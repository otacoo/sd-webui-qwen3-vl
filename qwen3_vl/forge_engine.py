import gc
import json
import os

import torch

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Qwen3-VL special token ids (fixed for this model)
ID_PAD = 151643        # <|endoftext|>
ID_IM_START = 151644   # <|im_start|>
ID_IM_END = 151645     # <|im_end|>
ID_VISION_START = 151652
ID_VISION_END = 151653
ID_IMAGE = 151655      # <|image_pad|>

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def build_chat_prompt(messages):
    parts = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            if isinstance(content, str):
                text = content
            else:
                chunks = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            chunks.append(part.get("text", ""))
                        elif part.get("type") == "image":
                            chunks.append(VISION_BLOCK)
                text = "".join(chunks)
            parts.append(f"<|im_start|>user\n{text}<|im_end|>\n")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def extract_images(messages):
    images = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                images.append(part["image"])
    return images


def _pil_to_tensor(image, device):
    import numpy as np

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _normalize_te_keys(sd):
    for prefix in (
        "text_encoder.qwen3vl_4b.transformer.",
        "qwen3vl_4b.transformer.",
        "qwen3vl_4b.",
    ):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
            break
    from backend.state_dict import state_dict_prefix_replace

    return state_dict_prefix_replace(
        sd,
        {"model.language_model.": "model.", "model.visual.": "visual."},
    )


class Qwen3VLForgeEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.storage_dtype = None
        self.quantized = False

    @property
    def is_loaded(self):
        return self.model is not None

    def load(self, path, device="auto"):
        import time

        import torch as _torch
        from transformers.modeling_utils import no_init_weights

        from backend import memory_management, utils
        from backend.nn.llm.llama import Qwen3VL
        from backend.operations import using_forge_operations
        from backend.state_dict import (
            convert_quantization,
            detect_quantization,
            load_state_dict,
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        t_start = time.time()
        print(f"[Qwen3-VL] Loading Qwen3-VL weights: {path}")

        sd, metadata = utils.load_torch_file(path, return_metadata=True)
        print(f"[Qwen3-VL] Weights read in {time.time() - t_start:.1f}s")

        t0 = time.time()
        sd, metadata = convert_quantization(sd, metadata)
        sd = _normalize_te_keys(sd)

        config_path = os.path.join(ASSETS_DIR, "config.json")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        storage_dtype = memory_management.text_encoder_dtype()
        state_dict_dtype = utils.weight_dtype(sd)
        quant_config = detect_quantization(sd)
        if quant_config is not None:
            storage_dtype = state_dict_dtype
        elif state_dict_dtype in [_torch.float8_e4m3fn, _torch.float8_e5m2, "nf4", "fp4"]:
            storage_dtype = state_dict_dtype

        print(f"[Qwen3-VL] Storage dtype: {storage_dtype}, quantized: {quant_config is not None}")

        t0 = time.time()
        with no_init_weights():
            with using_forge_operations(
                device=memory_management.cpu,
                dtype=storage_dtype,
                manual_cast_enabled=True,
                extra_dtype=quant_config,
            ):
                model = Qwen3VL(config)
        print(f"[Qwen3-VL] Model created in {time.time() - t0:.1f}s")

        t0 = time.time()
        load_state_dict(model, sd, log_name="Qwen3VL", ignore_start="lm_head.")
        print(f"[Qwen3-VL] Weights applied in {time.time() - t0:.1f}s")
        del sd
        gc.collect()
        model.eval()

        t0 = time.time()
        target = self._resolve_device(device)
        model.to(target)
        memory_management.soft_empty_cache()
        print(f"[Qwen3-VL] Moved to {target} in {time.time() - t0:.1f}s")

        self.model = model
        self.tokenizer = self._load_tokenizer()
        self.device = target
        self.storage_dtype = storage_dtype
        self.quantized = quant_config is not None
        self._head = None

        if self.quantized:
            try:
                import backend.nn.llm.qwen35 as _qwen35
                from backend.attention import attention_pytorch as _attention_pytorch

                if _qwen35.attention_function is not _attention_pytorch:
                    _qwen35.attention_function = _attention_pytorch
                    print(
                        "[Qwen3-VL] Vision tower attention: attention_pytorch "
                        "(flash_attn only supports fp16/bf16, qkv is fp8)"
                    )
            except Exception as e:
                print(f"[Qwen3-VL] WARNING: could not switch vision attention: {e}")

        print(f"[Qwen3-VL] Device: {target}, tokenizer: {self.tokenizer.__class__.__name__}")
        print(f"[Qwen3-VL] Model loaded. Total time: {time.time() - t_start:.1f}s")

    def _resolve_device(self, device):
        if device == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(ASSETS_DIR)

    def unload(self):
        self.model = None
        self.tokenizer = None
        self._head = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _prepare(self, prompt_text, images):
        from backend.nn.llm.qwen_vl import process_qwen2vl_images

        device = self.device
        model = self.model
        tokenizer = self.tokenizer
        compute_dtype = torch.bfloat16

        ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        text_ids = []
        image_slots = []
        images = list(images)
        for token_id in ids:
            if token_id == ID_IMAGE:
                if images:
                    image_slots.append((len(text_ids), images.pop(0)))
            else:
                text_ids.append(token_id)

        embed_tokens = model.get_input_embeddings()
        tokens_embed = embed_tokens(
            torch.tensor([text_ids], device=device, dtype=torch.long)
        ).to(compute_dtype)

        inserted = 0
        for pos, pil_image in image_slots:
            image, grid = process_qwen2vl_images(
                _pil_to_tensor(pil_image, device),
                patch_size=16,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
            merged, _deepstack = model.visual(
                image.to(device, dtype=compute_dtype), grid
            )
            emb = merged.to(compute_dtype).view(1, -1, merged.shape[-1])
            ind = pos + inserted
            tokens_embed = torch.cat(
                [tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1
            )
            inserted += emb.shape[1]

        seq = tokens_embed.shape[1]
        mask = torch.ones((1, seq), device=device, dtype=torch.long)

        return tokens_embed, mask

    def _logits(self, hidden):
        if self._head is None:
            weight = self.model.get_input_embeddings().weight
            self._head = weight.detach().float()
        return hidden.float() @ self._head.t()

    def _sample(self, logits, do_sample, temperature, top_p):
        logits = logits[:, -1, :]
        if not do_sample:
            return int(logits.argmax(dim=-1).item())
        if temperature and temperature != 1.0:
            logits = logits / max(float(temperature), 1e-6)
        if top_p and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            remove_mask = probs.cumsum(dim=-1) - probs > top_p
            sorted_logits = sorted_logits.masked_fill(remove_mask, -float("inf"))
            logits = logits.scatter(1, sorted_indices, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    @torch.inference_mode()
    def generate(
        self,
        prompt_text,
        images,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
    ):
        if not self.is_loaded:
            raise RuntimeError(
                "Qwen3-VL model is not loaded. Click 'Load Model' in the Qwen3-VL tab first."
            )

        model = self.model
        device = self.device
        stop_ids = (ID_IM_END, ID_IM_START, ID_PAD)

        tokens_embed, mask = self._prepare(prompt_text, images)

        out = model(
            None,
            embeds=tokens_embed,
            attention_mask=mask,
            past_key_values=[],
        )
        hidden = out[0]
        past = out[2]

        next_id = self._sample(
            self._logits(hidden[:, -1:]), do_sample, temperature, top_p
        )

        generated = []
        past_len = int(past[0][2])
        for _ in range(max(max_new_tokens, 1)):
            if next_id in stop_ids:
                break
            generated.append(next_id)

            one_mask = torch.ones((1, past_len + 1), device=device, dtype=torch.long)
            dec_embed = model.get_input_embeddings()(
                torch.tensor([[next_id]], device=device, dtype=torch.long)
            ).to(torch.bfloat16)
            out = model(
                None,
                embeds=dec_embed,
                attention_mask=one_mask,
                past_key_values=past,
            )
            hidden = out[0]
            past = out[2]
            past_len += 1

            next_id = self._sample(
                self._logits(hidden[:, -1:]), do_sample, temperature, top_p
            )

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
