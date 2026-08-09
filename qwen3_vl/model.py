import gc
import json
import os
import threading
from datetime import datetime

from . import settings
from .forge_engine import Qwen3VLForgeEngine, build_chat_prompt, extract_images


class Qwen3VLModel:
    def __init__(self):
        self.model_path = None
        self.device = None
        self.dtype = None
        self.loaded_at = None
        self._engine = None
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def _resolve_model_path(self, model_path):
        path = (model_path or "").strip()
        if not path:
            path = settings.get("model_path") or ""
        if not path:
            raise FileNotFoundError(
                "No model file selected. Pick a Qwen3-VL weights file from the "
                "list (e.g. models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors)."
            )
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.lower().endswith(".safetensors") and "qwen" in name.lower():
                    return os.path.join(path, name)
            raise FileNotFoundError(
                f"No Qwen3-VL .safetensors file found in directory: {path}"
            )
        return path

    def load(self, model_path=None, device="auto"):
        with self._lock:
            if self._loaded:
                self._unload_inner()

            resolved_path = self._resolve_model_path(model_path)

            engine = Qwen3VLForgeEngine()
            engine.load(resolved_path, device=device)

            self._engine = engine
            self.model_path = resolved_path
            self.device = str(engine.device)
            self.dtype = str(engine.storage_dtype)
            self.loaded_at = datetime.now()
            self._loaded = True

            print("[Qwen3-VL] VRAM: " + self.vram_info())

    def _unload_inner(self, torch=None):
        if torch is None:
            try:
                import torch
            except Exception:
                torch = None
        if self._engine is not None:
            try:
                self._engine.unload()
            except Exception:
                pass
        self._engine = None
        self.model_path = None
        self.loaded_at = None
        self._loaded = False
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def unload(self):
        with self._lock:
            was_loaded = self._loaded
            self._unload_inner()
            if was_loaded:
                print("[Qwen3-VL] Model unloaded, CUDA cache cleared.")

    def vram_info(self):
        try:
            import torch
            if torch.cuda.is_available() and self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return f"{allocated:.2f} GiB allocated / {reserved:.2f} GiB reserved / {total:.2f} GiB total"
        except Exception:
            pass
        return "n/a"

    def status_lines(self):
        lines = []
        if self._loaded:
            lines.append("Status: Loaded (Qwen3-VL-4B via Forge backend)")
            lines.append(f"Path: {self.model_path}")
            lines.append(f"Device: {self.device}")
            lines.append(f"Dtype: {self.dtype}")
            lines.append("VRAM: " + self.vram_info())
        else:
            lines.append("Status: Unloaded")
        return "\n".join(lines)

    def generate_messages(self, messages, max_new_tokens=512, do_sample=False,
                          temperature=None, top_p=None, repetition_penalty=1.0):
        if not self._loaded or self._engine is None:
            raise RuntimeError(
                "Qwen3-VL model is not loaded. Click 'Load Model' in the Qwen3-VL tab first."
            )

        prompt_text = build_chat_prompt(messages)
        images = extract_images(messages)

        with self._lock:
            return self._engine.generate(
                prompt_text,
                images,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if temperature is not None else 0.7,
                top_p=top_p if top_p is not None else 0.9,
            )

    def load_check_and_report(self):
        if not self._loaded:
            raise RuntimeError("Qwen3-VL model is not loaded.")


_singleton = None
_singleton_lock = threading.Lock()


def ensure_model(model_path=None, device="auto"):
    global _singleton
    with _singleton_lock:
        if _singleton is None or not _singleton.is_loaded:
            _singleton = Qwen3VLModel()
            _singleton.load(model_path=model_path, device=device)
        return _singleton


def get_model():
    return _singleton


def unload_model():
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.unload()
            _singleton = None


def memory_report():
    try:
        import torch
        if torch.cuda.is_available():
            return (
                f"CUDA: {torch.cuda.memory_allocated()/1024**3:.2f} GiB allocated, "
                f"{torch.cuda.memory_reserved()/1024**3:.2f} GiB reserved"
            )
    except Exception:
        pass
    return "No CUDA available"


def _model_search_dirs():
    dirs = []
    try:
        from modules import paths
        models_path = getattr(paths, "models_path", None)
        if models_path:
            dirs.append(str(models_path))
    except Exception:
        pass
    try:
        from modules import shared
        data_path = getattr(shared, "data_path", None)
        if data_path:
            dirs.append(os.path.join(str(data_path), "models"))
    except Exception:
        pass
    forge_root = os.path.dirname(settings.EXTENSION_ROOT)
    dirs.append(os.path.join(forge_root, "models"))
    seen = []
    for d in dirs:
        if d and d not in seen:
            seen.append(d)
    return seen


def discover_models():
    found = []
    for base in _model_search_dirs():
        te = os.path.join(base, "text_encoder")
        if not os.path.isdir(te):
            continue
        for name in sorted(os.listdir(te)):
            if name.lower().endswith(".safetensors") and "qwen" in name.lower():
                full = os.path.join(te, name)
                if full not in found:
                    found.append(full)
    return found
