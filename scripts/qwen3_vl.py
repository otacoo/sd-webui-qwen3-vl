import os
import sys
import time
import traceback
import hashlib

import gradio as gr

_EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, _EXTENSION_ROOT)

from qwen3_vl import settings as q_settings
from qwen3_vl import model as q_model
from qwen3_vl import inference
from qwen3_vl import image_utils
from qwen3_vl import batch as q_batch
from qwen3_vl import prompts
from qwen3_vl import captioning


image_utils.register_callbacks()

GRADIO_MAJOR = 3
try:
    import gradio as gr_mod
    GRADIO_MAJOR = int(str(getattr(gr_mod, "__version__", "3")).split(".")[0])
except Exception:
    pass


def _click_js(button, js_code):
    kwargs = {"js": js_code} if GRADIO_MAJOR >= 4 else {"_js": js_code}
    button.click(None, None, None, **kwargs)


def _error_text(exc, context="operation"):
    traceback.print_exc()
    return f"Qwen3-VL Error during {context}:\n{exc}"


def _status_markdown():
    m = q_model.get_model()
    if m is not None and m.is_loaded:
        return (
            "**Status: Loaded**\n\n"
            f"- Model: `{os.path.basename(m.model_path)}`\n"
            f"- Device: `{m.device}`\n"
            f"- Dtype: `{m.dtype}`\n"
            f"- VRAM: {m.vram_info()}\n"
        )
    return (
        "**Status: Unloaded**\n\n"
        "Load the model before analyzing images. Qwen3-VL only uses VRAM "
        "after loading.\n"
    )


def _save_result(text, image_path):
    directory = None
    filename = None
    if image_path:
        directory = os.path.dirname(image_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{stem}.txt"
    else:
        directory = q_settings.get("caption_out_dir") or "outputs/qwen3-vl-captions"
        filename = f"caption_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


_TE_EXTENSIONS = ("ckpt", "pt", "pth", "bin", "safetensors", "sft")


def _text_encoder_dirs():
    dirs = []
    try:
        from modules import paths

        dirs.append(
            os.path.abspath(os.path.join(paths.models_path, "text_encoder"))
        )
    except Exception:
        pass
    try:
        from modules import shared as forge_shared

        for d in getattr(forge_shared.cmd_opts, "text_encoder_dirs", ()) or ():
            dirs.append(os.path.abspath(d))
    except Exception:
        pass
    return dirs


def _text_encoder_files():
    files = {}
    try:
        from modules_forge.main_entry import find_files_with_extensions
    except Exception:
        find_files_with_extensions = None
    for d in _text_encoder_dirs():
        if not os.path.isdir(d):
            continue
        if find_files_with_extensions is not None:
            files.update(find_files_with_extensions(d, _TE_EXTENSIONS))
        else:
            for root, _, fnames in os.walk(d):
                for f in fnames:
                    if f.endswith(_TE_EXTENSIONS):
                        files[f] = os.path.join(root, f)
    return files


def _resolve_model_selection(selection):
    selection = (selection or "").strip()
    if not selection:
        return ""
    if os.path.exists(selection):
        return selection
    return _text_encoder_files().get(selection, selection)


def _model_choices():
    files = _text_encoder_files()
    return sorted((fname, fpath) for fname, fpath in files.items())


def _model_choices_and_values():
    choices = _model_choices()
    choice_values = [value for _, value in choices]
    stored = q_settings.get("model_path") or ""
    if stored and stored not in choice_values:
        if os.path.exists(stored):
            choices.insert(0, (os.path.basename(stored), stored))
    value = stored if stored in choice_values else (
        choices[0][1] if choices else None
    )
    return choices, value


def on_refresh_models():
    try:
        from modules_forge.main_entry import refresh_models

        refresh_models()
    except Exception:
        pass
    return gr.update(choices=_model_choices())


def register_ui():
    from modules import script_callbacks

    def on_ui_tabs():
        with gr.Blocks(analytics_enabled=False) as qwen_tab:
            _build_tab()
        return [(qwen_tab, "Qwen3-VL", "qwen3-vl")]

    script_callbacks.on_ui_tabs(on_ui_tabs)


def _build_tab():
    choices, model_value = _model_choices_and_values()

    with gr.Row(elem_id="qwen3-vl-row"):
        with gr.Column(scale=5, elem_id="qwen3-vl-left"):
            with gr.Accordion("Model", open=True):
                with gr.Row():
                    model_path = gr.Dropdown(
                        label="Model",
                        choices=choices,
                        value=model_value,
                        elem_id="qwen3-vl-model",
                    )
                    try:
                        from modules.ui_common import create_refresh_button
                    except Exception:
                        create_refresh_button = None
                    if create_refresh_button is not None:
                        create_refresh_button(
                            [model_path],
                            on_refresh_models,
                            lambda: {"choices": _model_choices()},
                            "qwen3-vl-model-refresh",
                        )
                    else:
                        gr.Button(
                            "Refresh", elem_id="qwen3-vl-model-refresh"
                        ).click(on_refresh_models, [], [model_path])
                with gr.Row():
                    device = gr.Dropdown(
                        label="Device",
                        choices=["auto", "cuda", "cpu"],
                        value=q_settings.get("device"),
                        elem_id="qwen3-vl-device",
                    )
                    load_btn = gr.Button("Load Model", elem_id="qwen3-vl-load")
                    unload_btn = gr.Button("Unload Model", elem_id="qwen3-vl-unload")

            with gr.Accordion("Image", open=True):
                image = gr.Image(
                    label="Image",
                    type="pil",
                    interactive=True,
                    height=300 if GRADIO_MAJOR >= 4 else None,
                    elem_id="qwen3-vl-image",
                )
                if GRADIO_MAJOR < 4:
                    image.style(height=300)
                with gr.Row():
                    use_last_btn = gr.Button(
                        "Use Last Generated", elem_id="qwen3-vl-use-last"
                    )
                    image_info = gr.Markdown(
                        "No image loaded.", elem_id="qwen3-vl-image-info"
                    )

            with gr.Accordion("Analyze", open=True):
                with gr.Row():
                    mode = gr.Dropdown(
                        label="Mode",
                        choices=prompts.MODES,
                        value="Description",
                        elem_id="qwen3-vl-mode",
                    )
                    custom_prompt = gr.Textbox(
                        label="Custom instruction (used when mode = Custom)",
                        value=q_settings.get("custom_prompt"),
                        lines=2,
                        elem_id="qwen3-vl-custom",
                    )
                with gr.Row():
                    max_tokens = gr.Slider(
                        label="Max new tokens",
                        minimum=16,
                        maximum=2048,
                        step=16,
                        value=q_settings.get("max_new_tokens"),
                        elem_id="qwen3-vl-max-tokens",
                    )
                    sampling = gr.Checkbox(
                        label="Random sampling",
                        value=q_settings.get("sampling"),
                        elem_id="qwen3-vl-sampling",
                    )
                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        value=q_settings.get("temperature"),
                        elem_id="qwen3-vl-temperature",
                    )
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=q_settings.get("top_p"),
                        elem_id="qwen3-vl-top-p",
                    )
                analyze_btn = gr.Button(
                    "Analyze", variant="primary", elem_id="qwen3-vl-analyze"
                )

            with gr.Accordion("Chat", open=False):
                with gr.Row():
                    chat_temperature = gr.Slider(
                        label="Chat temperature",
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        value=q_settings.get("chat_temperature"),
                        elem_id="qwen3-vl-chat-temperature",
                    )
                    chat_max_tokens = gr.Slider(
                        label="Chat max new tokens",
                        minimum=16,
                        maximum=2048,
                        step=16,
                        value=q_settings.get("chat_max_new_tokens"),
                        elem_id="qwen3-vl-chat-max-tokens",
                    )
                chatbot = gr.Chatbot(label="Chat", elem_id="qwen3-vl-chatbot")
                with gr.Row():
                    message = gr.Textbox(
                        label="Message",
                        placeholder="Ask about the image...",
                        elem_id="qwen3-vl-chat-message",
                    )
                    send_chat_btn = gr.Button(
                        "Send", variant="primary", elem_id="qwen3-vl-chat-send"
                    )
                clear_chat_btn = gr.Button(
                    "Clear Chat", elem_id="qwen3-vl-chat-clear"
                )

            with gr.Accordion("Batch Captioning", open=False):
                with gr.Row():
                    batch_input = gr.Textbox(
                        label="Input directory",
                        value=q_settings.get("batch_input_dir"),
                        elem_id="qwen3-vl-batch-input",
                    )
                    batch_output = gr.Textbox(
                        label="Output directory (empty = next to images)",
                        value=q_settings.get("batch_output_dir"),
                        elem_id="qwen3-vl-batch-output",
                    )
                with gr.Row():
                    batch_mode = gr.Dropdown(
                        label="Mode",
                        choices=[m for m in prompts.MODES],
                        value=q_settings.get("batch_mode"),
                        elem_id="qwen3-vl-batch-mode",
                    )
                    batch_overwrite = gr.Checkbox(
                        label="Overwrite existing captions",
                        value=q_settings.get("batch_overwrite"),
                        elem_id="qwen3-vl-batch-overwrite",
                    )
                    batch_max_tokens = gr.Slider(
                        label="Max new tokens",
                        minimum=16,
                        maximum=2048,
                        step=16,
                        value=q_settings.get("batch_max_new_tokens"),
                        elem_id="qwen3-vl-batch-max-tokens",
                    )
                batch_btn = gr.Button(
                    "Start", variant="primary", elem_id="qwen3-vl-batch-start"
                )
                batch_result = gr.Textbox(
                    label="Batch log", lines=6, interactive=False,
                    elem_id="qwen3-vl-batch-result",
                )

        with gr.Column(scale=4, elem_id="qwen3-vl-right"):
            status = gr.Markdown(_status_markdown(), elem_id="qwen3-vl-status")
            result = gr.Textbox(
                label="Result (editable)",
                lines=10,
                elem_id="qwen3-vl-result",
            )
            with gr.Row():
                save_btn = gr.Button("Save Caption (.txt)", elem_id="qwen3-vl-save")
                send_txt_btn = gr.Button(
                    "Send to txt2img prompt", elem_id="qwen3-vl-send-txt"
                )
                append_txt_btn = gr.Button(
                    "Append to txt2img prompt", elem_id="qwen3-vl-append-txt"
                )
                send_img_btn = gr.Button(
                    "Send to img2img prompt", elem_id="qwen3-vl-send-img"
                )
            action_msg = gr.Markdown("", elem_id="qwen3-vl-action-msg")

    current_image_path = gr.State(None)
    chat_fingerprint = gr.State("")

    def on_load(model_val, device_val):
        try:
            path = _resolve_model_selection(model_val)
            if not path:
                return "**Qwen3-VL Error:** No model selected. Pick a model "
                "from the list."
            q_model.ensure_model(model_path=path, device=device_val)
            return _status_markdown()
        except Exception as exc:
            return _error_text(exc, "model load")

    load_btn.click(on_load, [model_path, device], [status])

    def on_unload():
        try:
            q_model.unload_model()
            return _status_markdown()
        except Exception as exc:
            return _error_text(exc, "model unload")

    unload_btn.click(on_unload, [], [status])

    def on_use_last():
        try:
            img, path = image_utils.get_last_generated_image()
            return gr.update(value=img), (
                f"Loaded last generated image ({os.path.basename(path)})" if path
                else "Loaded last generated image."
            ), path
        except Exception as exc:
            return gr.update(), _error_text(exc, "loading last generated image"), None

    use_last_btn.click(on_use_last, [], [image, image_info, current_image_path])

    def on_analyze(mode_val, custom_val, image_val, max_tokens_val,
                   sampling_val, temperature_val, top_p_val):
        try:
            img = image_utils.validate_image(image_val)
            t0 = time.time()
            text = inference.generate(
                mode_val,
                img,
                custom_prompt=custom_val,
                system_prompt=q_settings.get("system_prompt"),
                max_new_tokens=max_tokens_val,
                do_sample=sampling_val,
                temperature=temperature_val,
                top_p=top_p_val,
            )
            dt = time.time() - t0
            return text, f"Analysis complete. ({dt:.1f}s)"
        except Exception as exc:
            return "", _error_text(exc, "analysis")

    analyze_btn.click(
        on_analyze,
        [mode, custom_prompt, image, max_tokens, sampling, temperature, top_p],
        [result, action_msg],
    )

    def on_save(result_val, img_path):
        try:
            if not (result_val or "").strip():
                return "Nothing to save."
            path = _save_result(result_val, img_path)
            return f"Saved to `{path}`"
        except Exception as exc:
            return _error_text(exc, "saving caption")

    save_btn.click(on_save, [result, current_image_path], [action_msg])

    def _send_js(target_id, append, switch_tab):
        switch = ""
        if switch_tab:
            switch = (
                "const tab = document.querySelector('#%s');"
                "if (tab) tab.click();"
            ) % switch_tab
        return (
            "() => {"
            "const src = document.querySelector('#qwen3-vl-result textarea');"
            "const dst = document.querySelector('#%s textarea');"
            "if (!dst || !src) return;"
            "const value = src.value;"
            "const proto = Object.getOwnPropertyDescriptor("
            "window.HTMLTextAreaElement.prototype, 'value');"
            "let target = value;"
            "if (%s && dst.value.trim()) target = dst.value.replace(/\\s*$/, '') + ', ' + value;"
            "if (proto && proto.set) proto.set.call(dst, target);"
            "else dst.value = target;"
            "dst.dispatchEvent(new Event('input', {bubbles:true}));"
            "dst.dispatchEvent(new Event('change', {bubbles:true}));"
            "%s"
            "}" % (target_id, "true" if append else "false", switch)
        )

    _click_js(send_txt_btn, _send_js("txt2img_prompt", False, "tab_txt2img"))
    _click_js(append_txt_btn, _send_js("txt2img_prompt", True, None))
    _click_js(send_img_btn, _send_js("img2img_prompt", False, "tab_img2img"))

    def on_chat(chatbot_val, message_val, image_val, fp_val,
                chat_temperature_val, chat_max_tokens_val):
        try:
            img = image_utils.validate_image(image_val)
            fingerprint = hashlib.md5(img.tobytes()).hexdigest()
            if fp_val and fp_val != fingerprint:
                chatbot_val = []
            history = chatbot_val if isinstance(chatbot_val, list) else []
            t0 = time.time()
            reply = inference.chat(
                history,
                message_val,
                img,
                system_prompt=q_settings.get("system_prompt"),
                max_new_tokens=chat_max_tokens_val,
                do_sample=True,
                temperature=chat_temperature_val,
                top_p=q_settings.get("chat_top_p"),
            )
            dt = time.time() - t0
            history = history + [[message_val, reply]]
            return history, "", fingerprint, f"Chat updated. ({dt:.1f}s)"
        except Exception as exc:
            return (
                chatbot_val if isinstance(chatbot_val, list) else [],
                message_val,
                fp_val,
                _error_text(exc, "chat"),
            )

    send_chat_btn.click(
        on_chat,
        [chatbot, message, image, chat_fingerprint, chat_temperature, chat_max_tokens],
        [chatbot, message, chat_fingerprint, action_msg],
    )

    def on_clear_chat():
        return [], "", "Chat cleared."

    clear_chat_btn.click(on_clear_chat, [], [chatbot, chat_fingerprint, action_msg])

    def on_batch(input_dir, output_dir, batch_mode_val, overwrite_val,
                 batch_max_tokens_val, progress=gr.Progress()):
        try:
            def report(done, total, name):
                progress(done / max(total, 1), desc=f"{done}/{total} {name}")

            results, errors = q_batch.run_batch(
                input_dir, output_dir, mode=batch_mode_val,
                system_prompt=q_settings.get("system_prompt"),
                overwrite=overwrite_val,
                max_new_tokens=batch_max_tokens_val,
                progress_cb=report,
            )
            lines = []
            for img, state, out_path, _ in results:
                if state == "ok":
                    lines.append(f"[OK] {os.path.basename(img)} -> {out_path}")
                elif state == "skipped":
                    lines.append(f"[SKIP] {os.path.basename(img)} (caption exists)")
                else:
                    lines.append(f"[ERROR] {os.path.basename(img)}")
            lines.append(f"\nDone. {len(results)} files, {errors} errors.")
            return "\n".join(lines)
        except Exception as exc:
            return _error_text(exc, "batch captioning")

    batch_btn.click(
        on_batch,
        [batch_input, batch_output, batch_mode, batch_overwrite, batch_max_tokens],
        [batch_result],
    )

    model_path.change(
        lambda v: q_settings.update(model_path=v), [model_path], []
    )
    device.change(
        lambda v: q_settings.update(device=v), [device], []
    )
    custom_prompt.change(
        lambda v: q_settings.update(custom_prompt=v), [custom_prompt], []
    )
    max_tokens.release(
        lambda v: q_settings.update(max_new_tokens=v), [max_tokens], []
    )
    sampling.change(
        lambda v: q_settings.update(sampling=v), [sampling], []
    )
    temperature.release(
        lambda v: q_settings.update(temperature=v), [temperature], []
    )
    top_p.release(
        lambda v: q_settings.update(top_p=v), [top_p], []
    )
    chat_temperature.release(
        lambda v: q_settings.update(chat_temperature=v), [chat_temperature], []
    )
    chat_max_tokens.release(
        lambda v: q_settings.update(chat_max_new_tokens=v), [chat_max_tokens], []
    )
    batch_input.change(
        lambda v: q_settings.update(batch_input_dir=v), [batch_input], []
    )
    batch_output.change(
        lambda v: q_settings.update(batch_output_dir=v), [batch_output], []
    )
    batch_mode.change(
        lambda v: q_settings.update(batch_mode=v), [batch_mode], []
    )
    batch_overwrite.change(
        lambda v: q_settings.update(batch_overwrite=v), [batch_overwrite], []
    )
    batch_max_tokens.release(
        lambda v: q_settings.update(batch_max_new_tokens=v), [batch_max_tokens], []
    )


register_ui()