import threading

from . import model
from . import captioning
from . import prompts


def _mode_prompt(mode, custom_prompt=""):
    if mode == "Custom":
        return (custom_prompt or "").strip()
    return prompts.MODE_PROMPTS.get(mode, prompts.DESCRIPTION_PROMPT)


def _messages_for_single(prompt_text, image, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    })
    return messages


def generate(mode, image, custom_prompt="", system_prompt=None,
             max_new_tokens=None, do_sample=None, temperature=None,
             top_p=None):
    if image is None:
        raise ValueError("No image provided.")

    m = model.get_model()
    if m is None or not m.is_loaded:
        raise RuntimeError(
            "Qwen3-VL model is not loaded. Click 'Load Model' in the Qwen3-VL tab first."
        )

    text = _mode_prompt(mode, custom_prompt)
    if not text:
        raise ValueError("No instruction text provided.")

    messages = _messages_for_single(text, image, system_prompt)
    with inference_lock:
        raw = m.generate_messages(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    cleaned = captioning.clean_output(raw)
    if mode == "Tags":
        cleaned = captioning.normalize_tags(cleaned)
    return cleaned


def chat(history, message, image, system_prompt=None,
         max_new_tokens=None, do_sample=None, temperature=None,
         top_p=None):
    if image is None:
        raise ValueError("No image provided. Add an image before chatting.")
    if not (message or "").strip():
        raise ValueError("Message is empty.")

    m = model.get_model()
    if m is None or not m.is_loaded:
        raise RuntimeError(
            "Qwen3-VL model is not loaded. Click 'Load Model' in the Qwen3-VL tab first."
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": message.strip()},
        ],
    })
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    with inference_lock:
        raw = m.generate_messages(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    return captioning.clean_output(raw)


inference_lock = threading.Lock()