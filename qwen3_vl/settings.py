import json
import os

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(EXTENSION_ROOT, "settings.json")

DEFAULTS = {
    "model_path": "",
    "device": "auto",
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.9,
    "sampling": False,
    "chat_max_new_tokens": 512,
    "chat_temperature": 0.6,
    "chat_top_p": 0.9,
    "chat_sampling": True,
    "system_prompt": "You are an image captioning assistant. Only describe visible information. Never infer identities or hidden information.",
    "custom_prompt": "",
    "caption_out_dir": "outputs/qwen3-vl-captions",
    "batch_input_dir": "",
    "batch_output_dir": "",
    "batch_mode": "Tags",
    "batch_overwrite": False,
    "batch_max_new_tokens": 256,
}

_settings = None


def load_settings():
    global _settings
    data = dict(DEFAULTS)
    if os.path.isfile(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                data.update(stored)
        except Exception:
            pass
    _settings = data
    return data


def save_settings(updates=None):
    global _settings
    if not _settings:
        load_settings()
    if updates:
        _settings.update(updates)
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(_settings, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return _settings


def get(key):
    if not _settings:
        load_settings()
    return _settings.get(key, DEFAULTS.get(key))


def update(**kwargs):
    return save_settings(kwargs)