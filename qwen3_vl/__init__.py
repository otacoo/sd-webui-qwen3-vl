__version__ = "0.1.0"

from . import settings, prompts, captioning, image_utils, vision, model, inference, batch


def log(message):
    print(f"[Qwen3-VL] {message}")