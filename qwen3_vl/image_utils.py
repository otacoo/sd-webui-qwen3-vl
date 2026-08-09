import os
from pathlib import Path

_LAST_IMAGE = None
_LAST_IMAGE_PATH = None

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

CALLBACK_REGISTERED = False


def _pil_from_path(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def load_image(source):
    from PIL import Image
    import numpy as np

    if source is None:
        raise ValueError("No image provided.")
    if isinstance(source, str):
        path = source
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        return _pil_from_path(path)
    if isinstance(source, Path):
        return _pil_from_path(str(source))
    if isinstance(source, np.ndarray):
        return Image.fromarray(source).convert("RGB")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    raise ValueError(f"Unsupported image source type: {type(source).__name__}")


def validate_image(image):
    if image is None:
        raise ValueError("No image provided. Upload an image or use the last generated image.")
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.width < 1 or image.height < 1:
        raise ValueError("Image has invalid dimensions.")
    return image


def resize_image_if_needed(image, max_side=2048):
    longest = max(image.width, image.height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, image.LANCZOS if hasattr(image, "LANCZOS") else 3)


def _on_image_saved(params):
    global _LAST_IMAGE, _LAST_IMAGE_PATH
    try:
        image = getattr(params, "image", None)
        if image is None:
            filename = getattr(params, "filename_path", None) or getattr(params, "filename", None)
            path = getattr(params, "path", None) or filename
            if path and os.path.isfile(path):
                _LAST_IMAGE = _pil_from_path(path).copy()
                _LAST_IMAGE_PATH = str(path)
        else:
            _LAST_IMAGE = image.copy()
            _LAST_IMAGE_PATH = getattr(params, "filename_path", None) or getattr(params, "path", None)
    except Exception:
        pass


def _on_image_saved_image(params):
    global _LAST_IMAGE, _LAST_IMAGE_PATH
    try:
        image = getattr(params, "image", None)
        if image is not None:
            _LAST_IMAGE = image.copy()
            _LAST_IMAGE_PATH = getattr(params, "filename_path", None) or getattr(params, "path", None)
    except Exception:
        pass


def register_callbacks():
    global CALLBACK_REGISTERED
    if CALLBACK_REGISTERED:
        return
    try:
        from modules import script_callbacks
        script_callbacks.on_image_saved(_on_image_saved)
        script_callbacks.on_image_saved_image(_on_image_saved_image)
        CALLBACK_REGISTERED = True
    except Exception:
        CALLBACK_REGISTERED = True


def _candidate_dirs():
    dirs = []
    try:
        from modules import shared
        for key in ("outdir_img_samples", "outdir_txt_samples", "outdir_samples"):
            value = getattr(shared.opts, key, None)
            if value:
                dirs.append(str(value))
    except Exception:
        pass
    try:
        from modules import paths
        base = getattr(paths, "data_path", None)
        if base:
            for sub in ("outputs/img2img-images", "outputs/txt2img-images", "outputs"):
                dirs.append(os.path.join(base, sub))
    except Exception:
        pass
    seen = []
    for d in dirs:
        if d and d not in seen:
            seen.append(d)
    return seen


def _newest_image_file(dirs):
    best_path, best_mtime = None, None
    for directory in dirs:
        if not os.path.isdir(directory):
            continue
        try:
            for name in os.listdir(directory):
                if os.path.splitext(name)[1].lower() not in _IMAGE_EXTS:
                    continue
                full = os.path.join(directory, name)
                mtime = os.path.getmtime(full)
                if best_mtime is None or mtime > best_mtime:
                    best_path, best_mtime = full, mtime
        except Exception:
            continue
    return best_path


def get_last_generated_image():
    global _LAST_IMAGE, _LAST_IMAGE_PATH
    if _LAST_IMAGE is not None:
        return _LAST_IMAGE.copy(), _LAST_IMAGE_PATH
    path = _newest_image_file(_candidate_dirs())
    if path:
        _LAST_IMAGE = _pil_from_path(path).copy()
        _LAST_IMAGE_PATH = path
        return _LAST_IMAGE, path
    raise FileNotFoundError(
        "No generated image found. Generate an image in Forge first, "
        "or upload an image manually."
    )


def save_image_reference(image, directory=None, name="chat_image.png"):
    if directory is None:
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    image.save(path)
    return path


def default_caption_path_for(image_path):
    if image_path:
        p = Path(image_path)
        return str(p.with_suffix(".txt"))
    return None


def is_image_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in _IMAGE_EXTS