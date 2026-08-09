import math
from pathlib import Path

FACTOR = 28
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS,
                 max_pixels=MAX_PIXELS):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio too extreme.")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        b = math.ceil(height / beta / factor) * factor
        a = math.ceil(width / beta / factor) * factor
        while a * b > max_pixels:
            if b > a:
                b -= factor
            else:
                a -= factor
        h_bar, w_bar = b, a
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        b = math.floor(height * beta / factor) * factor
        a = math.floor(width * beta / factor) * factor
        while a * b < min_pixels:
            if b < a:
                b += factor
            else:
                a += factor
        h_bar, w_bar = b, a
    return h_bar, w_bar


def _fetch_image(image):
    from PIL import Image

    if isinstance(image, str):
        if not Path(image).is_file():
            raise FileNotFoundError(f"Image file not found: {image}")
        with Image.open(image) as im:
            image = im.convert("RGB")
    elif isinstance(image, Path):
        with Image.open(image) as im:
            image = im.convert("RGB")

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        width, height = smart_resize(image.height, image.width)
        if (image.width, image.height) != (width, height):
            resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else 1
            image = image.resize((width, height), resample)
        return image
    return image


def process_vision_info(messages):
    image_inputs = []
    video_inputs = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "image":
                image_inputs.append(_fetch_image(part["image"]))
            elif part_type == "video":
                video_inputs.append(part["video"])
    return image_inputs, video_inputs