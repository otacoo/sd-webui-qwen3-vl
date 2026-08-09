import os
import traceback
from pathlib import Path

from . import image_utils
from . import inference
from . import captioning
from . import settings

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def iter_images(input_dir):
    if not input_dir or not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = []
    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(full)
    return sorted(files)


def caption_file(image_path, mode, custom_prompt="", system_prompt=None,
                 max_new_tokens=None, do_sample=False, temperature=None,
                 top_p=None):
    image = image_utils.load_image(image_path)
    result = inference.generate(
        mode,
        image,
        custom_prompt=custom_prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    if mode == "Tags":
        result = captioning.normalize_tags(result)
    return result


def run_batch(input_dir, output_dir, mode="Tags", custom_prompt="",
              system_prompt=None, overwrite=False, max_new_tokens=None,
              progress_cb=None):
    images = iter_images(input_dir)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    total = len(images)
    results = []
    errors = 0

    for index, image_path in enumerate(images):
        stem = Path(image_path).stem
        caption_path = os.path.join(output_dir, f"{stem}.txt") if output_dir else (
            image_path.rsplit(".", 1)[0] + ".txt"
        )

        if os.path.isfile(caption_path) and not overwrite:
            results.append((image_path, "skipped", caption_path, ""))
            if progress_cb:
                progress_cb(index + 1, total, os.path.basename(image_path))
            continue

        try:
            text = caption_file(
                image_path, mode, custom_prompt=custom_prompt,
                system_prompt=system_prompt, max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(text)
            results.append((image_path, "ok", caption_path, ""))
        except Exception as exc:
            errors += 1
            results.append((image_path, "error", "", str(exc)))
            print(f"[Qwen3-VL] Batch error for {image_path}:")
            traceback.print_exc()
            error_log_path = os.path.join(
                settings.EXTENSION_ROOT, "batch_errors.log"
            )
            try:
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{image_path}: {exc}\n")
            except Exception:
                pass

        if progress_cb:
            progress_cb(index + 1, total, os.path.basename(image_path))

    return results, errors