# SD WebUI Qwen3-VL

SD WebUI Forge extension that integrates **Qwen3-VL-4B-Instruct** as a local
vision-language model: image description, Stable Diffusion prompt generation,
WD14-style tag generation, chat, and batch captioning.


> [!Important]
> This extension is totally LLM coded and will only work on **Forge Neo** and **Nvidia** GPUs. Use at your own risk!
> **TL;DR:** Werks on mah machine.

## Installation

1. Download the model:
    - [Qwen3 VL 4B](https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders)
    - Save it into Forge's `models/text_encoder` folder
2. Go into Extensions tab > Install from URL
3. Paste `https://github.com/otacoo/sd-webui-qwen3-vl.git`
4. Press Install
5. Apply and Restart the UI

## Info

- **Mode** - `Description`, `SD Prompt`, `Tags`, `Caption`, or `Custom`.
  - Pick from various ways to describe an image, e.g. tags for Anima, caption for Krea2.
  - Use *Custom* to give the model your own instructions.
- **Save Caption (.txt)** - save next to the source image (last generated) or
  into `outputs/qwen3-vl-captions`.
- **Chat** - conversation about the current image; resets when the
  image changes. **Clear Chat** resets the history.
- **Batch Captioning** - captions every image in a directory to
  `<name>.txt` (skips existing files unless *Overwrite existing captions* is
  checked).

Generation is deterministic by default. Check **Random sampling** to enable 
temperature / top-p sampling. Inference is serialized
with a lock so chat, analysis and batch never run concurrently.

## Settings

`settings.json` in the extension folder is written automatically when you
change values in the UI (model path, device, token counts, temperatures, batch
dirs, ...).

## Troubleshooting

- **No Qwen3-VL model class found** — `transformers` is too old.
  Run `install.py` or `pip install "transformers>=4.57.0"`.
- **Out of memory** — unload Forge's checkpoint first
  (`Unload Forge Checkpoint` button), or use `cpu` (slow).
- Qwen3-VL-4B uses ~10 GB VRAM in bfloat16.

## License
MIT
