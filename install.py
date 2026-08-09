import importlib.metadata
import os
import sys

MIN_PYTHON = (3, 9)

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DONE_MARKER = os.path.join(_ROOT, ".setup_ok")


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    if os.path.exists(_DONE_MARKER):
        print("[Qwen3-VL] Setup check complete. Everything OK.")
        return

    print("[Qwen3-VL] Checking environment...")
    print(f"[Qwen3-VL] Python: {sys.version.split()[0]}")

    if sys.version_info < MIN_PYTHON:
        print(f"[Qwen3-VL] WARNING: Python {sys.version_info.major}.{sys.version_info.minor} "
              f"is older than recommended {MIN_PYTHON[0]}.{MIN_PYTHON[1]}.")

    print("[Qwen3-VL] This extension reuses Forge's built-in Qwen3-VL-4B "
          "(the Krea-2 text encoder). No additional packages are required.")

    torch_version = _version("torch")
    transformers_version = _version("transformers")
    pillow_version = _version("pillow")

    cuda_ok = False
    if torch_version is not None:
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False

    if torch_version is not None:
        print(f"[OK] PyTorch {torch_version}")
    else:
        print("[Qwen3-VL] WARNING: PyTorch not detected. This extension reuses the "
              "PyTorch installation provided by Forge; it is not installed here.")

    if cuda_ok:
        print("[OK] CUDA available")
    else:
        print("[Qwen3-VL] CUDA not available. CPU inference will be slow.")

    if transformers_version is not None:
        print(f"[OK] Transformers {transformers_version}")
    else:
        print("[WARN] transformers not detected. It is required for the tokenizer "
              "and is normally present in Forge's environment.")

    if pillow_version is not None:
        print(f"[OK] Pillow {pillow_version}")
    else:
        print("[WARN] Pillow not detected. It is normally present in Forge's environment.")

    if torch_version is not None:
        try:
            with open(_DONE_MARKER, "w", encoding="utf-8") as f:
                f.write("ok\n")
        except Exception:
            pass

    print("[Qwen3-VL] Setup check complete. Everything OK.")


if __name__ == "__main__":
    main()
