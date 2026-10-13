"""Pre-download a faster-whisper model so the first app launch is instant.

Usage:
    python scripts/download_model.py [model] [--mirror]

    model:    large-v3 (default) | large-v2 | medium | small | base
    --mirror: use hf-mirror.com (recommended in China; also disables the Xet
              transfer protocol, which the mirror cannot proxy -> HTTP 401)
"""
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default="large-v3",
                    choices=["large-v3", "large-v2", "medium", "small", "base"])
parser.add_argument("--mirror", action="store_true")
args = parser.parse_args()

if args.mirror:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download

t0 = time.time()
path = snapshot_download(f"Systran/faster-whisper-{args.model}")
print(f"[OK] {args.model} downloaded to:\n  {path}\n  elapsed: {time.time() - t0:.0f}s")
