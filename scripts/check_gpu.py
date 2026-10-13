"""GPU availability check for faster-whisper (CTranslate2).

Exit code 0 always: no CUDA is a valid outcome (the app falls back to CPU).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transcriber import setup_cuda_libs

setup_cuda_libs()

import ctranslate2

print("ctranslate2:", ctranslate2.__version__)
n = ctranslate2.get_cuda_device_count()
print("CUDA devices detected:", n)
if n:
    types = ", ".join(sorted(ctranslate2.get_supported_compute_types("cuda")))
    print(f"[OK] GPU acceleration ready (supported compute types: {types})")
    print("     faster-whisper will run on CUDA with int8_float16.")
else:
    print("[WARN] No usable CUDA device found.")
    print("       The app will still work on CPU (int8), just slower (~2-4x realtime).")
