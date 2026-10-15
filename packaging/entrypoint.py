"""PyInstaller entry point. Keep windowed builds' diagnostics in a user log."""

import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if sys.stdout is None or sys.stderr is None:
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "FastMatch"
    else:
        root = Path.home() / "Library" / "Logs" / "FastMatch"
    root.mkdir(parents=True, exist_ok=True)
    log = (root / "fastmatch.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log

from fastmatch.external_runtime import configure_runtime

configure_runtime()

from fastmatch.__main__ import main


def missing_runtime(error):
    message = (f"FastMatch could not load a required library: {error}\n\n"
               "For CUDA/ROCm builds, install the system GPU runtime described in README.md "
               "and expose its library directories. See _internal/runtime-requirements.json "
               "for this build's library list. Use the CPU package if no GPU runtime is installed.")
    print(message, file=sys.stderr)
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "FastMatch: missing runtime", 0x10)
    return 1

if __name__ == "__main__":
    try:
        code = main()
    except (ImportError, OSError) as error:
        code = missing_runtime(error)
    raise SystemExit(code)
