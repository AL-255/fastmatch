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

from fastmatch.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
