"""GUI launcher with a persistent error log and no console window."""
import logging
from pathlib import Path
import sys
import importlib.util
import os

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
# Keep the GUI usable while no GPU torch runtime has been installed yet.
if importlib.util.find_spec("torch") is None and (root / ".cpu-runtime").is_dir():
    sys.path.insert(0, str(root / ".cpu-runtime"))

if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            log_dir = Path.home() / "Library" / "Logs" / "FastMatch"
        else:
            log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "FastMatch" / "logs"
    else:
        log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    handler = RotatingFileHandler(log_dir / "fastmatch.log",
                                  maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from fastmatch.__main__ import main
        raise SystemExit(main())
    except Exception:
        logging.exception("FastMatch failed")
        raise SystemExit(1)
