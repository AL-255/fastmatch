"""Release metadata embedded by the standalone build, absent in source trees."""

import json
from pathlib import Path


def bundled_build_info() -> dict:
    try:
        data = json.loads(Path(__file__).with_name("_build_info.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
