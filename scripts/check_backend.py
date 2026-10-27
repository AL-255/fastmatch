"""Print runtime details and exercise the same GPU kernels as startup."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from fastmatch.device import device_banner_text, resolve_device

if __name__ == "__main__":
    resolved = resolve_device("auto")
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name,
                        "vram_bytes": props.total_memory,
                        "architecture": getattr(props, "gcnArchName", None)})
    print(json.dumps({"python": sys.version, "torch": torch.__version__,
                      "hip": torch.version.hip, "cuda": torch.version.cuda,
                      "devices": devices, "resolved": str(resolved),
                      "banner": device_banner_text(resolved)}, indent=2))
