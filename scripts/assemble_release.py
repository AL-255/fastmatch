"""Join large release archives and verify every part plus the final SHA256.

Usage: python assemble_release.py FastMatch-vX.Y.Z-linux-x64-rocm7.2.tar.gz.parts.json
"""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(manifest_path: Path):
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.resolve().parent
    name = data["archive"]
    if Path(name).name != name:
        raise ValueError("Archive must be a filename")
    output = root / name
    temporary = root / (name + ".assembling")
    digest, size = hashlib.sha256(), 0
    try:
        with temporary.open("wb") as target:
            for part in data["parts"]:
                if Path(part["name"]).name != part["name"]:
                    raise ValueError("Part must be a filename")
                part_digest, part_size = hashlib.sha256(), 0
                with (root / part["name"]).open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        part_digest.update(block)
                        digest.update(block)
                        target.write(block)
                        part_size += len(block)
                if part_digest.hexdigest() != part["sha256"] or part_size != part["size"]:
                    raise ValueError(f"Corrupt part: {part['name']}")
                size += part_size
        if digest.hexdigest() != data["sha256"] or size != data["size"]:
            raise ValueError("Assembled archive checksum mismatch")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    print(assemble(parser.parse_args().manifest))
