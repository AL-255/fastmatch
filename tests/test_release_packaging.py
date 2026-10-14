"""Archive splitting/reassembly must detect corruption before shipping binaries."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.assemble_release import assemble
from scripts.build_release import sha256, split_archive


def test_large_archive_round_trip_and_corrupt_part(tmp_path):
    archive = tmp_path / "FastMatch.tar.gz"
    payload = bytes(range(256)) * 40
    archive.write_bytes(payload)
    expected = sha256(archive)
    outputs = split_archive(archive, limit=2048, part_size=1024)
    manifest = outputs[-1]
    assert not archive.exists()
    assert len(outputs) == 11
    restored = assemble(manifest)
    assert restored.read_bytes() == payload
    assert sha256(restored) == expected
    outputs[0].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt part"):
        assemble(manifest)
    assert restored.read_bytes() == payload  # an existing good archive survives
    assert not restored.with_name(restored.name + ".assembling").exists()


def test_small_archive_is_not_split(tmp_path):
    archive = tmp_path / "FastMatch.zip"
    archive.write_bytes(b"small")
    assert split_archive(archive, limit=1024) == [archive]


def test_reassembly_rejects_path_traversal(tmp_path):
    manifest = tmp_path / "bad.parts.json"
    manifest.write_text(json.dumps(dict(archive="../outside.zip", parts=[])))
    with pytest.raises(ValueError, match="filename"):
        assemble(manifest)


@pytest.mark.skipif(not (shutil.which("pwsh") or shutil.which("powershell")),
                    reason="PowerShell is not installed")
def test_powershell_reassembly_and_corruption(tmp_path):
    archive = tmp_path / "FastMatch.zip"
    payload = bytes(range(256)) * 16
    archive.write_bytes(payload)
    parts = split_archive(archive, limit=1024, part_size=1024)
    helper = Path(__file__).resolve().parent.parent / "scripts" / "assemble_release.ps1"
    command = [shutil.which("pwsh") or shutil.which("powershell"), "-NoProfile", "-File",
               str(helper), "-Manifest", str(parts[-1])]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert archive.read_bytes() == payload
    parts[0].write_bytes(b"bad")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert archive.read_bytes() == payload
