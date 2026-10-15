"""Build, exercise and archive one native FastMatch executable.

Install requirements-build.txt after installing the desired torch wheel, then:
  python scripts/build_release.py --backend cpu --label dev
Tag builds use --label vX.Y.Z. GPU archives over 2 GiB are split into 1 GiB
parts for GitHub release assets, with per-part and full-archive SHA256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent.parent
ASSET_LIMIT = 1900 * 1024 * 1024
PART_SIZE = 1024 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_archive(path: Path, limit=ASSET_LIMIT, part_size=PART_SIZE) -> list[Path]:
    if path.stat().st_size < limit:
        return [path]
    full_hash, size = sha256(path), path.stat().st_size
    parts, records = [], []
    with path.open("rb") as source:
        index = 1
        while source.tell() < size:
            part = path.with_name(f"{path.name}.part{index:03d}")
            remaining = part_size
            with part.open("wb") as target:
                while remaining and (block := source.read(min(8 * 1024 * 1024, remaining))):
                    target.write(block)
                    remaining -= len(block)
            records.append(dict(name=part.name, size=part.stat().st_size, sha256=sha256(part)))
            parts.append(part)
            index += 1
    manifest = path.with_name(path.name + ".parts.json")
    manifest.write_text(json.dumps(dict(archive=path.name, size=size, sha256=full_hash, parts=records),
                                   indent=2), encoding="utf-8")
    path.unlink()  # avoid retaining both the whole archive and its parts
    return [*parts, manifest]


def verify_backend(backend):
    import torch
    if backend == "cpu":
        assert not torch.version.cuda and not torch.version.hip, "CPU build contains a GPU wheel"
    elif backend == "cuda-cu128":
        assert torch.version.cuda and torch.version.cuda.startswith("12.8"), "Expected CUDA 12.8 wheel"
    elif backend == "rocm7.2":
        assert torch.version.hip and torch.version.hip.startswith("7.2"), "Expected ROCm 7.2 wheel"
    elif backend == "mps":
        assert sys.platform == "darwin" and torch.backends.mps.is_built(), "Expected Apple MPS wheel"


def stage_external_runtime(files, directory):
    """Expose only omitted libraries, never the build's original libtorch files."""
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for name, filename in files.items():
        source = Path(filename)
        aliases = {name, source.name}
        if ".so" in source.name:
            for candidate in source.parent.glob(source.name.split(".so", 1)[0] + ".so*"):
                if candidate.resolve() == source.resolve():
                    aliases.add(candidate.name)
        for alias in aliases:
            target = directory / alias
            if target.exists():
                continue
            try:
                if sys.platform == "win32":
                    os.link(source, target)
                else:
                    target.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, target)
    return directory


def build(label: str, backend: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", label):
        raise ValueError("Invalid release/build label")
    verify_backend(backend)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    version_file = (ROOT / "fastmatch" / "__init__.py").read_text(encoding="utf-8")
    version = re.search(r'^__version__ = "([^"]+)"', version_file, re.M).group(1)
    if label.startswith("v"):
        if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", label):
            raise ValueError("Release tags must be vMAJOR.MINOR.PATCH, optionally with a prerelease suffix")
        version = label[1:]
    else:
        version += f"+dev.{commit[:7]}"
    metadata = ROOT / "build" / "metadata" / "_build_info.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(dict(version=version, commit=commit, backend=backend)), encoding="utf-8")
    model = ROOT / "models" / "dinov2-small"
    for filename in ("config.json", "model.safetensors"):
        if not (model / filename).is_file():
            raise FileNotFoundError("Run python -m fastmatch --download-semantic-model before packaging")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "packaging/FastMatch.spec"], cwd=ROOT, check=True)
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    architecture = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    bundle = ROOT / "dist" / ("FastMatch.app" if sys.platform == "darwin" else "FastMatch")
    executable = bundle / ("Contents/MacOS/FastMatch" if sys.platform == "darwin"
                           else "FastMatch.exe" if sys.platform == "win32" else "FastMatch")
    # Run the packaged code, with pretrained inference forced offline.
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "HF_HUB_OFFLINE": "1",
                   "HF_HUB_DISABLE_PROGRESS_BARS": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    environment.pop("FASTMATCH_SEMANTIC_MODEL", None)
    runtime_files = json.loads((ROOT / "build" / "metadata" / "runtime-files.json").read_text())
    if backend in ("cuda-cu128", "rocm7.2"):
        if not runtime_files:
            raise RuntimeError("GPU build did not identify any external runtime libraries")
        # Test the trimmed bundle against external installed libraries. Keep
        # these build-machine paths out of the distributable.
        runtime = stage_external_runtime(runtime_files, ROOT / "build" / "external-runtime")
        variable = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
        environment[variable] = os.pathsep.join([str(runtime), environment.get(variable, "")])
        environment["FASTMATCH_GPU_LIBRARY_PATH"] = str(runtime)
    report = ROOT / "build" / f"smoke-{system}-{backend}.json"
    subprocess.run([str(executable), "--smoke-test", str(report), "--device", "cpu"],
                   cwd=ROOT / "dist", env=environment, check=True, timeout=240)
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["ok"] and result["version"] == version, result
    # GitHub's macOS runners may not expose MPS. Check it on hardware whenever
    # available; never claim accelerator execution from a CPU-only runner.
    if backend == "mps":
        import torch
        if torch.backends.mps.is_available():
            gpu_report = report.with_name("smoke-mps-hardware.json")
            subprocess.run([str(executable), "--smoke-test", str(gpu_report), "--device", "mps"],
                           cwd=ROOT / "dist", env=environment, check=True, timeout=240)
    assets = ROOT / "release-assets"
    assets.mkdir(exist_ok=True)
    name = f"FastMatch-{label}-{system}-{architecture}-{backend}"
    if backend in ("cuda-cu128", "rocm7.2"):
        name += "-system"
    if sys.platform == "darwin":
        archive = assets / f"{name}.zip"
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(bundle), str(archive)], check=True)
    elif sys.platform == "win32":
        archive = Path(shutil.make_archive(str(assets / name), "zip", root_dir=bundle.parent, base_dir=bundle.name))
    else:
        archive = assets / f"{name}.tar.gz"
        if shutil.which("pigz"):
            # AMD runtimes contain many gigabytes of kernels. Compress using
            # hosted-runner cores rather than spending most of CI in gzip.
            with archive.open("wb") as output:
                producer = subprocess.Popen(["tar", "-C", str(bundle.parent), "-cf", "-", bundle.name],
                                            stdout=subprocess.PIPE)
                try:
                    subprocess.run(["pigz", "-6", "-p", str(min(4, os.cpu_count() or 2))],
                                   stdin=producer.stdout, stdout=output, check=True)
                finally:
                    producer.stdout.close()
                if producer.wait() != 0:
                    raise RuntimeError("Tar archive creation failed")
        else:
            with tarfile.open(archive, "w:gz", compresslevel=6) as stream:
                stream.add(bundle, arcname=bundle.name)
    if os.environ.get("CI") == "true":
        # The validated executable now lives in the archive. Free its onedir
        # copy before splitting multi-gigabyte GPU archives on hosted runners.
        shutil.rmtree(bundle)
    # Ship smoke evidence next to each native bundle.
    smoke_asset = assets / f"{name}.smoke.json"
    shutil.copy2(report, smoke_asset)
    size_asset = assets / f"{name}.packaging.json"
    shutil.copy2(ROOT / "build" / "metadata" / "packaging-report.json", size_asset)
    runtime_asset = assets / f"{name}.runtime.json"
    shutil.copy2(ROOT / "build" / "metadata" / "runtime-requirements.json", runtime_asset)
    outputs = [*split_archive(archive), smoke_asset, size_asset, runtime_asset]
    checksums = assets / f"{name}.sha256"
    checksums.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in outputs), encoding="utf-8")
    print(f"Packaged {name}; frozen smoke passed. Assets: {assets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda-cu128", "rocm7.2", "mps"), required=True)
    args = parser.parse_args()
    build(args.label, args.backend)
