# Executable release builds

The `Build executables` GitHub Actions workflow builds six native bundles:

| Runner | Architecture | PyTorch runtime |
|---|---|---|
| Windows 2025 | x64 | CPU |
| Windows 2025 | x64 | CUDA 12.8 |
| Ubuntu 22.04 | x64 | CPU |
| Ubuntu 22.04 | x64 | CUDA 12.8 |
| Ubuntu 22.04 | x64 | ROCm 7.2 |
| macOS 15 | ARM64 | Apple MPS / CPU |

PyTorch 2.11 does not publish Intel macOS wheels; the macOS application targets
Apple Silicon. MPS is Apple's GPU backend, not the Neural Engine.

## Triggers and publication

- Pushing a version tag matching `vMAJOR.MINOR.PATCH` (optionally a prerelease
  suffix) builds every platform and publishes the matching GitHub release after
  every job succeeds.
- Pull requests touching the application, tests, packaging, or workflow run the
  same builds and upload temporary Actions artifacts, without publishing releases.
- `workflow_dispatch` also builds temporary artifacts for testing.

Each native job installs an explicit CPU/CUDA/ROCm wheel first, then pinned
runtime and build dependencies from `requirements-build.txt`. The DINOv2-small
weights are included, so semantic search works offline in the executable.

The source test suite runs before packaging. The frozen application's
`--smoke-test REPORT --device cpu` exercises Qt startup, license resources,
NCC/SSD/CCORR, fuzzy retrieval, and real offline pretrained inference. CUDA/ROCm
runtime versions are checked even on runners without GPUs. MPS hardware tests
run when a runner actually exposes a usable GPU. CPU smoke success alone is not
proof of GPU execution; run an explicit accelerator smoke on supported hardware:

```bash
FastMatch --smoke-test smoke-cuda.json --device cuda
FastMatch --smoke-test smoke-rocm.json --device rocm
FastMatch --smoke-test smoke-mps.json --device mps
```

CUDA/ROCm archives have a `-system` suffix and deliberately exclude vendor
runtime libraries and GPU kernel databases. Users install the corresponding
CUDA 12.8/cuDNN/NCCL or ROCm 7.2 stack separately; see
[GPU runtime installation](../README.md#gpu-runtime-installation).
PyTorch's own ABI-specific libraries and required framework dependencies
(MAGMA/AOTriton for ROCm) stay bundled; these are not supplied by the system SDK. CPU/macOS remain self-contained
apart from normal OS graphics libraries/frameworks.

`scripts/bundle_policy.py` removes vendor libraries after dependency analysis,
including aliases that would otherwise point at omitted files. It also removes
PyTorch tests/tools and unrelated Transformers model sources. Each build exports
`.runtime.json` (external library inventory) and `.packaging.json` (byte counts
and largest retained files). CI smoke tests use the build environment's installed
external GPU libraries through `LD_LIBRARY_PATH`/Windows DLL search directories;
those machine-specific paths are never included in the distributable. A GPU
package still needs its user-space libraries even to run in CPU fallback mode.

The publisher requires all six build artifacts, creates a draft release, uploads
one platform at a time, and publishes only after all uploads succeed. Only the
publisher has `contents: write`. A failed build creates no release; a failed new
release upload leaves a draft that can be completed by rerunning the job.

## Preparing a tag

Merge the tested changes into `main`, update the source version as appropriate,
then create and push a new tag. Existing release tags should stay fixed.

```bash
git switch main
git pull --ff-only origin main
git tag -a vX.Y.Z -m "FastMatch vX.Y.Z"
git push origin vX.Y.Z
```

The build embeds the tag version, commit hash and runtime in the application.
The About dialog reads those values without needing Git on the user's machine.
Prerelease tags are published as GitHub prereleases.
Write curated release notes in `docs/releases/vX.Y.Z.md` before tagging; the
publisher uses that file instead of generated commit notes when present.

## Local builds

Build on the target OS; PyInstaller is not a cross-compiler. Use Python 3.13 and
a clean virtual environment to keep unrelated packages out of the bundle.
Install the desired runtime from one of PyTorch's indexes before the pinned
requirements (use `cpu` on macOS, where the wheel also includes MPS support):

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-build.txt
python -m fastmatch --download-semantic-model
python scripts/build_release.py --label dev --backend cpu
```

For NVIDIA use the `cu128` index and `--backend cuda-cu128`; for AMD use the
`rocm7.2` index and `--backend rocm7.2` (Linux); for Apple use `--backend mps`.
Linux requires desktop graphics libraries; CI installs `libegl1`, `libopengl0`,
`libxcb-cursor0`, and `libxkbcommon-x11-0` before building.

Archives and checksums are written to `release-assets/`. Windows ships a ZIP
containing `FastMatch/FastMatch.exe`; macOS ships a ZIP containing `FastMatch.app`;
Linux ships a tar.gz containing `FastMatch/FastMatch`. Keep the full extracted
directory together: native libraries live alongside the executable.

## Large GPU downloads

GitHub release assets must be smaller than 2 GiB. GPU archives above the build's
1,900 MiB threshold are split into ordered 1 GiB pieces. The `.parts.json`
manifest records every part's size and SHA256, plus the complete archive hash.
Download all matching parts and the manifest into one directory, then use the
release's `assemble_release.py` or `assemble_release.ps1` helper to reconstruct
and verify the archive before extraction. A bad or missing part fails the
operation; an existing verified archive is preserved on failure.

Every build has a `.sha256` file covering the uploaded files and a `.smoke.json`
report. System-runtime GPU builds can still be large because PyTorch itself
contains accelerator-specific operators; inspect `.packaging.json` for details.

## Signing and writable data

macOS bundles use PyInstaller's ad-hoc signing; Apple notarization and Windows
Authenticode signing require maintainer certificates and are not configured.
Semantic model overrides use `FASTMATCH_SEMANTIC_MODEL`. In a frozen application,
an explicit model download writes to a user-owned application-data directory,
so it does not modify a signed or read-only application bundle.
