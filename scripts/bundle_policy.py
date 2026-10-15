"""Keep application libraries; leave accelerator SDKs to the host system."""

from pathlib import Path, PurePosixPath
import importlib.metadata
import json


def exclusion_reason(destination, source, backend):
    dest = destination.replace("\\", "/")
    src = source.replace("\\", "/")
    parts = PurePosixPath(dest).parts
    name = PurePosixPath(dest).name.lower()
    if dest == "torch/bin/torch_shm_manager":
        return None  # torch checks this helper during import, even for inference
    if dest.startswith(("torch/test/", "torch/bin/", "torch/include/", "torch/share/")):
        return "development and test files"
    if name.startswith(("libtorchbind_test", "libjitbackend_test", "libbackend_with_compiler",
                        "libaoti_custom_ops", "torchbind_test", "jitbackend_test")):
        return "development and test files"
    if "transformers/models/" in dest:
        model = dest.split("transformers/models/", 1)[1].split("/", 1)[0]
        # AutoTokenizer imports EncoderDecoderConfig while DINOv2's decorators
        # construct documentation at import time.
        if model not in ("__init__.py", "auto", "dinov2", "encoder_decoder"):
            return "unused model implementations"
    if backend not in ("cuda-cu128", "rocm7.2"):
        return None
    if any(p == "nvidia" or p.startswith(("_rocm_sdk", "rocm_sdk_core", "rocm_sdk_libraries"))
           for p in (*parts, *PurePosixPath(src).parts)):
        return "system accelerator runtime"
    # Some torch wheels place the entire vendor runtime under torch/lib.
    # Keep libtorch/libc10 and CPU dependencies: a system CUDA/ROCm SDK cannot
    # replace PyTorch's own ABI-specific libraries.
    if dest.startswith("torch/lib/") or "/torch/lib/" in src:
        if name.startswith(("libtorch", "torch_", "libc10", "c10", "libshm", "shm",
                            "libgomp", "libomp", "libiomp", "libmkl", "mkl_",
                            "libnvfuser", "nvfuser", "libstdc++", "libgcc")):
            return None
        vendor_name = name.removeprefix("lib")
        prefixes = ("cublas", "cudart", "cufft", "curand", "cusolver", "cusparse", "cudnn",
                    "cupti", "cufile", "cutensor", "nccl", "nvrtc", "nvjitlink", "nvtoolsext",
                    "nvshmem", "nvfatbin", "npp", "amd", "hip", "roc", "hsa", "miopen",
                    "rccl", "aqlprofile", "comgr", "drm_amdgpu")
        if vendor_name.startswith(prefixes) and (".so" in name or name.endswith((".dll", ".dylib"))):
            return "system accelerator runtime"
        if any(p.lower() in ("rocblas", "hipblaslt", "hipsparselt", "miopen") for p in parts):
            return "system accelerator runtime"
    return None


def trim_collection(binaries, datas, backend, output):
    """Filter PyInstaller TOCs and record the exact external dependencies."""
    totals, external, runtime_files, kept = {}, set(), {}, []
    result = []
    for collection in (binaries, datas):
        filtered = []
        for destination, source, kind in collection:
            reason = exclusion_reason(destination, source, backend)
            if kind == "SYMLINK":
                reason = reason or exclusion_reason(source, source, backend)
            path = Path(source)
            size = path.stat().st_size if path.is_file() else 0
            if reason:
                totals[reason] = totals.get(reason, 0) + size
                if reason == "system accelerator runtime" and kind != "SYMLINK":
                    if ".so" in path.name or path.suffix.lower() in (".dll", ".dylib"):
                        external.add(path.name)
                        runtime_files[PurePosixPath(destination.replace("\\", "/")).name] = str(path.resolve())
            else:
                filtered.append((destination, source, kind))
                if kind != "SYMLINK":
                    kept.append((size, destination))
        result.append(filtered)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    requirements = dict(backend=backend, external_libraries=sorted(external),
                        instructions="See README.md: GPU runtime installation")
    if backend in ("cuda-cu128", "rocm7.2"):
        requirements["build_runtime_packages"] = {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name", "").lower().startswith(("nvidia-", "cuda-", "rocm-", "amd-rocm"))
        }
    (output / "runtime-requirements.json").write_text(json.dumps(requirements, indent=2), encoding="utf-8")
    # Build-machine paths are used only for the CI smoke test, never shipped.
    (output / "runtime-files.json").write_text(json.dumps(runtime_files), encoding="utf-8")
    report = dict(excluded_bytes=totals, retained_bytes=sum(size for size, _ in kept),
                  largest_files=[dict(path=name, bytes=size) for size, name in sorted(kept, reverse=True)[:25]])
    (output / "packaging-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Bundle exclusions (MiB):", {key: round(value / 2**20, 1) for key, value in totals.items()})
    return tuple(result)
