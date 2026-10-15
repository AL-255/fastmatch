"""Distribution policy must not mistake PyTorch itself for a vendor runtime."""

import json

import pytest

from scripts.bundle_policy import exclusion_reason, trim_collection


@pytest.mark.parametrize("backend", ["cpu", "cuda-cu128", "rocm7.2", "mps"])
@pytest.mark.parametrize("library", ["libtorch_cpu.so", "libtorch_cuda.so", "libtorch_hip.so",
                                      "libc10_hip.so", "torch_cuda.dll", "c10_cuda.dll", "libgomp.so.1",
                                      "fbgemm.dll", "dnnl.dll", "caffe2_nvrtc.dll"])
def test_preserves_framework_and_cpu_libraries(backend, library):
    assert exclusion_reason(f"torch/lib/{library}", f"/venv/torch/lib/{library}", backend) is None


def test_external_runtime_removes_aliases_and_reports_without_host_paths(tmp_path):
    vendor = tmp_path / "torch" / "lib" / "librocblas.so.5"
    vendor.parent.mkdir(parents=True)
    vendor.write_bytes(b"vendor library")
    core = vendor.with_name("libtorch_hip.so")
    core.write_bytes(b"framework library")
    binaries = [("torch/lib/librocblas.so.5", str(vendor), "BINARY"),
                ("librocblas.so.5", "torch/lib/librocblas.so.5", "SYMLINK"),
                ("torch/lib/libtorch_hip.so", str(core), "BINARY")]
    kept, _ = trim_collection(binaries, [], "rocm7.2", tmp_path / "metadata")
    assert kept == [binaries[-1]]
    shipped = (tmp_path / "metadata" / "runtime-requirements.json").read_text()
    assert str(tmp_path) not in shipped
    assert json.loads(shipped)["external_libraries"] == ["librocblas.so.5"]
    assert json.loads((tmp_path / "metadata" / "packaging-report.json").read_text())["excluded_bytes"][
        "system accelerator runtime"] == vendor.stat().st_size


def test_keeps_dinov2_and_removes_unrelated_models_and_torch_tests():
    assert exclusion_reason("transformers/models/dinov2/modeling_dinov2.py", "", "cpu") is None
    assert exclusion_reason("transformers/models/auto/modeling_auto.py", "", "cpu") is None
    assert exclusion_reason("transformers/models/llama/modeling_llama.py", "", "cpu")
    assert exclusion_reason("torch/test/test_api", "", "cpu")
    assert exclusion_reason("torch/bin/protoc", "", "cpu")
    assert exclusion_reason("torch/bin/torch_shm_manager", "", "cpu") is None
