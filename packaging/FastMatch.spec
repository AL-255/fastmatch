# Native onedir bundles; CUDA/ROCm runtime libraries are installed on the host.
from pathlib import Path
import json
import sys
import importlib.util

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
sys.path.insert(0, str(root))
from scripts.bundle_policy import trim_collection
build_info = json.loads((root / "build" / "metadata" / "_build_info.json").read_text(encoding="utf-8"))
datas = [(str(root / "LICENSE"), "."), (str(root / "README.md"), "."),
         (str(root / "readme_cn.md"), "."),
         (str(root / "build" / "metadata" / "_build_info.json"), "fastmatch"),
         (str(root / "models" / "dinov2-small" / "config.json"), "models/dinov2-small"),
         (str(root / "models" / "dinov2-small" / "model.safetensors"), "models/dinov2-small")]
datas += collect_data_files("transformers", include_py_files=True)
datas += collect_data_files("huggingface_hub", include_py_files=True)
if (root / "models" / "dinov2-small" / "README.md").is_file():
    datas.append((str(root / "models" / "dinov2-small" / "README.md"), "models/dinov2-small"))
binaries = []
hiddenimports = ["transformers.models.dinov2.modeling_dinov2",
                 "transformers.models.dinov2.configuration_dinov2"]
for distribution in ("transformers", "torch", "huggingface-hub", "safetensors", "numpy"):
    datas += copy_metadata(distribution)

# AMD's Windows wheels discover DLLs and kernel databases relative to these
# packages. Preserve their layout in the self-contained Windows ROCm build.
if build_info["backend"] == "rocm7.2.1-bundled":
    for module in ("rocm_sdk", "_rocm_sdk_core", "_rocm_sdk_libraries_custom"):
        spec = importlib.util.find_spec(module)
        if spec is None:
            raise RuntimeError(f"Missing Windows ROCm package: {module}")
        hiddenimports += collect_submodules(module, filter=lambda name: ".tests" not in name)
        datas.append((str(Path(spec.origin).parent), module))
    for distribution in ("rocm", "rocm-sdk-core", "rocm-sdk-libraries-custom"):
        datas += copy_metadata(distribution)

a = Analysis([str(root / "packaging" / "entrypoint.py")], pathex=[str(root)],
             datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             hookspath=[str(root / "packaging" / "hooks")],
             excludes=["torchvision", "torchaudio", "matplotlib", "IPython", "pytest",
                       "PyQt5", "PyQt6", "PySide2", "tensorflow", "jax", "scipy", "sklearn",
                       "triton"])
a.binaries, a.datas = trim_collection(a.binaries, a.datas, build_info["backend"], root / "build" / "metadata")
if sys.platform == "win32":
    # Do not shadow Windows' ICU API with an unrelated developer PATH DLL.
    a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() != "icuuc.dll"]
a.datas += [("runtime-requirements.json", str(root / "build" / "metadata" / "runtime-requirements.json"), "DATA")]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FastMatch",
          console=sys.platform not in ("win32", "darwin"), debug=False, upx=False)
collection = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="FastMatch")
if sys.platform == "darwin":
    app = BUNDLE(collection, name="FastMatch.app", bundle_identifier="com.al255.fastmatch",
                 info_plist={"CFBundleShortVersionString": build_info["version"].split("+")[0].split("-")[0],
                             "CFBundleVersion": build_info["version"].split("+")[0].split("-")[0],
                             "NSHighResolutionCapable": True})
