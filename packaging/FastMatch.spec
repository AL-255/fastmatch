# Native onedir bundles: avoid extracting multi-gigabyte GPU runtimes at launch.
from pathlib import Path
import importlib.metadata
import json
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata, collect_submodules

root = Path(SPECPATH).parent
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

# ROCm 7.x can ship its runtime in SDK wheels, rather than torch/lib. Preserve
# their package layout (and metadata) so the SDK's loader can find its libraries.
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name", "")
    if name.lower().startswith(("rocm", "amd-rocm", "nvidia-")):
        datas += copy_metadata(name)
        for package in (distribution.read_text("top_level.txt") or "").splitlines():
            if package and not package.startswith("#"):
                datas += collect_data_files(package)
                binaries += collect_dynamic_libs(package, search_patterns=["*.so", "*.so.*", "*.dll", "*.dylib"])
                hiddenimports += collect_submodules(package)

a = Analysis([str(root / "packaging" / "entrypoint.py")], pathex=[str(root)],
             datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             hookspath=[str(root / "packaging" / "hooks")],
             excludes=["torchvision", "torchaudio", "matplotlib", "IPython", "pytest",
                       "PyQt5", "PyQt6", "PySide2", "tensorflow", "jax", "scipy", "sklearn"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FastMatch",
          console=sys.platform not in ("win32", "darwin"), debug=False, upx=False)
collection = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="FastMatch")
if sys.platform == "darwin":
    app = BUNDLE(collection, name="FastMatch.app", bundle_identifier="com.al255.fastmatch",
                 info_plist={"CFBundleShortVersionString": build_info["version"].split("+")[0].split("-")[0],
                             "CFBundleVersion": build_info["version"].split("+")[0].split("-")[0],
                             "NSHighResolutionCapable": True})
