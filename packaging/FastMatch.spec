# Native onedir bundles; CUDA/ROCm runtime libraries are installed on the host.
from pathlib import Path
import json
import sys

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

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

a = Analysis([str(root / "packaging" / "entrypoint.py")], pathex=[str(root)],
             datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             hookspath=[str(root / "packaging" / "hooks")],
             excludes=["torchvision", "torchaudio", "matplotlib", "IPython", "pytest",
                       "PyQt5", "PyQt6", "PySide2", "tensorflow", "jax", "scipy", "sklearn",
                       "triton", "torch.testing._internal"])
a.binaries, a.datas = trim_collection(a.binaries, a.datas, build_info["backend"], root / "build" / "metadata")
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
