"""Configure Windows' DLL search before importing a frozen GPU PyTorch build."""

import os
from pathlib import Path
import sys

_dll_handles = []


def configure_runtime():
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    candidates = os.environ.get("FASTMATCH_GPU_LIBRARY_PATH", "").split(os.pathsep)
    for variable in ("CUDA_PATH_V12_8", "CUDA_PATH", "CUDNN_PATH"):
        if os.environ.get(variable):
            candidates.extend(str(Path(os.environ[variable]) / suffix)
                              for suffix in ("bin", "bin/12.8", "bin/12.9"))
    candidates += os.environ.get("PATH", "").split(os.pathsep)
    for directory in dict.fromkeys(candidates):
        if directory and Path(directory).is_dir():
            # Handles must stay alive while torch loads libraries dynamically.
            _dll_handles.append(os.add_dll_directory(str(Path(directory).resolve())))
