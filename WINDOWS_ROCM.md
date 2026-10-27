# Windows EXE 与 AMD ROCm

双击项目根目录的 `FastMatch.exe`。可以把图片拖到 EXE 上打开。
这是本地启动器，会使用同目录的 `.venv` 和源代码；不能单独复制 EXE 到另一台电脑。
启动失败时查看 `logs/fastmatch.log`。
如果 `.venv` 尚无 PyTorch，但目录中有 `.cpu-runtime`，启动器会先使用该 CPU 备用环境。

## 后端

- Engine → Auto：优先使用通过实际计算测试的 GPU，否则使用 CPU。
- Engine → ROCm (AMD GPU)：需要 HIP 版 PyTorch 和兼容的 AMD 驱动。
- 状态栏显示 `Engine: ROCm / HIP (AMD Radeon RX 7900 XT)` 才表示已启用 AMD 加速。
- PyTorch 的 ROCm 后端沿用 `torch.cuda` API，内部设备名仍是 `cuda`，这是正常行为。
- NCC、SSD、CCORR 共用 GPU 计算路径；特征匹配仍使用 OpenCV CPU。

## RDNA 1 / 2 / 3 / 4

应用不按显卡代际设置白名单，也不伪造 GPU 架构。只要安装的 PyTorch/ROCm
包含该 GPU 的内核并通过启动检查，就可使用相同的匹配算法。
这不等于 AMD 官方运行库支持每一代的所有型号。

| 代际 | 本软件的处理 |
|---|---|
| RDNA 1 | 允许兼容运行库；随附官方 Windows 环境不保证支持，无可用内核时回退 CPU |
| RDNA 2 | 允许兼容运行库；需按具体型号核对 AMD 支持矩阵，不保证整代支持 |
| RDNA 3 | RX 7900 XT 为主要目标；其他型号需核对驱动和运行库支持 |
| RDNA 4 | 共用 HIP 路径；仍需相应型号受运行库支持 |

未在四代真实显卡上全面测试。不能通过本软件补齐驱动或 ROCm 缺失的硬件支持。

## 重建运行环境

使用 Python 3.12 x64。在项目目录运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-rocm-windows.txt
.\.venv\Scripts\python.exe -m fastmatch --device rocm
.\.venv\Scripts\python.exe scripts/check_backend.py
```

ROCm 清单固定为 AMD 官方 Windows 7.2.1 / PyTorch 2.9.1，便于重现。
请根据 AMD 文档安装兼容驱动。本软件不会自动更改系统显卡驱动。
不要再安装原版 `requirements.txt`，其中的 torch 最低版本可能替换 HIP 构建。
其他运行库可单独安装到此 `.venv`，EXE 不需要重新编译。

重新生成启动器：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-launcher.ps1
```

Linux 也可安装对应型号的 ROCm PyTorch，再运行 `python -m fastmatch --device rocm`；
Windows EXE 仅用于 Windows。

参考：
- [AMD Windows 安装](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
- [AMD Windows 支持矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
- [PyTorch HIP API](https://docs.pytorch.org/docs/stable/notes/hip.html)
