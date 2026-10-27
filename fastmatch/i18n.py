"""English / Simplified Chinese UI, switchable without losing the open image."""
from __future__ import annotations
import re
from PySide6.QtCore import QSettings, QLocale, QSignalBlocker, QTranslator, QLibraryInfo
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QAbstractButton,
                              QGroupBox, QComboBox, QSpinBox, QMenu, QTableWidget, QStatusBar)

_language = "en"
_reverse: dict[str, str] = {}
_qt_translator = None
ZH = {
    "&MPS (Apple GPU)": "MPS（Apple GPU）(&M)",
    "Use the Apple GPU.": "使用 Apple GPU 加速。",
    "No working Apple MPS device was detected.": "未检测到可用的 Apple MPS 设备。",
    "Fuzzy instance search": "模糊实例搜索",
    "Semantic similarity": "语义相似度",
    "Fuzzy instance search — RootSIFT + geometry (altered / occluded copies)": "模糊实例搜索 — RootSIFT 与几何匹配（变形或遮挡副本）",
    "Semantic similarity — DINOv2 (visually related objects)": "语义相似度 — DINOv2（视觉相关对象）",
    "Strict feature matching — keypoints + pixel appearance (rotated / scaled)": "严格特征匹配 — 特征点与像素外观（旋转或缩放）",
    "Min matches": "最少匹配点",
    "Find altered copies with RootSIFT and geometric consistency. Tolerates appearance changes and partial occlusion; requires distinctive local details. Runs on CPU.": "通过 RootSIFT 和几何一致性查找变形副本，容忍外观变化和局部遮挡，需要明显的局部细节。使用 CPU 运行。",
    "Find visually related objects using local DINOv2 features. Results are approximate regions with the selected aspect ratio. Uses RGB and the selected engine device. Large images can take time.": "使用本地 DINOv2 特征查找视觉相关对象，结果为保持选区宽高比的近似区域。使用 RGB 与所选计算引擎，大图搜索可能需要较长时间。",
    "Minimum independent, geometrically consistent keypoints. Lower finds weaker copies; higher reduces false matches.": "独立且几何一致的特征点数量下限。降低可找到较弱的匹配，提高可减少误匹配。",
    "Search multiple scales (0.75×, 1×, 1.5×)": "搜索多种尺度（0.75×、1×、1.5×）",
    "Rectilinear select": "直角多边形选区",
    "Rectilinear (right-angle polygon) select: click to drop vertices (edges snap horizontal/vertical); click the first vertex or double-click / Enter to close, Esc to cancel.": "直角多边形选区：点击添加顶点，边缘自动保持水平或垂直；点击首个顶点、双击或按 Enter 闭合，按 Esc 取消。",
    "&File": "文件(&F)", "&View": "视图(&V)", "&Tools": "工具(&T)",
    "&Theme": "主题(&T)", "&Engine": "计算引擎(&E)", "&Help": "帮助(&H)",
    "Language": "语言 / Language", "&About FastMatch": "关于 FastMatch(&A)",
    "About FastMatch": "关于 FastMatch", "&Memory panel": "记录面板(&M)",
    "&Search panel": "搜索面板(&S)", "&XOR with background": "与背景反色(&X)",
    "&Auto (prefer GPU)": "自动（优先 GPU）(&A)", "&CUDA (GPU)": "CUDA（NVIDIA GPU）(&C)",
    "&ROCm (AMD GPU)": "ROCm（AMD GPU）(&R)", "C&PU": "CPU(&P)",
    "&System": "跟随系统(&S)", "&Light": "浅色(&L)", "&Dark": "深色(&D)",
    "CPU threads": "CPU 计算线程数", "0 matches": "0 个匹配", "Unlimited": "不限",
    "0 means unlimited. Positive values limit the number of matches after overlapping detections are merged.": "0 表示不限数量；正数表示合并重叠匹配后的结果数量上限。",
    "A search is still finishing; please wait and try again.": "搜索尚未结束，请稍后重试。",
    "Add to Memory": "添加到记录", "Auto run": "自动搜索", "Busy": "正在处理",
    "Calibrate": "标定", "Calibrate scale": "标定比例尺", "Calibration cleared.": "标定已清除。",
    "Calibrate: drag a line along a span of known physical length.": "标定：沿已知实际长度拖动一条线。",
    "Calibration span too short — try a longer drag.": "标定线太短，请拖动更长的线。",
    "Channel mode": "颜色通道", "Clear all memory entries.": "清除全部记录。",
    "Clear calibration": "清除标定", "Clear matches": "清除匹配", "Clear measurement": "清除测量",
    "Clear the result boxes and the current selection (draw a new region to search again).": "清除匹配框和当前选区，重新框选即可再次搜索。",
    "Close Image": "关闭图像", "Close Memory": "关闭记录", "Close the current image and clear the view.": "关闭当前图像并清空视图。",
    "Could not read a positive length from that input.": "请输入有效的正数长度。",
    "Cursor: (-, -)": "光标：(-, -)", "Degenerate calibration; please try again.": "标定无效，请重试。",
    "Delete the selected memory entries.": "删除所选记录。", "Detector": "特征检测器",
    "Distribution of match scores. Bars at/above the threshold are shown; bars below it are filtered out. Bin count: View ▸ Score histogram.": "匹配分数分布：显示高于阈值的结果，过滤低于阈值的结果。可在“视图”中调整直方图分箱数。",
    "Double-click an entry to restore its boxes and selection.": "双击记录以恢复匹配框和选区。",
    "Drag a line of known physical length, then enter that length to set the pixel↔physical scale (uses the longer of the horizontal/vertical span; the first point becomes the physical origin).": "拖动已知实际长度的线段，再输入长度以建立像素与实际尺寸的比例。采用较长的水平或垂直跨度，起点作为实际坐标原点。",
    "Drag a line on the image to measure its physical distance.": "在图像上拖动线段以测量实际距离。",
    "Draw a selection box first.": "请先框选一个区域。",
    "Draw box outlines XORed with the pixels underneath, so they stay visible on any background.": "让匹配框边缘与背景像素反色，以适应不同背景。",
    "Enable flipping": "搜索镜像", "Enable rotation": "搜索旋转",
    "Also search the template rotated 90°, 180°, and 270°.": "同时搜索旋转 90°、180°、270° 的模板。",
    "Also search mirrored templates. Combined with rotation this adds the diagonal reflections too (8 orientations total).": "同时搜索镜像模板；配合旋转可覆盖全部 8 种方向。",
    "Feature matching": "特征匹配", "Feature matching runs on the CPU via OpenCV (independent of the engine GPU device).": "特征匹配通过 OpenCV 在 CPU 上执行，不受 GPU 引擎选择影响。",
    "Fit": "适应窗口", "GPU only — CPU is locked to 1.0×.": "仅 GPU 可用；CPU 固定为 1.0 倍。",
    "Generate a labelled sample, search one motif, and report recall.": "生成带标注的样本，搜索图案并报告召回率。",
    "Give the selected memory entry a custom name.": "为所选记录设置名称。",
    "Histogram &bins": "直方图分箱(&B)", "Image closed.": "图像已关闭。",
    "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)": "图像 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;所有文件 (*)",
    "Keypoint detector: ORB (fast binary, default), AKAZE, or SIFT. Feature matching runs on CPU via OpenCV.": "检测器可选 ORB（快速，默认）、AKAZE 或 SIFT，通过 OpenCV 在 CPU 上计算。",
    "License": "许可证", "Line &width": "线宽(&W)", "Main": "主工具栏", "Match &boxes": "匹配框(&B)",
    "Luminance: single BT.601 luma plane (faster, less VRAM). RGB / YCbCr: weighted multi-channel matching (per-channel weights below) — applies to NCC/SSD/CCORR and feature matching's appearance verification.": "亮度：单一 BT.601 亮度通道，速度更快且显存占用更少。RGB / YCbCr：按下方权重进行多通道匹配，适用于 NCC、SSD、CCORR 和特征匹配的外观验证。",
    "Match parameters": "匹配参数", "Max results": "结果上限", "Measure": "测量",
    "Measure: drag a line to measure its distance.": "测量：拖动线段以测量距离。", "Measurement cleared.": "测量已清除。",
    "Memory": "匹配记录", "Method": "匹配方法", "Min matches": "最少匹配点",
    "Minimum match score to display. This filters live (no re-run).": "显示结果的最低匹配分数；实时过滤，无需重新搜索。",
    "Minimum number of verified feature matches required to accept one instance. Lower finds more (and weaker) instances; higher is stricter.": "接受一个实例所需的最少验证匹配点数。降低可找到更多结果，提高则更严格。",
    "Multi-scale search is GPU-only; CPU is locked to 1.0× to keep queries responsive.": "多尺度搜索仅用于 GPU；CPU 固定为 1.0 倍以保持响应速度。",
    "Name:": "名称：", "No image": "未打开图像", "Open &Recent": "最近打开(&R)",
    "Open Recent": "最近打开", "Open Image…": "打开图像…", "Open Memory": "打开记录", "Open Memory…": "打开记录…",
    "Open an image first.": "请先打开图像。", "Open an image to search (PNG / JPEG / TIFF / BMP).": "打开需要搜索的图像（PNG / JPEG / TIFF / BMP）。",
    "Open failed": "打开失败", "Open image": "打开图像", "Open image…": "打开图像…", "Open source image?": "打开原始图像？",
    "Orientation": "方向", "Press Space to enter focus mode": "按空格键进入专注模式",
    "Quit": "退出", "Remove": "删除", "Rename": "重命名", "Rename entry": "重命名记录", "Rename…": "重命名…",
    "Run": "开始搜索", "Run a match first to add to Memory.": "请先完成搜索，再添加到记录。",
    "Run the search on the current selection.": "搜索与当前选区相似的图案。",
    "Save Memory": "保存记录", "Save Memory As": "记录另存为", "Save Memory As…": "记录另存为…", "Save failed": "保存失败",
    "Save the current selection and its matches as a memory entry.": "将当前选区及匹配结果保存为一条记录。",
    "Scale search": "尺度搜索", "Search": "搜索", "Search multiple scales": "搜索多个尺度",
    "Select a memory entry to rename.": "请选择需要重命名的记录。", "Select mode": "框选模式", "Pan mode": "平移模式",
    "Selection set — press Run to search.": "选区已设置，点击“开始搜索”。", "Self-test": "自检", "Self-test error": "自检错误", "Threshold": "相似度阈值",
    "Toggle between Select (draw region) and Pan (drag to move).": "切换框选区域与拖动平移模式。",
    "Toggle between Select (draw region) and Pan (drag to move). While selecting: Shift+drag adds another example of the same structure, Ctrl+drag marks something that must not match. Right-click a numbered example to delete it.": "切换框选区域与拖动平移模式。框选时：按住 Shift 拖框可再添加一个同类样本，按住 Ctrl 拖框可标记“不应匹配”的区域。右键点击带编号的样本框可删除它。",
    "Clear examples": "清空样本",
    "Examples": "样本", "&Examples panel": "样本面板(&E)",
    "#": "编号", "Type": "类型", "Position": "位置", "Size": "大小",
    "Positive": "正样本", "Negative": "负样本", "Delete selected": "删除所选",
    "Delete the selected examples (Delete key). Ctrl/Shift+click selects several.": "删除所选样本（Delete 键）。按住 Ctrl/Shift 点击可多选。",
    "No examples. Shift+drag adds one, Ctrl+drag a negative.": "暂无样本。按住 Shift 拖框添加正样本，按住 Ctrl 拖框添加负样本。",
    "Remove the extra examples (Shift+drag / Ctrl+drag boxes) and search with the selection alone.": "删除额外添加的样本（Shift/Ctrl 拖出的框），只用当前选区搜索。",
    "When on, the search runs automatically as you draw a selection or change settings. When off, click Run to search.": "启用后，框选或更改设置会自动搜索；关闭后需点击“开始搜索”。",
    "Zoom to fit the whole image (F).": "缩放至完整显示图像（F）。",
    "Use the GPU when available, otherwise fall back to the CPU.": "可用时使用 GPU，否则回退 CPU。",
    "Force the CUDA GPU backend.": "使用 NVIDIA CUDA GPU。", "No working CUDA device was detected on this machine.": "未检测到可用的 CUDA 设备。",
    "Use AMD ROCm / HIP.": "使用 AMD ROCm / HIP 加速。", "Requires compatible AMD drivers and ROCm PyTorch.": "需要兼容的 AMD 驱动和 ROCm 版 PyTorch。",
    "Force the CPU backend (slower, but always available).": "使用 CPU 计算后端。",
    "Luminance (single BT.601 luma plane)": "亮度（单一 BT.601 通道）",
    "RGB (weighted R, G, B channels)": "RGB（R、G、B 加权）", "YCbCr (weighted Y, Cb, Cr channels)": "YCbCr（Y、Cb、Cr 加权）",
    "NCC — normalized cross-correlation (textured, illumination-robust)": "NCC — 归一化互相关（纹理图案、耐光照变化）",
    "SSD — squared difference (flat / low-texture / exact appearance)": "SSD — 平方差（平坦区域、低纹理、精确外观）",
    "CCORR — cosine cross-correlation (bright / high-energy templates; no mean subtraction)": "CCORR — 余弦互相关（亮色或高能量模板，不减均值）",
    "Feature matching — ORB keypoints + appearance verify (rotated / scaled)": "特征匹配 — ORB 特征点与外观验证（旋转或缩放）",
    "Label": "名称", "Channel": "通道", "Selection": "选区", "Occurrences": "出现次数",
    "Score": "分数", "Orientations": "方向", "Focus mode — Space to exit": "专注模式 — 按空格键退出",
    "Running on CPU: install a compatible AMD ROCm or NVIDIA CUDA PyTorch build for GPU acceleration.": "当前使用 CPU；安装兼容的 AMD ROCm 或 NVIDIA CUDA PyTorch 可启用 GPU 加速。",
    "ROCm is compiling GPU kernels for this selection size; the first search can take a while, later ones are fast. Please wait.": "ROCm 正在为此选区尺寸编译 GPU 内核，首次搜索可能需要一些时间，之后会很快，请稍候。",
}

def language() -> str:
    return _language

def load_language() -> str:
    default = "zh" if QLocale.system().language() == QLocale.Language.Chinese else "en"
    value = QSettings("FastMatch", "FastMatch").value("ui/language", default)
    return value if value in ("en", "zh") else default

def tr(text: str) -> str:
    source = _reverse.get(text, text)
    if _language == "en":
        return source
    result = ZH.get(source, source)
    if result == source:
        match = re.fullmatch(r"(\d+) threads( \(all cores\))?", source)
        if match:
            result = f"{match[1]} 个线程" + ("（全部核心）" if match[2] else "")
        elif re.match(r"^\d+ matches", source):
            result = re.sub(r"^(\d+) matches", r"\1 个匹配", source)
        elif re.match(r"^\d+ orientations?:", source):
            result = re.sub(r"^(\d+) orientations?:", r"\1 种方向：", source)
        elif match := re.fullmatch(r"Examples: (\d+) positive, (\d+) negative", source):
            result = f"样本：正 {match[1]} 个，负 {match[2]} 个"
        elif match := re.fullmatch(r"Delete example #(\d+)", source):
            result = f"删除样本 #{match[1]}"
        elif match := re.fullmatch(r"Delete negative example ×(\d+)", source):
            result = f"删除负样本 ×{match[1]}"
        elif source.startswith("Matches shown: "):
            result = source.replace("Matches shown: ", "显示匹配数：", 1)
        elif source.startswith("Memory (") or source.startswith("Memory —"):
            result = source.replace("Memory", "匹配记录", 1).replace(" entries", " 条记录").replace(" entry", " 条记录")
        elif source.endswith(" channel weights"):
            result = source.replace(" channel weights", " 通道权重")
        elif source.endswith(" (sum 1.00)"):
            result = source.replace(" (sum 1.00)", "（总和 1.00）")
        elif source.startswith("Engine: "):
            result = source.replace("Engine: ", "计算引擎：", 1).replace("CPU (slow) - install a compatible AMD ROCm or NVIDIA CUDA PyTorch build for GPU acceleration", "CPU — 安装兼容的 AMD ROCm 或 NVIDIA CUDA PyTorch 以启用 GPU 加速")
        else:
            for en, zh in (("Cursor: ", "光标："), ("Match failed: ", "匹配失败："),
                           ("Selection: ", "选区："), ("Loaded ", "已加载 "), ("Distance: ", "距离："),
                           ("Could not load image:\n", "无法加载图像：\n"), ("Could not open memory:\n", "无法打开记录：\n"),
                           ("Could not save memory:\n", "无法保存记录：\n"), ("This image is no longer available:\n", "图像已不可用：\n"),
                           ("NCC —", "NCC（归一化互相关）—"), ("SSD —", "SSD（平方差）—"),
                           ("CCORR —", "CCORR（互相关）—"), ("Feature matching —", "特征匹配 —")):
                if source.startswith(en):
                    result = zh + source[len(en):]
                    break
    if result != source:
        _reverse[result] = source
    return result

def refresh_ui(root: QWidget) -> None:
    """Translate presentation only; preserve combo data, user text and selections."""
    for obj in [root, *root.findChildren(QWidget), *root.findChildren(QAction)]:
        with QSignalBlocker(obj):
            for getter, setter in (("toolTip", "setToolTip"), ("windowTitle", "setWindowTitle")):
                if hasattr(obj, getter):
                    old = getattr(obj, getter)()
                    new = tr(old)
                    if new != old:
                        getattr(obj, setter)(new)
            if isinstance(obj, (QLabel, QAbstractButton, QAction)):
                obj.setText(tr(obj.text()))
            if isinstance(obj, (QGroupBox, QMenu)):
                obj.setTitle(tr(obj.title()))
            if isinstance(obj, QSpinBox):
                obj.setSpecialValueText(tr(obj.specialValueText()))
            if isinstance(obj, QComboBox):
                for index in range(obj.count()):
                    obj.setItemText(index, tr(obj.itemText(index)))
            if isinstance(obj, QTableWidget):
                for index in range(obj.columnCount()):
                    item = obj.horizontalHeaderItem(index)
                    if item is not None:
                        item.setText(tr(item.text()))
            if isinstance(obj, QStatusBar) and obj.currentMessage():
                obj.showMessage(tr(obj.currentMessage()))

def set_language(value: str, *, persist: bool = True) -> None:
    global _language, _qt_translator
    if value not in ("en", "zh"):
        raise ValueError(value)
    _language = value
    if persist:
        QSettings("FastMatch", "FastMatch").setValue("ui/language", value)
    app = QApplication.instance()
    if app is not None:
        if _qt_translator is not None:
            app.removeTranslator(_qt_translator)
            _qt_translator = None
        if value == "zh":
            translator = QTranslator(app)
            if translator.load("qtbase_zh_CN", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
                app.installTranslator(translator)
                _qt_translator = translator
        for root in app.topLevelWidgets():
            refresh_ui(root)
