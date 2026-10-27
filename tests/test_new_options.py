import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
import torch
from PySide6.QtWidgets import QApplication
from fastmatch import cpu, i18n
from fastmatch.app import MainWindow
from fastmatch.engine import Matcher
from fastmatch.types import Match, MatchParams

def test_language_roundtrip_preserves_state():
    app = QApplication.instance() or QApplication([])
    i18n.set_language("en", persist=False)
    window = MainWindow(None, device="cpu")
    try:
        window._params_panel._threshold.setValue(700)
        before = window._params_panel.current_params()
        i18n.set_language("zh", persist=False)
        assert window._params_panel._run_button.text() == "开始搜索"
        assert window._params_panel._max_results.text() == "不限"
        methods = window._params_panel._method
        assert methods.itemText(methods.findData("semantic")).startswith("语义相似度")
        assert methods.itemText(methods.findData("fuzzy")).startswith("模糊实例搜索")
        assert i18n.tr("Matches shown: 123") == "显示匹配数：123"
        assert window._params_panel.current_params() == before
        i18n.set_language("en", persist=False)
        assert window._params_panel._run_button.text() == "Run"
        assert window._params_panel._max_results.text() == "Unlimited"
        assert window._params_panel.current_params() == before
    finally:
        window.close()
        i18n.set_language("en", persist=False)

def test_cpu_thread_setting_reaches_both_libraries():
    import cv2
    previous = torch.get_num_threads()
    previous_cv = cv2.getNumThreads()
    try:
        chosen = cpu.configure_threads(min(4, cpu.available_threads()))
        assert torch.get_num_threads() == chosen
        assert cv2.getNumThreads() == chosen
    finally:
        torch.set_num_threads(previous)
        cv2.setNumThreads(previous_cv)

def test_unlimited_results_after_nms():
    count = 520
    matcher = Matcher(device="cpu")
    cands = {"x": torch.arange(count, dtype=torch.float32) * 10,
             "y": torch.zeros(count), "w": torch.ones(count) * 3,
             "h": torch.ones(count) * 3, "scale": torch.ones(count),
             "score": torch.linspace(0.9, 1, count)}
    assert len(matcher._finalize(cands, MatchParams(max_results=0), None, 3, 3)) == count
    assert len(matcher._finalize(cands, MatchParams(max_results=5), None, 3, 3)) == 5

def test_unlimited_orientation_results():
    matches = [Match(x=i*10, y=0, w=3, h=3, score=0.99, scale=1.0) for i in range(520)]
    assert len(Matcher._finalize_orientations(matches, MatchParams())) == 520
    assert len(Matcher._finalize_orientations(matches, MatchParams(max_results=3))) == 3
