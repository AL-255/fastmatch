"""Optional-backend gating and method-specific controls in the Qt panel."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication
from fastmatch.params_panel import ParamsPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_new_modes_controls_and_defaults(qapp, monkeypatch):
    monkeypatch.setattr("fastmatch.semantic_matcher.semantic_available", lambda: (True, "ready"))
    panel = ParamsPanel("cpu")
    panel._select_method("fuzzy")
    assert panel._method_stack.currentIndex() == 2
    panel._fuzzy_min_inliers.setValue(8)
    assert panel.current_params().feature_min_inliers == 8
    panel._select_method("semantic")
    assert panel._method_stack.currentIndex() == 3
    assert panel.current_params().scales == (.75, 1, 1.5)
    assert panel.current_params().threshold == .5
    assert not panel._channel_mode.isEnabled()
    assert panel._semantic_multiscale.isEnabled()
    panel._semantic_multiscale.setChecked(False)
    assert panel.current_params().scales == (1,)
    panel._select_method("ncc")
    assert panel._channel_mode.isEnabled()
    assert panel.current_params().threshold == .85
    assert not panel._multiscale.isEnabled()
    panel.close()


def test_unavailable_backends_are_disabled_with_setup_hints(qapp, monkeypatch):
    monkeypatch.setattr("fastmatch.params_panel._cv2_available", lambda: False)
    monkeypatch.setattr("fastmatch.semantic_matcher.semantic_available", lambda: (False, "Download DINOv2"))
    panel = ParamsPanel("cpu")
    for method in ("fuzzy", "features", "semantic"):
        index = panel._method.findData(method)
        assert not panel._method.model().item(index).isEnabled()
    assert panel._method.model().item(panel._method.findData("ncc")).isEnabled()
    panel.close()
