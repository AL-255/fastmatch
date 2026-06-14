"""About-dialog tests (offscreen, Qt-only).

Pins the three things the About page must surface: the author/e-mail, a git
build ID, and the full (scrollable, read-only) license text.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="about dialog needs PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from fastmatch import __version__  # noqa: E402
from fastmatch.about import AUTHOR, EMAIL, AboutDialog, build_id, license_text  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_build_id_is_nonempty() -> None:
    """build_id() always returns a usable string (git hash, or a fallback)."""
    bid = build_id()
    assert isinstance(bid, str) and bid.strip()


def test_license_text_is_the_full_license() -> None:
    """The full license text is returned (not a stub) — here, GPL v2."""
    text = license_text()
    assert "GNU GENERAL PUBLIC LICENSE" in text
    assert len(text) > 1000  # the whole license, not a one-line notice


def test_about_dialog_shows_author_build_and_license(qapp) -> None:
    """The dialog surfaces author, e-mail, version, build ID, and the license."""
    dlg = AboutDialog()
    info = dlg._info.text()  # noqa: SLF001 - white-box check of the shown fields
    assert AUTHOR in info
    assert EMAIL in info
    assert __version__ in info
    assert build_id() in info
    # The license pane is read-only (scrollable) and holds the full text.
    assert dlg._license.isReadOnly()  # noqa: SLF001
    assert "GNU GENERAL PUBLIC LICENSE" in dlg._license.toPlainText()  # noqa: SLF001
    dlg.deleteLater()
