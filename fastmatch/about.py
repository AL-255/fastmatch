"""The *About FastMatch* dialog: author, build ID (git), and the full license.

Kept separate from :mod:`fastmatch.app` so the build-ID / license helpers are
unit-testable without standing up the whole main window. The build ID is derived
from git at runtime (short hash + ``-dirty`` when the tree has uncommitted
changes); the license is read from the project's ``LICENSE`` file and shown in a
scrollable, read-only view.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import __version__

#: Author shown in the dialog (the project maintainer).
AUTHOR = "Anhang Li (AL-255)"
EMAIL = "thelithcore@gmail.com"

#: Project root (parent of the ``fastmatch`` package), where ``LICENSE`` lives.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def build_id() -> str:
    """A build identifier from git: short commit hash, ``-dirty`` if modified.

    Falls back to reading ``.git/HEAD`` directly (no git binary needed), then to
    ``"unknown"`` when the source isn't a git checkout (e.g. an installed wheel).
    """
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=12"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    try:  # no git binary / not on PATH: parse the refs ourselves
        head = (_REPO_ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            return (_REPO_ROOT / ".git" / ref).read_text(encoding="utf-8").strip()[:12]
        return head[:12] or "unknown"
    except Exception:
        return "unknown"


def license_text() -> str:
    """Full text of the bundled ``LICENSE`` file, or a short fallback notice."""
    for cand in (_REPO_ROOT / "LICENSE", Path(__file__).resolve().parent / "LICENSE"):
        try:
            return cand.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return "License text not found. FastMatch is distributed under the GNU GPL v2."


class AboutDialog(QDialog):
    """Modal *About* dialog: app/version, author + e-mail, build ID, and the
    full license in a scrollable read-only pane."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About FastMatch")
        self.resize(600, 640)

        layout = QVBoxLayout(self)

        heading = QLabel("<h2>FastMatch</h2>", self)
        heading.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(heading)

        # Version + build ID + author + clickable e-mail.
        self._info = QLabel(self)
        self._info.setTextFormat(Qt.TextFormat.RichText)
        self._info.setOpenExternalLinks(True)
        self._info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self._info.setText(
            f"Version {__version__}<br>"
            f"Build {build_id()}<br><br>"
            f"{AUTHOR}<br>"
            f'<a href="mailto:{EMAIL}">{EMAIL}</a>'
        )
        layout.addWidget(self._info)

        layout.addWidget(QLabel("License", self))
        self._license = QPlainTextEdit(self)
        self._license.setReadOnly(True)
        self._license.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._license.setPlainText(license_text())
        mono = self._license.font()
        mono.setStyleHint(mono.StyleHint.Monospace)
        mono.setFamily("monospace")
        self._license.setFont(mono)
        layout.addWidget(self._license, 1)  # take the slack; this is the scroll area

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
