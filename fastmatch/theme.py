"""Application theming — Light / Dark / System palettes for the whole GUI.

Qt's modern ``QStyleHints.setColorScheme`` is *advisory*: several platform
plugins (notably ``offscreen``, used by the test suite) ignore it and leave the
palette untouched, so it cannot be the source of truth. Instead we drive the
look deterministically with **hand-built QPalettes** applied through the
palette-faithful **Fusion** style — identical on every platform and in headless
tests. ``setColorScheme`` is *also* nudged so platforms that honour it keep
their native dialogs/tooltips consistent, but the palette is the guarantee.

Three themes:

  * ``"light"`` — Fusion's standard light palette, on a light canvas.
  * ``"dark"``  — a hand-tuned dark palette, on the existing dark canvas.
  * ``"system"`` — restore whatever style + palette the app started with (the
    desktop's own theme), captured once before we ever override it.

:func:`apply_theme` is the single entry point the GUI calls; it also persists the
choice via :class:`QSettings` so it survives a restart. :func:`viewport_background`
hands the matching canvas colour to the image view (the canvas tracks the theme;
match/selection overlay colours deliberately do **not**, so a match always reads
the same green regardless of theme).
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

# Ordered theme keys (also the radio order in the menu) and their display labels.
THEME_KEYS: tuple[str, ...] = ("system", "light", "dark")
THEME_LABELS: dict[str, str] = {
    "system": "&System",
    "light": "&Light",
    "dark": "&Dark",
}
DEFAULT_THEME = "system"

# QSettings identity. Set lazily/idempotently so importing this module never
# requires a configured QApplication (tests, headless).
_ORG = "FastMatch"
_APP = "FastMatch"
_THEME_KEY = "ui/theme"

# Canvas (QGraphicsView background) per theme. The canvas tracks the UI chrome so
# the surrounding frame matches the rest of the window; a neutral grey is used
# rather than pure white/black so images of either polarity stay legible.
_BG_DARK = QColor(30, 30, 30)
_BG_LIGHT = QColor(225, 225, 225)

# The app's startup style + palette, captured once before the first override so
# "System" can faithfully restore the desktop's own theme.
_default_style: str | None = None
_default_palette: QPalette | None = None
_defaults_captured = False


def _capture_defaults(app: QApplication) -> None:
    """Record the app's original style/palette exactly once (for "System")."""
    global _default_style, _default_palette, _defaults_captured
    if _defaults_captured:
        return
    _default_style = app.style().objectName() or "Fusion"
    _default_palette = QPalette(app.palette())  # copy: app.palette() is shared
    _defaults_captured = True


def _dark_palette() -> QPalette:
    """A hand-tuned dark palette built atop Fusion's standard palette.

    Starting from Fusion's standard palette guarantees every colour role is
    populated; we then override the visible roles for a cohesive dark look,
    including the *Disabled* group so greyed-out controls stay legible.
    """
    pal = QStyleFactory.create("Fusion").standardPalette()
    C = QColor
    Role = QPalette.ColorRole
    Group = QPalette.ColorGroup

    window = C(53, 53, 53)
    base = C(35, 35, 35)
    text = C(220, 220, 220)
    highlight = C(58, 150, 235)

    pal.setColor(Role.Window, window)
    pal.setColor(Role.WindowText, text)
    pal.setColor(Role.Base, base)
    pal.setColor(Role.AlternateBase, window)
    pal.setColor(Role.ToolTipBase, window)
    pal.setColor(Role.ToolTipText, text)
    pal.setColor(Role.Text, text)
    pal.setColor(Role.Button, window)
    pal.setColor(Role.ButtonText, text)
    pal.setColor(Role.BrightText, C(255, 80, 80))
    pal.setColor(Role.Link, C(80, 170, 235))  # lighter than the selection fill
    pal.setColor(Role.Highlight, highlight)
    pal.setColor(Role.HighlightedText, C(15, 15, 15))
    pal.setColor(Role.PlaceholderText, C(140, 140, 140))

    # 3D bevel roles. Fusion's standardPalette leaves these at LIGHT greys, which
    # makes QGroupBox frames and the slider groove nearly invisible on a dark
    # window. Place the lighter bevels just ABOVE Window (53) and the darker ones
    # BELOW it so recessed edges read as real steps.
    pal.setColor(Role.Light, C(75, 75, 75))
    pal.setColor(Role.Midlight, C(64, 64, 64))
    pal.setColor(Role.Mid, C(44, 44, 44))
    pal.setColor(Role.Dark, C(25, 25, 25))
    pal.setColor(Role.Shadow, C(15, 15, 15))

    disabled = C(150, 150, 150)
    for role in (Role.WindowText, Role.Text, Role.ButtonText):
        pal.setColor(Group.Disabled, role, disabled)
    pal.setColor(Group.Disabled, Role.Highlight, C(70, 70, 70))
    pal.setColor(Group.Disabled, Role.HighlightedText, C(170, 170, 170))
    return pal


def _light_palette() -> QPalette:
    """Fusion's standard (light) palette, with disabled text and accent tuned.

    Fusion's stock disabled foreground (#bebebe ~1.6:1 on Window) is barely
    legible, so we darken it; the Highlight/Link are harmonized to the app accent
    so white selected-row text clears AA contrast and links lose the harsh pure blue.
    """
    pal = QStyleFactory.create("Fusion").standardPalette()
    Role = QPalette.ColorRole
    for role in (Role.WindowText, Role.Text, Role.ButtonText):
        # ~3.8:1 on the light window — legible-but-dimmed, matching the dark theme's
        # disabled parity (Fusion's stock #bebebe was a near-invisible ~1.6:1).
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(120, 120, 120))
    pal.setColor(Role.Highlight, QColor(23, 105, 170))
    pal.setColor(Role.Link, QColor(11, 95, 176))
    return pal


def build_palette(key: str) -> QPalette | None:
    """Palette for ``key``, or ``None`` for ``"system"`` (uses the default)."""
    if key == "dark":
        return _dark_palette()
    if key == "light":
        return _light_palette()
    return None  # system -> restore captured default


def theme_qss(key: str) -> str:
    """Narrow, objectName-scoped QSS to apply on the MAIN WINDOW for theme ``key``.

    Two purposes only: mute the secondary diagnostic text (``deviceBanner`` and
    the ``matchCount`` readout) and give the primary Run button (``runButton``) an
    accent — scoped to ``:enabled`` so the disabled state still uses the palette's
    Disabled group. Empty for ``"system"`` so nothing is overridden. Only these
    three object names are targeted, so no other widget changes.

    Applied at the *widget* level (``window.setStyleSheet(...)``), NOT on the
    QApplication: an app-level stylesheet wraps the style in a QStyleSheetStyle
    proxy (``app.style().objectName()`` becomes ""), whereas a widget stylesheet
    leaves the global Fusion style intact while still cascading to these children.
    """
    if key == "dark":
        secondary = "#9a9a9a"
        run = (
            "QPushButton#runButton:enabled{background:#3a96eb;color:#0f0f0f;"
            "font-weight:600;border:none;padding:5px;}"
            "QPushButton#runButton:enabled:hover{background:#52a6f2;}"
        )
    elif key == "light":
        secondary = "#5f6b76"
        run = (
            "QPushButton#runButton:enabled{background:#1769aa;color:#ffffff;"
            "border:1px solid #135a93;font-weight:600;padding:4px 12px;}"
            "QPushButton#runButton:enabled:hover{background:#1f7bc0;}"
        )
    else:
        return ""
    return (
        f"QLabel#deviceBanner{{color:{secondary};font-size:11px;}}"
        f"QLabel#matchCount{{color:{secondary};}}" + run
    )


def viewport_background(key: str) -> QColor:
    """Canvas colour for the image view under theme ``key``.

    ``system`` follows the desktop's detected colour scheme when the platform
    reports one, falling back to the dark canvas (the historical default).
    """
    if key == "dark":
        return QColor(_BG_DARK)
    if key == "light":
        return QColor(_BG_LIGHT)
    # system: honour the OS scheme if the platform exposes it.
    app = QApplication.instance()
    if app is not None:
        try:
            if app.styleHints().colorScheme() == Qt.ColorScheme.Light:
                return QColor(_BG_LIGHT)
        except (AttributeError, RuntimeError):
            pass
    return QColor(_BG_DARK)


def normalize_theme(key: str | None) -> str:
    """Coerce any input to a known theme key (defaults to ``DEFAULT_THEME``)."""
    k = (key or "").strip().lower()
    return k if k in THEME_KEYS else DEFAULT_THEME


def apply_theme(app: QApplication, key: str, *, persist: bool = True) -> str:
    """Apply theme ``key`` to ``app`` (palette + style), and persist the choice.

    Returns the normalized key actually applied. The first call captures the
    app's startup style/palette so a later switch to ``"system"`` restores the
    desktop theme exactly. Safe to call repeatedly.
    """
    _capture_defaults(app)
    key = normalize_theme(key)

    if key == "system":
        # Restore the desktop's own theme captured at startup.
        if _default_style:
            app.setStyle(_default_style)
        if _default_palette is not None:
            app.setPalette(_default_palette)
        scheme = Qt.ColorScheme.Unknown
    else:
        app.setStyle("Fusion")  # palette-faithful on every platform
        pal = build_palette(key)
        if pal is not None:
            app.setPalette(pal)
        scheme = Qt.ColorScheme.Dark if key == "dark" else Qt.ColorScheme.Light

    # Nudge platforms that honour colour-scheme hints (native menus, dialogs,
    # tooltips). Harmless where unsupported; the palette above is the guarantee.
    try:
        app.styleHints().setColorScheme(scheme)
    except (AttributeError, RuntimeError):
        pass

    if persist:
        save_theme(key)
    return key


# --------------------------------------------------------------- persistence
def _settings() -> QSettings:
    """A QSettings bound to the FastMatch identity (set idempotently)."""
    if not QCoreApplication.organizationName():
        QCoreApplication.setOrganizationName(_ORG)
    if not QCoreApplication.applicationName():
        QCoreApplication.setApplicationName(_APP)
    return QSettings()


def settings() -> QSettings:
    """Shared app :class:`QSettings` (FastMatch identity), for non-theme prefs too
    (e.g. the File ▸ Open Recent history) so they share one identity/store."""
    return _settings()


def save_theme(key: str) -> None:
    """Persist the chosen theme key."""
    _settings().setValue(_THEME_KEY, normalize_theme(key))


def load_theme() -> str:
    """The persisted theme key, or :data:`DEFAULT_THEME` if none/invalid."""
    return normalize_theme(_settings().value(_THEME_KEY, DEFAULT_THEME))
