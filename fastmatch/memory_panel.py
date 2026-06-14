"""The saved-search "Memory" side panel.

``MemoryPanel`` is the GUI surface for the Qt-free data model in
:mod:`fastmatch.memory`. Each completed search can be appended as a
:class:`~fastmatch.memory.MemoryEntry` (the boxed selection + the matches it
found + the params used); the panel renders one stats row per entry in a
read-only table, lets the user remove rows, and saves/loads the whole
:class:`~fastmatch.memory.MemoryStore` (source image header + all entries) to a
JSON file via :func:`~fastmatch.memory.save_store` / :func:`~fastmatch.memory.load_store`.

The panel deliberately keeps *no* state the model layer does not: a python list
``self._entries`` is held strictly aligned with the table rows (row ``i`` shows
``self._entries[i]``), and the source header lives in ``self._source_image`` /
``self._image_size`` (set via :meth:`set_source`). :meth:`store` rebuilds a
``MemoryStore`` on demand from those three fields, so what we save is always what
the panel currently shows.

Threading note: all of this runs on the GUI thread — the panel only renders
already-finished, plain-CPU :class:`~fastmatch.types.Match` data the controller
handed up, so there is no cross-thread payload here.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import memory
from .memory import MemoryEntry, MemoryStore

# Table columns, in display order. Each added entry populates exactly these from
# its derived stats (see :class:`~fastmatch.memory.MemoryEntry`).
_COLUMNS: tuple[str, ...] = (
    "Label",
    "Method",
    "Channel",
    "Selection",
    "Occurrences",
    "Score",
    "Orientations",
)

# QFileDialog name filter for the memory JSON files (matches save_store output).
_JSON_FILTER = "FastMatch memory (*.json)"

# Canonical extension for a saved store.
_JSON_EXT = ".json"


def _ensure_json_suffix(path: str) -> str:
    """Return ``path`` with a ``.json`` extension guaranteed.

    The Save dialog only offers the ``*.json`` filter, but native dialogs do not
    reliably append the extension when the user types a bare name (GTK/GNOME and
    the offscreen platform notably do not). We normalize on our side so a file
    typed as ``patterns`` is written as ``patterns.json``. An existing ``.json``
    suffix is kept as-is (case-insensitively); any other extension is preserved
    and ``.json`` is appended, since the dialog only ever meant a JSON file.
    """
    if path and os.path.splitext(path)[1].lower() == _JSON_EXT:
        return path
    return path + _JSON_EXT


class MemoryPanel(QWidget):
    """A list of saved searches with stats, plus add/remove and save/load.

    Signals:
        add_requested: Emitted when "Add to Memory" is clicked. The main window
            builds the current :class:`MemoryEntry` (selection + matches + params)
            and feeds it back via :meth:`add_entry`; the panel does not know how
            to capture the current search itself.
        entry_activated: Emitted with the :class:`MemoryEntry` of a double-clicked
            row, so the main window can *revisit* it (re-box the selection / re-run).
        store_loaded: Emitted with the freshly loaded :class:`MemoryStore` after a
            successful "Load…", so the main window can react to the new source.
    """

    add_requested = Signal()  # "Add to Memory" clicked
    entry_activated = Signal(object)  # MemoryEntry — a row was double-clicked (revisit)
    store_loaded = Signal(object)  # MemoryStore — emitted after a successful Load

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the empty panel (no source, no entries yet)."""
        super().__init__(parent)

        # The model state the panel owns, kept in lock-step with the table:
        # self._entries[i] is the entry rendered on row i.
        self._entries: list[MemoryEntry] = []
        # Source header for store(); populated via set_source / load_store.
        self._source_image: str = ""
        self._image_size: tuple[int, int] = (0, 0)
        # Path of the memory file currently associated with this list (set on a
        # successful Open/Save As); drives "Save Memory" vs "Save Memory As…".
        self._current_path: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)  # match the Search dock's tuned inset
        root.setSpacing(8)                    # ...and rhythm, so both docks read as one

        # --- Header: source image basename + entry count -------------------
        self._header_label = QLabel(self)
        self._header_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._header_label)
        # Surface the otherwise-invisible recall gesture (double-click). Muted via
        # the shared "matchCount" secondary-text style hook (themed in theme.py).
        self._hint_label = QLabel("Double-click an entry to restore its boxes and selection.", self)
        self._hint_label.setObjectName("matchCount")
        self._hint_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._hint_label)

        # --- Entry table (read-only, full-row, multi-row selection) --------
        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        # Read-only: rows are rendered from the model, never typed into.
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # Select whole rows; allow extending the selection to remove many at once.
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.verticalHeader().setVisible(False)
        # Let the "Label" column take the slack; keep the stat columns snug.
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(_COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        # Double-click a row -> revisit that entry.
        self._table.setToolTip("Double-click an entry to restore its boxes and selection.")
        self._table.itemDoubleClicked.connect(self._on_item_double_clicked)
        root.addWidget(self._table, 1)

        # --- Button row ----------------------------------------------------
        buttons = QHBoxLayout()
        self._add_button = QPushButton("Add to Memory", self)
        self._add_button.setToolTip("Save the current selection and its matches as a memory entry.")
        self._add_button.clicked.connect(self.add_requested)
        buttons.addWidget(self._add_button)

        self._rename_button = QPushButton("Rename…", self)
        self._rename_button.setToolTip("Give the selected memory entry a custom name.")
        self._rename_button.clicked.connect(self._on_rename_clicked)
        buttons.addWidget(self._rename_button)

        self._remove_button = QPushButton("Remove", self)
        self._remove_button.setToolTip("Delete the selected memory entries.")
        self._remove_button.clicked.connect(self._on_remove_clicked)
        buttons.addWidget(self._remove_button)

        # File operations (Open/Save/Save As/Close Memory) live in the window's
        # File menu; see ImageViewport's MainWindow. The public methods below
        # back those menu items.

        root.addLayout(buttons)

        self._sync_header()
        self._sync_button_state()

    # ------------------------------------------------------------------ helpers
    def _sync_button_state(self) -> None:
        """Enable Rename/Remove only when there is at least one entry to act on.

        "Add to Memory" stays enabled (it is the meaningful action on an empty
        list); the two row-acting buttons are dead with no entries — Remove
        no-ops and Rename only pops a "select an entry" prompt — so we grey them
        out to keep the empty-state affordance honest.
        """
        has_entries = len(self._entries) > 0
        self._rename_button.setEnabled(has_entries)
        self._remove_button.setEnabled(has_entries)

    def _sync_header(self) -> None:
        """Refresh the header label from the current source + entry count."""
        n = len(self._entries)
        plural = "entry" if n == 1 else "entries"
        if self._source_image:
            name = os.path.basename(self._source_image)
            self._header_label.setText(f"Memory — {name} ({n} {plural})")
        else:
            self._header_label.setText(f"Memory ({n} {plural})")

    def _row_cells(self, entry: MemoryEntry) -> list[str]:
        """Build the per-column display strings for one entry's stats row."""
        x, y, w, h = entry.selection
        lo, hi = entry.score_range()
        return [
            entry.summary(),
            entry.method,
            entry.channel_mode,
            f"({x},{y}) {w}×{h}",
            str(entry.occurrences()),  # matches + the reference selection
            f"{lo:.2f}–{hi:.2f}",
            entry.orientation_summary(),
        ]

    @staticmethod
    def _params_tooltip(entry: MemoryEntry) -> str:
        """A multi-line description of *all* settings in the entry's params.

        Surfaces every :class:`~fastmatch.types.MatchParams` field on hover so the
        full settings behind a saved search are visible without re-opening it.
        """
        p = entry.params
        lines = [
            f"method: {p.method}",
            f"channel_mode: {p.channel_mode}",
            f"threshold: {p.threshold:.2f}",
            f"threshold_floor: {p.threshold_floor:.2f}",
            f"scales: {', '.join(f'{s:g}' for s in p.scales)}",
            f"enable_rotation: {p.enable_rotation}",
            f"enable_flipping: {p.enable_flipping}",
            f"nms_iou: {p.nms_iou:.2f}",
            f"exclude_iou: {p.exclude_iou:.2f}",
            f"max_results: {p.max_results}",
            f"compute_dtype: {p.compute_dtype}",
            f"feature_detector: {p.feature_detector}",
            f"feature_ratio: {p.feature_ratio:.2f}",
            f"feature_min_inliers: {p.feature_min_inliers}",
            f"feature_max_instances: {p.feature_max_instances}",
        ]
        return "\n".join(lines)

    def _append_row(self, entry: MemoryEntry) -> int:
        """Append a read-only stats row for ``entry`` and return its row index."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        tooltip = self._params_tooltip(entry)
        for col, text in enumerate(self._row_cells(entry)):
            item = QTableWidgetItem(text)
            # Defensive: keep cells non-editable even if the table policy changes.
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            # Every cell on the row reveals the full settings on hover.
            item.setToolTip(tooltip)
            self._table.setItem(row, col, item)
        return row

    def _select_row(self, row: int) -> None:
        """Select ``row`` (full row) and scroll it into view."""
        self._table.clearSelection()
        self._table.selectRow(row)
        item = self._table.item(row, 0)
        if item is not None:
            self._table.scrollToItem(item)

    def _rebuild_rows(self) -> None:
        """Clear and repopulate the table from ``self._entries``."""
        self._table.clearContents()
        self._table.setRowCount(0)
        for entry in self._entries:
            self._append_row(entry)
        self._sync_header()
        self._sync_button_state()

    # ------------------------------------------------------------------- slots
    def _on_item_double_clicked(self, item: QTableWidgetItem) -> None:
        """A row was double-clicked: emit its entry for the main window to revisit."""
        row = item.row()
        if 0 <= row < len(self._entries):
            self.entry_activated.emit(self._entries[row])

    def _refresh_row(self, row: int) -> None:
        """Re-populate one row's cells + tooltip from its (possibly edited) entry."""
        if not (0 <= row < len(self._entries)):
            return
        entry = self._entries[row]
        tooltip = self._params_tooltip(entry)
        for col, text in enumerate(self._row_cells(entry)):
            item = self._table.item(row, col)
            if item is None:
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, col, item)
            else:
                item.setText(text)
            item.setToolTip(tooltip)

    def _on_rename_clicked(self) -> None:
        """Prompt for a custom name for the selected entry (empty -> auto label).

        Sets ``entry.label`` (which the Label column / ``summary()`` use and which
        is persisted to JSON); a blank name clears it back to the auto summary.
        """
        rows = sorted({idx.row() for idx in self._table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "Rename", "Select a memory entry to rename.")
            return
        row = rows[0]
        if not (0 <= row < len(self._entries)):
            return
        entry = self._entries[row]
        current = entry.label or entry.summary()
        name, ok = QInputDialog.getText(self, "Rename entry", "Name:", text=current)
        if not ok:
            return
        entry.label = name.strip()
        self._refresh_row(row)
        self._select_row(row)

    def _on_remove_clicked(self) -> None:
        """Delete the selected rows and their entries, keeping the lists aligned.

        Removing rows back-to-front keeps the still-pending row indices valid as
        earlier rows are deleted, and lets us drop the matching ``self._entries``
        element by the same index so the row<->entry alignment is preserved.
        """
        rows = sorted(
            {idx.row() for idx in self._table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row in rows:
            if 0 <= row < len(self._entries):
                self._table.removeRow(row)
                del self._entries[row]
        self._sync_header()
        self._sync_button_state()

    # --------------------------------------------------------- file operations
    def current_memory_path(self) -> str | None:
        """The JSON file this list is associated with (Open/Save As), or ``None``."""
        return self._current_path

    def open_memory(self) -> bool:
        """Prompt for a path, load a store, and replace the current list.

        A malformed or too-new file raises ``ValueError`` from
        :func:`memory.load_store`; we report it as a warning and leave the
        current entries untouched. On success the panel rebuilds from the loaded
        store, remembers the path, and re-emits it via ``store_loaded``.
        Returns whether a store was loaded.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Open Memory", "", _JSON_FILTER)
        if not path:
            return False  # user cancelled
        try:
            store = memory.load_store(path)
        except ValueError as exc:
            QMessageBox.warning(self, "Open failed", f"Could not open memory:\n{exc}")
            return False
        self.load_store(store)
        self._current_path = path
        self.store_loaded.emit(store)
        return True

    def save_memory(self) -> bool:
        """Save to the current file if known; otherwise behave like *Save As*."""
        if self._current_path:
            return self._write(self._current_path)
        return self.save_memory_as()

    def save_memory_as(self) -> bool:
        """Prompt for a path and write the current store as JSON; remember it."""
        start = self._current_path or ""
        path, _ = QFileDialog.getSaveFileName(self, "Save Memory As", start, _JSON_FILTER)
        if not path:
            return False  # user cancelled
        path = _ensure_json_suffix(path)  # write "name" as "name.json"
        if self._write(path):
            self._current_path = path
            return True
        return False

    def _write(self, path: str) -> bool:
        """Write the store to ``path``; surface any failure as a warning dialog."""
        try:
            memory.save_store(self.store(), path)
        except Exception as exc:  # I/O, permissions, etc.
            QMessageBox.warning(self, "Save failed", f"Could not save memory:\n{exc}")
            return False
        return True

    def close_memory(self) -> None:
        """Clear all entries and forget the associated file (Close Memory)."""
        self.clear()
        self._current_path = None

    # ------------------------------------------------------------------ public
    def add_entry(self, entry: MemoryEntry) -> None:
        """Append ``entry`` as a new stats row, then select and scroll to it."""
        self._entries.append(entry)
        row = self._append_row(entry)
        self._sync_header()
        self._sync_button_state()
        self._select_row(row)

    def set_source(self, source_image: str, image_size: tuple[int, int]) -> None:
        """Record the source image + size used as the store header on save."""
        self._source_image = source_image
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._sync_header()

    def store(self) -> MemoryStore:
        """Build a :class:`MemoryStore` from the current header + entries."""
        return MemoryStore(
            source_image=self._source_image,
            image_size=self._image_size,
            entries=list(self._entries),
        )

    def load_store(self, store: MemoryStore) -> None:
        """Replace the header + entries with ``store`` and rebuild the rows."""
        self._source_image = store.source_image
        self._image_size = (int(store.image_size[0]), int(store.image_size[1]))
        self._entries = list(store.entries)
        self._rebuild_rows()

    def clear(self) -> None:
        """Drop all entries (rows) but keep the recorded source header."""
        self._entries = []
        self._table.clearContents()
        self._table.setRowCount(0)
        self._sync_header()
        self._sync_button_state()
