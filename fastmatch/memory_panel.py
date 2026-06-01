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

        root = QVBoxLayout(self)

        # --- Header: source image basename + entry count -------------------
        self._header_label = QLabel(self)
        self._header_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._header_label)

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

        self._save_button = QPushButton("Save…", self)
        self._save_button.setToolTip("Save all memory entries to a JSON file.")
        self._save_button.clicked.connect(self._on_save_clicked)
        buttons.addWidget(self._save_button)

        self._load_button = QPushButton("Load…", self)
        self._load_button.setToolTip("Load memory entries from a JSON file (replaces the current list).")
        self._load_button.clicked.connect(self._on_load_clicked)
        buttons.addWidget(self._load_button)

        root.addLayout(buttons)

        self._sync_header()

    # ------------------------------------------------------------------ helpers
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

    def _on_save_clicked(self) -> None:
        """Prompt for a path and write the current store as JSON.

        Any write failure is surfaced as a warning dialog rather than crashing
        the GUI thread.
        """
        path, _ = QFileDialog.getSaveFileName(self, "Save memory", "", _JSON_FILTER)
        if not path:
            return  # user cancelled
        try:
            memory.save_store(self.store(), path)
        except Exception as exc:  # I/O, permissions, etc.
            QMessageBox.warning(self, "Save failed", f"Could not save memory:\n{exc}")

    def _on_load_clicked(self) -> None:
        """Prompt for a path, load a store, and replace the current list.

        A malformed or too-new file raises ``ValueError`` from
        :func:`memory.load_store`; we report it as a warning and leave the
        current entries untouched. On success the panel rebuilds from the loaded
        store and re-emits it via ``store_loaded``.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Load memory", "", _JSON_FILTER)
        if not path:
            return  # user cancelled
        try:
            store = memory.load_store(path)
        except ValueError as exc:
            QMessageBox.warning(self, "Load failed", f"Could not load memory:\n{exc}")
            return
        self.load_store(store)
        self.store_loaded.emit(store)

    # ------------------------------------------------------------------ public
    def add_entry(self, entry: MemoryEntry) -> None:
        """Append ``entry`` as a new stats row, then select and scroll to it."""
        self._entries.append(entry)
        row = self._append_row(entry)
        self._sync_header()
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
