"""Side panel listing the numbered example boxes of a multi-example search.

One row per example, numbered exactly like the labels drawn on the image:
positives ``1, 2, 3...`` (row 1 is the selection itself) and negatives
``×1, ×2...``. Rows can be multi-selected and deleted in one go (Delete key or
the button); selecting rows highlights their boxes and centres the view on the
last one clicked, so each example can be checked before searching.

The viewport owns the example boxes; this panel only renders a snapshot of
them (:meth:`set_examples`) and reports what the user asked for via signals.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

Item = tuple[str, int]  # ("pos" | "neg", number as labelled on the image)

_COLUMNS = ("#", "Type", "Position", "Size")
_EMPTY = "No examples. Shift+drag adds one, Ctrl+drag a negative."
_POS_COLOR = QColor(0, 150, 200)
_NEG_COLOR = QColor(230, 80, 0)


class ExamplesPanel(QWidget):
    """Numbered list of the example boxes with multi-select delete.

    Signals:
        delete_requested: ``list[(kind, number)]`` of the rows to delete.
        clear_requested: "Clear examples" was clicked.
        highlight_changed: ``list[(kind, number)]`` of the selected rows.
        focus_requested: ``(kind, number)`` of the row just clicked; the main
            window centres the view on that box.
    """

    delete_requested = Signal(object)
    clear_requested = Signal()
    highlight_changed = Signal(object)
    focus_requested = Signal(str, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[Item] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._summary = QLabel(_EMPTY, self)
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellClicked.connect(self._on_cell_clicked)
        root.addWidget(self._table, 1)

        # Delete / Backspace remove the selected rows while the table has focus.
        for key in (QKeySequence.StandardKey.Delete, QKeySequence(Qt.Key.Key_Backspace)):
            sc = QShortcut(QKeySequence(key), self._table)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(self._delete_selected)

        buttons = QHBoxLayout()
        self._delete_btn = QPushButton("Delete selected", self)
        self._delete_btn.setToolTip("Delete the selected examples (Delete key). Ctrl/Shift+click selects several.")
        self._delete_btn.clicked.connect(self._delete_selected)
        self._delete_btn.setEnabled(False)
        self._clear_btn = QPushButton("Clear examples", self)
        self._clear_btn.clicked.connect(self.clear_requested.emit)
        self._clear_btn.setEnabled(False)
        buttons.addWidget(self._delete_btn)
        buttons.addWidget(self._clear_btn)
        root.addLayout(buttons)

    # ------------------------------------------------------------------ data
    def set_examples(
        self,
        selection: QRect | None,
        positives: "list[tuple[int, int, int, int]]",
        negatives: "list[tuple[int, int, int, int]]",
    ) -> None:
        """Show the selection (positive #1), extra positives and negatives."""
        rows: list[tuple[Item, tuple[int, int, int, int]]] = []
        if selection is not None:
            rows.append((("pos", 1), (selection.x(), selection.y(), selection.width(), selection.height())))
            rows += [(("pos", i + 2), box) for i, box in enumerate(positives)]
        rows += [(("neg", i + 1), box) for i, box in enumerate(negatives)]
        self._items = [item for item, _ in rows]

        self._table.blockSignals(True)
        self._table.clearSelection()
        self._table.setRowCount(len(rows))
        for r, ((kind, number), (x, y, w, h)) in enumerate(rows):
            color = _POS_COLOR if kind == "pos" else _NEG_COLOR
            cells = (
                str(number) if kind == "pos" else f"×{number}",
                "Positive" if kind == "pos" else "Negative",
                f"({x}, {y})",
                f"{w} × {h}",
            )
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c < 2:
                    item.setForeground(color)
                self._table.setItem(r, c, item)
        self._table.blockSignals(False)

        n_pos = sum(1 for k, _ in self._items if k == "pos")
        n_neg = len(self._items) - n_pos
        self._summary.setText(
            f"Examples: {n_pos} positive, {n_neg} negative" if self._items
            else _EMPTY)
        self._clear_btn.setEnabled(bool(positives or negatives))
        self._delete_btn.setEnabled(False)
        self.highlight_changed.emit([])

    def items(self) -> list[Item]:
        """The ``(kind, number)`` shown on each row, top to bottom."""
        return list(self._items)

    def selected_items(self) -> list[Item]:
        rows = sorted({i.row() for i in self._table.selectedIndexes()})
        return [self._items[r] for r in rows]

    # --------------------------------------------------------------- actions
    def select_rows(self, rows: "list[int]") -> None:
        """Select the given table rows (programmatic; used by tests)."""
        self._table.clearSelection()
        mode = self._table.selectionMode()
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for r in rows:
            self._table.selectRow(r)
        self._table.setSelectionMode(mode)

    def _delete_selected(self) -> None:
        items = self.selected_items()
        if items:
            self.delete_requested.emit(items)

    def _on_selection_changed(self) -> None:
        items = self.selected_items()
        self._delete_btn.setEnabled(bool(items))
        self.highlight_changed.emit(items)

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._items):
            self.focus_requested.emit(*self._items[row])
