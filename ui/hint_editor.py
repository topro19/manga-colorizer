"""
Interactive Page-Level Color Hint Editor for MangaColorizer.
Provides character color consistency by placing hint dots onto manga pages,
which are passed directly into the model's 5-channel hint conditioning tensor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class HintCanvas(QWidget):
    """Interactive canvas that displays the manga page and records color hints."""

    hints_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._pixmap: Optional[QPixmap] = None
        self._orig_size: Tuple[int, int] = (1, 1)

        self.hints: List[Dict[str, Any]] = []
        self.current_color = QColor(90, 56, 37)  # Default: Brown hair
        self.current_radius = 12

    def set_image(self, img_path_or_pixmap: str | QPixmap) -> None:
        if isinstance(img_path_or_pixmap, str):
            self._pixmap = QPixmap(img_path_or_pixmap)
        else:
            self._pixmap = img_path_or_pixmap
        if self._pixmap and not self._pixmap.isNull():
            self._orig_size = (self._pixmap.width(), self._pixmap.height())
        self.update()

    def clear_hints(self) -> None:
        self.hints.clear()
        self.hints_changed.emit()
        self.update()

    def undo_hint(self) -> None:
        if self.hints:
            self.hints.pop()
            self.hints_changed.emit()
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap:
            # Map widget coordinates to image coordinates
            img_rect = self._get_image_rect()
            if img_rect.contains(event.pos()):
                rel_x = event.pos().x() - img_rect.left()
                rel_y = event.pos().y() - img_rect.top()

                orig_x = int(rel_x * (self._orig_size[0] / float(img_rect.width())))
                orig_y = int(rel_y * (self._orig_size[1] / float(img_rect.height())))

                self.hints.append({
                    "x": orig_x,
                    "y": orig_y,
                    "radius": self.current_radius,
                    "color": (self.current_color.red(), self.current_color.green(), self.current_color.blue()),
                })
                self.hints_changed.emit()
                self.update()

    def _get_image_rect(self) -> QRect:
        if not self._pixmap or self._pixmap.isNull():
            return self.rect()
        scaled_size = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - scaled_size.width()) // 2
        y = (self.height() - scaled_size.height()) // 2
        return QRect(x, y, scaled_size.width(), scaled_size.height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#12141c"))

        if not self._pixmap or self._pixmap.isNull():
            painter.setPen(QColor("#64748b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No page loaded for hint editing.")
            return

        img_rect = self._get_image_rect()
        painter.drawPixmap(img_rect, self._pixmap.scaled(img_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # Draw existing hints
        scale_x = img_rect.width() / float(self._orig_size[0])
        scale_y = img_rect.height() / float(self._orig_size[1])

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for h in self.hints:
            hx = img_rect.left() + int(h["x"] * scale_x)
            hy = img_rect.top() + int(h["y"] * scale_y)
            rad = max(3, int(h["radius"] * min(scale_x, scale_y)))

            c = h["color"]
            fill_col = QColor(c[0], c[1], c[2], 220)
            painter.setBrush(QBrush(fill_col))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(QPoint(hx, hy), rad, rad)


class ColorHintDialog(QDialog):
    """
    Dialog for editing character and page-level color consistency hints.
    """

    def __init__(self, page_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Color Consistency & Hint Editor")
        self.resize(900, 750)
        self.setStyleSheet("background-color: #0f1117; color: #e2e8f0;")

        self.page_path = page_path
        self.canvas = HintCanvas(self)
        self.canvas.set_image(page_path)

        # Palette buttons
        swatches = [
            ("Dark Hair", "#1c1917"),
            ("Brown Hair", "#5a3825"),
            ("Blonde Hair", "#eab308"),
            ("Pale Skin", "#ffedd5"),
            ("Tan Skin", "#d97706"),
            ("Red/Pink", "#e11d48"),
            ("Denim Blue", "#2563eb"),
            ("Green", "#16a34a"),
        ]

        swatch_layout = QHBoxLayout()
        swatch_layout.addWidget(QLabel("Presets:"))
        for name, hex_code in swatches:
            btn = QPushButton(name)
            btn.setStyleSheet(f"background-color: {hex_code}; color: {'#ffffff' if hex_code not in ('#ffedd5', '#eab308') else '#000000'}; font-size: 11px; padding: 4px 8px;")
            btn.clicked.connect(lambda _, col=hex_code: self._select_color(col))
            swatch_layout.addWidget(btn)

        self.btn_custom_color = QPushButton("Pick Custom Color...")
        self.btn_custom_color.clicked.connect(self._pick_custom_color)
        swatch_layout.addWidget(self.btn_custom_color)
        swatch_layout.addStretch()

        # Radius control
        rad_layout = QHBoxLayout()
        rad_layout.addWidget(QLabel("Hint Radius:"))
        self.slider_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_radius.setRange(4, 50)
        self.slider_radius.setValue(12)
        self.lbl_radius = QLabel("12 px")
        self.slider_radius.valueChanged.connect(self._on_radius_changed)
        rad_layout.addWidget(self.slider_radius)
        rad_layout.addWidget(self.lbl_radius)

        self.btn_undo = QPushButton("Undo Last")
        self.btn_undo.clicked.connect(self.canvas.undo_hint)
        self.btn_clear = QPushButton("Clear All Hints")
        self.btn_clear.clicked.connect(self.canvas.clear_hints)
        rad_layout.addSpacing(20)
        rad_layout.addWidget(self.btn_undo)
        rad_layout.addWidget(self.btn_clear)
        rad_layout.addStretch()

        # Consistency explanation banner
        info_banner = QLabel(
            "<b>Color Consistency Guide:</b> Click anywhere on the page (hair, skin, clothes) to place a color hint.<br>"
            "MangaColorization-v2 uses these hints as conditional targets for the generator.<br>"
            "Check <i>'Apply hints to all pages in this volume'</i> to keep hair and skin consistent across all pages."
        )
        info_banner.setStyleSheet("background-color: #1e1b4b; border: 1px solid #4338ca; border-radius: 6px; padding: 8px; color: #c7d2fe; font-size: 12px;")

        # Options
        self.chk_apply_all = QCheckBox("Apply these hints as default reference for ALL pages in this volume")
        self.chk_apply_all.setChecked(True)

        # Action buttons
        btn_box = QHBoxLayout()
        self.btn_save = QPushButton("Apply Hints & Close")
        self.btn_save.setObjectName("PrimaryBtn")
        self.btn_save.setStyleSheet("background-color: #4f46e5; color: white; padding: 8px 16px; font-weight: bold; border-radius: 6px;")
        self.btn_save.clicked.connect(self.accept)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_save)

        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(info_banner)
        layout.addLayout(swatch_layout)
        layout.addLayout(rad_layout)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.chk_apply_all)
        layout.addLayout(btn_box)

    def _select_color(self, hex_code: str) -> None:
        self.canvas.current_color = QColor(hex_code)

    def _pick_custom_color(self) -> None:
        col = QColorDialog.getColor(self.canvas.current_color, self, "Select Hint Color")
        if col.isValid():
            self.canvas.current_color = col

    def _on_radius_changed(self, val: int) -> None:
        self.canvas.current_radius = val
        self.lbl_radius.setText(f"{val} px")

    def get_hints(self) -> List[Dict[str, Any]]:
        return list(self.canvas.hints)

    def is_apply_to_all(self) -> bool:
        return self.chk_apply_all.isChecked()
