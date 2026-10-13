"""
Palette and Reference Color Inspector Panel.
Allows user to:
- Load colored character reference illustrations.
- Extract dominant character palettes (hair, skin, clothing, eyes).
- Lock color hex codes to enforce consistent coloring across 180 pages.
- Persist character definitions to color_memory.json.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.palette_manager import PaletteManager


class ColorSwatchButton(QPushButton):
    color_changed = Signal(str)

    def __init__(self, hex_color: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.hex_color = hex_color
        self.setFixedSize(28, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()
        self.clicked.connect(self._choose_color)

    def set_color(self, hex_color: str):
        self.hex_color = hex_color
        self._update_style()

    def _update_style(self):
        self.setStyleSheet(
            f"QPushButton {{ background-color: {self.hex_color}; border: 2px solid #475569; border-radius: 4px; }} "
            f"QPushButton:hover {{ border-color: #ffffff; }}"
        )

    def _choose_color(self):
        col = QColorDialog.getColor(QColor(self.hex_color), self, "Select Reference Color")
        if col.isValid():
            self.hex_color = col.name()
            self._update_style()
            self.color_changed.emit(self.hex_color)


class PalettePanel(QWidget):
    palette_updated = Signal()

    def __init__(self, palette_manager: PaletteManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.palette_manager = palette_manager

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(14)

        # Title
        title = QLabel("Character Consistency & Color Memory")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #f1f5f9;")
        layout.addWidget(title)

        desc = QLabel(
            "Lock specific colors for hair, skin, and clothing to prevent color drift across pages. "
            "Upload an official colored reference art or anime screenshot to extract palette automatically."
        )
        desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Reference Art Uploader Card
        ref_box = QGroupBox("Reference Character Art")
        ref_layout = QHBoxLayout(ref_box)

        self.btn_load_ref = QPushButton("Load Reference Image")
        self.btn_load_ref.setObjectName("PrimaryBtn")
        self.btn_load_ref.clicked.connect(self._load_reference_image)
        ref_layout.addWidget(self.btn_load_ref)

        self.ref_info = QLabel("No reference loaded")
        self.ref_info.setStyleSheet("color: #94a3b8; font-size: 12px;")
        ref_layout.addWidget(self.ref_info)
        ref_layout.addStretch()

        layout.addWidget(ref_box)

        # Extracted Palette Swatches
        self.swatch_box = QGroupBox("Extracted Palette (Dominant Tones)")
        self.swatch_layout = QHBoxLayout(self.swatch_box)
        self._refresh_swatches()
        layout.addWidget(self.swatch_box)

        # Character Attribute Locks
        locks_box = QGroupBox("Character Color Locks (Enforced Across Batch)")
        locks_layout = QVBoxLayout(locks_box)

        self.hair_btn = ColorSwatchButton("#332211")
        self.hair_btn.color_changed.connect(lambda c: self._on_lock_changed("hair", c))
        row_hair = self._create_lock_row("Hair Tone:", self.hair_btn, "Enforce character hair color")
        locks_layout.addLayout(row_hair)

        self.skin_btn = ColorSwatchButton("#ffd5b8")
        self.skin_btn.color_changed.connect(lambda c: self._on_lock_changed("skin", c))
        row_skin = self._create_lock_row("Skin Tone:", self.skin_btn, "Enforce natural warm skin palette")
        locks_layout.addLayout(row_skin)

        self.cloth_btn = ColorSwatchButton("#225588")
        self.cloth_btn.color_changed.connect(lambda c: self._on_lock_changed("clothing", c))
        row_cloth = self._create_lock_row("Main Outfit:", self.cloth_btn, "Enforce clothing / uniform tone")
        locks_layout.addLayout(row_cloth)

        layout.addWidget(locks_box)
        layout.addStretch()

    def _create_lock_row(self, label_text: str, swatch: ColorSwatchButton, help_text: str) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(90)
        lbl.setStyleSheet("font-weight: 600; color: #f1f5f9;")
        row.addWidget(lbl)
        row.addWidget(swatch)

        hlp = QLabel(help_text)
        hlp.setStyleSheet("color: #64748b; font-size: 11px;")
        row.addWidget(hlp)
        row.addStretch()
        return row

    def _refresh_swatches(self):
        while self.swatch_layout.count():
            item = self.swatch_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        palette = self.palette_manager.get_active_palette()
        if not palette:
            placeholder = QLabel("Load a reference image above to extract color palette.")
            placeholder.setStyleSheet("color: #64748b; font-style: italic;")
            self.swatch_layout.addWidget(placeholder)
            return

        for hex_code in palette:
            btn = ColorSwatchButton(hex_code)
            self.swatch_layout.addWidget(btn)
        self.swatch_layout.addStretch()

    def _load_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Character Illustration",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not file_path:
            return

        bgr = cv2.imread(file_path)
        if bgr is None:
            return

        colors = self.palette_manager.extract_palette_from_image(bgr, k=8)
        self.palette_manager.set_active_palette(colors)
        self.ref_info.setText(f"Loaded: {os.path.basename(file_path)} ({len(colors)} tones extracted)")
        self._refresh_swatches()
        self.palette_updated.emit()

    def _on_lock_changed(self, category: str, hex_color: str):
        self.palette_manager.set_color_lock(category, hex_color, enabled=True)
        self.palette_updated.emit()
