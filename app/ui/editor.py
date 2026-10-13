"""
Interactive region color correction editor dialog.
Allows selecting a bounded region (e.g. hair, clothing) and applying a new color tint
without re-running the entire chapter pipeline.
"""

from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QMouseEvent
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QColorDialog, QSlider
)


class RegionCorrectionDialog(QDialog):
    region_corrected = Signal(np.ndarray)  # returns updated BGR image

    def __init__(self, original_bgr: np.ndarray, current_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Region Color Correction")
        self.resize(900, 750)
        self.original_bgr = original_bgr.copy()
        self.current_bgr = current_bgr.copy()
        self.selected_point: Optional[Tuple[int, int]] = None
        self.chosen_color = QColor("#2563EB")

        self._build_ui()
        self._update_display()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.lbl_info = QLabel("Click any character region (hair, skin, cloth) to recolor:")
        self.lbl_info.setStyleSheet("font-weight: 600; font-size: 13px; color: #a5b4fc;")
        layout.addWidget(self.lbl_info)

        self.lbl_canvas = QLabel()
        self.lbl_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_canvas.mousePressEvent = self._on_canvas_click
        layout.addWidget(self.lbl_canvas, 1)

        btn_row = QHBoxLayout()
        self.btn_pick_color = QPushButton("Choose Color...")
        self.btn_pick_color.setStyleSheet(f"background-color: {self.chosen_color.name()}; color: #ffffff; padding: 6px 12px; font-weight: bold; border-radius: 4px;")
        self.btn_pick_color.clicked.connect(self._on_pick_color)
        btn_row.addWidget(self.btn_pick_color)

        self.btn_apply = QPushButton("Apply to Region")
        self.btn_apply.setStyleSheet("background-color: #4338ca; color: #ffffff; padding: 6px 16px; font-weight: bold; border-radius: 4px;")
        self.btn_apply.clicked.connect(self._on_apply_correction)
        btn_row.addWidget(self.btn_apply)

        self.btn_done = QPushButton("Save & Close")
        self.btn_done.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_done)

        layout.addLayout(btn_row)

    def _on_canvas_click(self, event: QMouseEvent):
        # Map click to image coordinates
        pixmap = self.lbl_canvas.pixmap()
        if not pixmap:
            return
        w_lbl = self.lbl_canvas.width()
        h_lbl = self.lbl_canvas.height()
        w_pix = pixmap.width()
        h_pix = pixmap.height()

        x_off = (w_lbl - w_pix) // 2
        y_off = (h_lbl - h_pix) // 2

        cx = event.pos().x() - x_off
        cy = event.pos().y() - y_off

        if 0 <= cx < w_pix and 0 <= cy < h_pix:
            scale_x = self.current_bgr.shape[1] / float(w_pix)
            scale_y = self.current_bgr.shape[0] / float(h_pix)
            orig_x = int(cx * scale_x)
            orig_y = int(cy * scale_y)
            self.selected_point = (orig_x, orig_y)
            self.lbl_info.setText(f"Target selected at ({orig_x}, {orig_y}). Choose color and click Apply.")

    def _on_pick_color(self):
        col = QColorDialog.getColor(self.chosen_color, self, "Select Color")
        if col.isValid():
            self.chosen_color = col
            self.btn_pick_color.setStyleSheet(f"background-color: {col.name()}; color: #ffffff; padding: 6px 12px; font-weight: bold; border-radius: 4px;")

    def _on_apply_correction(self):
        if not self.selected_point:
            return
        px, py = self.selected_point
        gray = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Flood fill or region mask from ink contours
        ink = (gray < 85).astype(np.uint8) * 255
        ink_closed = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        flooded = ink_closed.copy()
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flooded, mask, (px, py), 128, flags=4 | (128 << 8))

        region_mask = (flooded == 128).astype(np.float32)
        r, g, b = self.chosen_color.red(), self.chosen_color.green(), self.chosen_color.blue()

        # Recolor region preserving high-frequency luminance
        tint_bgr = np.array([b, g, r], dtype=np.float32)
        cur_float = self.current_bgr.astype(np.float32)
        factor = region_mask[:, :, np.newaxis] * 0.70
        recolored = cur_float * (1.0 - factor) + tint_bgr * factor
        self.current_bgr = np.clip(recolored, 0, 255).astype(np.uint8)
        self._update_display()
        self.region_corrected.emit(self.current_bgr)

    def _update_display(self):
        rgb = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.lbl_canvas.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_canvas.setPixmap(scaled)
