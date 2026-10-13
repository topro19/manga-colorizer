"""
Interactive Multi-Mode Before/After/Difference Viewer for MangaColorizer.
Features:
- Modes:
    1. Split Slider: Draggable divider between two selected image states
    2. Side-by-Side: Dual comparison panels
    3. Blink: Alternating temporal flick
    4. Difference: Chrominance heat-delta
- Image states:
    - Original B&W
    - Raw AI
    - Guided / L-Preserved
    - Final Composited
- Smooth pan & zoom (mouse wheel + click-and-drag)
- Fit to window & 1:1 original scale
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class SplitComparisonWidget(QWidget):
    """
    Custom painting canvas displaying Original, Raw AI, Guided, Final, and Difference images
    with draggable split slider, side-by-side view, blink inspection, and pan/zoom.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmaps: Dict[str, QPixmap] = {
            "Original": QPixmap(),
            "Raw AI": QPixmap(),
            "Guided": QPixmap(),
            "Final": QPixmap(),
        }
        self._diff_pixmap: Optional[QPixmap] = None

        self.left_state = "Original"
        self.right_state = "Final"
        self.mode = "split"  # 'split', 'side_by_side', 'blink', 'diff'

        self.split_pos = 0.5
        self._is_dragging_split = False

        self._blink_state = False
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)

        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self._is_panning = False
        self._pan_start = QPoint(0, 0)

        self.setMinimumSize(400, 300)

    def set_layer_pixmap(self, layer_name: str, pixmap: QPixmap):
        self._pixmaps[layer_name] = pixmap
        if layer_name in (self.left_state, self.right_state):
            self._update_difference_map()
            self.update()

    def set_all_images(self, orig_pix: QPixmap, raw_pix: QPixmap, guided_pix: QPixmap, final_pix: QPixmap):
        self._pixmaps["Original"] = orig_pix
        self._pixmaps["Raw AI"] = raw_pix if not raw_pix.isNull() else final_pix
        self._pixmaps["Guided"] = guided_pix if not guided_pix.isNull() else final_pix
        self._pixmaps["Final"] = final_pix
        self._update_difference_map()
        self.fit_to_screen()
        self.update()

    def set_compared_layers(self, left_layer: str, right_layer: str):
        self.left_state = left_layer
        self.right_state = right_layer
        self._update_difference_map()
        self.update()

    def _update_difference_map(self):
        p_left = self._pixmaps.get(self.left_state)
        p_right = self._pixmaps.get(self.right_state)
        if not p_left or not p_right or p_left.isNull() or p_right.isNull():
            self._diff_pixmap = None
            return

        img_left = p_left.toImage().convertToFormat(QImage.Format_RGB32)
        img_right = p_right.toImage().convertToFormat(QImage.Format_RGB32)
        w, h = img_left.width(), img_left.height()
        diff_img = QImage(w, h, QImage.Format_RGB32)

        for y in range(0, h, 2):
            for x in range(0, w, 2):
                c1 = img_left.pixelColor(x, y)
                c2 = img_right.pixelColor(x, y)
                dr = abs(c1.red() - c2.red())
                dg = abs(c1.green() - c2.green())
                db = abs(c1.blue() - c2.blue())
                delta = min(255, int((dr + dg + db) * 1.5))
                heat = QColor(delta, int(delta * 0.4), 255 - delta)
                diff_img.setPixelColor(x, y, heat)
                if x + 1 < w: diff_img.setPixelColor(x + 1, y, heat)
                if y + 1 < h: diff_img.setPixelColor(x, y + 1, heat)
                if x + 1 < w and y + 1 < h: diff_img.setPixelColor(x + 1, y + 1, heat)

        self._diff_pixmap = QPixmap.fromImage(diff_img)

    def set_mode(self, mode: str):
        self.mode = mode
        if self.mode == "blink":
            self._blink_timer.start(500)
        else:
            self._blink_timer.stop()
        self.fit_to_screen()
        self.update()

    def _toggle_blink(self):
        self._blink_state = not self._blink_state
        self.update()

    def get_base_pixmap(self) -> Optional[QPixmap]:
        for name in [self.left_state, self.right_state, "Original", "Final"]:
            p = self._pixmaps.get(name)
            if p and not p.isNull():
                return p
        return None

    def fit_to_screen(self):
        base = self.get_base_pixmap()
        if not base or base.isNull():
            return
        img_w, img_h = base.width(), base.height()
        avail_w = self.width() - 40
        avail_h = self.height() - 40

        if self.mode == "side_by_side":
            scale_x = (avail_w / 2.0) / float(img_w)
            scale_y = avail_h / float(img_h)
            self.zoom_factor = max(0.05, min(scale_x, scale_y, 1.0))
            total_w = img_w * 2 * self.zoom_factor + 20
            self.pan_offset = QPoint(int((self.width() - total_w) / 2), int((self.height() - img_h * self.zoom_factor) / 2))
        else:
            scale_x = avail_w / float(img_w)
            scale_y = avail_h / float(img_h)
            self.zoom_factor = max(0.05, min(scale_x, scale_y, 1.0))
            self.pan_offset = QPoint(int((self.width() - img_w * self.zoom_factor) / 2), int((self.height() - img_h * self.zoom_factor) / 2))
        self.update()

    def set_100_percent(self):
        base = self.get_base_pixmap()
        if not base or base.isNull():
            return
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(int((self.width() - base.width()) / 2), int((self.height() - base.height()) / 2))
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        base = self.get_base_pixmap()
        if not base:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        old_zoom = self.zoom_factor
        new_zoom = max(0.05, min(5.0, old_zoom * factor))

        mouse_pos = event.position().toPoint()
        offset_from_anchor = mouse_pos - self.pan_offset
        new_offset = mouse_pos - offset_from_anchor * (new_zoom / old_zoom)

        self.zoom_factor = new_zoom
        self.pan_offset = new_offset
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            base = self.get_base_pixmap()
            if self.mode == "split" and base:
                target_w = int(base.width() * self.zoom_factor)
                split_x = self.pan_offset.x() + int(target_w * self.split_pos)
                if abs(event.pos().x() - split_x) <= 15:
                    self._is_dragging_split = True
                    return

            self._is_panning = True
            self._pan_start = event.pos() - self.pan_offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        base = self.get_base_pixmap()
        if self._is_dragging_split and base:
            target_w = int(base.width() * self.zoom_factor)
            if target_w > 0:
                rel_x = event.pos().x() - self.pan_offset.x()
                self.split_pos = max(0.0, min(1.0, rel_x / float(target_w)))
                self.update()
            return

        if self._is_panning:
            self.pan_offset = event.pos() - self._pan_start
            self.update()
            return

        if self.mode == "split" and base:
            target_w = int(base.width() * self.zoom_factor)
            split_x = self.pan_offset.x() + int(target_w * self.split_pos)
            if abs(event.pos().x() - split_x) <= 15:
                self.setCursor(Qt.CursorShape.SplitHCursor)
                return

        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging_split = False
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#12141c"))

        base = self.get_base_pixmap()
        if not base or base.isNull():
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a manga page or volume to begin colorization.")
            return

        img_w, img_h = base.width(), base.height()
        target_w = int(img_w * self.zoom_factor)
        target_h = int(img_h * self.zoom_factor)
        base_rect = QRect(self.pan_offset.x(), self.pan_offset.y(), target_w, target_h)

        p_left = self._pixmaps.get(self.left_state) or base
        p_right = self._pixmaps.get(self.right_state) or base

        if self.mode == "side_by_side":
            spacing = 15
            left_rect = QRect(self.pan_offset.x(), self.pan_offset.y(), target_w, target_h)
            painter.drawPixmap(left_rect, p_left)

            right_rect = QRect(self.pan_offset.x() + target_w + spacing, self.pan_offset.y(), target_w, target_h)
            painter.drawPixmap(right_rect, p_right)

            self._draw_label_pill(painter, left_rect.left() + 10, left_rect.top() + 10, self.left_state.upper(), "#1e293b", "#cbd5e1")
            self._draw_label_pill(painter, right_rect.left() + 10, right_rect.top() + 10, self.right_state.upper(), "#312e81", "#a5b4fc")

        elif self.mode == "blink":
            active_pix = p_left if self._blink_state else p_right
            tag = self.left_state if self._blink_state else self.right_state
            painter.drawPixmap(base_rect, active_pix)
            self._draw_label_pill(painter, base_rect.left() + 10, base_rect.top() + 10, f"BLINK: {tag.upper()}", "#312e81", "#a5b4fc")

        elif self.mode == "diff":
            if self._diff_pixmap:
                painter.drawPixmap(base_rect, self._diff_pixmap)
            else:
                painter.drawPixmap(base_rect, p_right)
            self._draw_label_pill(painter, base_rect.left() + 10, base_rect.top() + 10, f"DELTA: {self.left_state} vs {self.right_state}", "#4c0519", "#fda4af")

        else:
            # Split slider
            split_x = self.pan_offset.x() + int(target_w * self.split_pos)
            painter.drawPixmap(base_rect, p_right)

            clip_left = QRect(base_rect.left(), base_rect.top(), max(0, split_x - base_rect.left()), target_h)
            painter.save()
            painter.setClipRect(clip_left)
            painter.drawPixmap(base_rect, p_left)
            painter.restore()

            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(split_x, base_rect.top(), split_x, base_rect.bottom())

            handle_y = base_rect.top() + target_h // 2
            painter.setBrush(QColor("#6366f1"))
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(QPoint(split_x, handle_y), 13, 13)

            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawLine(split_x - 4, handle_y, split_x - 1, handle_y - 3)
            painter.drawLine(split_x - 4, handle_y, split_x - 1, handle_y + 3)
            painter.drawLine(split_x + 4, handle_y, split_x + 1, handle_y - 3)
            painter.drawLine(split_x + 4, handle_y, split_x + 1, handle_y + 3)

            self._draw_label_pill(painter, base_rect.left() + 10, base_rect.top() + 10, self.left_state.upper(), "#1e293b", "#cbd5e1")
            self._draw_label_pill(painter, base_rect.right() - 85, base_rect.top() + 10, self.right_state.upper(), "#312e81", "#a5b4fc")

    def _draw_label_pill(self, painter: QPainter, x: int, y: int, text: str, bg_hex: str, fg_hex: str):
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        txt_w = metrics.horizontalAdvance(text)
        txt_h = metrics.height()
        pad_x, pad_y = 8, 4
        pill_rect = QRect(x, y, txt_w + pad_x * 2, txt_h + pad_y * 2)

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(bg_hex))
        painter.drawRoundedRect(pill_rect, 4, 4)
        painter.setPen(QColor(fg_hex))
        painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class ViewerPanel(QWidget):
    """
    Viewer panel with top toolbar controls for zoom, fit, 1:1, Split/Side-by-Side/Blink/Diff toggles,
    and A/B comparison state selectors (Original, Raw AI, Guided, Final).
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.comparison_widget = SplitComparisonWidget(self)

        self.btn_fit = QPushButton("Fit")
        self.btn_100 = QPushButton("100%")

        self.radio_split = QRadioButton("Split")
        self.radio_split.setChecked(True)
        self.radio_side = QRadioButton("Side-by-Side")
        self.radio_blink = QRadioButton("Blink")
        self.radio_diff = QRadioButton("Diff")

        self.btn_group = QButtonGroup(self)
        self.btn_group.addButton(self.radio_split)
        self.btn_group.addButton(self.radio_side)
        self.btn_group.addButton(self.radio_blink)
        self.btn_group.addButton(self.radio_diff)

        # State Selectors: Left vs Right
        self.combo_left = QComboBox()
        self.combo_left.addItems(["Original", "Raw AI", "Guided", "Final"])
        self.combo_left.setCurrentText("Original")

        self.combo_right = QComboBox()
        self.combo_right.addItems(["Original", "Raw AI", "Guided", "Final"])
        self.combo_right.setCurrentText("Final")

        self.info_label = QLabel("Ready")
        self.info_label.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 500;")

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 5, 10, 5)
        toolbar.setSpacing(8)
        toolbar.addWidget(QLabel("View:"))
        toolbar.addWidget(self.radio_split)
        toolbar.addWidget(self.radio_side)
        toolbar.addWidget(self.radio_blink)
        toolbar.addWidget(self.radio_diff)
        toolbar.addSpacing(10)
        toolbar.addWidget(QLabel("Compare:"))
        toolbar.addWidget(self.combo_left)
        toolbar.addWidget(QLabel("vs"))
        toolbar.addWidget(self.combo_right)
        toolbar.addSpacing(10)
        toolbar.addWidget(self.btn_fit)
        toolbar.addWidget(self.btn_100)
        toolbar.addStretch()
        toolbar.addWidget(self.info_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(toolbar)
        layout.addWidget(self.comparison_widget, stretch=1)

        self.btn_fit.clicked.connect(self.comparison_widget.fit_to_screen)
        self.btn_100.clicked.connect(self.comparison_widget.set_100_percent)
        self.radio_split.toggled.connect(lambda chk: self.comparison_widget.set_mode("split") if chk else None)
        self.radio_side.toggled.connect(lambda chk: self.comparison_widget.set_mode("side_by_side") if chk else None)
        self.radio_blink.toggled.connect(lambda chk: self.comparison_widget.set_mode("blink") if chk else None)
        self.radio_diff.toggled.connect(lambda chk: self.comparison_widget.set_mode("diff") if chk else None)

        self.combo_left.currentTextChanged.connect(self._on_layer_changed)
        self.combo_right.currentTextChanged.connect(self._on_layer_changed)

    def _on_layer_changed(self):
        self.comparison_widget.set_compared_layers(
            self.combo_left.currentText(),
            self.combo_right.currentText()
        )

    def set_images_all_layers(
        self,
        orig_bytes: bytes,
        raw_bytes: bytes,
        guided_bytes: bytes,
        final_bytes: bytes,
        page_info: str = ""
    ):
        p_orig = QPixmap()
        p_orig.loadFromData(orig_bytes)
        p_raw = QPixmap()
        p_raw.loadFromData(raw_bytes)
        p_guided = QPixmap()
        p_guided.loadFromData(guided_bytes)
        p_final = QPixmap()
        p_final.loadFromData(final_bytes)

        self.comparison_widget.set_all_images(p_orig, p_raw, p_guided, p_final)
        if page_info:
            self.info_label.setText(page_info)

    def set_images(self, orig_bgr: np.ndarray, final_bgr: np.ndarray, page_info: str = ""):
        import cv2
        _, o_enc = cv2.imencode(".png", orig_bgr)
        _, f_enc = cv2.imencode(".png", final_bgr)
        self.set_images_all_layers(
            orig_bytes=o_enc.tobytes(),
            raw_bytes=f_enc.tobytes(),
            guided_bytes=f_enc.tobytes(),
            final_bytes=f_enc.tobytes(),
            page_info=page_info
        )
