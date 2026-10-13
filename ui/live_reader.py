"""
Interactive Live Manga Reader for MangaColorizer Studio.
Allows user to:
- Read black & white and colored pages in real-time as the colorizer processes them in the background.
- Toggle between Colorized and Original B&W view instantly (or split-view).
- Navigate pages with Next / Previous, slider, and keyboard arrows (Left/Right).
- Zoom & pan smoothly.
- View live colorization status badge for the currently viewed page.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class MangaReaderCanvas(QWidget):
    """
    Renders the active manga page with smooth pan, zoom, fit to window,
    and instantaneous toggling between Original B&W and Colorized.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.orig_pixmap: Optional[QPixmap] = None
        self.color_pixmap: Optional[QPixmap] = None

        self.view_mode = "color"  # 'color', 'original', 'split'
        self.split_pos = 0.5
        self._is_dragging_split = False

        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self._is_panning = False
        self._pan_start = QPoint(0, 0)

        self.setMinimumSize(400, 300)

    def set_page_pixmaps(self, orig_pix: Optional[QPixmap], color_pix: Optional[QPixmap]):
        self.orig_pixmap = orig_pix
        self.color_pixmap = color_pix
        self.fit_to_screen()
        self.update()

    def set_view_mode(self, mode: str):
        self.view_mode = mode
        self.update()

    def fit_to_screen(self):
        base = self.color_pixmap or self.orig_pixmap
        if not base or base.isNull():
            return
        img_w, img_h = base.width(), base.height()
        avail_w = self.width() - 40
        avail_h = self.height() - 40
        scale_x = avail_w / float(img_w)
        scale_y = avail_h / float(img_h)
        self.zoom_factor = max(0.05, min(scale_x, scale_y, 1.0))
        self.pan_offset = QPoint(
            int((self.width() - img_w * self.zoom_factor) / 2),
            int((self.height() - img_h * self.zoom_factor) / 2)
        )
        self.update()

    def set_100_percent(self):
        base = self.color_pixmap or self.orig_pixmap
        if not base or base.isNull():
            return
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(
            int((self.width() - base.width()) / 2),
            int((self.height() - base.height()) / 2)
        )
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        base = self.color_pixmap or self.orig_pixmap
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
            base = self.color_pixmap or self.orig_pixmap
            if self.view_mode == "split" and base:
                target_w = int(base.width() * self.zoom_factor)
                split_x = self.pan_offset.x() + int(target_w * self.split_pos)
                if abs(event.pos().x() - split_x) <= 15:
                    self._is_dragging_split = True
                    return

            self._is_panning = True
            self._pan_start = event.pos() - self.pan_offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        base = self.color_pixmap or self.orig_pixmap
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

        if self.view_mode == "split" and base:
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
        painter.fillRect(self.rect(), QColor("#0d0f17"))

        base = self.color_pixmap or self.orig_pixmap
        if not base or base.isNull():
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No manga pages loaded.\nEnter MangaDot URL or open a local chapter to start reading."
            )
            return

        img_w, img_h = base.width(), base.height()
        target_w = int(img_w * self.zoom_factor)
        target_h = int(img_h * self.zoom_factor)
        base_rect = QRect(self.pan_offset.x(), self.pan_offset.y(), target_w, target_h)

        if self.view_mode == "original" and self.orig_pixmap:
            painter.drawPixmap(base_rect, self.orig_pixmap)
            self._draw_label_pill(painter, base_rect.left() + 12, base_rect.top() + 12, "ORIGINAL B&W", "#1e293b", "#cbd5e1")

        elif self.view_mode == "color" and self.color_pixmap:
            painter.drawPixmap(base_rect, self.color_pixmap)
            self._draw_label_pill(painter, base_rect.left() + 12, base_rect.top() + 12, "COLORIZED", "#312e81", "#a5b4fc")

        elif self.view_mode == "split" and self.orig_pixmap and self.color_pixmap:
            split_x = self.pan_offset.x() + int(target_w * self.split_pos)
            painter.drawPixmap(base_rect, self.color_pixmap)

            clip_left = QRect(base_rect.left(), base_rect.top(), max(0, split_x - base_rect.left()), target_h)
            painter.save()
            painter.setClipRect(clip_left)
            painter.drawPixmap(base_rect, self.orig_pixmap)
            painter.restore()

            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(split_x, base_rect.top(), split_x, base_rect.bottom())

            handle_y = base_rect.top() + target_h // 2
            painter.setBrush(QColor("#6366f1"))
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(QPoint(split_x, handle_y), 13, 13)

            self._draw_label_pill(painter, base_rect.left() + 12, base_rect.top() + 12, "ORIGINAL", "#1e293b", "#cbd5e1")
            self._draw_label_pill(painter, base_rect.right() - 85, base_rect.top() + 12, "COLORIZED", "#312e81", "#a5b4fc")
        else:
            painter.drawPixmap(base_rect, base)

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


class LiveMangaReaderPanel(QWidget):
    """
    Dedicated Manga Reader workspace with page navigation, live colorization monitoring,
    MangaDot URL downloader, and chapter range targeting.
    """
    request_start_colorization = Signal(str, str)  # input_folder, output_folder

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.canvas = MangaReaderCanvas(self)

        self.pages_orig: List[str] = []
        self.pages_colored: Dict[str, str] = {}  # page_stem -> colorized_path
        self.current_index = 0

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Top Bar: MangaDot URL Input & Range Controls
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        top_bar.addWidget(QLabel("MangaDot URL:"))
        self.edit_url = QLineEdit("https://mangadot.net/chapter/37925")
        self.edit_url.setMinimumWidth(260)
        top_bar.addWidget(self.edit_url)

        top_bar.addWidget(QLabel("Ch From:"))
        self.edit_start_ch = QLineEdit("99")
        self.edit_start_ch.setFixedWidth(50)
        top_bar.addWidget(self.edit_start_ch)

        top_bar.addWidget(QLabel("To:"))
        self.edit_end_ch = QLineEdit("100")
        self.edit_end_ch.setFixedWidth(50)
        top_bar.addWidget(self.edit_end_ch)

        self.btn_fetch = QPushButton("📥 Fetch & Colorize")
        self.btn_fetch.setObjectName("PrimaryBtn")
        top_bar.addWidget(self.btn_fetch)

        self.btn_open_folder = QPushButton("📂 Open Local")
        self.btn_open_folder.setStyleSheet("background-color: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 5px 10px; font-weight: 600; border-radius: 4px;")
        top_bar.addWidget(self.btn_open_folder)

        self.combo_chapters = QComboBox()
        self.combo_chapters.setMinimumWidth(130)
        self.combo_chapters.setStyleSheet("background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 4px 8px; border-radius: 4px;")
        top_bar.addWidget(self.combo_chapters)

        top_bar.addSpacing(10)

        # Reader View Controls
        self.radio_color = QRadioButton("Color")
        self.radio_color.setChecked(True)
        self.radio_bw = QRadioButton("B&W")
        self.radio_split = QRadioButton("Split")

        btn_group = QButtonGroup(self)
        btn_group.addButton(self.radio_color)
        btn_group.addButton(self.radio_bw)
        btn_group.addButton(self.radio_split)

        top_bar.addWidget(QLabel("View:"))
        top_bar.addWidget(self.radio_color)
        top_bar.addWidget(self.radio_bw)
        top_bar.addWidget(self.radio_split)

        top_bar.addSpacing(10)
        self.btn_fit = QPushButton("Fit")
        self.btn_100 = QPushButton("100%")
        top_bar.addWidget(self.btn_fit)
        top_bar.addWidget(self.btn_100)

        layout.addLayout(top_bar)

        # Center Canvas
        layout.addWidget(self.canvas, stretch=1)

        # Bottom Reader Navigation Bar
        bot_bar = QHBoxLayout()
        bot_bar.setSpacing(10)

        self.btn_prev = QPushButton("◀ Previous (←)")
        self.btn_next = QPushButton("Next (→) ▶")
        self.btn_prev.setMinimumWidth(110)
        self.btn_next.setMinimumWidth(110)

        self.lbl_page_info = QLabel("Page 0 / 0")
        self.lbl_page_info.setStyleSheet("font-weight: 700; color: #f1f5f9; font-size: 13px;")

        self.slider_page = QSlider(Qt.Orientation.Horizontal)
        self.slider_page.setRange(0, 0)
        self.slider_page.valueChanged.connect(self._on_slider_changed)

        self.lbl_live_status = QLabel("Ready")
        self.lbl_live_status.setStyleSheet("color: #34d399; font-weight: 600; font-size: 12px;")

        bot_bar.addWidget(self.btn_prev)
        bot_bar.addWidget(self.lbl_page_info)
        bot_bar.addWidget(self.slider_page, stretch=1)
        bot_bar.addWidget(self.btn_next)
        bot_bar.addWidget(self.lbl_live_status)

        layout.addLayout(bot_bar)

        # Signals
        self.btn_prev.clicked.connect(self.prev_page)
        self.btn_next.clicked.connect(self.next_page)
        self.btn_fit.clicked.connect(self.canvas.fit_to_screen)
        self.btn_100.clicked.connect(self.canvas.set_100_percent)

        self.radio_color.toggled.connect(lambda c: self.canvas.set_view_mode("color") if c else None)
        self.radio_bw.toggled.connect(lambda c: self.canvas.set_view_mode("original") if c else None)
        self.radio_split.toggled.connect(lambda c: self.canvas.set_view_mode("split") if c else None)

        self.btn_open_folder.clicked.connect(self._on_browse_local_folder)
        self.combo_chapters.currentIndexChanged.connect(self._on_chapter_selected)

    def _on_browse_local_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Manga Chapter Folder")
        if folder:
            self.load_chapter_folder(folder)

    def load_chapter_folder(self, folder_path: str):
        from utils.image_utils import get_image_files
        from pathlib import Path
        imgs = [str(p) for p in get_image_files(Path(folder_path))]
        if imgs:
            ch_name = os.path.basename(folder_path)
            # Add to dropdown if not present
            found = False
            for i in range(self.combo_chapters.count()):
                if self.combo_chapters.itemData(i) == folder_path:
                    self.combo_chapters.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.combo_chapters.addItem(ch_name, folder_path)
                self.combo_chapters.setCurrentIndex(self.combo_chapters.count() - 1)
            self.load_chapter_pages(imgs)

    def _on_chapter_selected(self, idx: int):
        if idx < 0:
            return
        folder = self.combo_chapters.itemData(idx)
        if folder and os.path.isdir(folder):
            from utils.image_utils import get_image_files
            from pathlib import Path
            imgs = [str(p) for p in get_image_files(Path(folder))]
            if imgs:
                self.load_chapter_pages(imgs)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Left:
            self.prev_page()
        elif event.key() == Qt.Key.Key_Right:
            self.next_page()
        else:
            super().keyPressEvent(event)

    def load_chapter_pages(self, page_files: List[str], scan_output_dirs: Optional[List[str]] = None):
        """Populates pages for reading and automatically links any existing colorized pages."""
        self.pages_orig = sorted(page_files)
        self.current_index = 0
        total = len(self.pages_orig)
        self.slider_page.setRange(1, total if total > 0 else 0)
        if total > 0:
            self.slider_page.setValue(1)

        # Check for already colorized pages in common output folders or alongside original files
        for page_path in self.pages_orig:
            stem = os.path.splitext(os.path.basename(page_path))[0]
            dir_name = os.path.dirname(page_path)

            candidates = [
                dir_name.replace("manga_downloads", "output_colorized"),
                os.path.join(os.path.dirname(dir_name), "..", "output"),
                os.path.join(os.path.dirname(dir_name), "..", "output_colorized"),
                os.path.join(dir_name, "colorized"),
            ]
            if scan_output_dirs:
                candidates.extend(scan_output_dirs)

            for cand in candidates:
                cand = os.path.abspath(cand)
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    pot = os.path.join(cand, f"{stem}{ext}")
                    if os.path.isfile(pot) and os.path.getsize(pot) > 1000:
                        self.pages_colored[stem] = pot
                        break
                if stem in self.pages_colored:
                    break

        self._update_current_page_display()

    def update_colorized_page(self, page_name: str, color_file: str):
        """Called when a page finishes colorizing in background."""
        stem = os.path.splitext(page_name)[0]
        self.pages_colored[stem] = color_file
        # If currently viewing this page, refresh view immediately!
        if self.pages_orig and self.current_index < len(self.pages_orig):
            cur_stem = os.path.splitext(os.path.basename(self.pages_orig[self.current_index]))[0]
            if cur_stem == stem:
                self._update_current_page_display()
                self.lbl_live_status.setText("✓ Just Colorized!")

    def _update_current_page_display(self):
        if not self.pages_orig or self.current_index >= len(self.pages_orig):
            self.canvas.set_page_pixmaps(None, None)
            self.lbl_page_info.setText("Page 0 / 0")
            return

        orig_path = self.pages_orig[self.current_index]
        orig_pix = QPixmap(orig_path)
        stem = os.path.splitext(os.path.basename(orig_path))[0]

        color_pix = None
        if stem in self.pages_colored and os.path.exists(self.pages_colored[stem]):
            color_pix = QPixmap(self.pages_colored[stem])
            self.lbl_live_status.setText("● Colorized (Live)")
            self.lbl_live_status.setStyleSheet("color: #34d399; font-weight: 600;")
        else:
            self.lbl_live_status.setText("⏳ Coloring in background...")
            self.lbl_live_status.setStyleSheet("color: #fbbf24; font-weight: 600;")

        self.canvas.set_page_pixmaps(orig_pix, color_pix or orig_pix)
        self.lbl_page_info.setText(f"Page {self.current_index + 1} / {len(self.pages_orig)}")

    def prev_page(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.slider_page.setValue(self.current_index + 1)
            self._update_current_page_display()

    def next_page(self):
        if self.current_index < len(self.pages_orig) - 1:
            self.current_index += 1
            self.slider_page.setValue(self.current_index + 1)
            self._update_current_page_display()

    def _on_slider_changed(self, val: int):
        if val > 0 and val - 1 != self.current_index and val - 1 < len(self.pages_orig):
            self.current_index = val - 1
            self._update_current_page_display()
