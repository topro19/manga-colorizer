"""
Polished Consumer-grade Manga Colorizer Studio Window.
Simple, intuitive workflow:
[ Upload Manga / CBZ / PDF ]
[ Color instructions & Reference ]
[ Mode: Fast / Balanced / Quality ]
[ COLORIZE ]
-> Live Chapter Progress & Scene Continuity -> Before/After Split Viewer & Region Editor.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QRadioButton, QButtonGroup, QProgressBar,
    QStackedWidget, QSplitter, QFrame, QMessageBox
)

from app.gpu.providers import detect_gpu_environment
from app.characters.character_memory import CharacterMemoryManager
from app.models.local_onnx import LocalMangaV2Backend
from app.color.pipeline import ChapterPipelineWorker
from app.ui.editor import RegionCorrectionDialog
from app.storage.project import ProjectState, ProjectManager
from ui.viewer import ViewerPanel
from ui.styles import DARK_THEME_QSS
from app.utils.logging import logger


class ConsumerMainWindow(QMainWindow):
    """
    Watashi-style Consumer Manga Colorizer Window.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Colorizer Studio — AMD Radeon RX 9060 XT DirectML")
        self.resize(1360, 880)
        self.setStyleSheet(DARK_THEME_QSS)

        # Detect GPU
        self.gpu_info = detect_gpu_environment()
        self.char_memory = CharacterMemoryManager()

        # Initialize strong local backend
        model_path = Path("models/manga-colorize-fp16.onnx")
        self.backend = LocalMangaV2Backend(model_path, provider=self.gpu_info["active_provider"])
        self.backend.load()

        self.selected_source: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.pipeline_worker: Optional[ChapterPipelineWorker] = None

        self.completed_pages: List[Dict[str, Any]] = []
        self.current_preview_idx = 0

        self._build_ui()

    def _build_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(12)

        # Header Bar with GPU badge
        header_bar = QHBoxLayout()
        title_lbl = QLabel("MANGA COLORIZER STUDIO")
        title_font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet("color: #ffffff; letter-spacing: 1px;")
        header_bar.addWidget(title_lbl)

        header_bar.addStretch()

        gpu_badge = QLabel(self.gpu_info["display_text"])
        gpu_badge.setStyleSheet(
            "background-color: #1e1b4b; color: #a5b4fc; border: 1px solid #4338ca; "
            "padding: 5px 12px; border-radius: 12px; font-weight: 600; font-size: 11px;"
        )
        header_bar.addWidget(gpu_badge)
        main_layout.addLayout(header_bar)

        # Stacked Screens: 0 = Input & Setup, 1 = Processing, 2 = Results Viewer
        self.stack = QStackedWidget()
        self.stack.addWidget(self._create_setup_view())
        self.stack.addWidget(self._create_processing_view())
        self.stack.addWidget(self._create_results_view())
        main_layout.addWidget(self.stack, 1)

    def _create_setup_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setSpacing(16)
        layout.setContentsMargins(40, 20, 40, 20)

        # 1. Source Import Card
        import_box = QFrame()
        import_box.setStyleSheet("background-color: #171922; border: 2px dashed #374151; border-radius: 12px; padding: 24px;")
        import_layout = QVBoxLayout(import_box)
        import_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_import_status = QLabel("Drop Manga Folder, CBZ, or PDF here")
        self.lbl_import_status.setStyleSheet("font-size: 15px; font-weight: 600; color: #e2e8f0;")
        import_layout.addWidget(self.lbl_import_status)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.btn_browse_folder = QPushButton("📁 Select Folder")
        self.btn_browse_folder.clicked.connect(self._on_browse_folder)
        self.btn_browse_cbz = QPushButton("📦 Select CBZ")
        self.btn_browse_cbz.clicked.connect(self._on_browse_cbz)
        self.btn_browse_pdf = QPushButton("📄 Select PDF")
        self.btn_browse_pdf.clicked.connect(self._on_browse_pdf)

        self.btn_scrape_web = QPushButton("🌐 Scrape Chapter / PDF")
        self.btn_scrape_web.clicked.connect(self._on_scrape_web)

        for btn in (self.btn_browse_folder, self.btn_browse_cbz, self.btn_browse_pdf, self.btn_scrape_web):
            btn.setStyleSheet("background-color: #262a3b; color: #ffffff; padding: 8px 16px; border-radius: 6px; font-weight: 600;")
            btn_row.addWidget(btn)
        import_layout.addLayout(btn_row)
        layout.addWidget(import_box)

        # 2. Color Instructions
        inst_label = QLabel("Color Instructions (Optional):")
        inst_label.setStyleSheet("font-weight: 600; color: #cbd5e1; font-size: 13px;")
        layout.addWidget(inst_label)

        self.edit_instructions = QLineEdit()
        self.edit_instructions.setPlaceholderText("e.g. Black hair, blue jacket, pale skin, warm evening sunset lighting...")
        self.edit_instructions.setStyleSheet("background-color: #1a1c26; border: 1px solid #33384c; border-radius: 6px; padding: 10px; color: #ffffff;")
        layout.addWidget(self.edit_instructions)

        # 3. Quality Mode Radios
        mode_box = QHBoxLayout()
        mode_box.addWidget(QLabel("Quality Mode:"))
        self.radio_fast = QRadioButton("Fast (DirectML)")
        self.radio_balanced = QRadioButton("Balanced (Natural Anime)")
        self.radio_vivid = QRadioButton("🔥 Vivid Heavy (High Saturation)")
        self.radio_quality = QRadioButton("Quality (Chapter Synthesis)")
        self.radio_cloud = QRadioButton("☁️ Cloud Generative (Free AI SDXL)")
        self.radio_vivid.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radio_fast, 0)
        self.mode_group.addButton(self.radio_balanced, 1)
        self.mode_group.addButton(self.radio_vivid, 2)
        self.mode_group.addButton(self.radio_quality, 3)
        self.mode_group.addButton(self.radio_cloud, 4)

        mode_box.addWidget(self.radio_fast)
        mode_box.addWidget(self.radio_balanced)
        mode_box.addWidget(self.radio_vivid)
        mode_box.addWidget(self.radio_quality)
        mode_box.addWidget(self.radio_cloud)
        mode_box.addStretch()
        layout.addLayout(mode_box)

        # 4. Colorize Button
        self.btn_start = QPushButton("✨ COLORIZE CHAPTER")
        self.btn_start.setFixedHeight(48)
        self.btn_start.setStyleSheet(
            "background-color: #4f46e5; color: #ffffff; font-size: 15px; font-weight: bold; border-radius: 8px;"
        )
        self.btn_start.clicked.connect(self._start_pipeline)
        layout.addWidget(self.btn_start)
        layout.addStretch()

        return view

    def _create_processing_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        self.lbl_proc_stage = QLabel("Analyzing Chapter...")
        self.lbl_proc_stage.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        layout.addWidget(self.lbl_proc_stage)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #1e2230; border-radius: 6px; text-align: center; } "
            "QProgressBar::chunk { background-color: #6366f1; border-radius: 6px; }"
        )
        layout.addWidget(self.progress_bar)

        self.lbl_proc_details = QLabel("Page 0 / 0 | Scene 1 | 94% Consistency")
        self.lbl_proc_details.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(self.lbl_proc_details)

        btn_row = QHBoxLayout()
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        return view

    def _create_results_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Nav bar
        nav = QHBoxLayout()
        self.btn_prev_page = QPushButton("◀ Previous Page")
        self.btn_prev_page.clicked.connect(self._on_prev_page)
        self.lbl_page_counter = QLabel("Page 1 / 1")
        self.lbl_page_counter.setStyleSheet("font-weight: 600; color: #ffffff;")
        self.btn_next_page = QPushButton("Next Page ▶")
        self.btn_next_page.clicked.connect(self._on_next_page)

        self.btn_fix_region = QPushButton("🎨 Fix Region Color")
        self.btn_fix_region.setStyleSheet("background-color: #4338ca; color: #ffffff; padding: 6px 14px; font-weight: bold; border-radius: 4px;")
        self.btn_fix_region.clicked.connect(self._on_fix_region)

        nav.addWidget(self.btn_prev_page)
        nav.addWidget(self.lbl_page_counter)
        nav.addWidget(self.btn_next_page)
        nav.addStretch()
        nav.addWidget(self.btn_fix_region)
        layout.addLayout(nav)

        # Split before/after viewer
        self.split_viewer = ViewerPanel()
        layout.addWidget(self.split_viewer, 1)

        return view

    def _on_browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select Manga Folder")
        if f:
            self.selected_source = Path(f)
            self.lbl_import_status.setText(f"Folder: {self.selected_source.name}")

    def _on_browse_cbz(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select CBZ Archive", "", "CBZ Files (*.cbz);;All Files (*)")
        if f:
            self.selected_source = Path(f)
            self.lbl_import_status.setText(f"CBZ: {self.selected_source.name}")

    def _on_browse_pdf(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select PDF Document", "", "PDF Files (*.pdf);;All Files (*)")
        if f:
            self.selected_source = Path(f)
            self.lbl_import_status.setText(f"PDF: {self.selected_source.name}")

    def _on_scrape_web(self):
        try:
            from ui.scraper_dialog import MangaScraperDialog
            dlg = MangaScraperDialog(self)
            if dlg.exec():
                if getattr(dlg, 'result_pdf', None) and dlg.result_pdf.exists():
                    self.selected_source = dlg.result_pdf
                    self.lbl_import_status.setText(f"Scraped PDF: {self.selected_source.name}")
                elif getattr(dlg, 'result_dir', None) and dlg.result_dir.exists():
                    self.selected_source = dlg.result_dir
                    self.lbl_import_status.setText(f"Scraped Folder: {self.selected_source.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error Launching Scraper", str(e))

    def _start_pipeline(self):
        if not self.selected_source:
            QMessageBox.warning(self, "No Source", "Please select a Manga Folder, CBZ, or PDF first.")
            return

        out = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out:
            return
        self.output_dir = Path(out)

        self.completed_pages.clear()
        self.stack.setCurrentIndex(1)

        if self.radio_cloud.isChecked():
            mode = "CLOUD"
        elif self.radio_vivid.isChecked():
            mode = "VIVID"
        elif self.radio_balanced.isChecked():
            mode = "BALANCED"
        elif self.radio_fast.isChecked():
            mode = "FAST"
        else:
            mode = "QUALITY"
        
        # Select appropriate backend based on mode
        if mode == "CLOUD":
            from app.models.cloud_backend import CloudGenerativeBackend
            active_backend = CloudGenerativeBackend()
            active_backend.load()
        elif mode == "QUALITY":
            from app.models.diffusion_backend import DiffusionQualityBackend
            active_backend = DiffusionQualityBackend()
            active_backend.load()
        else:
            active_backend = self.backend

        self.pipeline_worker = ChapterPipelineWorker(
            source_path=self.selected_source,
            output_dir=self.output_dir,
            backend=active_backend,
            char_memory=self.char_memory,
            color_instructions=self.edit_instructions.text().strip(),
            quality_mode=mode
        )

        self.pipeline_worker.status_updated.connect(self.lbl_proc_stage.setText)
        self.pipeline_worker.page_progress.connect(self._on_page_progress)
        self.pipeline_worker.page_finished.connect(self._on_page_finished)
        self.pipeline_worker.chapter_finished.connect(self._on_chapter_finished)
        self.pipeline_worker.error_occurred.connect(self._on_pipeline_error)

        self.pipeline_worker.start()

    def _on_page_progress(self, cur: int, total: int, name: str):
        pct = int((cur / max(1, total)) * 100)
        self.progress_bar.setValue(pct)
        self.lbl_proc_details.setText(f"Page {cur} / {total} — {name}")

    def _on_page_finished(self, idx: int, name: str, orig_bgr: np.ndarray, color_bgr: np.ndarray):
        self.completed_pages.append({
            "idx": idx,
            "name": name,
            "orig_bgr": orig_bgr,
            "color_bgr": color_bgr
        })

    def _on_chapter_finished(self, total: int, out_dir: str):
        self.stack.setCurrentIndex(2)
        if self.completed_pages:
            self.current_preview_idx = 0
            self._display_preview_page()

    def _on_pipeline_error(self, err: str):
        QMessageBox.critical(self, "Colorization Error", err)
        self.stack.setCurrentIndex(0)

    def _on_pause_clicked(self):
        if self.pipeline_worker:
            if self.btn_pause.text() == "Pause":
                self.pipeline_worker.pause()
                self.btn_pause.setText("Resume")
            else:
                self.pipeline_worker.resume()
                self.btn_pause.setText("Pause")

    def _on_cancel_clicked(self):
        if self.pipeline_worker:
            self.pipeline_worker.cancel()
            self.stack.setCurrentIndex(0)

    def _display_preview_page(self):
        if not self.completed_pages:
            return
        p = self.completed_pages[self.current_preview_idx]
        self.lbl_page_counter.setText(f"Page {self.current_preview_idx + 1} / {len(self.completed_pages)}")
        self.split_viewer.set_images(p["orig_bgr"], p["color_bgr"])

    def _on_prev_page(self):
        if self.current_preview_idx > 0:
            self.current_preview_idx -= 1
            self._display_preview_page()

    def _on_next_page(self):
        if self.current_preview_idx < len(self.completed_pages) - 1:
            self.current_preview_idx += 1
            self._display_preview_page()

    def _on_fix_region(self):
        if not self.completed_pages:
            return
        p = self.completed_pages[self.current_preview_idx]
        dlg = RegionCorrectionDialog(p["orig_bgr"], p["color_bgr"], self)
        dlg.region_corrected.connect(lambda updated_bgr: self._on_region_updated(updated_bgr))
        dlg.exec()

    def _on_region_updated(self, updated_bgr: np.ndarray):
        p = self.completed_pages[self.current_preview_idx]
        p["color_bgr"] = updated_bgr
        self.split_viewer.set_images(p["orig_bgr"], updated_bgr)
        # Update on disk
        if self.output_dir:
            out_file = self.output_dir / f"color_{p['name']}"
            cv2.imwrite(str(out_file), updated_bgr)
