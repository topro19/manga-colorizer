"""
Main Application Window for MangaColorizer Studio.
Professional desktop UI featuring:
- 4 Workspaces:
    1. Studio: Colorization with FAST/BALANCED/QUALITY modes, L-preservation, Line protection, speech bubble masks.
    2. Model Lab: Multi-model comparison.
    3. Color Memory: Character consistency, color locks, and reference palette extraction.
    4. Benchmark: Developer hardware & latency profiling.
- Interactive 4-State Viewer: Original vs Raw AI vs Guided vs Final (Split, Side-by-Side, Blink, Diff).
- Batch volume colorization with pause, resume, cancel, and project_manifest.json logging.
- DirectML hardware acceleration on AMD Radeon RX 9060 XT 16GB.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from PIL import Image
from PySide6.QtCore import QSize, Qt, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.backends.colorizer_registry import ColorizerRegistry
from core.batch_processor import BatchProcessor
from core.model_manager import ModelManager
from core.palette_manager import PaletteManager
from core.vlm_director import LocalRuleVLMDirector
from utils.cbz import extract_cbz
from utils.gpu_monitor import get_windows_gpu_info
from utils.image_utils import get_image_files
from utils.pdf import extract_pdf_pages

from .benchmark_screen import DeveloperBenchmarkScreen
from .hint_editor import ColorHintDialog
from .live_reader import LiveMangaReaderPanel
from .model_lab import ModelLabPanel
from .palette_panel import PalettePanel
from .progress import ProgressPanel
from .styles import DARK_THEME_QSS
from .viewer import ViewerPanel
from utils.mangadot_downloader import MangaDotDownloadWorker


class MainWindow(QMainWindow):
    def __init__(self, model_manager: ModelManager):
        super().__init__()
        self.setWindowTitle("MangaColorizer Studio — AMD Radeon RX 9060 XT DirectML")
        self.resize(1440, 920)
        self.setStyleSheet(DARK_THEME_QSS)

        self.model_manager = model_manager
        
        # Pluggable components
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        self.registry = ColorizerRegistry(models_dir=models_dir, provider="DmlExecutionProvider")
        self.palette_manager = PaletteManager()
        self.vlm_director = LocalRuleVLMDirector()

        self.batch_worker: Optional[BatchProcessor] = None
        self.download_worker: Optional[MangaDotDownloadWorker] = None

        # State
        self.input_path: Optional[Path] = None
        self.input_type: str = "folder"
        self.current_pages: List[Path] = []
        self.hints_store: Dict[str, List[Dict[str, Any]]] = {}

        self._build_ui()
        self._update_gpu_status()
        self._connect_reader_events()

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # 1. Top Header Bar
        header_bar = self._create_header_bar()
        main_layout.addLayout(header_bar)

        # 2. Main Content (Nav Rail + Workspaces)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        nav_panel = self._create_nav_rail()
        content_layout.addWidget(nav_panel)

        self.stacked_workspaces = QStackedWidget()
        
        # Workspace 0: Studio
        self.view_studio = self._create_studio_workspace()
        self.stacked_workspaces.addWidget(self.view_studio)

        # Workspace 1: Live Manga Reader
        self.live_reader = LiveMangaReaderPanel(self)
        self.stacked_workspaces.addWidget(self.live_reader)

        # Workspace 2: Model Lab
        self.model_lab = ModelLabPanel(self.registry, self)
        self.stacked_workspaces.addWidget(self.model_lab)

        # Workspace 3: Character Consistency & Palette
        self.palette_view = PalettePanel(self.palette_manager, self)
        self.stacked_workspaces.addWidget(self.palette_view)

        # Workspace 4: Developer Benchmark Screen
        self.benchmark_screen = DeveloperBenchmarkScreen(self.registry, self)
        self.stacked_workspaces.addWidget(self.benchmark_screen)

        content_layout.addWidget(self.stacked_workspaces, stretch=1)
        main_layout.addLayout(content_layout, stretch=1)

    def _create_header_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 6)

        title = QLabel("MangaColorizer Studio")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc; letter-spacing: 0.5px;")

        subtitle = QLabel("AMD Radeon RX 9060 XT • DirectML • Production Pipeline")
        subtitle.setStyleSheet("font-size: 12px; color: #818cf8; margin-left: 10px; font-weight: 600;")

        bar.addWidget(title)
        bar.addWidget(subtitle)
        bar.addStretch()

        # Engine selector
        bar.addWidget(QLabel("Active Engine:"))
        self.combo_active_model = QComboBox()
        self.combo_active_model.setMinimumWidth(210)
        for m in self.registry.list_available_models():
            self.combo_active_model.addItem(m)
        bar.addWidget(self.combo_active_model)

        bar.addSpacing(10)

        # GPU Status Badge
        self.lbl_gpu_badge = QLabel("DirectML: Detecting...")
        self.lbl_gpu_badge.setObjectName("GpuBadge")
        bar.addWidget(self.lbl_gpu_badge)

        self.btn_test_gpu = QPushButton("⚡ Test GPU")
        self.btn_test_gpu.setStyleSheet("background-color: #312e81; color: #c7d2fe; border: 1px solid #4f46e5; font-weight: 600; padding: 5px 12px;")
        self.btn_test_gpu.clicked.connect(self._run_gpu_test)
        bar.addWidget(self.btn_test_gpu)

        return bar

    def _create_nav_rail(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(160)
        panel.setStyleSheet("QFrame { background-color: #13151f; border: 1px solid #232738; border-radius: 8px; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(8)

        lbl = QLabel("WORKSPACES")
        lbl.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        layout.addWidget(lbl)

        self.btn_nav_studio = QPushButton("🎨 Studio")
        self.btn_nav_reader = QPushButton("📖 Manga Reader")
        self.btn_nav_lab = QPushButton("🧪 Model Lab")
        self.btn_nav_palette = QPushButton("🎭 Color Memory")
        self.btn_nav_bench = QPushButton("📊 Benchmark")

        for b in [self.btn_nav_studio, self.btn_nav_reader, self.btn_nav_lab, self.btn_nav_palette, self.btn_nav_bench]:
            b.setCheckable(True)
            b.setStyleSheet(
                "QPushButton { text-align: left; padding: 10px 12px; font-weight: 600; border-radius: 6px; } "
                "QPushButton:checked { background-color: #4338ca; color: #ffffff; border: 1px solid #6366f1; }"
            )

        self.nav_group = QButtonGroup(self)
        self.nav_group.addButton(self.btn_nav_studio, 0)
        self.nav_group.addButton(self.btn_nav_reader, 1)
        self.nav_group.addButton(self.btn_nav_lab, 2)
        self.nav_group.addButton(self.btn_nav_palette, 3)
        self.nav_group.addButton(self.btn_nav_bench, 4)
        self.btn_nav_studio.setChecked(True)

        self.nav_group.idClicked.connect(self._on_workspace_switched)

        layout.addWidget(self.btn_nav_studio)
        layout.addWidget(self.btn_nav_reader)
        layout.addWidget(self.btn_nav_lab)
        layout.addWidget(self.btn_nav_palette)
        layout.addWidget(self.btn_nav_bench)
        layout.addStretch()

        return panel

    def _on_workspace_switched(self, idx: int):
        self.stacked_workspaces.setCurrentIndex(idx)
        if self.current_pages:
            active_img = str(self.current_pages[0])
            if idx == 2:
                self.model_lab.set_active_image(active_img)
            elif idx == 4:
                self.benchmark_screen.set_active_image(active_img)

    def _create_studio_workspace(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Left Sidebar (Inputs & Pipeline Sliders)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(360)
        sidebar_scroll.setMaximumWidth(430)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(4, 0, 10, 0)
        sidebar_layout.setSpacing(12)

        sidebar_layout.addWidget(self._create_input_group())
        sidebar_layout.addWidget(self._create_pipeline_tuning_group())
        sidebar_layout.addWidget(self._create_output_group())
        sidebar_layout.addStretch()

        sidebar_scroll.setWidget(sidebar_content)
        layout.addWidget(sidebar_scroll)

        # Right Area (Viewer + Progress + Actions + Logs)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.viewer_panel = ViewerPanel(self)
        right_layout.addWidget(self.viewer_panel, stretch=1)

        right_layout.addLayout(self._create_action_bar())

        self.progress_panel = ProgressPanel(self)
        right_layout.addWidget(self.progress_panel)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(85)
        self.log_view.setStyleSheet("background-color: #0b0d14; border: 1px solid #212638; color: #94a3b8; font-family: Consolas, monospace; font-size: 11px;")
        self.log_view.setPlaceholderText("DirectML logs and batch execution status...")
        right_layout.addWidget(self.log_view)

        layout.addWidget(right_container, stretch=1)
        return container

    def _create_input_group(self) -> QGroupBox:
        group = QGroupBox("1. Input Manga Source")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        btn_row = QHBoxLayout()
        self.btn_sel_folder = QPushButton("📁 Folder")
        self.btn_sel_cbz = QPushButton("📦 CBZ")
        self.btn_sel_pdf = QPushButton("📄 PDF")

        self.btn_sel_folder.clicked.connect(self._on_select_folder)
        self.btn_sel_cbz.clicked.connect(self._on_select_cbz)
        self.btn_sel_pdf.clicked.connect(self._on_select_pdf)

        btn_row.addWidget(self.btn_sel_folder)
        btn_row.addWidget(self.btn_sel_cbz)
        btn_row.addWidget(self.btn_sel_pdf)
        layout.addLayout(btn_row)

        self.lbl_input_path = QLabel("No manga source selected.")
        self.lbl_input_path.setWordWrap(True)
        self.lbl_input_path.setStyleSheet("color: #94a3b8; font-size: 11px; font-family: monospace;")
        layout.addWidget(self.lbl_input_path)

        self.lbl_page_total = QLabel("Pages found: 0")
        self.lbl_page_total.setStyleSheet("color: #a5b4fc; font-weight: 600;")
        layout.addWidget(self.lbl_page_total)

        self.list_thumbnails = QListWidget()
        self.list_thumbnails.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_thumbnails.setIconSize(QSize(70, 100))
        self.list_thumbnails.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_thumbnails.setMaximumHeight(120)
        self.list_thumbnails.setStyleSheet("background-color: #0f111a; border: 1px solid #232738; border-radius: 6px;")
        self.list_thumbnails.itemClicked.connect(self._on_thumbnail_clicked)
        layout.addWidget(self.list_thumbnails)

        return group

    def _create_pipeline_tuning_group(self) -> QGroupBox:
        group = QGroupBox("2. Anti-Bleeding & Quality Pipeline")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # Pipeline Mode (FAST / BALANCED / QUALITY)
        mode_box = QHBoxLayout()
        mode_box.addWidget(QLabel("Pipeline Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["FAST", "BALANCED", "QUALITY"])
        self.combo_mode.setCurrentText("BALANCED")
        mode_box.addWidget(self.combo_mode)
        layout.addLayout(mode_box)

        # Color Strength Slider (Default 85%)
        cs_box = QVBoxLayout()
        self.lbl_color_strength = QLabel("Color Strength: 85%")
        self.lbl_color_strength.setStyleSheet("font-weight: 600;")
        self.slider_color_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_color_strength.setRange(0, 100)
        self.slider_color_strength.setValue(85)
        self.slider_color_strength.valueChanged.connect(
            lambda v: self.lbl_color_strength.setText(f"Color Strength: {v}%")
        )
        cs_box.addWidget(self.lbl_color_strength)
        cs_box.addWidget(self.slider_color_strength)
        layout.addLayout(cs_box)

        # L-Preservation Slider (Default 95%)
        lp_box = QVBoxLayout()
        self.lbl_l_preservation = QLabel("L-Channel Preservation: 95%")
        self.lbl_l_preservation.setStyleSheet("font-weight: 600;")
        self.slider_l_preservation = QSlider(Qt.Orientation.Horizontal)
        self.slider_l_preservation.setRange(0, 100)
        self.slider_l_preservation.setValue(95)
        self.slider_l_preservation.valueChanged.connect(
            lambda v: self.lbl_l_preservation.setText(f"L-Channel Preservation: {v}%")
        )
        lp_box.addWidget(self.lbl_l_preservation)
        lp_box.addWidget(self.slider_l_preservation)
        layout.addLayout(lp_box)

        # Line Art Preservation Slider (Default 95%)
        la_box = QVBoxLayout()
        self.lbl_line_preservation = QLabel("Line Art Protection: 95%")
        self.lbl_line_preservation.setStyleSheet("font-weight: 600;")
        self.slider_line_preservation = QSlider(Qt.Orientation.Horizontal)
        self.slider_line_preservation.setRange(0, 100)
        self.slider_line_preservation.setValue(95)
        self.slider_line_preservation.valueChanged.connect(
            lambda v: self.lbl_line_preservation.setText(f"Line Art Protection: {v}%")
        )
        la_box.addWidget(self.lbl_line_preservation)
        la_box.addWidget(self.slider_line_preservation)
        layout.addLayout(la_box)

        # Speech bubble protection checkbox
        self.chk_speech_bubbles = QCheckBox("Speech bubble & text protection (clean white)")
        self.chk_speech_bubbles.setChecked(True)
        layout.addWidget(self.chk_speech_bubbles)

        # Fast guided filter checkbox
        self.chk_guided_filter = QCheckBox("Edge-guided cross-bilateral filter (anti-bleed)")
        self.chk_guided_filter.setChecked(True)
        layout.addWidget(self.chk_guided_filter)

        # Interactive Hint Editor launcher
        self.btn_open_hints = QPushButton("🎨 Add Color Hints to Current Page...")
        self.btn_open_hints.setStyleSheet("background-color: #1e1b4b; border: 1px solid #4338ca; color: #a5b4fc; font-weight: 600; padding: 7px;")
        self.btn_open_hints.clicked.connect(self._open_hint_editor)
        layout.addWidget(self.btn_open_hints)

        return group

    def _create_output_group(self) -> QGroupBox:
        group = QGroupBox("3. Output & Resuming")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        dir_layout = QHBoxLayout()
        default_out = str(Path(__file__).resolve().parent.parent / "output")
        self.edit_output_dir = QLineEdit(default_out)
        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self._on_browse_output)
        dir_layout.addWidget(self.edit_output_dir)
        dir_layout.addWidget(btn_browse_out)
        layout.addLayout(dir_layout)

        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("Format:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["PNG (Lossless)", "JPEG"])
        fmt_layout.addWidget(self.combo_format)
        layout.addLayout(fmt_layout)

        self.chk_create_cbz = QCheckBox("Export as .CBZ archive")
        self.chk_create_pdf = QCheckBox("Export as .PDF document")
        self.chk_reprocess = QCheckBox("Reprocess existing pages (force overwrite)")
        layout.addWidget(self.chk_create_cbz)
        layout.addWidget(self.chk_create_pdf)
        layout.addWidget(self.chk_reprocess)

        return group

    def _create_action_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(10)

        self.btn_colorize = QPushButton("▶ Colorize Manga Volume")
        self.btn_colorize.setObjectName("PrimaryBtn")
        self.btn_colorize.setMinimumHeight(44)
        self.btn_colorize.clicked.connect(self._start_colorization)

        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._pause_colorization)

        self.btn_resume = QPushButton("⏯ Resume")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self._resume_colorization)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_colorization)

        bar.addWidget(self.btn_colorize, stretch=2)
        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_resume)
        bar.addWidget(self.btn_stop)

        return bar

    def _update_gpu_status(self) -> None:
        badge, detailed, is_gpu = self.model_manager.get_status_labels()
        self.lbl_gpu_badge.setText(detailed)
        if is_gpu:
            self.lbl_gpu_badge.setStyleSheet("background-color: #064e3b; color: #34d399; border: 1px solid #059669; border-radius: 12px; padding: 4px 12px; font-weight: 700;")
        else:
            self.lbl_gpu_badge.setStyleSheet("background-color: #451a03; color: #fbbf24; border: 1px solid #d97706; border-radius: 12px; padding: 4px 12px; font-weight: 700;")

    def _run_gpu_test(self) -> None:
        self.log_message("Executing DirectML GPU diagnostic...")
        self.btn_test_gpu.setEnabled(False)
        try:
            diag = self.model_manager.test_gpu()
            title = "DirectML Test Passed" if diag["success"] else "DirectML Test Notice"
            icon = QMessageBox.Icon.Information if diag["success"] else QMessageBox.Icon.Warning
            QMessageBox(icon, title, diag["message"], QMessageBox.StandardButton.Ok, self).exec()
            self.log_message(diag["message"])
            self._update_gpu_status()
        finally:
            self.btn_test_gpu.setEnabled(True)

    def _on_select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Manga Image Folder")
        if folder:
            self.input_path = Path(folder)
            self.input_type = "folder"
            self._scan_and_load_thumbnails()

    def _on_select_cbz(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Comic Archive (.cbz)", filter="CBZ Archives (*.cbz *.zip)")
        if file_path:
            self.input_path = Path(file_path)
            self.input_type = "cbz"
            self.chk_create_cbz.setChecked(True)
            self._scan_and_load_thumbnails()

    def _on_select_pdf(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Manga PDF Document (.pdf)", filter="PDF Documents (*.pdf)")
        if file_path:
            self.input_path = Path(file_path)
            self.input_type = "pdf"
            self.chk_create_pdf.setChecked(True)
            self._scan_and_load_thumbnails()

    def _scan_and_load_thumbnails(self) -> None:
        if not self.input_path or not self.input_path.exists():
            return

        self.list_thumbnails.clear()
        self.lbl_input_path.setText(str(self.input_path))
        self.log_message(f"Scanning manga source: {self.input_path.name}...")

        if self.input_type == "folder":
            self.current_pages = get_image_files(self.input_path, recursive=False)
        elif self.input_type == "cbz":
            import zipfile
            self.current_pages = []
            with zipfile.ZipFile(self.input_path, "r") as archive:
                for name in archive.namelist():
                    if Path(name).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        self.current_pages.append(Path(name))
        elif self.input_type == "pdf":
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(self.input_path)
            self.current_pages = [Path(f"Page_{i+1:04d}.png") for i in range(len(doc))]

        count = len(self.current_pages)
        self.lbl_page_total.setText(f"Pages found: {count}")
        self.log_message(f"Discovered {count} page(s).")

        for p in self.current_pages[:30]:
            item = QListWidgetItem(p.name)
            self.list_thumbnails.addItem(item)

        if self.input_type == "folder" and self.current_pages:
            first_path = self.current_pages[0]
            orig_pix = QPixmap(str(first_path))
            self.viewer_panel.comparison_widget.set_all_images(orig_pix, orig_pix, orig_pix, orig_pix)
            self.model_lab.set_active_image(str(first_path))
            self.benchmark_screen.set_active_image(str(first_path))

    def _on_thumbnail_clicked(self, item: QListWidgetItem) -> None:
        if self.input_type == "folder":
            target = self.input_path / item.text()
            if target.is_file():
                pix = QPixmap(str(target))
                self.viewer_panel.comparison_widget.set_all_images(pix, pix, pix, pix)
                self.viewer_panel.info_label.setText(f"Selected: {item.text()}")
                self.model_lab.set_active_image(str(target))
                self.benchmark_screen.set_active_image(str(target))

    def _on_browse_output(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.edit_output_dir.setText(dir_path)

    def _open_hint_editor(self) -> None:
        if not self.current_pages or self.input_type != "folder":
            QMessageBox.information(self, "Hint Editor", "Please select a manga folder first to open a page in the Color Hint Editor.")
            return

        first_page = str(self.current_pages[0])
        dlg = ColorHintDialog(first_page, self)
        if dlg.exec():
            hints = dlg.get_hints()
            if dlg.is_apply_to_all():
                self.hints_store["*"] = hints
            else:
                self.hints_store[Path(first_page).name] = hints
            self.log_message(f"Saved {len(hints)} interactive color hints.")

    def _start_colorization(self) -> None:
        if not self.input_path:
            QMessageBox.warning(self, "No Input", "Please select a manga source first.")
            return

        output_dir = self.edit_output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "No Output", "Please choose an output directory.")
            return

        selected_model_name = self.combo_active_model.currentText()
        backend = self.registry.get_backend(selected_model_name)
        if not backend:
            QMessageBox.critical(self, "Model Error", f"Model '{selected_model_name}' could not be initialized.")
            return

        fmt = "JPEG" if self.combo_format.currentIndex() == 1 else "PNG"
        mode = self.combo_mode.currentText()
        color_strength = float(self.slider_color_strength.value()) / 100.0
        l_preservation = float(self.slider_l_preservation.value()) / 100.0
        line_protection = float(self.slider_line_preservation.value()) / 100.0
        protect_bubbles = self.chk_speech_bubbles.isChecked()
        use_guided_filter = self.chk_guided_filter.isChecked()
        reprocess = self.chk_reprocess.isChecked()
        create_cbz_out = self.chk_create_cbz.isChecked()
        create_pdf_out = self.chk_create_pdf.isChecked()

        self.btn_colorize.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_panel.reset()

        self.batch_worker = BatchProcessor(
            colorizer=backend,
            vlm_director=self.vlm_director,
            palette_manager=self.palette_manager,
            input_source=self.input_path,
            source_type=self.input_type,
            output_dir=output_dir,
            output_format=fmt,
            pipeline_mode=mode,
            color_strength=color_strength,
            l_preservation=l_preservation,
            line_protection=line_protection,
            protect_bubbles=protect_bubbles,
            use_guided_filter=use_guided_filter,
            reprocess_existing=reprocess,
            create_cbz_output=create_cbz_out,
            create_pdf_output=create_pdf_out,
            page_hints=self.hints_store,
        )

        self.batch_worker.page_started.connect(self._on_page_started)
        self.batch_worker.page_completed.connect(self._on_page_completed)
        self.batch_worker.page_failed.connect(self._on_page_failed)
        self.batch_worker.progress_updated.connect(self._on_progress_updated)
        self.batch_worker.batch_finished.connect(self._on_batch_finished)
        self.batch_worker.status_message.connect(self.log_message)

        self.batch_worker.start()

    def _pause_colorization(self) -> None:
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.pause()
            self.btn_pause.setEnabled(False)
            self.btn_resume.setEnabled(True)

    def _resume_colorization(self) -> None:
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.resume()
            self.btn_pause.setEnabled(True)
            self.btn_resume.setEnabled(False)

    def _stop_colorization(self) -> None:
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop()
            self.btn_stop.setEnabled(False)

    @Slot(int, int, str)
    def _on_page_started(self, current: int, total: int, name: str) -> None:
        self.progress_panel.set_status(f"Colorizing page {current}/{total}: {name}...")

    @Slot(int, int, str, bytes, bytes, bytes, bytes, float, dict, str)
    def _on_page_completed(
        self,
        current: int,
        total: int,
        name: str,
        orig_bytes: bytes,
        raw_bytes: bytes,
        guided_bytes: bytes,
        final_bytes: bytes,
        ms: float,
        timings: dict,
        out_path: str = ""
    ) -> None:
        info_str = f"Page {current} / {total} — {name} ({ms:.0f} ms)"
        self.viewer_panel.set_images_all_layers(orig_bytes, raw_bytes, guided_bytes, final_bytes, info_str)
        self.progress_panel.update_latency_stages(timings)
        self.log_message(f"✓ Completed {name} in {ms:.0f} ms (Inference: {timings.get('inference_ms', 0):.0f} ms)")
        if out_path:
            self.live_reader.update_colorized_page(name, out_path)

    @Slot(int, int, str, str)
    def _on_page_failed(self, current: int, total: int, name: str, reason: str) -> None:
        self.progress_panel.set_status(f"Page {current} ({name}) failed: {reason}", is_error=True)
        self.log_message(f"✗ Failed {name}: {reason}")

    @Slot(int, int, float, float, str, str, str)
    def _on_progress_updated(
        self,
        current: int,
        total: int,
        percent: float,
        ppm: float,
        eta: str,
        elapsed: str,
        stage_info: str
    ) -> None:
        self.progress_panel.update_progress(current, total, percent, ppm, eta, elapsed, stage_info)

    @Slot(int, int, str, str)
    def _on_batch_finished(self, processed: int, failed: int, elapsed: str, final_path: str) -> None:
        self.btn_colorize.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)

        summary = f"Volume colorization completed in {elapsed}.\nSuccessfully processed: {processed} page(s)\nFailed: {failed} page(s)\nManifest: {final_path}"
        self.progress_panel.set_status("Colorization completed!")
        self.log_message(summary)
        QMessageBox.information(self, "Volume Completed", summary)

    def _connect_reader_events(self) -> None:
        self.live_reader.btn_fetch.clicked.connect(self._start_mangadot_download)
        self.live_reader.request_start_colorization.connect(self._on_reader_request_colorization)

        # Pre-scan any existing downloaded chapters into the reader dropdown
        manga_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "input", "manga_downloads"))
        if os.path.isdir(manga_base):
            for root, dirs, files in os.walk(manga_base):
                if any(f.lower().endswith((".webp", ".png", ".jpg", ".jpeg")) for f in files):
                    rel = os.path.relpath(root, manga_base).replace("\\", " / ")
                    self.live_reader.combo_chapters.addItem(f"📖 {rel}", root)
            if self.live_reader.combo_chapters.count() > 0:
                self.live_reader.combo_chapters.setCurrentIndex(0)

    def _start_mangadot_download(self) -> None:
        url = self.live_reader.edit_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Manga Reader", "Please enter a valid MangaDot URL (e.g. https://mangadot.net/chapter/37925).")
            return

        try:
            ch_start = float(self.live_reader.edit_start_ch.text().strip())
            ch_end = float(self.live_reader.edit_end_ch.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Manga Reader", "Please specify valid numeric chapter numbers.")
            return

        out_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "input", "manga_downloads"))
        os.makedirs(out_base, exist_ok=True)

        self.live_reader.btn_fetch.setEnabled(False)
        self.live_reader.lbl_live_status.setText("Connecting to MangaDot...")
        self.live_reader.lbl_live_status.setStyleSheet("color: #38bdf8; font-weight: 600;")
        self.log_message(f"Starting MangaDot fetch for URL: {url} (Chapters {ch_start} - {ch_end})")

        self.download_worker = MangaDotDownloadWorker(
            target_url=url,
            start_chapter=ch_start,
            end_chapter=ch_end,
            base_output_dir=out_base,
            parent=self
        )

        self.download_worker.manga_info_loaded.connect(self._on_manga_info_loaded)
        self.download_worker.page_downloaded.connect(self._on_reader_page_downloaded)
        self.download_worker.chapter_finished.connect(self._on_reader_chapter_finished)
        self.download_worker.all_downloads_completed.connect(self._on_reader_all_downloads_completed)
        self.download_worker.error_occurred.connect(self._on_reader_download_error)
        self.download_worker.start()

    def _on_manga_info_loaded(self, title: str, chapters: list) -> None:
        self.log_message(f"Found manga '{title}' with {len(chapters)} available chapters.")
        self.live_reader.lbl_live_status.setText(f"Found: {title}")

    def _on_reader_page_downloaded(self, cur: int, tot: int, ch_label: str, fpath: str) -> None:
        self.live_reader.lbl_live_status.setText(f"Downloading {ch_label}: {cur}/{tot}")
        # If this is the first page of the chapter, show it immediately so the user can start reading!
        if cur == 1:
            self.live_reader.load_chapter_pages([fpath])

    def _on_reader_chapter_finished(self, ch_label: str, folder_path: str) -> None:
        self.log_message(f"✓ Downloaded {ch_label} into {folder_path}")
        colored_dir = folder_path.replace("manga_downloads", "output_colorized")
        os.makedirs(colored_dir, exist_ok=True)

        # Populate chapter in reader dropdown and scan already colored files
        self.live_reader.load_chapter_folder(folder_path)

        # Auto-trigger DirectML background colorization for this downloaded chapter!
        self._start_reader_background_colorization(folder_path, colored_dir)

    def _on_reader_all_downloads_completed(self, folders: list) -> None:
        self.live_reader.btn_fetch.setEnabled(True)
        self.live_reader.lbl_live_status.setText("All Chapters Ready")
        self.live_reader.lbl_live_status.setStyleSheet("color: #34d399; font-weight: 600;")
        self.log_message("All targeted manga chapters have been fetched.")

    def _on_reader_download_error(self, err: str) -> None:
        self.live_reader.btn_fetch.setEnabled(True)
        self.live_reader.lbl_live_status.setText(f"Fetch Error: {err[:20]}...")
        self.live_reader.lbl_live_status.setStyleSheet("color: #ef4444; font-weight: 600;")
        self.log_message(f"✗ MangaDot download error: {err}")
        QMessageBox.warning(self, "Download Error", f"Failed to download chapter:\n{err}")

    def _on_reader_request_colorization(self, in_dir: str, out_dir: str) -> None:
        self._start_reader_background_colorization(in_dir, out_dir)

    def _start_reader_background_colorization(self, input_folder: str, output_folder: str) -> None:
        selected_model_name = self.combo_active_model.currentText()
        backend = self.registry.get_backend(selected_model_name)
        if not backend:
            self.log_message(f"Error: model backend '{selected_model_name}' unavailable for live colorization.")
            return

        self.log_message(f"Launching background colorization on AMD DirectML for: {input_folder}")
        self.batch_worker = BatchProcessor(
            colorizer=backend,
            vlm_director=self.vlm_director,
            palette_manager=self.palette_manager,
            input_source=Path(input_folder),
            source_type="folder",
            output_dir=Path(output_folder),
            output_format="PNG",
            pipeline_mode="BALANCED",
            color_strength=0.65,
            l_preservation=0.95,
            line_protection=0.95,
            protect_bubbles=True,
            use_guided_filter=True,
            reprocess_existing=False,
            create_cbz_output=False,
            create_pdf_output=False,
            page_hints={},
        )

        self.batch_worker.page_completed.connect(self._on_page_completed)
        self.batch_worker.page_started.connect(self._on_page_started)
        self.batch_worker.page_failed.connect(self._on_page_failed)
        self.batch_worker.progress_updated.connect(self._on_progress_updated)
        self.batch_worker.status_message.connect(self.log_message)
        self.batch_worker.start()

    def log_message(self, text: str) -> None:
        self.log_view.append(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Gracefully terminate background batch worker thread
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop()
            self.batch_worker.wait(2000)

        # Gracefully terminate background downloader worker thread
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.stop()
            self.download_worker.wait(2000)

        # Stop benchmark workers if running
        if hasattr(self.benchmark_screen, "_worker") and self.benchmark_screen._worker and self.benchmark_screen._worker.isRunning():
            self.benchmark_screen._worker.terminate()
            self.benchmark_screen._worker.wait(1000)

        if hasattr(self.model_lab, "_worker") and self.model_lab._worker and self.model_lab._worker.isRunning():
            self.model_lab._worker.terminate()
            self.model_lab._worker.wait(1000)

        event.accept()
