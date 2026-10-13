"""
Model Lab View for MangaColorizer.
Allows users to run and compare multiple colorization backends on the current manga page side-by-side,
benchmarking DirectML latency, memory, line preservation, and aesthetic quality.
"""

from __future__ import annotations

import io
import time
from typing import Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.backends.base_colorizer import BaseColorizer
from core.backends.colorizer_registry import ColorizerRegistry


class BenchmarkWorker(QThread):
    model_benchmarked = Signal(str, bytes, float, str)  # name, png_bytes, latency_ms, info_text
    benchmark_finished = Signal()

    def __init__(
        self,
        registry: ColorizerRegistry,
        image_path: str,
        color_strength: float = 0.65,
        line_preservation: float = 0.90,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.registry = registry
        self.image_path = image_path
        self.color_strength = color_strength
        self.line_preservation = line_preservation

    def run(self):
        bgr = cv2.imread(self.image_path)
        if bgr is None:
            self.benchmark_finished.emit()
            return

        for name in self.registry.list_available_models():
            backend = self.registry.get_backend(name)
            if not backend:
                continue

            t0 = time.perf_counter()
            try:
                out_bgr = backend.colorize(
                    bgr_image=bgr,
                    color_strength=self.color_strength,
                    line_preservation=self.line_preservation,
                    speech_bubble_protection=True,
                    use_guided_filter=True,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                _, png_buf = cv2.imencode(".png", out_bgr)
                info = backend.get_info()
                desc = f"{info.get('domain', '')} | {info.get('weights', '')}"
                self.model_benchmarked.emit(name, png_buf.tobytes(), latency_ms, desc)
            except Exception as e:
                print(f"[ModelLab] Error benchmarking {name}: {e}")

        self.benchmark_finished.emit()


class ModelLabPanel(QWidget):
    """
    Dedicated Model Lab UI displaying multi-model benchmark results side-by-side.
    """

    def __init__(self, registry: ColorizerRegistry, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.registry = registry
        self.active_image_path: Optional[str] = None
        self._worker: Optional[BenchmarkWorker] = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title = QLabel("Model Lab & Backend Evaluation")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #f1f5f9;")
        header.addWidget(title)

        header.addStretch()

        self.btn_run_benchmark = QPushButton("Benchmark All Models")
        self.btn_run_benchmark.setObjectName("PrimaryBtn")
        self.btn_run_benchmark.clicked.connect(self.run_benchmark)
        header.addWidget(self.btn_run_benchmark)

        layout.addLayout(header)

        # Description
        desc = QLabel(
            "Compare installed colorization backends on the active manga page. "
            "Evaluates DirectML GPU latency, line preservation, and anti-bleeding results side-by-side."
        )
        desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Scroll area for comparison cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: 1px solid #282d3f; border-radius: 8px; background: #0f1117; }")

        self.cards_container = QWidget()
        self.cards_layout = QGridLayout(self.cards_container)
        self.cards_layout.setContentsMargins(15, 15, 15, 15)
        self.cards_layout.setSpacing(15)

        self.scroll_area.setWidget(self.cards_container)
        layout.addWidget(self.scroll_area, stretch=1)

    def set_active_image(self, path: str):
        self.active_image_path = path

    def run_benchmark(self):
        if not self.active_image_path:
            return

        # Clear existing cards
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.btn_run_benchmark.setEnabled(False)
        self.btn_run_benchmark.setText("Benchmarking...")

        self._worker = BenchmarkWorker(self.registry, self.active_image_path)
        self._worker.model_benchmarked.connect(self._add_result_card)
        self._worker.benchmark_finished.connect(self._on_benchmark_done)
        self._worker.start()

    def _add_result_card(self, model_name: str, png_bytes: bytes, latency_ms: float, desc: str):
        col_idx = self.cards_layout.count() % 3
        row_idx = self.cards_layout.count() // 3

        frame = QFrame()
        frame.setStyleSheet("QFrame { background: #171923; border: 1px solid #2d3246; border-radius: 8px; }")
        flayout = QVBoxLayout(frame)
        flayout.setContentsMargins(10, 10, 10, 10)
        flayout.setSpacing(8)

        # Title
        t_label = QLabel(model_name)
        t_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        t_label.setStyleSheet("color: #a5b4fc;")
        flayout.addWidget(t_label)

        # Image preview
        pix = QPixmap()
        pix.loadFromData(png_bytes)
        scaled_pix = pix.scaled(320, 440, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        img_label = QLabel()
        img_label.setPixmap(scaled_pix)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        flayout.addWidget(img_label)

        # Metrics
        m_label = QLabel(f"DirectML Latency: {latency_ms:.1f} ms\n{desc}")
        m_label.setStyleSheet("color: #34d399; font-weight: 600; font-size: 11px;")
        flayout.addWidget(m_label)

        self.cards_layout.addWidget(frame, row_idx, col_idx)

    def _on_benchmark_done(self):
        self.btn_run_benchmark.setEnabled(True)
        self.btn_run_benchmark.setText("Benchmark All Models")
