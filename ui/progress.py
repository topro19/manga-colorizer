"""
Progress tracking and statistical telemetry panel for MangaColorizer Studio.
Displays:
- Current page / total pages & percentage
- Processing speed (pages/min)
- Estimated time remaining (ETA)
- Total elapsed time
- Per-stage latency breakdown (VLM, GPU Inference, Postprocessing)
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class ProgressPanel(QFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ProgressPanel")
        self.setStyleSheet("""
            QFrame#ProgressPanel {
                background-color: #151824;
                border: 1px solid #282d3f;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        # Main Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")

        # Stat Labels
        self.lbl_page_count = QLabel("Page 0 / 0")
        self.lbl_page_count.setStyleSheet("font-weight: 700; color: #f8fafc; font-size: 13px;")

        self.lbl_speed = QLabel("Speed: -- pages/min")
        self.lbl_speed.setStyleSheet("color: #a5b4fc;")

        self.lbl_eta = QLabel("ETA: --")
        self.lbl_eta.setStyleSheet("color: #34d399; font-weight: 600;")

        self.lbl_elapsed = QLabel("Elapsed: 00:00:00")
        self.lbl_elapsed.setStyleSheet("color: #94a3b8;")

        self.lbl_stages = QLabel("VLM: -- ms | GPU: -- ms | Post: -- ms")
        self.lbl_stages.setStyleSheet("color: #cbd5e1; font-family: monospace; font-size: 11px;")

        # Status / Error banner
        self.lbl_status = QLabel("Ready to colorize.")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-style: italic;")

        # Layouts
        top_row = QHBoxLayout()
        top_row.addWidget(self.lbl_page_count)
        top_row.addSpacing(16)
        top_row.addWidget(self.lbl_speed)
        top_row.addSpacing(16)
        top_row.addWidget(self.lbl_eta)
        top_row.addSpacing(16)
        top_row.addWidget(self.lbl_elapsed)
        top_row.addSpacing(16)
        top_row.addWidget(self.lbl_stages)
        top_row.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addLayout(top_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_status)

    def update_progress(
        self,
        current: int,
        total: int,
        percent: float,
        pages_per_min: float,
        eta_str: str,
        elapsed_str: str,
        stage_info: str = "",
    ) -> None:
        self.progress_bar.setValue(int(round(percent)))
        self.lbl_page_count.setText(f"Page {current} / {total} ({percent:.1f}%)")
        self.lbl_speed.setText(f"Speed: {pages_per_min:.1f} pages/min")
        self.lbl_eta.setText(f"ETA: {eta_str}")
        self.lbl_elapsed.setText(f"Elapsed: {elapsed_str}")
        if stage_info:
            self.lbl_stages.setText(stage_info)

    def update_latency_stages(self, timings_ms: dict) -> None:
        vlm_ms = timings_ms.get("vlm_plan_ms", 0.0)
        gpu_ms = timings_ms.get("inference_ms", 0.0)
        post_ms = timings_ms.get("postprocessing_ms", 0.0)
        tot_ms = timings_ms.get("total_ms", 0.0)
        self.lbl_stages.setText(f"VLM: {vlm_ms:.0f}ms | GPU: {gpu_ms:.0f}ms | Post: {post_ms:.0f}ms (Total: {tot_ms:.0f}ms)")

    def set_status(self, text: str, is_error: bool = False) -> None:
        color = "#f87171" if is_error else "#94a3b8"
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color};")

    def reset(self) -> None:
        self.progress_bar.setValue(0)
        self.lbl_page_count.setText("Page 0 / 0")
        self.lbl_speed.setText("Speed: -- pages/min")
        self.lbl_eta.setText("ETA: --")
        self.lbl_elapsed.setText("Elapsed: 00:00:00")
        self.lbl_stages.setText("VLM: -- ms | GPU: -- ms | Post: -- ms")
        self.lbl_status.setText("Ready to colorize.")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-style: italic;")
