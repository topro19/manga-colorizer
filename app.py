#!/usr/bin/env python3
"""
MangaColorizer — Desktop Application Entry Point.
Runs locally on Windows using AMD Radeon DirectML GPU acceleration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
    QProgressDialog,
)

from core.model_manager import ModelManager
from ui.main_window import MainWindow


def main() -> None:
    # High-DPI scaling configuration for crisp rendering
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("MangaColorizer")
    app.setOrganizationName("MangaColorizer")

    # Clean default system font
    app.setFont(QFont("Segoe UI", 10))

    # Initialize model manager
    model_mgr = ModelManager()

    # If model is not downloaded, prompt user and download it automatically
    if not model_mgr.is_model_downloaded():
        reply = QMessageBox.question(
            None,
            "Download Required Model",
            "MangaColorization-v2 FP16 ONNX model (~58.8 MB) was not found.\n\n"
            "Would you like to download it now from Hugging Face?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            sys.exit(0)

        progress_dlg = QProgressDialog("Downloading manga-colorize-fp16.onnx...", "Cancel", 0, 100)
        progress_dlg.setWindowTitle("Downloading Model")
        progress_dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress_dlg.setMinimumDuration(0)
        progress_dlg.setValue(0)

        def on_download_progress(downloaded: int, total: int, percent: float) -> None:
            progress_dlg.setValue(int(percent))
            mb_down = downloaded / (1024 * 1024)
            mb_tot = total / (1024 * 1024)
            progress_dlg.setLabelText(f"Downloading model: {mb_down:.1f} MB / {mb_tot:.1f} MB ({percent:.0f}%)")
            QApplication.processEvents()

        try:
            model_mgr.download_model(progress_callback=on_download_progress)
            progress_dlg.setValue(100)
        except Exception as e:
            QMessageBox.critical(
                None,
                "Download Error",
                f"Failed to download model from Hugging Face:\n{str(e)}\n\n"
                f"You can manually place manga-colorize-fp16.onnx inside:\n"
                f"{model_mgr.models_dir}",
            )
            sys.exit(1)

    # Initialize session with DirectML
    try:
        model_mgr.load_session(prefer_directml=True)
    except Exception as e:
        print(f"[Warning] Session initialization note: {e}")

    # Launch GUI (defaults to consumer chapter-level flow, or pass --studio for dev lab)
    if "--studio" in sys.argv:
        window = MainWindow(model_manager=model_mgr)
    else:
        from app.ui.main_window import ConsumerMainWindow
        window = ConsumerMainWindow()

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
