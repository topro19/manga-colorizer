"""
Manga Scraper and PDF Converter Dialog Window.
Integrated into MangaColorizer Studio or runnable as standalone utility.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QCheckBox, QApplication
)

from utils.manga_scraper_tool import MangaScraperTool
from ui.styles import DARK_THEME_QSS


class ScraperWorker(QThread):
    progress = Signal(int, int, str)
    status = Signal(str)
    finished = Signal(str, str)  # chapter_title, pdf_path
    error = Signal(str)

    def __init__(self, url: str, output_dir: str, create_pdf: bool = True):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self.create_pdf = create_pdf
        self.tool = MangaScraperTool(output_dir=self.output_dir)

    def run(self):
        try:
            self.status.emit("Connecting to manga source...")
            if "mangadot.net" in self.url:
                title, images = self.tool.scrape_mangadot_chapter(
                    self.url,
                    progress_cb=lambda cur, tot, f: self.progress.emit(cur, tot, f)
                )
            else:
                title, images = self.tool.scrape_generic_web_chapter(
                    self.url,
                    progress_cb=lambda cur, tot, f: self.progress.emit(cur, tot, f)
                )

            if not images:
                self.error.emit("No images could be scraped from this chapter.")
                return

            pdf_path_str = ""
            if self.create_pdf:
                self.status.emit(f"Converting {len(images)} pages into PDF...")
                pdf_file = Path(self.output_dir) / f"{title}.pdf"
                pdf_path = self.tool.images_to_pdf(images, pdf_file)
                pdf_path_str = str(pdf_path)

            self.finished.emit(title, pdf_path_str)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.tool.close()


class MangaScraperDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manga Chapter Scraper & PDF Exporter")
        self.resize(680, 480)
        self.setStyleSheet(DARK_THEME_QSS)
        self.worker: Optional[ScraperWorker] = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title_lbl = QLabel("Manga Chapter Scraper & PDF Converter")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title_lbl)

        # URL Input
        url_lbl = QLabel("Manga Chapter URL:")
        url_lbl.setStyleSheet("font-weight: 600; color: #a5b4fc;")
        layout.addWidget(url_lbl)

        self.edit_url = QLineEdit("https://mangadot.net/chapter/37925")
        self.edit_url.setPlaceholderText("Paste manga chapter URL (e.g. https://mangadot.net/chapter/37925)...")
        self.edit_url.setStyleSheet("padding: 8px; border-radius: 4px; background-color: #1e2230; color: #ffffff;")
        layout.addWidget(self.edit_url)

        # Output Folder
        out_lbl = QLabel("Output Folder:")
        out_lbl.setStyleSheet("font-weight: 600; color: #a5b4fc;")
        layout.addWidget(out_lbl)

        out_row = QHBoxLayout()
        self.edit_output = QLineEdit(str(Path("downloads/manga").absolute()))
        self.edit_output.setStyleSheet("padding: 8px; border-radius: 4px; background-color: #1e2230; color: #ffffff;")
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse_output)
        out_row.addWidget(self.edit_output)
        out_row.addWidget(self.btn_browse)
        layout.addLayout(out_row)

        self.chk_create_pdf = QCheckBox("Automatically convert scraped chapter into single PDF")
        self.chk_create_pdf.setChecked(True)
        layout.addWidget(self.chk_create_pdf)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        layout.addWidget(self.progress_bar)

        # Log box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #0f111a; color: #94a3b8; font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_box, 1)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("🚀 Scrape & Convert to PDF")
        self.btn_start.setStyleSheet("background-color: #4f46e5; color: #ffffff; padding: 10px 20px; font-weight: bold; border-radius: 6px;")
        self.btn_start.clicked.connect(self._on_start)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_start)
        layout.addLayout(btn_row)

    def _on_browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.edit_output.setText(d)

    def _on_start(self):
        url = self.edit_url.text().strip()
        out_dir = self.edit_output.text().strip()
        if not url:
            QMessageBox.warning(self, "Missing URL", "Please enter a manga chapter URL.")
            return

        self.btn_start.setEnabled(False)
        self.log_box.append(f"Starting scrape for: {url}")
        self.progress_bar.setValue(0)

        self.worker = ScraperWorker(url=url, output_dir=out_dir, create_pdf=self.chk_create_pdf.isChecked())
        self.worker.status.connect(lambda s: self.log_box.append(f"[*] {s}"))
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, cur: int, tot: int, path: str):
        pct = int((cur / max(1, tot)) * 100)
        self.progress_bar.setValue(pct)
        self.log_box.append(f"[+] Downloaded page {cur}/{tot}: {Path(path).name}")

    def _on_finished(self, title: str, pdf_path: str):
        self.btn_start.setEnabled(True)
        self.progress_bar.setValue(100)
        self.result_pdf = Path(pdf_path) if pdf_path else None
        self.result_dir = Path(self.edit_output.text().strip()) / title
        msg = f"Chapter '{title}' successfully scraped!"
        if pdf_path:
            msg += f"\nPDF Created: {pdf_path}"
        self.log_box.append(f"[SUCCESS] {msg}")
        QMessageBox.information(self, "Scrape Complete", msg)
        self.accept()

    def _on_error(self, err: str):
        self.btn_start.setEnabled(True)
        self.log_box.append(f"[ERROR] {err}")
        QMessageBox.critical(self, "Scrape Failed", err)


def main():
    app = QApplication(sys.argv)
    dlg = MangaScraperDialog()
    dlg.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
