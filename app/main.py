"""
Manga Colorizer Studio Application Entrypoint.
"""

import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication

# Ensure scratch/MangaColorizer is on python path
sys.path.insert(0, str(Path(__file__).parent))

from app.ui.main_window import ConsumerMainWindow
from app.utils.logging import logger

def main():
    logger.info("Starting Manga Colorizer Studio...")
    app = QApplication(sys.argv)
    window = ConsumerMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
