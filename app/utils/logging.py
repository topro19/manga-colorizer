"""
Structured logging module for MangaColorizer Studio.
"""

import logging
import os
from pathlib import Path

def setup_logger(name: str = "MangaColorizer", log_file: str = "logs/colorizer.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    Path("logs").mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Minimal console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()
