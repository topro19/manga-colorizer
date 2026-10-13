"""
Image utility routines for high-res colorization and color space conversions.
"""

from typing import List, Tuple
from pathlib import Path
import cv2
import numpy as np

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VALID_IMAGE_EXTS

def natural_sort_key(s: str | Path) -> List[int | str]:
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def load_image_bgr(path: str | Path) -> np.ndarray:
    path_str = str(path)
    img = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback for unicode paths on Windows
        with open(path_str, "rb") as f:
            bytes_data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image from {path_str}")
    return img

def save_image_bgr(path: str | Path, img: np.ndarray, quality: int = 95) -> bool:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    else:
        params = []
    return cv2.imwrite(str(path), img, params)
