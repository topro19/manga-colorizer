"""
Utility functions for image handling, sorting, color space transformations,
and metadata persistence.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_sort_key(s: str) -> List[Any]:
    """Sort strings with embedded numbers naturally (e.g. 1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def get_image_files(directory: str | Path, recursive: bool = False) -> List[Path]:
    """Return naturally-sorted list of valid image files in directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []

    files: List[Path] = []
    pattern = "**/*" if recursive else "*"
    for item in dir_path.glob(pattern):
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            if not item.name.startswith("._") and not item.name.startswith("."):
                files.append(item)

    files.sort(key=lambda p: natural_sort_key(p.name))
    return files


def load_image_rgb(path_or_image: str | Path | Image.Image) -> Image.Image:
    """Load an image, orient it via EXIF if needed, and convert to RGB."""
    if isinstance(path_or_image, Image.Image):
        img = path_or_image
    else:
        img = Image.open(path_or_image)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


def load_image_gray(path_or_image: str | Path | Image.Image) -> Image.Image:
    """Load an image and convert to grayscale ('L')."""
    rgb = load_image_rgb(path_or_image)
    return rgb.convert("L")


def apply_luminance_preservation(
    original_gray: Image.Image,
    colorized_rgb: Image.Image,
) -> Image.Image:
    """
    Preserve crisp original line art and screentones by replacing the
    Y (luminance) channel of the upscaled colorized image with the
    full-resolution original grayscale image in YCbCr color space.
    """
    target_size = original_gray.size  # (W, H)
    if colorized_rgb.size != target_size:
        # Resize colorized chrominance smoothly with Lanczos
        resized_color = colorized_rgb.resize(target_size, Image.Resampling.LANCZOS)
    else:
        resized_color = colorized_rgb

    # Convert colorized to YCbCr
    ycbcr = resized_color.convert("YCbCr")
    y, cb, cr = ycbcr.split()

    # Recombine with original crisp Y channel
    blended_ycbcr = Image.merge("YCbCr", (original_gray, cb, cr))
    return blended_ycbcr.convert("RGB")


def read_metadata(output_dir: str | Path) -> Dict[str, Any]:
    """Read processing metadata JSON file if it exists."""
    meta_path = Path(output_dir) / "_colorization_metadata.json"
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def write_metadata(output_dir: str | Path, data: Dict[str, Any]) -> None:
    """Save processing metadata JSON file safely."""
    meta_path = Path(output_dir) / "_colorization_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = meta_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.replace(meta_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
