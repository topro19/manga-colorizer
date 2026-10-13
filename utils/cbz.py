"""
Comic Book Archive (.cbz) utilities for extraction and repacking.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Union

from .image_utils import SUPPORTED_IMAGE_EXTENSIONS, get_image_files, natural_sort_key


def extract_cbz(cbz_path: str | Path, target_dir: str | Path) -> List[Path]:
    """
    Extract a .cbz (ZIP) file to target_dir.
    Returns naturally-sorted list of extracted image Paths.
    """
    cbz_file = Path(cbz_path)
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(cbz_file, "r") as archive:
        for member in archive.infolist():
            # Skip OS junk and directories
            filename = member.filename
            if member.is_dir() or "__MACOSX" in filename or filename.startswith("."):
                continue
            ext = Path(filename).suffix.lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                archive.extract(member, out_dir)

    return get_image_files(out_dir, recursive=True)


def create_cbz(
    images_or_directory: Union[List[Path], List[str], Path, str],
    output_cbz: str | Path,
    compression: int = zipfile.ZIP_DEFLATED,
) -> Path:
    """
    Package images into a new .cbz archive with natural page naming.
    """
    out_path = Path(output_cbz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(images_or_directory, (str, Path)) and Path(images_or_directory).is_dir():
        image_files = get_image_files(images_or_directory, recursive=False)
    else:
        image_files = [Path(p) for p in images_or_directory]
        image_files.sort(key=lambda p: natural_sort_key(p.name))

    with zipfile.ZipFile(out_path, "w", compression=compression) as archive:
        for img_path in image_files:
            if img_path.is_file():
                archive.write(img_path, arcname=img_path.name)

    return out_path
