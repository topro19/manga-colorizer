"""
Chapter ingestion module supporting Folders, CBZ archives, and PDF documents.
Always extracts to a working cache without mutating user sources.
"""

from __future__ import annotations
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional
from app.utils.image import is_image_file, natural_sort_key
from app.utils.logging import logger

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class ChapterExtractor:
    """
    Extracts pages from directories, CBZ files, or PDF documents into a clean chapter directory.
    """

    @staticmethod
    def extract_source(source_path: str | Path, target_dir: str | Path) -> List[Path]:
        source = Path(source_path)
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        if not source.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")

        if source.is_dir():
            return ChapterExtractor._copy_images_from_folder(source, target)
        elif source.suffix.lower() == ".cbz" or (source.is_file() and zipfile.is_zipfile(str(source))):
            return ChapterExtractor._extract_cbz(source, target)
        elif source.suffix.lower() == ".pdf":
            return ChapterExtractor._extract_pdf(source, target)
        else:
            raise ValueError(f"Unsupported manga input source: {source}")

    @staticmethod
    def _copy_images_from_folder(source_dir: Path, target_dir: Path) -> List[Path]:
        files = [p for p in source_dir.iterdir() if p.is_file() and is_image_file(p)]
        files.sort(key=natural_sort_key)
        
        extracted_paths: List[Path] = []
        for idx, f in enumerate(files, start=1):
            dst = target_dir / f"{idx:04d}_{f.name}"
            shutil.copy2(f, dst)
            extracted_paths.append(dst)
        logger.info(f"Ingested {len(extracted_paths)} pages from directory {source_dir}")
        return extracted_paths

    @staticmethod
    def _extract_cbz(cbz_path: Path, target_dir: Path) -> List[Path]:
        with zipfile.ZipFile(cbz_path, 'r') as zf:
            namelist = [name for name in zf.namelist() if is_image_file(name)]
            namelist.sort(key=natural_sort_key)
            extracted_paths: List[Path] = []
            for idx, member in enumerate(namelist, start=1):
                clean_name = Path(member).name
                dst = target_dir / f"{idx:04d}_{clean_name}"
                with zf.open(member) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
                extracted_paths.append(dst)
        logger.info(f"Extracted {len(extracted_paths)} pages from CBZ {cbz_path.name}")
        return extracted_paths

    @staticmethod
    def _extract_pdf(pdf_path: Path, target_dir: Path) -> List[Path]:
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF rendering.")
        doc = fitz.open(str(pdf_path))
        extracted_paths: List[Path] = []
        for idx, page in enumerate(doc, start=1):
            # Render at 300 DPI for high fidelity line-art extraction
            pix = page.get_pixmap(dpi=300)
            dst = target_dir / f"{idx:04d}_page_{idx}.png"
            pix.save(str(dst))
            extracted_paths.append(dst)
        doc.close()
        logger.info(f"Rendered {len(extracted_paths)} pages from PDF {pdf_path.name}")
        return extracted_paths
