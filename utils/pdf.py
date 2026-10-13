"""
PDF utilities for extracting pages as high-resolution images and exporting colorized pages back to PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image

from .image_utils import natural_sort_key


def extract_pdf_pages(
    pdf_path: str | Path,
    target_dir: str | Path,
    dpi: int = 200,
    prefix: str = "page",
) -> List[Path]:
    """
    Extract each page of a PDF file to a high-resolution PNG image using pypdfium2.
    Returns list of extracted image paths.
    """
    import pypdfium2 as pdfium

    pdf_file = Path(pdf_path)
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(pdf_file)
    extracted_paths: List[Path] = []
    try:
        total_pages = len(pdf)
        scale = dpi / 72.0

        for idx in range(total_pages):
            page = pdf[idx]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()

            filename = f"{prefix}_{idx + 1:04d}.png"
            save_path = out_dir / filename
            pil_image.save(save_path, format="PNG")
            extracted_paths.append(save_path)
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    return extracted_paths


def create_pdf(
    image_paths: List[Path | str],
    output_pdf: str | Path,
) -> Path:
    """
    Compile multiple images into a single multi-page PDF document using Pillow.
    """
    out_path = Path(output_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_paths = [Path(p) for p in image_paths]
    sorted_paths.sort(key=lambda p: natural_sort_key(p.name))

    if not sorted_paths:
        raise ValueError("No images provided to create PDF")

    first_image = Image.open(sorted_paths[0]).convert("RGB")
    remaining_images = []

    for p in sorted_paths[1:]:
        img = Image.open(p).convert("RGB")
        remaining_images.append(img)

    first_image.save(
        out_path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=remaining_images,
    )

    return out_path
