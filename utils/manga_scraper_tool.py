"""
Manga Scraper & PDF Exporter Tool.
Features:
- Scrapes manga chapters from URLs (MangaDot, AsuraScans, Mangadex, image URLs, etc.)
- Downloads and organizes chapter pages.
- Converts downloaded pages into high-resolution single or multi-chapter PDFs.
- Can be run as a standalone CLI or launched via the Desktop UI.
"""

from __future__ import annotations
import base64
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# Add project root to sys.path if run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
import requests
from bs4 import BeautifulSoup

from app.utils.image import natural_sort_key, is_image_file
from app.utils.logging import logger


class MangaScraperTool:
    """
    Universal Manga Chapter Scraper and PDF Converter.
    """

    def __init__(self, output_dir: str | Path = "downloads/manga"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Referer": "https://mangadot.net/"
        })
        self.driver = None

    def _ensure_browser(self):
        if self.driver is None:
            import undetected_chromedriver as uc
            options = uc.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            self.driver = uc.Chrome(options=options)

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def scrape_mangadot_chapter(
        self,
        chapter_url_or_id: str | int,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[str, List[Path]]:
        """
        Scrapes a MangaDot chapter (e.g. https://mangadot.net/chapter/37925).
        Returns (chapter_title, list_of_downloaded_image_paths).
        """
        match = re.search(r'chapter/(\d+)', str(chapter_url_or_id))
        chapter_id = int(match.group(1)) if match else int(chapter_url_or_id)

        self._ensure_browser()
        url = f"https://mangadot.net/chapter/{chapter_id}"
        logger.info(f"Navigating to {url}...")
        self.driver.get(url)

        # Allow Cloudflare turnstile clearance
        for _ in range(10):
            time.sleep(1)
            if "Just a moment" not in self.driver.title and len(self.driver.title) > 0:
                break

        # Fetch chapter image list from MangaDot internal API
        script = f"""
            const resp = await fetch('https://mangadot.net/api/chapters/{chapter_id}/images');
            return await resp.json();
        """
        data = self.driver.execute_script(script)
        images = data.get("images", [])
        manga_title = data.get("manga", {}).get("title", f"Manga_{chapter_id}")
        # Clean title for directory
        clean_title = re.sub(r'[\\/*?:"<>|]', "", manga_title).strip()
        ch_num = data.get("chapter", {}).get("chapter_number", chapter_id)

        chapter_folder = self.output_dir / clean_title / f"Chapter_{ch_num}"
        chapter_folder.mkdir(parents=True, exist_ok=True)

        downloaded_paths: List[Path] = []

        for idx, item in enumerate(images, start=1):
            raw_url = item.get("url", "")
            img_url = f"https://mangadot.net{raw_url}" if raw_url.startswith("/") else raw_url
            ext = ".png" if ".png" in img_url.lower() else ".jpg"
            out_file = chapter_folder / f"{idx:03d}{ext}"

            if out_file.exists() and out_file.stat().st_size > 2000:
                downloaded_paths.append(out_file)
                if progress_cb:
                    progress_cb(idx, len(images), str(out_file))
                continue

            # Fetch via authenticated browser session
            b64 = self.driver.execute_script(f"""
                const resp = await fetch('{img_url}');
                const blob = await resp.blob();
                return new Promise((resolve) => {{
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result.split(',')[1]);
                    reader.readAsDataURL(blob);
                }});
            """)

            with open(out_file, "wb") as f:
                f.write(base64.b64decode(b64))

            downloaded_paths.append(out_file)
            if progress_cb:
                progress_cb(idx, len(images), str(out_file))

        return f"{clean_title}_Chapter_{ch_num}", downloaded_paths

    def scrape_generic_web_chapter(
        self,
        chapter_url: str,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[str, List[Path]]:
        """
        Scrapes generic manga reader sites by extracting reader container images.
        """
        self._ensure_browser()
        self.driver.get(chapter_url)
        time.sleep(4)

        title = self.driver.title or "Manga_Chapter"
        clean_title = re.sub(r'[\\/*?:"<>|]', "", title).strip()[:50]
        chapter_folder = self.output_dir / clean_title
        chapter_folder.mkdir(parents=True, exist_ok=True)

        # Extract image source urls from page
        img_urls = self.driver.execute_script("""
            const imgs = Array.from(document.querySelectorAll('img'));
            return imgs
                .map(i => i.src || i.dataset.src || i.getAttribute('data-lazy-src'))
                .filter(src => src && (src.includes('.jpg') || src.includes('.png') || src.includes('.webp') || src.includes('chapter')));
        """)

        downloaded: List[Path] = []
        for idx, src in enumerate(img_urls, start=1):
            out_file = chapter_folder / f"{idx:03d}.jpg"
            try:
                r = self.session.get(src, timeout=10)
                if r.status_code == 200 and len(r.content) > 3000:
                    with open(out_file, "wb") as f:
                        f.write(r.content)
                    downloaded.append(out_file)
                    if progress_cb:
                        progress_cb(idx, len(img_urls), str(out_file))
            except Exception:
                continue

        return clean_title, downloaded

    @staticmethod
    def images_to_pdf(image_paths: List[str | Path], output_pdf: str | Path) -> Path:
        """
        Converts an ordered list of manga images to a crisp, high-resolution PDF document.
        """
        pdf_path = Path(output_pdf)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        sorted_paths = [Path(p) for p in image_paths if is_image_file(p)]
        sorted_paths.sort(key=lambda p: natural_sort_key(p.name))

        if not sorted_paths:
            raise ValueError("No images provided for PDF conversion.")

        pil_images = []
        first_img = None

        for p in sorted_paths:
            try:
                img = Image.open(p).convert("RGB")
                if first_img is None:
                    first_img = img
                else:
                    pil_images.append(img)
            except Exception as e:
                logger.warning(f"Could not open image {p}: {e}")

        if first_img is None:
            raise ValueError("Could not parse any valid images for PDF.")

        first_img.save(
            pdf_path,
            "PDF",
            resolution=150.0,
            save_all=True,
            append_images=pil_images
        )
        logger.info(f"Successfully compiled {len(sorted_paths)} pages to PDF: {pdf_path}")
        return pdf_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape manga chapters and compile them into high-resolution PDFs.")
    parser.add_argument("url", nargs="?", help="URL of the manga chapter to scrape")
    parser.add_argument("--output", "-o", default="scraped_manga", help="Output directory to save images and PDF")
    parser.add_argument("--no-pdf", action="store_true", help="Only download images without compiling into PDF")
    parser.add_argument("--pdf-only", help="Compile an existing folder of images into a PDF: --pdf-only <path_to_images_folder>")
    
    args = parser.parse_args()

    scraper = MangaScraperTool(output_dir=args.output)

    if args.pdf_only:
        folder = Path(args.pdf_only)
        if not folder.exists() or not folder.is_dir():
            print(f"Error: Folder '{folder}' does not exist.")
            sys.exit(1)
        images = [p for p in folder.iterdir() if is_image_file(p)]
        pdf_out = folder.parent / f"{folder.name}.pdf"
        print(f"Compiling {len(images)} images in '{folder.name}' to PDF...")
        res = scraper.images_to_pdf(images, pdf_out)
        print(f"[SUCCESS] PDF created: {res}")
        return

    if not args.url:
        parser.print_help()
        sys.exit(1)

    print(f"Scraping chapter from: {args.url}")
    if "mangadot.net" in args.url:
        title, downloaded = scraper.scrape_mangadot_chapter(
            args.url,
            progress_cb=lambda cur, tot, f: print(f"[{cur}/{tot}] Downloaded {Path(f).name}")
        )
    else:
        title, downloaded = scraper.scrape_generic_web_chapter(
            args.url,
            progress_cb=lambda cur, tot, f: print(f"[{cur}/{tot}] Downloaded {Path(f).name}")
        )

    if not downloaded:
        print("[ERROR] No images could be downloaded.")
        sys.exit(1)

    print(f"\n[SUCCESS] Downloaded {len(downloaded)} pages to {scraper.output_dir / title}")

    if not args.no_pdf:
        pdf_target = scraper.output_dir / f"{title}.pdf"
        print(f"Compiling into PDF: {pdf_target} ...")
        pdf_file = scraper.images_to_pdf(downloaded, pdf_target)
        print(f"[SUCCESS] High-resolution PDF generated: {pdf_file}")


if __name__ == "__main__":
    main()
