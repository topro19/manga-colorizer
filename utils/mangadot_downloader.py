"""
MangaDot Downloader & Scraper Utility.
Handles:
- Extracting manga title, ID, and chapter list from MangaDot URLs (e.g. https://mangadot.net/chapter/37925)
- Filtering by target chapter range (e.g. Chapter 95 to 100)
- Downloading chapter pages via authenticated browser session
- Saving organized page files into input/manga_downloads/<manga_name>/chapter_<num>/
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional
from PySide6.QtCore import QObject, QThread, Signal


class MangaDotChapterInfo:
    def __init__(self, chapter_id: int, chapter_num: float, title: str, page_count: int):
        self.chapter_id = chapter_id
        self.chapter_num = chapter_num
        self.title = title
        self.page_count = page_count


class MangaDotScraper:
    def __init__(self):
        self.driver = None

    def start_session(self, target_url: str = "https://mangadot.net"):
        import undetected_chromedriver as uc
        options = uc.ChromeOptions()
        # Non-headless solves Cloudflare Turnstile instantly
        self.driver = uc.Chrome(options=options)
        self.driver.get(target_url)
        # Wait up to 10s for Cloudflare check to clear
        for _ in range(10):
            time.sleep(1)
            if "Just a moment" not in self.driver.title and len(self.driver.title) > 0:
                break

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def get_manga_and_chapters(self, chapter_url_or_id: str) -> Dict[str, Any]:
        """
        Takes URL like https://mangadot.net/chapter/37925 or ID 37925.
        Returns dict with: manga_id, manga_title, current_chapter, chapters_list
        """
        match = re.search(r'chapter/(\d+)', str(chapter_url_or_id))
        chapter_id = int(match.group(1)) if match else int(chapter_url_or_id)

        if not self.driver:
            self.start_session(f"https://mangadot.net/chapter/{chapter_id}")
        else:
            self.driver.get(f"https://mangadot.net/chapter/{chapter_id}")
            time.sleep(3)

        script = f"""
            return (async () => {{
                const r_ch = await fetch('https://mangadot.net/api/chapters/{chapter_id}/images');
                const chData = await r_ch.json();
                
                const mangaId = chData.chapter ? chData.chapter.manga_id : 214;
                const r_list = await fetch(`https://mangadot.net/api/manga/${{mangaId}}/chapters/list`);
                const listData = await r_list.json();
                
                return {{
                    manga_id: mangaId,
                    manga_title: (chData.manga && chData.manga.title) ? chData.manga.title : "Manga_" + mangaId,
                    current_chapter: chData.chapter,
                    chapters: listData
                }};
            }})();
        """
        data = self.driver.execute_script(script)
        return data

    def download_chapter_pages(
        self,
        chapter_id: int,
        output_dir: str,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> List[str]:
        """
        Fetches all images for chapter_id and writes them to output_dir.
        Returns list of downloaded file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        img_info = self.driver.execute_script(f"""
            const resp = await fetch('https://mangadot.net/api/chapters/{chapter_id}/images');
            return await resp.json();
        """)

        images = img_info.get("images", [])
        downloaded_paths = []

        for idx, item in enumerate(images):
            raw_url = item.get("url")
            full_url = f"https://mangadot.net{raw_url}" if raw_url.startswith("/") else raw_url
            page_num = idx + 1
            filename = f"{page_num:03d}.webp"
            out_file = os.path.join(output_dir, filename)

            if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
                downloaded_paths.append(out_file)
                if progress_cb:
                    progress_cb(page_num, len(images), out_file)
                continue

            # Fetch via browser context to inherit Cloudflare clearance cookies
            b64 = self.driver.execute_script(f"""
                const resp = await fetch('{full_url}');
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
                progress_cb(page_num, len(images), out_file)

        return downloaded_paths


class MangaDotDownloadWorker(QThread):
    manga_info_loaded = Signal(str, list)  # manga_title, list_of_chapters
    page_downloaded = Signal(int, int, str, str)  # current, total, chapter_label, file_path
    chapter_finished = Signal(str, str)  # chapter_label, folder_path
    all_downloads_completed = Signal(list)  # list of folder paths
    error_occurred = Signal(str)

    def __init__(
        self,
        target_url: str,
        start_chapter: float,
        end_chapter: float,
        base_output_dir: str,
        parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self.target_url = target_url
        self.start_chapter = start_chapter
        self.end_chapter = end_chapter
        self.base_output_dir = base_output_dir
        self.scraper = MangaDotScraper()
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True
        try:
            if self.scraper:
                self.scraper.close()
        except Exception:
            pass

    def run(self):
        try:
            data = self.scraper.get_manga_and_chapters(self.target_url)
            manga_title = data.get("manga_title", "Manga")
            all_chapters = data.get("chapters", [])
            self.manga_info_loaded.emit(manga_title, all_chapters)

            # Filter chapters by range
            matched_chapters = []
            for ch in all_chapters:
                num = float(ch.get("chapter_number", 0.0))
                if self.start_chapter <= num <= self.end_chapter:
                    matched_chapters.append(ch)

            matched_chapters.sort(key=lambda c: float(c.get("chapter_number", 0.0)))
            clean_title = re.sub(r'[\/:*?"<>|]', '_', manga_title)
            completed_folders = []

            for ch in matched_chapters:
                if self._is_stopped:
                    break

                ch_num = ch.get("chapter_number")
                ch_id = ch.get("id")
                ch_folder = os.path.join(self.base_output_dir, clean_title, f"Chapter_{ch_num}")
                os.makedirs(ch_folder, exist_ok=True)

                def cb(cur, tot, fpath):
                    self.page_downloaded.emit(cur, tot, f"Chapter {ch_num}", fpath)

                self.scraper.download_chapter_pages(ch_id, ch_folder, progress_cb=cb)
                completed_folders.append(ch_folder)
                self.chapter_finished.emit(f"Chapter {ch_num}", ch_folder)

            self.all_downloads_completed.emit(completed_folders)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.scraper.close()
