import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import undetected_chromedriver as uc
from app.models.local_onnx import LocalMangaV2Backend
from utils.manga_scraper_tool import MangaScraperTool

DEST_DIR = Path.home() / "Downloads" / "dorohedoro"
DEST_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("downloads/cache_chapters")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== DOROHEDORO CHAPTER 100+ BATCH RUNNER ===")
print(f"Target Output Folder: {DEST_DIR}")

# 1. Initialize DirectML Balanced Backend
backend = LocalMangaV2Backend(Path("models/manga-colorize-fp16.onnx"), provider="DmlExecutionProvider")
backend.load()

# 2. Launch browser for Cloudflare-cleared downloads
options = uc.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
driver = uc.Chrome(options=options)

try:
    print("Initializing MangaDot authenticated session...")
    driver.get("https://mangadot.net/chapter/37926")  # Chapter 100
    for _ in range(10):
        time.sleep(1)
        if "Just a moment" not in driver.title and len(driver.title) > 0:
            break

    cur_id = 37926  # Chapter 100

    while cur_id:
        print(f"\n---> Fetching chapter metadata for ID: {cur_id} ...")
        res = driver.execute_script("""
            const cid = arguments[0];
            try {
                const resp = await fetch('https://mangadot.net/api/chapters/' + cid + '/images');
                if (!resp.ok) return { error: 'HTTP ' + resp.status };
                return await resp.json();
            } catch(e) {
                return { error: e.toString() };
            }
        """, cur_id)

        if not res or "error" in res:
            print(f"Reached end of chapters or error: {res}")
            break

        ch_num = res.get("chapter", {}).get("chapter_number", str(cur_id))
        ch_title = res.get("chapter", {}).get("chapter_title", f"Chapter {ch_num}")
        images_list = res.get("images", [])
        next_id = res.get("next_chapter_id")

        print(f"==> Processing Dorohedoro Chapter {ch_num} ('{ch_title}') [{len(images_list)} pages]")

        # Check if PDF already exists in destination
        target_pdf = DEST_DIR / f"Dorohedoro_Chapter_{ch_num}.pdf"
        if target_pdf.exists() and target_pdf.stat().st_size > 50000:
            print(f"Chapter {ch_num} already exists at {target_pdf.name}. Skipping to next...")
            cur_id = next_id
            continue

        # Download raw pages
        ch_cache = CACHE_DIR / f"Chapter_{ch_num}"
        ch_cache.mkdir(parents=True, exist_ok=True)
        downloaded_raw_paths = []

        for idx, item in enumerate(images_list, start=1):
            raw_url = item.get("url", "")
            img_url = f"https://mangadot.net{raw_url}" if raw_url.startswith("/") else raw_url
            ext = ".png" if ".png" in img_url.lower() else ".jpg"
            out_raw = ch_cache / f"{idx:03d}{ext}"

            if not (out_raw.exists() and out_raw.stat().st_size > 2000):
                # Download via browser
                b64 = driver.execute_script("""
                    const url = arguments[0];
                    try {
                        const r = await fetch(url);
                        const b = await r.blob();
                        return new Promise((resolve) => {
                            const reader = new FileReader();
                            reader.onloadend = () => resolve(reader.result.split(',')[1]);
                            reader.readAsDataURL(b);
                        });
                    } catch(e) { return null; }
                """, img_url)

                if b64:
                    import base64
                    with open(out_raw, "wb") as f:
                        f.write(base64.b64decode(b64))

            if out_raw.exists():
                downloaded_raw_paths.append(out_raw)

        if not downloaded_raw_paths:
            print(f"Warning: No images downloaded for chapter {ch_num}.")
            cur_id = next_id
            continue

        # Colorize pages using BALANCED MODE
        colorized_pages = []
        color_cache = ch_cache / "colorized"
        color_cache.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        for idx, raw_p in enumerate(downloaded_raw_paths, start=1):
            out_c = color_cache / f"{idx:03d}_color.png"
            if not (out_c.exists() and out_c.stat().st_size > 5000):
                bgr = cv2.imread(str(raw_p))
                if bgr is not None:
                    # Balanced mode settings: natural anime colors, speech bubble protection, guided contours
                    colored = backend.colorize(
                        bgr,
                        color_strength=0.85,
                        line_preservation=0.90,
                        speech_bubble_protection=True,
                        use_guided_filter=True,
                        options={"vivid_mode": False}
                    )
                    cv2.imwrite(str(out_c), colored)
            if out_c.exists():
                colorized_pages.append(out_c)

        # Compile into PDF in user's Downloads/dorohedoro
        print(f"Compiling Chapter {ch_num} into PDF: {target_pdf.name} ...")
        MangaScraperTool.images_to_pdf(colorized_pages, target_pdf)
        elapsed = time.time() - t0
        print(f"[COMPLETED] Chapter {ch_num} saved to {target_pdf} ({target_pdf.stat().st_size / (1024*1024):.2f} MB in {elapsed:.1f}s)")

        cur_id = next_id

finally:
    driver.quit()

print(f"\nAll chapters processed and saved to {DEST_DIR}!")
