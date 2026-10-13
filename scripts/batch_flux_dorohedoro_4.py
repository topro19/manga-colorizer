import os
import sys
import re
import time
import subprocess
import cv2
import numpy as np
from PIL import Image
import pypdfium2 as pdfium

SRC_DIR = r"C:\Users\ginto\Downloads\dorohedoro"
OUT_DIR = r"C:\Users\ginto\Downloads\dorohedoro 4"
CLI_EXE = r"tools\sd_cpp\sd-cli.exe"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("scratch_flux", exist_ok=True)

# FLUX Model Paths
DIFF_MODEL = "models/flux_gguf/flux1-schnell-Q4_0.gguf"
CLIP_L = "models/flux_gguf/clip_l.safetensors"
T5XXL = "models/flux_gguf/t5-v1_1-xxl-encoder-Q3_K_M.gguf"
VAE = "models/flux_gguf/ae.safetensors"
LORA_DIR = "models/flux_manga_lora"

PROMPT = "clean anime manga colorization, dorohedoro characters, nikaido blonde hair, fair smooth skin, blush, toned anatomy, caiman lizard scales, shin suit, noi, en, vibrant colors <lora:anime_schnell:0.8>"

def run_flux_page(input_bgr: np.ndarray, page_idx: int) -> np.ndarray:
    """Colorize a manga page using FLUX Schnell + Anime LoRA + Ink Restoration."""
    h_orig, w_orig = input_bgr.shape[:2]
    gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)

    # Calculate optimal 16/32 multiple dimension for FLUX
    target_w = 512
    target_h = int(round(h_orig * (target_w / w_orig) / 16) * 16)
    
    init_path = f"scratch_flux/init_p{page_idx}.png"
    out_path = f"scratch_flux/out_p{page_idx}.png"

    # Save resized input for FLUX
    im_pil = Image.fromarray(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))
    im_pil.resize((target_w, target_h), Image.Resampling.LANCZOS).save(init_path)

    # Execute native FLUX Schnell + Anime LoRA via sd-cli
    cmd = [
        CLI_EXE,
        "--diffusion-model", DIFF_MODEL,
        "--clip_l", CLIP_L,
        "--t5xxl", T5XXL,
        "--vae", VAE,
        "--lora-model-dir", LORA_DIR,
        "--init-img", init_path,
        "--strength", "0.15",
        "--prompt", PROMPT,
        "-W", str(target_w),
        "-H", str(target_h),
        "--steps", "4",
        "--cfg-scale", "1.0",
        "-o", out_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_path):
        print(f"[WARN] FLUX failed for page {page_idx}, falling back to grayscale.")
        return input_bgr

    # Load FLUX output and upscale back to original page resolution
    flux_raw = cv2.imread(out_path)
    flux_fit = cv2.resize(flux_raw, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)

    # 1. Optical Multiply underneath original drawing
    color_f = flux_fit.astype(np.float32) / 255.0
    raw_f = input_bgr.astype(np.float32) / 255.0
    mult_bgr = (color_f * raw_f * 255.0).astype(np.uint8)

    # 2. Strict Original Black Ink Restoration
    ink_mask = (gray < 85).astype(np.float32)[:, :, None]
    final_bgr = (mult_bgr.astype(np.float32) * (1.0 - ink_mask) + input_bgr.astype(np.float32) * ink_mask).astype(np.uint8)

    # 3. Clean Speech Bubbles & Gutter Borders
    white_bg = (gray > 225).astype(np.uint8) * 255
    ink_binary = (gray < 85).astype(np.uint8) * 255
    white_bg[ink_binary > 0] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_bg)
    gutter_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
    for i in range(1, num_labels):
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # Gutters / outer white panels
        if (x <= 5 or y <= 5 or x + bw >= w_orig - 5 or y + bh >= h_orig - 5) and area > 2000:
            if float(np.mean(gray[labels == i])) > 230:
                gutter_mask[labels == i] = 1.0

    if np.any(gutter_mask > 0):
        gutter_3ch = gutter_mask[:, :, None]
        final_bgr = (final_bgr.astype(np.float32) * (1.0 - gutter_3ch) + input_bgr.astype(np.float32) * gutter_3ch).astype(np.uint8)

    # Clean temporary scratch files
    try:
        os.remove(init_path)
        os.remove(out_path)
    except Exception:
        pass

    return final_bgr

def process_chapter(pdf_name: str):
    in_path = os.path.join(SRC_DIR, pdf_name)
    out_path = os.path.join(OUT_DIR, pdf_name)
    if os.path.exists(out_path):
        print(f"[SKIP] Already exists: {pdf_name}")
        return

    print(f"\n[START] Processing {pdf_name} with FLUX + Anime LoRA into 'dorohedoro 4'...")
    doc = pdfium.PdfDocument(in_path)
    total_pages = len(doc)
    colorized_pil_pages = []

    for idx in range(total_pages):
        t_page = time.time()
        page = doc[idx]
        pil_img = page.render(scale=2.0).to_pil()
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # FLUX Colorize
        col_bgr = run_flux_page(bgr, idx)
        col_rgb = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2RGB)
        colorized_pil_pages.append(Image.fromarray(col_rgb))
        print(f"  Page {idx + 1}/{total_pages} processed in {time.time() - t_page:.1f}s", flush=True)

    # Save compiled PDF
    if colorized_pil_pages:
        colorized_pil_pages[0].save(
            out_path,
            save_all=True,
            append_images=colorized_pil_pages[1:],
            resolution=150.0
        )
        print(f"[DONE] Saved {pdf_name} ({total_pages} pages, {os.path.getsize(out_path)} bytes)")

def main():
    files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith(".pdf")])
    target_chapters = []
    for f in files:
        m = re.search(r"Chapter_(\d+(?:\.\d+)?)", f)
        if m and float(m.group(1)) >= 145.0:
            target_chapters.append((float(m.group(1)), f))

    target_chapters.sort(key=lambda x: x[0])
    print(f"Total chapters to process with FLUX: {len(target_chapters)}")

    start_total = time.time()
    for ch_num, pdf_name in target_chapters:
        process_chapter(pdf_name)

    print(f"\nALL CHAPTERS COMPLETED WITH FLUX IN 'dorohedoro 4' IN {time.time() - start_total:.1f}s!")

if __name__ == "__main__":
    main()
