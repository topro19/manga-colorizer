import os
import sys
import re
import time
import cv2
import numpy as np
from PIL import Image
import pypdfium2 as pdfium
import onnxruntime as ort

SRC_DIR = r"C:\Users\ginto\Downloads\dorohedoro"
OUT_DIR = r"C:\Users\ginto\Downloads\dorohedoro 3"
MODEL_PATH = "models/manga-colorize-fp16.onnx"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Initialize ONNX Runtime with DirectML GPU acceleration
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
available = ort.get_available_providers()
prov = ["DmlExecutionProvider", "CPUExecutionProvider"] if "DmlExecutionProvider" in available else ["CPUExecutionProvider"]
sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=prov)
input_name = sess.get_inputs()[0].name
print(f"[INIT] Loaded MangaColorize ONNX on {prov[0]} for Batch 'dorohedoro 3'")

# 2. Canonical Dorohedoro Palette memory (Lab space for precise optical chrominance calibration)
# Nikaido: warm peach skin, blonde hair, dark suit
# Caiman: reptilian green scales, crimson cross-marks, black gas-mask
# Shin: pale peach skin, blonde hair, crimson heart mask
# Noi: pale skin, platinum/pinkish-white hair, red eyes
# En: russet-orange hair, dark pinstriped suit
# Ebisu: bone-white skull mask, red/purple trim
CANONICAL_SKIN_LAB = np.array([72.0, 131.0, 134.0], dtype=np.float32)
CANONICAL_BLONDE_LAB = np.array([70.0, 138.0, 142.0], dtype=np.float32)
CANONICAL_RED_LAB = np.array([50.0, 168.0, 148.0], dtype=np.float32)

def colorize_page(bgr_image: np.ndarray) -> np.ndarray:
    h_orig, w_orig = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Optimal inference scale (32-pixel multiple)
    max_dim = 1024
    scale = min(max_dim / float(h_orig), max_dim / float(w_orig), 1.0)
    th = int(round(h_orig * scale / 32.0)) * 32
    tw = int(round(w_orig * scale / 32.0)) * 32

    gray_resized = cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)
    inp = np.zeros((1, 5, th, tw), dtype=np.float32)
    inp[0, 0] = gray_resized.astype(np.float32) / 255.0

    raw_out = sess.run(None, {input_name: inp})[0]
    rgb = np.clip(raw_out[0].transpose(1, 2, 0), 0.0, 1.0)
    col_full = cv2.resize((rgb[:, :, ::-1] * 255.0).astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)

    # Convert to Lab for Character Palette Calibration
    col_lab = cv2.cvtColor(col_full, cv2.COLOR_BGR2LAB)
    
    # 1. Skin & Anatomy Calibration
    # Detect skin tones (warm chrominance with paper brightness > 150)
    skin_mask = (gray > 140) & (col_lab[:, :, 1] > 130) & (col_lab[:, :, 2] > 128)
    if np.any(skin_mask):
        # Softly anchor toward Dorohedoro canonical anime skin tones
        col_lab[skin_mask, 1] = np.clip(0.70 * col_lab[skin_mask, 1].astype(float) + 0.30 * CANONICAL_SKIN_LAB[1], 0, 255).astype(np.uint8)
        col_lab[skin_mask, 2] = np.clip(0.70 * col_lab[skin_mask, 2].astype(float) + 0.30 * CANONICAL_SKIN_LAB[2], 0, 255).astype(np.uint8)

    # 2. Blonde Hair Calibration
    hair_mask = (gray > 130) & (col_lab[:, :, 2] > 140) & (col_lab[:, :, 1] < 135)
    if np.any(hair_mask):
        col_lab[hair_mask, 1] = np.clip(0.75 * col_lab[hair_mask, 1].astype(float) + 0.25 * CANONICAL_BLONDE_LAB[1], 0, 255).astype(np.uint8)
        col_lab[hair_mask, 2] = np.clip(0.75 * col_lab[hair_mask, 2].astype(float) + 0.25 * CANONICAL_BLONDE_LAB[2], 0, 255).astype(np.uint8)

    # Bilateral smoothing on chrominance only to eliminate color-bleeding/roughness
    col_lab[:, :, 1] = cv2.bilateralFilter(col_lab[:, :, 1], 5, 25, 25)
    col_lab[:, :, 2] = cv2.bilateralFilter(col_lab[:, :, 2], 5, 25, 25)
    col_calibrated = cv2.cvtColor(col_lab, cv2.COLOR_LAB2BGR)

    # 3. Optical Multiply under Original Ink Lines (Preserves 100% Q Hayashida art)
    color_f = col_calibrated.astype(np.float32) / 255.0
    raw_f = bgr_image.astype(np.float32) / 255.0
    mult_bgr = (color_f * raw_f * 255.0).astype(np.uint8)

    # 4. Pure Black Ink Mask Restoration
    ink_mask = (gray < 85).astype(np.float32)[:, :, None]
    final_bgr = (mult_bgr.astype(np.float32) * (1.0 - ink_mask) + bgr_image.astype(np.float32) * ink_mask).astype(np.uint8)

    # 5. Protect Page Gutters, Margins, & Outer Borders
    white_bg = (gray > 225).astype(np.uint8) * 255
    ink_binary = (gray < 85).astype(np.uint8) * 255
    white_bg[ink_binary > 0] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_bg)
    gutter_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
    for i in range(1, num_labels):
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if (x <= 5 or y <= 5 or x + bw >= w_orig - 5 or y + bh >= h_orig - 5) and area > 2500:
            comp_mask = (labels == i)
            if float(np.mean(gray[comp_mask])) > 230:
                gutter_mask[comp_mask] = 1.0

    if np.any(gutter_mask > 0):
        gutter_3ch = gutter_mask[:, :, None]
        final_bgr = (final_bgr.astype(np.float32) * (1.0 - gutter_3ch) + bgr_image.astype(np.float32) * gutter_3ch).astype(np.uint8)

    return final_bgr

def process_chapter(pdf_name: str):
    in_path = os.path.join(SRC_DIR, pdf_name)
    out_path = os.path.join(OUT_DIR, pdf_name)
    if os.path.exists(out_path):
        print(f"[SKIP] Already exists: {pdf_name}")
        return

    print(f"[START] Processing {pdf_name} into 'dorohedoro 3'...")
    doc = pdfium.PdfDocument(in_path)
    total_pages = len(doc)
    colorized_pil_pages = []

    for idx in range(total_pages):
        page = doc[idx]
        pil_img = page.render(scale=2.0).to_pil()
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Colorize
        col_bgr = colorize_page(bgr)
        col_rgb = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2RGB)
        colorized_pil_pages.append(Image.fromarray(col_rgb))

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
        if m and float(m.group(1)) >= 144.0:
            target_chapters.append((float(m.group(1)), f))

    target_chapters.sort(key=lambda x: x[0])
    print(f"Total chapters to process in 'dorohedoro 3': {len(target_chapters)}")

    start_total = time.time()
    for ch_num, pdf_name in target_chapters:
        t0 = time.time()
        process_chapter(pdf_name)
        print(f"Chapter {ch_num} took {time.time() - t0:.1f}s")

    print(f"\nALL 27 CHAPTERS COMPLETED IN {time.time() - start_total:.1f}s!")

if __name__ == "__main__":
    main()
