import os
import re
import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image
import onnxruntime as ort

SRC_DIR = r"C:\Users\ginto\Downloads\dorohedoro"
OUT_DIR = r"C:\Users\ginto\Downloads\dorohedoro 2"
MODEL_PATH = "models/manga-colorize-fp16.onnx"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Initialize DirectML ONNX Session
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
available = ort.get_available_providers()
prov = ["DmlExecutionProvider"] if "DmlExecutionProvider" in available else ["CPUExecutionProvider"]
sess = ort.InferenceSession(MODEL_PATH, opts, providers=prov)
input_name = sess.get_inputs()[0].name
print(f"[INIT] Loaded MangaColorize ONNX on {prov[0]}")

def colorize_page(bgr_image: np.ndarray) -> np.ndarray:
    h_orig, w_orig = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Scale to 768 multiple
    scale = 768.0 / max(h_orig, w_orig)
    th = int(round(h_orig * scale / 32.0)) * 32
    tw = int(round(w_orig * scale / 32.0)) * 32

    gray_resized = cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)
    inp = np.zeros((1, 5, th, tw), dtype=np.float32)
    inp[0, 0] = gray_resized.astype(np.float32) / 255.0

    raw_out = sess.run(None, {input_name: inp})[0]
    rgb = np.clip(raw_out[0].transpose(1, 2, 0), 0.0, 1.0)
    col_full = cv2.resize((rgb[:, :, ::-1] * 255.0).astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)

    # Overlay pure original black ink lines (preserves 100% original sharp line-art)
    ink_raw = (gray < 75).astype(np.float32)
    ink_mask = cv2.GaussianBlur(ink_raw, (3, 3), 0)[:, :, None]
    final_bgr = (col_full.astype(np.float32) * (1.0 - ink_mask) + bgr_image.astype(np.float32) * ink_mask).astype(np.uint8)

    # Protect outer page gutters & margins
    ink_binary = (gray < 85).astype(np.uint8) * 255
    white = (gray > 220).astype(np.uint8) * 255
    white[ink_binary > 0] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    gutter_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
    for i in range(1, num_labels):
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if (x <= 5 or y <= 5 or x + bw >= w_orig - 5 or y + bh >= h_orig - 5) and area > 2500:
            comp_mask = (labels == i)
            if float(np.mean(gray[comp_mask])) > 230:
                gutter_mask[comp_mask] = 1.0

    if np.any(gutter_mask > 0):
        gutter_mask = cv2.dilate(gutter_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))[:, :, None]
        bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        final_bgr = (final_bgr * (1.0 - gutter_mask) + bgr_gray * gutter_mask).astype(np.uint8)

    return final_bgr

def process_chapter(pdf_name: str):
    in_path = os.path.join(SRC_DIR, pdf_name)
    out_path = os.path.join(OUT_DIR, pdf_name)
    if os.path.exists(out_path):
        print(f"[SKIP] Already exists: {pdf_name}")
        return

    print(f"[START] Processing {pdf_name}...")
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
        print(f"[DONE] Saved {pdf_name} ({total_pages} pages)")

def main():
    files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith(".pdf")])
    target_chapters = []
    for f in files:
        m = re.search(r"Chapter_(\d+(?:\.\d+)?)", f)
        if m and float(m.group(1)) >= 138.0:
            target_chapters.append((float(m.group(1)), f))

    target_chapters.sort(key=lambda x: x[0])
    print(f"Total chapters to process: {len(target_chapters)}")

    for ch_num, pdf_name in target_chapters:
        process_chapter(pdf_name)

    print("\nALL CHAPTERS COMPLETED INTO 'dorohedoro 2'!")

if __name__ == "__main__":
    main()
