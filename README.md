# MangaColorizer Studio (AMD Radeon RX 9060 XT DirectML)

A high-performance local Windows desktop application for manga colorization powered by a **Pluggable Multi-Model Architecture** (MangaColorization-v2, DDColor Chroma, and Reference-Guided engines) hardware-accelerated through **ONNX Runtime DirectML** on AMD Radeon GPUs.

Runs 100% locally on Windows without requiring CUDA, NVIDIA GPUs, Linux, WSL, or Docker.

---

## What Was Redesigned & Fixed

The colorization pipeline has been completely re-engineered from the ground up to prevent the "flat color filter" and color bleeding issues:

1. **Luminance Preservation ($L$ Channel in CIE Lab)**:
   - The original manga's grayscale shading, screentones, cross-hatching, and dark ink contrast are strictly preserved.
   - The AI models predict chrominance ($a, b$ channels or RGB chroma). Color is injected underneath the ink lines rather than painting over them.
2. **Line Art Protection (0–100%, Default 90%)**:
   - High-contrast ink lines are dynamically masked and desaturated back to neutral. Ink contours remain pure, crisp black without purple or orange halos.
3. **Edge-Guided Cross-Bilateral Filtering**:
   - A fast guided filter uses the original manga page's ink edges as the guide image. Chrominance is prevented from bleeding across panel boundaries, character clothing lines, and hair strands.
4. **Speech Bubble & Gutter Isolation**:
   - Dialog bubbles, sound effects backgrounds, and panel gutters are segmented and clamped to neutral white, keeping lettering pristine.
5. **Pluggable Multi-Model Architecture & Model Lab**:
   - **MangaColorization-v2**: Specifically trained generator for manga lines with interactive hint conditioning.
   - **DDColor-FP16 (Chroma)**: DirectML dual-decoder predicting native Lab $a, b$ chrominance with natural skin tones and rich ambient color.
   - **Reference-Guided Engine**: Injects color anchors based on uploaded reference artwork or character palettes.
   - **Model Lab**: Compare all backends side-by-side on the active manga page, benchmarking latency, line preservation, and aesthetic quality.
6. **Character Consistency & Color Memory (`color_memory.json`)**:
   - Extract dominant palettes from character reference artwork using k-means.
   - Lock specific hex codes for hair, skin, and clothing across 180 pages.
7. **Interactive 4-Way Viewer**:
   - Draggable split-slider, Side-by-side, Blink alternating view, and Difference Heatmap.

---

## Project Structure

```text
MangaColorizer/
│
├── app.py                     # Studio GUI entry point (PySide6)
├── requirements.txt           # Python dependencies (DirectML, PySide6, OpenCV, Pillow)
├── README.md                  # Documentation and user guide
├── setup.bat                  # Automated 1-click installer and launcher
├── run.bat                    # Fast desktop launcher
│
├── models/
│   ├── manga-colorize-fp16.onnx   # MangaColorize-v2 ONNX generator (58.8 MB)
│   └── ddcolor-fp16.onnx          # DDColor DirectML FP16 chroma model (108 MB)
│
├── core/
│   ├── filters.py             # Fast guided filter, line art mask, bubble isolation, Lab compositing
│   ├── palette_manager.py     # Color memory persistence (color_memory.json), k-means palette extractor
│   ├── batch_processor.py     # Resilient QThread batch processor with pause/resume and metadata
│   ├── model_manager.py       # DirectML session loading, auto-download, and GPU test
│   └── backends/
│       ├── base_colorizer.py      # Abstract colorizer backend interface
│       ├── manga_colorization_v2.py # MangaColorize-v2 backend with upstream scaling & hints
│       ├── ddcolor_colorizer.py     # DDColor native Lab AB chroma backend
│       ├── reference_guided.py    # Palette-conditioned consistent colorizer
│       └── colorizer_registry.py  # Factory and registry for hot-swapping models in UI
│
├── ui/
│   ├── main_window.py         # Main studio with 3 workspaces (Studio, Model Lab, Color Memory)
│   ├── model_lab.py           # Multi-model side-by-side benchmark panel
│   ├── palette_panel.py       # Color lock and character consistency inspector
│   ├── viewer.py              # Split-slider, side-by-side, blink, and difference viewer
│   ├── progress.py            # Real-time progress bar, speed (ppm), ETA, and latency
│   ├── hint_editor.py         # Interactive color hint canvas dialog
│   └── styles.py              # Modern dark studio styling
│
├── utils/
│   ├── image_utils.py         # Natural sorting, metadata persistence
│   ├── cbz.py                 # CBZ archive extraction and packaging
│   ├── pdf.py                 # PDF page rendering (pypdfium2) and export
│   └── gpu_monitor.py         # Windows CIM/WMI GPU telemetry
│
├── input/                     # Manga volumes, archives, and sample pages
└── output/                    # Colorized outputs and _colorization_metadata.json
```

---

## Quick Start

### 1-Click Launch (Recommended)
Double-click `run.bat` or run:
```cmd
run.bat
```

### Full Setup
If setting up a fresh machine:
```cmd
setup.bat
```

---

## Workspaces in the Studio

### 1. Studio Workspace
- Select input source (**Folder**, **CBZ**, or **PDF**).
- Adjust **Color Strength** (0–100%, default 65%).
- Adjust **Line Art Preservation** (0–100%, default 90%).
- Select active model in header (**MangaColorization-v2**, **DDColor-FP16**, or **Reference-Guided**).
- Inspect pages in the center viewer using **Split Slider**, **Side-by-Side**, **Blink**, or **Difference**.
- Colorize individual pages or entire volumes with full pause, resume, and progress tracking.

### 2. Model Lab Workspace
- Run all installed models concurrently on the active manga page.
- View side-by-side preview cards with DirectML latency (ms), model weights, and line-preservation comparison.

### 3. Color Memory Workspace
- Upload an official color illustration or anime screenshot.
- Auto-extract dominant character tones (hair, skin, clothing) via k-means clustering.
- Lock specific hex codes to enforce character consistency across all pages.

### 4. 📖 Live Manga Reader Workspace (Target Online Manga & Read While Coloring)
- **Target URL**: Paste any MangaDot URL (e.g. `https://mangadot.net/chapter/37925`).
- **Chapter Range**: Set chapter start and end range (e.g. `Ch From: 99` `To: 100`).
- **Bypasses Cloudflare**: Automates session authentication and chapter metadata discovery.
- **Read Immediately While Coloring**: The moment pages download, start reading in crisp B&W without waiting for the batch to finish.
- **Instant Hot-Reload to Color**: As the AMD Radeon RX 9060 XT (DirectML) finishes colorizing each page in the background, the reader dynamically swaps the page to the full-color version.
- **Reader Controls**: Color / B&W / Split views, Fit-to-screen, 100% zoom, smooth panning, slider, and Left/Right keyboard navigation (`←` / `→`).

---

## Verifying AMD Radeon RX 9060 XT DirectML Hardware Acceleration

1. Top header badge will display: `GPU: AMD Radeon RX 9060 XT / DirectML`.
2. Click **⚡ Test GPU**:
   - Confirms `DmlExecutionProvider` tensor execution and displays per-inference latency (~15 ms).
3. In Windows Task Manager (`Ctrl + Shift + Esc`) -> **Performance** -> **GPU (AMD Radeon RX 9060 XT)**, you will observe Direct3D/Compute spikes during colorization.

---

## ℹ️ Project Background & Notes
> **Note:** This project was developed with AI assistance. It was made public with the thought that someone out there might find it useful and enjoy having a local, GPU-accelerated manga colorization pipeline. It's released open-source so anyone can use, adapt, or build upon it!
