# Current Architecture & Model Audit

## 1. Overview
The existing application is a local Windows desktop manga colorizer built with PySide6 and ONNX Runtime DirectML targeting the AMD Radeon RX 9060 XT (16 GB VRAM).
It currently operates in a predominantly **page-by-page** manner, running an inference pass on each image file, applying hard masks (line art and speech bubbles), and blending with original CIE LAB luminance.

## 2. UI Framework
- **Framework**: PySide6 (Qt 6 for Python)
- **Main Window**: `ui/main_window.py` (`MainWindow`), supporting tabbed workspaces:
  - Studio (batch import, parameters, before/after slider)
  - Live Reader (`ui/live_reader.py`, web downloader + page viewing)
  - Model Lab (`ui/model_lab.py`, side-by-side backend comparison)
  - Color Palette & Memory (`ui/palette_panel.py`)
  - Benchmark Lab (`ui/benchmark_screen.py`)

## 3. Models & GPU Providers
- **Hardware Target**: AMD Radeon RX 9060 XT (16 GB), Windows DirectML.
- **Provider Priority**: `DmlExecutionProvider` -> fallback to `CPUExecutionProvider`.
- **Installed ONNX Models** in `models/`:
  1. `manga-colorize-fp16.onnx` (58.8 MB):
     - **Input Shape**: `[1, 5, H, W]` float32:
       - Channel 0: Grayscale image `(gray / 255.0 - 0.5) / 0.5`
       - Channels 1..3: Normalized Color Hints `(c / 255.0 - 0.5) / 0.5`
       - Channel 4: Hint binary mask `(0.0 or 1.0)`
     - **Dimensions**: H and W must be multiples of 32 (padded with edge reflection).
     - **Output Shape**: `[1, 3, H, W]` float32. Dynamically scaled based on whether output values are natively in `[0, 1]` or `[-1, 1]`.
  2. `ddcolor-fp16.onnx` (108 MB):
     - **Input Shape**: `[1, 3, 256, 256]` float32 RGB `[0, 1]`.
     - **Output Shape**: `[1, 2, 256, 256]` float32 representing CIE LAB AB channels centered at 0.
  3. `ddcolor-tiny-fp16.onnx` (129 MB):
     - Lightweight variant of DDColor.

## 4. Current Preprocessing & Postprocessing
- **Preprocessing**:
  - Image loaded via OpenCV BGR.
  - Multi-scale resizing & padding to divisible-by-32 dimensions.
  - Region segmentation & candidate bubble detection.
- **Postprocessing**:
  - `create_line_art_mask`: Adaptive threshold + Gaussian blur to protect ink.
  - `create_speech_bubble_mask`: Morphological connected component inspection + text-density checking to protect speech bubbles from color tinting.
  - `composite_luminance_and_chroma`:
    - Fast Guided Filter snapping predicted chrominance to original high-res luminance contours.
    - Soft white-area luminance modulation ($L > 195$) so bright skin tones and clothing don't wash out into pure white.
    - Preserves high-res original luminance $L_{\text{orig}}$.

## 5. Identified Bottlenecks & Missing Capabilities
1. **Lack of Chapter-Level Continuity**:
   - Each page is evaluated in isolation without global scene clustering or persistent character visual tracking across sequences.
2. **Missing Panel-Level Decomposition**:
   - Full pages are fed directly into the model rather than decomposing complex layouts into virtual panel crops with gutters protected.
3. **No Project Serialization (`.mcp`)**:
   - The user cannot save a work-in-progress chapter project, close the application, and reload character palettes and completed pages.
4. **No Region-Specific Manual Corrections**:
   - Clicking a specific element (e.g. hair) requires regenerating the whole page rather than editing that segmented cell.

## 6. Recommended Architecture (Refactor Roadmap)
Restructure into clear decoupled packages:
- `app/ingestion`: Chapter extraction (Folders, CBZ, PDF).
- `app/analysis`: Panel extraction, speech bubble detection, line art masking.
- `app/characters`: Character clustering, persistent memory (`character_memory.json`), appearance tracking.
- `app/scenes`: Scene grouping based on visual similarity and sequential continuity.
- `app/color`: Structured color plan generation, consistency enforcement.
- `app/models`: Model registry & backend abstraction (`ColorizationBackend`).
- `app/protection`: Watertight ink, text, bubble, and gutter protection masks.
- `app/reassembly`: Sub-panel recombination back into exact original page dimensions.
- `app/storage`: `.mcp` project format, resume state, and cache manager.
- `app/ui`: Streamlined consumer workflow (Upload -> Instructions/Ref -> Colorize -> Results/Editor).
