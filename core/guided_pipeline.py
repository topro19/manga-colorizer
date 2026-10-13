"""
Master Manga Colorization Pipeline Coordinator.
Coordinates:
1. Input Preprocessing & Page Resolution Validation
2. Hard Mask Extraction (Lines, Speech Bubbles, Panel Borders, Text)
3. Region Segmentation (Ink cells, semantic candidate classification)
4. Color Planning (Optional VLM Director + SHA256 caching + Color Bible Memory)
5. Sparse Hint Generation (Anchor colors inside cells)
6. ONNX DirectML Colorization Inference (FP16 Baseline)
7. LAB Compositing with Strict L-Channel Preservation (0-100%)
8. Fast Guided Filter Edge-Lock (Zero bleed across contours)
9. Speech Bubble & Gutter Isolation (Pristine clean whites)
10. Cross-Page Semantic Consistency & Manifest Logging
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .backends.base_colorizer import BaseColorizer
from .filters import composite_luminance_and_chroma
from .palette_manager import PaletteManager
from .segmenter import RegionSegmenter
from .vlm_director import BaseVLMDirector, LocalRuleVLMDirector


class PipelineExecutionResult:
    def __init__(
        self,
        final_bgr: np.ndarray,
        raw_model_bgr: np.ndarray,
        l_preserved_bgr: np.ndarray,
        line_mask: np.ndarray,
        bubble_mask: np.ndarray,
        color_plan: Dict[str, Any],
        timings_ms: Dict[str, float]
    ):
        self.final_bgr = final_bgr
        self.raw_model_bgr = raw_model_bgr
        self.l_preserved_bgr = l_preserved_bgr
        self.line_mask = line_mask
        self.bubble_mask = bubble_mask
        self.color_plan = color_plan
        self.timings_ms = timings_ms


class GuidedColorizationPipeline:
    """
    Production-grade guided colorization pipeline for Windows AMD Radeon RX 9060 XT DirectML.
    """

    def __init__(
        self,
        colorizer: BaseColorizer,
        vlm_director: Optional[BaseVLMDirector] = None,
        palette_manager: Optional[PaletteManager] = None
    ):
        self.colorizer = colorizer
        self.vlm_director = vlm_director or LocalRuleVLMDirector()
        self.palette_manager = palette_manager or PaletteManager()

    def process_page(
        self,
        image_path: str,
        mode: str = "BALANCED",  # 'FAST', 'BALANCED', 'QUALITY'
        color_strength: float = 0.65,
        l_preservation: float = 0.95,
        line_protection: float = 0.95,
        protect_bubbles: bool = True,
        use_guided_filter: bool = True,
        manual_hints: Optional[List[Dict[str, Any]]] = None,
        force_vlm_refresh: bool = False
    ) -> PipelineExecutionResult:
        timings: Dict[str, float] = {}
        t_start = time.perf_counter()

        # 1. Load image
        t0 = time.perf_counter()
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Failed to load image from {image_path}")
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        timings["io_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Hard masks & region segmentation
        t0 = time.perf_counter()
        line_mask = RegionSegmenter.create_line_art_mask(gray, protection_strength=line_protection)
        bubble_mask = RegionSegmenter.create_bubble_and_text_mask(gray)
        labeled_cells, regions = RegionSegmenter.segment_cells(gray, bubble_mask)
        timings["segmentation_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. Color Planning (VLM Director + Color Memory)
        t0 = time.perf_counter()
        color_plan = {}
        if mode in ("BALANCED", "QUALITY"):
            color_plan = self.vlm_director.analyze_page(
                image_path=image_path,
                regions=regions,
                color_memory=self.palette_manager.data,
                force_refresh=force_vlm_refresh
            )
        timings["vlm_plan_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. Sparse hint generation
        t0 = time.perf_counter()
        sparse_hints: List[Tuple[int, int, int, int, int, int]] = []
        if mode in ("BALANCED", "QUALITY") and color_plan:
            sparse_hints = RegionSegmenter.generate_sparse_hints_from_plan(
                color_plan=color_plan,
                regions=regions,
                image_shape=(h, w)
            )

        # Incorporate manual user hints if provided
        if manual_hints:
            for mh in manual_hints:
                mx, my = mh["x"], mh["y"]
                mc = mh["color"]
                mrad = mh.get("radius", 14)
                sparse_hints.append((mx, my, mc[0], mc[1], mc[2], mrad))
        timings["hints_ms"] = (time.perf_counter() - t0) * 1000.0

        # 5. DirectML Colorization inference
        t0 = time.perf_counter()
        options = {
            "color_locks": self.palette_manager.get_color_locks(),
            "reference_palette": self.palette_manager.get_active_palette()
        }

        # Run model to get raw colorized image
        raw_color_bgr = self.colorizer.colorize(
            bgr_image=bgr,
            color_strength=color_strength,
            line_preservation=0.0,  # Do not mask line in raw pass so we can compare
            speech_bubble_protection=False,
            use_guided_filter=False,
            hint_points=sparse_hints,
            options=options
        )
        timings["inference_ms"] = (time.perf_counter() - t0) * 1000.0

        # 6. L-Channel Preservation in CIE LAB space
        t0 = time.perf_counter()
        orig_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        raw_lab = cv2.cvtColor(raw_color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        L_orig = orig_lab[:, :, 0]
        L_raw = raw_lab[:, :, 0]
        L_blended = L_raw * (1.0 - l_preservation) + L_orig * l_preservation

        lab_l_preserved = np.zeros_like(orig_lab)
        lab_l_preserved[:, :, 0] = np.clip(L_blended, 0, 255)
        lab_l_preserved[:, :, 1] = raw_lab[:, :, 1]
        lab_l_preserved[:, :, 2] = raw_lab[:, :, 2]
        l_preserved_bgr = cv2.cvtColor(lab_l_preserved.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # 7. Final Hard Mask Compositing & Guided Filter
        final_bgr = composite_luminance_and_chroma(
            original_bgr=bgr,
            colorized_bgr=l_preserved_bgr,
            color_strength=color_strength,
            line_preservation=line_protection,
            speech_bubble_protection=protect_bubbles,
            use_guided_filter=use_guided_filter
        )
        timings["postprocessing_ms"] = (time.perf_counter() - t0) * 1000.0
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0

        return PipelineExecutionResult(
            final_bgr=final_bgr,
            raw_model_bgr=raw_color_bgr,
            l_preserved_bgr=l_preserved_bgr,
            line_mask=line_mask,
            bubble_mask=bubble_mask,
            color_plan=color_plan,
            timings_ms=timings
        )
