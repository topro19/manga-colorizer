"""
Reassembles colorized panels and protective masks back into the exact original page dimensions.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import cv2
import numpy as np
from app.analysis.panel_detector import PanelBox
from app.protection.masks import ProtectionMasks
from core.filters import composite_luminance_and_chroma


class PageReassembler:
    """
    Blends colorized panel crops back into the original full-resolution page canvas,
    strictly maintaining original dimensions, line art, gutters, and dialogue bubbles.
    """

    @staticmethod
    def reassemble_page(
        original_bgr: np.ndarray,
        colorized_panels: List[Tuple[PanelBox, np.ndarray]],
        color_strength: float = 0.85,
        line_preservation: float = 0.90,
        protect_bubbles: bool = True,
        vivid_mode: bool = False
    ) -> np.ndarray:
        h, w = original_bgr.shape[:2]
        canvas_bgr = original_bgr.copy()

        # Composite each colorized panel into its bounding box
        for panel, colored_crop in colorized_panels:
            x, y, pw, ph = panel.bbox
            if colored_crop.shape[:2] != (ph, pw):
                colored_crop = cv2.resize(colored_crop, (pw, ph), interpolation=cv2.INTER_CUBIC)
            canvas_bgr[y:y+ph, x:x+pw] = colored_crop

        # Final full-page anti-bleed & luminance preservation pass
        final_page = composite_luminance_and_chroma(
            original_bgr=original_bgr,
            colorized_bgr=canvas_bgr,
            color_strength=color_strength,
            line_preservation=line_preservation,
            speech_bubble_protection=protect_bubbles,
            use_guided_filter=True,
            vivid_mode=vivid_mode
        )

        return final_page
