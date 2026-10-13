"""
Color Consistency Engine.
Enforces chromatic continuity between consecutive manga pages and scenes.
Detects and corrects unexpected sudden color deviations.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


class ConsistencyEngine:
    """
    Maintains color harmony and character palette consistency across whole chapters.
    """

    @staticmethod
    def enforce_scene_harmony(
        page_bgr: np.ndarray,
        target_palette: Dict[str, str],
        protection_mask: np.ndarray,
        harmony_strength: float = 0.20
    ) -> np.ndarray:
        """
        Gently harmonizes the overall page ambient lighting to match the scene mood.
        Leaves protected ink lines and text bubbles completely untouched.
        """
        if harmony_strength <= 0.0 or not target_palette:
            return page_bgr

        # Extract dominant environment color
        env_hex = target_palette.get("sky", target_palette.get("ambient", ""))
        if not env_hex or len(env_hex) != 7:
            return page_bgr

        try:
            r = int(env_hex[1:3], 16)
            g = int(env_hex[3:5], 16)
            b = int(env_hex[5:7], 16)
            tint_bgr = np.array([b, g, r], dtype=np.float32)

            page_float = page_bgr.astype(np.float32)
            # Modulate ambient tone outside protected boundaries
            factor = (1.0 - protection_mask[:, :, np.newaxis]) * float(harmony_strength)
            harmonized = page_float * (1.0 - factor * 0.15) + tint_bgr * (factor * 0.15)
            return np.clip(harmonized, 0, 255).astype(np.uint8)
        except Exception:
            return page_bgr
