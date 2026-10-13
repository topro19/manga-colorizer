"""
Manga page decomposition and panel segmentation engine.
Extracts individual panels from full-resolution pages while preserving exact global coordinates.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


@dataclass
class PanelBox:
    panel_id: str
    page_idx: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    mean_lum: float
    is_splash: bool = False
    crop_bgr: Optional[np.ndarray] = None


class PanelDetector:
    """
    Decomposes manga pages into virtual panels bounded by panel frames and gutters.
    """

    @staticmethod
    def detect_panels(page_bgr: np.ndarray, page_idx: int = 1) -> List[PanelBox]:
        h, w = page_bgr.shape[:2]
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)

        # Panel borders are usually crisp black lines enclosing panels
        ink = (gray < 70).astype(np.uint8) * 255
        # Close thin breaks in panel frames
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)

        # Invert to find enclosed panel interiors
        interiors = cv2.bitwise_not(closed_ink)
        # Exclude extreme outer margins
        margin = max(4, int(min(h, w) * 0.01))
        interiors[:margin, :] = 0
        interiors[-margin:, :] = 0
        interiors[:, :margin] = 0
        interiors[:, -margin:] = 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interiors)
        panels: List[PanelBox] = []

        min_panel_area = int((h * w) * 0.04)  # at least 4% of page
        max_panel_area = int((h * w) * 0.96)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_panel_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            pw = stats[i, cv2.CC_STAT_WIDTH]
            ph = stats[i, cv2.CC_STAT_HEIGHT]

            # Require reasonable width and height
            if pw < (w * 0.15) or ph < (h * 0.10):
                continue

            crop = page_bgr[y:y+ph, x:x+pw].copy()
            mean_lum = float(np.mean(gray[y:y+ph, x:x+pw]))
            is_splash = area > (h * w * 0.70)

            panels.append(
                PanelBox(
                    panel_id=f"p{page_idx}_pan_{len(panels)+1}",
                    page_idx=page_idx,
                    bbox=(x, y, pw, ph),
                    mean_lum=mean_lum,
                    is_splash=is_splash,
                    crop_bgr=crop
                )
            )

        # Sort panels in typical manga reading order (Top to Bottom, Right to Left)
        # Using row grouping
        if panels:
            panels.sort(key=lambda p: (p.bbox[1] // (h // 6), -p.bbox[0]))
        else:
            # Fallback to whole page as single panel if no distinct panels found (e.g. splash art)
            panels.append(
                PanelBox(
                    panel_id=f"p{page_idx}_pan_full",
                    page_idx=page_idx,
                    bbox=(0, 0, w, h),
                    mean_lum=float(np.mean(gray)),
                    is_splash=True,
                    crop_bgr=page_bgr.copy()
                )
            )

        return panels
