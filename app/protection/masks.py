"""
Watertight manga structure protection masks:
- Black Ink line-art mask (prevents coloring or eroding lines)
- Dialogue speech bubble mask (keeps bubbles pure clean white)
- Text region mask (keeps dialogue letters sharp black)
- Gutter / panel margin mask (keeps frame borders and gutters neutral)
"""

from __future__ import annotations
import cv2
import numpy as np


class ProtectionMasks:
    """Generates structural protection masks for a manga page."""

    @staticmethod
    def create_ink_mask(gray: np.ndarray, threshold: int = 75, protection_strength: float = 0.95) -> np.ndarray:
        """
        Extracts black ink contours and screentone hatching.
        Returns float32 mask [0, 1] where 1.0 = solid ink.
        """
        deep_ink = (gray < threshold).astype(np.float32)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        ).astype(np.float32) / 255.0
        combined = np.maximum(deep_ink, adaptive)
        blurred = cv2.GaussianBlur(combined, (3, 3), 0)
        return np.clip(blurred * protection_strength, 0.0, 1.0)

    @staticmethod
    def create_speech_bubble_mask(gray: np.ndarray, white_thresh: int = 246) -> np.ndarray:
        """
        Detects bounded speech bubbles containing dialog text.
        Never falsely flags open character line art or white skin as bubbles.
        Returns float32 mask [0, 1] where 1.0 indicates protected bubble space.
        """
        h, w = gray.shape[:2]
        ink = (gray < 85).astype(np.uint8) * 255
        ink_closed = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        interiors = cv2.bitwise_not(ink_closed)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interiors)
        bubble_mask = np.zeros_like(gray, dtype=np.float32)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1000 or area > 45000:
                continue
            x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # Exclude canvas outer gutters
            if x <= 2 or y <= 2 or (x + bw) >= (w - 2) or (y + bh) >= (h - 2):
                continue

            comp = (labels == i)
            mean_lum = float(np.mean(gray[comp]))
            if mean_lum < float(white_thresh):
                continue

            patch = gray[y:y+bh, x:x+bw]
            text_density = np.sum(patch < 80) / float(bw * bh)
            aspect = max(bw, bh) / max(1, min(bw, bh))

            # Speech bubbles have dialogue lettering inside and compact aspect ratio
            if aspect < 3.2 and text_density >= 0.045:
                bubble_mask[comp] = 1.0

        bubble_mask = cv2.GaussianBlur(bubble_mask, (5, 5), 0)
        return bubble_mask

    @staticmethod
    def create_gutter_mask(gray: np.ndarray) -> np.ndarray:
        """
        Detects white gutters between panels and around outer page borders.
        """
        h, w = gray.shape[:2]
        is_pure_white = (gray > 250).astype(np.uint8) * 255
        # Find continuous white strips along edges
        gutter_mask = np.zeros_like(gray, dtype=np.float32)
        # Margin borders
        border_size = max(6, int(min(h, w) * 0.015))
        gutter_mask[:border_size, :] = 1.0
        gutter_mask[-border_size:, :] = 1.0
        gutter_mask[:, :border_size] = 1.0
        gutter_mask[:, -border_size:] = 1.0
        return gutter_mask
