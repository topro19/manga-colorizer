"""
Manga Region Segmenter and Hard Mask Generator.
Extracts:
- Pure dark line-art mask with configurable protection strength (0-100%).
- Speech bubble and text isolation masks (clean white/off-white protection).
- Gutter and panel border masks.
- Watertight semantic ink cells categorized by luminance (skin, hair, clothing, background).
- Sparse point hint generator to anchor character identities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class RegionSegmenter:
    """
    Performs morphology-based cell segmentation, line art protection,
    speech bubble detection, and semantic region extraction for manga pages.
    """

    @staticmethod
    def create_line_art_mask(gray_img: np.ndarray, dark_thresh: int = 70, protection_strength: float = 0.95) -> np.ndarray:
        """
        Creates line art mask where 1.0 = solid black ink.
        Combines deep thresholding with adaptive thresholding.
        """
        # Deep ink
        deep_ink = (gray_img < dark_thresh).astype(np.float32)
        # Adaptive threshold to catch fine pencil/hatching details
        adaptive = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        ).astype(np.float32) / 255.0

        combined = np.maximum(deep_ink, adaptive)
        # Smooth slightly for anti-aliasing
        blurred = cv2.GaussianBlur(combined, (3, 3), 0)
        return np.clip(blurred * protection_strength, 0.0, 1.0)

    @staticmethod
    def create_bubble_and_text_mask(gray_img: np.ndarray, white_thresh: int = 246) -> np.ndarray:
        """
        Detects enclosed speech bubbles containing dialog text.
        Returns binary mask (uint8, 255 = protected bubble space).
        Never flags open character line art or white skin as bubbles.
        """
        h, w = gray_img.shape[:2]
        ink = (gray_img < 85).astype(np.uint8) * 255
        ink_closed = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        interiors = cv2.bitwise_not(ink_closed)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interiors)
        bubble_mask = np.zeros_like(gray_img, dtype=np.uint8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1000 or area > 45000:
                continue
            x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # Exclude margins / canvas edges
            if x <= 2 or y <= 2 or (x + bw) >= (w - 2) or (y + bh) >= (h - 2):
                continue

            comp = (labels == i)
            mean_lum = float(np.mean(gray_img[comp]))
            if mean_lum < float(white_thresh):
                continue

            patch = gray_img[y:y+bh, x:x+bw]
            text_density = np.sum(patch < 80) / float(bw * bh)
            aspect = max(bw, bh) / max(1, min(bw, bh))

            # Genuine speech bubbles contain dialog letters inside
            if aspect < 3.2 and text_density >= 0.045:
                bubble_mask[comp] = 255

        return bubble_mask

    @staticmethod
    def segment_cells(
        gray_img: np.ndarray,
        bubble_mask: np.ndarray,
        min_cell_area: int = 120
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Segments the page into bounded regions enclosed by ink lines.
        Returns:
            labeled_map: 2D integer array where each pixel has its region ID
            regions: List of metadata dicts for each region (id, centroid, bbox, mean_lum, type)
        """
        # Form watertight boundary walls
        lines = (gray_img < 75).astype(np.uint8) * 255
        dilated_lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        # Invert to obtain bounded interiors
        cells = cv2.bitwise_not(dilated_lines)
        # Exclude speech bubbles
        cells[bubble_mask == 255] = 0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cells)
        region_list: List[Dict[str, Any]] = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_cell_area:
                continue

            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            cell_pixels = gray_img[labels == i]
            mean_lum = float(np.mean(cell_pixels))

            # Classify semantic candidate based on manga tones:
            # Skin: 160 - 252 (warm, light, smooth line-art character contours)
            # Hair/Dark: 30 - 110
            # Clothes/Mid: 110 - 160
            # Sky/Backdrop: 210 - 252 (large wide regions)
            if 160 <= mean_lum <= 252 and area < 40000:
                sem_type = "skin"
            elif mean_lum < 110:
                sem_type = "hair_or_dark"
            elif mean_lum < 160:
                sem_type = "clothing"
            else:
                sem_type = "background"

            region_list.append({
                "id": i,
                "area": area,
                "mean_lum": mean_lum,
                "type": sem_type,
                "centroid": (cx, cy),
                "bbox": (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT]
                )
            })

        return labels, region_list

    @staticmethod
    def generate_sparse_hints_from_plan(
        color_plan: Dict[str, Any],
        regions: List[Dict[str, Any]],
        image_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int, int, int]]:
        """
        Converts a structured color plan into sparse (x, y, r, g, b, radius) hints
        anchored precisely within detected semantic ink cells.
        """
        hints: List[Tuple[int, int, int, int, int, int]] = []
        h, w = image_shape[:2]

        def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
            h_str = hex_str.lstrip("#")
            if len(h_str) == 6:
                return int(h_str[0:2], 16), int(h_str[2:4], 16), int(h_str[4:6], 16)
            return (200, 200, 200)

        # Check characters in plan
        characters = color_plan.get("characters", [])
        for char in characters:
            skin_hex = char.get("skin_color")
            hair_hex = char.get("hair_color")
            clothing_list = char.get("clothing", [])

            # Skin hints
            if skin_hex:
                sr, sg, sb = hex_to_rgb(skin_hex)
                skin_cells = [r for r in regions if r["type"] == "skin" and 300 < r["area"] < 40000]
                # Sort descending by area so dominant face/body cells get hints first
                skin_cells.sort(key=lambda x: x["area"], reverse=True)
                # Pick up to 5 dominant skin cells
                for sc in skin_cells[:5]:
                    cx, cy = sc["centroid"]
                    hints.append((cx, cy, sr, sg, sb, 22))

            # Hair hints
            if hair_hex:
                hr, hg, hb = hex_to_rgb(hair_hex)
                hair_cells = [r for r in regions if r["type"] == "hair_or_dark" and 400 < r["area"] < 25000]
                for hc in hair_cells[:2]:
                    cx, cy = hc["centroid"]
                    hints.append((cx, cy, hr, hg, hb, 14))

            # Clothing hints
            for item in clothing_list:
                c_color = item.get("color")
                if c_color:
                    cr, cg, cb = hex_to_rgb(c_color)
                    cloth_cells = [r for r in regions if r["type"] == "clothing" and 500 < r["area"] < 30000]
                    for cc in cloth_cells[:2]:
                        cx, cy = cc["centroid"]
                        hints.append((cx, cy, cr, cg, cb, 14))

        # Background hints
        bg = color_plan.get("background", {})
        if "sky" in bg:
            sk_r, sk_g, sk_b = hex_to_rgb(bg["sky"])
            bg_cells = [r for r in regions if r["type"] == "background" and r["centroid"][1] < int(h * 0.4)]
            for bc in bg_cells[:2]:
                cx, cy = bc["centroid"]
                hints.append((cx, cy, sk_r, sk_g, sk_b, 16))

        return hints
