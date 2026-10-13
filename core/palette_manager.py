import os
import json
from typing import Dict, List, Optional
import cv2
import numpy as np

COLOR_MEMORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
    "color_memory.json"
)

class PaletteManager:
    """
    Manages persistent color consistency across manga volumes and chapters.
    Maintains character palettes (hair, skin, eyes, clothing) and reference color locks.
    """
    def __init__(self, memory_path: str = COLOR_MEMORY_PATH):
        self.memory_path = memory_path
        self.data: Dict[str, any] = {
            "characters": {},
            "active_palette": [],
            "color_locks": {}
        }
        self.load()

    def load(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"[PaletteManager] Error loading {self.memory_path}: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[PaletteManager] Error saving {self.memory_path}: {e}")

    def extract_palette_from_image(self, image_bgr: np.ndarray, k: int = 8) -> List[str]:
        """
        Extracts dominant representative color hex codes from an image using k-means clustering.
        Filters out pure white, pure black, and extreme neutral grays.
        """
        h, w = image_bgr.shape[:2]
        # Resize for fast clustering
        small = cv2.resize(image_bgr, (min(w, 200), min(h, 200)), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, _, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        
        hex_colors = []
        for col in centers.astype(np.uint8):
            b, g, r = int(col[0]), int(col[1]), int(col[2])
            # Skip pure white/near white or pure black
            if (r > 245 and g > 245 and b > 245) or (r < 15 and g < 15 and b < 15):
                continue
            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            if hex_code not in hex_colors:
                hex_colors.append(hex_code)

        return hex_colors

    def set_character_palette(self, char_name: str, attributes: Dict[str, str]):
        """
        E.g. set_character_palette("Protagonist", {"hair": "#ffaa22", "skin": "#ffdfc4", "eyes": "#3366cc"})
        """
        self.data.setdefault("characters", {})[char_name] = attributes
        self.save()

    def get_characters(self) -> Dict[str, Dict[str, str]]:
        return self.data.get("characters", {})

    def remove_character(self, char_name: str):
        if char_name in self.data.get("characters", {}):
            del self.data["characters"][char_name]
            self.save()

    def set_active_palette(self, colors: List[str]):
        self.data["active_palette"] = colors
        self.save()

    def get_active_palette(self) -> List[str]:
        return self.data.get("active_palette", [])

    def set_color_lock(self, category: str, hex_color: str, enabled: bool = True):
        self.data.setdefault("color_locks", {})[category] = {
            "hex": hex_color,
            "enabled": enabled
        }
        self.save()

    def get_color_locks(self) -> Dict[str, Dict[str, any]]:
        return self.data.get("color_locks", {})
