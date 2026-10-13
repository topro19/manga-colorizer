import os
import time
from typing import Dict, Any, Optional
import cv2
import numpy as np
import onnxruntime as ort

from .base_colorizer import BaseColorizer
from ..filters import composite_luminance_and_chroma

class MangaColorizationV2Backend(BaseColorizer):
    """
    Colorization backend for Faridzar/manga-colorization-v2-onnx (qweasdd generator).
    Trained specifically on manga line art.
    Input: [1, 5, H, W] float32 (Grayscale, Hint R, G, B, Hint Mask)
    Output: [1, 3, H, W] float32 in [-1, 1], scaled by (fake_color * 0.5 + 0.5)
    """
    def __init__(self, model_path: str, provider: str = "DmlExecutionProvider"):
        super().__init__("MangaColorization-v2", model_path, provider)
        self.session = None

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        
        available = ort.get_available_providers()
        prov_list = [self.provider] if self.provider in available else ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in prov_list:
            prov_list.append("CPUExecutionProvider")
            
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = True
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(self.model_path, opts, providers=prov_list)
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "MangaColorization-v2 (FP16)",
            "domain": "Manga Line Art & Screentones",
            "weights": "58.8 MB ONNX",
            "features": "DirectML, Interactive Hinting, Divisible by 32",
            "optimal_res": "576 - 768 px"
        }

    def colorize(
        self,
        bgr_image: np.ndarray,
        color_strength: float = 0.65,
        line_preservation: float = 0.90,
        speech_bubble_protection: bool = True,
        use_guided_filter: bool = True,
        hint_points: Optional[list] = None,
        inference_resolution: str = "Auto",
        options: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        if self.session is None and not self.load():
            raise RuntimeError(f"Could not load model at {self.model_path}")

        h_orig, w_orig = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Determine target resolution based on optimal manga generator sizing
        if inference_resolution == "576":
            base_size = 576
        elif inference_resolution == "768":
            base_size = 768
        elif inference_resolution == "1024":
            base_size = 1024
        else: # Auto
            base_size = 576

        if h_orig < w_orig:
            ratio = h_orig / (base_size * 1.5)
            w_target = int(np.ceil(w_orig / ratio))
            h_target = int(base_size * 1.5)
        else:
            ratio = w_orig / base_size
            h_target = int(np.ceil(h_orig / ratio))
            w_target = base_size

        resized_gray = cv2.resize(gray, (w_target, h_target), interpolation=cv2.INTER_AREA)

        # Pad to multiples of 32
        pad_h = (32 - h_target % 32) % 32
        pad_w = (32 - w_target % 32) % 32
        padded = np.pad(resized_gray, ((0, pad_h), (0, pad_w)), mode='edge')
        cur_h, cur_w = padded.shape

        # Channel 0: Grayscale [0, 1]
        # Channel 1..3: Normalized Color Hints ((c - 0.5) / 0.5)
        # Channel 4: Hint Mask [0 or 1]
        inp_tensor = np.zeros((1, 5, cur_h, cur_w), dtype=np.float32)
        inp_tensor[0, 0] = padded

        # Apply interactive color hints if provided
        if hint_points:
            scale_y = h_target / float(h_orig)
            scale_x = w_target / float(w_orig)
            for pt in hint_points:
                x, y, r, g, b, radius = pt
                px = int(x * scale_x)
                py = int(y * scale_y)
                rad = max(2, int(radius * scale_x))
                
                # Draw hint circle onto hint channels
                cv2.circle(inp_tensor[0, 1], (px, py), rad, float((r / 255.0 - 0.5) / 0.5), -1)
                cv2.circle(inp_tensor[0, 2], (px, py), rad, float((g / 255.0 - 0.5) / 0.5), -1)
                cv2.circle(inp_tensor[0, 3], (px, py), rad, float((b / 255.0 - 0.5) / 0.5), -1)
                cv2.circle(inp_tensor[0, 4], (px, py), rad, 1.0, -1)

        # If no hints were provided, auto-detect prominent subject/character ink cells
        # and anchor subtle skin/warm tones so pure white manga art gets naturally colored
        if not hint_points:
            auto_hints = []
            scale_y = h_target / float(h_orig)
            scale_x = w_target / float(w_orig)
            ink = (padded < 0.33).astype(np.uint8) * 255
            ink_closed = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            interiors = cv2.bitwise_not(ink_closed)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(interiors)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 800 or area > (cur_h * cur_w * 0.20):
                    continue
                x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                if x <= 2 or y <= 2 or (x + bw) >= (cur_w - 2) or (y + bh) >= (cur_h - 2):
                    continue
                patch = padded[y:y+bh, x:x+bw]
                text_density = np.sum(patch < 0.33) / float(bw * bh)
                mean_lum = float(np.mean(patch))
                # Skip speech bubbles (contain text dialogue)
                if text_density >= 0.045 and mean_lum > 0.95:
                    continue
                # Character cells: place subtle skin tone anchor
                if mean_lum > 0.70:
                    cx, cy = int(centroids[i][0]), int(centroids[i][1])
                    auto_hints.append((cx, cy, 248, 208, 180, 18))
                    if len(auto_hints) >= 4:
                        break

            for pt in auto_hints:
                px, py, r, g, b, rad = pt
                cv2.circle(inp_tensor[0, 1], (px, py), rad, float((r / 255.0 - 0.5) / 0.5), -1)
                cv2.circle(inp_tensor[0, 2], (px, py), rad, float((g / 255.0 - 0.5) / 0.5), -1)
                cv2.circle(inp_tensor[0, 3], (px, py), rad, float((b / 255.0 - 0.5) / 0.5), -1)
                cv2.circle(inp_tensor[0, 4], (px, py), rad, 1.0, -1)

        input_name = self.session.get_inputs()[0].name
        raw_output = self.session.run(None, {input_name: inp_tensor})[0]

        # Model output is [1, 3, H, W]
        raw_rgb = raw_output[0].transpose(1, 2, 0)
        # Adaptively scale based on dynamic range: if values are already in [0, 1], do not compress with *0.5+0.5
        if raw_output.min() < -0.1:
            colorized = raw_rgb * 0.5 + 0.5
        else:
            colorized = raw_rgb
        colorized = np.clip(colorized, 0.0, 1.0)

        # Unpad
        if pad_h > 0:
            colorized = colorized[:-pad_h, :]
        if pad_w > 0:
            colorized = colorized[:, :-pad_w]

        # Project color back to original manga dimensions
        colorized_full = cv2.resize(colorized, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        colorized_bgr = (colorized_full[:, :, ::-1] * 255.0).astype(np.uint8)

        # Pass through the anti-bleed line art & speech bubble compositing pipeline
        final_bgr = composite_luminance_and_chroma(
            original_bgr=bgr_image,
            colorized_bgr=colorized_bgr,
            color_strength=color_strength,
            line_preservation=line_preservation,
            speech_bubble_protection=speech_bubble_protection,
            use_guided_filter=use_guided_filter
        )

        return final_bgr
