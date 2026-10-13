import os
import time
from typing import Dict, Any, Optional
import cv2
import numpy as np
import onnxruntime as ort

from .base_colorizer import BaseColorizer
from ..filters import composite_luminance_and_chroma

class DDColorBackend(BaseColorizer):
    """
    Colorization backend for DDColor FP16 ONNX (Dual-Decoder Colorization).
    Trained for semantic scene coloring (natural skin tones, fabrics, foliage, architecture).
    Input: [1, 3, 256, 256] float32 RGB
    Output: [1, 2, 256, 256] float32 CIE Lab AB channels in [-128, 127]
    Directly predicts chrominance without degrading original luminance!
    """
    def __init__(self, model_path: str, provider: str = "DmlExecutionProvider"):
        super().__init__("DDColor-FP16", model_path, provider)
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
        self.session = ort.InferenceSession(self.model_path, opts, providers=prov_list)
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "DDColor (FP16 Native Chroma)",
            "domain": "Semantic Photoreal & Anime Colorization",
            "weights": "108 MB ONNX",
            "features": "Native CIE Lab AB Chroma, Ultra-fast (15ms DirectML)",
            "optimal_res": "256x256 Feature Encoder"
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
        
        # Prepare input: RGB [0, 1] resized to 256x256
        rgb_norm = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb_norm, (256, 256), interpolation=cv2.INTER_AREA)
        inp_tensor = resized.transpose(2, 0, 1)[np.newaxis, ...] # [1, 3, 256, 256]

        input_name = self.session.get_inputs()[0].name
        raw_output = self.session.run(None, {input_name: inp_tensor})[0]

        # Model output is [1, 2, 256, 256] Lab AB channels centered around 0
        ab_pred = raw_output[0].transpose(1, 2, 0)
        # Resize AB to original manga resolution
        ab_orig = cv2.resize(ab_pred, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

        # Original L in OpenCV format [0, 255]
        orig_lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_orig = orig_lab[:, :, 0]

        # OpenCV Lab AB are stored offset by +128 in [0, 255]
        A_cv = np.clip(ab_orig[:, :, 0] + 128.0, 0, 255)
        B_cv = np.clip(ab_orig[:, :, 1] + 128.0, 0, 255)

        # In OpenCV Lab, when L=255, all colors map to white regardless of A and B.
        # So we supply L_for_raw where bright colored regions have L around 220-235:
        ab_dist = np.sqrt(ab_orig[:, :, 0]**2 + ab_orig[:, :, 1]**2)
        bright_chroma = np.clip((L_orig - 200.0) / 55.0, 0.0, 1.0) * np.clip(ab_dist / 6.0, 0.0, 1.0)
        L_for_raw = L_orig * (1.0 - bright_chroma * 0.25)

        raw_lab = np.stack([L_for_raw, A_cv, B_cv], axis=2).astype(np.uint8)
        raw_color_bgr = cv2.cvtColor(raw_lab, cv2.COLOR_LAB2BGR)

        # Composite with guided filter, line art protection, and bubble mask
        final_bgr = composite_luminance_and_chroma(
            original_bgr=bgr_image,
            colorized_bgr=raw_color_bgr,
            color_strength=color_strength,
            line_preservation=line_preservation,
            speech_bubble_protection=speech_bubble_protection,
            use_guided_filter=use_guided_filter
        )

        return final_bgr
