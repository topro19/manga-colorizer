"""
Postprocessing for MangaColorizer.
Transforms raw model output into final RGB images, crops out multiple-of-32 padding,
and restores high-resolution original line art via luminance transfer.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image

from utils.image_utils import apply_luminance_preservation


def postprocess_output(
    model_output: np.ndarray,
    meta: Dict[str, Any],
    original_gray: Image.Image,
    preserve_original_resolution: bool = True,
    transfer_mode: str = "luminance",
) -> Image.Image:
    """
    Process raw ONNX output:
    - [1, 3, H, W] float32 in [0, 1]
    - Squeeze batch and transpose to [H, W, 3]
    - Clip and convert to uint8 RGB
    - Crop away padding
    - Optionally restore original resolution with crisp line-art luminance blending.
    """
    # model_output is [1, 3, H, W]
    out_rgb = np.squeeze(model_output, axis=0)  # [3, H, W]
    out_rgb = np.transpose(out_rgb, (1, 2, 0))  # [H, W, 3]

    # Clip to [0.0, 1.0] and scale to 255
    out_rgb = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    # Crop out the multiple-of-32 padding
    scaled_w, scaled_h = meta["scaled_size"]
    cropped_rgb = out_rgb[:scaled_h, :scaled_w]

    colorized_img = Image.fromarray(cropped_rgb, mode="RGB")
    orig_w, orig_h = meta["original_size"]

    # Resolution restoration
    if preserve_original_resolution and (scaled_w, scaled_h) != (orig_w, orig_h):
        if transfer_mode == "luminance":
            # Best quality: blends original high-res ink lines with upscaled colors
            return apply_luminance_preservation(original_gray, colorized_img)
        else:
            # Direct Lanczos resize
            return colorized_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    elif preserve_original_resolution and (scaled_w, scaled_h) == (orig_w, orig_h):
        if transfer_mode == "luminance":
            return apply_luminance_preservation(original_gray, colorized_img)

    return colorized_img
