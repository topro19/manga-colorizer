"""
Image Preprocessing for MangaColorizer.
Prepares manga pages for the MangaColorization-v2 FP16 ONNX model:
- Aspect-ratio preserving scaling based on inference resolution settings
- Multiple-of-32 padding
- 5-channel tensor construction (Grayscale + Color Hints + Hint Mask)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from utils.image_utils import load_image_gray


def calculate_target_dims(
    orig_w: int,
    orig_h: int,
    resolution_mode: str = "Auto",
) -> Tuple[int, int]:
    """
    Calculate scaled dimensions preserving aspect ratio before 32-px padding.
    Modes:
      - 'Auto': if max(w, h) > 1536, scale down to 1536; else keep original size
      - '768', '1024', '1536': scale max dimension to this target
      - 'Original': keep original dimensions
    """
    if resolution_mode == "Original":
        return orig_w, orig_h

    target_max = None
    if resolution_mode == "Auto":
        max_dim = max(orig_w, orig_h)
        if max_dim > 1536:
            target_max = 1536
        else:
            return orig_w, orig_h
    else:
        try:
            target_max = int(resolution_mode)
        except ValueError:
            target_max = 1536

    max_dim = max(orig_w, orig_h)
    if max_dim <= target_max:
        return orig_w, orig_h

    scale = target_max / float(max_dim)
    scaled_w = max(32, int(round(orig_w * scale)))
    scaled_h = max(32, int(round(orig_h * scale)))
    return scaled_w, scaled_h


def prepare_5channel_input(
    image_input: Union[str, Image.Image],
    resolution_mode: str = "Auto",
    hints: Optional[List[Dict[str, Any]]] = None,
    reference_palette: Optional[List[Tuple[int, int, int]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Image.Image]:
    """
    Convert a manga page into the [1, 5, H, W] float32 tensor required by the model.

    Args:
        image_input: File path or PIL Image.
        resolution_mode: 'Auto', '768', '1024', '1536', 'Original'.
        hints: Optional list of hint dicts: [{'x': int, 'y': int, 'radius': int, 'color': (R, G, B)}].
               Coordinates are relative to original image dimensions.
        reference_palette: Optional global reference colors to inject as ambient seeds if no explicit hints exist.

    Returns:
        tensor: [1, 5, H_pad, W_pad] np.float32
        meta: dictionary with original_size, scaled_size, padded_size
        original_gray: PIL Image of the original unscaled grayscale page
    """
    original_gray = load_image_gray(image_input)
    orig_w, orig_h = original_gray.size

    # Step 1: Compute scaled dimensions preserving aspect ratio
    scaled_w, scaled_h = calculate_target_dims(orig_w, orig_h, resolution_mode)

    if (scaled_w, scaled_h) != (orig_w, orig_h):
        scaled_gray = original_gray.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    else:
        scaled_gray = original_gray

    # Step 2: Compute dimensions divisible by 32
    pad_w = (32 - (scaled_w % 32)) % 32
    pad_h = (32 - (scaled_h % 32)) % 32
    padded_w = scaled_w + pad_w
    padded_h = scaled_h + pad_h

    # Step 3: Create padded canvas (manga page margins are white = 255)
    padded_gray = Image.new("L", (padded_w, padded_h), color=255)
    padded_gray.paste(scaled_gray, (0, 0))

    # Channel 0: Grayscale normalized to [0.0, 1.0]
    ch0 = np.array(padded_gray, dtype=np.float32) / 255.0

    # Channels 1-3: Color hints ((c / 255.0 - 0.5) / 0.5)
    # Channel 4: Hint mask (1.0 where hint exists, 0.0 elsewhere)
    ch1_3 = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    ch4 = np.zeros((1, padded_h, padded_w), dtype=np.float32)

    # Scale factor from original coordinates to scaled coordinates
    scale_x = scaled_w / float(orig_w)
    scale_y = scaled_h / float(orig_h)

    # Render manual color hints
    if hints:
        # Create an RGB image and mask image to draw hints
        hint_canvas = Image.new("RGB", (padded_w, padded_h), color=(0, 0, 0))
        mask_canvas = Image.new("L", (padded_w, padded_h), color=0)
        draw_hint = ImageDraw.Draw(hint_canvas)
        draw_mask = ImageDraw.Draw(mask_canvas)

        for h in hints:
            hx = int(round(h["x"] * scale_x))
            hy = int(round(h["y"] * scale_y))
            rad = max(1, int(round(h.get("radius", 6) * min(scale_x, scale_y))))
            color = tuple(h["color"])

            bbox = [hx - rad, hy - rad, hx + rad, hy + rad]
            draw_hint.ellipse(bbox, fill=color)
            draw_mask.ellipse(bbox, fill=255)

        hint_arr = np.array(hint_canvas, dtype=np.float32) / 255.0  # [H, W, 3]
        mask_arr = np.array(mask_canvas, dtype=np.float32) / 255.0  # [H, W]

        # Normalization: (c - 0.5) / 0.5
        norm_hints = (hint_arr - 0.5) / 0.5  # in [-1, 1]
        norm_hints = np.transpose(norm_hints, (2, 0, 1))  # [3, H, W]

        # Apply to channels
        active_mask = (mask_arr > 0.1)
        for c in range(3):
            ch1_3[c] = np.where(active_mask, norm_hints[c], 0.0)
        ch4[0] = np.where(active_mask, 1.0, 0.0)

    # Step 4: Assemble 5-channel tensor [1, 5, H, W]
    ch0_expanded = ch0[np.newaxis, :, :]  # [1, H, W]
    tensor_5ch = np.concatenate([ch0_expanded, ch1_3, ch4], axis=0)  # [5, H, W]
    tensor_5ch = np.expand_dims(tensor_5ch, axis=0).astype(np.float32)  # [1, 5, H, W]

    meta = {
        "original_size": (orig_w, orig_h),
        "scaled_size": (scaled_w, scaled_h),
        "padded_size": (padded_w, padded_h),
        "pad_w": pad_w,
        "pad_h": pad_h,
        "resolution_mode": resolution_mode,
    }

    return tensor_5ch, meta, original_gray
