"""
Inference engine for MangaColorizer.
Coordinates preprocessing, ONNX Runtime session execution, and postprocessing.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from .model_manager import ModelManager
from .postprocessing import postprocess_output
from .preprocessing import prepare_5channel_input


class InferenceEngine:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def colorize_image(
        self,
        image_input: Union[str, Image.Image],
        resolution_mode: str = "Auto",
        preserve_original_resolution: bool = True,
        transfer_mode: str = "luminance",
        hints: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Image.Image, float, Image.Image]:
        """
        Colorize a single manga page.

        Returns:
            colorized_image: PIL Image (RGB)
            inference_ms: inference latency in milliseconds
            original_gray: PIL Image of the original input
        """
        sess = self.model_manager.session
        if sess is None:
            sess = self.model_manager.load_session(prefer_directml=True)

        # 1. Preprocessing
        tensor_5ch, meta, original_gray = prepare_5channel_input(
            image_input=image_input,
            resolution_mode=resolution_mode,
            hints=hints,
        )

        # 2. Model Inference
        t0 = time.perf_counter()
        raw_output = sess.run(None, {"input": tensor_5ch})[0]
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Postprocessing
        colorized_image = postprocess_output(
            model_output=raw_output,
            meta=meta,
            original_gray=original_gray,
            preserve_original_resolution=preserve_original_resolution,
            transfer_mode=transfer_mode,
        )

        return colorized_image, inference_ms, original_gray
