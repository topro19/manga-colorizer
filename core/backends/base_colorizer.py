from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class BaseColorizer(ABC):
    """
    Abstract base class for all colorization backends.
    Allows hot-swapping between models (MangaColorize-v2, DDColor, Reference-Guided, etc.)
    and seamless benchmarking inside Model Lab.
    """
    def __init__(self, model_name: str, model_path: str, provider: str = "DmlExecutionProvider"):
        self.model_name = model_name
        self.model_path = model_path
        self.provider = provider

    @abstractmethod
    def load(self) -> bool:
        """Loads the model into ONNX Runtime session."""
        pass

    @abstractmethod
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
        """
        Colorizes the input BGR manga image.
        Returns the final composited BGR image preserving original line art and luminance.
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns model metadata, target domain, input requirement, and parameter size."""
        pass
