import os
from typing import Dict, List, Optional

from .base_colorizer import BaseColorizer
from .manga_colorization_v2 import MangaColorizationV2Backend
from .ddcolor_colorizer import DDColorBackend
from .reference_guided import ReferenceGuidedBackend

class ColorizerRegistry:
    """
    Central registry to query, list, and instantiate available colorization backends.
    Powers the Model Lab and backend selector in the UI.
    """
    def __init__(self, models_dir: str, provider: str = "DmlExecutionProvider"):
        self.models_dir = models_dir
        self.provider = provider
        self.backends: Dict[str, BaseColorizer] = {}
        self._init_backends()

    def _init_backends(self):
        manga_v2_path = os.path.join(self.models_dir, "manga-colorize-fp16.onnx")
        if os.path.exists(manga_v2_path):
            v2_backend = MangaColorizationV2Backend(manga_v2_path, self.provider)
            self.backends["MangaColorization-v2"] = v2_backend
            # Wrap with Reference Guided backend
            self.backends["Reference-Guided Engine"] = ReferenceGuidedBackend(v2_backend)

        ddcolor_path = os.path.join(self.models_dir, "ddcolor-fp16.onnx")
        if os.path.exists(ddcolor_path):
            self.backends["DDColor-FP16 (Chroma)"] = DDColorBackend(ddcolor_path, self.provider)

    def list_available_models(self) -> List[str]:
        return list(self.backends.keys())

    def get_backend(self, name: str) -> Optional[BaseColorizer]:
        return self.backends.get(name)

    def get_all_backends(self) -> Dict[str, BaseColorizer]:
        return self.backends
