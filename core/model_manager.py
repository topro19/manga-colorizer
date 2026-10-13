"""
Model Manager for MangaColorizer.
Handles automatic model download, ONNX Runtime session instantiation,
DirectML provider priority, and GPU diagnostic testing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import requests

from utils.gpu_monitor import get_active_provider_info, get_windows_gpu_info

DEFAULT_MODEL_URL = "https://huggingface.co/Faridzar/manga-colorization-v2-onnx/resolve/main/manga-colorize-fp16.onnx"
DEFAULT_MODEL_NAME = "manga-colorize-fp16.onnx"


class ModelManager:
    def __init__(self, models_dir: Optional[str | Path] = None):
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            # Default to <project_root>/models
            self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / DEFAULT_MODEL_NAME

        self.session: Optional[ort.InferenceSession] = None
        self.active_provider: Optional[str] = None
        self.available_providers: List[str] = ort.get_available_providers()
        self.gpu_info = get_windows_gpu_info()

    def is_model_downloaded(self) -> bool:
        """Check if model ONNX file exists and is greater than 10MB."""
        return self.model_path.is_file() and self.model_path.stat().st_size > 10 * 1024 * 1024

    def download_model(
        self,
        url: str = DEFAULT_MODEL_URL,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Path:
        """
        Download model file from Hugging Face with chunked streaming and progress reporting.
        progress_callback(downloaded_bytes, total_bytes, percent)
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.model_path.with_suffix(".download.tmp")

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB chunks

        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        percent = (downloaded / total_size) * 100.0
                        progress_callback(downloaded, total_size, percent)

        temp_file.replace(self.model_path)
        return self.model_path

    def load_session(
        self,
        prefer_directml: bool = True,
        device_id: int = 0,
    ) -> ort.InferenceSession:
        """
        Load ONNX Runtime inference session with DirectML provider preference.
        Falls back to CPUExecutionProvider if DirectML fails.
        """
        if not self.is_model_downloaded():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please download it first.")

        providers_to_try: List[Any] = []
        if prefer_directml and "DmlExecutionProvider" in self.available_providers:
            providers_to_try.append(("DmlExecutionProvider", {"device_id": device_id}))

        # Always include CPU fallback
        providers_to_try.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers_to_try,
            )
            current_providers = self.session.get_providers()
            self.active_provider = current_providers[0] if current_providers else "Unknown"
        except Exception as e:
            # Fallback to pure CPU
            print(f"[ModelManager] DirectML session init failed: {e}. Falling back to CPUExecutionProvider.")
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.active_provider = "CPUExecutionProvider"

        return self.session

    def get_status_labels(self) -> Tuple[str, str, bool]:
        """
        Returns:
            (badge_text, detailed_label, is_gpu_active)
        """
        providers = self.session.get_providers() if self.session else self.available_providers
        return get_active_provider_info(providers)

    def test_gpu(self) -> Dict[str, Any]:
        """
        Execute a diagnostic inference test on a small tensor and report speed, provider, and status.
        """
        result = {
            "success": False,
            "provider": "None",
            "device_name": self.gpu_info.get("name", "Unknown GPU"),
            "vram": self.gpu_info.get("vram_gb", "Unknown"),
            "inference_time_ms": 0.0,
            "message": "",
        }

        try:
            if not self.is_model_downloaded():
                result["message"] = "Model file not found. Download it before running the GPU test."
                return result

            # Create or reuse session
            sess = self.session
            if sess is None:
                sess = self.load_session(prefer_directml=True)

            providers = sess.get_providers()
            active = providers[0] if providers else "Unknown"
            result["provider"] = active

            # Run benchmark on 256x256 tensor
            dummy_input = np.zeros((1, 5, 256, 256), dtype=np.float32)

            # Warmup
            sess.run(None, {"input": dummy_input})

            # Timed run
            t0 = time.perf_counter()
            for _ in range(3):
                sess.run(None, {"input": dummy_input})
            elapsed = (time.perf_counter() - t0) / 3.0
            ms = elapsed * 1000.0

            result["inference_time_ms"] = round(ms, 1)
            result["success"] = True

            if "DmlExecutionProvider" in active:
                result["message"] = (
                    f"DirectML test: PASSED\n"
                    f"Provider: DmlExecutionProvider\n"
                    f"GPU: {result['device_name']} ({result['vram']} VRAM)\n"
                    f"Test Latency: {result['inference_time_ms']} ms"
                )
            else:
                result["message"] = (
                    f"DirectML test: FALLBACK TO CPU\n"
                    f"Provider: {active}\n"
                    f"Test Latency: {result['inference_time_ms']} ms\n"
                    f"Note: DmlExecutionProvider was not selected."
                )
        except Exception as e:
            result["success"] = False
            result["message"] = f"DirectML test: FAILED\nReason: {str(e)}"

        return result
