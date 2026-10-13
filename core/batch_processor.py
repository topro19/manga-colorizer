"""
Production Batch Processor for MangaColorizer.
Coordinates multi-page manga colorization volumes (Folder, CBZ, PDF) with:
- Guided Pipeline integration (VLM color plan + sparse hints + DirectML inference + LAB preservation)
- Manifest logging (project_manifest.json)
- Resilient Resume/Restart support (skips completed pages without recomputation)
- Per-stage latency calculation (IO, VLM, Inference, Postprocessing)
- Non-blocking asynchronous QThread execution with Pause, Resume, Stop controls.
"""

from __future__ import annotations

import gc
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QObject, QThread, QWaitCondition, Signal

from utils.cbz import create_cbz, extract_cbz
from utils.image_utils import get_image_files
from utils.pdf import create_pdf, extract_pdf_pages

from .backends.base_colorizer import BaseColorizer
from .guided_pipeline import GuidedColorizationPipeline, PipelineExecutionResult
from .palette_manager import PaletteManager
from .vlm_director import BaseVLMDirector


class BatchProcessor(QThread):
    # Signals for UI integration
    page_started = Signal(int, int, str)
    page_completed = Signal(
        int, int, str, bytes, bytes, bytes, bytes, float, dict, str
    )  # current, total, name, orig, raw, guided, final, ms, timings, out_path
    page_failed = Signal(int, int, str, str)
    progress_updated = Signal(int, int, float, float, str, str, str)  # cur, tot, pct, ppm, eta, elapsed, stage_info
    batch_finished = Signal(int, int, str, str)
    status_message = Signal(str)

    def __init__(
        self,
        colorizer: BaseColorizer,
        vlm_director: BaseVLMDirector,
        palette_manager: PaletteManager,
        input_source: str | Path,
        source_type: str,
        output_dir: str | Path,
        output_format: str = "PNG",
        jpeg_quality: int = 95,
        pipeline_mode: str = "BALANCED",  # 'FAST', 'BALANCED', 'QUALITY'
        color_strength: float = 0.80,
        l_preservation: float = 0.95,
        line_protection: float = 0.95,
        protect_bubbles: bool = True,
        use_guided_filter: bool = True,
        reprocess_existing: bool = False,
        create_cbz_output: bool = False,
        create_pdf_output: bool = False,
        page_hints: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.pipeline = GuidedColorizationPipeline(
            colorizer=colorizer,
            vlm_director=vlm_director,
            palette_manager=palette_manager
        )
        self.input_source = Path(input_source)
        self.source_type = source_type.lower()
        self.output_dir = Path(output_dir)
        self.output_format = output_format.upper()
        self.jpeg_quality = max(50, min(100, jpeg_quality))
        self.pipeline_mode = pipeline_mode
        self.color_strength = color_strength
        self.l_preservation = l_preservation
        self.line_protection = line_protection
        self.protect_bubbles = protect_bubbles
        self.use_guided_filter = use_guided_filter
        self.reprocess_existing = reprocess_existing
        self.create_cbz_output = create_cbz_output
        self.create_pdf_output = create_pdf_output
        self.page_hints = page_hints or {}

        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()
        self._is_paused = False
        self._is_stopped = False
        self._temp_dir: Optional[Path] = None

    def pause(self) -> None:
        self._mutex.lock()
        self._is_paused = True
        self._mutex.unlock()
        self.status_message.emit("Colorization paused.")

    def resume(self) -> None:
        self._mutex.lock()
        self._is_paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()
        self.status_message.emit("Colorization resumed.")

    def stop(self) -> None:
        self._mutex.lock()
        self._is_stopped = True
        self._is_paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()
        self.status_message.emit("Colorization stopping...")

    def _check_pause_or_stop(self) -> bool:
        self._mutex.lock()
        if self._is_stopped:
            self._mutex.unlock()
            return True
        while self._is_paused:
            self._pause_cond.wait(self._mutex)
            if self._is_stopped:
                self._mutex.unlock()
                return True
        self._mutex.unlock()
        return False

    def _read_manifest(self) -> Dict[str, Any]:
        manifest_path = self.output_dir / "project_manifest.json"
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"project": str(self.input_source), "pages": {}}

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        manifest_path = self.output_dir / "project_manifest.json"
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"[BatchProcessor] Manifest write error: {e}")

    def run(self) -> None:
        start_time = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        images_to_process: List[Path] = []

        try:
            # 1. Unpack input if needed
            if self.source_type == "cbz":
                self.status_message.emit(f"Extracting CBZ archive: {self.input_source.name}...")
                self._temp_dir = Path(tempfile.mkdtemp(prefix="mangacolor_cbz_"))
                images_to_process = extract_cbz(self.input_source, self._temp_dir)
            elif self.source_type == "pdf":
                self.status_message.emit(f"Extracting PDF pages: {self.input_source.name}...")
                self._temp_dir = Path(tempfile.mkdtemp(prefix="mangacolor_pdf_"))
                images_to_process = extract_pdf_pages(self.input_source, self._temp_dir)
            else:
                images_to_process = get_image_files(self.input_source, recursive=False)

            total_pages = len(images_to_process)
            if total_pages == 0:
                self.status_message.emit("No valid manga pages found to process.")
                self.batch_finished.emit(0, 0, "00:00:00", str(self.output_dir))
                return

            self.status_message.emit(
                f"Discovered {total_pages} page(s). Running mode: [{self.pipeline_mode}] with {self.pipeline.colorizer.model_name}..."
            )

            manifest = self._read_manifest()
            processed_count = 0
            failed_count = 0
            completed_output_paths: List[Path] = []

            for idx, img_path in enumerate(images_to_process):
                if self._check_pause_or_stop():
                    self.status_message.emit("Batch stopped by user.")
                    break

                page_num = idx + 1
                page_name = img_path.name
                base_stem = img_path.stem
                out_ext = ".jpg" if self.output_format == "JPEG" else ".png"
                out_file = self.output_dir / f"{base_stem}{out_ext}"

                # 2. Check resume condition via manifest + disk file
                page_record = manifest["pages"].get(page_name)
                is_already_done = (
                    not self.reprocess_existing
                    and page_record
                    and page_record.get("status") == "complete"
                    and out_file.is_file()
                )

                if is_already_done:
                    self.status_message.emit(f"Page {page_num}/{total_pages}: {page_name} already completed (resumed).")
                    completed_output_paths.append(out_file)
                    processed_count += 1
                    continue

                self.page_started.emit(page_num, total_pages, page_name)

                # 3. Process page through Guided Pipeline
                try:
                    hints = self.page_hints.get(page_name) or self.page_hints.get("*")
                    result: PipelineExecutionResult = self.pipeline.process_page(
                        image_path=str(img_path),
                        mode=self.pipeline_mode,
                        color_strength=self.color_strength,
                        l_preservation=self.l_preservation,
                        line_protection=self.line_protection,
                        protect_bubbles=self.protect_bubbles,
                        use_guided_filter=self.use_guided_filter,
                        manual_hints=hints
                    )

                    # Save to output file
                    if self.output_format == "JPEG":
                        cv2.imwrite(str(out_file), result.final_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    else:
                        cv2.imwrite(str(out_file), result.final_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                    completed_output_paths.append(out_file)
                    processed_count += 1

                    # Update manifest
                    manifest["pages"][page_name] = {
                        "status": "complete",
                        "timestamp": datetime.now().isoformat(),
                        "output_file": out_file.name,
                        "model": self.pipeline.colorizer.model_name,
                        "mode": self.pipeline_mode,
                        "processing_time": round(result.timings_ms.get("total_ms", 0.0) / 1000.0, 2),
                        "timings_ms": result.timings_ms,
                    }
                    self._write_manifest(manifest)

                    # Encode preview buffers for 4-layer comparison
                    orig_bgr = cv2.imread(str(img_path))
                    _, buf_orig = cv2.imencode(".png", orig_bgr)
                    _, buf_raw = cv2.imencode(".png", result.raw_model_bgr)
                    _, buf_guided = cv2.imencode(".png", result.l_preserved_bgr)
                    _, buf_final = cv2.imencode(".png", result.final_bgr)

                    self.page_completed.emit(
                        page_num,
                        total_pages,
                        page_name,
                        buf_orig.tobytes(),
                        buf_raw.tobytes(),
                        buf_guided.tobytes(),
                        buf_final.tobytes(),
                        result.timings_ms.get("total_ms", 0.0),
                        result.timings_ms,
                        str(out_file)
                    )

                except Exception as e:
                    failed_count += 1
                    err_msg = str(e)
                    print(f"[BatchProcessor] Error on page {page_num} ({page_name}): {err_msg}")
                    manifest["pages"][page_name] = {
                        "status": "failed",
                        "error": err_msg,
                        "timestamp": datetime.now().isoformat()
                    }
                    self._write_manifest(manifest)
                    self.page_failed.emit(page_num, total_pages, page_name, err_msg)
                    continue
                finally:
                    gc.collect()

                # Timing, ETA, Speed
                elapsed_sec = time.perf_counter() - start_time
                done_in_session = processed_count + failed_count
                if done_in_session > 0:
                    pages_per_sec = done_in_session / elapsed_sec
                    pages_per_min = pages_per_sec * 60.0
                    rem_pages = total_pages - page_num
                    eta_sec = rem_pages / pages_per_sec if pages_per_sec > 0 else 0
                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
                else:
                    pages_per_min = 0.0
                    eta_str = "--"

                elapsed_str = f"{int(elapsed_sec // 3600):02d}:{int((elapsed_sec % 3600) // 60):02d}:{int(elapsed_sec % 60):02d}"
                percent = (page_num / float(total_pages)) * 100.0
                stage_info = (
                    f"VLM: {result.timings_ms.get('vlm_plan_ms', 0):.0f}ms | "
                    f"GPU: {result.timings_ms.get('inference_ms', 0):.0f}ms | "
                    f"Post: {result.timings_ms.get('postprocessing_ms', 0):.0f}ms"
                )
                self.progress_updated.emit(page_num, total_pages, percent, pages_per_min, eta_str, elapsed_str, stage_info)

            # 4. Packaging
            final_path = str(self.output_dir)
            if self.create_cbz_output and completed_output_paths:
                cbz_path = self.output_dir / f"{self.input_source.stem}_colorized.cbz"
                self.status_message.emit("Assembling colorized CBZ archive...")
                create_cbz(completed_output_paths, cbz_path)
                final_path = str(cbz_path)

            if self.create_pdf_output and completed_output_paths:
                pdf_path = self.output_dir / f"{self.input_source.stem}_colorized.pdf"
                self.status_message.emit("Exporting colorized PDF document...")
                create_pdf(completed_output_paths, pdf_path)
                final_path = str(pdf_path)

            total_elapsed = time.perf_counter() - start_time
            total_elapsed_str = f"{int(total_elapsed // 3600):02d}:{int((total_elapsed % 3600) // 60):02d}:{int(total_elapsed % 60):02d}"
            self.batch_finished.emit(processed_count, failed_count, total_elapsed_str, final_path)

        finally:
            if self._temp_dir and self._temp_dir.is_dir():
                try:
                    shutil.rmtree(self._temp_dir, ignore_errors=True)
                except Exception:
                    pass
