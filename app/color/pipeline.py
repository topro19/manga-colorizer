"""
Chapter-level orchestration pipeline.
Coordinates:
Chapter Ingestion -> Page Analysis -> Panel Segmentation -> Scene Grouping
-> Character Memory -> Color Planning -> DirectML Colorization
-> Protection/Masking -> Consistency Pass -> Exact Reassembly.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QThread

from app.ingestion.chapter import ChapterExtractor
from app.analysis.panel_detector import PanelDetector, PanelBox
from app.scenes.scene_grouper import SceneGrouper, SceneGroup
from app.characters.character_memory import CharacterMemoryManager
from app.color.color_plan import ColorPlanGenerator, ColorPlan
from app.color.consistency import ConsistencyEngine
from app.models.backend import ColorizationBackend
from app.protection.masks import ProtectionMasks
from app.reassembly.page_reassembler import PageReassembler
from app.storage.project import ProjectState, ProjectManager
from app.utils.logging import logger
from app.utils.image import load_image_bgr, save_image_bgr


class ChapterPipelineWorker(QThread):
    """
    Asynchronous chapter processing worker thread with pausing and cancellation.
    """
    page_progress = Signal(int, int, str)  # cur_page, total_pages, page_name
    scene_detected = Signal(str, int)     # scene_id, panel_count
    status_updated = Signal(str)          # user-facing status message
    page_finished = Signal(int, str, np.ndarray, np.ndarray)  # page_idx, name, orig_bgr, color_bgr
    chapter_finished = Signal(int, str)   # total_completed, output_dir
    error_occurred = Signal(str)

    def __init__(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        backend: ColorizationBackend,
        char_memory: CharacterMemoryManager,
        color_instructions: str = "",
        quality_mode: str = "BALANCED",
        reference_palette: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        self.source_path = Path(source_path)
        self.output_dir = Path(output_dir)
        self.backend = backend
        self.char_memory = char_memory
        self.color_instructions = color_instructions
        self.quality_mode = quality_mode
        self.reference_palette = reference_palette

        self._is_paused = False
        self._is_cancelled = False

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            self.status_updated.emit("Ingesting chapter pages...")
            cache_dir = self.output_dir / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            page_paths = ChapterExtractor.extract_source(self.source_path, cache_dir)
            total_pages = len(page_paths)
            if total_pages == 0:
                self.error_occurred.emit("No valid manga pages found in input source.")
                return

            self.status_updated.emit(f"Analyzing {total_pages} pages for panels & scene flow...")

            # 1. Page & Panel Analysis across whole chapter
            all_panels: List[PanelBox] = []
            page_data: List[Dict[str, Any]] = []

            for idx, p_path in enumerate(page_paths, start=1):
                if self._is_cancelled:
                    return
                bgr = load_image_bgr(p_path)
                panels = PanelDetector.detect_panels(bgr, page_idx=idx)
                all_panels.extend(panels)
                page_data.append({
                    "idx": idx,
                    "path": p_path,
                    "bgr": bgr,
                    "panels": panels
                })

            # 2. Scene Continuity Grouping
            scenes = SceneGrouper.group_panels_into_scenes(all_panels)
            self.status_updated.emit(f"Discovered {len(scenes)} scenes across {len(all_panels)} panels.")

            # 3. Process pages in chapter context
            for p_info in page_data:
                while self._is_paused:
                    time.sleep(0.1)
                if self._is_cancelled:
                    return

                idx = p_info["idx"]
                orig_bgr = p_info["bgr"]
                page_name = p_info["path"].name
                self.page_progress.emit(idx, total_pages, page_name)

                # Determine active scene
                first_panel_id = p_info["panels"][0].panel_id if p_info["panels"] else ""
                active_scene = scenes[0]
                for sc in scenes:
                    if any(p.panel_id == first_panel_id for p in sc.panels):
                        active_scene = sc
                        break

                self.scene_detected.emit(active_scene.scene_id, len(active_scene.panels))

                # Color Plan synthesis
                color_plan = ColorPlanGenerator.create_plan(
                    scene=active_scene,
                    char_memory=self.char_memory,
                    user_prompt=self.color_instructions,
                    reference_palette=self.reference_palette
                )

                # Panel-level colorization
                colorized_panels = []
                is_vivid = (self.quality_mode == "VIVID")
                color_str = 0.95 if is_vivid else 0.85

                for panel in p_info["panels"]:
                    if panel.crop_bgr is None:
                        continue
                    colored_crop = self.backend.colorize(
                        bgr_image=panel.crop_bgr,
                        color_strength=color_str,
                        line_preservation=0.90,
                        speech_bubble_protection=True,
                        options={
                            "vivid_mode": is_vivid,
                            "prompt": self.color_instructions or "masterpiece, highly detailed anime illustration, vibrant colors"
                        }
                    )
                    colorized_panels.append((panel, colored_crop))

                # Reassemble exact original dimensions
                reassembled_bgr = PageReassembler.reassemble_page(
                    original_bgr=orig_bgr,
                    colorized_panels=colorized_panels,
                    color_strength=color_str,
                    line_preservation=0.90,
                    protect_bubbles=True,
                    vivid_mode=is_vivid
                )

                # Scene consistency harmony
                if color_plan.environment:
                    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
                    ink_mask = ProtectionMasks.create_ink_mask(gray)
                    bubble_mask = ProtectionMasks.create_speech_bubble_mask(gray)
                    prot_mask = np.maximum(ink_mask, bubble_mask)
                    reassembled_bgr = ConsistencyEngine.enforce_scene_harmony(
                        page_bgr=reassembled_bgr,
                        target_palette=color_plan.environment,
                        protection_mask=prot_mask,
                        harmony_strength=0.20
                    )

                # Save finalized page
                out_path = self.output_dir / f"color_{page_name}"
                save_image_bgr(out_path, reassembled_bgr)

                self.page_finished.emit(idx, page_name, orig_bgr, reassembled_bgr)

            self.chapter_finished.emit(total_pages, str(self.output_dir))

        except Exception as e:
            logger.exception("Error in chapter pipeline execution")
            self.error_occurred.emit(str(e))
