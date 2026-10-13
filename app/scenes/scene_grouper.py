"""
Scene continuity and sequential panel grouper.
Groups panels and pages by environmental and narrative continuity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
from app.analysis.panel_detector import PanelBox


@dataclass
class SceneGroup:
    scene_id: str
    panels: List[PanelBox] = field(default_factory=list)
    environment_palette: Dict[str, str] = field(default_factory=dict)
    average_luminance: float = 200.0


class SceneGrouper:
    """
    Groups panels across consecutive pages based on visual brightness and spatial proximity.
    """

    @staticmethod
    def group_panels_into_scenes(all_panels: List[PanelBox]) -> List[SceneGroup]:
        if not all_panels:
            return []

        scenes: List[SceneGroup] = []
        cur_panels: List[PanelBox] = []
        cur_lum_history: List[float] = []

        for p in all_panels:
            if not cur_panels:
                cur_panels.append(p)
                cur_lum_history.append(p.mean_lum)
                continue

            prev_lum = cur_lum_history[-1]
            # If large sudden shift in average panel lighting (> 60), trigger a scene boundary
            if abs(p.mean_lum - prev_lum) > 65.0 or len(cur_panels) >= 12:
                scene_num = len(scenes) + 1
                scenes.append(
                    SceneGroup(
                        scene_id=f"scene_{scene_num:02d}",
                        panels=list(cur_panels),
                        average_luminance=float(np.mean(cur_lum_history))
                    )
                )
                cur_panels = [p]
                cur_lum_history = [p.mean_lum]
            else:
                cur_panels.append(p)
                cur_lum_history.append(p.mean_lum)

        if cur_panels:
            scene_num = len(scenes) + 1
            scenes.append(
                SceneGroup(
                    scene_id=f"scene_{scene_num:02d}",
                    panels=list(cur_panels),
                    average_luminance=float(np.mean(cur_lum_history))
                )
            )

        return scenes
