"""
Unified Color Plan generator and consistency enforcement.
Converts user natural instructions and reference images into structured color plans.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from app.characters.character_memory import CharacterMemoryManager, CharacterProfile
from app.scenes.scene_grouper import SceneGroup


@dataclass
class ColorPlan:
    scene_id: str
    lighting: str = "natural daylight"
    characters: Dict[str, Dict[str, str]] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    saturation: str = "medium"
    contrast: str = "medium"


class ColorPlanGenerator:
    """
    Synthesizes structured color plans from user prompt instructions,
    character memory, and scene context.
    """

    @staticmethod
    def create_plan(
        scene: SceneGroup,
        char_memory: CharacterMemoryManager,
        user_prompt: str = "",
        reference_palette: Optional[Dict[str, str]] = None
    ) -> ColorPlan:
        plan = ColorPlan(scene_id=scene.scene_id)
        
        # Determine lighting from scene luminance
        if scene.average_luminance < 120:
            plan.lighting = "dim dramatic night"
        elif scene.average_luminance > 220:
            plan.lighting = "bright high-key daylight"
        else:
            plan.lighting = "balanced manga interior"

        # Apply character profiles from persistent memory
        for char in char_memory.get_active_characters():
            plan.characters[char.character_id] = {
                "hair": char.hair_color,
                "skin": char.skin_tone,
                "shirt": char.shirt_color,
                "jacket": char.jacket_color
            }

        # Parse user instructions if provided (e.g. 'black hair, blue jacket, sunset')
        prompt_lower = user_prompt.lower()
        if "sunset" in prompt_lower or "golden" in prompt_lower:
            plan.lighting = "warm sunset"
            plan.environment["sky"] = "#F89B68"
        elif "night" in prompt_lower or "dark" in prompt_lower:
            plan.lighting = "cool night"
            plan.environment["sky"] = "#1E293B"

        if "blue jacket" in prompt_lower:
            for cid in plan.characters:
                plan.characters[cid]["jacket"] = "#2563EB"
        if "red hair" in prompt_lower:
            for cid in plan.characters:
                plan.characters[cid]["hair"] = "#B91C1C"

        if reference_palette:
            plan.environment.update(reference_palette)

        return plan
