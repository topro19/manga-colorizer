"""
Project file serialization (.mcp) and session persistence.
Allows saving work-in-progress chapters, completed pages, character memories,
and resuming seamlessly.
"""

from __future__ import annotations
import json
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.utils.logging import logger


@dataclass
class ProjectState:
    chapter_name: str
    source_path: str
    output_dir: str
    total_pages: int = 0
    completed_pages: List[int] = field(default_factory=list)
    current_page: int = 1
    color_instructions: str = ""
    characters: Dict[str, Any] = field(default_factory=dict)
    palettes: Dict[str, Any] = field(default_factory=dict)
    quality_mode: str = "BALANCED"


class ProjectManager:
    """Manages creation, loading, and saving of .mcp project packages."""

    @staticmethod
    def save_project(state: ProjectState, target_path: str | Path) -> None:
        target = Path(target_path)
        if target.suffix.lower() != ".mcp":
            target = target.with_suffix(".mcp")

        data = asdict(state)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved project state to {target}")

    @staticmethod
    def load_project(project_path: str | Path) -> ProjectState:
        target = Path(project_path)
        if not target.is_file():
            raise FileNotFoundError(f"Project file not found: {target}")

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            return ProjectState(**data)
