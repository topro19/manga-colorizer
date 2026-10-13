"""
Persistent Character Memory and semantic identity tracking.
Maintains consistent palettes across pages and chapters (character_memory.json).
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.utils.logging import logger


@dataclass
class CharacterProfile:
    character_id: str
    name: str = "unknown"
    hair_color: str = "#2B2A33"
    eye_color: str = "#4A78C2"
    skin_tone: str = "#F5D0B5"
    shirt_color: str = "#3A506B"
    jacket_color: str = "#1C2541"
    palette_confidence: float = 0.90
    reference_regions: List[Dict[str, Any]] = field(default_factory=list)
    appearances: List[str] = field(default_factory=list)  # panel or page ids


class CharacterMemoryManager:
    """
    Loads, tracks, and persists character identities to character_memory.json.
    Prevents random hair/clothing color changes across manga chapters.
    """

    def __init__(self, memory_file: str | Path = "character_memory.json"):
        self.memory_file = Path(memory_file)
        self.profiles: Dict[str, CharacterProfile] = {}
        self.load()

    def load(self) -> None:
        if self.memory_file.is_file():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for cid, pdata in data.items():
                        self.profiles[cid] = CharacterProfile(**pdata)
                logger.info(f"Loaded {len(self.profiles)} character profiles from {self.memory_file}")
            except Exception as e:
                logger.error(f"Error loading character memory: {e}")
        else:
            # Seed with default protagonist profile
            self.profiles["char_protagonist"] = CharacterProfile(
                character_id="char_protagonist",
                name="Protagonist",
                hair_color="#1D2430",
                eye_color="#3B6FA5",
                skin_tone="#F2C7A5",
                shirt_color="#D93636",
                jacket_color="#233B8B",
                palette_confidence=0.92
            )

    def save(self) -> None:
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as f:
                data = {cid: asdict(prof) for cid, prof in self.profiles.items()}
                json.dump(data, f, indent=2)
            logger.debug(f"Saved character memory to {self.memory_file}")
        except Exception as e:
            logger.error(f"Failed to save character memory: {e}")

    def get_profile(self, character_id: str) -> Optional[CharacterProfile]:
        return self.profiles.get(character_id)

    def update_profile(self, profile: CharacterProfile) -> None:
        self.profiles[profile.character_id] = profile
        self.save()

    def get_active_characters(self) -> List[CharacterProfile]:
        return list(self.profiles.values())
