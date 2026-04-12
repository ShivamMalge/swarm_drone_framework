"""
scenario_loader.py — YAML save/load for ScenarioConfig
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from src.scenario.scenario_model import ScenarioConfig


class ScenarioLoader:
    """Handles YAML serialization of ScenarioConfig."""

    def __init__(self, directory: str | Path = "scenarios"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_scenario(self, config: ScenarioConfig, filename: str | None = None) -> Path:
        if not filename:
            filename = f"{config.name}.yaml"
            
        path = self.directory / filename
        
        # Convert to dict and dump
        d = config.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
            
        return path

    def load_scenario(self, filename: str) -> ScenarioConfig:
        path = self.directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
            
        return ScenarioConfig.from_dict(d)

    def list_scenarios(self) -> List[str]:
        """List available scenario YAML files."""
        return sorted([f.name for f in self.directory.glob("*.yaml")])
