"""
scenario_validator.py — Validates ScenarioConfig
"""

from __future__ import annotations

from typing import List

from src.scenario.scenario_model import ScenarioConfig


class ScenarioValidationError(Exception):
    pass


class ScenarioValidator:
    """Validates parameters of a ScenarioConfig."""

    @staticmethod
    def validate(config: ScenarioConfig) -> List[str]:
        """Returns a list of error messages. Empty list means valid."""
        errors: List[str] = []

        # Name
        if not config.name or not config.name.strip():
            errors.append("Scenario name cannot be empty.")

        # Seed
        if config.seed < 0:
            errors.append("Seed must be a non-negative integer.")

        # Num agents
        if config.num_agents <= 0 or config.num_agents > 500:
            errors.append("Number of agents must be between 1 and 500.")

        # Comm radius
        if config.communication_radius <= 0:
            errors.append("Communication radius must be > 0.")
            
        # Energy
        if config.energy_params.initial_energy < 0:
            errors.append("Initial energy cannot be negative.")
        if config.energy_params.drain_rate < 0:
            errors.append("Drain rate cannot be negative.")

        # Interference
        if not (0.0 <= config.interference.intensity <= 1.0):
            errors.append("Interference intensity must be between 0.0 and 1.0.")
            
        for idx, zone in enumerate(config.interference.spatial_zones):
            if zone.radius <= 0:
                errors.append(f"Spatial zone {idx} radius must be > 0.")
            if not (0.0 <= zone.intensity <= 1.0):
                errors.append(f"Spatial zone {idx} intensity must be between 0.0 and 1.0.")

        # Tasks
        if config.tasks.count < 0:
            errors.append("Task count cannot be negative.")
        if config.tasks.distribution not in ["uniform", "clustered"]:
            errors.append("Task distribution must be 'uniform' or 'clustered'.")

        # Simulation
        if config.simulation.duration <= 0:
            errors.append("Simulation duration must be > 0.")
        if config.simulation.dt <= 0:
            errors.append("Simulation dt must be > 0.")

        return errors

    @staticmethod
    def check_performance_safety(config: ScenarioConfig) -> List[str]:
        """Returns a list of warnings (not strict errors)."""
        warnings = []
        if config.num_agents > 150:
            warnings.append(f"High agent count ({config.num_agents}). Performance may degrade.")
        if config.simulation.duration > 1000:
            warnings.append(f"Long simulation duration ({config.simulation.duration}s) will generate large export files.")
        return warnings
