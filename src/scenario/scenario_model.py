"""
scenario_model.py — Schema for Scenario Configuration (Phase 2P)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from src.core.config import SimConfig


@dataclass
class EnergyParams:
    initial_energy: float = 100.0
    drain_rate: float = 0.001


@dataclass
class SpatialZone:
    center_x: float = 50.0
    center_y: float = 50.0
    radius: float = 20.0
    intensity: float = 0.5


@dataclass
class InterferenceParams:
    enabled: bool = False
    intensity: float = 0.3
    spatial_zones: list[SpatialZone] = field(default_factory=list)


@dataclass
class TaskParams:
    count: int = 0
    distribution: str = "uniform"  # "uniform", "clustered"


@dataclass
class SimulationParams:
    duration: float = 200.0
    dt: float = 1.0


@dataclass
class ScenarioConfig:
    name: str = "custom_scenario"
    seed: int = 42
    num_agents: int = 50
    communication_radius: float = 20.0
    
    energy_params: EnergyParams = field(default_factory=EnergyParams)
    interference: InterferenceParams = field(default_factory=InterferenceParams)
    tasks: TaskParams = field(default_factory=TaskParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to flat standard dictionary."""
        return {
            "name": self.name,
            "seed": self.seed,
            "num_agents": self.num_agents,
            "communication_radius": self.communication_radius,
            "energy_params": {
                "initial_energy": self.energy_params.initial_energy,
                "drain_rate": self.energy_params.drain_rate,
            },
            "interference": {
                "enabled": self.interference.enabled,
                "intensity": self.interference.intensity,
                "spatial_zones": [
                    {
                        "center_x": z.center_x,
                        "center_y": z.center_y,
                        "radius": z.radius,
                        "intensity": z.intensity,
                    } for z in self.interference.spatial_zones
                ],
            },
            "tasks": {
                "count": self.tasks.count,
                "distribution": self.tasks.distribution,
            },
            "simulation": {
                "duration": self.simulation.duration,
                "dt": self.simulation.dt,
            }
        }

    def to_sim_config(self, base_cfg: SimConfig) -> SimConfig:
        """Create a new SimConfig built on base_cfg but overriding with scenario parameters."""
        import dataclasses
        d = dataclasses.asdict(base_cfg)
        d["num_agents"] = self.num_agents
        d["seed"] = self.seed
        d["comm_radius"] = self.communication_radius
        d["energy_initial"] = self.energy_params.initial_energy
        d["p_idle"] = self.energy_params.drain_rate
        d["max_time"] = self.simulation.duration
        d["dt"] = self.simulation.dt
        
        # Interference logic maps roughly to p_drop scaling in Phase 2
        # Without deeper kernel hacks, we'll map intensity to a baseline global packet drop
        # or rely on SimulationWorker APIs. We'll set psi_max.
        if self.interference.enabled:
            d["psi_max"] = self.interference.intensity
            d["p_drop"] = max(base_cfg.p_drop, self.interference.intensity * 0.5)
            
        return SimConfig(**d)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> ScenarioConfig:
        ep = d.get("energy_params", {})
        inf = d.get("interference", {})
        sz = inf.get("spatial_zones", [])
        zones = [SpatialZone(**z) for z in sz]
        tp = d.get("tasks", {})
        sp = d.get("simulation", {})
        
        return ScenarioConfig(
            name=d.get("name", "custom_scenario"),
            seed=int(d.get("seed", 42)),
            num_agents=int(d.get("num_agents", 50)),
            communication_radius=float(d.get("communication_radius", 20.0)),
            energy_params=EnergyParams(
                initial_energy=float(ep.get("initial_energy", 100.0)),
                drain_rate=float(ep.get("drain_rate", 0.001)),
            ),
            interference=InterferenceParams(
                enabled=bool(inf.get("enabled", False)),
                intensity=float(inf.get("intensity", 0.3)),
                spatial_zones=zones,
            ),
            tasks=TaskParams(
                count=int(tp.get("count", 0)),
                distribution=str(tp.get("distribution", "uniform")),
            ),
            simulation=SimulationParams(
                duration=float(sp.get("duration", 200.0)),
                dt=float(sp.get("dt", 1.0)),
            )
        )
