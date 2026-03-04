import numpy as np
import pytest

from src.metrics.connectivity_metrics import (
    compute_largest_connected_component,
    classify_connectivity_phase,
    ConnectivityPhase
)

def test_fully_connected():
    """Tightly clustered agents should form a single connected component."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 5, size=(10, 2))
    comm_radius = 20.0
    
    lcc, count, ratio = compute_largest_connected_component(positions, comm_radius)
    assert lcc == 10
    assert count == 1
    assert ratio == 1.0
    assert classify_connectivity_phase(ratio) == ConnectivityPhase.CONNECTED

def test_fragmented():
    """Agents spaced widely apart should yield isolated subgraphs."""
    positions = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [0.0, 100.0],
        [100.0, 100.0]
    ])
    comm_radius = 10.0
    
    lcc, count, ratio = compute_largest_connected_component(positions, comm_radius)
    
    # 4 components of size 1 each
    assert lcc == 1
    assert count == 4
    assert ratio == 0.25
    assert classify_connectivity_phase(ratio) == ConnectivityPhase.FRAGMENTED

def test_deterministic_replay():
    """Execution with identical seeds must yield identical connectivity time series."""
    from src.core.config import SimConfig
    from src.simulation import Phase1Simulation
    from src.environment.interference_field import InterferenceField, FieldMode
    
    def run_sim(seed: int) -> list[tuple]:
        config = SimConfig(num_agents=10, grid_width=50, grid_height=50, comm_radius=15.0, seed=seed, max_time=5.0)
        sim = Phase1Simulation(config)
        # Suppress arbitrary interference variation for purity
        sim.interference = InterferenceField(mode=FieldMode.CONSTANT, psi_max=config.psi_max)
        sim.run()
        return sim.connectivity_log
        
    log1 = run_sim(42)
    log2 = run_sim(42)
    
    assert len(log1) > 0
    assert log1 == log2
    
def test_empty_graph():
    """Should cleanly handle 0 agents."""
    positions = np.empty((0, 2))
    lcc, count, ratio = compute_largest_connected_component(positions, 10.0)
    assert lcc == 0
    assert count == 0
    assert ratio == 0.0
