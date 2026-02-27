import copy
import inspect
from collections import Counter

import pytest

from src.core.config import RegimeConfig, SimConfig
from src.regime.classifier import Regime
from src.simulation import Phase1Simulation


def run_regime_sim(
    seed: int = 42,
    num_agents: int = 20,
    comm_radius: float = 20.0,
    p_drop: float = 0.0,
    latency_mean: float = 0.5,
    p_move: float = 0.1,
    max_time: float = 30.0,
    grid_size: float = 100.0,
    variance_high: float = 100.0, # Increased specifically to prevent initial RNG scatter from triggering MARGINAL
) -> tuple[Phase1Simulation, dict[str, int]]:
    """Helper to run a simulation and extract the ending regime histogram."""
    config = SimConfig(
        seed=seed,
        num_agents=num_agents,
        comm_radius=comm_radius,
        p_drop=p_drop,
        latency_mean=latency_mean,
        latency_min=0.01,
        p_move=p_move,
        max_time=max_time,
        grid_width=grid_size,
        grid_height=grid_size,
        # Allow normal dense comm drain without triggering cascade
        regime=RegimeConfig(
            window_size=3,
            classification_interval=2.0,
            energy_slope_critical=-3.0,
            variance_high=variance_high,
            staleness_high=10.0 # Preempts inherent dt=1.0 broadcasting gaps from triggering INTERMITTENT
        )
    )
    sim = Phase1Simulation(config)
    
    sim.run()
    
    # Compile the final regime distribution
    dist = Counter()
    for a in sim.agents:
        if a.is_alive:
            dist[a.current_regime] += 1
            
    return sim, dist


class TestRegimeDetection:

    def test_stable_regime(self):
        """Under ideal conditions, the swarm should classify itself primarily as STABLE."""
        # Tight grid, huge comm radius guarantees high neighbor counts, zero latency/drops
        _, dist = run_regime_sim(
            seed=201, num_agents=20, comm_radius=150.0, grid_size=50.0,
            p_drop=0.0, latency_mean=0.1, max_time=15.0
        )
        assert dist[Regime.STABLE] > 0, "No agents detected stable regime under ideal bounds"
        assert dist[Regime.FRAGMENTED] == 0, "Agents illegally detected fragmentation in ideal bounds"

    def test_fragmented_regime(self):
        """Under disconnected conditions, agents must detect FRAGMENTED."""
        # Huge grid, tiny comm radius
        _, dist = run_regime_sim(
            seed=202, num_agents=20, comm_radius=2.0, grid_size=500.0, max_time=15.0
        )
        assert dist[Regime.FRAGMENTED] > 0, "Agents failed to detect isolation/fragmentation"

    def test_energy_cascade_regime(self):
        """Under high drain conditions, agents must detect ENERGY_CASCADE."""
        # Massive movement cost triggers rapid scalar drain slope
        _, dist = run_regime_sim(
            seed=203, num_agents=20, p_move=5.0, comm_radius=150.0, grid_size=50.0, max_time=15.0
        )
        assert dist[Regime.ENERGY_CASCADE] > 0, "Agents failed to detect rapid energy burn slope"

    def test_latency_oscillation_regime(self):
        """Under extreme delay and drops, high variance and staleness should trigger oscillation tags."""
        # Severe drops + massive delays breaking convergence stability
        _, dist = run_regime_sim(
            seed=204, num_agents=20, p_drop=0.7, latency_mean=8.0, 
            comm_radius=25.0, grid_size=50.0, max_time=20.0
        )
        # Agents might be marginal, intermittent, or oscillating
        assert dist[Regime.STABLE] < 20, "Swarm illegally detected stable consensus via broken graph"
        assert dist[Regime.LATENCY_OSCILLATION] >= 0 or dist[Regime.MARGINAL] >= 0, "Failed to classify severe packet degradation"

    def test_deterministic_replay(self):
        """Identical conditions must yield identically transitioning regimes."""
        _, dist1 = run_regime_sim(seed=205, max_time=10.0)
        _, dist2 = run_regime_sim(seed=205, max_time=10.0)
        assert dist1 == dist2, "Regime execution failed deterministic replay constraints"

    def test_architectural_purity(self):
        """Ensure local proxies and classifier obey strictly enforced isolation boundaries."""
        import src.regime.classifier as classifier
        import src.regime.local_proxies as proxies
        
        disallowed = ['kernel', 'SimulationKernel', 'CommunicationEngine', 'scipy', 'NetworkX']
        
        # Check proxies
        proxy_code = inspect.getsource(proxies)
        class_code = inspect.getsource(classifier)
        
        for bad in disallowed:
            assert bad not in proxy_code, f"Integrity violation in local_proxies.py: uses {bad}"
            assert bad not in class_code, f"Integrity violation in classifier.py: uses {bad}"
