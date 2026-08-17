"""
Regression pins: this project's audit history, encoded as permanent tests.

Each test pins one historical defect that was found by measurement and fixed.
They are deliberately cheap, and each carries the incident it guards against.
The companion runtime pin for the oracle channel lives in
test_no_global_access.py::test_oracle_channel_closed_when_disabled.
"""

import numpy as np
import pytest

from src.core.config import SimConfig
from src.simulation import Phase1Simulation


# ── F-10 / MS-14: one bound, one owner ───────────────────────────────────

def test_gossip_applies_epsilon_verbatim():
    """
    compute_gossip_update must apply exactly the epsilon it is given.

    Incident: an internal clamp here recomputed the projector's stability
    bound (with a DIFFERENT tau discretisation, F-24) and won on 98.5% of
    calls, silently overriding Algorithm 1 -- the paper's stated adaptation
    mechanism -- and making two experimental arms bit-identical (F-02).
    An unsafe epsilon must propagate, not be hidden: if it arrives here
    unsafe, the defect is upstream in the projector.
    """
    from src.coordination.gossip_consensus import compute_gossip_update

    own, nbrs = 0.0, [10.0, 10.0, 10.0, 10.0]
    # A deliberately unstable epsilon for degree 4: the old clamp would have
    # capped it at 0.99 / (4 * (tau+1)) <= 0.2475.
    eps = 0.9
    got = compute_gossip_update(own, nbrs, [0.0] * 4, 1.0, eps)
    assert got == pytest.approx(own + eps * 40.0), (
        "epsilon was not applied verbatim -- a hidden clamp is back"
    )


# ── The 1e154 overflow: monitors must survive what they observe ──────────

def test_variance_proxy_saturates_instead_of_crashing():
    """
    Incident: with the duplicate clamp removed, the Unconstrained arm's
    consensus genuinely diverges; one seed passed 1e154, at which point
    statistics.variance's exact-fraction arithmetic raised OverflowError and
    killed the entire 50-seed suite. The divergence is real behaviour that
    must stay observable; the observer dying on it was the defect.
    """
    from src.regime.local_proxies import (
        VARIANCE_PROXY_CEILING,
        compute_local_consensus_variance,
    )

    class _Nb:
        def __init__(self, v):
            self.consensus_state = v

    class _Map:
        def __init__(self, vals):
            self._v = [_Nb(v) for v in vals]

        def get_all_neighbors(self):
            return self._v

    class _Agent:
        def __init__(self, own, nbrs):
            self.consensus_state = own
            self.local_map = _Map(nbrs)

    # The exact scale that crashed the suite, and beyond.
    for scale in (1e154, 1e200, float("inf")):
        var = compute_local_consensus_variance(_Agent(scale, [-scale]))
        assert np.isfinite(var)
        assert var == VARIANCE_PROXY_CEILING


# ── F-39: interference is a fraction ─────────────────────────────────────

def test_interference_clamped_to_unit_interval():
    """
    Incident: the percolation experiment ramped psi_max to 2.55 ("255%
    jamming") with no clamp; callers use (1 - psi), so survival probability
    went negative and 62% of Figure 2's x-axis was a saturated blackout
    presented as a jamming gradient.
    """
    from src.environment.interference_field import FieldMode, InterferenceField

    f = InterferenceField(mode=FieldMode.CONSTANT, psi_max=2.55)
    assert f.evaluate(np.zeros(2), 0.0) == 1.0
    f.psi_max = -0.3
    assert f.evaluate(np.zeros(2), 0.0) == 0.0
    f.psi_max = 0.4
    assert f.evaluate(np.zeros(2), 0.0) == pytest.approx(0.4)


# ── F-17: observing a simulation must not change it ──────────────────────

def test_metrics_logging_does_not_perturb_trajectories():
    """
    Incident: the metrics logger called compute_velocity() for reporting;
    that method's random-walk branch draws from the agent RNG, so LOGGED runs
    diverged from unlogged runs -- "deterministic reproducibility" was false
    whenever a kernel logger was attached.

    'stability' test_mode is the clean comparison: identical config, logging
    on vs off. ('thermodynamics' also randomises initial energy, so it is not
    a pure logging toggle.)
    """
    base = dict(seed=5, num_agents=20, max_time=100.0, coverage_enabled=True,
                log_dir="logs")
    a = Phase1Simulation(SimConfig(test_mode=None, **base))
    a.run()
    b = Phase1Simulation(SimConfig(test_mode="stability", **base))
    b.run()
    b.close_loggers()

    pa = np.array([x._position for x in a.agents])
    pb = np.array([x._position for x in b.agents])
    ea = np.array([x.energy for x in a.agents])
    eb = np.array([x.energy for x in b.agents])
    assert np.array_equal(pa, pb), "logging changed agent trajectories"
    assert np.array_equal(ea, eb), "logging changed agent energies"


# ── F-03: connectivity is a property of the living ───────────────────────

def test_extinct_swarm_reports_zero_connectivity():
    """
    Incident: lambda_2 was computed over ALL agent positions, dead included.
    A fully extinct swarm logged lambda_2 = 0.29 with a largest connected
    component of 100 -- Table III's connectivity column measured the
    geometric arrangement of corpses.

    Fixture note: the halt fixed point (F-04) lets isolated agents survive on
    idle drain alone -- with default p_idle=0.001, 3.0 energy lasts 3000
    ticks, and a first version of this fixture left 3 halted survivors.
    p_idle=0.1 makes even a fully halted agent expire within 30 ticks, which
    guarantees the extinct-swarm case this pin exists to exercise.
    """
    sim = Phase1Simulation(SimConfig(
        seed=7, num_agents=10, grid_width=50.0, grid_height=50.0,
        energy_initial=3.0, p_idle=0.1, coverage_enabled=True, max_time=60.0,
    ))
    sim.run()
    assert sim.alive_mask.sum() == 0, "fixture must produce total swarm death"
    last = sim.connectivity_log[-1]
    assert last["spectral_gap"] == 0.0
    assert last["lcc"] == 0


# ── F-12: beliefs must expire ────────────────────────────────────────────

def test_belief_age_bounded_by_eviction_horizon():
    """
    Incident: LocalMap never evicted. Dead agents persisted in neighbours'
    maps at frozen positions for the rest of the run (max observed belief age
    1925 ticks against a 30-tick horizon), inflating tau_max, distorting
    regime timing, and steering the coverage law toward corpses. A secondary
    placement bug then let broadcast-thinned agents skip eviction (ages
    drifted to 35-37); eviction now runs before the thinning checks.
    """
    cfg = SimConfig(seed=7, num_agents=20, grid_width=100.0, grid_height=100.0,
                    coverage_enabled=True, max_time=150.0)
    sim = Phase1Simulation(cfg)

    worst = 0.0
    orig = Phase1Simulation._handle_metrics_log

    def probe(self, event):
        nonlocal worst
        t = event.timestamp
        for a in self.agents:
            if a.is_alive:
                for nb in a._local_map.get_all_neighbors():
                    worst = max(worst, t - nb.timestamp)
        orig(self, event)

    sim._handle_metrics_log = probe.__get__(sim)
    sim.kernel.register_handler(
        __import__("src.core.event", fromlist=["EventType"]).EventType.METRICS_LOG,
        sim._handle_metrics_log,
    )
    sim.run()
    assert worst <= cfg.belief_max_age + 1e-9, (
        f"belief age {worst} exceeded the eviction horizon {cfg.belief_max_age}"
    )
