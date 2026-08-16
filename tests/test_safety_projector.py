"""
Tests for Algorithm 1 (Bounded Heuristic Clamping).

These tests were dead for an unknown period: `project_to_theta_safe` gained a
required `theta_nominal` parameter and every test here kept calling the old
two-argument signature, so all four failed with TypeError and the framework's
headline algorithm had zero passing coverage (audit F-08).

The two-stage tests at the bottom exist because the manuscript's Algorithm 1
pseudocode documents only stage 2 (the bisection), while stage 1 (the static
box clamp) is what actually produces the surviving experimental result: it
raises coverage_gain from a proposed 0.0 to its lower bound 0.5, which keeps an
isolated agent inside the Voronoi coverage law rather than dropping it into the
random-walk fallback (audit F-04, F-11).
"""

from __future__ import annotations

import pytest

from src.adaptation.hybrid_supervisor import HybridSupervisor, Strategy
from src.adaptation.safety_projector import THETA_SAFE_BOUNDS, project_to_theta_safe


def _nominal() -> dict[str, float]:
    """The nominal parameter vector the projector bisects back toward."""
    return HybridSupervisor().propose_parameters(Strategy.NORMAL_OPERATION, 0.05)


# ── Stage 1: static box clamp ────────────────────────────────────────────

def test_safety_projector_clips_values() -> None:
    # Intentionally extreme values strictly outside safe domain boundaries
    proposed = {
        "coverage_gain": 0.1,          # Too low (bound 0.5)
        "gossip_epsilon": 0.1,         # Too high (bound 0.05)
        "broadcast_rate": 2.0,         # Too high (bound 1.5)
        "auction_participation": -0.5, # Too low (bound 0.0)
        "velocity_scale": 1.9,         # Too high (bound 1.5)
    }

    safe, count = project_to_theta_safe(proposed, theta_nominal=_nominal())

    # Assure identical clamping occurrences matches boundary breaks
    assert count == 5

    assert safe["coverage_gain"] == THETA_SAFE_BOUNDS["coverage_gain"][0]
    assert safe["gossip_epsilon"] == THETA_SAFE_BOUNDS["gossip_epsilon"][1]
    assert safe["broadcast_rate"] == THETA_SAFE_BOUNDS["broadcast_rate"][1]
    assert safe["auction_participation"] == THETA_SAFE_BOUNDS["auction_participation"][0]
    assert safe["velocity_scale"] == THETA_SAFE_BOUNDS["velocity_scale"][1]


def test_safety_projector_passes_valid_values() -> None:
    # Within bounds (Should incur 0 projection events)
    valid = {
        "coverage_gain": 1.0,
        "gossip_epsilon": 0.03,
        "broadcast_rate": 1.0,
        "auction_participation": 0.5,
        "velocity_scale": 1.0,
    }

    safe, count = project_to_theta_safe(valid, theta_nominal=_nominal())

    assert count == 0
    for k in valid:
        assert safe[k] == valid[k]


def test_safety_projector_handles_unknown_keys() -> None:
    proposed = {
        "coverage_gain": 3.0,
        "custom_metric": 100.0,
    }

    safe, count = project_to_theta_safe(proposed, theta_nominal=_nominal())
    assert count == 1
    assert safe["coverage_gain"] == 2.0
    # Keys without a declared bound pass through untouched.
    assert safe["custom_metric"] == 100.0


def test_safety_projector_determinism() -> None:
    # Validates stateless mapping output deterministic equality
    proposed = {"velocity_scale": -1.0}

    r1, _ = project_to_theta_safe(proposed, theta_nominal=_nominal())
    r2, _ = project_to_theta_safe(proposed, theta_nominal=_nominal())

    assert r1 == r2


# ── The two-stage contract (what Algorithm 1 must document) ──────────────

def test_box_clamp_raises_coverage_gain_from_zero_to_lower_bound() -> None:
    """
    Stage 1 is load-bearing: CONNECTIVITY_RECOVERY and ENERGY_CONSERVATION both
    propose coverage_gain = 0.0, and the box clamp raises it to 0.5. That is
    what keeps a fragmented agent in the coverage law instead of the random-walk
    fallback, and is the mechanism behind the reported energy result.
    """
    supervisor = HybridSupervisor()
    for strategy in (Strategy.CONNECTIVITY_RECOVERY, Strategy.ENERGY_CONSERVATION):
        proposed = supervisor.propose_parameters(strategy, 0.05)
        assert proposed["coverage_gain"] == 0.0, "supervisor no longer proposes 0.0"

        safe, _ = project_to_theta_safe(proposed, theta_nominal=_nominal())

        assert safe["coverage_gain"] == 0.5
        assert safe["coverage_gain"] == THETA_SAFE_BOUNDS["coverage_gain"][0]
        # The coverage branch in AgentCore.compute_velocity is gated on > 0.05.
        assert safe["coverage_gain"] > 0.05, (
            "clamped coverage_gain must keep the agent in the Voronoi coverage law"
        )


def test_velocity_scale_zero_is_NOT_clamped() -> None:
    """
    Counterpart to the above, and a documented source of confusion: a code
    comment in hybrid_supervisor claims velocity_scale = 0.0 is "Clamped to 0.5".
    It is not -- the lower bound is 0.0, so it passes through untouched.
    """
    proposed = HybridSupervisor().propose_parameters(Strategy.ENERGY_CONSERVATION, 0.05)
    assert proposed["velocity_scale"] == 0.0

    safe, _ = project_to_theta_safe(proposed, theta_nominal=_nominal())
    assert safe["velocity_scale"] == 0.0
    assert THETA_SAFE_BOUNDS["velocity_scale"][0] == 0.0


def test_bisection_operates_on_the_box_clamped_value() -> None:
    """
    Stage 2 runs after stage 1, on the clamped value, not on the raw proposal.

    gossip_epsilon is proposed at 0.10, box-clamped to 0.05, and only then
    tested against the dynamic bound. With a bound below 0.05 the bisection
    fires and returns a value strictly under it.
    """
    proposed = {"gossip_epsilon": 0.10}
    nominal = _nominal()                       # gossip_epsilon = 0.05
    bound = 0.01                               # tighter than both

    safe, count = project_to_theta_safe(proposed, theta_nominal=nominal,
                                        dynamic_bounds={"gossip_epsilon": bound})

    assert safe["gossip_epsilon"] < bound, "bisection must return a value under the bound"
    # One event for the box clamp, one for the bisection.
    assert count == 2

    # The search starts from the CLAMPED upper value (0.05), not the raw 0.10.
    # Nominal 0.05 also violates the bound, so low is reset to bound*0.99 and
    # the 5-step search converges just below it.
    assert safe["gossip_epsilon"] == pytest.approx(bound * 0.99, rel=1e-9)


def test_bisection_does_not_fire_when_value_is_already_safe() -> None:
    """A parameter already under the dynamic bound is left alone by stage 2."""
    proposed = {"gossip_epsilon": 0.02}
    safe, count = project_to_theta_safe(proposed, theta_nominal=_nominal(),
                                        dynamic_bounds={"gossip_epsilon": 0.5})
    assert safe["gossip_epsilon"] == 0.02
    assert count == 0
