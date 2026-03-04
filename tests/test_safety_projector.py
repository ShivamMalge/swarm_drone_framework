from __future__ import annotations

import pytest

from src.adaptation.safety_projector import THETA_SAFE_BOUNDS, project_to_theta_safe


def test_safety_projector_clips_values() -> None:
    # Intentionally extreme values strictly outside safe domain boundaries
    proposed = {
        "coverage_gain": 0.1,         # Too low (bound 0.5)
        "gossip_epsilon": 0.1,        # Too high (bound 0.05)
        "broadcast_rate": 2.0,        # Too high (bound 1.5)
        "auction_participation": -0.5,# Too low (bound 0.0)
        "velocity_scale": 1.9,        # Too high (bound 1.5)
    }

    safe, count = project_to_theta_safe(proposed)

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

    safe, count = project_to_theta_safe(valid)

    assert count == 0
    for k in valid:
        assert safe[k] == valid[k]

def test_safety_projector_handles_unknown_keys() -> None:
    proposed = {
        "coverage_gain": 3.0,
        "custom_metric": 100.0,
    }

    safe, count = project_to_theta_safe(proposed)
    assert count == 1
    assert safe["coverage_gain"] == 2.0
    assert safe["custom_metric"] == 100.0

def test_safety_projector_determinism() -> None:
    # Validates stateless mapping output deterministic equality
    proposed = {"velocity_scale": -1.0}
    
    r1, _ = project_to_theta_safe(proposed)
    r2, _ = project_to_theta_safe(proposed)
    
    assert r1 == r2
