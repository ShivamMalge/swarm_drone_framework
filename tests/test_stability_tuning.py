import pytest

from src.adaptation.stability_tuner import smooth_update


def test_smooth_update_boundaries():
    """Ensure EMA update operates linearly around logical bounds."""
    # alpha = 1.0 -> jump straight to target
    assert smooth_update(0.0, 10.0, 1.0) == 10.0
    
    # alpha = 0.0 -> freeze at start
    assert smooth_update(5.0, 10.0, 0.0) == 5.0
    
    # alpha = 0.5 -> perfect half point
    assert smooth_update(0.0, 10.0, 0.5) == 5.0
    
    # alpha = 0.1 -> fractional progression
    assert round(smooth_update(0.0, 10.0, 0.1), 1) == 1.0

def test_smooth_update_alpha_defensive_clamping():
    """Ensure alpha is actively clamped inside [0.0, 1.0] mathematically."""
    
    # alpha > 1.0 should map to exactly 1.0
    assert smooth_update(0.0, 10.0, 1.5) == 10.0
    
    # alpha < 0.0 should map to exactly 0.0
    assert smooth_update(5.0, 10.0, -0.5) == 5.0

def test_smooth_update_negative_domain_transition():
    """Ensure EMA correctly slides values across the zero bound."""
    assert smooth_update(-10.0, 10.0, 0.5) == 0.0
    assert smooth_update(10.0, -10.0, 0.5) == 0.0
    assert smooth_update(-5.0, -15.0, 0.1) == -6.0
