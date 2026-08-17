"""
Unit tests for PacketDropSampler.

Contract under test (see packet_drop.py, and Eq. (1)-(2) of the manuscript):

    p_survive(d) = (1 - p_drop) * (1 - psi) * max(0, 1 - (d / R_tx)^2)

These tests were dead for an unknown period: `should_drop` gained distance and
tx_radius parameters when the path-loss factor was introduced, and every test
here kept the old single-argument call, so all failed on TypeError. On repair,
the CONTRACT changed with them: "zero drop never drops" is only true at d = 0 —
at any positive distance the path-loss factor drops packets even with
p_drop = 0 and psi = 0, and at d = R_tx delivery probability is exactly zero.
That boundary property is now pinned deliberately, because the manuscript
states it (the effective communication range is smaller than the nominal
radius).
"""

import numpy as np
import pytest

from src.communication.packet_drop import PacketDropSampler


R = 20.0  # tx radius used throughout


class TestPacketDropSampler:

    def test_zero_drop_never_drops_at_zero_distance(self):
        # Path loss is 1.0 at d=0, so with p_drop=0 and psi=0 nothing drops.
        rng = np.random.default_rng(0)
        sampler = PacketDropSampler(base_p_drop=0.0, rng=rng)
        drops = sum(sampler.should_drop(0.0, 0.0, R) for _ in range(1000))
        assert drops == 0

    def test_zero_drop_still_drops_with_distance(self):
        # The deliberate contract change: p_drop=0 is NOT a delivery guarantee.
        # At d/R = 0.8, p_survive = 1 - 0.64 = 0.36 even with no baseline drop.
        rng = np.random.default_rng(0)
        sampler = PacketDropSampler(base_p_drop=0.0, rng=rng)
        n = 10000
        drops = sum(sampler.should_drop(0.0, 0.8 * R, R) for _ in range(n))
        expected_drop = 1.0 - (1.0 - (0.8) ** 2)  # 0.64
        assert abs(drops / n - expected_drop) < 0.03

    def test_delivery_probability_zero_at_nominal_radius(self):
        # Manuscript-stated boundary property: p_survive(R_tx) = 0 exactly.
        rng = np.random.default_rng(0)
        sampler = PacketDropSampler(base_p_drop=0.0, rng=rng)
        drops = sum(sampler.should_drop(0.0, R, R) for _ in range(1000))
        assert drops == 1000

    def test_beyond_radius_always_drops(self):
        rng = np.random.default_rng(0)
        sampler = PacketDropSampler(base_p_drop=0.0, rng=rng)
        assert all(sampler.should_drop(0.0, R * 1.01, R) for _ in range(100))

    def test_full_drop_always_drops(self):
        rng = np.random.default_rng(0)
        sampler = PacketDropSampler(base_p_drop=1.0, rng=rng)
        drops = sum(sampler.should_drop(0.0, 0.0, R) for _ in range(1000))
        assert drops == 1000

    def test_full_interference_always_drops(self):
        rng = np.random.default_rng(0)
        sampler = PacketDropSampler(base_p_drop=0.0, rng=rng)
        drops = sum(sampler.should_drop(1.0, 0.0, R) for _ in range(1000))
        assert drops == 1000

    def test_empirical_drop_rate_matches_three_factor_model(self):
        """p_drop=0.2, psi=0.1, d/R=0.5: survive = 0.8 * 0.9 * 0.75 = 0.54."""
        rng = np.random.default_rng(42)
        sampler = PacketDropSampler(base_p_drop=0.2, rng=rng)
        n = 10000
        drops = sum(sampler.should_drop(0.1, 0.5 * R, R) for _ in range(n))
        expected_drop = 1.0 - (0.8 * 0.9 * 0.75)  # 0.46
        assert abs(drops / n - expected_drop) < 0.03

    def test_invalid_p_drop_rejected(self):
        with pytest.raises(ValueError):
            PacketDropSampler(base_p_drop=1.5, rng=np.random.default_rng(0))
        with pytest.raises(ValueError):
            PacketDropSampler(base_p_drop=-0.1, rng=np.random.default_rng(0))
