"""
Unit tests for PacketDropSampler — drop rate matches expected Bernoulli.
"""

import numpy as np
from src.communication.packet_drop import PacketDropSampler


class TestPacketDropSampler:

    def test_zero_drop_never_drops(self):
        rng = np.random.default_rng(0)
        sampler = PacketDropSampler(base_p_drop=0.0, rng=rng)
        drops = sum(sampler.should_drop(0.0) for _ in range(1000))
        assert drops == 0

    def test_full_drop_always_drops(self):
        rng = np.random.default_rng(0)
        sampler = PacketDropSampler(base_p_drop=1.0, rng=rng)
        drops = sum(sampler.should_drop(0.0) for _ in range(1000))
        assert drops == 1000

    def test_full_interference_always_drops(self):
        rng = np.random.default_rng(0)
        sampler = PacketDropSampler(base_p_drop=0.0, rng=rng)
        drops = sum(sampler.should_drop(1.0) for _ in range(1000))
        assert drops == 1000

    def test_empirical_drop_rate_matches(self):
        """With p_drop=0.2 and ψ=0.1, effective survive = 0.8*0.9 = 0.72."""
        rng = np.random.default_rng(42)
        sampler = PacketDropSampler(base_p_drop=0.2, rng=rng)
        n = 10000
        drops = sum(sampler.should_drop(0.1) for _ in range(n))
        empirical_drop = drops / n
        expected_drop = 1.0 - 0.8 * 0.9  # 0.28
        assert abs(empirical_drop - expected_drop) < 0.03  # ±3% tolerance

    def test_invalid_p_drop_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            PacketDropSampler(base_p_drop=1.5, rng=np.random.default_rng(0))
        with pytest.raises(ValueError):
            PacketDropSampler(base_p_drop=-0.1, rng=np.random.default_rng(0))
