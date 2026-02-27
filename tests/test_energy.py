"""
Unit tests for EnergyModel — monotonic decay, death, no negatives.
"""

import pytest

from src.agent.energy_model import EnergyModel


class TestEnergyModel:

    def test_initial_energy(self):
        e = EnergyModel(100.0)
        assert e.energy == 100.0
        assert e.is_alive is True

    def test_consume_decreases_energy(self):
        e = EnergyModel(100.0)
        e.consume(30.0)
        assert e.energy == 70.0

    def test_monotonic_decay(self):
        e = EnergyModel(50.0)
        prev = e.energy
        for _ in range(10):
            e.consume(3.0)
            assert e.energy <= prev
            prev = e.energy

    def test_energy_never_negative(self):
        e = EnergyModel(10.0)
        e.consume(999.0)
        assert e.energy == 0.0
        assert e.is_alive is False

    def test_death_is_permanent(self):
        e = EnergyModel(5.0)
        e.consume(5.0)
        assert e.is_alive is False
        with pytest.raises(RuntimeError, match="dead"):
            e.consume(1.0)

    def test_negative_consumption_rejected(self):
        e = EnergyModel(50.0)
        with pytest.raises(ValueError, match="non-negative"):
            e.consume(-10.0)

    def test_negative_initial_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            EnergyModel(-5.0)

    def test_zero_consumption(self):
        e = EnergyModel(50.0)
        e.consume(0.0)
        assert e.energy == 50.0
        assert e.is_alive is True

    def test_exact_depletion(self):
        e = EnergyModel(10.0)
        e.consume(10.0)
        assert e.energy == 0.0
        assert e.is_alive is False
