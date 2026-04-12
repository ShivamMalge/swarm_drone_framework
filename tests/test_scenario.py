"""
Phase 2P verification: Scenario Configuration Studio.
"""

import os
import shutil
import sys
from pathlib import Path

from src.core.config import SimConfig
from src.scenario.scenario_model import ScenarioConfig
from src.scenario.scenario_validator import ScenarioValidator
from src.scenario.scenario_loader import ScenarioLoader

TEST_DIR = Path(os.path.dirname(__file__)) / ".." / "scenarios_test"

def test_scenario_serialization():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    loader = ScenarioLoader(TEST_DIR)
    
    cfg1 = ScenarioConfig(
        name="test_export",
        seed=123,
        num_agents=75
    )
    cfg1.interference.enabled = True
    cfg1.interference.intensity = 0.8
    
    path = loader.save_scenario(cfg1)
    assert path.exists(), "YAML file not created"
    
    cfg2 = loader.load_scenario(path.name)
    assert cfg2.name == "test_export"
    assert cfg2.seed == 123
    assert cfg2.num_agents == 75
    assert cfg2.interference.enabled is True
    assert cfg2.interference.intensity == 0.8
    
    # Test valid
    errors = ScenarioValidator.validate(cfg2)
    assert len(errors) == 0, "Valid config should have no errors"

    # Test invalid validation
    cfg3 = ScenarioConfig(num_agents=600, seed=-5)
    errors = ScenarioValidator.validate(cfg3)
    assert len(errors) > 0, "Invalid config should produce errors"
    assert any("500" in e for e in errors), "Agent count error missing"
    assert any("non-negative" in e.lower() for e in errors), "Seed error missing"
    
    print("[PASS] YAML Serialization & Validation")
    
    # Clean up
    shutil.rmtree(TEST_DIR, ignore_errors=True)

def test_simconfig_conversion():
    base_cfg = SimConfig(seed=1, num_agents=10, p_drop=0.1)
    
    scen_cfg = ScenarioConfig(
        name="my_custom",
        seed=999,
        num_agents=150,
        communication_radius=35.0,
    )
    scen_cfg.interference.enabled = True
    scen_cfg.interference.intensity = 0.5
    
    sim_cfg = scen_cfg.to_sim_config(base_cfg)
    
    assert sim_cfg.seed == 999
    assert sim_cfg.num_agents == 150
    assert sim_cfg.comm_radius == 35.0
    assert sim_cfg.psi_max == 0.5
    # Since intensity=0.5, p_drop should scale as max(base.p_drop, intensity*0.5)
    # max(0.1, 0.5 * 0.5) = max(0.1, 0.25) = 0.25
    assert sim_cfg.p_drop == 0.25
    
    print("[PASS] ScenarioConfig -> SimConfig override determinism")

if __name__ == "__main__":
    test_scenario_serialization()
    test_simconfig_conversion()
    print("\nAll Phase 2P tests passed.")
