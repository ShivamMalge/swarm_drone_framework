
import numpy as np
from src.core.config import SimConfig, RegimeConfig
from src.simulation import Phase1Simulation
import os

def verify_patches():
    # 1. Config Setup
    config = SimConfig(
        num_agents=100,
        max_time=100.0,
        seed=42,
        coverage_enabled=True,
        r_task=2.0
    )
    
    sim = Phase1Simulation(config)
    
    # 2. Run simulation
    print("Running simulation with N=100...")
    sim.run()
    
    # 3. Verify Regime Asynchrony (Patch 1)
    # Collect regime transition times for each agent
    # Since we don't have a direct log of transitions, we'll check the agents' current regimes at the end
    # and maybe look at the classification configs if they were jittered.
    
    print("\n--- Patch 1 Verification: Regime Classification Jitters ---")
    unique_neighbors = set(a.regime_classifier.config.neighbor_low for a in sim.agents)
    unique_variances = set(round(a.regime_classifier.config.variance_high, 4) for a in sim.agents)
    
    print(f"Number of unique 'neighbor_low' parameters (int): {len(unique_neighbors)}")
    print(f"Number of unique 'variance_high' parameters (float): {len(unique_variances)}")
    
    if len(unique_neighbors) > 1 or len(unique_variances) > 1:
        print("SUCCESS: Regime parameters are jittered.")
    else:
        print("FAILURE: Regime parameters are identical.")

    # 4. Verify Physical Task Resolution (Patch 2)
    print("\n--- Patch 2 Verification: Physical Task Consumption ---")
    print(f"Tasks remaining in simulation.active_tasks: {len(sim.active_tasks)}")
    
    # Check if any agents reached their tasks
    tasks_visited = 0
    for a in sim.agents:
        if a.active_task_id is None and hasattr(a, '_local_map') and len(a._local_map.task_metadata) > 0:
            # If agent had a task but it's now None, it likely completed it or observed it missing
            tasks_visited += 1
            
    print(f"Agents that completed or cleared tasks: {tasks_visited}")
    
    # Verify deterministic replay (basic check: run twice and compare agent positions)
    print("\n--- Replay Verification ---")
    pos1 = [a.position for a in sim.agents]
    
    sim2 = Phase1Simulation(config)
    sim2.run()
    pos2 = [a.position for a in sim2.agents]
    
    mismatch = False
    for p1, p2 in zip(pos1, pos2):
        if not np.allclose(p1, p2):
            mismatch = True
            break
            
    if mismatch:
        print("FAILURE: Deterministic replay is broken.")
    else:
        print("SUCCESS: Deterministic replay preserved.")

if __name__ == "__main__":
    verify_patches()
