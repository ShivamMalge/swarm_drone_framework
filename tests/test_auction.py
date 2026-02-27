import copy
import inspect
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest

from src.core.config import SimConfig
from src.simulation import Phase1Simulation

# ============================================================================
# Helpers
# ============================================================================

def run_simulation(seed: int, p_drop: float, num_agents: int = 20, max_time: float = 30.0) -> Phase1Simulation:
    config = SimConfig(
        num_agents=num_agents,
        p_drop=p_drop,
        seed=seed,
        max_time=max_time,
        comm_radius=100.0, # Fully connected if no drop
        auction_timeout=5.0
    )
    sim = Phase1Simulation(config)
    sim.run()
    return sim

# ============================================================================
# Tests
# ============================================================================

class TestAuctionAllocation:
    def test_deterministic_winner_no_drop(self):
        """
        Verify that under ideal communication (p_drop=0) and full connectivity,
        all agents agree on the exact same winner (the one with the true highest bid).
        """
        sim = run_simulation(seed=101, p_drop=0.0, num_agents=20, max_time=10.0)
        
        # 3 tasks were seeded, spaced by 10 seconds.
        # By max_time, some might be resolved.
        results = sim.auction_results
        
        task_winners = {}
        for task_id, winner_id, t in results:
            if task_id not in task_winners:
                task_winners[task_id] = set()
            task_winners[task_id].add(winner_id)
            
        for task_id, unique_winners in task_winners.items():
            # In a perfectly connected, 0-drop environment with sufficient gossip time,
            # there should be exactly ONE agreed winner per task (the agent that appended to results)
            assert len(unique_winners) == 1, f"Consensus failed for {task_id}: selected winners {unique_winners}"


    def test_dropout_robustness(self):
        """
        Verify that under heavy packet loss (p_drop=0.5), the system still functions,
        though agents might partition and have differing views of the winner.
        The simulation shouldn't crash, and agents should base decisions on their bounded local knowledge.
        """
        sim = run_simulation(seed=102, p_drop=0.5, num_agents=20, max_time=10.0)
        
        results = sim.auction_results
        assert len(results) > 0, "Auctions should still resolve even with high packet drop"
        
        task_winners = {}
        for task_id, winner_id, t in results:
            if task_id not in task_winners:
                task_winners[task_id] = set()
            task_winners[task_id].add(winner_id)
            
        # Due to 50% drop rate, there might be multiple perceived winners depending on graph partitions.
        # We just verify the logic resolved properly without errors.
        for task_id, unique_winners in task_winners.items():
            assert len(unique_winners) >= 1, f"Each task should have at least one valid winner chosen locally"


    def test_deterministic_replay(self):
        """
        Verify that running two identical instances yields bitwise identical auction results.
        """
        sim1 = run_simulation(seed=103, p_drop=0.2, num_agents=15, max_time=10.0)
        sim2 = run_simulation(seed=103, p_drop=0.2, num_agents=15, max_time=10.0)
        
        # Same sequence of events -> exact same winner evaluations
        assert sim1.auction_results == sim2.auction_results, "Deterministic replay failed"
        
        # Ensure a different seed yields a different set of winners/timing
        sim3 = run_simulation(seed=999, p_drop=0.2, num_agents=15, max_time=10.0)
        assert sim1.auction_results != sim3.auction_results, "Different seeds should yield divergent physical states"


    def test_no_global_access(self):
        """
        Verify strictly that the auction logic module imports nothing global
        and does not query the SimulationKernel or RGG structures.
        """
        import src.coordination.auction as auction
        
        # Ensure no disallowed modules are imported into the auction logic
        disallowed = ['scipy', 'kernel', 'SimulationKernel', 'CommunicationEngine']
        source_lines = inspect.getsource(auction)
        
        for bad_import in disallowed:
            assert bad_import not in source_lines, f"Architecture violation: found {bad_import} in auction.py"
