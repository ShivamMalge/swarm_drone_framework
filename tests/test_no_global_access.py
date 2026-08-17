"""
Global purity test — verifies AgentCore enforces decentralization.

This test statically inspects the AgentCore module source to ensure
it does NOT import or reference any global-scope objects.
"""

import ast
import inspect
import os


class TestNoGlobalAccess:
    """Verify AgentCore maintains strict decentralization."""

    def _get_agent_source_path(self) -> str:
        """Return the file path of agent_core.py."""
        from src.agent import agent_core
        return inspect.getfile(agent_core)

    def _get_imports(self, filepath: str) -> list[str]:
        """Parse all import targets from a Python source file."""
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)
        return imports

    def test_no_kernel_import(self):
        """AgentCore must not import SimulationKernel."""
        path = self._get_agent_source_path()
        imports = self._get_imports(path)
        for imp in imports:
            assert "kernel" not in imp.lower(), (
                f"AgentCore imports '{imp}' which references the kernel. "
                "Agents must not hold kernel references."
            )

    def test_no_adjacency_import(self):
        """AgentCore must not import RGG/adjacency modules."""
        path = self._get_agent_source_path()
        imports = self._get_imports(path)
        forbidden = ["rgg", "adjacency", "graph", "laplacian", "spectral_analyzer"]
        for imp in imports:
            for keyword in forbidden:
                assert keyword not in imp.lower(), (
                    f"AgentCore imports '{imp}' which references global "
                    f"graph structure ({keyword})."
                )

    def test_no_global_state_reference_in_source(self):
        """AgentCore source must not reference global state objects."""
        path = self._get_agent_source_path()
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        # "lambda_2" was previously on this list and matched the agent's own
        # LOCAL estimator attribute (_lambda_2_proxy), failing the test for
        # months on a false positive. The patterns below name specifically
        # GLOBAL quantities an agent must never reference.
        forbidden_patterns = [
            "global_adjacency",
            "adjacency_matrix",
            "agent_list",
            "all_agents",
            "true_lambda_2",
            "global_lambda_2",
            "spectral_gap",
            "algebraic_connectivity",
            "SimulationKernel",
            "connectivity_metrics",
        ]
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"AgentCore source contains forbidden reference: '{pattern}'"
            )

    def test_oracle_channel_closed_when_disabled(self):
        """
        The one REAL global-information channel into AgentCore is
        oracle_centroid, which the orchestrator computes from true global
        positions and writes onto the agent in oracle mode. When
        global_info_enabled is False that channel must stay shut for the whole
        run -- this is a runtime check, not a source grep, because this exact
        flag once silently failed to propagate (audit 0.5 history).
        """
        import numpy as np
        from src.core.config import SimConfig
        from src.simulation import Phase1Simulation

        sim = Phase1Simulation(SimConfig(
            num_agents=20, max_time=30.0, seed=42,
            coverage_enabled=True, global_info_enabled=False,
        ))
        sim.run()
        for agent in sim.agents:
            assert agent.global_info_enabled is False
            assert agent.oracle_centroid is None, (
                f"Agent {agent.agent_id} received an oracle centroid with "
                "global_info_enabled=False: global information leaked."
            )

    def test_agent_does_not_store_other_agents(self):
        """AgentCore instance must not hold references to other agent objects."""
        from src.agent.agent_core import AgentCore
        from src.agent.energy_model import EnergyModel
        import numpy as np

        agent = AgentCore(
            agent_id=0,
            position=np.array([10.0, 10.0]),
            energy_model=EnergyModel(100.0),
            rng=np.random.default_rng(0),
        )

        # Inspect all attributes — none should be an AgentCore
        for attr_name in dir(agent):
            if attr_name.startswith("__"):
                continue
            attr = getattr(agent, attr_name)
            assert not isinstance(attr, AgentCore), (
                f"Agent stores another AgentCore in attribute '{attr_name}'"
            )
            # Check lists and dicts for agent references
            if isinstance(attr, (list, tuple)):
                for item in attr:
                    assert not isinstance(item, AgentCore)
            if isinstance(attr, dict):
                for v in attr.values():
                    assert not isinstance(v, AgentCore)
