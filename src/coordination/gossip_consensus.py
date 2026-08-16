"""
Decentralized Gossip Consensus Module.

Implements the standard Laplacian consensus update:
    x_i(t+1) = x_i(t) + epsilon * SUM(x_j - x_i)

Stability policy — SINGLE enforcement point:
    The delay-tolerant step-size bound
        0 < epsilon <= 0.99 / (d_i * (tau_max + 1))
    is enforced by the Theta_safe projector (Algorithm 1 in the manuscript,
    src/adaptation/safety_projector.py, dynamic bound computed in
    src/agent/agent_core.py::_apply_strategy_parameters). This function applies
    the epsilon it is given, verbatim.

    A previous revision ALSO recomputed the same bound here and clamped on
    ~98.5% of calls, silently overriding the projector — the paper's stated
    adaptation mechanism — with a second implementation of itself that used a
    different tau discretisation (ceil here vs floor in the projector; audit
    F-10 correction, F-24). One bound, one owner: do not reintroduce a clamp
    here. If epsilon arrives unsafe, the defect is upstream in the projector,
    and hiding it here makes the paper describe a mechanism that is not the
    one governing behaviour.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_gossip_update(
    own_state: float,
    neighbor_states: list[float],
    neighbor_delays: list[float],
    dt: float,
    base_epsilon: float,
) -> float:
    """
    Compute the next state value using the discrete-time consensus protocol.

    Parameters
    ----------
    own_state : float
        The agent's current state value x_i(t).
    neighbor_states : list[float]
        The state values of all known neighbors from the stale LocalMap.
    neighbor_delays : list[float]
        The observed continuous time delays for each neighbor's message.
        (Unused here; retained so the call site documents what information
        the stability bound upstream is computed from.)
    dt : float
        The simulation discrete time step duration. (Unused here; see above.)
    base_epsilon : float
        The step size to apply. Already projected onto the safe manifold by
        the Theta_safe pipeline — applied verbatim, see module docstring.

    Returns
    -------
    float
        The updated consensus state x_i(t+1).
    """
    if not neighbor_states:
        return own_state

    # SUM(x_j - x_i)
    diff_sum = sum((x_j - own_state) for x_j in neighbor_states)

    return own_state + base_epsilon * diff_sum
