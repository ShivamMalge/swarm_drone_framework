"""
Agent core — the autonomous entity.

Strictly decentralized: holds ONLY its own state and a LocalMap of
stale neighbor beliefs.  Has NO access to:
  - The kernel or any global dispatcher
  - Any collection of other agents
  - Any global graph or topology structure
  - The interference field
  - True spectral connectivity metrics

All inter-agent communication flows through the event queue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.agent.energy_model import EnergyModel
from src.agent.local_map import LocalMap


@dataclass
class AgentMessage:
    """Payload carried by inter-agent messages."""

    sender_id: int
    position: np.ndarray
    energy: float
    send_time: float


class AgentCore:
    """
    Autonomous agent with position, energy, and local belief map.

    Parameters
    ----------
    agent_id : int
        Unique identifier.
    position : np.ndarray
        Initial position (2D).
    energy_model : EnergyModel
        Energy tracker (monotonically decreasing).
    rng : np.random.Generator
        Per-agent independent RNG stream (injected, not global).
    v_max : float
        Maximum velocity magnitude.
    """

    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        energy_model: EnergyModel,
        rng: np.random.Generator,
        v_max: float = 2.0,
    ) -> None:
        self.agent_id = agent_id
        self._position: np.ndarray = position.copy()
        self._energy = energy_model
        self._rng = rng
        self._local_map = LocalMap()
        self._v_max = v_max

        # Communication queues (processed only via events)
        self._inbox: deque[AgentMessage] = deque()
        self._outbox: deque[AgentMessage] = deque()

    # ── Position ────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        """Current position (read-only copy)."""
        return self._position.copy()

    @property
    def energy(self) -> float:
        """Current energy scalar."""
        return self._energy.energy

    @property
    def is_alive(self) -> bool:
        """Whether the agent is operational."""
        return self._energy.is_alive

    @property
    def local_map(self) -> LocalMap:
        """The agent's stale neighbor belief map."""
        return self._local_map

    # ── Kinematics (called by KINEMATIC_UPDATE handler) ─────

    def compute_velocity(self) -> np.ndarray:
        """
        Compute desired velocity from local state.

        Phase 1: random walk bounded by v_max.
        Phase 2+: Voronoi centroid seeking via coordination layer.
        """
        if not self.is_alive:
            return np.zeros(2)
        # Simple random walk placeholder (deterministic backbone)
        direction = self._rng.standard_normal(2)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        return direction * self._v_max * 0.5

    def apply_movement(
        self,
        velocity: np.ndarray,
        dt: float,
        clamp_fn: callable,
    ) -> float:
        """
        Update position and return distance moved.

        Energy is NOT consumed here — that is done by the event handler.

        Parameters
        ----------
        velocity : np.ndarray
            Velocity vector.
        dt : float
            Time step.
        clamp_fn : callable
            Boundary clamping function from SpatialGrid.

        Returns
        -------
        float
            Euclidean distance actually moved.
        """
        if not self.is_alive:
            return 0.0
        # Clamp velocity magnitude
        speed = float(np.linalg.norm(velocity))
        if speed > self._v_max:
            velocity = velocity / speed * self._v_max

        old_pos = self._position.copy()
        self._position = clamp_fn(self._position + velocity * dt)
        return float(np.linalg.norm(self._position - old_pos))

    def consume_movement_energy(self, distance: float, p_move: float) -> None:
        """Deduct energy for movement. Called by event handler only."""
        self._energy.consume(distance * p_move)

    def consume_comm_energy(self, num_recipients: int, p_comm: float) -> None:
        """Deduct energy for communication. Called by event handler only."""
        self._energy.consume(num_recipients * p_comm)

    def consume_idle_energy(self, dt: float, p_idle: float) -> None:
        """Deduct idle energy. Called by event handler only."""
        self._energy.consume(dt * p_idle)

    # ── Communication ───────────────────────────────────────

    def prepare_broadcast(self, current_time: float) -> AgentMessage:
        """Create an outgoing message with current state snapshot."""
        return AgentMessage(
            sender_id=self.agent_id,
            position=self._position.copy(),
            energy=self._energy.energy,
            send_time=current_time,
        )

    def receive_message(self, msg: AgentMessage) -> None:
        """
        Buffer an incoming message.

        Called ONLY by the kernel's MSG_DELIVER handler — never directly
        by another agent or by the communication engine.
        """
        self._inbox.append(msg)

    def process_inbox(self) -> int:
        """
        Drain the inbox and update the local map.

        Returns the number of messages processed.
        """
        count = 0
        while self._inbox:
            msg = self._inbox.popleft()
            self._local_map.update_neighbor(
                agent_id=msg.sender_id,
                position=msg.position,
                energy=msg.energy,
                timestamp=msg.send_time,
            )
            count += 1
        return count
