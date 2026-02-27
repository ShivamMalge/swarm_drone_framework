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
    consensus_state: float
    send_time: float
    
    # Phase 2C
    auction_bid: tuple[str, float, int] | None = None


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
    coverage_enabled : bool
        Whether to use distributed Voronoi coverage control.
    comm_radius : float
        Agent's communication radius, bounding its local Voronoi cell.
    """

    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        energy_model: EnergyModel,
        rng: np.random.Generator,
        v_max: float = 2.0,
        coverage_enabled: bool = False,
        comm_radius: float = 20.0,
    ) -> None:
        self.agent_id = agent_id
        self._position: np.ndarray = position.copy()
        self._energy = energy_model
        self._rng = rng
        self._local_map = LocalMap()
        self._v_max = v_max
        self._coverage_enabled = coverage_enabled
        self._comm_radius = comm_radius

        # Communication queues (processed only via events)
        self._inbox: deque[AgentMessage] = deque()
        self._outbox: deque[AgentMessage] = deque()

        # Phase 2B: Consensus State
        self.consensus_state: float = float(agent_id)

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
        Phase 2: Voronoi centroid seeking via distributed Lloyd logic.
        """
        if not self.is_alive:
            return np.zeros(2)

        if self._coverage_enabled:
            from src.coordination.voronoi_coverage import compute_local_centroid
            
            neighbors = self._local_map.get_all_neighbors()
            neighbor_positions = [info.position for info in neighbors]
            
            # Compute centroid based ONLY on own position + stale neighbor beliefs
            centroid = compute_local_centroid(
                self._position, neighbor_positions, self._comm_radius
            )
            
            # Proportional feedback law: v_i = k * (c_i - x_i)
            k_gain = 0.5
            velocity = k_gain * (centroid - self._position)
            
            # Bound velocity safely
            speed = float(np.linalg.norm(velocity))
            if speed > self._v_max:
                velocity = (velocity / speed) * self._v_max
            return velocity
        else:
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
        # Phase 2C: Attach one random known auction bid to gossip (O(1) payload size)
        auction_bid = None
        if self._local_map.active_auctions:
            # Pick a random active auction to gossip
            task_ids = list(self._local_map.active_auctions.keys())
            task_id = self._rng.choice(task_ids)
            best_bid, winner_id = self._local_map.active_auctions[task_id]
            auction_bid = (task_id, best_bid, winner_id)

        return AgentMessage(
            sender_id=self.agent_id,
            position=self._position.copy(),
            energy=self._energy.energy,
            consensus_state=self.consensus_state,
            send_time=current_time,
            auction_bid=auction_bid,
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
                consensus_state=msg.consensus_state,
                timestamp=msg.send_time,
            )
            
            # Phase 2C: process attached auction bid if present
            if msg.auction_bid is not None:
                task_id, bid_val, bidder_id = msg.auction_bid
                self._local_map.update_auction(task_id, bid_val, bidder_id)
                
            count += 1
        return count

    # ── Consensus (called by CONSENSUS_UPDATE handler) ──────

    def handle_consensus_update(self, epsilon: float) -> None:
        """
        Execute one iteration of the discrete-time distributed gossip protocol.
        """
        if not self.is_alive:
            return

        from src.coordination.gossip_consensus import compute_gossip_update
        
        # Fog-of-war constraint: We only use the local, delayed map beliefs.
        neighbors = self._local_map.get_all_neighbors()
        neighbor_states = [nb.consensus_state for nb in neighbors]
        
        self.consensus_state = compute_gossip_update(
            self.consensus_state, neighbor_states, epsilon
        )

    # ── Auction (called by AUCTION handlers) ────────────────
    
    def handle_auction_start(
        self, 
        task_id: str, 
        task_position: np.ndarray, 
        task_reward: float
    ) -> None:
        """
        Process a newly spawned task. Compute local bid and cache it.
        The bid will naturally gossip out via periodic MSG_TRANSMIT.
        """
        if not self.is_alive:
            return
            
        from src.coordination.auction import compute_bid
        
        bid = compute_bid(
            agent_position=self._position,
            agent_energy=self._energy.energy,
            task_position=task_position,
            task_reward=task_reward
        )
        
        if bid > float('-inf'):
            self._local_map.update_auction(task_id, bid, self.agent_id)

    def handle_auction_resolve(self, task_id: str) -> bool:
        """
        Evaluate if this agent won the auction based on its LocalMap.
        Returns True if won, False otherwise.
        """
        if not self.is_alive:
            return False
            
        from src.coordination.auction import resolve_local_winner
        
        winner_id = resolve_local_winner(task_id, self._local_map)
        
        # Win evaluated safely solely from local belief buffer
        if winner_id == self.agent_id:
            # We won. Optional: update state (e.g. consume energy) or mark task done.
            # In Phase 2C, returning True is sufficient to log the victory.
            return True
            
        return False
