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
from src.core.config import RegimeConfig
from src.regime.classifier import Regime, RegimeClassifier
from src.regime.telemetry_buffer import TelemetryBuffer, TelemetrySnapshot
from src.adaptation.hybrid_supervisor import HybridSupervisor, Strategy
from src.adaptation.safety_projector import project_to_theta_safe
from src.adaptation.stability_tuner import smooth_update


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
        regime_config: RegimeConfig | None = None,
        tuning_alpha: float = 0.15,
        theta_safe_enabled: bool = True,
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

        # Phase 2D: Passive Regime Detection
        reg_cfg = regime_config if regime_config is not None else RegimeConfig()
        
        # Patch 1: Proxy Synchronization Drift
        # Apply deterministic jitter to monitoring parameters once at initialization
        jitter_neighbor = self._rng.integers(-1, 2)  # -1, 0, or 1
        jitter_variance = self._rng.uniform(-0.2, 0.2)
        jitter_staleness = self._rng.uniform(-0.5, 0.5)
        
        self.telemetry_buffer = TelemetryBuffer(window_size=reg_cfg.window_size)
        
        # Derived config with jitters
        self.regime_classifier = RegimeClassifier(config=RegimeConfig(
            window_size=reg_cfg.window_size,
            classification_interval=reg_cfg.classification_interval,
            neighbor_low=max(1, reg_cfg.neighbor_low + jitter_neighbor),
            variance_high=max(0.1, reg_cfg.variance_high + jitter_variance),
            energy_slope_critical=reg_cfg.energy_slope_critical,
            staleness_high=max(0.1, reg_cfg.staleness_high + jitter_staleness)
        ))
        self.current_regime = Regime.STABLE

        # Patch 2: Active Task Tracking
        self.active_task_id: str | None = None
        self.active_task_pos: np.ndarray | None = None

        # Phase 3C: Adaptive Hybrid Supervisor
        self.supervisor = HybridSupervisor()
        self.current_strategy = Strategy.NORMAL_OPERATION
        
        # Phase 4: Theta Safe Parameters
        self.coverage_gain: float = 1.0
        self.gossip_epsilon: float = 0.05
        self.broadcast_rate: float = 1.0
        self.auction_participation: float = 1.0
        self.velocity_scale: float = 1.0
        self.projection_events: int = 0
        
        # Phase 4B: Stability Constrained Tuning
        self.tuning_alpha = tuning_alpha
        self.total_tuning_updates: int = 0
        self.total_parameter_shift: float = 0.0
        self.max_parameter_shift: float = 0.0
        self._theta_safe_enabled = theta_safe_enabled
        
        # Stored initial states
        self._base_epsilon = 0.05
        self._coverage_enabled = coverage_enabled

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

        if self.active_task_pos is not None:
            # Patch 2: Prioritize moving toward active task if won auction
            direction = self.active_task_pos - self._position
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                velocity = (direction / dist) * self._v_max * self.velocity_scale
                return velocity

        if self.coverage_gain > 0.05:
            from src.coordination.voronoi_coverage import compute_local_centroid
            
            neighbors = self._local_map.get_all_neighbors()
            neighbor_positions = [info.position for info in neighbors]
            
            # Compute centroid based ONLY on own position + stale neighbor beliefs
            centroid = compute_local_centroid(
                self._position, neighbor_positions, self._comm_radius
            )
            
            # Proportional feedback law: v_i = k * (c_i - x_i)
            k_gain = 0.5 * self.coverage_gain
            velocity = k_gain * (centroid - self._position)
            
            # Bound velocity safely
            speed = float(np.linalg.norm(velocity))
            max_spd = self._v_max * self.velocity_scale
            if speed > max_spd:
                velocity = (velocity / speed) * max_spd
            return velocity
        else:
            # Simple random walk placeholder (deterministic backbone)
            direction = self._rng.standard_normal(2)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            return direction * self._v_max * 0.5 * self.velocity_scale

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

    def prepare_broadcast(self, current_time: float) -> AgentMessage | None:
        """Create an outgoing message with current state snapshot."""
        if self.broadcast_rate <= 0.05:
            # Structurally deactivated
            return None
        elif self.broadcast_rate < 1.0:
            # Probabilistically thin transmissions to conserve energy
            if self._rng.random() > self.broadcast_rate:
                return None
            
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
        
        # Apply local tuning (completely obscuring the global event epsilon argument)
        effective_epsilon = self.gossip_epsilon
        
        self.consensus_state = compute_gossip_update(
            self.consensus_state, neighbor_states, effective_epsilon
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
        if not self.is_alive or self.auction_participation <= 0.05:
            return
            
        from src.coordination.auction import compute_bid
        
        # Patch 2: Cache task position in LocalMap
        self._local_map.task_metadata[task_id] = task_position.copy()
        
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
        if not self.is_alive or self.auction_participation <= 0.05:
            return False
            
        from src.coordination.auction import resolve_local_winner
        
        winner_id = resolve_local_winner(task_id, self._local_map)
        
        # Win evaluated safely solely from local belief buffer
        if winner_id == self.agent_id:
            # Patch 2: Store task target for physical navigation
            self.active_task_id = task_id
            self.active_task_pos = self._local_map.task_metadata.get(task_id)
            return True
            
        return False

    def handle_task_completion(self, task_id: str) -> None:
        """
        Patch 2: Mark task as completed in LocalMap and clear active task state.
        This is called when the agent discovers the task is missing at the location.
        """
        if self.active_task_id == task_id:
            self.active_task_id = None
            self.active_task_pos = None
            
        # Clean up auctions in LocalMap
        if task_id in self._local_map.active_auctions:
            del self._local_map.active_auctions[task_id]

    # ── Regime Detection (called by REGIME_UPDATE handler) ──
    
    def handle_regime_update(self, current_time: float) -> None:
        """
        Passive evaluation layer. Computes purely local fog-of-war proxy metrics,
        stores them in a sliding window, and heuristically maps to a Regime enum.
        Does NOT alter kinematics, consensus, or auction behaviors.
        """
        if not self.is_alive:
            return

        from src.regime.local_proxies import (
            compute_information_staleness,
            compute_local_consensus_variance,
            compute_neighbor_density,
        )

        # 1. Compute instantaneous fog-of-war metrics bounds
        density = compute_neighbor_density(self._local_map)
        staleness = compute_information_staleness(self._local_map, current_time)
        variance = compute_local_consensus_variance(self)
        
        # 2. Package into a snapshot
        snapshot = TelemetrySnapshot(
            time=current_time,
            neighbor_count=density,
            mean_neighbor_age=staleness,
            local_consensus_variance=variance,
            local_energy=self._energy.energy
        )
        
        # 3. Slide window
        self.telemetry_buffer.append(snapshot)
        
        # 4. If buffer has stabilized, classify the behavioral regime
        if self.telemetry_buffer.is_ready():
            smoothed_metrics = self.telemetry_buffer.get_smoothed_metrics()
            self.current_regime = self.regime_classifier.classify(smoothed_metrics)
            
            # 5. Phase 3C/4: Pass regime to supervisor, receive proposed theta, project to safe bounds
            self.current_strategy = self.supervisor.select_strategy(self.current_regime)
            theta_proposed = self.supervisor.propose_parameters(self.current_strategy, self._base_epsilon)
            
            if not self._coverage_enabled:
                theta_proposed["coverage_gain"] = 0.0
                
            self._apply_strategy_parameters(theta_proposed)
            
    def _apply_strategy_parameters(self, theta_proposed: dict[str, float]) -> None:
        """
        Projects proposed modifications onto the Safe Manifold, then smoothly
        interpolates current agent bounds toward the required targets.
        """
        if self._theta_safe_enabled:
            theta_safe, clip_count = project_to_theta_safe(theta_proposed)
        else:
            theta_safe = theta_proposed
            clip_count = 0
        
        old_params = [
            self.coverage_gain, self.gossip_epsilon, self.broadcast_rate, 
            self.auction_participation, self.velocity_scale
        ]
        
        # Phase 4B: Apply exponential moving average stability smoothing
        self.coverage_gain = smooth_update(self.coverage_gain, theta_safe.get("coverage_gain", self.coverage_gain), self.tuning_alpha)
        self.gossip_epsilon = smooth_update(self.gossip_epsilon, theta_safe.get("gossip_epsilon", self.gossip_epsilon), self.tuning_alpha)
        self.broadcast_rate = smooth_update(self.broadcast_rate, theta_safe.get("broadcast_rate", self.broadcast_rate), self.tuning_alpha)
        self.auction_participation = smooth_update(self.auction_participation, theta_safe.get("auction_participation", self.auction_participation), self.tuning_alpha)
        self.velocity_scale = smooth_update(self.velocity_scale, theta_safe.get("velocity_scale", self.velocity_scale), self.tuning_alpha)
        
        new_params = [
            self.coverage_gain, self.gossip_epsilon, self.broadcast_rate, 
            self.auction_participation, self.velocity_scale
        ]
        
        shift = sum(abs(n - o) for n, o in zip(new_params, old_params)) / len(new_params)
        
        if shift > 0.0:
            self.total_tuning_updates += 1
            self.total_parameter_shift += shift
            if shift > self.max_parameter_shift:
                self.max_parameter_shift = shift
        
        self.projection_events += clip_count

