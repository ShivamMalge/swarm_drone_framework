"""
Phase 1 Simulation Orchestrator.

Wires the kernel, environment, agents, communication engine, and metrics
into a single event-driven simulation loop.

The orchestrator is the ONLY place where the kernel and agent array co-exist.
Agents never hold references to the kernel, orchestrator, or each other.
"""

from __future__ import annotations

import numpy as np

from src.agent.agent_core import AgentCore
from src.agent.energy_model import EnergyModel
from src.communication.comm_engine import CommunicationEngine
from src.communication.latency_model import LatencyModel
from src.communication.packet_drop import PacketDropSampler
from src.communication.rgg_builder import RGGBuilder
from src.core.config import SimConfig
from src.core.event import Event, EventType, reset_sequence_counter
from src.core.kernel import SimulationKernel
from src.environment.interference_field import FieldMode, InterferenceField
from src.environment.spatial_grid import SpatialGrid
from src.metrics.drop_rate_tracker import DropRateTracker
from src.metrics.energy_profiler import EnergyProfiler
from src.metrics.position_logger import PositionLogger


class Phase1Simulation:
    """
    Event-driven simulation orchestrator for Phase 1.

    Initialises all modules and registers event handlers with the kernel.
    Agents are stored in a flat array; they NEVER receive a reference
    to this orchestrator, the kernel, or each other.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config

        # ── Spawn independent RNG streams from SeedSequence ─
        self._streams = config.spawn_rng_streams()

        # ── Core ────────────────────────────────────────────
        reset_sequence_counter()
        self.kernel = SimulationKernel()

        # ── Environment ─────────────────────────────────────
        self.grid = SpatialGrid(config.grid_width, config.grid_height)
        self.interference = InterferenceField(
            mode=FieldMode.CONSTANT, psi_max=config.psi_max
        )

        # ── Agents (each gets its own independent RNG stream) ─
        positions = self.grid.random_positions(
            config.num_agents, self._streams["positions"]
        )
        self.agents: list[AgentCore] = []
        for i in range(config.num_agents):
            agent = AgentCore(
                agent_id=i,
                position=positions[i],
                energy_model=EnergyModel(config.energy_initial),
                rng=self._streams[f"agent_{i}"],
                v_max=config.v_max,
                coverage_enabled=config.coverage_enabled,
                comm_radius=config.comm_radius,
                regime_config=config.regime,
            )
            self.agents.append(agent)

        self.alive_mask = np.ones(config.num_agents, dtype=bool)

        # ── Communication (dedicated RNG streams) ───────────
        self.comm_engine = CommunicationEngine(
            rgg_builder=RGGBuilder(config.comm_radius),
            drop_sampler=PacketDropSampler(
                config.p_drop, self._streams["packet_drop"]
            ),
            latency_model=LatencyModel(
                config.latency_mean,
                config.latency_min,
                self._streams["latency"],
            ),
            interference_field=self.interference,
        )

        # ── Metrics (centralized, read-only) ────────────────
        self.energy_profiler = EnergyProfiler()
        self.position_logger = PositionLogger()
        self.drop_tracker = DropRateTracker()
        
        # Phase 2C: Global metric tracker for testing
        self.auction_results: list[tuple[str, int, float]] = [] # task_id, winner_id, time

        # Phase 3D/4: Global metric logs
        self.connectivity_log: list[dict] = []
        self.adaptation_log: list[dict] = []
        

        # ── Register handlers ───────────────────────────────
        self.kernel.register_handler(
            EventType.KINEMATIC_UPDATE, self._handle_kinematic_update
        )
        self.kernel.register_handler(
            EventType.MSG_TRANSMIT, self._handle_msg_transmit
        )
        self.kernel.register_handler(
            EventType.MSG_DELIVER, self._handle_msg_deliver
        )
        self.kernel.register_handler(
            EventType.ENERGY_UPDATE, self._handle_energy_update
        )
        self.kernel.register_handler(
            EventType.METRICS_LOG, self._handle_metrics_log
        )
        self.kernel.register_handler(
            EventType.CONSENSUS_UPDATE, self._handle_consensus_update
        )
        self.kernel.register_handler(
            EventType.AUCTION_START, self._handle_auction_start
        )
        self.kernel.register_handler(
            EventType.AUCTION_RESOLVE, self._handle_auction_resolve
        )
        self.kernel.register_handler(
            EventType.REGIME_UPDATE, self._handle_regime_update
        )

    # ── Initial event scheduling ────────────────────────────

    def seed_events(self) -> None:
        """Schedule the first round of events for all agents."""
        dt = self.config.dt
        for agent in self.agents:
            # Kinematic updates at regular intervals
            self.kernel.schedule_event(Event(
                timestamp=dt,
                event_type=EventType.KINEMATIC_UPDATE,
                agent_id=agent.agent_id,
            ))
            # Communication broadcasts at regular intervals
            self.kernel.schedule_event(Event(
                timestamp=dt,
                event_type=EventType.MSG_TRANSMIT,
                agent_id=agent.agent_id,
            ))
            # Idle energy drain at regular intervals
            self.kernel.schedule_event(Event(
                timestamp=dt,
                event_type=EventType.ENERGY_UPDATE,
                agent_id=agent.agent_id,
            ))
            
            # Phase 2B: Staggered initial consensus events
            # small_delta << dt to break artificial synchrony but maintain determinism
            small_delta = self.config.consensus_dt / 1000.0
            consensus_start_time = dt + (agent.agent_id * small_delta)
            
            self.kernel.schedule_event(Event(
                timestamp=consensus_start_time,
                event_type=EventType.CONSENSUS_UPDATE,
                agent_id=agent.agent_id,
            ))
            
            # Phase 2D: Staggered initial regime events
            regime_delta = self.config.regime.classification_interval / float(self.config.num_agents)
            regime_start_time = dt + (agent.agent_id * regime_delta)
            
            self.kernel.schedule_event(Event(
                timestamp=regime_start_time,
                event_type=EventType.REGIME_UPDATE,
                agent_id=agent.agent_id,
            ))
        # Phase 2C: Seed a batch of tasks to auction dynamically
        # Space them out to allow O(1) gossip to converge per task
        for task_idx in range(3):
            task_id = f"task_{task_idx}"
            task_pos = self.grid.random_positions(1, self._streams["task_spawner"])[0]
            task_reward = float(self._streams["task_spawner"].uniform(50.0, 150.0))
            
            start_time = 0.5 + float(task_idx * 15.0)
            
            for agent in self.agents:
                # Agent receives knowledge of new task
                self.kernel.schedule_event(Event(
                    timestamp=start_time,
                    event_type=EventType.AUCTION_START,
                    agent_id=agent.agent_id,
                    payload={"task_id": task_id, "position": task_pos, "reward": task_reward}
                ))
                
                # Agent resolves local winner at exactly timeout seconds later
                self.kernel.schedule_event(Event(
                    timestamp=start_time + self.config.auction_timeout,
                    event_type=EventType.AUCTION_RESOLVE,
                    agent_id=agent.agent_id,
                    payload={"task_id": task_id}
                ))

        # Metrics logging at regular intervals
        self.kernel.schedule_event(Event(
            timestamp=dt,
            event_type=EventType.METRICS_LOG,
        ))

    # ── Event handlers ──────────────────────────────────────

    def _handle_kinematic_update(self, event: Event) -> None:
        """Process a kinematic update for one agent."""
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return

        # Agent computes velocity from local state only
        velocity = agent.compute_velocity()

        # Apply movement (returns distance)
        distance = agent.apply_movement(
            velocity, self.config.dt, self.grid.clamp_position
        )

        # Energy consumed for movement (event-driven only)
        agent.consume_movement_energy(distance, self.config.p_move)

        # Check for death
        if not agent.is_alive:
            self.alive_mask[event.agent_id] = False
            return

        # Schedule next kinematic event
        self.kernel.schedule_event(Event(
            timestamp=event.timestamp + self.config.dt,
            event_type=EventType.KINEMATIC_UPDATE,
            agent_id=event.agent_id,
        ))

    def _handle_msg_transmit(self, event: Event) -> None:
        """Process a broadcast request from one agent."""
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return

        # Phase 2C: Prepare the outbound payload from the agent
        out_msg = agent.prepare_broadcast(event.timestamp)
        
        if out_msg is None:
            # Agent intentionally silenced broadcast per local parameters
            self.kernel.schedule_event(Event(
                timestamp=event.timestamp + self.config.dt,
                event_type=EventType.MSG_TRANSMIT,
                agent_id=event.agent_id,
            ))
            return

        # Gather all positions (for KD-Tree — agents don't touch this)
        all_positions = self._get_all_positions()

        # Communication engine handles RGG, drop, latency, scheduling
        num_delivered = self.comm_engine.process_broadcasts(
            sender_id=out_msg.sender_id,
            sender_position=out_msg.position,
            sender_energy=out_msg.energy,
            sender_consensus=out_msg.consensus_state,
            sender_auction_bid=out_msg.auction_bid,
            send_time=out_msg.send_time,
            all_positions=all_positions,
            alive_mask=self.alive_mask,
            kernel=self.kernel,
        )

        # Energy consumed for communication (event-driven only)
        # Cost is per-neighbor-attempted, not per-delivered
        neighbors_count = sum(
            1 for aid in range(self.config.num_agents)
            if aid != event.agent_id and self.alive_mask[aid]
            and np.linalg.norm(
                all_positions[aid] - all_positions[event.agent_id]
            ) <= self.config.comm_radius
        )
        agent.consume_comm_energy(max(neighbors_count, 0), self.config.p_comm)

        if not agent.is_alive:
            self.alive_mask[event.agent_id] = False
            return

        # Update drop tracker
        self.drop_tracker.total_sent = self.comm_engine.total_sent
        self.drop_tracker.total_dropped = self.comm_engine.total_dropped
        self.drop_tracker.total_delivered = self.comm_engine.total_delivered

        # Schedule next broadcast
        self.kernel.schedule_event(Event(
            timestamp=event.timestamp + self.config.dt,
            event_type=EventType.MSG_TRANSMIT,
            agent_id=event.agent_id,
        ))

    def _handle_msg_deliver(self, event: Event) -> None:
        """Deliver a message to the target agent's inbox."""
        msg = event.payload
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return

        # This is the ONLY place receive_message is called —
        # via kernel dispatch, never directly by another agent.
        from src.agent.agent_core import AgentMessage

        agent.receive_message(AgentMessage(
            sender_id=msg.sender_id,
            position=msg.position,
            energy=msg.energy,
            consensus_state=msg.consensus_state,
            send_time=msg.send_time,
            auction_bid=msg.auction_bid,
        ))
        agent.process_inbox()

    def _handle_energy_update(self, event: Event) -> None:
        """Process idle energy drain."""
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return

        agent.consume_idle_energy(self.config.dt, self.config.p_idle)

        if not agent.is_alive:
            self.alive_mask[event.agent_id] = False
            return

        # Schedule next idle drain
        self.kernel.schedule_event(Event(
            timestamp=event.timestamp + self.config.dt,
            event_type=EventType.ENERGY_UPDATE,
            agent_id=event.agent_id,
        ))

    def _handle_consensus_update(self, event: Event) -> None:
        """Process an asynchronous gossip consensus update for one agent."""
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return

        agent.handle_consensus_update(epsilon=self.config.consensus_epsilon)

        if not agent.is_alive:
            self.alive_mask[event.agent_id] = False
            return

        # Schedule the next consensus update
        self.kernel.schedule_event(Event(
            timestamp=event.timestamp + self.config.consensus_dt,
            event_type=EventType.CONSENSUS_UPDATE,
            agent_id=event.agent_id,
        ))

    def _handle_auction_start(self, event: Event) -> None:
        """Process an auction start event for a single agent."""
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return
            
        payload = event.payload
        agent.handle_auction_start(
            task_id=payload["task_id"],
            task_position=payload["position"],
            task_reward=payload["reward"]
        )
        
    def _handle_auction_resolve(self, event: Event) -> None:
        """Process auction resolution on a single agent."""
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return
            
        payload = event.payload
        won = agent.handle_auction_resolve(
            task_id=payload["task_id"]
        )
        
        # Store global results for metrics/validation
        if won:
            self.auction_results.append((payload["task_id"], agent.agent_id, event.timestamp))

    def _handle_regime_update(self, event: Event) -> None:
        """Process a passive regime local evaluation."""
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return

        agent.handle_regime_update(event.timestamp)

        if not agent.is_alive:
            self.alive_mask[event.agent_id] = False
            return

        # Schedule the next regime update
        self.kernel.schedule_event(Event(
            timestamp=event.timestamp + self.config.regime.classification_interval,
            event_type=EventType.REGIME_UPDATE,
            agent_id=event.agent_id,
        ))

    def _handle_metrics_log(self, event: Event) -> None:
        """Log metrics snapshot (centralized, read-only)."""
        t = event.timestamp
        for agent in self.agents:
            self.energy_profiler.record(t, agent.agent_id, agent.energy)
            self.position_logger.record(t, agent.agent_id, agent.position)

        # Phase 3D: Global connectivity metrics compilation
        # Fully centralized analytical operation; agents are blind to this.
        from src.metrics.connectivity_metrics import compute_connectivity_metrics
        
        all_pos = self._get_all_positions()
        metrics = compute_connectivity_metrics(
            all_pos, self.config.comm_radius
        )
        
        self.connectivity_log.append({
            "time": t,
            "lcc": metrics["largest_component"],
            "component_count": metrics["component_count"],
            "connectivity_ratio": metrics["connectivity_ratio"],
            "spectral_gap": metrics["spectral_gap"]
        })

        # Phase 4: System-wide adaptation metrics compilation
        alive_agents = [a for a in self.agents if a.is_alive]
        if alive_agents:
            self.adaptation_log.append({
                "time": t,
                "coverage_gain": sum(a.coverage_gain for a in alive_agents) / len(alive_agents),
                "gossip_epsilon": sum(a.gossip_epsilon for a in alive_agents) / len(alive_agents),
                "broadcast_rate": sum(a.broadcast_rate for a in alive_agents) / len(alive_agents),
                "auction_participation": sum(a.auction_participation for a in alive_agents) / len(alive_agents),
                "velocity_scale": sum(a.velocity_scale for a in alive_agents) / len(alive_agents),
                "projection_events": sum(a.projection_events for a in self.agents)
            })

        # Schedule next metrics log
        next_t = t + self.config.dt * 5  # Log every 5 dt intervals
        if next_t <= self.config.max_time:
            self.kernel.schedule_event(Event(
                timestamp=next_t,
                event_type=EventType.METRICS_LOG,
            ))

    # ── Helpers ─────────────────────────────────────────────

    def _get_all_positions(self) -> np.ndarray:
        """Collect all agent positions into a single array (for KD-Tree)."""
        return np.array([a.position for a in self.agents])

    # ── Run ─────────────────────────────────────────────────

    def run(self) -> int:
        """Seed events and run the simulation. Returns total events dispatched."""
        self.seed_events()
        return self.kernel.run(until=self.config.max_time)

    # ── Summary ─────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary of simulation results."""
        alive = sum(1 for a in self.agents if a.is_alive)
        return {
            "total_agents": self.config.num_agents,
            "alive_agents": alive,
            "dead_agents": self.config.num_agents - alive,
            "total_energy_remaining": sum(a.energy for a in self.agents),
            "comm_sent": self.comm_engine.total_sent,
            "comm_dropped": self.comm_engine.total_dropped,
            "comm_delivered": self.comm_engine.total_delivered,
            "drop_rate": self.drop_tracker.drop_rate,
            "simulation_time": self.kernel.now,
            "regimes_distribution": self._get_regime_distribution(),
        }

    def _get_regime_distribution(self) -> dict[str, int]:
        dist = {}
        for agent in self.agents:
            if agent.is_alive:
                dist[agent.current_regime.name] = dist.get(agent.current_regime.name, 0) + 1
        return dist
