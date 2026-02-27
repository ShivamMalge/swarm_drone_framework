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

        # Gather all positions (for KD-Tree — agents don't touch this)
        all_positions = self._get_all_positions()

        # Communication engine handles RGG, drop, latency, scheduling
        num_delivered = self.comm_engine.process_broadcasts(
            sender_id=event.agent_id,
            sender_position=agent.position,
            sender_energy=agent.energy,
            sender_consensus=agent.consensus_state,
            send_time=event.timestamp,
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

    def _handle_metrics_log(self, event: Event) -> None:
        """Log metrics snapshot (centralized, read-only)."""
        t = event.timestamp
        for agent in self.agents:
            self.energy_profiler.record(t, agent.agent_id, agent.energy)
            self.position_logger.record(t, agent.agent_id, agent.position)

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
        }
