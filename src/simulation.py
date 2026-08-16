"""
Phase 1 Simulation Orchestrator.

Wires the kernel, environment, agents, communication engine, and metrics
into a single event-driven simulation loop.

The orchestrator is the ONLY place where the kernel and agent array co-exist.
Agents never hold references to the kernel, orchestrator, or each other.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

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
from src.regime.classifier import Regime
from src.metrics.drop_rate_tracker import DropRateTracker
from src.metrics.energy_profiler import EnergyProfiler
from src.metrics.position_logger import PositionLogger
from src.metrics.kernel_logger import KernelLogger


class Phase1Simulation:
    """
    Event-driven simulation orchestrator for Phase 1.

    Initialises all modules and registers event handlers with the kernel.
    Agents are stored in a flat array; they NEVER receive a reference
    to this orchestrator, the kernel, or each other.
    """

    def __init__(self, config: SimConfig, log_suffix: str = "") -> None:
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
                theta_safe_enabled=config.theta_safe_enabled,
                test_mode=config.test_mode,
                global_info_enabled=config.global_info_enabled,
                tuning_alpha=config.tuning_alpha,
            )
            
            # Experiment 2: Thermodynamics - Randomized low initial energy
            if config.test_mode == "thermodynamics":
                # Override energy model with randomized low values (e.g., 5.0 to 15.0)
                low_energy = self._streams["positions"].uniform(5.0, 15.0)
                agent._energy._energy = low_energy
                agent._energy._initial_energy = low_energy
                
            self.agents.append(agent)

        if config.global_info_enabled:
            assert self.agents[0].global_info_enabled is True, "global_info_enabled did not propagate to AgentCore"

        self.alive_mask = np.ones(config.num_agents, dtype=bool)

        # Live (N, 2) position matrix, kept in sync by _handle_kinematic_update.
        # apply_movement is the only mutator of agent positions, so this stays
        # exact; it replaces rebuilding the array from N copies on every call.
        self._positions = np.array([a._position for a in self.agents], dtype=float)

        # K-D Tree over _positions, rebuilt lazily. _positions_dirty is set by
        # _handle_kinematic_update, the only mutator of agent positions, so the
        # tree can never be served stale regardless of how events interleave.
        #
        # In practice KINEMATIC_UPDATE (priority 1) fully precedes MSG_TRANSMIT
        # (priority 3) at each timestamp, so positions are static across the
        # whole broadcast phase and this costs one rebuild per tick. That is a
        # performance property, not a correctness assumption: any interleaving
        # simply triggers more rebuilds.
        self._position_tree = None
        self._positions_dirty = True
        self.tree_rebuilds = 0

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
        self.kernel_logger = KernelLogger(test_mode=config.test_mode, output_dir=config.log_dir, filename_suffix=log_suffix)
        
        # Phase 2C: Global metric tracker for testing
        self.auction_results: list[tuple[str, int, float]] = [] # task_id, winner_id, time
        self.active_tasks: dict[str, np.ndarray] = {}  # Patch 2: Global record of unconsumed tasks

        # Phase 3D/4: Global metric logs
        self.connectivity_log: list[dict] = []
        self.adaptation_log: list[dict] = []
        self.time_of_death: float | None = None
        self.time_to_50pct_attrition: float | None = None
        

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
        self.kernel.register_handler(
            EventType.TASK_SPAWN, self._handle_task_spawn
        )
        self.kernel.register_handler(
            EventType.ENV_UPDATE, self._handle_env_update
        )

    def _check_death_conditions(self, timestamp: float) -> None:
        alive_count = self.alive_mask.sum()
        if self.time_to_50pct_attrition is None and alive_count <= 0.5 * self.config.num_agents:
            self.time_to_50pct_attrition = timestamp
        if self.time_of_death is None and alive_count == 0:
            self.time_of_death = timestamp

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
            
            # Stagger initial regime checks to prevent CPU spikes
            regime_delta = self.config.regime.dwell_time / float(self.config.num_agents)
            regime_start_time = dt + (agent.agent_id * regime_delta)
            
            self.kernel.schedule_event(Event(
                timestamp=regime_start_time,
                event_type=EventType.REGIME_UPDATE,
                agent_id=agent.agent_id,
            ))
        # Phase 2C: Seed the first task spawn (Poisson process will chain subsequent spawns)
        self.kernel.schedule_event(Event(
            timestamp=0.5,
            event_type=EventType.TASK_SPAWN,
            payload={"task_idx": 0}
        ))

        # Metrics logging at regular intervals
        self.kernel.schedule_event(Event(
            timestamp=dt,
            event_type=EventType.METRICS_LOG,
        ))

        # Exogenous environment ramp (percolation experiment only), on the same
        # cadence the metrics handler previously applied it at.
        if self.config.test_mode == "percolation":
            self.kernel.schedule_event(Event(
                timestamp=dt,
                event_type=EventType.ENV_UPDATE,
            ))

    # ── Event handlers ──────────────────────────────────────

    def _handle_kinematic_update(self, event: Event) -> None:
        """Process a kinematic update for one agent."""
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return

        # True Oracle: compute true global Voronoi centroid
        if self.config.global_info_enabled and agent.coverage_gain > 0.05:
            agent.oracle_centroid = self._compute_oracle_centroid(event.agent_id)

        # Agent computes velocity from local state only.
        # Tick accounting lives HERE, not inside compute_velocity(): a tick is a
        # kinematic event, and the metrics logger also calls compute_velocity()
        # for reporting. Counting inside it conflated logging with simulation
        # (audit F-17).
        agent.total_ticks += 1
        if agent.global_info_enabled or agent.coverage_gain > 0.05:
            agent.coverage_ticks += 1

        velocity = agent.compute_velocity()
        # Cached so the metrics logger can report the commanded velocity without
        # re-invoking compute_velocity(), whose random-walk branch draws from the
        # agent RNG and would otherwise perturb the trajectory it is measuring.
        agent.last_velocity = velocity

        # Apply movement (returns distance)
        distance = agent.apply_movement(
            velocity, self.config.dt, self.grid.clamp_position
        )
        # Keep the orchestrator's position matrix exact. apply_movement is the
        # sole mutator of agent positions, so syncing here is sufficient.
        self._positions[event.agent_id] = agent._position
        self._positions_dirty = True

        # Energy consumed for movement (event-driven only)
        agent.consume_movement_energy(distance, self.config.p_move)

        # Check for death
        if not agent.is_alive:
            self.alive_mask[event.agent_id] = False
            self._check_death_conditions(event.timestamp)
            return

        # Patch 2: Physical Spatial Resolution of Tasks
        if agent.active_task_id is not None:
            task_id = agent.active_task_id
            if task_id in self.active_tasks:
                task_pos = self.active_tasks[task_id]
                dist = float(np.linalg.norm(agent.position - task_pos))
                if dist <= self.config.r_task:
                    # Task consumed physically!
                    del self.active_tasks[task_id]
                    agent.handle_task_completion(task_id)
                    # NOTE: We do NOT broadcast. Other agents only find out by arriving.
            else:
                # Task is already missing (consumed by someone else)
                # Check if agent has arrived at the location to "observe" it's missing
                if agent.active_task_pos is not None:
                    dist_to_loc = float(np.linalg.norm(agent.position - agent.active_task_pos))
                    if dist_to_loc <= self.config.r_task:
                        agent.handle_task_completion(task_id)

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

        if self.config.global_info_enabled:
            # Oracle bypasses the CommunicationEngine entirely (instant, lossless).
            # This branch is evaluated BEFORE prepare_broadcast(): the oracle builds
            # no message, so calling prepare_broadcast() here would advance the
            # agent's RNG stream (agent_core.py:304, 312) for a payload that is
            # immediately discarded, perturbing the oracle's trajectory relative to
            # every other arm. Same class of determinism leak as the metrics-logger
            # side effects; see audit F-17.
            agent.consume_comm_energy(
                self._oracle_recipient_count(event.agent_id),
                self.config.p_comm,
                1.0,
            )

            if not agent.is_alive:
                self.alive_mask[event.agent_id] = False
                self._check_death_conditions(event.timestamp)

            self.kernel.schedule_event(Event(
                timestamp=event.timestamp + self.config.dt,
                event_type=EventType.MSG_TRANSMIT,
                agent_id=event.agent_id,
            ))
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
        num_delivered, neighbors_count = self.comm_engine.process_broadcasts(
            sender_id=out_msg.sender_id,
            sender_position=out_msg.position,
            sender_energy=out_msg.energy,
            sender_consensus=out_msg.consensus_state,
            sender_auction_bid=out_msg.auction_bid,
            send_time=out_msg.send_time,
            sender_tx_radius=out_msg.tx_radius, # Phase 1
            all_positions=all_positions,
            alive_mask=self.alive_mask,
            kernel=self.kernel,
            position_tree=self._get_position_tree(),
        )

        # Energy consumed for communication (event-driven only).
        # Cost is per-neighbor-attempted, scaled quadratically by transmission
        # radius. neighbors_count comes back from process_broadcasts, which
        # already evaluated these exact distances vectorised; recomputing them
        # here as a per-element Python loop was ~28% of total runtime
        # (3.3M scalar np.linalg.norm calls at N=200).
        power_multiplier = out_msg.tx_radius / self.config.comm_radius_base
        agent.consume_comm_energy(max(neighbors_count, 0), self.config.p_comm, power_multiplier)

        if not agent.is_alive:
            self.alive_mask[event.agent_id] = False
            self._check_death_conditions(event.timestamp)
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
            tx_radius=msg.tx_radius,
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
            self._check_death_conditions(event.timestamp)
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

        if self.config.global_info_enabled:
            # True Oracle: Instant access to true global average
            alive_agents = [a for a in self.agents if a.is_alive]
            if alive_agents:
                global_avg = sum(a.consensus_state for a in alive_agents) / len(alive_agents)
                agent.consensus_state = global_avg
        else:
            agent.handle_consensus_update(
                current_time=event.timestamp,
                dt=self.config.dt,
                epsilon=self.config.consensus_epsilon
            )

        if not agent.is_alive:
            self.alive_mask[event.agent_id] = False
            self._check_death_conditions(event.timestamp)
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
        
        if self.config.global_info_enabled:
            # Oracle handles auctions globally at resolution time, agents don't bid locally
            return
            
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
        task_id = payload["task_id"]
        
        won = False
        if self.config.global_info_enabled:
            # True Oracle: Centralized Hungarian Assignment
            from scipy.optimize import linear_sum_assignment
            alive_indices = np.where(self.alive_mask)[0]
            if len(alive_indices) > 0 and task_id in self.active_tasks:
                task_pos = self.active_tasks[task_id]
                # Distance array
                cost_matrix = []
                for idx in alive_indices:
                    a = self.agents[idx]
                    dist = float(np.linalg.norm(a.position - task_pos))
                    cost_matrix.append([dist])
                
                # Assign 1 agent to 1 task
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                if len(row_ind) > 0:
                    winner_idx = alive_indices[row_ind[0]]
                    if winner_idx == agent.agent_id:
                        agent.active_task_id = task_id
                        agent.active_task_pos = task_pos
                        won = True
        else:
            won = agent.handle_auction_resolve(
                task_id=task_id
            )
        
        # Store global results for metrics/validation
        if won:
            self.auction_results.append((payload["task_id"], agent.agent_id, event.timestamp))

    def _handle_task_spawn(self, event: Event) -> None:
        """Continuously spawn tasks across the simulation using a Poisson process."""
        task_idx = event.payload["task_idx"]
        task_id = f"task_{task_idx}"
        task_pos = self.grid.random_positions(1, self._streams["task_spawner"])[0]
        task_reward = float(self._streams["task_spawner"].uniform(50.0, 150.0))
        
        start_time = event.timestamp
        
        for agent in self.agents:
            self.kernel.schedule_event(Event(
                timestamp=start_time,
                event_type=EventType.AUCTION_START,
                agent_id=agent.agent_id,
                payload={"task_id": task_id, "position": task_pos, "reward": task_reward}
            ))
            
            self.kernel.schedule_event(Event(
                timestamp=start_time + self.config.auction_timeout,
                event_type=EventType.AUCTION_RESOLVE,
                agent_id=agent.agent_id,
                payload={"task_id": task_id}
            ))
            
        self.active_tasks[task_id] = task_pos
        
        # Schedule next task spawn using an exponential inter-arrival time (lambda = 1/15)
        next_interval = self._streams["task_spawner"].exponential(15.0)
        next_time = start_time + next_interval
        
        if next_time <= self.config.max_time:
            self.kernel.schedule_event(Event(
                timestamp=next_time,
                event_type=EventType.TASK_SPAWN,
                payload={"task_idx": task_idx + 1}
            ))

    def _handle_regime_update(self, event: Event) -> None:
        """Process a passive regime local evaluation."""
        agent = self.agents[event.agent_id]
        if not agent.is_alive:
            return

        agent.handle_regime_update(event.timestamp)

        if not agent.is_alive:
            self.alive_mask[event.agent_id] = False
            self._check_death_conditions(event.timestamp)
            return

        # Schedule the next regime update
        self.kernel.schedule_event(Event(
            timestamp=event.timestamp + self.config.regime.dwell_time,
            event_type=EventType.REGIME_UPDATE,
            agent_id=event.agent_id,
        ))

    def _handle_env_update(self, event: Event) -> None:
        """
        Advance the exogenous environment.

        Owns the percolation experiment's interference ramp. This is the only
        place environmental physics is mutated on a schedule; the metrics
        handler must remain a pure observer.
        """
        self.interference.psi_max += (
            self.config.interference_growth_rate * self.config.dt * 5
        )
        next_t = event.timestamp + self.config.dt * 5
        if next_t <= self.config.max_time:
            self.kernel.schedule_event(Event(
                timestamp=next_t,
                event_type=EventType.ENV_UPDATE,
            ))

    def _handle_metrics_log(self, event: Event) -> None:
        """Log metrics snapshot (centralized, strictly read-only)."""
        t = event.timestamp
        alive_agents = [a for a in self.agents if a.is_alive]
        hibernating = [a for a in alive_agents if a.current_strategy.name == "ENERGY_CONSERVATION"]
        hib_cov = sum(a.coverage_gain for a in hibernating) / len(hibernating) if hibernating else 0.0
        hib_auc = sum(a.auction_participation for a in hibernating) / len(hibernating) if hibernating else 0.0
        
        # We will dynamically inject these into the snapshot dictionary
        hib_data = {"hibernation_coverage": hib_cov, "hibernation_auction": hib_auc}
        
        for agent in self.agents:
            self.energy_profiler.record(t, agent.agent_id, agent.energy)
            self.position_logger.record(t, agent.agent_id, agent.position)

        # Phase 3D: Global connectivity metrics compilation
        # Fully centralized analytical operation; agents are blind to this.
        from src.metrics.connectivity_metrics import compute_connectivity_metrics
        from src.regime.classifier import Regime
        
        # Connectivity is a property of the LIVING swarm. Dead agents keep
        # their final position forever, and including them meant the reported
        # lambda_2 described the geometric arrangement of corpses: a fully
        # extinct swarm still logged lambda_2 = 0.2919 with LCC = 100 while the
        # alive-only value was 0.0 (audit F-03).
        alive_pos = self._get_all_positions()[self.alive_mask]
        n_alive = len(alive_pos)
        if n_alive > 1:
            metrics = compute_connectivity_metrics(
                alive_pos, self.config.comm_radius
            )
        else:
            # 0 or 1 survivors: no edges can exist, so lambda_2 is 0 by definition.
            metrics = {
                "largest_component": n_alive,
                "component_count": n_alive,
                "connectivity_ratio": float(n_alive),
                "spectral_gap": 0.0,
            }


        # ── Experiment 1: Percolation & Fragmentation ──
        if self.config.test_mode == "percolation":
            # NOTE: the interference ramp that drives this experiment used to be
            # applied HERE, inside the metrics handler -- a function documented
            # as "centralized, read-only" that was in fact driving the physics
            # (audit F-17). It now has its own ENV_UPDATE event; see
            # _handle_env_update.

            # Count regimes
            regime_dist = self._get_regime_distribution()
            stable_count = regime_dist.get(Regime.STABLE.name, 0)
            fragmented_count = regime_dist.get(Regime.FRAGMENTED.name, 0)
            
            # Mean of the agents' own Eq.(2) local mixing proxy.
            # This column was previously the mean believed-NEIGHBOUR-COUNT, logged
            # and plotted under the name avg_local_lambda_proxy and the legend
            # \hat{lambda}_2 -- a quantity with no relation to Eq.(2), scaled by an
            # unexplained /100 at plot time. Meanwhile the actual proxy
            # (AgentCore._lambda_2_proxy) was computed every regime update and
            # never read by anything (audit F-19).
            avg_local_lambda = 0.0
            if alive_agents:
                avg_local_lambda = sum(a._lambda_2_proxy for a in alive_agents) / len(alive_agents)
            
            # Count edges: query_pairs returns undirected pairs.
            # Alive-only, for the same reason as lambda_2 above: this column
            # previously ROSE to 1101 edges while only 7 of 200 agents were
            # alive, because corpses kept contributing edges (audit F-03).
            if n_alive > 1:
                tree = KDTree(alive_pos)
                active_edges = len(tree.query_pairs(self.config.comm_radius))
            else:
                active_edges = 0
            
            self.kernel_logger.log_snapshot({
                "time": t,
                "true_lambda_2": metrics["spectral_gap"],
                "avg_local_lambda_proxy": avg_local_lambda,
                "active_network_edges": active_edges,
                "count_regime_stable": stable_count,
                "count_regime_fragmented": fragmented_count
            ,
                "hibernation_coverage": hib_cov,
                "hibernation_auction": hib_auc
            })

        # ── Experiment 2: Thermodynamic Energy Cascade ──
        elif self.config.test_mode == "thermodynamics":
            total_energy = sum(a.energy for a in alive_agents) if alive_agents else 0.0
            # Mean cell area = Total area / count
            cell_area = (self.config.grid_width * self.config.grid_height) / len(alive_agents) if alive_agents else 0.0
            
            # Average kinematic velocity, read from the cached command rather
            # than recomputed: compute_velocity() consumes agent RNG (audit F-17).
            all_velocities = [float(np.linalg.norm(a.last_velocity)) for a in alive_agents]
            avg_v = sum(all_velocities) / len(all_velocities) if all_velocities else 0.0
            
            self.kernel_logger.log_snapshot({
                "time": t,
                "active_drone_count": len(alive_agents),
                "total_system_energy": total_energy,
                "avg_voronoi_cell_area": cell_area,
                "avg_kinematic_velocity": avg_v
            ,
                "hibernation_coverage": hib_cov,
                "hibernation_auction": hib_auc
            })

        # ── Experiment 3: Θ_safe Control Stability ──
        elif self.config.test_mode == "stability":
            # Track max velocity cmd, from the cached command (see above).
            all_v = [float(np.linalg.norm(a.last_velocity)) for a in alive_agents]
            max_v = max(all_v) if all_v else 0.0
            
            snap = {
                "time": t,
                "avg_network_delay": self.config.latency_mean,
                "clamped_events_count": sum(a.projection_events for a in self.agents),
                "hibernation_coverage": hib_cov,
                "hibernation_auction": hib_auc
            }
            if not self.config.theta_safe_enabled:
                snap["max_velocity_cmd_run_A"] = max_v
            else:
                snap["max_velocity_cmd_run_B"] = max_v
                
            self.kernel_logger.log_snapshot(snap)

        self.connectivity_log.append({
            "time": t,
            "n_alive": int(n_alive),
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
                "projection_events": sum(a.projection_events for a in self.agents),
                "total_tuning_updates": sum(a.total_tuning_updates for a in self.agents),
                "total_parameter_shift": sum(a.total_parameter_shift for a in self.agents),
                "max_parameter_shift": max((a.max_parameter_shift for a in self.agents), default=0.0)
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
        """
        Return the live (N, 2) position matrix.

        READ-ONLY for callers. This is the orchestrator's own array, kept in
        sync by _handle_kinematic_update rather than rebuilt per call; it was
        previously reconstructed from N .position copies on every broadcast
        (~11% of runtime at N=200, see rust_conversion_plan.md R0 results).
        """
        return self._positions

    def _get_position_tree(self):
        """
        Return a K-D Tree over current positions, rebuilding only if they moved.

        Rebuild count is instrumented so the O(N log N) claim in the manuscript
        can be checked rather than asserted.
        """
        if self._positions_dirty or self._position_tree is None:
            self._position_tree = RGGBuilder.build_tree(self._positions)
            self._positions_dirty = False
            self.tree_rebuilds += 1
        return self._position_tree

    def _oracle_recipient_count(self, sender_id: int) -> int:
        """
        Number of recipients the oracle is billed for on one broadcast.

        Selected by config.oracle_comm_billing; see SimConfig for the rationale
        behind each accounting model.
        """
        if self.config.oracle_comm_billing == "per_neighbour":
            # Identical accounting to the decentralized arms: living agents
            # inside the sender's transmission radius.
            positions = self._get_all_positions()
            distances = np.linalg.norm(positions - positions[sender_id], axis=1)
            in_range = (distances <= self.config.comm_radius) & self.alive_mask
            return int(in_range.sum()) - (1 if self.alive_mask[sender_id] else 0)

        if self.config.oracle_comm_billing != "all_to_all":
            raise ValueError(
                f"Unknown oracle_comm_billing: {self.config.oracle_comm_billing!r}. "
                "Expected 'all_to_all' or 'per_neighbour'."
            )

        # Global awareness is billed as a global broadcast.
        return max(0, int(self.alive_mask.sum()) - 1)

    def _compute_oracle_centroid(self, agent_id: int) -> np.ndarray | None:
        """Centralized Voronoi centroid computation for True Oracle."""
        from scipy.spatial import Voronoi
        if sum(self.alive_mask) < 4:
            return None
            
        alive_pos = self._get_all_positions()[self.alive_mask]
        w, h = self.config.grid_width, self.config.grid_height
        boundary = np.array([[-w, -h], [-w, 2*h], [2*w, 2*h], [2*w, -h]])
        pts = np.vstack([alive_pos, boundary])
        
        try:
            vor = Voronoi(pts)
            alive_indices = np.where(self.alive_mask)[0]
            idx_in_alive = np.where(alive_indices == agent_id)[0]
            if len(idx_in_alive) == 0:
                return None
                
            region_idx = vor.point_region[idx_in_alive[0]]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                return None
                
            polygon = vor.vertices[region]
            polygon = np.clip(polygon, 0, [w, h])
            return np.mean(polygon, axis=0)
        except Exception:
            return None

    # ── Run ─────────────────────────────────────────────────

    def run(self) -> int:
        """Seed events and run the simulation. Returns total events dispatched."""
        self.seed_events()
        return self.kernel.run(until=self.config.max_time)

    # ── Summary ─────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary of simulation results."""
        alive = sum(1 for a in self.agents if a.is_alive)
        
        total_ticks_sum = sum(a.total_ticks for a in self.agents)
        coverage_ticks_sum = sum(a.coverage_ticks for a in self.agents)
        coverage_rate = (coverage_ticks_sum / total_ticks_sum) if total_ticks_sum > 0 else 0.0
        
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
            "time_of_death": self.time_of_death,
            "time_to_50pct_attrition": self.time_to_50pct_attrition,
            "regimes_distribution": self._get_regime_distribution(),
            "coverage_completion_rate": coverage_rate,
        }

    def close_loggers(self) -> None:
        """Close external loggers (e.g., for experiments)."""
        self.kernel_logger.close()

    def _get_regime_distribution(self) -> dict[str, int]:
        dist = {}
        for agent in self.agents:
            if agent.is_alive:
                dist[agent.current_regime.name] = dist.get(agent.current_regime.name, 0) + 1
        return dist
