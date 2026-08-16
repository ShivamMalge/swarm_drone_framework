"""
Communication Engine — orchestrates RGG, drop, latency, and event scheduling.

This is the ONLY module that bridges agent broadcasts to the kernel event queue.
Agents submit broadcasts; this engine evaluates topology, drops packets,
samples latency, and schedules MSG_DELIVER events through the kernel.

No agent ever calls another agent's receive_message() directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.communication.latency_model import LatencyModel
from src.communication.message import Message
from src.communication.packet_drop import PacketDropSampler
from src.communication.rgg_builder import RGGBuilder
from src.core.event import Event, EventType

if TYPE_CHECKING:
    from src.core.kernel import SimulationKernel
    from src.environment.interference_field import InterferenceField


class CommunicationEngine:
    """
    Processes agent broadcasts into scheduled MSG_DELIVER events.

    Parameters
    ----------
    rgg_builder : RGGBuilder
        Ephemeral neighbor-list builder.
    drop_sampler : PacketDropSampler
        Bernoulli drop evaluator.
    latency_model : LatencyModel
        Delivery delay sampler.
    interference_field : InterferenceField
        Exogenous ψ(q,t) evaluator.
    """

    def __init__(
        self,
        rgg_builder: RGGBuilder,
        drop_sampler: PacketDropSampler,
        latency_model: LatencyModel,
        interference_field: InterferenceField,
    ) -> None:
        self._rgg = rgg_builder
        self._drop = drop_sampler
        self._latency = latency_model
        self._psi = interference_field

        # Statistics (for metrics, NOT accessible by agents)
        self.total_sent: int = 0
        self.total_dropped: int = 0
        self.total_delivered: int = 0

    def process_broadcasts(
        self,
        sender_id: int,
        sender_position: np.ndarray,
        sender_energy: float,
        sender_consensus: float,
        sender_auction_bid: tuple[str, float, int] | None,
        send_time: float,
        sender_tx_radius: float,  # Phase 1: Dynamic Transmission Power
        all_positions: np.ndarray,
        alive_mask: np.ndarray,
        kernel: SimulationKernel,
        position_tree=None,
    ) -> tuple[int, int]:
        """
        Evaluate connectivity, drop/deliver messages, schedule events.

        Parameters
        ----------
        sender_id : int
            Broadcasting agent index.
        sender_position : np.ndarray
            Sender's current position.
        sender_energy : float
            Sender's current energy.
        sender_consensus : float
            Sender's current consensus state.
        send_time : float
            Current simulation time.
        all_positions : np.ndarray
            Positions of ALL agents (used by RGG builder for KD-Tree).
            This is read-only — agents do NOT access this array.
        alive_mask : np.ndarray
            Boolean mask of alive agents.
        kernel : SimulationKernel
            Event queue for scheduling deliveries.

        Returns
        -------
        tuple[int, int]
            (delivery events scheduled, living agents inside sender_tx_radius).

            The second value is returned so the orchestrator can bill
            transmission energy without recomputing the same distances. It
            counts exactly the agents for which total_sent was incremented:
            alive, not the sender, and within sender_tx_radius.
        """
        # Evaluate neighbors dynamically using sender's unique tx_radius
        psi_val = self._psi.evaluate(sender_position, send_time)
        delivered = 0
        in_range = 0

        # Spatial candidate retrieval. Previously this scanned every agent,
        # O(N) per broadcast and therefore O(N^2) per tick, while the manuscript
        # claimed O(N log N) K-D Tree partitioning (audit F-22). The tree is
        # supplied by the orchestrator, which owns position state and knows when
        # it changes; if none is supplied we fall back to the full scan so this
        # module stays usable standalone.
        #
        # The tree is only a CANDIDATE filter. The exact admission test below is
        # unchanged (vectorised norm, `<= sender_tx_radius`, ascending index
        # order), so per-neighbour RNG consumption is bit-identical to the scan.
        if position_tree is not None:
            candidates = RGGBuilder.query_radius(
                position_tree, sender_position, sender_tx_radius
            )
        else:
            candidates = range(len(all_positions))

        candidates = [
            i for i in candidates if i != sender_id and alive_mask[i]
        ]
        if not candidates:
            return 0, 0

        distances = np.linalg.norm(all_positions[candidates] - sender_position, axis=1)

        for slot, nbr_id in enumerate(candidates):
            dist = float(distances[slot])
            if dist > sender_tx_radius:
                continue

            in_range += 1
            self.total_sent += 1

            # Evaluate non-linear packet drop based on SNR / distance
            if self._drop.should_drop(psi_val, dist, sender_tx_radius):
                self.total_dropped += 1
                continue  # Silent discard — sender is unaware

            # Sample delivery latency
            delay = self._latency.sample()

            # Create the message
            msg = Message(
                sender_id=sender_id,
                receiver_id=nbr_id,
                position=sender_position.copy(),
                energy=sender_energy,
                consensus_state=sender_consensus,
                send_time=send_time,
                tx_radius=sender_tx_radius,
                auction_bid=sender_auction_bid,
            )

            # Schedule MSG_DELIVER event through the kernel
            deliver_event = Event(
                timestamp=send_time + delay,
                event_type=EventType.MSG_DELIVER,
                agent_id=nbr_id,
                payload=msg,
            )
            kernel.schedule_event(deliver_event)
            self.total_delivered += 1
            delivered += 1

        return delivered, in_range
