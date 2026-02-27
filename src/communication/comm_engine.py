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
        send_time: float,
        all_positions: np.ndarray,
        alive_mask: np.ndarray,
        kernel: SimulationKernel,
    ) -> int:
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
        int
            Number of delivery events scheduled.
        """
        # Build ephemeral neighbor list (not persisted)
        neighbor_lists = self._rgg.build_neighbor_lists(all_positions)
        neighbors = neighbor_lists.get(sender_id, [])

        delivered = 0
        psi_val = self._psi.evaluate(sender_position, send_time)

        for nbr_id in neighbors:
            if not alive_mask[nbr_id]:
                continue

            self.total_sent += 1

            # Evaluate packet drop
            if self._drop.should_drop(psi_val):
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
                send_time=send_time,
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

        return delivered
