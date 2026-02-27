"""
Message data structure for inter-agent communication.

Messages are created by agents (outbox) and delivered by the kernel
via MSG_DELIVER events.  The payload contains the sender's state
snapshot at send-time — inherently stale upon receipt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Message:
    """
    An inter-agent communication packet.

    Attributes
    ----------
    sender_id : int
        Agent that originated this message.
    receiver_id : int
        Target agent.
    position : np.ndarray
        Sender's position at send_time.
    energy : float
        Sender's energy at send_time.
    consensus_state : float
        Sender's consensus value at send_time.
    send_time : float
        Simulation time at which the sender created the message.
    """

    sender_id: int
    receiver_id: int
    position: np.ndarray
    energy: float
    consensus_state: float
    send_time: float
