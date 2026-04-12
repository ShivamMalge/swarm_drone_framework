"""
system_event.py — Immutable event schema for the Event Fusion layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemEvent:
    """
    Immutable record of a high-level system event detected by the
    EventFusionEngine.  All numeric fields are clamped at construction
    time so downstream consumers never encounter NaN or out-of-range
    values.

    Fields
    ------
    type : str
        Short label, e.g. ``"collapse"``, ``"instability"``, ``"recovery"``.
    severity : float
        Impact magnitude, clamped to [0.0, 1.0].
    confidence : float
        Fusion confidence, clamped to [0.0, 1.0].
    timestamp : float
        Simulation clock at the moment of detection.
    frame_index : int
        Frame counter at the moment of detection.
    source : str
        Originating module, e.g. ``"fusion"``, ``"percolation"``.
    """

    type: str
    severity: float
    confidence: float
    timestamp: float
    frame_index: int
    source: str

    def __post_init__(self) -> None:
        # frozen=True prevents normal setattr, use object.__setattr__
        object.__setattr__(self, "severity",
                           max(0.0, min(1.0, float(self.severity))))
        object.__setattr__(self, "confidence",
                           max(0.0, min(1.0, float(self.confidence))))
