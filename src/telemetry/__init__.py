"""
Telemetry sub-package — read-only observer pipeline (Phase 2).

Public surface:
  TelemetryFrame   – structured DES snapshot dataclass   (2A)
  TelemetryEmitter – extraction bridge (DES → frame)     (2A)
  TelemetryBuffer  – single-slot latest-frame holder     (2B)
"""

from src.telemetry.telemetry_frame import TelemetryFrame
from src.telemetry.telemetry_emitter import TelemetryEmitter
from src.telemetry.telemetry_buffer import TelemetryBuffer

__all__ = ["TelemetryFrame", "TelemetryEmitter", "TelemetryBuffer"]
