"""
Unit tests for SimulationKernel — event ordering, causal dispatch, clock.
"""

from src.core.event import Event, EventType, reset_sequence_counter
from src.core.kernel import SimulationKernel


class TestEventOrdering:
    """Events must be dispatched in strict (timestamp, priority, sequence_id) order."""

    def setup_method(self):
        reset_sequence_counter()
        self.kernel = SimulationKernel()
        self.dispatch_log = []
        for et in EventType:
            self.kernel.register_handler(
                et, lambda e, log=self.dispatch_log: log.append(e)
            )

    def test_timestamp_ordering(self):
        """Events with different timestamps are dispatched chronologically."""
        self.kernel.schedule_event(Event(3.0, EventType.KINEMATIC_UPDATE, agent_id=0))
        self.kernel.schedule_event(Event(1.0, EventType.KINEMATIC_UPDATE, agent_id=1))
        self.kernel.schedule_event(Event(2.0, EventType.KINEMATIC_UPDATE, agent_id=2))
        self.kernel.run(until=10.0)
        assert [e.agent_id for e in self.dispatch_log] == [1, 2, 0]

    def test_priority_ordering_at_same_timestamp(self):
        """At the same timestamp, lower EventType value fires first."""
        self.kernel.schedule_event(Event(1.0, EventType.MSG_DELIVER, agent_id=0))
        self.kernel.schedule_event(Event(1.0, EventType.ENV_UPDATE, agent_id=1))
        self.kernel.schedule_event(Event(1.0, EventType.KINEMATIC_UPDATE, agent_id=2))
        self.kernel.run(until=10.0)
        types = [e.event_type for e in self.dispatch_log]
        assert types == [EventType.ENV_UPDATE, EventType.KINEMATIC_UPDATE, EventType.MSG_DELIVER]

    def test_sequence_id_ordering_at_same_timestamp_and_priority(self):
        """At same timestamp and priority, earlier-scheduled fires first."""
        reset_sequence_counter()
        e1 = Event(1.0, EventType.KINEMATIC_UPDATE, agent_id=0)
        e2 = Event(1.0, EventType.KINEMATIC_UPDATE, agent_id=1)
        e3 = Event(1.0, EventType.KINEMATIC_UPDATE, agent_id=2)
        self.kernel.schedule_event(e1)
        self.kernel.schedule_event(e2)
        self.kernel.schedule_event(e3)
        self.kernel.run(until=10.0)
        assert [e.agent_id for e in self.dispatch_log] == [0, 1, 2]

    def test_no_past_scheduling(self):
        """Cannot schedule events in the past."""
        self.kernel.schedule_event(Event(5.0, EventType.KINEMATIC_UPDATE))
        self.kernel.run(until=5.0)
        import pytest
        with pytest.raises(ValueError, match="past"):
            self.kernel.schedule_event(Event(3.0, EventType.KINEMATIC_UPDATE))

    def test_clock_advances_monotonically(self):
        """Clock never goes backward."""
        self.kernel.schedule_event(Event(1.0, EventType.KINEMATIC_UPDATE))
        self.kernel.schedule_event(Event(3.0, EventType.KINEMATIC_UPDATE))
        self.kernel.run(until=5.0)
        assert self.kernel.now == 3.0

    def test_run_stops_at_until(self):
        """Events beyond the 'until' boundary are not dispatched."""
        self.kernel.schedule_event(Event(1.0, EventType.KINEMATIC_UPDATE))
        self.kernel.schedule_event(Event(10.0, EventType.KINEMATIC_UPDATE))
        dispatched = self.kernel.run(until=5.0)
        assert dispatched == 1
        assert self.kernel.pending_count == 1
