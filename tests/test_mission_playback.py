"""
Phase 2Z verification: Mission Playback determinism tests
"""

import time
import numpy as np

from src.telemetry.telemetry_frame import TelemetryFrame
from src.replay.replay_loader import ReplayLoader
from src.replay.replay_engine import ReplayEngine
from src.replay.mission_controller import MissionController


class MockLoader(ReplayLoader):
    def __init__(self, n_frames=10):
        # Generate dummy frames
        self._mock_frames = []
        for i in range(n_frames):
            f = TelemetryFrame(time=i * 0.1, positions=np.zeros((1, 2)),
                               energies=np.zeros(1), adjacency=np.zeros((1,1), dtype=np.uint8),
                               connected_components=[], spectral_gap=0.0, consensus_variance=0.0,
                               packet_drop_rate=0.0, latency=0.0, regime_state={},
                               adaptive_parameters={}, drone_failure_flags=np.zeros(1, dtype=bool),
                               agent_states=np.zeros(1))
            self._mock_frames.append(f)
            
    def frames(self):
        return self._mock_frames
        
    def __len__(self):
        return len(self._mock_frames)
        
    def __getitem__(self, idx):
        return self._mock_frames[idx]


def test_seek_and_sync():
    loader = MockLoader(100)
    engine = ReplayEngine(loader)
    controller = MissionController(engine)
    
    emitted_frames = []
    controller.frame_ready.connect(lambda f: emitted_frames.append(f))
    
    reset_count = 0
    def handle_reset():
        nonlocal reset_count
        reset_count += 1
    controller.reset_state.connect(handle_reset)
    
    # Test seeking
    controller.seek(25)
    
    assert reset_count == 1
    assert len(emitted_frames) == 1
    assert emitted_frames[0].time == 2.5 # 25 * 0.1
    
    controller.seek(50)
    assert reset_count == 2
    assert emitted_frames[-1].time == 5.0
    
    print("[PASS] MissionController synchronizes state resets with explicit seek triggers")


def test_playback_progression():
    loader = MockLoader(10)
    engine = ReplayEngine(loader)
    controller = MissionController(engine)
    
    emitted_frames = []
    controller.frame_ready.connect(lambda f: emitted_frames.append(f))
    
    controller._engine.play()
    
    # Tick manually 3 times
    controller._tick()
    controller._tick()
    controller._tick()
    
    assert len(emitted_frames) == 3
    assert emitted_frames[0].time == 0.0
    assert emitted_frames[2].time == 0.2
    
    print("[PASS] Tick iterations gracefully emit deterministic sequences")


def test_stop_state_loop():
    loader = MockLoader(10)
    engine = ReplayEngine(loader)
    controller = MissionController(engine)
    
    resets = 0
    controller.reset_state.connect(lambda: globals().update(resets=globals().get("resets", 0) + 1))
    # Wait, using globals in test is bad, let's keep it simple:
    val = [0]
    controller.reset_state.connect(lambda: val.__setitem__(0, val[0] + 1))
    
    controller.seek(5)
    assert engine.index == 5
    
    controller.stop()
    assert engine.index == 0
    assert val[0] == 2 # 1 for seek, 1 for stop
    
    print("[PASS] Stop explicitly anchors to origin isolating deterministic arrays safely")


if __name__ == "__main__":
    test_seek_and_sync()
    test_playback_progression()
    test_stop_state_loop()
    print("\nAll Phase 2Z tests passed.")
