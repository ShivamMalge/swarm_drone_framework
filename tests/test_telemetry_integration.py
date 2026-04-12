"""Integration test for Phase 2A: TelemetryEmitter end-to-end verification."""

from src.core.config import SimConfig
from src.simulation import Phase1Simulation
from src.telemetry import TelemetryEmitter, TelemetryFrame

# --- Integration snippet ---
cfg = SimConfig(num_agents=20, max_time=10.0, seed=42)
sim = Phase1Simulation(cfg)
sim.seed_events()

# Emitter is initialized OUTSIDE the kernel; reads only.
emitter = TelemetryEmitter(simulation_ref=sim, config=cfg)

# Step kernel in chunks, extract frame after each step
for t_target in [2.0, 5.0, 10.0]:
    sim.kernel.run(until=t_target)
    frame: TelemetryFrame = emitter.extract_frame()
    alive = int((~frame.drone_failure_flags).sum())
    print(
        f"t={frame.time:6.2f} | alive={alive:3d} | "
        f"comps={len(frame.connected_components):2d} | "
        f"lambda2={frame.spectral_gap:.4f} | "
        f"cv={frame.consensus_variance:.4f} | "
        f"drop={frame.packet_drop_rate:.4f} | "
        f"regimes={set(frame.regime_state.values())} | "
        f"proj_events={frame.adaptive_parameters['projection_events_total']}"
    )

print("Integration OK")
