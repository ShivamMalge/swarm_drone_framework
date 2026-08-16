import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

LOG_DIR = "logs"
OUTPUT_DIR = "plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Configure plot style for IEEE standards
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (8, 5),
    "figure.dpi": 300
})

DT = 0.1  # 100ms per tick

# Which columns each figure requires, and which script produces them.
# A CSV that parses but carries the wrong columns is the dangerous case: it
# yields a plausible-looking but incorrect figure, which is exactly how the
# fabricated Figure 4 survived (audit F-01). Validate explicitly and name the
# generating script in the error so the fix is obvious.
REQUIRED_COLUMNS = {
    "experiment_3_stability_merged.csv": (
        ["time", "max_velocity_cmd_run_A", "max_velocity_cmd_run_B"],
        "experiments/run_stability_test.py",
    ),
    "experiment_1_percolation.csv": (
        ["time", "true_lambda_2", "avg_local_lambda_proxy"],
        "experiments/run_percolation.py",
    ),
    "experiment_2_thermodynamics_merged.csv": (
        ["time", "framework_total_energy", "baseline_total_energy",
         "framework_active_drones", "baseline_active_drones"],
        "experiments/run_energy_cascade.py",
    ),
}


def load_checked(filename):
    """Load a figure's source CSV, failing loudly if it is not the expected file."""
    path = os.path.join(LOG_DIR, filename)
    required, producer = REQUIRED_COLUMNS[filename]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate it with: python {producer}"
        )
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}.\n"
            f"Found: {list(df.columns)}\n"
            f"This file is not the output of {producer}. It was most likely "
            f"overwritten by an ad-hoc simulation run writing into '{LOG_DIR}'. "
            f"Regenerate with: python {producer}"
        )
    if len(df) == 0:
        raise ValueError(f"{path} is empty. Regenerate with: python {producer}")
    print(f"  [{filename}] {len(df)} rows, columns verified")
    return df

def plot_stability():
    df = load_checked("experiment_3_stability_merged.csv")
    # Fix time scale
    real_time = df['time'] * DT
    
    plt.figure()
    plt.plot(real_time, df['max_velocity_cmd_run_A'], label='Unconstrained Heuristic', color='gray', alpha=0.6, linewidth=1.5)
    plt.plot(real_time, df['max_velocity_cmd_run_B'], label='Stability-Constrained Bisection', color='darkred', linewidth=2)
    plt.axhline(y=2.0, color='black', linestyle='--', label='Local Stability Boundary')
    
    # Adjust annotations for physical time (assuming original tick indices were 50 and 150)
    plt.axvline(x=50 * DT, color='k', linestyle=':', alpha=0.5)
    plt.text(52 * DT, 2.0, 'Obstacle Avoidance Trigger', rotation=90, va='bottom', fontsize=10, alpha=0.7)
    plt.axvline(x=150 * DT, color='k', linestyle=':', alpha=0.5)
    plt.text(152 * DT, 2.0, 'Severe Packet Loss Spike', rotation=90, va='bottom', fontsize=10, alpha=0.7)

    plt.ylim(-0.2, 4.5)
    plt.title("Kinematic Stability During Heuristic Adaptation")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Maximum Kinematic Variance / Velocity ($v$)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kinematic_stability.png"))
    print("Generated kinematic_stability.png")

def plot_percolation():
    df = load_checked("experiment_1_percolation.csv")
    real_time = df['time'] * DT
    
    plt.figure()
    # No rescaling. The previous /100.0 had no derivation; it existed to bring a
    # neighbour-count column onto the same axis as lambda_2 (audit F-19).
    plt.plot(real_time, df['true_lambda_2'], label='True Algebraic Connectivity ($\lambda_2$), living agents', color='blue', linewidth=2)
    plt.plot(real_time, df['avg_local_lambda_proxy'], label='Local Mixing Proxy ($\hat{\lambda}_2$), Eq. 2', color='orange', linestyle='--', linewidth=2)
    plt.axhline(y=0.1, color='red', linestyle=':', label='$\lambda_{crit}$ (Fragmentation Threshold)')

    # Mark where the jamming ramp saturates. run_percolation.py starts at
    # psi_max = 0.05 and _handle_env_update adds interference_growth_rate * dt * 5
    # = 0.025 every 5 ticks, so psi reaches 1.0 (total blackout, zero delivery
    # probability) at t = 191 ticks. Everything to the right of this line is a
    # saturated regime, not a jamming gradient -- audit F-39.
    blackout_tick = 191.0
    if real_time.max() > blackout_tick * DT:
        plt.axvspan(blackout_tick * DT, real_time.max(), color='red', alpha=0.07)
        plt.axvline(x=blackout_tick * DT, color='red', linestyle='--', alpha=0.6)
        plt.text(blackout_tick * DT + 0.4, plt.ylim()[1] * 0.55,
                 'total blackout ($\psi = 1$)\nno packet can be delivered',
                 rotation=90, va='center', fontsize=9, alpha=0.8, color='darkred')

    plt.title("Spectral Stability Under Environmental Jamming")
    plt.xlabel("Simulation Time (s)")
    # NOT "Fiedler Value": only the blue curve is one. The proxy is a different
    # quantity on a different scale, and labelling the shared axis as a Fiedler
    # value is precisely the conflation the old /100 rescaling created (F-19).
    plt.ylabel("Magnitude (curves are not on a common scale)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spectral_stability.png"))
    print("Generated spectral_stability.png")

def plot_thermodynamics():
    df = load_checked("experiment_2_thermodynamics_merged.csv")
    real_time = df['time'] * DT
    
    fig, ax1 = plt.subplots()

    color1 = 'tab:green'
    ax1.set_xlabel('Simulation Time (s)')
    ax1.set_ylabel('Total System Energy (J)', color='black')
    
    ax1.plot(real_time, df['framework_total_energy'], color=color1, linewidth=2, label='Framework (Total Energy)')
    ax1.plot(real_time, df['baseline_total_energy'], color='gray', linestyle='-.', linewidth=2, label='Unconstrained RGG Baseline')
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Active Drone Count ($N$)', color=color2)  
    
    ax2.plot(real_time, df['framework_active_drones'], color=color2, linestyle='--', linewidth=2, label='Framework Active Drones')
    ax2.plot(real_time, df['baseline_active_drones'], color='black', linestyle=':', linewidth=2, label='Baseline Active Drones')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Thermodynamic Decay and Node Attrition")
    fig.tight_layout()  
    plt.savefig(os.path.join(OUTPUT_DIR, "thermodynamic_decay.png"))
    print("Generated thermodynamic_decay.png")

if __name__ == "__main__":
    plot_stability()
    plot_percolation()
    plot_thermodynamics()
