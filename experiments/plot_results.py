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

def plot_stability():
    file_path = os.path.join(LOG_DIR, "experiment_3_stability.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
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
    file_path = os.path.join(LOG_DIR, "experiment_1_percolation.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    real_time = df['time'] * DT
    
    plt.figure()
    plt.plot(real_time, df['true_lambda_2'], label='True Algebraic Connectivity ($\lambda_2$)', color='blue', linewidth=2)
    plt.plot(real_time, df['avg_local_lambda_proxy'] / 100.0, label='Scaled Local Proxy ($\hat{\lambda}_2$)', color='orange', linestyle='--', linewidth=2)
    plt.axhline(y=0.1, color='red', linestyle=':', label='$\lambda_{crit}$ (Fragmentation Threshold)')
    
    plt.title("Spectral Stability Under Environmental Jamming")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Fiedler Value ($\lambda_2$)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spectral_stability.png"))
    print("Generated spectral_stability.png")

def plot_thermodynamics():
    file_path = os.path.join(LOG_DIR, "experiment_2_thermodynamics.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
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
