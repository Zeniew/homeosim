import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import MFGOGrFunctions_synaptic_scaling as mfgogr

# ==========================================
# SIMULATION SETTINGS
# ==========================================
numBins = 5000
input_rates_hz = np.linspace(0, 1500, 100) 

# Variables to sweep for the two plots
threshold_sweep = [-34.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0] # Intrinsic Plasticity changing resting threshold
weight_sweep = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]       # GOGO Plasticity changing synaptic weights
tau = 10.0                              # Synaptic decay constant
eGABA = -64.0                           # GABA reversal potential

# Dictionaries to store results
results_hz = {}
results_vm = {}

print("Generating Data for Presentation Plots. This will take a moment...")

# 1. Run the sweeps for different thresholds
for thresh in threshold_sweep:
    print(f"Simulating F-I Curve for Threshold = {thresh} mV...")
    GO = mfgogr.Golgi(n=1, csON=0, csOFF=0, useCS=0, numBins=numBins, plast_ratio=0.0)
    
    out_hz_list = []
    avg_vm_list = []
    
    for in_rate in input_rates_hz:
        # Reset state
        GO.Vm.fill(GO.eLeak)
        GO.threshRest = thresh  # Inject the modified resting threshold here
        GO.gSum_MFGO.fill(0)
        GO.gSum_GRGO.fill(0)
        GO.gSum_GOGO.fill(0)
        GO.gNMDA_MFGO.fill(0)
        GO.act.fill(0)
        
        vm_history = np.zeros(numBins)
        
        for t in range(numBins):
            spikes = np.random.poisson(in_rate / 1000.0)
            GO.inputMFGO[0] = spikes
            GO.do_Golgi(t)
            vm_history[t] = GO.Vm[0]
            
        out_hz = np.sum(GO.act[:, 0]) / (numBins / 1000.0)
        out_hz_list.append(out_hz)
        avg_vm_list.append(np.mean(vm_history))
        
    results_hz[thresh] = np.array(out_hz_list)
    results_vm[thresh] = np.array(avg_vm_list)

print("Simulations complete. Generating plots...")

# ==========================================
# PLOT 1: The F-I Curve Shift (Intrinsic Plasticity)
# ==========================================
plt.figure(figsize=(10, 6))
colors_thresh = plt.cm.coolwarm(np.linspace(0, 1, len(threshold_sweep)))

for i, thresh in enumerate(threshold_sweep):
    plt.plot(input_rates_hz, results_hz[thresh], label=f'Threshold = {thresh} mV', 
             color=colors_thresh[i], linewidth=3, alpha=0.8)

plt.axhline(y=28, color='black', linestyle=':', alpha=0.5, label='28 Hz (Explosion Zone)')

plt.title("How Intrinsic Plasticity Stabilizes the Network (F-I Curve Flattening)", fontsize=16, fontweight='bold')
plt.xlabel("Input Drive (Total MF Spikes / sec)", fontsize=14)
plt.ylabel("Output Firing Rate (Hz)", fontsize=14)

# --- MOVED LEGEND OUTSIDE ---
plt.legend(fontsize=11, bbox_to_anchor=(1.04, 1), loc="upper left")

plt.grid(True, alpha=0.3)
plt.tight_layout()
# --- ADDED BBOX_INCHES='TIGHT' TO PREVENT CROPPING ---
plt.savefig("Plot1_Intrinsic_FI_Shift.png", dpi=300, bbox_inches='tight')

# ==========================================
# PLOT 2: The Loop Gain Landscape (GOGO Plasticity)
# ==========================================
# For Plot 2, we use the baseline threshold (-34.0) to show why it exploded before adapting.
baseline_hz = results_hz[-34.0]
baseline_vm = results_vm[-34.0]

# Smooth the F-I curve slightly to calculate a clean derivative (Gain, g)
smoothed_hz = gaussian_filter1d(baseline_hz, sigma=2)
# Calculate Gain (g) = df/dx
gain_g = np.gradient(smoothed_hz, input_rates_hz) 

plt.figure(figsize=(10, 6))
colors_weight = plt.cm.magma(np.linspace(0.1, 0.9, len(weight_sweep)))

for i, W in enumerate(weight_sweep):
    # Calculate effective inhibitory drive per spike
    driving_force = np.abs(baseline_vm - eGABA)
    effective_inhibition = W * driving_force * tau
    
    # Calculate Loop Gain
    loop_gain = effective_inhibition * gain_g
    
    plt.plot(baseline_hz, loop_gain, label=f'GOGO Weight = {W}', 
             color=colors_weight[i], linewidth=3)

# The Stability Threshold (Red Line of Death)
plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Instability Threshold (Loop Gain = 1.0)')

# Highlight the 28-30 Hz Danger Zone
plt.axvspan(25, 32, color='gray', alpha=0.15, label='Maximum Gain Zone (~28 Hz)')

plt.title("The Race to Instability: Loop Gain vs Firing Rate", fontsize=16, fontweight='bold')
plt.xlabel("Golgi Firing Rate (Hz)", fontsize=14)
plt.ylabel("Network Loop Gain (L = W * g)", fontsize=14)
plt.ylim(0, np.max(loop_gain)*1.2)

# --- MOVED LEGEND OUTSIDE ---
plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")

plt.grid(True, alpha=0.3)
plt.tight_layout()
# --- ADDED BBOX_INCHES='TIGHT' TO PREVENT CROPPING ---
plt.savefig("Plot2_LoopGain_Explosion.png", dpi=300, bbox_inches='tight')

print("Plots saved successfully as PNGs!")
plt.show()