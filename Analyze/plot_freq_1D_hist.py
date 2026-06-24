import matplotlib.pyplot as plt
import numpy as np
import os

def plotFiringFrequencyDrift(raster, cell_type, timestep_ms=1.0, save_path=None, target = 10.0):
    """
    Plots a 1D histogram of the average firing frequency per cell across the LAST 100 trials.
    Handles subsampling for large populations (Granule cells).
    Skips the first cell (index 0) to avoid artifacts.
    """
    if cell_type != 3:
        if raster.ndim != 3:
            raise ValueError(f"Expected 3D array (num_trials, num_timesteps, num_cells), got {raster.shape}")
        else:
            # --- MODIFICATION: REMOVE FIRST CELL ---
            print(f"Original shape: {raster.shape}")
            raster = raster[:, :, 1:] 
            print(f"Shape after removing first cell: {raster.shape}")

            # --- MODIFICATION: KEEP ONLY LAST 100 TRIALS ---
            num_trials_total = raster.shape[0]
            trials_to_keep = min(100, num_trials_total)
            print(f"Slicing data to use only the last {trials_to_keep} trials...")
            raster = raster[-trials_to_keep:, :, :]
            
            num_trials, num_timesteps, num_cells = raster.shape
    
    num_cells = raster.shape[2] if cell_type != 3 else raster.shape[1]
    # --- SUBSAMPLING LOGIC ---
    if cell_type == 3 and num_cells > 5000:
        print(f"Granule Cells detected ({num_cells}). Subsampling 5,000 for visualization...")
        selected_indices = np.random.choice(num_cells, 5000, replace=False)
        num_trials_total = raster.shape[0]
        trials_to_keep = min(100, num_trials_total)
        raster = raster[-trials_to_keep:, selected_indices]
        num_cells = raster.shape[1] 

    # --- CALCULATE FREQUENCY ---
    trial_duration_sec = (5000 * timestep_ms) / 1000.0
    if cell_type != 3:
        freq = np.sum(raster, axis=1) / trial_duration_sec  # shape: (num_trials, num_cells)
    else:
        freq = raster / trial_duration_sec  # shape: (num_trials, num_cells) for granule cells since we already sliced the time axis

    # Define titles
    cell_names = {1: "Mossy Fiber", 2: "Golgi Cell", 3: "Granule Cell"}
    type_name = cell_names.get(cell_type, "Unknown Cell")

    # ==========================================
    # PLOT: 1D HISTOGRAM
    # ==========================================
    plt.figure(figsize=(10, 6))

    # Flatten the array to get the distribution of every cell's frequency across the sliced trials
    
    cell_avg_freqs = np.mean(freq, axis=0)
    global_avg = np.mean(cell_avg_freqs)

    # Plot the histogram
    plt.hist(cell_avg_freqs, bins=100, color='skyblue', edgecolor='black', alpha=0.8)

    # Add vertical reference lines
    plt.axvline(global_avg, color='red', linestyle='dashed', linewidth=2, label=f'Global Average: {global_avg:.2f} Hz')
    plt.axvline(target, color='green', linestyle='dashed', linewidth=2, label=f'Homeostatic Target: {target:.2f} Hz')

    plt.xlabel(f"Average Firing Frequency (Hz) over last {trials_to_keep} trials")
    plt.ylabel("Count (Number of Cell-Trials)")
    plt.title(f"{type_name} Firing Frequency Distribution (Last {trials_to_keep} Trials)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.xlim(0, 200) # For SS
    # plt.xlim(0, 50) # For IE

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")

    plt.show()

    print(f"Global Average Frequency ({type_name}, Last {trials_to_keep} Trials): {global_avg:.4f} Hz")


# --- EXECUTION BLOCK ---

# 1. Load the raster data
raster_path = '/home/data/einez/homeostat_SS/MFGoGr_SS_shuffleMF10percent_noCS_noplast_100_trial/MFGoGr_SS_shuffleMF10percent_noCS_noplast_100_trial_GOrasters.npy' 

if os.path.exists(raster_path):
    raster_data = np.load(raster_path)
    print("Finished loading data")
    
    # # Just the first 10 trials
    # raster_data = raster_data[:10, :]
 
    # 2. Define Cell Type (1=MF, 2=Golgi, 3=Granule)
    current_cell_type = 2

    # 3. Define save location
    save_filename = "MFGoGr_SS_shuffleMF10percent_noCS_noplast_100_trial/GO_Firing_Frequency_Histogram" 
    plot_save_path = f"/home/aw39625/minisim/Results/{save_filename}"

    # 4. Run the function
    plotFiringFrequencyDrift(raster_data, cell_type=current_cell_type, timestep_ms=1.0, save_path=plot_save_path, target = 10.0)
else:
    print(f"File not found: {raster_path}")