import matplotlib.pyplot as plt
import numpy as np
import os

def plotFiringFrequencyDrift(raster, cell_type, timestep_ms=1.0, save_path=None):
    """
    Plots average firing frequency per cell across trials.
    Handles subsampling for large populations (Granule cells).
    Skips the first cell (index 0) to avoid artifacts.
    """
    if raster.ndim != 3:
        raise ValueError(f"Expected 3D array (num_trials, num_timesteps, num_cells), got {raster.shape}")

    # --- MODIFICATION: REMOVE FIRST CELL ---
    # We slice the array to exclude index 0 before doing any calculations
    print(f"Original shape: {raster.shape}")
    raster = raster[:, :, 1:] 
    print(f"Shape after removing first cell: {raster.shape}")

    num_trials, num_timesteps, num_cells = raster.shape
    
    # --- SUBSAMPLING LOGIC ---
    # If Granule cells (Type 3) and population is large, subsample 5,000 cells
    if cell_type == 3 and num_cells > 5000:
        print(f"Granule Cells detected ({num_cells}). Subsampling 5,000 for visualization...")
        # Randomly select 5000 indices without replacement
        selected_indices = np.random.choice(num_cells, 5000, replace=False)
        raster = raster[:, :, selected_indices]
        # Update num_cells after slicing
        num_cells = raster.shape[2] 
    else:
        # For MF (1) and Golgi (2), or small Granule populations, use all cells
        pass

    # --- CALCULATE FREQUENCY ---
    # Convert timestep count to seconds
    trial_duration_sec = (num_timesteps * timestep_ms) / 1000.0
    
    # Compute firing frequency per cell per trial (Hz)
    # Sum spikes over time axis (axis 1), divide by duration
    freq = np.sum(raster, axis=1) / trial_duration_sec  # shape: (num_trials, num_cells)

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # Define title based on cell type
    cell_names = {1: "Mossy Fiber", 2: "Golgi Cell", 3: "Granule Cell"}
    type_name = cell_names.get(cell_type, "Unknown Cell")
    
    # # Plot individual traces (lighter, thinner)
    # for cell in range(num_cells):
    #     plt.plot(range(1, num_trials + 1), freq[:, cell], alpha=0.3, lw=0.8)

    # Plot the population average in bold (black)
    plt.plot(range(1, num_trials + 1), np.mean(freq, axis=1), color='black', lw=2, label='Mean firing rate')

    plt.xlabel("Trial")
    plt.ylabel("Firing frequency (Hz)")
    plt.title(f"{type_name} Firing Frequency Drift Across Trials (Cell 0 Excluded)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    print(f"Global Average Frequency ({type_name}): {np.mean(freq):.4f} Hz")
    plt.show()

# --- EXECUTION BLOCK ---

# 1. Load the raster data
raster_path = '/home/data/einez/MFGoGr_MFGOplast_discrete_trace_symmetric_20_trials_GOrasters.npy' 
raster_data = np.load(raster_path)

print("Finished loading data")

# 2. Define Cell Type (1=MF, 2=Golgi, 3=Granule)
current_cell_type = 2

# 3. Define save location
save_filename = "MFGoGr_MFGOplast_discrete_trace_symmetric_20_trials_average.png"
plot_save_path = f"/home/aw39625/minisim/Results/Firing_Freq_Plots/{save_filename}"

# 4. Run the function
plotFiringFrequencyDrift(raster_data, cell_type=current_cell_type, timestep_ms=1.0, save_path=plot_save_path)