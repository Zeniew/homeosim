import matplotlib.pyplot as plt
import numpy as np
import os

def plotFiringFrequencyDrift(raster, cell_type, timestep_ms=1.0, save_path=None):
    """
    Plots average firing frequency per cell across trials.
    Handles subsampling for large populations (Granule cells).
    
    Parameters
    ----------
    raster : np.ndarray
        3D array with shape (num_trials, num_timesteps, num_cells)
        Each entry should be 1 if the cell spiked at that timestep, else 0.
    cell_type : int
        1 = MF (Mossy Fiber)
        2 = Golgi Cell
        3 = Granule Cell (Triggers subsampling)
    timestep_ms : float
        Duration of each timestep in milliseconds.
    save_path : str or None
        Optional path to save the figure.
    """
    if raster.ndim != 3:
        raise ValueError(f"Expected 3D array (num_trials, num_timesteps, num_cells), got {raster.shape}")

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
    # # We iterate range(num_cells) because we might have sliced the array
    # for cell in range(num_cells):
    #     plt.plot(range(1, num_trials + 1), freq[:, cell], alpha=0.3, lw=0.8)

    # Plot the population average in bold (black)
    plt.plot(range(1, num_trials + 1), np.mean(freq, axis=1), color='black', lw=2, label='Mean firing rate')

    plt.xlabel("Trial")
    plt.ylabel("Firing frequency (Hz)")
    plt.title(f"{type_name} Firing Frequency Drift Across Trials")
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
# Update this path to your Granule raster file if testing cell_type=3
raster_path = '/home/data/einez/MFGoGr_MFGOplast_discrete_trace_20_trials_GOrasters.npy' 
raster_data = np.load(raster_path)

print("Finished loading data")
print(f"Original Shape: {raster_data.shape}")

# 2. Define Cell Type (CHANGE THIS: 1=MF, 2=Golgi, 3=Granule)
current_cell_type = 2

# 3. Define save location
save_filename = "MFGoGr_MFGOplast_discrete_trace_20_trials_average.png"
plot_save_path = f"/home/aw39625/minisim/Results/Firing_Freq_Plots/{save_filename}"

# 4. Run the function
# Note: Ensure timestep_ms matches your simulation (usually 1.0 ms)
plotFiringFrequencyDrift(raster_data, cell_type=current_cell_type, timestep_ms=1.0, save_path=plot_save_path)