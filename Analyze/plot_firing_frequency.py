import matplotlib.pyplot as plt
import numpy as np
import os

def plotFiringFrequencyDrift(raster, timestep_ms=1.0, save_path=None):
    """
    Plots average firing frequency per cell across trials.
    
    Parameters
    ----------
    raster : np.ndarray
        3D array with shape (num_trials, num_timesteps, num_cells)
        Each entry should be 1 if the cell spiked at that timestep, else 0.
    timestep_ms : float
        Duration of each timestep in milliseconds.
    save_path : str or None
        Optional path to save the figure.
    """
    if raster.ndim != 3:
        raise ValueError(f"Expected 3D array (num_trials, num_timesteps, num_cells), got {raster.shape}")

    num_trials, num_timesteps, num_cells = raster.shape
    trial_duration_sec = (num_timesteps) / 1000.0
    # Compute firing frequency per cell per trial (Hz)
    freq = np.sum(raster, axis=1) / trial_duration_sec  # shape: (num_trials, num_cells)

    # Plot each cell’s firing frequency trajectory
    plt.figure(figsize=(12, 6))
    for cell in range(1, num_cells):
        plt.plot(range(1, num_trials + 1), freq[:, cell], alpha=0.3, lw=0.8)

    # Plot the population average in bold
    plt.plot(range(1, num_trials + 1), np.mean(freq, axis=1), color='black', lw=2, label='Mean firing rate')

    plt.xlabel("Trial")
    plt.ylabel("Firing frequency (Hz)")
    plt.title("Firing Frequency Drift Across Trials")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    print("Average frequency: ", np.mean(freq))

    plt.show()

# Load the raster data
raster_data = np.load('/home/data/einez/MFGoGr_MFGOplast_plast_ratio_12000_20_trials_GOrasters.npy')

print("Finished loading data")
print(raster_data.shape)
print(np.where(np.mean(np.sum(raster_data, axis = 1) / 5, axis = 0) > 100)) # where is the average frequency across trials greater than 100 Hz (threshold 500 because I didn't divide by timestep)
print(np.mean(np.sum(raster_data, axis = 1) / 5, axis = 0)[0]) # average frequency across trials of cell

# Define save location
plot_save_path = "/home/aw39625/minisim/Results/Firing_Freq_Plots/MFGoGr_MFGOplast_plast_ratio_12000_20_trials_GO_Average_Firing_Frequency.png"

# Show and save
plotFiringFrequencyDrift(raster_data, timestep_ms=1, save_path=plot_save_path)