import matplotlib.pyplot as plt
import numpy as np
import os

def plotFiringFrequencyDensity(raster, cell_type, timestep_ms=1.0, save_path=None, num_bins=50):
    """
    Plots the density distribution of firing frequencies per cell across trials.
    Includes a 2D Heatmap (Histogram) and a Violin Plot.
    """
    if raster.ndim != 3:
        raise ValueError(f"Expected 3D array (num_trials, num_timesteps, num_cells), got {raster.shape}")

    # --- MODIFICATION: REMOVE FIRST CELL ---
    print(f"Original shape: {raster.shape}")
    raster = raster[:, :, 1:] 
    print(f"Shape after removing first cell: {raster.shape}")

    num_trials, num_timesteps, num_cells = raster.shape
    
    # --- SUBSAMPLING LOGIC ---
    if cell_type == 3 and num_cells > 5000:
        print(f"Granule Cells detected ({num_cells}). Subsampling 5,000 for visualization...")
        selected_indices = np.random.choice(num_cells, 5000, replace=False)
        raster = raster[:, :, selected_indices]
        num_cells = raster.shape[2] 

    # --- CALCULATE FREQUENCY ---
    trial_duration_sec = (num_timesteps * timestep_ms) / 1000.0
    freq = np.sum(raster, axis=1) / trial_duration_sec  # shape: (num_trials, num_cells)

    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    cell_names = {1: "Mossy Fiber", 2: "Golgi Cell", 3: "Granule Cell"}
    type_name = cell_names.get(cell_type, "Unknown Cell")
    mean_freq = np.mean(freq, axis=1)

    # ---------------------------------------------------------
    # PLOT 1: 2D Density Heatmap (Stacked Histograms)
    # ---------------------------------------------------------
    ax1 = axes[0]
    
    # Define consistent bins across all trials based on global min/max
    f_min, f_max = np.min(freq), np.max(freq)
    bin_edges = np.linspace(f_min, f_max, num_bins + 1)
    
    # Build a matrix of shape (num_bins, num_trials)
    hist_matrix = np.zeros((num_bins, num_trials))
    for t in range(num_trials):
        counts, _ = np.histogram(freq[t, :], bins=bin_edges)
        hist_matrix[:, t] = counts

    # Create meshgrid for plotting
    # Shift X by 0.5 so trial numbers align with the center of the bins
    X, Y = np.meshgrid(np.arange(1, num_trials + 2) - 0.5, bin_edges)
    
    # Plot the heatmap
    c = ax1.pcolormesh(X, Y, hist_matrix, cmap='viridis', shading='flat')
    fig.colorbar(c, ax=ax1, label='Number of Cells')
    
    # Overlay the population average
    ax1.plot(range(1, num_trials + 1), mean_freq, color='red', lw=2, marker='o', markersize=4, label='Mean firing rate')
    
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Firing frequency (Hz)")
    ax1.set_title(f"{type_name} Firing Frequency Density (Heatmap)")
    ax1.set_xticks(range(1, num_trials + 1))
    ax1.legend()

    # ---------------------------------------------------------
    # PLOT 2: Violin Plot (Smoothed Density Distributions)
    # ---------------------------------------------------------
    ax2 = axes[1]
    
    # Format data as a list of arrays (one array per trial)
    data_to_plot = [freq[t, :] for t in range(num_trials)]
    
    parts = ax2.violinplot(data_to_plot, positions=range(1, num_trials + 1), showmeans=True, showextrema=True)
    
    # Optional: Customize violin plot colors
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Firing frequency (Hz)")
    ax2.set_title(f"{type_name} Firing Frequency Distribution (Violin Plot)")
    ax2.set_xticks(range(1, num_trials + 1))

    plt.tight_layout()

    # --- SAVE & SHOW ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    print(f"Global Average Frequency ({type_name}): {np.mean(freq):.4f} Hz")
    plt.show()

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Load the raster data
    raster_path = '/home/data/einez/MFGoGr_shuffledMFisi_noCS_noGoGo_mfgoplast_halfcell_50_trial_GOrasters.npy' 
    
    # Add simple mock data generation for testing if file doesn't exist
    if os.path.exists(raster_path):
        raster_data = np.load(raster_path)
        print("Finished loading data")
    else:
        print(f"Warning: File {raster_path} not found. Using random noise for demonstration.")
        raster_data = np.random.binomial(1, 0.05, (20, 1000, 4096))

    # 2. Define Cell Type (1=MF, 2=Golgi, 3=Granule)
    current_cell_type = 2

    # 3. Define save location
    save_filename = "MFGoGr_shuffledMFisi_noCS_noGoGo_mfgoplast_halfcell_50_trial_density.png"
    plot_save_path = f"/home/aw39625/minisim/Results/Firing_Freq_Plots/{save_filename}"

    # 4. Run the function
    plotFiringFrequencyDensity(raster_data, cell_type=current_cell_type, timestep_ms=1.0, save_path=plot_save_path, num_bins=60)