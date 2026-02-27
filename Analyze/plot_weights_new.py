import matplotlib.pyplot as plt
import numpy as np
import os

def plotWeights(weights, save_path=None, weights_type=3):
    """
    Plots weights across trials given shape (num_trials, num_cells).
    Input weights_type: 1=GRGO, 2=GOGO, 3=MFGO, 4=MFGR, 5=GOGR
    """
    
    # 1. Title Selection (Matches your old script)
    titles = {
        1: "GRGO Weights Across Time",
        2: "GOGO Weights Across Time",
        3: "MFGO Weights Across Time",
        4: "MFGR Weights Across Time",
        5: "GOGR Weights Across Time"
    }
    title = titles.get(weights_type, "Weights Across Time")

    print(f"Processing {title}...")
    print("Array shape:", weights.shape) # Expected: (Trials, Cells)
    
    num_trials = weights.shape[0]
    num_cells = weights.shape[1]
    trials_x = np.arange(num_trials)

    plt.figure(figsize=(18, 6))

    # # =================================================================
    # # OPTION A: PLOT GLOBAL AVERAGE (Default)
    # # =================================================================
    # # This matches your old script: averages all cells to show general drift.
    
    # weights_per_trial = np.mean(weights, axis=1) # Average across cells
    # plt.plot(
    #     trials_x, 
    #     weights_per_trial, 
    #     color='blue',       
    #     linewidth=3,
    #     marker='o',        
    #     markersize=4,
    #     label='Population Mean (All Cells)'
    # )


    # =================================================================
    # OPTION B: PLOT SPECIFIC CELLS (For Plasticity Debugging)
    # =================================================================
    # Uncomment this block to compare your Plastic Cell vs. a Static Cell.
    # This effectively proves if your plasticity logic is working.
    
    plastic_cell_id = 1   # The cell you forced to learn
    static_cell_id = 2    # A neighbor that should be flat

    # Plot Plastic Cell
    plt.plot(trials_x, weights[:, plastic_cell_id], 
             label=f'Plastic Cell (ID {plastic_cell_id})', 
             color='red', linewidth=2, marker='x')

    # Plot Static Cell
    plt.plot(trials_x, weights[:, static_cell_id], 
             label=f'Static Cell (ID {static_cell_id})', 
             color='black', linestyle='--', alpha=0.7)


    # =================================================================
    # OPTION C: PLOT ALL CELLS (The "Cloud")
    # =================================================================
    # Uncomment this to see every single cell as a faint line.
    # Warning: Can be slow if you have >10,000 cells.
    
    # for i in range(min(num_cells, 500)): # Limit to first 500 for speed
    #     plt.plot(trials_x, weights[:, i], color='grey', alpha=0.1, linewidth=1)


    # =================================================================
    # FORMATTING & SAVING
    # =================================================================
    plt.xlabel("Trial Number")
    plt.ylabel("Synaptic Weight Strength")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    # Save logic
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


# --- Execution ---
# Define paths
input_file = '/home/data/einez/MFGoGr_simple_mfgoplast_onecell_20_trial_mfgoW.npy'
output_img = "/home/aw39625/minisim/Results/MFGoGr_simple_mfgoplast_onecell_20_trial_mfgoW.png"

# Check if file exists before running
if os.path.exists(input_file):
    weights_data = np.load(input_file)
    
    # Set weights_type: 1=GRGO, 2=GOGO, 3=MFGO, etc.
    plotWeights(weights_data, save_path=output_img, weights_type=3)
else:
    print(f"File not found: {input_file}")
    