import matplotlib.pyplot as plt
import numpy as np
import os

def plotWeightsContinuous(weights, save_path=None, weights_type=1):
    """
    Plots weights as a continuous timeline across all trials.
    Input weights shape expected: (num_trials, num_timesteps)
    """
    
    # 1. Title Selection
    titles = {
        1: "GRGO Weights Across Time",
        2: "GOGO Weights Across Time",
        3: "MFGO Weights Across Time",
        4: "MFGR Weights Across Time",
        5: "GOGR Weights Across Time"
    }
    title = titles.get(weights_type, "Weights Across Time")
    # weights = weights[:, 1:numcells]
    print("Array shape:", weights.shape)
    num_trials = weights.shape[0]
    steps_per_trial = weights.shape[1]

    # 2. Process data by Trial (Average across time steps)
    # Instead of flattening, we average axis 1 (time) to get one value per trial
    weights_per_trial = np.mean(weights, axis=1)

    # #---
    # # TEST: ONLY PLOTTING WEIGHT OF FIRST CELL
    # weights_per_trial = weights[:, 2]  # Take only the first weight from each trial
    # # ---   
    
    # Create the X-axis for Trials
    time_axis = np.arange(num_trials) + 1 # Start from 1 for better readability on the plot

    plt.figure(figsize=(8, 6))

    # Plot individual cell weight traces across trials (lighter, thinner)
    for cell in range(1, weights.shape[1]):
        plt.plot(
            time_axis,
            weights[:, cell], 
            alpha=0.3, 
            linewidth=0.8
        )

     # 3. Plot the trial averages
    plt.plot(
        time_axis, 
        weights_per_trial, 
        color='blue',       
        linewidth=2,
        marker='o',        # Added dots so you can see each trial clearly
        markersize=3,
        label='Mean Weight per Trial'
    )


    plt.xlabel("Trial Number") # Fixed X-axis label

    # ----
    # # 2. Flatten the data (Concatenate trials end-to-end)
    # # This turns shape (20, 1000) into shape (20000,)
    # weights_continuous = weights.flatten()
    # total_steps = len(weights_continuous)
    
    # # Create the X-axis for the continuous time
    # time_axis = np.arange(total_steps)

    # plt.figure(figsize=(18, 6))

    # # 3. Plot the continuous trace
    # plt.plot(
    #     time_axis, 
    #     weights_continuous, 
    #     color='blue',       
    #     linewidth=1.5,
    #     label='Weight Value'
    # )

    ## 4. Add Vertical Lines for Trial Boundaries
    ## This helps you see if resets happen between trials
    # for i in range(1, num_trials):
    #     trial_boundary = i * steps_per_trial
    #     plt.axvline(x=trial_boundary, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # ----

    # Add a dummy line for the legend to explain the red dashes
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Trial Boundary')

    plt.xlabel("Continuous Time Step (ms)")
    plt.ylabel("Weight Strength")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")


# --- Execution ---
weights_data = np.load('/home/data/einez/homeostat_SS/MFGoGr_SS_shuffleMF10percent_noCS_yesGoGo_yesgrGo_mfgrplast_allcell_1000_trial/MFGoGr_SS_shuffleMF10percent_noCS_yesGoGo_yesgrGo_mfgrplast_allcell_1000_trial_mfgrW.npy')
plot_save_path = "/home/aw39625/minisim/Results/MFGoGr_SS_shuffleMF10percent_noCS_yesGoGo_yesgrGo_mfgrplast_allcell_1000_trial/MFGoGr_SS_shuffleMF10percent_noCS_yesGoGo_yesgrGo_mfgrplast_allcell_1000_trial_mfgrW.png"

if weights_data.shape[1] > 4096:
    weights_data = weights_data[:, 1:5000]  # Remove the first column (cell 0) to exclude it from the plot
else:
    weights_data = weights_data[:, 1:weights_data.shape[1]]  # Remove the first column (cell 0) to exclude it from the plot

# Check if the values are "close enough"
is_consistent = np.allclose(weights_data, weights_data[0], atol=1e-10)
print(f"Are weights effectively the same? {is_consistent}")

# Print weights to check if changing
print(f"Sample weights from first trial:", weights_data[0, :5])  # Print first 5 weights from the first trial
print(f"Sample weights from second trial:", weights_data[1, :5])  # Print first 5 weights from the last trial
print(f"Sample weights from third trial:", weights_data[2, :5])  # Print first 5 weights from the third trial

plotWeightsContinuous(weights_data, save_path=plot_save_path, weights_type=4)