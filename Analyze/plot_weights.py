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

    print("Array shape:", weights.shape)
    num_trials = weights.shape[0]
    steps_per_trial = weights.shape[1]

    # 2. Process data by Trial (Average across time steps)
    # Instead of flattening, we average axis 1 (time) to get one value per trial
    weights_per_trial = np.mean(weights, axis=1)
    
    # Create the X-axis for Trials
    time_axis = np.arange(num_trials)

    plt.figure(figsize=(18, 6))

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

    # (Vertical lines for trial boundaries are no longer needed since X-axis IS trials)

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
weights_data = np.load('/home/data/einez/MFGoGr_simple_mfgoplast_diff_1000_trial_mfgoW.npy')
plot_save_path = "/home/aw39625/minisim/Results/MFGoGr_simple_mfgoplast_diff_1000_trial_mfgoW.png"

# Plotting MFGO (Type 3)
plotWeightsContinuous(weights_data, save_path=plot_save_path, weights_type=3)