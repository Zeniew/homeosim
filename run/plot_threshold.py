import matplotlib.pyplot as plt
import numpy as np
import os

def plotThresholdContinuous(thr, save_path=None, thr_type=1):
    """
    Plots thr as a continuous timeline across all trials.
    Input thr shape expected: (num_trials, num_timesteps)
    """
    
    # 1. Title Selection
    titles = {
        1: "GO Thresholds Aross Time",
        2: "GR Thresholds Across Time"
    }
    title = titles.get(thr_type, "Thresholds Across Time")

    print("Array shape:", thr.shape)
    num_trials = thr.shape[0]
    steps_per_trial = thr.shape[1]

    # 2. Process data by Trial (Average across time steps)
    # Instead of flattening, we average axis 1 (time) to get one value per trial
    weights_per_trial = np.mean(thr, axis=1)

    # #---
    # # TEST: ONLY PLOTTING threshold OF FIRST CELL
    # weights_per_trial = thr[:, 2]  # Take only the first threshold from each trial
    # # ---   
    
    # Create the X-axis for Trials
    time_axis = np.arange(num_trials)

    plt.figure(figsize=(8, 6))

    # Plot individual cell threshold traces across trials (lighter, thinner)
    for cell in range(thr.shape[1]):
        plt.plot(
            time_axis,
            thr[:, cell], 
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
        label='Mean threshold per Trial'
    )


    plt.xlabel("Trial Number") # Fixed X-axis label

    # ----
    # # 2. Flatten the data (Concatenate trials end-to-end)
    # # This turns shape (20, 1000) into shape (20000,)
    # weights_continuous = thr.flatten()
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
    #     label='threshold Value'
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
    plt.ylabel("Threshold Strength")
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
thr_data = np.load('/home/data/einez/homeostat_IE/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_GOplast_{numTrial}_trial/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_GOplast_{numTrial}_trial_GoThr.npy')
plot_save_path = "/home/aw39625/minisim/Results/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_GOplast_{numTrial}_trial/GoThr.png"

# Plotting MFGO (Type 3)
plotThresholdContinuous(thr_data, save_path=plot_save_path, thr_type=1)