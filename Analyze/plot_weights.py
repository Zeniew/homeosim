import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

def plotWeights(weights, save_path = None, weights_type = 1):
    colors2 = 'red'
    title = ""

    if weights_type == 1: # GRGO
        title = "GRGO Weights Across Trials"
    if weights_type == 2: # GOGO
        title = "GOGO Weights Across Trials"
    if weights_type == 3: # MFGO
        title = "MFGO Weights Across Trials"
    if weights_type == 4: # MFGR
        title = "MFGR Weights Across Trials"
    if weights_type == 5: # GOGR
        title = "GOGR Weights Across Trials"

    print("Array shape:", weights.shape)
    numTrials = weights.shape[0]
    numTimeSteps = weights.shape[1]

    plt.figure(figsize = (18,9))
    for trial in range(numTrials):
        plt.plot(
        range(numTimeSteps), 
        weights[trial], 
        marker='o',     # dots
        linestyle='-',  # lines
        label=f"Trial {trial+1}"
        )

    plt.xlabel("Time Step")
    plt.ylabel("Weight")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        print(f"Plot saved to {save_path}")

weights_data = np.load('/home/data/einez/MFGoGr_mfgrplast_10_trials_mfgrW.npy')
plot_save_path = plot_save_path = "/home/aw39625/minisim/Results/Eventplot_MFGoGr_mfgrplast_10_trials_mfgrW.png"

plotWeights(weights_data, save_path = plot_save_path, weights_type = 4)