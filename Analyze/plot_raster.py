import numpy as np
import matplotlib.pyplot as plt
import os

def showRasters(raster, save_path=None, raster_type = 1):
    if raster_type == 1: # MF
        print("Raster shape:", raster.shape)
        if raster.ndim != 2:
            raise ValueError(f"Expected a 2D raster array, but got shape {raster.shape}")

        numCell = raster.shape[1]
        plotarray = [np.where(raster[:, i] == 1)[0] for i in range(numCell)]

        plt.figure(figsize=(18, 9))
        plt.eventplot(plotarray, colors='black', linelengths=0.8)
        plt.xlabel("Timestep")
        plt.ylabel("Mossy Fiber Number")
        plt.title("MF Spike Raster")

    if raster_type == 2: # Go
        averaged_raster = np.mean(raster, axis=0)

        print("Averaged raster shape:", averaged_raster.shape)

        plt.figure(figsize=(18, 9))
        plt.imshow(raster.T, aspect='auto', cmap='Greys', origin='lower')
        plt.xlabel("Timestep")
        plt.ylabel("Golgi Cell Number")
        plt.title("Golgi Spike Raster")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


# Load the raster data
raster_data = np.load('/home/data/einez/MFGoGr_no_GoGo_GOrasters.npy')

# Define save location
plot_save_path = "/home/aw39625/minisim/Results/MFGoGr_no_GoGo_GOrasters.png"

# Show and save
showRasters(raster_data, save_path=plot_save_path, raster_type = 2)
