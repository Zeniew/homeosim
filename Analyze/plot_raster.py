import numpy as np
import matplotlib as plt

def showRasters(raster):
    print("Raster shape:", raster.shape)
    if raster.ndim != 2:
        raise ValueError(f"Expected a 2D raster array, but got shape {raster.shape}")

    numCell = raster.shape[1]
    plotarray = [np.where(raster[:, i] == 1)[0] for i in range(numCell)]

    plt.figure(figsize=(18, 9))
    plt.eventplot(plotarray, colors='black')
    plt.xlabel("Timestep")
    plt.ylabel("Neuron #")
    plt.show()

raster_data = np.load("/home/aw39625/minisim/Results/MFGoGr_Experiment_TestMF_MFrasters.npy")
showRasters(raster_data)
