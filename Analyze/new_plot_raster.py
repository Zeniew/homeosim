import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

'''
Here is a brief snippet of code that shows how to use "eventplot"

This function takes in a 2D array where one dimension is the number 
of rows in your raster plot (cells) and the other dimension is the list of times 
that a tick should occur (spike times for you, or time bins?).

The crappy thing about this function is that because it inputs an array, 
you have to anticipate the largest number of spikes that might occur in all 
of the rows, and because most rows have fewer spikes than that you are 1) wasting memory 
and 2) what do you do with all of the blank cells of the array. My solution to that is to 
start with an array created by numpy.zeros so that the default value if there is no spike is 
zero.  This works, but all of those zeros show up as a vertical line at the very left edge of 
your raster plot.
'''

# np.random.seed(19680801)


# # create another set of random data.
# # the gamma distribution is only used for aesthetic purposes
# test_data = np.random.gamma(4, size=[50, 50])
# exp_data = np.load('/home/data/einez/MFGoGr_full_GRrasters.npy')

# colors2 = 'black'
# lineoffsets2 = 1
# linelengths2 = 1
# save_path = "/home/aw39625/minisim/Results/eventplot_test.png"

# # create a horizontal plot
# plt.eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
#                     linelengths=linelengths2)

# # create a vertical plot
# #axs[1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
# #                    linelengths=linelengths2, orientation='vertical')

# plt.show()


# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# print(f"Plot saved to {save_path}")

def showRasters(raster, save_path=None, raster_type = 1):
    colors2 = 'black'
    lineoffsets2 = 1
    linelengths2 = 1
    if raster_type == 1: # MF
        print("Raster shape:", raster.shape)
        # raster = np.mean(raster, axis = 0) # Average across trials
        if raster.ndim != 2:
            raise ValueError(f"Expected a 2D raster array, but got shape {raster.shape}")

        numCell = raster.shape[1]
        plotarray = [np.where(raster[:, i] == 1)[0] for i in range(numCell)]

        plt.figure(figsize=(18, 9))
        plt.eventplot(plotarray, colors='black', linelengths=0.8)
        plt.xlabel("Timestep")
        plt.ylabel("Mossy Fiber Number")
        plt.title("MF Spike Raster")
        plt.xlim(0, 5000)

    if raster_type == 2: # Go
        # raster = raster[0]
        # raster = np.mean(raster, axis = 0)
        raster = (np.mean(raster, axis=0) > 0.0).astype(np.uint8)
        numCell = raster.shape[1]
        plotarray = [np.where(raster[:, i] == 1)[0] for i in range(numCell)]

        plt.figure(figsize=(18, 9))
        plt.eventplot(plotarray, colors='black', linelengths=0.8)
        plt.xlabel("Timestep")
        plt.ylabel("Golgi Cell Number")
        plt.title("Golgi Spike Raster")
        plt.xlim(0, 5000)
    
    if raster_type == 3: # Gr
        print("Entered granule raster plotting function")
        # raster = raster[0]
        # raster = np.mean(raster, axis = 0)
        raster = (np.mean(raster, axis=0) > 0.0).astype(np.uint8)
        print("Raster shape:", raster.shape)
        # raster = downsample_granule_cells_only(raster)
        # print("Downsampled shape (only granule cells):", raster.shape)
        
        numCell = 1000 #5000 # raster.shape[1]
        plotarray = [np.where(raster[:, i] == 1)[0] for i in range(numCell)]

        plt.figure(figsize=(18, 9))
        plt.eventplot(plotarray, colors='black', linelengths=0.8)
        plt.xlabel("Timestep")
        plt.ylabel("Granule Cell Number")
        plt.title("Granule Spike Raster")
        plt.xlim(0, 5000)


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def downsample_granule_cells_only(raster, max_cells=10000):
    raster = np.asarray(raster)
    print("Using CPU downsampling. Shape:", raster.shape)

    num_timesteps, num_cells = raster.shape
    downsample_factor = max(1, num_cells // max_cells)

    return raster[:, ::downsample_factor]

# Load the raster data
raster_data = np.load('/home/data/einez/MFGoGr_GRGOplast_150_trials_GOrasters.npy')
print("Finished loading data")
# Define save location
plot_save_path = "/home/aw39625/minisim/Results/Rasters/Eventplot_MFGoGr_GRGOplast_150_trials_GOrasters.png"

# Show and save
showRasters(raster_data, save_path=plot_save_path, raster_type = 2)