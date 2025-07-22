import numpy as np
import os
import time

import MFGOGRFunctions as mfgogr
import importConnect as connect
import WireFunctions

#def gen_filepaths(exp_name)

# function gets param globally bc they're all in same file
# anything that is being changed each session needs to be passed in

### Input Params###

# Cell Numbers
numGO = 4096
numMF = 4096
numGR = 1048576

# Go Activity
# upper lim and lower lim are global variables
upper_lim_GO = 0.014
lower_lim_GR = 0.01

# GR Activity ! will have to change this later
upper_lim_GR = 0.014
lower_lim_GR = 0.01

# GOGO Connect Params
conv_list = [25]
gogoW_list = [0.05] # range cant iter by floats
recip_list = [0.75] 
#span = 6 # changing span below

# Trial Params
numBins = 50000
useCS = 1
CSon, CSoff = 500, 3500
numTrial = 150

# saving to hard drive
saveDir = 'C:/Users/Einez (School)/Desktop/Mauk Lab Notes/small_sim_skeleton/Results/'
expName = 'MFGoGr_Experiment'
# Save Rasters
saveGORaster = True
saveGRRaster = True

##### Experiment Loop #####

##### Methods #####

def run_Session(recip, filpath, filepath_g, conv, grgoW, gogrW, RA = False, mfgoW = 0.0042, mfgrW = 0.0042):
    # Init MF class and create ISI Distributions
    MF = mfgogr.MF(numMF, CSon, CSoff)
    MFrasters = np.zeros((numBins, numMF), dtype = int)

    # Init GO class
    GO = mfgogr.Golgi(numGO, CSon, CSoff, useCS, numBins)
    GOrasters = np.zeros((numTrial, numBins, numGO), dtype = int)

    # Init GR class
    GR = mfgogr.Granule(numGR, CSon, CSoff, useCS, numBins)
    GRrasters = np.zeros((numTrial, numBins, numGR), dtype = int)

    # Get connect arrays
    MFGOimportPath = ''
    MFGO_connect_arr = connect.read_connect(MFGOimportPath, numMF, 20)
    MFGRimportPath = ''
    MFGR_connect_arr = connect.read_connect(MFGRimportPath, numMF, 20)
    GOGRimportPath = ''
    GOGR_connect_arr = connect.read_connect(GOGRimportPath, numGO, 20)
    GRGOimportPath = ''
    GRGO_connect_arr = connect.read_connect(GRGOimportPath, numGR, 20)
    GOGO_connect_arr = WireFunctions.wire_up_cerified(conv, recip, span)

    # Sim Core
    #####################
    for trial in range (numTrial):
        # Run trial
        all_start = time.time()
        for t in range(0, numBins):
            MFact = MF.do_MF_dist(t, useCS)
            # MF input --> GO and GR
            GO.update_input_activity(MFGO_connect_arr, 1, mfAct = MFact)
            GR.update_input_activity(MFGR_connect_arr, 1, mfAct = MFact)
            
            # Update Vm and thresh
            GO.do_Golgi(t)
            GR.do_Granule(t)
            
            # GOGO
            GO.update_input_activity(GOGO_connect_arr, 2, t = t)
            GO.do_Golgi(t)
            
            # Grab activity
            GRact = GR.get_act()
            GOact = GO.get_act()
            
            # GRGO, GOGR
            GO.update_input_activity(GRGO_connect_arr, 3, grAct = GRact)
            GR.update_input_activity(GOGR_connect_arr, 2, goAct = GOact)
            
            MFrasters[t, :] = MFact
        GOrasters[trial] = GO.get_act()
        GRrasters[trial] = GR.get_act()
        all_end = time.time()
        print(f"Trial: {trials+1}, Time:{(all_end - all_start):.3f}s")

    # Save rasters
    if saveGORaster:
        np.save(os.path.join(saveDir, f"{expName}_GOrasters.npy"), GOrasters)
    if saveGRRaster:
        np.save(os.path.join(saveDir, f"{expName}_GRrasters.npy"), GRrasters)


