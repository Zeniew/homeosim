import numpy as np
import cupy as cp
import os
import time

import MFGoGrFunctions as mfgogr
import importConnect as connect
import WireFunctions

##### Methods #####
def gen_filepaths(exp_name, convergence, gogoW):
    # made for conv and gogoW rn, current naming convention defined here
    filename = exp_name + "_C" + str(convergence) + "_W" + str(int(gogoW * 10000)) + ".cpy"
    filepath = os.path.join(saveDir, filename)
    # filename_g = exp_name + "_g_" + str(sessionNum)+ ".npy"
    # filepath_g = os.path.join(saveDir, filename_g)
    filepath_g = None
    return filepath, filepath_g

def run_session(recip, filpath, filepath_g, conv, grgoW = 0.0007, gogrW = 0.015, RA = False, mfgoW = 0.0042, mfgrW = 0.0042, gogoW = 0.05):
    # Init MF class and create ISI Distributions
    MF = mfgogr.Mossy(numMF, CSon, CSoff)
    MFrasters = cp.zeros((numBins, numMF), dtype = int)

    # # Init GO class
    # GO = mfgogr.Golgi(numGO, CSon, CSoff, useCS, numBins)
    # GOrasters = cp.zeros((numTrial, numBins, numGO), dtype = int)

    # # Init GR class
    # GR = mfgogr.Granule(numGR, CSon, CSoff, useCS, numBins)
    # GRrasters = cp.zeros((numTrial, numBins, numGR), dtype = int)

    # Get connect arrays
    MFGOimportPath = 'connect_arr/connect_arr_PRE.mfgo'
    MFGO_connect_arr = connect.read_connect(MFGOimportPath, numMF, 20)
    MFGRimportPath = 'connect_arr/connect_arr_PRE.mfgr'
    MFGR_connect_arr = connect.read_connect(MFGRimportPath, numMF, 20)
    # GOGRimportPath = "connect_arr/connect_arr_PRE.gogr"
    # GOGR_connect_arr = connect.read_connect(GOGRimportPath, numGO, 20)
    # GRGOimportPath = "connect_arr/connect_arr_PRE.grgo"
    # GRGO_connect_arr = connect.read_connect(GRGOimportPath, numGR, 20)
    # GOGO_connect_arr = WireFunctions.wire_up_verified(conv, recip, span, verbose=False)

    # Sim Core
    #####################
    for trial in range (numTrial):
        # Run trial
        all_start = time.time()
        for t in range(0, numBins):
            MFact = MF.do_MF_dist(t, useCS)
            # # MF input --> GO and GR
            # GO.update_input_activity(MFGO_connect_arr, 1, mfAct = MFact)
            # GR.update_input_activity(MFGR_connect_arr, 1, mfAct = MFact)
            
            # # Update Vm and thresh
            # GO.do_Golgi(t)
            # GR.do_Granule(t)
            
            # # GOGO
            # GO.update_input_activity(GOGO_connect_arr, 2, t = t)
            # GO.do_Golgi(t)
            
            # # Grab activity
            # GRact = GR.get_act()
            # GOact = GO.get_act()
            
            # # GRGO, GOGR
            # GO.update_input_activity(GRGO_connect_arr, 3, grAct = GRact)
            # GR.update_input_activity(GOGR_connect_arr, 2, goAct = GOact)
            
            MFrasters[t, :] = MFact
        # GOrasters[trial] = GO.get_act()
        # GRrasters[trial] = GR.get_act()
        all_end = time.time()
        print(f"Trial: {trial+1}, Time:{(all_end - all_start):.3f}s")

    # Save rasters
    if saveGORaster:
        os.makedirs(saveDir, exist_ok = True)
        cp.save(filepath, GOrasters[:, CSon:CSoff,:])
        print(f"Raster array saved to '{filepath}'")
    if saveGRRaster:
        cp.save(filepath_g, GRrasters[:, CSon:CSoff,:])
        print(f"Raster array saved to '{filepath_g}'")
    if saveMFRaster:
        cp.save(os.path.join(saveDir, f"{expName}_MFrasters.npy"), MFrasters[:, CSon:CSoff,:])
        print(f"Raster array saved to '{saveDir}/{expName}_MFrasters.npy'")

    # if saveGORaster:
    #     cp.save(os.path.join(saveDir, f"{expName}_GOrasters.npy"), GOrasters)
    # if saveGRRaster:
    #     cp.save(os.path.join(saveDir, f"{expName}_GRrasters.npy"), GRrasters)




### Input Params###

# Cell Numbers
numGO = 4096
numMF = 4096
numGR = 10 # 1048576

# Go Activity
# upper lim and lower lim are global variables
upper_lim_GO = 0.014
lower_lim_GO = 0.01

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
saveDir = 'C:/Users/Einez (School)/Desktop/homeosim/Results/'
expName = 'MFGoGr_Experiment_TestMF'
# Save Rasters
saveGORaster = False
saveGRRaster = False
saveMFRaster = True

# GOGO Connect Params
conv_list = [25]
gogoW_list = [0.05] # range can't iter by floats
recip_list = [0.75]
# span = 6 # changing span below

##### Experiment Loop #####

for i in range(len(recip_list)):
    for conv in conv_list:
        for gogoW in gogoW_list:
            if (conv < 15):
                continue
            print(f"Starting Session for GoGo Conv: {conv} W: {gogoW} ...")
            span = int(conv/2) if conv > 5 else 6 
            filepath, filepath_g = gen_filepaths(expName, conv, gogoW)
            recip = round(conv * recip_list[i])
            run_session(recip, filepath, filepath_g, conv, grgoW=0, gogrW=0, mfgrW=0, gogoW = 0)