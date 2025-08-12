import numpy as np
import cupy as cp
import os
import time

import MFGoGrFunctions as mfgogr
import importConnect as connect
import WireFunctions

##### Methods #####
def gen_filepaths(exp_name, convergence, gogoW):
    # Match printed file names
    filename_m = f"{exp_name}_MFrasters.npy"
    filename_g = f"{exp_name}_GRrasters.npy"
    filename_go = f"{exp_name}_GOrasters.npy"
    
    filepath_m = os.path.join(saveDir,filename_m)
    filepath_go = os.path.join(saveDir, filename_go)
    filepath_gr = os.path.join(saveDir, filename_g)
    
    return filepath_m, filepath_go, filepath_gr

def run_session(recip, filpath_m, filepath_go, filepath_gr, conv, grgoW = 0.0007, gogrW = 0.015, RA = False, mfgoW = 0.0042, mfgrW = 0.0042, gogoW = 0.05):
    
    print("Initializing objects...")

    # Init MF class and create ISI Distributions
    MF = mfgogr.Mossy(numMF, CSon, CSoff)
    MFrasters = np.zeros((numBins, numMF), dtype = np.uint8)

    # # Init GO class
    GO = mfgogr.Golgi(numGO, CSon, CSoff, useCS, numBins, gogo_weight = gogoW, mfgo_weight = mfgoW, grgo_weight = grgoW)
    GOrasters = np.zeros((numTrial, numBins, numGO), dtype = np.uint8)

    # # Init GR class
    GR = mfgogr.Granule(numGR, CSon, CSoff, useCS, numBins)
    GRrasters = np.zeros((numTrial, numBins, numGR), dtype = np.uint8)
    print("Objects initialized.")

    print("Loading connectivity arrays...")

    # Get connect arrays
    MFGO_importPath = '/home/data/einez/connect_arr/connect_arr_PRE.mfgo'
    MFGO_connect_arr = connect.read_connect(MFGO_importPath, numMF, 20)
    print("MFGO Connectivity Array Loaded.")
    MFGR_importPath = '/home/data/einez/connect_arr/connect_arr_PRE.mfgr'
    MFGR_connect_arr = connect.read_connect(MFGR_importPath, numMF, 4000)
    print("MFGR Connectivity Array Loaded.")
    GOGR_importPath = "/home/data/einez/connect_arr/connect_arr_PRE.gogr"
    GOGR_connect_arr = connect.read_connect(GOGR_importPath, numGO, 12800)
    GOGR_connect_arr[GOGR_connect_arr == -1] = 0 # changes the -1 padding to index 0
    print("GOGR Connectivity Array Loaded.")
    GRGO_importPath = "/home/data/einez/connect_arr/connect_arr_PRE.grgo"
    GRGO_connect_arr = connect.read_connect(GRGO_importPath, numGR, 50)
    # GRGO_connect_arr[GRGO_connect_arr == -1] = 0 # changes the -1 padding to index 0
    print("GRGO Connectivity Array Loaded.")
    # GOGO_connect_arr = WireFunctions.wire_up_verified(conv, recip, span, verbose=False)
    GOGO_importPath = "/home/data/einez/connect_arr/connect_arr_PRE.gogo"
    GOGO_connect_arr = connect.read_connect(GOGO_importPath, numGO, 12)
    print("GOGO Connectivity Array Loaded.")

    print("Connectivity Arrays Loaded")

    # Sim Core
    #####################
    for trial in range (numTrial):
        # Run trial
        all_start = time.time()
        for t in range(0, numBins):
            timestep_start = time.time()
            MFact = MF.do_MF_dist(t, useCS)
            MF_end = time.time()

            # MF -> GR update
            GR.update_input_activity(MFGR_connect_arr, 1, mfAct = MFact)

            MFGR_end = time.time()

            # do gr spikes
            GR.do_Granule(t)

            GR_end = time.time()

            # grab GR activity
            GRact = GR.get_act()

            # GR -> GO update
            GO.update_input_activity(GOGR_connect_arr, 3, grAct = GRact[trial])

            GRGO_end = time.time()
            # print("GRGO time taken:", GRGO_end - GRGO_start, "seconds")

            # MF -> GO
            GO.update_input_activity(MFGO_connect_arr, 1, mfAct = MFact)

            MFGO_end = time.time()

            # GO spikes
            GO.do_Golgi(t)

            GOspike_end = time.time()

            # GO -> GO update
            GO.update_input_activity(GOGO_connect_arr, 2, t = t)

            GOGO_end = time.time()

            GOact = GO.get_act()
            # GO -> GR update
            GR.update_input_activity(GOGR_connect_arr, 2, goAct = GOact[trial])

            GOGR_end = time.time()
            
            MFrasters[t, :] = MFact

            # print("MF time:", MF_end - timestep_start)
            # print("MFGR time:", MFGR_end - MF_end)
            # print("GR time:", GR_end - MFGR_end)
            # print("GRGO time:", GRGO_end - GR_end)
            # print("MFGO time:", MFGO_end - GRGO_end)
            # print("GO spikes time:", GOspike_end - MFGO_end)
            # print("GOGO time:", GOGO_end - GOspike_end)
            # print("GOGR time:", GOGR_end - GOGO_end)

            # print("Time step:", t)
        # Final update
        GR.updateFinalState()
        # Rasters
        GOrasters[trial] = GO.get_act()
        GRrasters[trial] = GR.get_act()
        all_end = time.time()
        print(f"Trial: {trial+1}, Time:{(all_end - all_start):.3f}s")

    # Save rasters
    if saveGORaster:
        os.makedirs(saveDir, exist_ok = True)
        cp.save(filepath_go, GOrasters)
        print(f"Raster array saved to '{filepath_go}'")
    if saveGRRaster:
        cp.save(filepath_gr, GRrasters)
        print(f"Raster array saved to '{filepath_gr}'")
    if saveMFRaster:
        # cp.save(os.path.join(saveDir, f"{expName}_MFrasters.npy"), MFrasters)
        cp.save(filepath_m, MFrasters)
        print(f"Raster array saved to '{filepath_m}'")

    # if saveGORaster:
    #     cp.save(os.path.join(saveDir, f"{expName}_GOrasters.npy"), GOrasters)
    # if saveGRRaster:
    #     cp.save(os.path.join(saveDir, f"{expName}_GRrasters.npy"), GRrasters)




### Input Params###

# Cell Numbers
numGO = 4096
numMF = 4096
numGR = 1048576

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
numBins =  5000 # 5000
useCS = 1
CSon, CSoff = 500, 3500
numTrial = 1 # 150

# saving to hard drive
saveDir = '/home/data/einez'
expName = 'MFGoGr_full'

# Save Rasters
saveGORaster = True
saveGRRaster = True
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
            filepath_m, filepath_go, filepath_gr = gen_filepaths(expName, conv, gogoW)
            recip = round(conv * recip_list[i])
            run_session(recip, filepath_m, filepath_go, filepath_gr, conv, grgoW=0, gogrW=0, gogoW = 0)