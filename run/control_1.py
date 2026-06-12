import numpy as np
import cupy as cp
import os
import time
# import json 

import MFGOGrFunctions_synaptic_scaling as mfgogr
# import playground_MFGOGRFunctions as mfgogr
import importConnect as connect
import WireFunctions

##### Methods #####
def gen_filepaths(exp_name):
    # Match printed file names
    filename_m = f"{exp_name}_MFrasters.npy"
    filename_g = f"{exp_name}_GRrasters.npy"
    filename_go = f"{exp_name}_GOrasters.npy"
    filename_w_grgo = f"{exp_name}_grgoW.npy"
    filename_w_gogo = f"{exp_name}_gogoW.npy"
    filename_w_mfgo = f"{exp_name}_mfgoW.npy"
    filename_w_gogr = f"{exp_name}_gogrW.npy"
    filename_w_mfgr = f"{exp_name}_mfgrW.npy"
    
    filepath_m = os.path.join(saveDir,filename_m)
    filepath_go = os.path.join(saveDir, filename_go)
    filepath_gr = os.path.join(saveDir, filename_g)
    filepath_w_grgo = os.path.join(saveDir,filename_w_grgo)
    filepath_w_gogo = os.path.join(saveDir,filename_w_gogo)
    filepath_w_mfgo = os.path.join(saveDir,filename_w_mfgo)
    filepath_w_gogr = os.path.join(saveDir,filename_w_gogr)
    filepath_w_mfgr = os.path.join(saveDir,filename_w_mfgr)
    
    return filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr #, filepath_params

def run_session(filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr):
    
    print("Initializing objects...")

    # Init MF class and create ISI Distributions
    MF = mfgogr.Mossy(numMF, CSon, CSoff)
    MFrasters = np.zeros((numBins, numMF), dtype = np.uint8)

    # # Init GO class
    GO = mfgogr.Golgi(numGO, CSon, CSoff, useCS, numBins, plast_ratio = 1.0)
    # GO = mfgogr.Golgi(numGO, CSon, CSoff, useCS, numBins, gogo_weight = gogoW, mfgo_weight = mfgoW, grgo_weight = grgoW) # playground version
    GOrasters = np.zeros((numTrial, numBins, numGO), dtype = np.uint8)
    GO_gogoW = np.zeros((numTrial, numGO), dtype = np.float64)
    GO_grgoW = np.zeros((numTrial, numGO), dtype = np.float64)
    GO_mfgoW = np.zeros((numTrial, numGO), dtype = np.float64)

    # # Init GR class
    GR = mfgogr.Granule(numGR, CSon, CSoff, useCS, numBins)
    # GR = mfgogr.Granule(numGR, CSon, CSoff, useCS, numBins) # playground version
    # GRrasters = np.zeros((numTrial, numBins, numGR), dtype = np.uint8)
    GRrasters = np.zeros((numTrial, numGR), dtype = np.int32)
    GR_mfgrW = np.zeros((numTrial, numGR), dtype = np.float64)
    GR_gogrW = np.zeros((numTrial, numGR), dtype = np.float64)

    print("Objects initialized.")

    print("Loading connectivity arrays...")

    # Get connect arrays
    MFGO_importPath = '/home/data/einez/connect_arr/connect_arr_PRE.mfgo'
    MFGO_connect_arr = connect.read_connect(MFGO_importPath, numMF, 20)
    MFGO_connect_arr = MFGO_connect_arr[:, :16]
    MFGO_connect_arr[MFGO_connect_arr == -1] = 0
    print("MFGO Connectivity Array Loaded.")

    MFGR_importPath = '/home/data/einez/connect_arr/connect_arr_PRE.mfgr'
    MFGR_connect_arr = connect.read_connect(MFGR_importPath, numMF, 4000)
    MFGR_connect_arr = MFGR_connect_arr[:, :1289]
    MFGR_connect_arr[MFGR_connect_arr == -1] = 0
    print("MFGR Connectivity Array Loaded.")

    GOGR_importPath = "/home/data/einez/connect_arr/connect_arr_PRE.gogr"
    GOGR_connect_arr = connect.read_connect(GOGR_importPath, numGO, 12800) # TRUNCATE: just grab the first 975 columns, the furthest start of -1 is at 975
    GOGR_connect_arr = GOGR_connect_arr[:, :975]
    # GOGR_connect_arr[GOGR_connect_arr == -1] = 0
    # print(GOGR_connect_arr.shape)
    # max_val = 0
    # for i in range(4096):
    #     indices = np.where(GOGR_connect_arr[i, :] == -1)[0]  # grab the row indices
    #     if indices.size > 0:  # check if any matches
    #         local_max = indices.min()
    #         if local_max > max_val:
    #             max_val = local_max
    # print("max:", max_val)
    # exit()
    print("GOGR Connectivity Array Loaded.")

    GRGO_importPath = "/home/data/einez/connect_arr/connect_arr_PRE.grgo"
    GRGO_connect_arr = connect.read_connect(GRGO_importPath, numGR, 50)
    GRGO_connect_arr = GRGO_connect_arr[:, :30]
    GRGO_connect_arr[GRGO_connect_arr == -1] = 0 # changes the -1 padding to index 0
    print("GRGO Connectivity Array Loaded.")

    # GOGO_connect_arr = WireFunctions.wire_up_verified(conv, recip, span, verbose=False)
    GOGO_importPath = "/home/data/einez/connect_arr/connect_arr_PRE.gogo"
    GOGO_connect_arr = connect.read_connect(GOGO_importPath, numGO, 12)
    GOGO_connect_arr[GOGO_connect_arr == -1] = 0
    print("GOGO Connectivity Array Loaded.")


    print("Connectivity Arrays Loaded")

    # Sim Core
    #####################
    for trial in range (numTrial):
        # Run trial
        all_start = time.time()
        MF_time = 0
        MFGR_time = 0
        GR_time = 0
        GRGO_time = 0
        MFGO_time = 0
        GO_time = 0
        GOGO_time = 0
        GOGR_time = 0
        for t in range(numBins):
            MFact = MF.do_MF_dist(t, useCS)

            # MF -> GR update
            GR.update_input_activity(MFGR_connect_arr, 1, mfAct = MFact)

            # do gr spikes
            GR.do_Granule(t)

            # grab GR activity
            GRact = GR.get_act()

            # GR -> GO update
            GO.update_input_activity(GRGO_connect_arr, 3, grAct = GRact[t]) # for the new version of GRGO

            # MF -> GO
            GO.update_input_activity(MFGO_connect_arr, 1, mfAct = MFact)

            # GO spikes
            GO.do_Golgi(t)

            # GO -> GO update
            GO.update_input_activity(GOGO_connect_arr, 2, t = t)

            GOact = GO.get_act()
            # GO -> GR update
            GR.update_input_activity(GOGR_connect_arr, 2, goAct = GOact[t])

            MFrasters[t, :] = MFact

            # print(f"Finished timestep: {t}")
        
        # Update plasticity weight
        if MFGO_PLAST == 1:
            GO.mfgoW = GO.update_weight(trial, exc_or_inh = 1, weight_array = GO.get_mfgoW())
            # --- DEBUG: PRINT TRUTH ---
            # Print Cell 1 (Plastic) vs Cell 2 (Should be Static)
            w = GO.get_mfgoW()
            print(f"Trial {trial}: Cell 1 (Static) = {w[1]:.6f} | Cell 2 (Plastic) = {w[2]:.6f}")
            # --------------------------
        if GOGO_PLAST == 1:
            GO.gogoW = GO.update_weight(trial, exc_or_inh = 2, weight_array = GO.get_gogoW())
        if GRGO_PLAST == 1:
            GO.grgoW = GO.update_weight(trial, exc_or_inh = 1, weight_array = GO.get_grgoW())
        if MFGR_PLAST == 1:
            GR.mfgrW = GR.update_weight(trial, exc_or_inh = 1, weight_array = GR.get_mfgrW())
            GR.GPU_mfgrW[:] = cp.asarray(GR.mfgrW, dtype = cp.float32) # update GPU copy of weights
        if GOGR_PLAST == 1:
            GR.gogrW = GR.update_weight(trial, exc_or_inh = 2, weight_array = GR.get_gogrW())
            GR.GPU_gogrW[:] = cp.asarray(GR.gogrW, dtype = cp.float32) # update GPU copy of weights
    
        GO_gogoW[trial] = (GO.get_gogoW().copy()) 
        GO_grgoW[trial] = (GO.get_grgoW().copy())
        GO_mfgoW[trial] = (GO.get_mfgoW().copy())
        GR_gogrW[trial] = (GR.get_gogrW().copy())
        GR_mfgrW[trial] = (GR.get_mfgrW().copy())

        # Final update
        GR.updateFinalState()
        # Rasters
        GOrasters[trial] = GO.get_act()
        # GRrasters[trial] = GR.get_act()
        GRrasters[trial] = GR.get_summed_act()
        print(np.sum(GRrasters[trial]))
        GR.reset_GPU_summed_act()
        all_end = time.time()
        # Shuffling MF
        MF.generate_MFisiDistribution()
        # print(f"MF_time: {MF_time:.3f}s | MFGR_time: {MFGR_time:.3f}s | GR_time: {GR_time:.3f}s | GRGO_time: {GRGO_time:.3f}s | MFGO_time: {MFGO_time:.3f}s | GO_time: {GO_time:.3f}s | GOGO_time: {GOGO_time:.3f}s | GOGR_time: {GOGR_time:.3f}s")
        print(f"Trial: {trial+1}, Time:{(all_end - all_start):.3f}s")
    

    # Save rasters
    if saveGORaster:
        os.makedirs(saveDir, exist_ok = True)
        cp.save(filepath_go, GOrasters)
        print(f"Raster array saved to '{filepath_go}'")
    if saveGRRaster:
        os.makedirs(saveDir, exist_ok = True)
        cp.save(filepath_gr, GRrasters)
        print(f"Raster array saved to '{filepath_gr}'")
    if saveMFRaster:
        # cp.save(os.path.join(saveDir, f"{expName}_MFrasters.npy"), MFrasters)
        os.makedirs(saveDir, exist_ok = True)
        cp.save(filepath_m, MFrasters)
        print(f"Raster array saved to '{filepath_m}'")
    if saveWeights:
        os.makedirs(saveDir, exist_ok = True)
        cp.save(filepath_w_gogo, GO_gogoW)
        print(f"Raster array saved to '{filepath_w_gogo}'")
        cp.save(filepath_w_grgo, GO_grgoW)
        print(f"Raster array saved to '{filepath_w_grgo}'")
        cp.save(filepath_w_mfgo, GO_mfgoW)
        print(f"Raster array saved to '{filepath_w_mfgo}'")
        cp.save(filepath_w_gogr, GR_gogrW)
        print(f"Raster array saved to '{filepath_w_gogr}'")
        cp.save(filepath_w_mfgr, GR_mfgrW)
        print(f"Raster array saved to '{filepath_w_mfgr}'")
   
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


# Trial Params
numBins = 5000 
useCS = 0
CSon, CSoff = 500, 3500
numTrial = 1000
MFGO_PLAST = 0
GOGO_PLAST = 1
GRGO_PLAST = 0
MFGR_PLAST = 0
GOGR_PLAST = 0

# saving to hard drive
expName = f'MFGoGr_SS_shuffleMF10percent_noCS_yesGoGo_yesgrGo_gogoplast_{numTrial}_trial'
saveDir = f'/home/data/einez/homeostat_SS/{expName}'

# Save Rasters
saveGORaster = True
saveGRRaster = True
saveMFRaster = True
saveWeights = True



##### Experiment Loop #####
print(f"Starting Session...")
filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr = gen_filepaths(expName)
run_session(filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr)