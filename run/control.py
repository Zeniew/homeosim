import numpy as np
import cupy as cp
import os
import time

import MFGOGrFunctions_discrete_trace as mfgogr
# import playground_MFGOGRFunctions as mfgogr
import importConnect as connect
import WireFunctions

##### Methods #####
def gen_filepaths(exp_name, convergence, gogoW):
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
    
    return filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr

def run_session(recip, filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr, conv, grgoW = 0.0007, gogrW = 0.015, RA = False, mfgoW = 0.0042, mfgrW = 0.0042, gogoW = 0.05):
    
    print("Initializing objects...")

    # Init MF class and create ISI Distributions
    MF = mfgogr.Mossy(numMF, CSon, CSoff)
    MFrasters = np.zeros((numBins, numMF), dtype = np.uint8)

    # # Init GO class
    GO = mfgogr.Golgi(numGO, CSon, CSoff, useCS, numBins, mfgo_plast = MFGO_PLAST, gogo_plast = GOGO_PLAST, grgo_plast = GRGO_PLAST, gogo_weight = gogoW, mfgo_weight = mfgoW, grgo_weight = grgoW)
    # GO = mfgogr.Golgi(numGO, CSon, CSoff, useCS, numBins, gogo_weight = gogoW, mfgo_weight = mfgoW, grgo_weight = grgoW) # playground version
    GOrasters = np.zeros((numTrial, numBins, numGO), dtype = np.uint8)
    GO_avg_gogoW = np.zeros((numTrial, numBins), dtype = np.float32)
    GO_avg_grgoW = np.zeros((numTrial, numBins), dtype = np.float32)
    GO_avg_mfgoW = np.zeros((numTrial, numBins), dtype = np.float32)

    # # Init GR class
    GR = mfgogr.Granule(numGR, CSon, CSoff, useCS, numBins, mfgr_plast = MFGR_PLAST, gogr_plast = GOGR_PLAST)
    # GR = mfgogr.Granule(numGR, CSon, CSoff, useCS, numBins) # playground version
    # GRrasters = np.zeros((numTrial, numBins, numGR), dtype = np.uint8)
    GR_avg_mfgrW = np.zeros((numTrial, numBins), dtype = np.float32)
    GR_avg_gogrW = np.zeros((numTrial, numBins), dtype = np.float32)

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
        for t in range(0, numBins):
            # print(t, ":", GO.get_grgoW())

            # timestep_start = time.time()
            MFact = MF.do_MF_dist(t, useCS)
            # MF_end = time.time()

            # MF -> GR update
            GR.update_input_activity(MFGR_connect_arr, 1, mfAct = MFact)

            # MFGR_end = time.time()

            # do gr spikes
            GR.do_Granule(t)

            # GR_end = time.time()

            # grab GR activity
            GRact = GR.get_act()

            # GR -> GO update
            # GO.update_input_activity(GOGR_connect_arr, 3, grAct = GRact[trial])
            GO.update_input_activity(GRGO_connect_arr, 3, grAct = GRact[trial]) # for the new version of GRGO

            # GRGO_end = time.time()
            # print("GRGO time taken:", GRGO_end - GRGO_start, "seconds")

            # MF -> GO
            GO.update_input_activity(MFGO_connect_arr, 1, mfAct = MFact)

            # MFGO_end = time.time()

            # GO spikes
            GO.do_Golgi(t)

            # GOspike_end = time.time()

            # GO -> GO update
            GO.update_input_activity(GOGO_connect_arr, 2, t = t)

            # test - see what happens when I add another do_Golgi
            # GO.do_Golgi(t)

            # GOGO_end = time.time()

            GOact = GO.get_act()
            # GO -> GR update
            GR.update_input_activity(GOGR_connect_arr, 2, goAct = GOact[trial])

            # test - see what happens when I add another do_Granule
            # GR.do_Granule(t)

            # GOGR_end = time.time()
            
            # grab the weight array at each time step and averaging them for recording
            MFrasters[t, :] = MFact
            GO_avg_gogoW[trial][t] = np.mean(GO.get_gogoW()) 
            GO_avg_grgoW[trial][t] = np.mean(GO.get_grgoW())
            GO_avg_mfgoW[trial][t] = np.mean(GO.get_mfgoW())
            GR_avg_gogrW[trial][t] = np.mean(GR.get_gogrW())
            GR_avg_mfgrW[trial][t] = np.mean(GR.get_mfgrW())

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
        # GRrasters[trial] = GR.get_act()
        all_end = time.time()
        print(f"Trial: {trial+1}, Time:{(all_end - all_start):.3f}s")
        
    print(GO_avg_grgoW)

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
        cp.save(filepath_w_gogo, GO_avg_gogoW)
        print(f"Raster array saved to '{filepath_w_gogo}'")
        cp.save(filepath_w_grgo, GO_avg_grgoW)
        print(f"Raster array saved to '{filepath_w_grgo}'")
        cp.save(filepath_w_mfgo, GO_avg_mfgoW)
        print(f"Raster array saved to '{filepath_w_mfgo}'")
        cp.save(filepath_w_mfgr, GR_avg_mfgrW)
        print(f"Raster array saved to '{filepath_w_mfgr}'")
        cp.save(filepath_w_gogr, GR_avg_gogrW)
        print(f"Raster array saved to '{filepath_w_gogr}'")
        

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
numTrial = 20 # 150
MFGO_PLAST = 1
GOGO_PLAST = 0
GRGO_PLAST = 0
MFGR_PLAST = 0
GOGR_PLAST = 0

# saving to hard drive
saveDir = '/home/data/einez'
expName = 'MFGoGr_MFGOplast_discrete_trace_20_trials'

# Save Rasters
saveGORaster = True
saveGRRaster = False
saveMFRaster = False
saveWeights = True

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
            filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr = gen_filepaths(expName, conv, gogoW)
            recip = round(conv * recip_list[i])
            run_session(recip, filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr, conv, grgoW=0.0007, gogrW=0.015, gogoW = 0.0125)