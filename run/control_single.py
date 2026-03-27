import numpy as np
import cupy as cp
import os
import time

import MFGOGrFunctions_single as mfgogr

##### Methods #####
def gen_filepaths(exp_name):
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

def run_session(filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr, grgoW=0.0007, gogrW=0.015, mfgoW=0.0042, mfgrW=0.001, gogoW=0.0125):
    
    print("Initializing single-cell circuit objects...")

    # Init MF class
    MF = mfgogr.Mossy(numMF, CSon, CSoff)
    MFrasters = np.zeros((numBins, numMF), dtype = np.uint8)

    # Init GO class
    GO = mfgogr.Golgi(numGO, CSon, CSoff, useCS, numBins, mfgo_plast = MFGO_PLAST, gogo_plast = GOGO_PLAST, grgo_plast = GRGO_PLAST, gogo_weight = gogoW, mfgo_weight = mfgoW, grgo_weight = grgoW, plast_ratio = 1.0)
    GOrasters = np.zeros((numTrial, numBins, numGO), dtype = np.uint8)
    GO_gogoW = np.zeros((numTrial, numGO), dtype = np.float32)
    GO_grgoW = np.zeros((numTrial, numGO), dtype = np.float32)
    GO_mfgoW = np.zeros((numTrial, numGO), dtype = np.float32)

    # Init GR class
    GR = mfgogr.Granule(numGR, CSon, CSoff, useCS, numBins, mfgr_plast = MFGR_PLAST, gogr_plast = GOGR_PLAST, mfgr_weight = mfgrW, gogr_weight = gogrW)

    print("Objects initialized. Connectivity arrays removed for single-cell circuit.")

    # Sim Core
    #####################
    for trial in range (numTrial):
        all_start = time.time()
        for t in range(0, numBins):

            MFact = MF.do_MF_dist(t, useCS)

            # MF -> GR update (Direct 1-to-1)
            GR.update_input_activity(1, mfAct = MFact)

            # do gr spikes
            GR.do_Granule(t)
            GRact = GR.get_act()

            # GR -> GO update (Direct 1-to-1)
            GO.update_input_activity(3, grAct = GRact[t])

            # MF -> GO (Direct 1-to-1)
            GO.update_input_activity(1, mfAct = MFact)

            # GO spikes
            GO.do_Golgi(t)

            # GO -> GO update (Direct 1-to-1)
            GO.update_input_activity(2, t = t)

            GOact = GO.get_act()
            
            # GO -> GR update (Direct 1-to-1)
            GR.update_input_activity(2, goAct = GOact[t])

            MFrasters[t, :] = MFact
        
        # Update plasticity weight
        if MFGO_PLAST == 1:
            GO.mfgoW = GO.update_weight(trial, exc_or_inh = 1, weight_array = GO.get_mfgoW())
        if GOGO_PLAST == 1:
            GO.gogoW = GO.update_weight(trial, exc_or_inh = 2, weight_array = GO.get_gogoW())
        if GRGO_PLAST == 1:
            GO.grgoW = GO.update_weight(trial, exc_or_inh = 1, weight_array = GO.get_grgoW())
    
        GO_gogoW[trial] = GO.get_gogoW()
        GO_grgoW[trial] = GO.get_grgoW()
        GO_mfgoW[trial] = GO.get_mfgoW()

        # Final update
        GR.updateFinalState()
        GOrasters[trial] = GO.get_act()
        all_end = time.time()
        # MF.generate_MFisiDistribution()
        # print(f"Trial: {trial+1}, Time:{(all_end - all_start):.3f}s")
    
    # Save rasters
    if saveGORaster:
        os.makedirs(saveDir, exist_ok = True)
        cp.save(filepath_go, GOrasters)
        print(f"Raster array saved to '{filepath_go}'")
    if saveMFRaster:
        os.makedirs(saveDir, exist_ok = True)
        cp.save(filepath_m, MFrasters)
        print(f"Raster array saved to '{filepath_m}'")
    if saveWeights:
        os.makedirs(saveDir, exist_ok = True)
        cp.save(filepath_w_gogo, GO_gogoW)
        cp.save(filepath_w_grgo, GO_grgoW)
        cp.save(filepath_w_mfgo, GO_mfgoW)
        print("Weight arrays saved.")

### Input Params###

# Cell Numbers (Set to 1 for Single-Cell Circuit)
numGO = 1
numMF = 1
numGR = 1

# Trial Params
numBins = 5000 
useCS = 0
CSon, CSoff = 500, 3500
numTrial = 100
MFGO_PLAST = 1
GOGO_PLAST = 0
GRGO_PLAST = 0
MFGR_PLAST = 0
GOGR_PLAST = 0

# saving to hard drive
saveDir = '/home/data/einez'
expName = 'SingleCell_stagnantMF_MFGoGr_circuit_mfgoplast_100_trials'

# Save Rasters
saveGORaster = True
saveGRRaster = True
saveMFRaster = True
saveWeights = True

##### Experiment Loop #####
print("Starting Single-Cell Session...")
filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr = gen_filepaths(expName)
run_session(filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo, filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr, gogoW = 0)