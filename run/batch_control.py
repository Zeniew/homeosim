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
def gen_filepaths(exp_name, save_dir):
    # Match printed file names
    filename_m = f"{exp_name}_MFrasters.npy"
    filename_g = f"{exp_name}_GRrasters.npy"
    filename_go = f"{exp_name}_GOrasters.npy"
    filename_w_grgo = f"{exp_name}_grgoW.npy"
    filename_w_gogo = f"{exp_name}_gogoW.npy"
    filename_w_mfgo = f"{exp_name}_mfgoW.npy"
    filename_w_gogr = f"{exp_name}_gogrW.npy"
    filename_w_mfgr = f"{exp_name}_mfgrW.npy"

    filepath_m = os.path.join(save_dir, filename_m)
    filepath_go = os.path.join(save_dir, filename_go)
    filepath_gr = os.path.join(save_dir, filename_g)
    filepath_w_grgo = os.path.join(save_dir, filename_w_grgo)
    filepath_w_gogo = os.path.join(save_dir, filename_w_gogo)
    filepath_w_mfgo = os.path.join(save_dir, filename_w_mfgo)
    filepath_w_gogr = os.path.join(save_dir, filename_w_gogr)
    filepath_w_mfgr = os.path.join(save_dir, filename_w_mfgr)

    return (filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo,
            filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr)


def run_session(filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo,
                filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr,
                numMF, numGO, numGR, CSon, CSoff, useCS, numBins, numTrial,
                MFGO_PLAST, GOGO_PLAST, GRGO_PLAST, MFGR_PLAST, GOGR_PLAST,
                saveDir, saveGORaster, saveGRRaster, saveMFRaster, saveWeights):

    print("Initializing objects...")

    # Init MF class and create ISI Distributions
    MF = mfgogr.Mossy(numMF, CSon, CSoff)
    MFrasters = np.zeros((numBins, numMF), dtype=np.uint8)

    # # Init GO class
    GO = mfgogr.Golgi(numGO, CSon, CSoff, useCS, numBins, plast_ratio=1.0)
    GOrasters = np.zeros((numTrial, numBins, numGO), dtype=np.uint8)
    GO_gogoW = np.zeros((numTrial, numGO), dtype=np.float64)
    GO_grgoW = np.zeros((numTrial, numGO), dtype=np.float64)
    GO_mfgoW = np.zeros((numTrial, numGO), dtype=np.float64)

    # # Init GR class
    GR = mfgogr.Granule(numGR, CSon, CSoff, useCS, numBins)
    GRrasters = np.zeros((numTrial, numGR), dtype=np.int32)
    GR_mfgrW = np.zeros((numTrial, numGR), dtype=np.float64)
    GR_gogrW = np.zeros((numTrial, numGR), dtype=np.float64)

    print("Objects initialized.")

    print("Loading connectivity arrays...")

    # Get connect arrays
    MFGO_importPath = '/home/data/einez/connect_arr/connect_arr_PRE.mfgo'
    MFGO_connect_arr = connect.read_connect(MFGO_importPath, numMF, 20)
    MFGO_connect_arr = MFGO_connect_arr[:, :16]
    print("MFGO Connectivity Array Loaded.")

    MFGR_importPath = '/home/data/einez/connect_arr/connect_arr_PRE.mfgr'
    MFGR_connect_arr = connect.read_connect(MFGR_importPath, numMF, 4000)
    MFGR_connect_arr = MFGR_connect_arr[:, :1289]
    print("MFGR Connectivity Array Loaded.")

    GOGR_importPath = "/home/data/einez/connect_arr/connect_arr_PRE.gogr"
    GOGR_connect_arr = connect.read_connect(GOGR_importPath, numGO, 12800)
    GOGR_connect_arr = GOGR_connect_arr[:, :975]
    print("GOGR Connectivity Array Loaded.")

    GRGO_importPath = "/home/data/einez/connect_arr/connect_arr_PRE.grgo"
    GRGO_connect_arr = connect.read_connect(GRGO_importPath, numGR, 50)
    GRGO_connect_arr = GRGO_connect_arr[:, :30]
    print("GRGO Connectivity Array Loaded.")

    GOGO_importPath = "/home/data/einez/connect_arr/R75_C12_PRE.gogo"
    GOGO_connect_arr = connect.read_connect(GOGO_importPath, numGO, 12)
    print("GOGO Connectivity Array Loaded.")

    print("Connectivity Arrays Loaded")

    # Sim Core
    #####################
    for trial in range(numTrial):
        # Run trial
        all_start = time.time()
        for t in range(numBins):
            MFact = MF.do_MF_dist(t, useCS)

            # MF -> GR update
            GR.update_input_activity(MFGR_connect_arr, 1, mfAct=MFact)

            # do gr spikes
            GR.do_Granule(t)

            # grab GR activity
            GRact = GR.get_act()

            # GR -> GO update
            GO.update_input_activity(GRGO_connect_arr, 3, grAct=GRact[t])

            # MF -> GO
            GO.update_input_activity(MFGO_connect_arr, 1, mfAct=MFact)

            # GO spikes
            GO.do_Golgi(t)

            # GO -> GO update
            GO.update_input_activity(GOGO_connect_arr, 2, t=t)

            GOact = GO.get_act()
            # GO -> GR update
            GR.update_input_activity(GOGR_connect_arr, 2, goAct=GOact[t])

            MFrasters[t, :] = MFact

        # Update plasticity weight
        if MFGO_PLAST == 1:
            GO.mfgoW = GO.update_weight(trial, exc_or_inh=1, weight_array=GO.get_mfgoW())
            w = GO.get_mfgoW()
            print(f"Trial {trial}: Cell 1 (Static) = {w[1]:.6f} | Cell 2 (Plastic) = {w[2]:.6f}")
        if GOGO_PLAST == 1:
            GO.gogoW = GO.update_weight(trial, exc_or_inh=2, weight_array=GO.get_gogoW())
        if GRGO_PLAST == 1:
            GO.grgoW = GO.update_weight(trial, exc_or_inh=1, weight_array=GO.get_grgoW())
        if MFGR_PLAST == 1:
            GR.mfgrW = GR.update_weight(trial, exc_or_inh=1, weight_array=GR.get_mfgrW())
            GR.GPU_mfgrW[:] = cp.asarray(GR.mfgrW, dtype=cp.float32)
        if GOGR_PLAST == 1:
            GR.gogrW = GR.update_weight(trial, exc_or_inh=2, weight_array=GR.get_gogrW())
            GR.GPU_gogrW[:] = cp.asarray(GR.gogrW, dtype=cp.float32)

        GO_gogoW[trial] = (GO.get_gogoW().copy())
        GO_grgoW[trial] = (GO.get_grgoW().copy())
        GO_mfgoW[trial] = (GO.get_mfgoW().copy())
        GR_gogrW[trial] = (GR.get_gogrW().copy())
        GR_mfgrW[trial] = (GR.get_mfgrW().copy())

        # Final update
        GR.updateFinalState()
        # Rasters
        GOrasters[trial] = GO.get_act()
        GRrasters[trial] = GR.get_summed_act()
        print(np.sum(GRrasters[trial]))
        GR.reset_GPU_summed_act()
        all_end = time.time()
        # Shuffling MF
        MF.generate_MFisiDistribution()
        print(f"Trial: {trial+1}, Time:{(all_end - all_start):.3f}s")

    # Save rasters
    if saveGORaster:
        os.makedirs(saveDir, exist_ok=True)
        cp.save(filepath_go, GOrasters)
        print(f"Raster array saved to '{filepath_go}'")
    if saveGRRaster:
        os.makedirs(saveDir, exist_ok=True)
        cp.save(filepath_gr, GRrasters)
        print(f"Raster array saved to '{filepath_gr}'")
    if saveMFRaster:
        os.makedirs(saveDir, exist_ok=True)
        cp.save(filepath_m, MFrasters)
        print(f"Raster array saved to '{filepath_m}'")
    if saveWeights:
        os.makedirs(saveDir, exist_ok=True)
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


def run_experiment(expName, numTrial, useCS, MFGO_PLAST, GOGO_PLAST, GRGO_PLAST,
                    MFGR_PLAST, GOGR_PLAST, saveGORaster, saveGRRaster, saveMFRaster,
                    saveWeights, numGO=4096, numMF=4096, numGR=1048576,
                    numBins=5000, CSon=500, CSoff=3500,
                    base_dir='/home/data/einez/homeostat_SS'):
    """
    Run a single MFGOGR experiment with the given configuration.

    Parameters that vary between experiments (expName, numTrial, useCS, the
    five *_PLAST flags, and the four save* flags) are required arguments.
    Cell counts / bin timing / base_dir have sensible defaults matching the
    original script but can be overridden if needed.
    """

    saveDir = os.path.join(base_dir, expName)

    (filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo,
     filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr) = gen_filepaths(expName, saveDir)

    print(f"Starting Session: {expName}...")

    run_session(filepath_m, filepath_go, filepath_gr, filepath_w_grgo, filepath_w_gogo,
                filepath_w_mfgo, filepath_w_gogr, filepath_w_mfgr,
                numMF, numGO, numGR, CSon, CSoff, useCS, numBins, numTrial,
                MFGO_PLAST, GOGO_PLAST, GRGO_PLAST, MFGR_PLAST, GOGR_PLAST,
                saveDir, saveGORaster, saveGRRaster, saveMFRaster, saveWeights)

    print(f"Finished Session: {expName}.")


##### Experiment Loop #####
if __name__ == "__main__":
    # run_experiment(
    #         expName='75recip_MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial',
    #         numTrial=1000,
    #         useCS=0,
    #         MFGO_PLAST=0,
    #         GOGO_PLAST=1,
    #         GRGO_PLAST=0,
    #         MFGR_PLAST=0,
    #         GOGR_PLAST=0,
    #         saveGORaster=True,
    #         saveGRRaster=True,
    #         saveMFRaster=True,
    #         saveWeights=True,
    #     )


    # # Example: run the original experiment
    # run_experiment(
    #     expName='75recip_MFGoGr_SS_shuffleMF10percent_noCS_mfgoplast_1000_trial',
    #     numTrial=1000,
    #     useCS=0,
    #     MFGO_PLAST=1,
    #     GOGO_PLAST=0,
    #     GRGO_PLAST=0,
    #     MFGR_PLAST=0,
    #     GOGR_PLAST=0,
    #     saveGORaster=True,
    #     saveGRRaster=True,
    #     saveMFRaster=True,
    #     saveWeights=True,
    # )

    
    # run_experiment(
    #     expName='75recip_MFGoGr_SS_shuffleMF10percent_noCS_mfgrplast_1000_trial',
    #     numTrial=1000,
    #     useCS=0,
    #     MFGO_PLAST=0,
    #     GOGO_PLAST=0,
    #     GRGO_PLAST=0,
    #     MFGR_PLAST=1,
    #     GOGR_PLAST=0,
    #     saveGORaster=True,
    #     saveGRRaster=True,
    #     saveMFRaster=True,
    #     saveWeights=True,
    # )

    run_experiment(
        expName='75recip_MFGoGr_SS_shuffleMF10percent_noCS_grgoplast_1000_trial',
        numTrial=1000,
        useCS=0,
        MFGO_PLAST=0,
        GOGO_PLAST=0,
        GRGO_PLAST=1,
        MFGR_PLAST=0,
        GOGR_PLAST=0,
        saveGORaster=True,
        saveGRRaster=True,
        saveMFRaster=True,
        saveWeights=True,   
    )

    # run_experiment(
    #     expName='75recip_MFGoGr_SS_shuffleMF10percent_noCS_allGOplast_1000_trial',
    #     numTrial=1000,
    #     useCS=0,
    #     MFGO_PLAST=1,
    #     GOGO_PLAST=1,
    #     GRGO_PLAST=1,
    #     MFGR_PLAST=0,
    #     GOGR_PLAST=0,
    #     saveGORaster=True,
    #     saveGRRaster=True,
    #     saveMFRaster=True,
    #     saveWeights=True,
    # )

    # run_experiment(
    #     expName='75recip_MFGoGr_SS_shuffleMF10percent_noCS_allGRplast_1000_trial',
    #     numTrial=1000,
    #     useCS=0,
    #     MFGO_PLAST=0,
    #     GOGO_PLAST=0,
    #     GRGO_PLAST=0,
    #     MFGR_PLAST=1,
    #     GOGR_PLAST=1,
    #     saveGORaster=True,
    #     saveGRRaster=True,
    #     saveMFRaster=True,
    #     saveWeights=True,
    # )