import numpy as np
import cupy as cp
from cupy import ElementwiseKernel
import math
import matplotlib.pyplot as plt
import time

class Mossy(): # MF Objects, entire network of MF per object
    def __init__(self,n,CSon, CSoff):
        self.numMossy = n # number of MFs
        self.CSon, self.CSoff = CSon, CSoff # time of CS on and off
        self.minBackground, self.maxBackground = 1, 30 # background frequency range
        self.minCS, self.maxCS = 90, 100 # CS frequency range

        self.act = np.zeros(self.numMossy, dtype = np.uint8) # activity of MFs, indicated by 0/1
        self.sizeOfDist = 1000 # size of the ISI distribution
        self.MFisiDistribution = np.zeros((101, self.sizeOfDist), dtype = int) # 2D array, with 101 rows of arrays that consist of 1000 zeroes , each row corresponds to ISI of particular frequency, where we select the new ISI distribution from
        self.MFisi = np.random.randint(5, 40, self.numMossy) # numMossy amounts of random integers selected from the range 5 - 40, the initial ISI of each MF, randomly selected from 5 to 40 time steps

        for i in range(0, 101): # for each integer from 0 to 100 <-- for gnenerating frequencies?
            if i == 0 : f = 1000 # f = firing frequency
            else: f = 1000.0/(i) 
            f_stdev = f/5.0  # why 5.0?
            # cp.random.normal(loc, scale, size) <-- generate random numbers from a normal distribution, characterized by mean, stdev, and last param = number of random numbers to generate
            ISItemp = np.random.normal(loc = f, scale = f_stdev, size = self.sizeOfDist) # create an array of random integers selected from normal distribution of ISI values with mean f and standard deviation f_stdev, these will be our "countdown" ISI values
            # check for values less than 5 and replace with 5
            for j in range(0, self.sizeOfDist):
                if ISItemp[j] < 5: ISItemp[j] = 5
            # fill temp normal values for row i of MFisiDistribution <-- generates the ISI distribution for each frequency
            self.MFisiDistribution[i, :] = ISItemp

        # Set Frequency index selection arrays <-- just curious, why not use normal distribution for the MF freq? is it completely uniform
        self.MFfreqs = np.random.randint(self.minBackground, self.maxBackground, self.numMossy)
        self.CSfreqs = np.random.randint(self.minCS, self.maxCS, self.numMossy)
        # choose CS MF
        self.CSMFindex = np.random.randint(0, self.numMossy, 80) # choose 80 random MFs to be CS MFs

    def do_MF_dist(self, timestep, useCS):
        self.act.fill(0) # reset activity of MFs to 0 for new timestep
        isi_mask = (self.MFisi == 0) # boolean array indicating which MF fired, true where MFisi is 0, false otherwise
        self.act = isi_mask.astype(np.uint8) # convert boolean array to int array, true = 1, false = 0
        if useCS == 1:
            # get random indices the size of all spiked cells
            random_idx = np.random.randint(0, self.sizeOfDist - 1, size=int(np.sum(isi_mask).item())) # array of random integers from the range of sizeOfDist - 1, For each spiked cell, get a random index to select new ISI from distribution
            # for starting and ending artifacts
            random_CSMF_idx = np.random.randint(0, self.sizeOfDist - 1, size = int(len(self.CSMFindex))) # For each CS MF (in CS, the CSMF definitely spiked, so there's no "checking"), get a random index to select new ISI from distribution
            # only need to worry about CS MF if in CS
            if (timestep >= self.CSon) and (timestep < self.CSoff): # if in CS period
                # create boolean array for CS cells
                is_CSMF = np.zeros(self.numMossy, dtype = bool)
                is_CSMF[self.CSMFindex] = True # set the CS MF indices to true
                # find cells that meet all conditions <-- what are theses conditions?
                spiking_cs_cell = isi_mask & is_CSMF # boolean arrays, which MF fired and are CSMF
                # don't need to run if no cells spike

                # **ASK CARTER
                if spiking_cs_cell.any(): # if there are any spiking CS cells that are also part of the ones that spiked
                    # pulls out actual index values
                    cs_idx = np.where(spiking_cs_cell)[0] # get the indices of the MF that fired and are CSMF
                    # store frequencies for the indexes
                    cs_freqs = self.CSfreqs[cs_idx] # get the preset firing frequencies of the CSMF that fired
                    # up to len cs idx get these rand idx, non cs indx get the rest
                    cs_rand_idx = random_idx[:len(cs_idx)] # get the random indices for the CSMF that fired
                    # select new isi
                    self.MFisi[cs_idx] = self.MFisiDistribution[cs_freqs, cs_rand_idx] # update the ISI values of the CSMF that fired with the new ISI values from the distribution
                #**
                spiking_non_cs_cell = isi_mask & ~is_CSMF # boolean arrays, which MF fired and are not CSMF
                if spiking_non_cs_cell.any(): # if there are any spiking non-CS cells that are also part of the ones that spiked
                    non_cs_idx = np.where(spiking_non_cs_cell)[0]
                    non_cs_freq = self.MFfreqs[non_cs_idx]
                    non_cs_rand_idx = random_idx[-len(non_cs_idx):] # get the random indices for the non-CS MFs that fired
                    self.MFisi[non_cs_idx] = self.MFisiDistribution[non_cs_freq, non_cs_rand_idx] # update the ISI values of the non-CS MFs that fired with the new ISI values from the distribution
                if timestep == self.CSon: # dealing with starting CS artifact
                    temp_isi = self.MFisiDistribution[self.CSfreqs[self.CSMFindex], random_CSMF_idx] # get new ISI values from distribution for the CSMF
                    self.MFisi[self.CSMFindex] = np.minimum(temp_isi, self.MFisi[self.CSMFindex]) # update the ISI values of the CSMF with the new ISI values, taking the minimum of the current ISI and the new ISI
                    # why take the minimum?
            else:
                freq_idx = self.MFfreqs[isi_mask] # get the frequencies of the MFs that fired
                new_isi = self.MFisiDistribution[freq_idx, random_idx] # get new ISI values from distribution for the MFs that fired
                self.MFisi[isi_mask] = new_isi
                if timestep == self.CSoff: # dealing with ending CS artifact
                    temp_isi = self.MFisiDistribution[self.CSfreqs[self.CSMFindex], random_CSMF_idx] # get new ISI values from distribution for the CSMF
                    min_isi = np.minimum(temp_isi, self.MFisi[self.CSMFindex]) # take the minimum of the current ISI and the new ISI
                    max_isi = np.maximum(temp_isi, self.MFisi[self.CSMFindex]) # take the maximum of the current ISI and the new ISI
                    # get new isi inbetween temp and current ISI <-- why do this for the ending artifact?
                    self.MFisi[self.CSMFindex] = np.random.randint(min_isi, max_isi + 1, (int(len(self.CSMFindex)),)) # update the ISI values of the CSMF with a random integer between the min and max ISI values        
        else: # No CS, if MF fired, then get new ISI from background distribution
            # only need the MFfreqs where firing = true
            freq_idx = self.MFfreqs[isi_mask] # get the frequencies of the MFs that fired
           # for selecting MF isi's from distribution
            random_idx = np.random.randint(0, self.sizeOfDist - 1, size=int(np.sum(isi_mask).item()))
           # generating new isi's
            new_isi = self.MFisiDistribution[freq_idx, random_idx] # get new ISI values from distribution for the MFs that fired
            # place into main MFisi array
            self.MFisi[isi_mask] = new_isi # update the ISI values of the MFs that fired with the new ISI values
            #decrement ISI for all firing = false
        self.MFisi[~isi_mask] -= 1 # decrement the ISI values of the MFs that did not fire by 1
        return self.act # return the activity of the MFs for this timestep
    def get_act(self):
        return self.act
            
class Golgi(): # class of Golgi cells, entire network of Golgi cells
    def __init__(self, n, csON, csOFF, useCS, trialSize, gogo_weight = 0.0125, mfgo_weight = 0.0042, grgo_weight = 0.0007):
        ### Constants
        self.numGolgi = n
        self.useCS = useCS
        self.csOn = csON
        self.csOff = csOFF
        self.gLeak = 0.02 #
        self.eLeak = -70.0 #
        self.eGABA = -64.0 #
        self.thresholdMax = 10.0 #
        
        
        self.gogoW = gogo_weight # weight of Golgi to Golgi synapse
        self.gGABA_decayGOGO = math.exp(-1.0/10.0) # (-msPerTimestep / gGABADecTauGOtoGO), decay constant for GABA conductance from Golgi to Golgi
        self.mfgoW = mfgo_weight # weight of MF to Golgi synapse
        self.g_decayMFGO = math.exp(-1.0/3.0) # (-msPerTimestep / gDecayTauMFtoGO), decay constant for excitatory conductance from MF to Golgi
        self.grgoW = grgo_weight # weight of granule to Golgi synapse
        self.g_decayGRGO = math.exp(-1.0/4.0) # (-msPerTimestep / gDecayTauGRtoGO), decay constant for excitatory conductance from granule to Golgi
        self.g_decay_NMDA_MFGO = math.exp(-1.0/30.0) # (-msPerTimestep / decayTauNMDA_MFtoGO), decay constant for NMDA conductance from MF to Golgi
        self.NMDA_AMPA_ratioMFGO = 1.3 # ratio of NMDA to AMPA conductance for MF to Golgi synapse

        self.threshDecGo = 1 - math.exp(-1.0/11.0) # (-msPerTimestep / threshDecTauGO), decay constant for Golgi threshold
        self.threshRest = -34.0 # resting threshold for Golgi cells

        ### Arrays
        self.Vm = np.full(self.numGolgi, self.eLeak, dtype = np.float32) # membrane potential of Golgi cells, initialized to leak reversal potential
        self.gSum_GOGO = np.zeros(self.numGolgi, dtype = np.float32) # sum of GABA conductances from Golgi to Golgi
        self.gSum_MFGO = np.zeros(self.numGolgi, dtype = np.float32) # sum of excitatory conductances from MF to Golgi
        self.gSum_GRGO = np.zeros(self.numGolgi, dtype = np.float32) # sum of excitatory conductances from granule to Golgi
        self.gNMDA_inc_MFGO = np.zeros(self.numGolgi, dtype = np.float32) # NMDA conductance increment from MF to Golgi
        self.gNMDA_MFGO = np.zeros(self.numGolgi, dtype = np.float32) # NMDA conductance from MF to Golgi

        # Input Arrays GPU only
        self.GPU_inputGOGO = cp.zeros(self.numGolgi, dtype = cp.uint8) # input GABA conductance from Golgi to Golgi
        self.GPU_inputMFGO = cp.zeros(self.numGolgi, dtype = cp.uint8) # input excitatory conductance from MF to Golgi
        self.GPU_inputGRGO = cp.zeros(self.numGolgi, dtype = cp.uint8) # input excitatory conductance from granule to Golgi

        # Threshold
        self.currentThresh = np.full(self.numGolgi, self.threshRest, dtype = np.float32) # current threshold of Golgi cells, initialized to resting threshold
        self.act = np.zeros((trialSize, self.numGolgi), dtype = np.uint8) # activity of Golgi cells over the entire trial, 2D array of size (trialSize, numGolgi)
        
        # Kernel Stuff
        self.block_size = 256
        self.grid_size = (self.numGolgi + self.block_size - 1) // self.block_size

        doGolgiKernel_code = """
        extern "C" __global__ void doGolgi(int size, float *GPU_gNMDA_inc_MFGO, float *GPU_Vm, 
        float *GPU_gSum_GOGO, unsigned char *GPU_inputGOGO, float GPU_gogoW, float GPU_gGABA_decayGOGO,
        float *GPU_gSum_MFGO, unsigned char *GPU_inputMFGO, float GPU_mfgoW, float GPU_g_decayMFGO,
        float *GPU_gSum_GRGO, unsigned char *GPU_inputGRGO, float GPU_grgoW, float GPU_g_decayGRGO,
        float *GPU_gNMDA_MFGO, float GPU_NMDA_AMPA_ratioMFGO, float GPU_g_decay_NMDA_MFGO, float *GPU_currentThresh,
        float GPU_threshRest, float GPU_threshDecGo, float GPU_gLeak, float GPU_eLeak, unsigned char *GPU_spike_mask)
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= size) return;
            GPU_gNMDA_inc_MFGO[idx] = (
                (0.00000011969 * GPU_Vm[idx] * GPU_Vm[idx] * GPU_Vm[idx]) +
                (0.000089369 * GPU_Vm[idx] * GPU_Vm[idx]) +
                (0.0151 * GPU_Vm[idx]) + 0.7713
            );

            GPU_gSum_GOGO[idx] = GPU_inputGOGO[idx] * GPU_gogoW + GPU_gSum_GOGO[idx] * GPU_gGABA_decayGOGO;

            GPU_gSum_MFGO[idx] = GPU_inputMFGO[idx] * GPU_mfgoW + GPU_gSum_MFGO[idx] * GPU_g_decayMFGO;

            GPU_gSum_GRGO[idx] = GPU_inputGRGO[idx] * GPU_grgoW + GPU_gSum_GRGO[idx] * GPU_g_decayGRGO;

            GPU_gNMDA_MFGO[idx] = (
                GPU_inputMFGO[idx] * (GPU_mfgoW * GPU_NMDA_AMPA_ratioMFGO * GPU_gNMDA_inc_MFGO[idx]) + 
                GPU_gNMDA_MFGO[idx] * GPU_g_decay_NMDA_MFGO
            );

            GPU_currentThresh[idx] = GPU_currentThresh[idx] + (GPU_threshRest - GPU_currentThresh[idx]) * (GPU_threshDecGO);

            GPU_spike_mask = GPU_Vm > GPU_currentThresh

        """
        update_input_kernel_code = """
        extern "C" __global__ void updateInputActivity(
            unsigned char *act,           // activity vector of golgi cells (0/1)
            const int *connArr,     // flattened 2D array: numGolgi x numInputsPerCell       
            unsigned char *inputArr,             // output sum for each Golgi
            int numGolgi,
            int numInputsPerCell
        ) {

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= numGolgi) return;

            int sum = 0;
            for (int j = 0; j < numInputsPerCell; j++) {
                int inputIndex = connArr[idx * numInputsPerCell + j]; // get connected golgi index
                sum += act[inputIndex];                             // add 0 or 1
            }

            inputArr[idx] = sum;

        }
        """

        self.GOGPU = cp.RawKernel(doGolgiKernel_code, 'doGolgi')
        self.updateGOgpu = cp.RawKernel(update_input_kernel_code, 'updateInputActivity')
        self.GPU_gNMDA_inc_MFGO = cp.array(self.gNMDA_inc_MFGO, dtype = cp.float32)
        self.GPU_Vm = cp.array(self.Vm, dtype = cp.float32)
        self.GPU_gSum_GOGO = cp.array(self.gSum_GOGO, dtype = cp.float32)
        self.GPU_gogoW = cp.float32(self.gogoW)
        self.GPU_gGABA_decayGOGO = cp.float32(self.gGABA_decayGOGO)
        self.GPU_gSum_MFGO = cp.array(self.gSum_MFGO, dtype = cp.float32)
        self.GPU_mfgoW = cp.float32(self.mfgoW)
        self.GPU_g_decayMFGO = cp.float32(self.g_decayMFGO)
        self.GPU_gSum_GRGO = cp.array(self.gSum_GRGO, dtype = cp.float32)
        self.GPU_grgoW = cp.float32(self.grgoW)
        self.GPU_g_decayGRGO = cp.float32(self.g_decayGRGO)
        self.GPU_gNMDA_MFGO = cp.array(self.gNMDA_MFGO, dtype = cp.float32)
        self.GPU_NMDA_AMPA_ratioMFGO = cp.float32(self.NMDA_AMPA_ratioMFGO)
        self.GPU_g_decay_NMDA_MFGO = cp.float32(self.g_decay_NMDA_MFGO)
        self.GPU_currentThresh = cp.array(self.currentThresh, dtype = cp.float32)
        self.GPU_threshRest = cp.float32(self.threshRest)
        self.GPU_threshDecGo = cp.float32(self.threshDecGo)
        self.GPU_gLeak = cp.float32(self.gLeak)
        self.GPU_eLeak = cp.float32(self.eLeak)
        self.GPU_spike_mask = cp.zeros(self.numGolgi, dtype = cp.uint8)


    def do_Golgi(self, t):
        with cp.cuda.Device(0):
            self.GOgpu((self.grid_size,),(self.block_size,), (self.numGolgi, self.GPU_gNMDA_inc_MFGO, self.GPU_Vm, 
            self.GPU_gSum_GOGO, self.GPU_inputGOGO, self.GPU_gogoW, self.GPU_gGABA_decayGOGO, self.GPU_gSum_MFGO, self.GPU_inputMFGO, 
            self.GPU_mfgoW, self.GPU_g_decayMFGO, self.GPU_gSum_GRGO, self.GPU_inputGRGO, self.GPU_grgoW, self.GPU_g_decayGRGO, 
            self.GPU_gNMDA_MFGO, self.GPU_NMDA_AMPA_ratioMFGO, self.GPU_g_decay_NMDA_MFGO, self.GPU_currentThresh, self.GPU_threshRest, 
            self.GPU_threshDecGo, self.GPU_gLeak, self.GPU_eLeak, self.GPU_spike_mask))
            self.act[t] = self.GPU_spike_mask.get()
            self.GPU_inputMFGO.fill(0)
            self.GPU_inputGOGO.fill(0)
            self.GPU_inputGRGO.fill(0)
        
    # generic function for updating GOGO and MFGO input arrays
    # def update_input_activity(self, connectArr, inputArrayChoice, mfAct = None, t = None, grAct = None):
    def update_input_activity(self, connectArr, inputArrayChoice, activity):
        ''' inputArrayChoice selects between 1 = MFGO, 2 = GOGO, 3 = GRGO'''
        activity = cp.array(activity)
        # MFGO
        if inputArrayChoice == 1:
            with cp.cuda.Device(0):
                self.updateGOgpu((self.grid_size,), (self.block_size,), (activity, connectArr, self.GPU_inputMFGO, self.numGolgi, 20))
        # GOGO
        elif inputArrayChoice == 2:
            with cp.cuda.Device(0):
                self.updateGOgpu((self.grid_size,), (self.block_size,), (activity, connectArr, self.GPU_inputGOGO, self.numGolgi, 12))
        elif inputArrayChoice == 3:
            with cp.cuda.Device(0):
                self.updateGOgpu((self.grid_size,), (self.block_size,), (activity, connectArr, self.GPU_inputGRGO, self.numGolgi, 50))

    def updateFinalState(self):
        with cp.cuda.Device(0):
            self.Vm = cp.asnumpy(self.GPU_Vm)
            self.gNMDA_inc_MFGO = cp.asnumpy(self.GPU_gNMDA_inc_MFGO)
            self.gSum_GOGO = cp.asnumpy(self.GPU_gSum_GOGO)
            self.gSum_MFGO = cp.asnumpy(self.GPU_gSum_MFGO)
            self.gSum_GRGO = cp.asnumpy(self.GPU_gSum_GRGO)
            self.gNMDA_MFGO = cp.asnumpy(self.GPU_gNMDA_MFGO)
            self.currentThresh = cp.asnumpy(self.GPU_currentThresh)
    
    def get_act(self):
        return self.act
    
    def get_gGOGO(self):
        return self.gSum_GOGO


class Granule():
    def __init__(self, n, csOFF, csON, useCS, trialSize, mfgr_weight = 0.0042, gogr_weight = 0.015):
        ### Constants
        self.numGranule = n
        self.csOFF, self.csON = csOFF, csON
        self.useCS = useCS # whether to use CS or not
        self.eLeak = -65.0
        self.eGOGR = -75.0
        self.thresholdMax = 10.0 # maximum Vm threshold
        self.gLeak_base = 0.1  # leak conductance
        self.g_decay_NMDA_MFGR = math.exp(-1.0/30.0) # 0.9672
        self.gDirectInc_MFGR = 0.0320

        self.g_decay_MFGR = 0.9355 # math.exp(-1.0/15.0), which compiles as 0 in C++? # !! FIX THIS I'm confused bc in the big sim it's 0.0 but this isn't mathematically possible???  (-msPerTimestep / gDecayTauMFtoGR), decay constant for excitatory conductance from MF to Granule
        self.gogrW = gogr_weight
        self.mfgrW = mfgr_weight
        self.gGABA_decayGOGR = math.exp(-1.0/7.0) # (-msPerTimestep / gGABADecTauGOtoGR), decay constant for GABA conductance from Golgi to Granule

        self.threshDecGR = 1 - math.exp(-1.0/3.0) 
        self.threshRest = -40.0

        ### Arrays
        # self.mfgrW = cp.full(self.numGranule, mfgr_weight, dtype = float) # synaptic weight of mossy fiber to granule 
        self.Vm = np.full(self.numGranule, self.eLeak, dtype = np.float32) # membrane potential of Granule cells, initialized to leak reversal potential
        self.gSum_MFGR = np.zeros(self.numGranule, dtype = np.float32) # sum of excitatory conductances from MF to Granule
        self.gSum_GOGR = np.zeros(self.numGranule, dtype = np.float32) # sum of GABA conductances from Golgi to Granule
        self.gNMDA_MFGR= np.zeros(self.numGranule, dtype = np.float32) # NMDA conductance from MF to Granule
        self.gNMDA_Inc_MFGR = np.zeros(self.numGranule, dtype = np.float32) # NMDA conductance increment from MF to Granule
        self.gLeak = np.full(self.numGranule, self.gLeak_base, dtype = np.float32) # leak conductance for Granule cells, initialized to leak conductance
        self.gKCa = np.zeros(self.numGranule, dtype = np.float32) # experimental K and Ca conductance, initialized to 0
        # Input Arrays GPU
        self.GPU_inputMFGR = cp.zeros(self.numGranule, dtype = cp.uint8) # input excit
        self.GPU_inputGOGR = cp.zeros(self.numGranule, dtype = cp.uint8) # input GABA conductance from Golgi to Granule

        # Threshold
        self.currentThresh = np.full(self.numGranule, self.threshRest, dtype = np.float32) # current threshold of Granule cells, initialized to resting threshold
        self.act = np.zeros((trialSize, self.numGranule), dtype = np.uint8) # activity of Granule cells over the entire trial, 2D array of size
        
        # Kernel stuff
        self.block_size = 256
        self.grid_size = (self.numGranule + self.block_size - 1) // self.block_size

        doGranulekernel_code = """
        extern "C" __global__ void doGranule(int size, float *GPU_gLeak, float *GPU_Vm, unsigned char *inputMFGR, float *GPU_gSum_MFGR,
        unsigned char *inputGOGR, float *GPU_gSum_GOGR, float *GPU_gNMDA_Inc_MFGR, float *GPU_gNMDA_MFGR, float *GPU_currentThresh, float GPU_mfgrW, 
        float GPU_g_decay_MFGR, float GPU_gogrW, float GPU_gGABA_decayGOGR, float GPU_g_decay_NMDA_MFGR, float GPU_gDirectInc_MFGR,
        float GPU_threshRest, float GPU_threshDecGR, float GPU_eLeak, float GPU_eGOGR, unsigned char *GPU_spike_mask){
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= size) return;
            GPU_gLeak[idx] = (0.0000001021370733 * GPU_Vm[idx] * GPU_Vm[idx] * GPU_Vm[idx] * GPU_Vm[idx]) + 
            (0.00001636462 * GPU_Vm[idx] * GPU_Vm[idx] * GPU_Vm[idx]) +
            (0.00113971219 * GPU_Vm[idx] * GPU_Vm[idx] + 0.038772 * GPU_Vm[idx] + 0.6234929);

            GPU_gNMDA_Inc_MFGR[idx] = (
            (0.00000011969 * GPU_Vm[idx] * GPU_Vm[idx] * GPU_Vm[idx]) + 
            (0.000089369 * GPU_Vm[idx] * GPU_Vm[idx]) +
            (0.0151 * GPU_Vm[idx]) + 0.7713
            );

            GPU_gSum_MFGR[idx] = inputMFGR[idx] * GPU_mfgrW + GPU_gSum_MFGR[idx] * GPU_g_decay_MFGR; 
            
            GPU_gSum_GOGR[idx] = inputGOGR[idx] * GPU_gogrW + GPU_gSum_GOGR[idx] * GPU_gGABA_decayGOGR;

            GPU_gNMDA_MFGR[idx] = GPU_gNMDA_Inc_MFGR[idx] * GPU_gDirectInc_MFGR * inputMFGR[idx] + GPU_gNMDA_MFGR[idx] * 0.9672;

            GPU_Vm[idx] = GPU_Vm[idx] + GPU_gLeak[idx] * (GPU_eLeak - GPU_Vm[idx]) - GPU_gSum_MFGR[idx] * GPU_Vm[idx] - 
            GPU_gNMDA_MFGR[idx] * GPU_Vm[idx] + GPU_gSum_GOGR[idx] * (GPU_eGOGR - GPU_Vm[idx]);

            GPU_currentThresh[idx] = GPU_currentThresh[idx] + (GPU_threshRest - GPU_currentThresh[idx]) * (GPU_threshDecGR);

            GPU_spike_mask[idx] = (GPU_Vm[idx] > GPU_currentThresh[idx]) ? 1 : 0;
        }
        """

        update_input_kernel_code = """
        extern "C" __global__ void updateInputActivity(
            unsigned char *act,           // activity vector of golgi cells (0/1)
            const int *connArr,     // flattened 2D array: numGolgi x numInputsPerCell       
            unsigned char *inputArr,             // output sum for each Golgi
            int numGranule,
            int numInputsPerCell
        ) {

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= numGranule) return;

            unsigned char sum = 0;
            for (int j = 0; j < numInputsPerCell; j++) {
                int inputIndex = connArr[idx * numInputsPerCell + j]; // get connected golgi index
                sum += act[inputIndex];                             // add 0 or 1
            }

            inputArr[idx] = sum;

        }
        """

        self.GRgpu = cp.RawKernel(doGranulekernel_code, 'doGranule')
        self.updateGRgpu = cp.RawKernel(update_input_kernel_code, 'updateInputActivity')
        self.GPU_gLeak = cp.array(self.gLeak, dtype = cp.float32)
        self.GPU_Vm = cp.array(self.Vm, dtype = cp.float32)
        self.GPU_gSum_MFGR = cp.array(self.gSum_MFGR, dtype = cp.float32)
        self.GPU_gSum_GOGR = cp.array(self.gSum_GOGR, dtype = cp.float32)
        self.GPU_gNMDA_Inc_MFGR = cp.array(self.gNMDA_Inc_MFGR, dtype = cp.float32)
        self.GPU_gNMDA_MFGR = cp.array(self.gNMDA_MFGR, dtype = cp.float32)
        self.GPU_currentThresh = cp.array(self.currentThresh, dtype = cp.float32)
        self.GPU_mfgrW = cp.float32(self.mfgrW)
        self.GPU_g_decay_MFGR = cp.float32(self.g_decay_MFGR)
        self.GPU_gogrW  = cp.float32(self.gogrW)
        self.GPU_gGABA_decayGOGR = cp.float32(self.gGABA_decayGOGR)
        self.GPU_g_decay_NMDA_MFGR = cp.float32(self.g_decay_NMDA_MFGR)
        self.GPU_gDirectInc_MFGR = cp.float32(self.gDirectInc_MFGR)
        self.GPU_threshRest = cp.float32(self.threshRest)
        self.GPU_threshDecGR = cp.float32(self.threshDecGR)
        self.GPU_eLeak = cp.float32(self.eLeak)
        self.GPU_eGOGR = cp.float32(self.eGOGR)
        self.GPU_spike_mask = cp.zeros(self.numGranule, dtype = cp.uint8)

    def do_Granule(self, t):
        with cp.cuda.Device(0):
            self.GRgpu((self.grid_size,),(self.block_size,), (self.numGranule, self.GPU_gLeak, self.GPU_Vm, self.GPU_inputMFGR, self.GPU_gSum_MFGR, self.GPU_inputGOGR, 
            self.GPU_gSum_GOGR, self.GPU_gNMDA_Inc_MFGR, self.GPU_gNMDA_MFGR, self.GPU_currentThresh, self.GPU_mfgrW, self.GPU_g_decay_MFGR, self.GPU_gogrW, 
            self.GPU_gGABA_decayGOGR, self.GPU_g_decay_NMDA_MFGR, self.GPU_gDirectInc_MFGR, self.GPU_threshRest, self.GPU_threshDecGR, self.GPU_eLeak, 
            self.GPU_eGOGR, self.GPU_spike_mask))
            self.act[t] = self.GPU_spike_mask.get()
            self.GPU_inputMFGR.fill(0)
            self.GPU_inputGOGR.fill(0)

    # generic function for updating GOGR and MFGR input arrays
    def update_input_activity(self, connectArr, inputArrayChoice, activity):
        ''' inputArrayChoice selects between 1 = MFGR, 2 = GOGR '''
        activity = cp.array(activity)
        # MFGR
        if inputArrayChoice == 1:
            with cp.cuda.Device(0):
                self.updateGRgpu((self.grid_size,), (self.block_size,), (activity, connectArr, self.GPU_inputMFGR, self.numGranule, 4000))
        # GOGR
        elif inputArrayChoice == 2:
            with cp.cuda.Device(0):
                self.updateGRgpu((self.grid_size,), (self.block_size,), (activity, connectArr, self.GPU_inputGOGR, self.numGranule, 12800))

    def get_act(self):
        return self.act

    def updateFinalState(self):
        with cp.cuda.Device(0):
            self.Vm = cp.asnumpy(self.GPU_Vm)
            self.gSum_MFGR = cp.asnumpy(self.GPU_gSum_MFGR)
            self.gSum_GOGR = cp.asnumpy(self.GPU_gSum_GOGR)
            self.gNMDA_Inc_MFGR = cp.asnumpy(self.GPU_gNMDA_Inc_MFGR)
            self.gNMDA_MFGR = cp.asnumpy(self.GPU_gNMDA_MFGR)
            self.currentThresh = cp.asnumpy(self.GPU_currentThresh)
    