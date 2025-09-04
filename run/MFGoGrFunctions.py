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
    def __init__(self, n, csON, csOFF, useCS, trialSize, mfgo_plast = 0, gogo_plast = 0, grgo_plast = 0, gogo_weight = 0.0125, mfgo_weight = 0.0042, grgo_weight = 0.0007):
        ### Constants
        self.numGolgi = n
        self.useCS = useCS
        self.csOn = csON
        self.csOff = csOFF
        self.gLeak = 0.02 #
        self.eLeak = -70.0 #
        self.eGABA = -64.0 #
        self.thresholdMax = 10.0 #
        self.gGABA_decayGOGO = math.exp(-1.0/10.0) # (-msPerTimestep / gGABADecTauGOtoGO), decay constant for GABA conductance from Golgi to Golgi
        self.g_decayMFGO = math.exp(-1.0/3.0) # (-msPerTimestep / gDecayTauMFtoGO), decay constant for excitatory conductance from MF to Golgi
        self.g_decayGRGO = math.exp(-1.0/4.0) # (-msPerTimestep / gDecayTauGRtoGO), decay constant for excitatory conductance from granule to Golgi
        self.g_decay_NMDA_MFGO = math.exp(-1.0/30.0) # (-msPerTimestep / decayTauNMDA_MFtoGO), decay constant for NMDA conductance from MF to Golgi
        self.NMDA_AMPA_ratioMFGO = 1.3 # ratio of NMDA to AMPA conductance for MF to Golgi synapse

        self.threshDecGo = 1- math.exp(-1.0/11.0) # (-msPerTimestep / threshDecTauGO), decay constant for Golgi threshold
        self.threshRest = -34.0 # resting threshold for Golgi cells

        ##### Plasticity
        self.mfgo_plast = mfgo_plast
        self.gogo_plast = gogo_plast
        self.grgo_plast = grgo_plast
        self.plast_ratio = 1/200 # LTP / LTD, 5 Hz
        self.mfgo_LTD_inc = 1/1000 * mfgo_weight * -1
        self.mfgo_LTP_inc = self.plast_ratio * self.mfgo_LTD_inc # negative due to computation
        self.gogo_LTD_inc = 1/1000 * gogo_weight * -1
        self.gogo_LTP_inc = self.plast_ratio * self.gogo_LTD_inc 
        self.grgo_LTD_inc = 1/1000 * grgo_weight * -1
        self.grgo_LTP_inc = self.plast_ratio * self.grgo_LTD_inc 
        self.mf_input = 0 
        self.gr_input = 0
        self.go_input = 0

        ### Arrays
        self.grgoW = np.full(self.numGolgi, grgo_weight, dtype = np.float32) # array of synaptic weight
        self.mfgoW = np.full(self.numGolgi, mfgo_weight, dtype = np.float32) # array of synaptic weight
        self.gogoW = np.full(self.numGolgi, gogo_weight, dtype = np.float32) # array of synaptic weight
        self.Vm = np.full(self.numGolgi, self.eLeak, dtype = np.float32) # membrane potential of Golgi cells, initialized to leak reversal potential
        self.gSum_GOGO = np.zeros(self.numGolgi, dtype = np.float32) # sum of GABA conductances from Golgi to Golgi
        self.inputGOGO = np.zeros(self.numGolgi, dtype = np.uint8) # input GABA conductance from Golgi to Golgi
        self.gSum_MFGO = np.zeros(self.numGolgi, dtype = np.float32) # sum of excitatory conductances from MF to Golgi
        self.inputMFGO = np.zeros(self.numGolgi, dtype = np.uint8) # input excitatory conductance from MF to Golgi
        self.gSum_GRGO = np.zeros(self.numGolgi, dtype = np.float32) # sum of excitatory conductances from granule to Golgi
        self.inputGRGO = np.zeros(self.numGolgi, dtype = np.uint8) # input excitatory conductance from granule to Golgi
        self.gNMDA_inc_MFGO = np.zeros(self.numGolgi, dtype = np.float32) # NMDA conductance increment from MF to Golgi
        self.gNMDA_MFGO = np.zeros(self.numGolgi, dtype = np.float32) # NMDA conductance from MF to Golgi
        
        # Threshold
        self.currentThresh = np.full(self.numGolgi, self.threshRest, dtype = np.float32) # current threshold of Golgi cells, initialized to resting threshold
        self.act = np.zeros((trialSize, self.numGolgi), dtype = np.uint8) # activity of Golgi cells over the entire trial, 2D array of size (trialSize, numGolgi)
        # NTS: Properties of gr that are different from Go? think abt it

    def do_Golgi(self, t):
        # Vectorized
        # NMDA High (voltage-dep step size for nMDa conductance update mf -> go)
        self.gNMDA_inc_MFGO = (
            (0.00000011969 * self.Vm * self.Vm * self.Vm) +
            (0.000089369 * self.Vm * self.Vm) +
            (0.0151 * self.Vm) + 0.7713
        ) # calculated NMDA increment based on current Vm, vectorized for all Golgi cells
        # update total go -> go input conductance
        self.gSum_GOGO = (self.inputGOGO * self.gogoW) + self.gSum_GOGO * self.gGABA_decayGOGO # decay previous conductance and add new input conductance scaled by weight
        # update total mf -> go input conductance
        self.gSum_MFGO = (self.inputMFGO * self.mfgoW) + self.gSum_MFGO * self.g_decayMFGO # decay previous conductance and add new input conductance scaled by weight
        # update total gr -> go input conductance
        self.gSum_GRGO = (self.inputGRGO * self.grgoW) + self.gSum_GRGO * self.g_decayGRGO # decay previous conductance and add new input conductance scaled by weight
        # update NMDA mf -> go conductance
        self.gNMDA_MFGO = (
            self.inputMFGO * (self.mfgoW * self.NMDA_AMPA_ratioMFGO * self.gNMDA_inc_MFGO) +
            self.gNMDA_MFGO * self.g_decay_NMDA_MFGO
        ) # decay previous NMDA conductance and add new input conductance scaled by weight and NMDA increment
        # update current threshold
        self.currentThresh += (self.threshRest - self.currentThresh) * self.threshDecGo # decay current threshold towards resting threshold
        # calc new Vm
        self.Vm += (
            (self.gLeak * (self.eLeak - self.Vm)) +
            (self.gSum_GOGO * (self.eGABA - self.Vm)) -
            (self.gSum_MFGO + self.gSum_GRGO + self.gNMDA_MFGO) *
            self.Vm
        ) # update Vm based on conductances and reversal potentials

        # reset inputs
        self.inputMFGO.fill(0)
        self.inputGRGO.fill(0)
        self.inputGOGO.fill(0)


        ## Do spikes
        # Get minimum values between arrays
        self.Vm = np.minimum(self.Vm, self.thresholdMax)
        # Calculate spikse with boolean mask, fit to int array
        spike_mask = self.Vm > self.currentThresh # boolean array indicating which Golgi cells fired, true where Vm exceeds current threshold
        self.act[t] = spike_mask.astype(np.uint8) # convert boolean array to int array, true = 1, false = 0

        # plasticity
        self.grgoW += self.grgo_plast * (self.gr_input * ((self.act[t] * self.grgo_LTD_inc) + ((self.act[t] - 1) * self.grgo_LTP_inc)))
        self.mfgoW += self.mfgo_plast * (self.mf_input * ((self.act[t] * self.mfgo_LTD_inc) + ((self.act[t] - 1) * self.mfgo_LTP_inc)))
        self.gogoW += self.gogo_plast * (self.go_input * ((self.act[t] * self.gogo_LTD_inc) + ((self.act[t] - 1) * self.gogo_LTP_inc)))

        self.mf_input = 0 
        self.gr_input = 0
        self.go_input = 0

        # Update thresholds where spikes occurred (condition, value if true, if false)
        self.currentThresh = np.where(spike_mask, self.thresholdMax, self.currentThresh) # set current threshold to max where spikes occurred

    # generic function for updating GOGO and MFGO input arrays
    # def update_input_activity(self, connectArr, inputArrayChoice, mfAct = None, t = None, grAct = None):
    def update_input_activity(self, connectArr, inputArrayChoice, mfAct = None, t = None, grAct = None):
        ''' inputArrayChoice selects between 1 = MFGO, 2 = GOGO, 3 = GRGO'''
        ### Playground version, adapt to the main code

        # MFGO - faster, so kept
        if inputArrayChoice == 1:
            self.mf_input = 1
            goInputs = np.zeros(self.numGolgi, dtype = np.uint8)
            spiked_idx = np.where(mfAct)[0]
            for mf in spiked_idx:
                goInputs[connectArr[mf, :]] += 1
            self.inputMFGO = goInputs
            # mfAct[0] = 0 # hardcode to 0 for optimization
            # self.inputMFGO = np.sum(mfAct[connectArr], axis = 1)
        
        if inputArrayChoice == 2:
            self.go_input = 1
            goInputs = np.zeros(self.numGolgi, dtype = np.uint8)
            spiked_idx = np.where(self.act[t])[0]
            for go in spiked_idx:
                goInputs[connectArr[go, :]] += 1
            self.inputGOGO = goInputs
        # # GOGO
        # if inputArrayChoice == 2:
        #     spike_mask = (self.act[t] == 1)
        #     spiked_idx = np.where(spike_mask)[0]
        #     # get connections for each spiked cell
        #     spiked_connections = []
        #     for cell in spiked_idx:
        #         # eliminate -1 terminator
        #         valid_idx = connectArr[cell] != -1
        #         valid_conns = connectArr[cell, valid_idx]
        #         spiked_connections.extend(valid_conns)
        #     # use bincount to count occurrances of target idx
        #     spiked_connections = np.array(spiked_connections, dtype=int)
        #     # count occurances of each index
        #     counts = np.bincount(spiked_connections)
        #     # add to input array
        #     self.inputGOGO[:len(counts)] = counts
        # GRGO
        if inputArrayChoice == 3: 
            self.gr_input = 1
            goInputs = np.zeros(self.numGolgi, dtype = np.uint8)
            spiked_idx = np.where(grAct)[0]
            for gr in spiked_idx:
                goInputs[connectArr[gr,:]] += 1
            self.inputGRGO = goInputs
            # # Version 1 
            # grAct[0] = 0 # hardcode to 0 for optimization
            # self.inputGRGO = np.sum(grAct[connectArr], axis = 1)

    def get_act(self):
        return self.act
    
    def get_gGOGO(self):
        return self.gSum_GOGO


class Granule():
    def __init__(self, n, csOFF, csON, useCS, trialSize, mfgr_plast = 0, gogr_plast = 0, mfgr_weight = 0.0042, gogr_weight = 0.015):
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
        self.gGABA_decayGOGR = math.exp(-1.0/7.0) # (-msPerTimestep / gGABADecTauGOtoGR), decay constant for GABA conductance from Golgi to Granule

        self.threshDecGR = 1 - math.exp(-1.0/3.0) 
        self.threshRest = -40.0

        ##### Plasticity
        self.GPU_mfgr_plast = cp.uint8(mfgr_plast)
        self.GPU_gogr_plast = cp.uint8(gogr_plast)
        self.plast_ratio = 1/10000 # LTP / LTD, 0.1 Hz
        self.GPU_mfgr_LTD_inc = cp.float32(1/1000 * mfgr_weight * -1)
        self.GPU_mfgr_LTP_inc = cp.float32(self.plast_ratio * self.GPU_mfgr_LTD_inc) # negative due to computation
        self.GPU_gogr_LTD_inc = cp.float32(1/1000 * gogr_weight * -1)
        self.GPU_gogr_LTP_inc = cp.float32(self.plast_ratio * self.GPU_gogr_LTD_inc)
        self.GPU_mf_input = cp.uint8(0) 
        self.GPU_go_input = cp.uint8(0)


        ### Arrays
        self.mfgrW = np.full(self.numGranule, mfgr_weight, dtype = np.float32) # synaptic weight of mossy fiber to granule
        self.gogrW = np.full(self.numGranule, gogr_weight, dtype = np.float32) # synaptic weight of mossy fiber to granule 
        self.Vm = np.full(self.numGranule, self.eLeak, dtype = np.float32) # membrane potential of Granule cells, initialized to leak reversal potential
        self.gSum_MFGR = np.zeros(self.numGranule, dtype = np.float32) # sum of excitatory conductances from MF to Granule
        self.inputMFGR = np.zeros(self.numGranule, dtype = np.uint8) # input excit
        self.gSum_GOGR = np.zeros(self.numGranule, dtype = np.float32) # sum of GABA conductances from Golgi to Granule
        self.inputGOGR = np.zeros(self.numGranule, dtype = np.uint8) # input GABA conductance from Golgi to Granule
        self.gNMDA_MFGR= np.zeros(self.numGranule, dtype = np.float32) # NMDA conductance from MF to Granule
        self.gNMDA_Inc_MFGR = np.zeros(self.numGranule, dtype = np.float32) # NMDA conductance increment from MF to Granule
        self.gLeak = np.full(self.numGranule, self.gLeak_base, dtype = np.float32) # leak conductance for Granule cells, initialized to leak conductance
        self.gKCa = np.zeros(self.numGranule, dtype = np.float32) # experimental K and Ca conductance, initialized to 0
        # Threshold
        self.currentThresh = np.full(self.numGranule, self.threshRest, dtype = np.float32) # current threshold of Granule cells, initialized to resting threshold
        self.act = np.zeros((trialSize, self.numGranule), dtype = np.uint8) # activity of Granule cells over the entire trial, 2D array of size
        
        # Kernel stuff
        doGranulekernel_code = """
        extern "C" __global__ void doGranule(int size, float *GPU_gLeak, float *GPU_Vm, unsigned char *inputMFGR, float *GPU_gSum_MFGR,
        unsigned char *inputGOGR, float *GPU_gSum_GOGR, float *GPU_gNMDA_Inc_MFGR, float *GPU_gNMDA_MFGR, float *GPU_currentThresh, float *GPU_mfgrW, 
        float GPU_g_decay_MFGR, float *GPU_gogrW, float GPU_gGABA_decayGOGR, float GPU_g_decay_NMDA_MFGR, float GPU_gDirectInc_MFGR,
        float GPU_threshRest, float GPU_threshDecGR, float GPU_eLeak, float GPU_eGOGR, unsigned char *GPU_spike_mask, float GPU_gogr_LTD_inc, float GPU_gogr_LTP_inc, 
        float GPU_mfgr_LTD_inc, float GPU_mfgr_LTP_inc, unsigned char GPU_go_input, unsigned char GPU_mf_input, unsigned char GPU_mfgr_plast, unsigned char GPU_gogr_plast){
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

            GPU_gSum_MFGR[idx] = inputMFGR[idx] * GPU_mfgrW[idx] + GPU_gSum_MFGR[idx] * GPU_g_decay_MFGR; 
            
            GPU_gSum_GOGR[idx] = inputGOGR[idx] * GPU_gogrW[idx] + GPU_gSum_GOGR[idx] * GPU_gGABA_decayGOGR;

            GPU_gNMDA_MFGR[idx] = GPU_gNMDA_Inc_MFGR[idx] * GPU_gDirectInc_MFGR * inputMFGR[idx] + GPU_gNMDA_MFGR[idx] * 0.9672;

            GPU_Vm[idx] = GPU_Vm[idx] + GPU_gLeak[idx] * (GPU_eLeak - GPU_Vm[idx]) - GPU_gSum_MFGR[idx] * GPU_Vm[idx] - 
            GPU_gNMDA_MFGR[idx] * GPU_Vm[idx] + GPU_gSum_GOGR[idx] * (GPU_eGOGR - GPU_Vm[idx]);

            GPU_currentThresh[idx] = GPU_currentThresh[idx] + (GPU_threshRest - GPU_currentThresh[idx]) * (GPU_threshDecGR);

            GPU_spike_mask[idx] = (GPU_Vm[idx] > GPU_currentThresh[idx]) ? 1 : 0;

            GPU_gogrW[idx] = GPU_gogrW[idx] + GPU_gogr_plast * (GPU_go_input * ((GPU_spike_mask[idx] * GPU_gogr_LTD_inc) + ((GPU_spike_mask[idx] - 1) * GPU_gogr_LTP_inc)));
            GPU_mfgrW[idx] = GPU_mfgrW[idx] + GPU_mfgr_plast * (GPU_mf_input * ((GPU_spike_mask[idx] * GPU_mfgr_LTD_inc) + ((GPU_spike_mask[idx] - 1) * GPU_mfgr_LTP_inc)));

            GPU_mf_input = 0;
            GPU_go_input = 0;

        }
        """

        self.GRgpu = cp.RawKernel(doGranulekernel_code, 'doGranule')
        self.GPU_gLeak = cp.array(self.gLeak, dtype = cp.float32)
        self.GPU_Vm = cp.array(self.Vm, dtype = cp.float32)
        self.GPU_gSum_MFGR = cp.array(self.gSum_MFGR, dtype = cp.float32)
        self.GPU_gSum_GOGR = cp.array(self.gSum_GOGR, dtype = cp.float32)
        self.GPU_gNMDA_Inc_MFGR = cp.array(self.gNMDA_Inc_MFGR, dtype = cp.float32)
        self.GPU_gNMDA_MFGR = cp.array(self.gNMDA_MFGR, dtype = cp.float32)
        self.GPU_currentThresh = cp.array(self.currentThresh, dtype = cp.float32)
        self.GPU_mfgrW = cp.array(self.mfgrW)
        self.GPU_g_decay_MFGR = cp.float32(self.g_decay_MFGR)
        self.GPU_gogrW  = cp.array(self.gogrW)
        self.GPU_gGABA_decayGOGR = cp.float32(self.gGABA_decayGOGR)
        self.GPU_g_decay_NMDA_MFGR = cp.float32(self.g_decay_NMDA_MFGR)
        self.GPU_gDirectInc_MFGR = cp.float32(self.gDirectInc_MFGR)
        self.GPU_threshRest = cp.float32(self.threshRest)
        self.GPU_threshDecGR = cp.float32(self.threshDecGR)
        self.GPU_eLeak = cp.float32(self.eLeak)
        self.GPU_eGOGR = cp.float32(self.eGOGR)
        self.GPU_spike_mask = cp.zeros(self.numGranule, dtype = cp.uint8)

    def doGRGPU(self):
        block_size = 256
        grid_size = (self.numGranule + block_size - 1) // block_size
        with cp.cuda.Device(0):
            GPU_inputMFGR = cp.array(self.inputMFGR)
            GPU_inputGOGR = cp.array(self.inputGOGR)
            self.GRgpu((grid_size,),(block_size,), (self.numGranule, self.GPU_gLeak, self.GPU_Vm, GPU_inputMFGR, self.GPU_gSum_MFGR, GPU_inputGOGR, 
            self.GPU_gSum_GOGR, self.GPU_gNMDA_Inc_MFGR, self.GPU_gNMDA_MFGR, self.GPU_currentThresh, self.GPU_mfgrW, self.GPU_g_decay_MFGR, self.GPU_gogrW, 
            self.GPU_gGABA_decayGOGR, self.GPU_g_decay_NMDA_MFGR, self.GPU_gDirectInc_MFGR, self.GPU_threshRest, self.GPU_threshDecGR, self.GPU_eLeak, 
            self.GPU_eGOGR, self.GPU_spike_mask, self.GPU_gogr_LTD_inc, self.GPU_gogr_LTP_inc, self.GPU_mfgr_LTD_inc, self.GPU_mfgr_LTP_inc, self.GPU_go_input, 
            self.GPU_mf_input, self.GPU_mfgr_plast, self.GPU_gogr_plast))
            return self.GPU_spike_mask.get()

    # generic function for updating GOGR and MFGR input arrays
    def update_input_activity(self, connectArr, inputArrayChoice, mfAct = None, goAct = None):
        ''' inputArrayChoice selects between 1 = MFGR, 2 = GOGR '''
        #### Playground version, adapt to main code
        if inputArrayChoice == 1:
            self.GPU_mf_input = cp.uint8(1) 
            grInputs = np.zeros(self.numGranule, dtype = np.uint8) # array that stores how many inputs each gr gets
            MFdiverge = int((self.numGranule * 5)/4096) # how many gr each MF connects to  <-- where is this used lmao
            spiked_idx = np.where(mfAct)[0]
            # print(spiked_idx)
            for mf in spiked_idx: # for every MF that spikes, this can be done in parallel
            # Important here
                grInputs[connectArr[mf, :]] += 1 # for each active MF, update each gr listed in its row by 1

            # Vectorized below
            # np.add.at(grInputs, connectArr[spiked_idx, :].ravel(), 1)
            self.inputMFGR = grInputs

        ### THIS VERSION IS ACTUALLY SLOWER
        # if inputArrayChoice == 2: 
        #     spiked_idx = np.where(goAct)[0]
        #     for go in spiked_idx: # for every GO that spikes
        #         grInputs[connectArr[go, :]] += 1
        #     # Vectorized below
        #     # np.add.at(grInputs, connectArr[spiked_idx, :].ravel(), 1)
        #     self.inputGOGR = grInputs


        # Version 1
        # ## MF input update
        # if inputArrayChoice == 1:
        #     spike_mask = (mfAct == 1)
        if inputArrayChoice == 2:
            self.GPU_go_input = cp.uint8(1) 
            spike_mask = (goAct == 1)
            spiked_idx = np.where(spike_mask)[0]
            # get connections for all spiked cells
            spiked_connections = [] # list to hold connections for all spiked MFs
            for cell in spiked_idx:
                # eliminate -1 terminator
                valid_idx = connectArr[cell] != -1
                # get valid connections for spiked cell
                valid_conns = connectArr[cell, valid_idx]
                # add to list
                spiked_connections.extend(valid_conns)
            spiked_connections = np.array(spiked_connections, dtype = int) # convert list to
            counts = np.bincount(spiked_connections) # count occurrences of each index in spiked_connections
            self.inputGOGR[:len(counts)] = counts # update inputGOGR with counts, only up to length of counts to avoid index error
        
        # if inputArrayChoice == 1:
        #     # add to MFGR input  
        #     self.inputMFGR[:len(counts)] = counts # update inputMFGR with counts, only up to length of counts to avoid index error
        # elif inputArrayChoice == 2:
            # add to GOGR input
            


    def do_Granule(self, t): # t for MF output, golgi for Golgi activity
        self.act[t] = self.doGRGPU() # convert boolean array to int array,
        # print(t, ":", np.mean(self.GPU_gogrW))
        # print(t, ":", np.sum(self.act[t]))
        # reset inputs
        self.inputMFGR.fill(0)
        self.inputGOGR.fill(0)

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
            self.mfgrW = cp.asnumpy(self.GPU_mfgrW)
            self.g_decay_MFGR = cp.asnumpy(self.GPU_g_decay_MFGR)
            self.gogrW  = cp.asnumpy(self.GPU_gogrW)
            self.gGABA_decayGOGR = cp.asnumpy(self.GPU_gGABA_decayGOGR)
            self.g_decay_NMDA_MFGR = cp.asnumpy(self.GPU_g_decay_NMDA_MFGR)
            self.gDirectInc_MFGR = cp.asnumpy(self.GPU_gDirectInc_MFGR)
            self.threshRest = cp.asnumpy(self.GPU_threshRest)
            self.threshDecGR = cp.asnumpy(self.GPU_threshDecGR)
            self.eLeak = cp.asnumpy(self.GPU_eLeak)
            self.eGOGR = cp.asnumpy(self.GPU_eGOGR)
    

