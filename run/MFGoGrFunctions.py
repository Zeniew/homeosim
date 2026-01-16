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

        ## Multiplicative
        self.target_hz = 5.0
        self.tau_trace = 500.0 # ms (Calcium Integration Window)
        self.UP_LIMIT = 2.73 # Turrigiano 1998 TTX
        self.DOWN_LIMIT = 0.66 # Turrigiano 1998 bicuculline
        self.alpha = 0.0001 # homeostastic learning rates
        self.alpha_up = self.alpha #  * self.UP_LIMIT
        self.alpha_down = self.alpha # * self.DOWN_LIMIT
        self.target_trace_val = self.target_hz * (self.tau_trace / 1000.0) # target trace value for homeostasis

        ## Discrete
        # self.plast_ratio = np.float32(1/200) # LTP / LTD, 5 Hz
        # self.mfgo_LTD_inc = np.float32((1/100000) * mfgo_weight * -1)
        # self.mfgo_LTP_inc = np.float32(self.plast_ratio * self.mfgo_LTD_inc) # negative due to computation
        # self.gogo_LTD_inc =  np.float32((1/100000) * gogo_weight * -1)
        # self.gogo_LTP_inc = np.float32(self.plast_ratio * self.gogo_LTD_inc) 
        # self.grgo_LTD_inc =  np.float32((1/100000) * grgo_weight * -1)
        # self.grgo_LTP_inc = np.float32(self.plast_ratio * self.grgo_LTD_inc) 

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

        # Trace
        self.MFGO_trace = np.zeros(self.numGolgi, dtype = np.float32) # trace for MF to Golgi input
        self.GRGO_trace = np.zeros(self.numGolgi, dtype = np.float32) # trace for granule to Golgi input
        self.GOGO_trace = np.zeros(self.numGolgi, dtype = np.float32) # trace for Golgi to Golgi input


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
        ## Multiplicative
        if self.mfgo_plast == 1:  # update MF to Golgi weight
            self.update_weight(self.mfgoW, self.MFGO_trace, 1, t) # update MF to Golgi weight
        if self.grgo_plast == 1:  # update granule to Golgi weight
            self.update_weight(self.grgoW, self.GRGO_trace, 1, t) # update granule to Golgi weight
        if self.gogo_plast == 1:  # update Golgi to Golgi weight
            self.update_weight(self.gogoW, self.GOGO_trace, 2, t) # update Golgi to Golgi weight

        ## Discrete
        # self.grgoW += self.grgo_plast * (((self.act[t] * self.grgo_LTD_inc) + ((self.act[t] - 1) * self.grgo_LTP_inc)))
        # self.mfgoW += self.mfgo_plast * (((self.act[t] * self.mfgo_LTD_inc) + ((self.act[t] - 1) * self.mfgo_LTP_inc))) # this is working!
        # self.gogoW += self.gogo_plast * (((1-self.act[t]) * self.gogo_LTD_inc) + ((self.act[t]) * -1.0 * self.gogo_LTP_inc)) # inhibitory, so inverted

        # # capping
        # self.grgoW = np.clip(self.grgoW, 0.0, 1.0)
        # self.mfgoW = np.clip(self.mfgoW, 0.0, 1.0)
        # self.gogoW = np.clip(self.gogoW, 0.0, 1.0)

        # Update thresholds where spikes occurred (condition, value if true, if false)
        self.currentThresh = np.where(spike_mask, self.thresholdMax, self.currentThresh) # set current threshold to max where spikes occurred

    # generic function for updating GOGO and MFGO input arrays
    # def update_input_activity(self, connectArr, inputArrayChoice, mfAct = None, t = None, grAct = None):
    def update_input_activity(self, connectArr, inputArrayChoice, mfAct = None, t = None, grAct = None):
        ''' inputArrayChoice selects between 1 = MFGO, 2 = GOGO, 3 = GRGO'''
        ### Playground version, adapt to the main code

        # MFGO - faster, so kept
        if inputArrayChoice == 1:
            goInputs = np.zeros(self.numGolgi, dtype = np.uint8)
            spiked_idx = np.where(mfAct)[0]
            for mf in spiked_idx:
                goInputs[connectArr[mf, :]] += 1
            self.inputMFGO = goInputs
            # mfAct[0] = 0 # hardcode to 0 for optimization
            # self.inputMFGO = np.sum(mfAct[connectArr], axis = 1)
        
        # GOGO
        if inputArrayChoice == 2:
            goInputs = np.zeros(self.numGolgi, dtype = np.uint8)
            spiked_idx = np.where(self.act[t])[0]
            for go in spiked_idx:
                goInputs[connectArr[go, :]] += 1
            self.inputGOGO = goInputs

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
            goInputs = np.zeros(self.numGolgi, dtype = np.uint8)
            spiked_idx = np.where(grAct)[0]
            for gr in spiked_idx:
                goInputs[connectArr[gr,:]] += 1
            self.inputGRGO = goInputs
            # # Version 1 
            # grAct[0] = 0 # hardcode to 0 for optimization
            # self.inputGRGO = np.sum(grAct[connectArr], axis = 1)

    def update_weight(self, weight_array, trace_array, exc_or_inh, t):
        # 1. Update Calcium Trace (Leaky Integrator)
        # Decay the trace
        decay = (-trace_array / self.tau_trace) * 1.0 # 1.0 ms
        trace_array += decay 
        # Add spike
        trace_array += self.act[t] # 1.0 ms

        # 2. Calculate Error
        error = self.target_trace_val - trace_array # error between target trace value and current trace value

        # 3. Excitatory vs Inhibitory
        if exc_or_inh == 2:
            error = -error 
        
        # 4. Asymmetric Learning Rate
        learning_rates = np.where(error > 0, self.alpha_up, self.alpha_down) # use alpha_up if error is positive, alpha_down otherwise

        # 5. Multiplicative Plasticity
        delta_w = learning_rates * error * weight_array * 1.0 # 1.0 ms
        weight_array += delta_w 

        # Clip weights
        weight_array = np.clip(weight_array, 0.0, 1.0)



    def get_act(self):
        return self.act
    
    def get_mfgoW(self):
        return self.mfgoW
    
    def get_grgoW(self):
        return self.grgoW

    def get_gogoW(self):
        return self.gogoW
    
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

        self.GPU_tau_trace = cp.float32(2000.0) # NTS: need to find literature supporting this
        self.GPU_target_hz = cp.float32(0.1)
        self.GPU_target_trace_val = cp.float32(self.GPU_target_hz * (self.GPU_tau_trace / 1000.0))

        # Turrigiano Learning Rates
        base_alpha = 0.00001 # 1/10 of Golgi, need to find literature supporting this
        self.GPU_alpha_up = cp.float32(base_alpha) # * 2.73
        self.GPU_alpha_down = cp.float32(base_alpha) # * (1.0 / 0.66)

        # self.plast_ratio =  cp.float32(1/10000) # LTP / LTD, 0.1 Hz
        # self.GPU_mfgr_LTD_inc = cp.float32((1/100000) * mfgr_weight * -1) # 1/100000
        # self.GPU_mfgr_LTP_inc = cp.float32(self.plast_ratio * self.GPU_mfgr_LTD_inc) # negative due to computation
        # self.GPU_gogr_LTD_inc = cp.float32((1/100000) * gogr_weight * -1) # 1/100000
        # self.GPU_gogr_LTP_inc = cp.float32(self.plast_ratio * self.GPU_gogr_LTD_inc)


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
        float GPU_threshRest, float GPU_threshDecGR, float GPU_eLeak, float GPU_eGOGR, unsigned char *GPU_spike_mask, 
        // float GPU_gogr_LTD_inc, 
        // float GPU_gogr_LTP_inc, 
        // float GPU_mfgr_LTD_inc, 
        // float GPU_mfgr_LTP_inc, 
        unsigned char GPU_mfgr_plast, unsigned char GPU_gogr_plast,
        float *GPU_mfgr_trace,    // Trace array for MF->GR synapses
        float *GPU_gogr_trace,    // Trace array for GO->GR synapses
        float tau_trace,          // 500.0
        float target_trace_val,   // 2.5 (for 5Hz)
        float alpha_up,           // calculated alpha * 2.73
        float alpha_down          // calculated alpha * (1/0.66)
        ){
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

            // Multiplicative Scaling
            // Cast spike to float for math
            float spike_val = (float)GPU_spike_mask[idx];

            // A. MF -> GR Update (Excitatory)
            // ----------------------------------------------------
            float mf_trace = GPU_mfgr_trace[idx];
            // Update Trace: Decay + Spike
            mf_trace += (-mf_trace / tau_trace);
            mf_trace += spike_val;
            GPU_mfgr_trace[idx] = mf_trace; // Write back state
            
            // Calc Error (Positive = Need Potentiation)
            float mf_error = target_trace_val - mf_trace;
            
            // Determine Rate (Asymmetric)
            float mf_rate = (mf_error > 0.0f) ? alpha_up : alpha_down;
            
            // Update Weight (Multiplicative)
            // Formula: W_new = W_old + (alpha * error * W_old)
            float mf_weight = GPU_mfgrW[idx];
            mf_weight += GPU_mfgr_plast * (mf_rate * mf_error * mf_weight);
            
            // Clamp between 0.0 and 1.0
            GPU_mfgrW[idx] = fminf(1.0f, fmaxf(0.0f, mf_weight));

            // B. GO -> GR Update (Inhibitory)
            // ----------------------------------------------------
            float go_trace = GPU_gogr_trace[idx];
            
            // Update Trace
            go_trace += (-go_trace / tau_trace);
            go_trace += spike_val;
            GPU_gogr_trace[idx] = go_trace; // Write back
            
            // Calc Error (Inverted for Inhibition)
            // If Trace > Target (Error < 0), we are firing too fast.
            // For inhibition, we need to STRENGTHEN the weight to stop firing.
            // So we invert the error sign to make it Positive (triggering Potentiation logic).
            float go_error = -(target_trace_val - go_trace); 
            
            // Determine Rate
            float go_rate = (go_error > 0.0f) ? alpha_up : alpha_down;
            
            // Update Weight
            float go_weight = GPU_gogrW[idx];
            go_weight += GPU_gogr_plast * (go_rate * go_error * go_weight);
            
            // Clamp
            GPU_gogrW[idx] = fminf(1.0f, fmaxf(0.0f, go_weight));

            //----------------------------------------------------------------------------------------------//
            
            // Discrete Scaling
            
            //GPU_gogrW[idx] = GPU_gogrW[idx] + GPU_gogr_plast * (((1-GPU_spike_mask[idx]) * GPU_gogr_LTD_inc) + (-1 * GPU_spike_mask[idx] * GPU_gogr_LTP_inc));
            //GPU_mfgrW[idx] = GPU_mfgrW[idx] + GPU_mfgr_plast * ((GPU_spike_mask[idx] * GPU_mfgr_LTD_inc) + ((GPU_spike_mask[idx] - 1) * GPU_mfgr_LTP_inc));
            //// --- CAPPING (Added Here) ---
            //// Using fminf/fmaxf logic: "Take the max of 0 and value (floor), then take the min of 1 and that result (ceiling)"
            
            //// Clamp MF->GR (Excitatory)
            //GPU_mfgrW[idx] = fminf(1.0f, fmaxf(0.0f, GPU_mfgrW[idx]));

            //// Clamp GO->GR (Inhibitory)
            //GPU_gogrW[idx] = fminf(1.0f, fmaxf(0.0f, GPU_gogrW[idx]));
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
        # New Trace Arrays (Float32) on GPU
        self.GPU_mfgr_trace = cp.zeros(self.numGranule, dtype=cp.float32)
        self.GPU_gogr_trace = cp.zeros(self.numGranule, dtype=cp.float32)

    def doGRGPU(self):
        block_size = 256
        grid_size = (self.numGranule + block_size - 1) // block_size
        with cp.cuda.Device(0):
            GPU_inputMFGR = cp.array(self.inputMFGR)
            GPU_inputGOGR = cp.array(self.inputGOGR)
            # self.GRgpu((grid_size,),(block_size,), (self.numGranule, self.GPU_gLeak, self.GPU_Vm, GPU_inputMFGR, self.GPU_gSum_MFGR, GPU_inputGOGR, 
            # self.GPU_gSum_GOGR, self.GPU_gNMDA_Inc_MFGR, self.GPU_gNMDA_MFGR, self.GPU_currentThresh, self.GPU_mfgrW, self.GPU_g_decay_MFGR, self.GPU_gogrW, 
            # self.GPU_gGABA_decayGOGR, self.GPU_g_decay_NMDA_MFGR, self.GPU_gDirectInc_MFGR, self.GPU_threshRest, self.GPU_threshDecGR, self.GPU_eLeak, 
            # self.GPU_eGOGR, self.GPU_spike_mask, self.GPU_gogr_LTD_inc, self.GPU_gogr_LTP_inc, self.GPU_mfgr_LTD_inc, self.GPU_mfgr_LTP_inc, self.GPU_mfgr_plast, self.GPU_gogr_plast))
            
            self.GRgpu((grid_size,),(block_size,), (self.numGranule, self.GPU_gLeak, self.GPU_Vm, GPU_inputMFGR, self.GPU_gSum_MFGR, GPU_inputGOGR, 
            self.GPU_gSum_GOGR, self.GPU_gNMDA_Inc_MFGR, self.GPU_gNMDA_MFGR, self.GPU_currentThresh, self.GPU_mfgrW, self.GPU_g_decay_MFGR, self.GPU_gogrW, 
            self.GPU_gGABA_decayGOGR, self.GPU_g_decay_NMDA_MFGR, self.GPU_gDirectInc_MFGR, self.GPU_threshRest, self.GPU_threshDecGR, self.GPU_eLeak, 
            self.GPU_eGOGR, self.GPU_spike_mask, self.GPU_mfgr_plast, self.GPU_gogr_plast, self.GPU_mfgr_trace, self.GPU_gogr_trace, self.GPU_tau_trace,self.GPU_target_trace_val,
            self.GPU_alpha_up, self.GPU_alpha_down))
            return self.GPU_spike_mask.get()

    # generic function for updating GOGR and MFGR input arrays
    def update_input_activity(self, connectArr, inputArrayChoice, mfAct = None, goAct = None):
        ''' inputArrayChoice selects between 1 = MFGR, 2 = GOGR '''
        #### Playground version, adapt to main code
        # MFGR
        if inputArrayChoice == 1:
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

        # GOGR
        if inputArrayChoice == 2:
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
    
    def get_mfgrW(self):
        return self.mfgrW
    
    def get_gogrW(self):
        return self.gogrW

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
    

