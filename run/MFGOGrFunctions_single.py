import numpy as np
import cupy as cp
from cupy import ElementwiseKernel
import math
import matplotlib.pyplot as plt
import time

class Mossy(): 
    def __init__(self,n,CSon, CSoff):
        self.numMossy = n 
        self.CSon, self.CSoff = CSon, CSoff 
        self.minBackground, self.maxBackground = 1, 30 
        self.minCS, self.maxCS = 90, 100 

        self.act = np.zeros(self.numMossy, dtype = np.uint8) 
        self.sizeOfDist = 1000 
        self.MFisiDistribution = np.zeros((101, self.sizeOfDist), dtype = int) 
        self.generate_MFisiDistribution() 

    def generate_MFisiDistribution(self):
        self.MFisi = np.random.randint(5, 40, self.numMossy) 

        for i in range(0, 101): 
            if i == 0 : f = 1000 
            else: f = 1000.0/(i) 
            f_stdev = f/5.0  
            ISItemp = np.random.normal(loc = f, scale = f_stdev, size = self.sizeOfDist) 
            for j in range(0, self.sizeOfDist):
                if ISItemp[j] < 5: ISItemp[j] = 5
            self.MFisiDistribution[i, :] = ISItemp

        self.MFfreqs = np.random.randint(self.minBackground, self.maxBackground, self.numMossy)
        self.CSfreqs = np.random.randint(self.minCS, self.maxCS, self.numMossy)
        self.CSMFindex = np.random.randint(0, self.numMossy, min(80, self.numMossy)) 
    
    def do_MF_dist(self, timestep, useCS):
        self.act.fill(1) 
        isi_mask = (self.MFisi == 0) 
        self.act = isi_mask.astype(np.uint8) 
        if useCS == 1:
            random_idx = np.random.randint(0, self.sizeOfDist - 1, size=int(np.sum(isi_mask).item())) 
            random_CSMF_idx = np.random.randint(0, self.sizeOfDist - 1, size = int(len(self.CSMFindex))) 
            if (timestep >= self.CSon) and (timestep < self.CSoff): 
                is_CSMF = np.zeros(self.numMossy, dtype = bool)
                is_CSMF[self.CSMFindex] = True 
                spiking_cs_cell = isi_mask & is_CSMF 

                if spiking_cs_cell.any(): 
                    cs_idx = np.where(spiking_cs_cell)[0] 
                    cs_freqs = self.CSfreqs[cs_idx] 
                    cs_rand_idx = random_idx[:len(cs_idx)] 
                    self.MFisi[cs_idx] = self.MFisiDistribution[cs_freqs, cs_rand_idx] 
                
                spiking_non_cs_cell = isi_mask & ~is_CSMF 
                if spiking_non_cs_cell.any(): 
                    non_cs_idx = np.where(spiking_non_cs_cell)[0]
                    non_cs_freq = self.MFfreqs[non_cs_idx]
                    non_cs_rand_idx = random_idx[-len(non_cs_idx):] 
                    self.MFisi[non_cs_idx] = self.MFisiDistribution[non_cs_freq, non_cs_rand_idx] 
                if timestep == self.CSon: 
                    temp_isi = self.MFisiDistribution[self.CSfreqs[self.CSMFindex], random_CSMF_idx] 
                    self.MFisi[self.CSMFindex] = np.minimum(temp_isi, self.MFisi[self.CSMFindex]) 
            else:
                freq_idx = self.MFfreqs[isi_mask] 
                new_isi = self.MFisiDistribution[freq_idx, random_idx] 
                self.MFisi[isi_mask] = new_isi
                if timestep == self.CSoff: 
                    temp_isi = self.MFisiDistribution[self.CSfreqs[self.CSMFindex], random_CSMF_idx] 
                    min_isi = np.minimum(temp_isi, self.MFisi[self.CSMFindex]) 
                    max_isi = np.maximum(temp_isi, self.MFisi[self.CSMFindex]) 
                    self.MFisi[self.CSMFindex] = np.random.randint(min_isi, max_isi + 1, (int(len(self.CSMFindex)),))       
        else: 
            freq_idx = self.MFfreqs[isi_mask] 
            random_idx = np.random.randint(0, self.sizeOfDist - 1, size=int(np.sum(isi_mask).item()))
            new_isi = self.MFisiDistribution[freq_idx, random_idx] 
            self.MFisi[isi_mask] = new_isi 
        self.MFisi[~isi_mask] -= 1 
        return self.act 
    def get_act(self):
        return self.act
            
class Golgi(): 
    def __init__(self, n, csON, csOFF, useCS, numBins, mfgo_plast = 0, gogo_plast = 0, grgo_plast = 0, gogo_weight = 0.0125, mfgo_weight = 0.0042, grgo_weight = 0.0007, plast_ratio = 1):
        self.numGolgi = n
        self.useCS = useCS
        self.csOn = csON
        self.csOff = csOFF
        self.gLeak = 0.02 
        self.eLeak = -70.0 
        self.eGABA = -64.0 
        self.thresholdMax = 10.0 
        self.gGABA_decayGOGO = math.exp(-1.0/10.0) 
        self.g_decayMFGO = math.exp(-1.0/3.0) 
        self.g_decayGRGO = math.exp(-1.0/4.0) 
        self.g_decay_NMDA_MFGO = math.exp(-1.0/30.0) 
        self.NMDA_AMPA_ratioMFGO = 1.3 

        self.threshDecGo = 1- math.exp(-1.0/11.0) 
        self.threshRest = -34.0 

        self.plast_pop_portion = plast_ratio 
        self.mfgo_plast = mfgo_plast
        self.gogo_plast = gogo_plast
        self.grgo_plast = grgo_plast

        self.target_spikes = 25 
        self.plast_mult_constant = 1.003 

        self.grgoW = np.full(self.numGolgi, grgo_weight, dtype = np.float32) 
        self.mfgoW = np.full(self.numGolgi, mfgo_weight, dtype = np.float32) 
        self.gogoW = np.full(self.numGolgi, gogo_weight, dtype = np.float32) 
        self.Vm = np.full(self.numGolgi, self.eLeak, dtype = np.float32) 
        self.gSum_GOGO = np.zeros(self.numGolgi, dtype = np.float32) 
        self.inputGOGO = np.zeros(self.numGolgi, dtype = np.uint8) 
        self.gSum_MFGO = np.zeros(self.numGolgi, dtype = np.float32) 
        self.inputMFGO = np.zeros(self.numGolgi, dtype = np.uint8) 
        self.gSum_GRGO = np.zeros(self.numGolgi, dtype = np.float32) 
        self.inputGRGO = np.zeros(self.numGolgi, dtype = np.uint8) 
        self.gNMDA_inc_MFGO = np.zeros(self.numGolgi, dtype = np.float32) 
        self.gNMDA_MFGO = np.zeros(self.numGolgi, dtype = np.float32) 
        
        self.currentThresh = np.full(self.numGolgi, self.threshRest, dtype = np.float32) 
        self.act = np.zeros((numBins, self.numGolgi), dtype = np.uint8) 

        self.MFGO_trace = np.zeros(self.numGolgi, dtype = np.float32) 
        self.GRGO_trace = np.zeros(self.numGolgi, dtype = np.float32) 
        self.GOGO_trace = np.zeros(self.numGolgi, dtype = np.float32) 


    def do_Golgi(self, t):
        self.gNMDA_inc_MFGO = (
            (0.00000011969 * self.Vm * self.Vm * self.Vm) +
            (0.000089369 * self.Vm * self.Vm) +
            (0.0151 * self.Vm) + 0.7713
        ) 

        self.gSum_GOGO = (self.inputGOGO * self.gogoW) + self.gSum_GOGO * self.gGABA_decayGOGO 
        self.gSum_MFGO = (self.inputMFGO * self.mfgoW) + self.gSum_MFGO * self.g_decayMFGO 
        self.gSum_GRGO = (self.inputGRGO * self.grgoW) + self.gSum_GRGO * self.g_decayGRGO 
        
        self.gNMDA_MFGO = (
            self.inputMFGO * (self.mfgoW * self.NMDA_AMPA_ratioMFGO * self.gNMDA_inc_MFGO) +
            self.gNMDA_MFGO * self.g_decay_NMDA_MFGO
        ) 
        
        self.currentThresh += (self.threshRest - self.currentThresh) * self.threshDecGo 
        self.Vm += (
            (self.gLeak * (self.eLeak - self.Vm)) +
            (self.gSum_GOGO * (self.eGABA - self.Vm)) -
            (self.gSum_MFGO + self.gSum_GRGO + self.gNMDA_MFGO) *
            self.Vm
        ) 

        self.inputMFGO.fill(0)
        self.inputGRGO.fill(0)
        self.inputGOGO.fill(0)

        self.Vm = np.minimum(self.Vm, self.thresholdMax)
        spike_mask = self.Vm > self.currentThresh 
        self.act[t] = spike_mask.astype(np.uint8) 

        self.currentThresh = np.where(spike_mask, self.thresholdMax, self.currentThresh) 

    # --- SIMPLIFIED FOR SINGLE CELL NO ARRAYS ---
    def update_input_activity(self, inputArrayChoice, mfAct = None, t = None, grAct = None):
        ''' inputArrayChoice selects between 1 = MFGO, 2 = GOGO, 3 = GRGO. Direct 1-to-1 connection.'''
        # MFGO
        if inputArrayChoice == 1:
            self.inputMFGO = mfAct.copy()
        # GOGO
        if inputArrayChoice == 2:
            self.inputGOGO = self.act[t].copy()
        # GRGO
        if inputArrayChoice == 3: 
            self.inputGRGO = grAct.copy()

    def update_weight(self, t, exc_or_inh, weight_array):
        # Handled dynamic length for 1 cell
        num_plast = max(1, int(self.numGolgi * self.plast_pop_portion))
        plast_cells = np.random.choice(self.numGolgi, num_plast, replace = False) 

        count = np.sum(self.act, axis = 0) 
        count_float = count.astype(float)
        error = self.target_spikes - count_float 

        if exc_or_inh == 2:
            error = -error 
        
        learning_rates = np.where(error > 0, self.plast_mult_constant, 1/self.plast_mult_constant) 
        learning_rates = np.where(error == 0, 1, learning_rates) 

        is_plastic = np.zeros(self.numGolgi, dtype=bool)
        is_plastic[plast_cells] = True 
        
        learning_rates = np.where(is_plastic, learning_rates, 1.0)
        weight_array = weight_array * learning_rates
        weight_array = np.clip(weight_array, 0.0, 1.0)

        return weight_array

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
    def __init__(self, n, csOFF, csON, useCS, trialSize, mfgr_plast = 0, gogr_plast = 0, mfgr_weight = 0.0007, gogr_weight = 0.015):
        self.numGranule = n
        self.csOFF, self.csON = csOFF, csON
        self.useCS = useCS 
        self.eLeak = -65.0
        self.eGOGR = -75.0
        self.thresholdMax = 10.0 
        self.gLeak_base = 0.1  
        self.g_decay_NMDA_MFGR = math.exp(-1.0/30.0) 
        self.gDirectInc_MFGR = 0.0320
        self.g_decay_MFGR = 0.9355 
        self.gGABA_decayGOGR = math.exp(-1.0/7.0) 

        self.threshDecGR = 1 - math.exp(-1.0/3.0) 
        self.threshRest = -40.0

        self.GPU_mfgr_plast = cp.uint8(mfgr_plast)
        self.GPU_gogr_plast = cp.uint8(gogr_plast)

        self.GPU_tau_trace = cp.float32(2000.0) 
        self.GPU_target_hz = cp.float32(0.1)
        self.GPU_target_trace_val = cp.float32(self.GPU_target_hz * (self.GPU_tau_trace / 1000.0))

        base_alpha = 0.00001 
        self.GPU_alpha_up = cp.float32(base_alpha) 
        self.GPU_alpha_down = cp.float32(base_alpha) 

        self.mfgrW = np.full(self.numGranule, mfgr_weight, dtype = np.float32) 
        self.gogrW = np.full(self.numGranule, gogr_weight, dtype = np.float32) 
        self.Vm = np.full(self.numGranule, self.eLeak, dtype = np.float32) 
        self.gSum_MFGR = np.zeros(self.numGranule, dtype = np.float32) 
        self.inputMFGR = np.zeros(self.numGranule, dtype = np.uint8) 
        self.gSum_GOGR = np.zeros(self.numGranule, dtype = np.float32) 
        self.inputGOGR = np.zeros(self.numGranule, dtype = np.uint8) 
        self.gNMDA_MFGR= np.zeros(self.numGranule, dtype = np.float32) 
        self.gNMDA_Inc_MFGR = np.zeros(self.numGranule, dtype = np.float32) 
        self.gLeak = np.full(self.numGranule, self.gLeak_base, dtype = np.float32) 
        self.gKCa = np.zeros(self.numGranule, dtype = np.float32) 
        self.currentThresh = np.full(self.numGranule, self.threshRest, dtype = np.float32) 
        self.act = np.zeros((trialSize, self.numGranule), dtype = np.uint8) 
        
        # Kernel stuff unchanged
        doGranulekernel_code = """
        extern "C" __global__ void doGranule(int size, float *GPU_gLeak, float *GPU_Vm, unsigned char *inputMFGR, float *GPU_gSum_MFGR,
        unsigned char *inputGOGR, float *GPU_gSum_GOGR, float *GPU_gNMDA_Inc_MFGR, float *GPU_gNMDA_MFGR, float *GPU_currentThresh, float *GPU_mfgrW, 
        float GPU_g_decay_MFGR, float *GPU_gogrW, float GPU_gGABA_decayGOGR, float GPU_g_decay_NMDA_MFGR, float GPU_gDirectInc_MFGR,
        float GPU_threshRest, float GPU_threshDecGR, float GPU_eLeak, float GPU_eGOGR, unsigned char *GPU_spike_mask, 
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

            float spike_val = (float)GPU_spike_mask[idx];

            float mf_trace = GPU_mfgr_trace[idx];
            mf_trace += (-mf_trace / tau_trace);
            mf_trace += spike_val;
            GPU_mfgr_trace[idx] = mf_trace; 
            
            float mf_error = target_trace_val - mf_trace;
            
            float mf_rate = (mf_error > 0.0f) ? alpha_up : alpha_down;
            
            float mf_weight = GPU_mfgrW[idx];
            mf_weight += GPU_mfgr_plast * (mf_rate * mf_error * mf_weight);
            
            GPU_mfgrW[idx] = fminf(1.0f, fmaxf(0.0f, mf_weight));

            float go_trace = GPU_gogr_trace[idx];
            
            go_trace += (-go_trace / tau_trace);
            go_trace += spike_val;
            GPU_gogr_trace[idx] = go_trace; 
            
            float go_error = -(target_trace_val - go_trace); 
            
            float go_rate = (go_error > 0.0f) ? alpha_up : alpha_down;
            
            float go_weight = GPU_gogrW[idx];
            go_weight += GPU_gogr_plast * (go_rate * go_error * go_weight);
            
            GPU_gogrW[idx] = fminf(1.0f, fmaxf(0.0f, go_weight));
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
        self.GPU_mfgr_trace = cp.zeros(self.numGranule, dtype=cp.float32)
        self.GPU_gogr_trace = cp.zeros(self.numGranule, dtype=cp.float32)

    def doGRGPU(self):
        block_size = 256
        grid_size = (self.numGranule + block_size - 1) // block_size
        with cp.cuda.Device(0):
            GPU_inputMFGR = cp.array(self.inputMFGR)
            GPU_inputGOGR = cp.array(self.inputGOGR)
            
            self.GRgpu((grid_size,),(block_size,), (self.numGranule, self.GPU_gLeak, self.GPU_Vm, GPU_inputMFGR, self.GPU_gSum_MFGR, GPU_inputGOGR, 
            self.GPU_gSum_GOGR, self.GPU_gNMDA_Inc_MFGR, self.GPU_gNMDA_MFGR, self.GPU_currentThresh, self.GPU_mfgrW, self.GPU_g_decay_MFGR, self.GPU_gogrW, 
            self.GPU_gGABA_decayGOGR, self.GPU_g_decay_NMDA_MFGR, self.GPU_gDirectInc_MFGR, self.GPU_threshRest, self.GPU_threshDecGR, self.GPU_eLeak, 
            self.GPU_eGOGR, self.GPU_spike_mask, self.GPU_mfgr_plast, self.GPU_gogr_plast, self.GPU_mfgr_trace, self.GPU_gogr_trace, self.GPU_tau_trace,self.GPU_target_trace_val,
            self.GPU_alpha_up, self.GPU_alpha_down))
            return self.GPU_spike_mask.get()

    # --- SIMPLIFIED FOR SINGLE CELL NO ARRAYS ---
    def update_input_activity(self, inputArrayChoice, mfAct = None, goAct = None):
        ''' inputArrayChoice selects between 1 = MFGR, 2 = GOGR. Direct 1-to-1 connection.'''
        # MFGR
        if inputArrayChoice == 1:
            self.inputMFGR = mfAct.copy()
        # GOGR
        if inputArrayChoice == 2:
            self.inputGOGR = goAct.copy()

    def do_Granule(self, t): 
        self.act[t] = self.doGRGPU() 
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