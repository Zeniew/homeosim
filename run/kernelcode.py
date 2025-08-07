import cupy as cp 
import numpy as np 

block_size = 256
grid_size = (GR.numGranule + block_size - 1) // block_size

# This part probably goes into initialization

doGranulekernel_code = """
extern "C" __global__ void doGranule(int size, float64 *GPU_gLeak, float64 *GPU_Vm, uint8 inputMFGR, float64 *GPU_gSum_MFGR,
uint8 inputGOGR, float64 *GPU_gSum_GOGR, float64 *GPU_gNMDA_Inc_MFGR,
float64 *GPU_gNMDA_MFGR, float64 *GPU_currentThresh, float64 *GPU_mfgrW, float64 *GPU_g_decay_MFGR,
float64 *GPU_gogrW, float64 *GPU_gGABA_decayGOGR, float64 *GPU_g_decay_NMDA_MFGR, float64 *GPU_gDirectInc_MFGR,
float64 *GPU_threshRest, float64 *GPU_threshDecGR, float64 *GPU_eLeak, float64 *GPU_eGOGR, bool *GPU_spike_mask)
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
    
    GPU_gSum_GOGR[idx] = inputGOGR * GPU_gogrW + GPU_gSum_GOGR[idx] * GPU_gGABA_decayGOGR;

    GPU_gNMDA_MFGR[idx] = GPU_gNMDA_Inc_MFGR[idx] * GPU_gDirectInc_MFGR * inputMFGR + GPU_gNMDA_MFGR[idx] * 0.9672;

    GPU_Vm[idx] = GPU_Vm[idx] + GPU_gLeak[idx] * (GPU_eLeak - GPU_Vm[idx]) - GPU_gSum_MFGR[idx] * GPU_Vm[idx] - 
    GPU_gNMDA_MFGR[idx] * GPU_Vm[idx] + GPU_gSum_GOGR[idx] * (GPU_eGOGR - GPU_Vm[idx]);

    GPU_currentThresh[idx] = GPU_currentThresh[idx] + (GPU_threshRest - GPU_currentThresh[idx]) * (GPU_threshDecGR);

    GPU_spike_mask = GPU_Vm > GPU_currentThresh

"""

self.GRgpu = cp.RawKernel(doGranulekernel_code, 'doGranule')
self.GPU_Vm = cp.array(self.Vm)
self.GPU_gSum_MFGR = cp.array(self.gSum_MFGR)
self.GPU_gSum_GOGR = cp.array(self.gSum_GOGR)
self.GPU_gNMDA_Inc_MFGR = cp.array(self.gNMDA_Inc_MFGR)
self.GPU_gNMDA_MFGR = cp.array(self.gNMDA_MFGR)
self.GPU_currentThresh = cp.array(self.currentThresh)
self.GPU_mfgrW = cp.array(self.mfgrW)
self.GPU_g_decay_MFGR = cp.array(self.g_decay_MFGR)
self.GPU_gogrW  = cp.array(self.gogrW)
self.GPU_gGABA_decayGOGR = cp.array(self.gGABA_decayGOGR)
self.GPU_g_decay_NMDA_MFGR = cp.array(self.g_decay_NMDA_MFGR)
self.GPU_gDirectInc_MFGR = cp.array(self.gDirectInc_MFGR)
self.GPU_threshRest = cp.array(self.threshRest)
self.GPU_threshDecGR = cp.array(self.threshDecGR)
self.GPU_eLeak = cp.array(self.eLeak)
self.GPU_eGOGR = cp.array(self.eGOGR)
self.GPU_spike_mask = cp.full(self.numGranule, False, dtype = bool)


# Calling the kernel inside the code
GRgpu((grid_size,),(block_size), (GR.numGRanule, GPU_Vm, GR.inputMFGR, GPU_gSum_MFGR, GR.inputGOGR, 
GPU_gSum_GOGR, GPU_gNMDA_Inc_MFGR, GPU_gNMDA_MFGR, GPU_currentThresh, GPU_mfgrW, GPU_g_decay_MFGR, GPU_gogrW, 
GPU_gGABA_decayGOGR, GPU_g_decay_NMDA_MFGR, GPU_gDirectInc_MFGR, GPU_threshRest, GPU_threshDecGR, GPU_eLeak, GPU_eGOGR))

# Function that gets called each time step, component of the doGranule
def doGRGPU(self):
    with cp.cudaDevice(0):
        GPU_inputMFGR = cp.array(self.inputMFGR)
        GPU_inputGOGR = cp.array(self.inputGOGR)
        GRgpu((grid_size,),(block_size), (self.numGRanule, GPU_Vm, self.inputMFGR, GPU_gSum_MFGR, self.inputGOGR, 
        GPU_gSum_GOGR, GPU_gNMDA_Inc_MFGR, GPU_gNMDA_MFGR, GPU_currentThresh, GPU_mfgrW, GPU_g_decay_MFGR, GPU_gogrW, 
        GPU_gGABA_decayGOGR, GPU_g_decay_NMDA_MFGR, GPU_gDirectInc_MFGR, GPU_threshRest, GPU_threshDecGR, GPU_eLeak, GPU_eGOGR))

        spike_mask = GPU_spike_mask.get()
        return spike_mask

# At the end of experiment, pull from the GPU once and update the class on the CPU
def updateFinalState(self):
    wih cp.cudaDevice(0):
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
