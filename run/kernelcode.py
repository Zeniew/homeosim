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
GRgpu((grid_size,),(block_size), (GR.numGranule, GPU_Vm, GR.inputMFGR, GPU_gSum_MFGR, GR.inputGOGR, 
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

#### Golgi

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

self.GOGPU = cp.RawKernel(doGolgiKernel_code, 'doGolgi')
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
self.GPU_spike_mask = cp.array(self.spike_mask, dtype = cp.uint8)

def doGOGPU(self):
    block_size = 256
    grid_size = (self.numGolgi + block_size - 1) // block_size
    with cp.cuda.Device(0):
        GPU_inputMFGO = cp.array(self.inputMFGO) # might be able to get rid of later when update_input_activity also on GPU
        GPU_inputGOGO = cp.array(self.inputGOGO)
        GPU_inputGRGO = cp.array(self.inputGRGO)
        self.GOgpu((grid_size,), (block_size), (self.numGolgi, self.GPU_gNMDA_inc_MFGO, self.GPU_Vm, 
        self.GPU_gSum_GOGO, self.GPU_inputGOGO, self.GPU_gogoW, self.GPU_gGABA_decayGOGO, self.GPU_gSum_MFGO, self.GPU_inputMFGO,
        self.GPU_mfgoW, self.GPU_g_decayMFGO, self.GPU_gSum_GRGO, self.GPU_inputGRGO, self.GPU_grgoW, self.GPU_g_decayGRGO, self.GPU_gNMDA_MFGO, 
        self.GPU_NMDA_AMPA_ratioMFGO, self.GPU_g_decay_NMDA_MFGO, self.GPU_currentThresh, self.GPU_threshRest, self.GPU_threshDecGo, 
        self.GPU_gLeak, self.GPU_eLeak, self.GPU_spike_mask))
        return self.GPU_spike_mask.get()

def do_Golgi(self, t):
    self.act[t] = self.doGOGPU()
    self.inputMFGO.fill(0)
    self.inputGOGO.fill(0)
    self.inputGRGO.fill(0)


######################## Update Input Activity #######################################################
# Update Input Activity Kernel Code for Golgi

block_size = 256
grid_size = (numGOs + block_size - 1) // block_size

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

self.updateGOGPU = cp.RawKernel(update_input_kernel_code, 'updateInputActivity')

def update_input_activity(self, inputArrayChoice, activity, mfgoconnArr, gogoconnArr, grgoconnArr):
    act = cp.array(activity, dtype = cp.uint8)
    block_size = 256
    grid_size = (self.numGolgi + block_size - 1) // block_size
    if inputArrayChoice == 1:
        numInputsPerCell = 20
        with cp.cuda.Device(0):
            self.updateGOGPU((grid_size,), (block_size,), (act, mfgoconnArr, self.GPU_inputMFGO, self.numGolgi, numInputsPerCell))
        self.inputMFGO = cp.asnumpy(self.GPU_inputMFGO)
    elif inputArrayChoice == 2:
        numInputsPerCell = 12
        with cp.cuda.Device(0):
            self.updateGOGPU((grid_size,), (block_size,), (act, gogoconnArr, self.GPU_inputGOGO, self.numGolgi, numInputsPerCell))
        self.inputGOGO = cp.asnumpy(self.GPU_inputGOGO)
    elif inputArrayChoice == 3:
        numInputsPerCell = 50
        with cp.cuda.Device(0):
            self.updateGOGPU((grid_size,), (block_size,), (act, grgoconnArr, self.GPU_inputGRGO, self.numGolgi, numInputsPerCell))
        self.inputGRGO = cp.asnumpy(self.GPU_inputGRGO)

GO.update_input_activity(1, mfAct, MFGO_connect_arr, GOGO_connect_arr, GRGO_connect_arr)
GO.update_input_activity(2, goAct, MFGO_connect_arr, GOGO_connect_arr, GRGO_connect_arr)
GO.update_input_activity(3, grAct, MFGO_connect_arr, GOGO_connect_arr, GRGO_connect_arr)

# Update Input Activity Kernel Code for Granule

block_size = 256
grid_size = (numGRs + block_size - 1) // block_size

update_input_kernel_code = """
extern "C" __global__ void updateInputActivity(
    unsigned char *act,           // activity vector of golgi cells (0/1)
    const int *connArr,     // flattened 2D array: numGolgi x numInputsPerCell       
    int *inputArr,             // output sum for each Golgi
    int numGranule,
    int numInputsPerCell
) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numGranule) return;

    int sum = 0;
    for (int j = 0; j < numInputsPerCell; j++) {
        int inputIndex = connArr[idx * numInputsPerCell + j]; // get connected granule index
        sum += act[inputIndex];                             // add 0 or 1
    }

    inputArr[idx] = sum;

}
"""

def update_input_activity(self, inputArrayChoice, activity, mfgrconnArr, gogrconnArr):
    act = cp.array(activity, dtype = cp.uint8)
    block_size = 256
    grid_size = (self.numGranule + block_size - 1) // block_size
    if inputArrayChoice == 1:
        numInputsPerCell = 4000
        with cp.cuda.Device(0):
            self.updateGRGPU((grid_size,), (block_size,), (act, mfgrconnArr, self.GPU_inputMFGR, self.numGranule, numInputsPerCell))
        self.inputMFGR = cp.asnumpy(self.GPU_inputMFGR)
    elif inputArrayChoice == 2:
        numInputsPerCell = 12800
        with cp.cuda.Device(0):
            self.updateGRGPU((grid_size,), (block_size,), (act, gogrconnArr, self.GPU_inputGOGR, self.numGranule, numInputsPerCell))
        self.inputGOGR = cp.asnumpy(self.GPU_inputGOGR)

GR.update_input_activity(1, mfAct, MFGR_connect_arr, GOGR_connect_arr)
GR.update_input_activity(2, goAct, MFGR_connect_arr, GOGR_connect_arr)