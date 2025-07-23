import numpy as np
import cupy as cp
import struct

def read_connect(filename, rows, cols, testing = False):
    # Calculate the expected number of elements and bytes
    num_elements = rows * cols
    element_size = 4  # Assuming int is 4 bytes
    
    with open(filename, 'rb') as file:
        data = file.read()
    
    # Check if the data size matches expectations
    expected_bytes = num_elements * element_size
    if len(data) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes, got {len(data)}")
    
    # Unpack the binary data into integers
    format_str = f"{num_elements}i"  # 'i' for int, repeated num_elements times
    array_flat = struct.unpack(format_str, data)
    
    # Reshape the flat list into a 2D array
    array_2d = [array_flat[i*cols : (i+1)*cols] for i in range(rows)]
    array_2d = cp.array(array_2d)

    if testing == True:
        print("Shape:", array_2d.shape)
        print("Size:", array_2d.size)
        print("Row 0: ", array_2d[0][:10])
        print("Row 1: ", array_2d[1][:10])
        print("Row 2: ", array_2d[2][:10])
    
    return array_2d 

def read_all_connect(filename, numGO, numMF, numGR, testing = False):
    # num connections per cell
    pGOGO = 12
    pMFGO = 20
    pMFGR = 4000
    pGOGR = 12800
    pGRGO = 50
    
    suffix = [".gogo", ".mfgo", ".mfgr", ".gogr", ".grgo"]
    # Connect array files
    GOGOfile = filename + suffix[0]
    MFGOfile = filename + suffix[1]
    MFGRfile = filename + suffix[2]
    GOGRfile = filename + suffix[3]
    GRGOfile = filename + suffix[4]

    # Read arrays
    gogo_connect = read_connect(GOGOfile, numGO, pGOGO, testing=testing)
    mfgo_connect = read_connect(MFGOfile, numMF, pMFGO, testing=testing)
    mfgr_connect = read_connect(MFGRfile, numMF, pMFGR, testing=testing)
    gogr_connect = read_connect(GOGRfile, numGO, pGOGR, testing=testing)
    grgo_connect = read_connect(GRGOfile, numGR, pGRGO, testing=testing)

    # Save arrays to list
    connect_arrays = []
    connect_arrays.append(gogo_connect)
    connect_arrays.append(mfgo_connect)
    connect_arrays.append(mfgr_connect)
    connect_arrays.append(gogr_connect)
    connect_arrays.append(grgo_connect)
    
    return connect_arrays