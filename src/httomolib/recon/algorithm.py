#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Tomography Team at DLS <scientificsoftware@diamond.ac.uk>
# Created Date: 01 November 2022
# version ='0.1'
# ---------------------------------------------------------------------------
"""Modules for tomographic reconstruction"""

from typing import Optional

import cupy as cp
#from cupy import ndarray, swapaxes
import scipy.fftpack
import numpy as np
from numpy import ndarray

__all__ = [
    'reconstruct_tomobar',
]

## %%%%%%%%%%%%%%%%%%%%%%% ToMoBAR reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def reconstruct_tomobar(
    data: ndarray,
    angles: ndarray,
    center: float = None,
    objsize: int = None,
    algorithm: str = 'FBP3D_device',
    gpu_id : int = 0
    ) -> ndarray:
    """
    Perform reconstruction using ToMoBAR wrappers around ASTRA toolbox. 

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : cp.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    objsize : int, optional
        The size in pixels of the reconstructed object.
    algorithm : str, optional
        The name of the reconstruction method, FBP3D_device or FBP3D_host.
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The reconstructed volumes a CuPy array.
    """
    from tomobar.methodsDIR import RecToolsDIR    
    from tomobar.supp.astraOP import AstraTools3D
    
    if center is None:
        center = 0.0
    if objsize is None:
        objsize = data.shape[2]
    if algorithm == "FBP3D_host":
        # set parameters and initiate a TomoBar class object for direct reconstruction
        RectoolsDIR = RecToolsDIR(DetectorsDimH=data.shape[2],  # DetectorsDimH # detector dimension (horizontal)
                                DetectorsDimV=data.shape[1],  # DetectorsDimV # detector dimension (vertical) for 3D case only
                                CenterRotOffset=data.shape[2] * 0.5 - center,  # The center of rotation combined with the shift offsets
                                AnglesVec=-angles,  # the vector of angles in radians
                                ObjSize=objsize,  # a scalar to define the reconstructed object dimensions
                                device_projector=gpu_id)
        
        # ------------------------------------------------------- # 
        # performs 3D FBP with filtering on the CPU (host (filtering) -> device (backprojection) -> host (if needed))
                
        reconstruction = RectoolsDIR.FBP(np.swapaxes(cp.asnumpy(data), 0, 1))
        
        # ------------------------------------------------------- #     
    if algorithm == "FBP3D_device":
        # Perform filtering of the data on the GPU and then pass a pointer to CuPy array to do backprojection, i.e.
        # (host -> device (filtering) -> device (backprojection) -> host (if needed))
        cp.cuda.Device(gpu_id).use()
        
        # initiate a 3D ASTRA class object    
        Atools = AstraTools3D(DetectorsDimH=data.shape[2],  # DetectorsDimH # detector dimension (horizontal)
                                DetectorsDimV=data.shape[1],  # DetectorsDimV # detector dimension (vertical) for 3D case only
                                AnglesVec=-angles,  # the vector of angles in radians
                                CenterRotOffset=data.shape[2] * 0.5 - center,  # The center of rotation combined with the shift offsets                              
                                ObjSize=objsize,  # a scalar to define the reconstructed object dimensions
                                OS_number = 1, # OS recon disabled
                                device_projector = 'gpu',
                                GPUdevice_index=gpu_id)    
        # ------------------------------------------------------- # 
        data = _filtersinc3D_cupy(cp.swapaxes(data, 0, 1)) # filter the data on the GPU and keep the result there
        cp._default_memory_pool.free_all_blocks()       
        
        reconstruction = Atools.backprojCuPy(data) # backproject the filtered data while keeping data on the GPU
        #reconstruction = cp.asnumpy(reconstruction) # if needed
        cp._default_memory_pool.free_all_blocks()
        # ------------------------------------------------------- # 
    return reconstruction

def _filtersinc3D_cupy(projection3D):
    """applies a filter to 3D projection data

    Args:
        projection3D (ndarray): projection data must be a CuPy array.

    Returns:
        ndarray: a CuPy array of filtered projection data.
    """
    a = 1.1
    [DetectorsLengthV, projectionsNum, DetectorsLengthH] = np.shape(projection3D)
    w = np.linspace(-np.pi,np.pi-(2*np.pi)/DetectorsLengthH, DetectorsLengthH,dtype='float32')
    
    # prepearing a ramp-like filter to apply to every projection
    rn1 = np.abs(2.0/a*np.sin(a*w/2.0))
    rn2 = np.sin(a*w/2.0)
    rd = (a*w)/2.0
    rd_c = np.zeros([1,DetectorsLengthH])
    rd_c[0,:] = rd
    r = rn1*(np.dot(rn2, np.linalg.pinv(rd_c)))**2
    multiplier = (1.0/projectionsNum)
    f = scipy.fftpack.fftshift(r)
    f_2d = np.zeros((DetectorsLengthV,DetectorsLengthH), dtype='float32') # 2d filter
    f_2d[0::,:] = f
    filter_gpu = cp.asarray(f_2d)
    
    filtered = cp.zeros(cp.shape(projection3D), dtype='float32') # convert to cupy array

    for i in range(0,projectionsNum):
        IMG = cp.fft.fft2(projection3D[:,i,:])
        fimg = IMG*filter_gpu
        filtered[:,i,:] = cp.real(cp.fft.ifft2(fimg))
    return multiplier*filtered
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##