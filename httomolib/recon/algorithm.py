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
import numpy as np
import nvtx

__all__ = [
    'reconstruct_tomobar',
    'reconstruct_tomopy',
]

## %%%%%%%%%%%%%%%%%%%%%%% ToMoBAR reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
@nvtx.annotate()
def reconstruct_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: float = None,
    objsize: int = None,
    algorithm: str = 'FBP3D_device',
    gpu_id : int = 0
    ) -> cp.ndarray:
    """
    Perform reconstruction using ToMoBAR wrappers around ASTRA toolbox.
    This is a 3D recon using 3D astra geometry routines.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : np.ndarray
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
        The reconstructed volume as cp or np array.
    """
    from tomobar.methodsDIR import RecToolsDIR
    from tomobar.supp.astraOP import AstraTools3D
    
    cp._default_memory_pool.free_all_blocks()
    
    if center is None:
        center = 0.0
    if objsize is None:
        objsize = data.shape[2]
    if algorithm == "FBP3D_host":
        # set parameters and initiate a TomoBar class object for direct reconstruction
        RectoolsDIR = RecToolsDIR(DetectorsDimH=data.shape[2],  # DetectorsDimH # detector dimension (horizontal)
                                DetectorsDimV=data.shape[1],  # DetectorsDimV # detector dimension (vertical) for 3D case only
                                CenterRotOffset=data.shape[2] / 2 - center - 0.5,  # The center of rotation combined with the shift offsets
                                AnglesVec=-angles,  # the vector of angles in radians
                                ObjSize=objsize,  # a scalar to define the reconstructed object dimensions
                                device_projector=gpu_id)
        
        # ------------------------------------------------------- # 
        # performs 3D FBP with filtering on the CPU (host (filtering) -> device (backprojection) -> host (if needed))
                
        reconstruction = RectoolsDIR.FBP(np.swapaxes(cp.asnumpy(data), 0, 1)) # the output stored as a numpy array
        
        # ------------------------------------------------------- #     
    elif algorithm == "FBP3D_device":
        # Perform filtering of the data on the GPU and then pass a pointer to CuPy array to do backprojection, i.e.
        # (host -> device (filtering) -> device (backprojection) -> host (if needed))
        cp.cuda.Device(gpu_id).use()
        
        # initiate a 3D ASTRA class object    
        Atools = AstraTools3D(DetectorsDimH=data.shape[2],  # DetectorsDimH # detector dimension (horizontal)
                                DetectorsDimV=data.shape[1],  # DetectorsDimV # detector dimension (vertical) for 3D case only
                                AnglesVec=-angles,  # the vector of angles in radians
                                CenterRotOffset=data.shape[2] / 2 - center - 0.5,  # The center of rotation combined with the shift offsets                              
                                ObjSize=objsize,  # a scalar to define the reconstructed object dimensions
                                OS_number = 1, # OS recon disabled
                                device_projector = 'gpu',
                                GPUdevice_index=gpu_id)    
        # ------------------------------------------------------- # 
        data = _filtersinc3D_cupy(cp.swapaxes(data, 0, 1)) # filter the data on the GPU and keep the result there        
        
        reconstruction = Atools.backprojCuPy(data) # backproject the filtered data while keeping data on the GPU
        cp._default_memory_pool.free_all_blocks()
        # ------------------------------------------------------- #
    else:
        raise ValueError("Unknown algorithm type, please specify FBP3D_device or FBP3D_host")
    return reconstruction

def _filtersinc3D_cupy(projection3D):
    """applies a filter to 3D projection data

    Args:
        projection3D (ndarray): projection data must be a CuPy array.

    Returns:
        ndarray: a CuPy array of filtered projection data.
    """
    a = 1.1
    [DetectorsLengthV, projectionsNum, DetectorsLengthH] = cp.shape(projection3D)
    w = cp.linspace(-cp.pi,cp.pi-(2*cp.pi)/DetectorsLengthH, DetectorsLengthH,dtype='float32')
    
    # prepearing a ramp-like filter to apply to every projection
    rn1 = cp.abs(2.0/a*cp.sin(a*w/2.0))
    rn2 = cp.sin(a*w/2.0)
    rd = (a*w)/2.0
    rd_c = cp.zeros([1,DetectorsLengthH])
    rd_c[0,:] = rd
    r = rn1*(cp.dot(rn2, cp.linalg.pinv(rd_c)))**2
    multiplier = (1.0/projectionsNum)
    f = cp.fft.fftshift(r)
    filter_2d = cp.zeros((DetectorsLengthV,DetectorsLengthH), dtype='float32') # 2d filter
    filter_2d[0::,:] = f
    
    filtered = cp.zeros(cp.shape(projection3D), dtype='float32') # convert to cupy array

    for i in range(0,projectionsNum):
        IMG = cp.fft.fft2(projection3D[:,i,:])
        fimg = IMG*filter_2d
        filtered[:,i,:] = cp.real(cp.fft.ifft2(fimg))

    del projection3D
    cp._default_memory_pool.free_all_blocks()
    return multiplier*filtered
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##


## %%%%%%%%%%%%%%%%%%%%%%% Tomopy/ASTRA reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%  ##
@nvtx.annotate()
def reconstruct_tomopy(
    data: np.ndarray,
    angles: np.ndarray,
    center: float = None,
    algorithm: str = 'FBP_CUDA',
    gpu_id : int = 0
    ) -> np.ndarray:
    """
    Perform reconstruction using tomopy with wrappers around ASTRA toolbox.
    This is a 3D recon using 2D (slice-by-slice) astra geometry routines.

    Parameters
    ----------
    data : np.ndarray
        Projection data as a numpy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    algorithm : str, optional
        The name of the reconstruction method, see available ASTRA methods.
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    np.ndarray
        The reconstructed volume.
    """
    from tomopy import astra, recon
    
    reconstruction = recon(cp.asnumpy(data),
                           theta=angles,
                           center=center,
                           algorithm=astra,
                           options={
                               "method": algorithm,
                               "proj_type": "cuda",
                               "gpu_list": [gpu_id],},
                           ncore=1,)   
    
    return reconstruction
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
