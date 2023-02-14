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
import cupyx
import numpy as np
import nvtx

__all__ = [
    'reconstruct_tomobar',
    'reconstruct_tomopy',
]

## %%%%%%%%%%%%%%%%%%%%%%% ToMoBAR reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
@nvtx.annotate()
def reconstruct_tomobar(
    data : cp.ndarray,
    angles : np.ndarray,
    center : float = None,
    objsize : int = None,
    algorithm : str = 'FBP3D_device',
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
                
        reconstruction = RectoolsDIR.FBP(np.swapaxes(data, 0, 1)) # the output stored as a numpy array
        
        # ------------------------------------------------------- #     
    elif algorithm == "FBP3D_device":
        # Perform filtering of the data on the GPU and then pass a pointer to CuPy array to do backprojection, i.e.
        # (host -> device (filtering) -> device (backprojection) -> host (if needed))
        
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
        data = _filtersinc3D_cupy(data) # filter the data on the GPU and keep the result there
        # the Astra toolbox requires C-contiguous arrays, and swapaxes seems to return a sliced view which 
        # is neither C nor F contiguous. 
        # So we have to make it C-contiguous first
        data = cp.ascontiguousarray(cp.swapaxes(data, 0, 1))
        reconstruction = Atools.backprojCuPy(data) # backproject the filtered data while keeping data on the GPU
        cp._default_memory_pool.free_all_blocks()
        # ------------------------------------------------------- #
    else:
        raise ValueError("Unknown algorithm type, please specify FBP3D_device or FBP3D_host")
    return reconstruction

@nvtx.annotate()
def _filtersinc3D_cupy(projection3D):
    """applies a filter to 3D projection data

    Args:
        projection3D (ndarray): projection data must be a CuPy array.

    Returns:
        ndarray: a CuPy array of filtered projection data.
    """
    
    # prepearing a ramp-like filter to apply to every projection
    filter_prep = cp.RawKernel(
        r"""
        #define MY_PI 3.1415926535897932384626433832795f

        extern "C" __global__ 
        void generate_filtersinc(float a, float* f, int n, float multiplier) {
            int tid = threadIdx.x;    // using only one block

            float dw = 2 * MY_PI / n;

            extern __shared__ char smem_raw[];
            float* smem = reinterpret_cast<float*>(smem_raw);

            // from: cp.linalg.pinv(rd_c)
            // pseudo-inverse of vector is x/sum(x**2),
            // so we need to compute sum(x**2) in shared memory
            float sum = 0.0;
            for (int i = tid; i < n; i += blockDim.x) {
                float w = -MY_PI  + i * dw;
                float rn2 = a * w / 2.0f;
                sum += rn2 * rn2;
            }
            
            smem[tid] = sum;
            __syncthreads();
            int nt = blockDim.x;
            int c = nt;
            while (c > 1) {
                int half = c / 2;
                if (tid < half) {
                    smem[tid] += smem[c - tid - 1];
                }
                __syncthreads();
                c = c - half;
            }
            float sum_aw2_sqr = smem[0];

            // cp.dot(rn2, cp.linalg.pinv(rd_c))**2
            // now we can calclate the dot product, preparing summing in shared memory
            float dot_partial = 0.0;
            for (int i = tid; i < n; i += blockDim.x) {
                float w = -MY_PI  + i * dw;
                float rd = a*w/2.0f;
                float rn2 = sin(rd);
                
                dot_partial += rn2 * rd / sum_aw2_sqr;
            }
                
            // now reduce dot_partial to full dot-product result
            smem[tid] = dot_partial;
            __syncthreads();
            c = nt;
            while (c > 1) {
                int half = c / 2;
                if (tid < half) {
                    smem[tid] += smem[c - tid - 1];
                }
                __syncthreads();
                c = c - half;
            }
            float dotprod_sqr = smem[0] * smem[0];

            // now compute actual result
            for (int i = tid; i < n; i += blockDim.x) {
                float w = -MY_PI  + i * dw;
                float rd = a*w/2.0f;
                float rn2 = sin(rd);
                float rn1 = abs(2.0/a*rn2);
                float r = rn1 * dotprod_sqr;

                
                // write to ifftshifted positions
                int shift = n / 2;
                int outidx = (i + shift) % n;

                // apply multiplier here - which does FFT scaling too
                f[outidx] = r * multiplier;
            }
        }
        """, "generate_filtersinc"
    )


    # since the fft is complex-to-complex, it makes a copy of the real input array anyway,
    # so we do that copy here explicitly, and then do everything in-place
    projection3D = projection3D.astype(cp.complex64)
    projection3D = cupyx.scipy.fft.fft2(projection3D, axes=(1, 2), overwrite_x=True, norm="backward")
    
    # generating the filter here so we can schedule/allocate while FFT is keeping the GPU busy
    a = 1.1
    (projectionsNum, DetectorsLengthV, DetectorsLengthH) = cp.shape(projection3D)
    f = cp.empty((1,1,DetectorsLengthH), dtype=np.float32)
    bx = 256
    # because FFT is linear, we can apply the FFT scaling + multiplier in the filter
    multiplier = 1.0/projectionsNum/DetectorsLengthV/DetectorsLengthH
    filter_prep(grid=(1, 1, 1), block=(bx, 1, 1), 
                args=(cp.float32(a), f, np.int32(DetectorsLengthH), np.float32(multiplier)),
                shared_mem=bx*4)
    # actual filtering
    projection3D *= f
    
    # avoid normalising here - we have included that in the filter
    return cp.real(cupyx.scipy.fft.ifft2(projection3D, axes=(1, 2), overwrite_x=True, norm="forward"))
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##


## %%%%%%%%%%%%%%%%%%%%%%% Tomopy/ASTRA reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%  ##
@nvtx.annotate()
def reconstruct_tomopy(
    data : np.ndarray,
    angles : np.ndarray,
    center : float = None,
    algorithm : str = 'FBP_CUDA',
    proj_type : str = "cuda",
    gpu_id : int = 0,
    ncore : int = 1
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
    proj_type : str, optional
        Define projector type, e.g., "cuda" for "FBP_CUDA" algorithm 
        or "linear" for "FBP" (CPU) algorithm, see more available ASTRA projectors.
    gpu_id : int, optional
        A GPU device index to perform operation on.
    ncore : int, optional
        Number of cores that will be assigned to jobs.

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
                               "proj_type": proj_type,
                               "gpu_list": [gpu_id],},
                           ncore=ncore,)
    
    return reconstruction
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
