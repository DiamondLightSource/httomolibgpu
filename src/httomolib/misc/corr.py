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
# Created By  : Daniil Kazantsev <scientificsoftware@diamond.ac.uk>
# Created Date: 21/October/2022
# version ='0.1'
# ---------------------------------------------------------------------------
""" Module for data correction """

import numpy as np
import cupy as cp
from numpy import ndarray

__all__ = [
    'median_filter3d_cupy',
    'inpainting_filter3d',
]


def median_filter3d_cupy(data: ndarray,
                         size: int = 3) -> ndarray:
    """
    Apply 3D median filter to a 3D array.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array either float32 or uint16 data type.
    size : int, optional
        The size of the filter's kernel.

    Returns
    -------
    ndarray
        Median filtered 3D CuPy array either float32 or uint16 data type.

    Raises
    ------
    ValueError
        If the input array is not three dimensional.

    """
    input_type = data.dtype
    if input_type not in ["float32", "uint16"]:
        raise ValueError("The input data should be either float32 or uint16 data type")
    out = cp.zeros(data.shape, dtype=input_type, order="C")

    # convert the full kernel size (odd int) to a half size as the C function requires it
    kernel_half_size = (max(int(size), 3) - 1) // 2

    if data.ndim == 3:
        dz, dy, dx = data.shape
        if (dz == 0) or (dy == 0) or (dx == 0):
            raise ValueError("The length of one of dimensions is equal to zero")
    else:
        raise ValueError("The input array must be a 3D array")
    
    #TODO: The code bellow needs further work in terms of:
    # 1. Templating for different kernel sizes as in here https://github.com/dkazanc/larix/blob/06f3952504dc97b84ec0688b6ab8dbdf1b6bdfc1/src/Core/GPU_modules/MedianFilt_GPU_core.cu#L101
    # 2. Adding the template for uint18 data type
    loaded_from_source = r'''
        extern "C" {
        inline __device__ void sort_bubble(float *x, int n_size)
        {
            for (int i = 0; i < n_size - 1; i++)
            {
                for(int j = 0; j < n_size - i - 1; j++)
                {
                    if (x[j] > x[j+1])
                    {
                        float temp = x[j];
                        x[j] = x[j+1];
                        x[j+1] = temp;
                    }
                }
            }
        }               
        __global__ void medianfilter_float32(const float* in, float* out, int N, int M, int Z, long num_total) 
        {
            
            int radius = 1; 
            int diameter = 3; 
            int midpoint = 13;
                        
            float ValVec[27];
            long i1, j1, k1, i_m, j_m, k_m, counter;
            int i3, j3;
            
            const long i = blockDim.x * blockIdx.x + threadIdx.x;
            const long j = blockDim.y * blockIdx.y + threadIdx.y;
            const long k = blockDim.z * blockIdx.z + threadIdx.z;
            
            const unsigned long long index = (unsigned long long)i + (unsigned long long)N*(unsigned long long)j + (unsigned long long)N*(unsigned long long)M*(unsigned long long)k;          

            if (index < num_total && i < N && j < M && k < Z)     
            {            
                counter = 0l;
                for(i_m=-radius; i_m<=radius; i_m++) {
                        i1 = i + i_m;
                        if ((i1 < 0) || (i1 >= N)) i1 = i;
                        for(j_m=-radius; j_m<=radius; j_m++) {
                        j1 = j + j_m;
                        if ((j1 < 0) || (j1 >= M)) j1 = j;
                            for(k_m=-radius; k_m<=radius; k_m++) {
                            k1 = k + k_m;
                            if ((k1 < 0) || (k1 >= Z)) k1 = k;
                            ValVec[counter] = in[i1 + N*j1 + N*M*k1];
                            counter++;
                    }}}
                sort_bubble(ValVec, diameter*diameter*diameter);
            out[index] = ValVec[midpoint];
            }
        }
        }'''          
    module = cp.RawModule(code=loaded_from_source)
    median_filter = module.get_function('medianfilter_float32')
    
    # setting grid/block parameters
    # TODO: check that this is correct bellow as the output is not correct
    blockdim = 8
    block = (blockdim, blockdim, blockdim)
    grid = (dz//blockdim, dy//blockdim, dx//blockdim)    
    params = (cp.asarray((data), order="C"), out, dz, dy, dx, dx*dy*dz)
        
    if (input_type == 'float32'):    
        print("medianfilter_float_kernel")
        median_filter(grid, block, params)
    else:
        print("medianfilter_uint16_kernel")
        # TODO: median filter kernel to work with uint16
    return out

def inpainting_filter3d(
    data: ndarray,
    mask: ndarray,
    iter: int = 3,
    windowsize_half: int = 5,
    method_type: str = "random",
    ncore: int = 1
) -> ndarray:
    """
    Inpainting filter for 3D data, taken from the Larix toolbox
    (C - implementation).

    A morphological inpainting scheme which progresses from the
    edge of the mask inwards. It acts like a diffusion-type process
    but significantly faster in convergence.

    Parameters
    ----------
    data : ndarray
        Input array.
    mask : ndarray
        Input binary mask (uint8) the same size as data,
        integer 1 will define the inpainting area.
    iter : int, optional
        An additional number of iterations to run after the region
        has been inpainted (smoothing effect).
    windowsize_half : int, optional
        Half-window size of the searching window (neighbourhood window).
    method_type : str, optional
        Method type to select for a value in the neighbourhood: mean, median,
        or random. Defaults to "random".
    ncore : int, optional
        The number of CPU cores to use.

    Returns
    -------
    ndarray
        Inpainted array.
    """

    from larix.methods.misc import INPAINT_EUCL_WEIGHTED
  
    return INPAINT_EUCL_WEIGHTED(data, mask, iter, windowsize_half, method_type, ncore)
