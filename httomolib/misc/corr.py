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

import cupy as cp
import numpy as np

__all__ = [    
    'median_filter3d_cupy',
    'remove_outlier3d_cupy',
    'inpainting_filter3d',
]


def median_filter3d_cupy(
    data: cp.ndarray,
    kernel_size: int = 3,
    dif: float = 0.0
) -> cp.ndarray:
    """
    Apply 3D median or dezinger (when dif>0) filter to a 3D array.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array either float32 or uint16 data type.
    kernel_size : int, optional
        The size of the filter's kernel.
    dif : float, optional
        Expected difference value between outlier value and the 
        median value of the array, leave equal to 0 for classical median.

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

    
    if data.ndim == 3:
        if 0 in data.shape:
            raise ValueError("The length of one of dimensions is equal to zero")
    else:
        raise ValueError("The input array must be a 3D array")

    out = cp.empty(data.shape, dtype=input_type, order="C")
    
    median_kernel = r'''
        template <typename Type, int diameter>
        __global__ void median_general_kernel(
            const Type* in, Type* out, float dif, int Z, int M, int N)
        {
            constexpr int radius = diameter / 2;
            constexpr int d3 = diameter*diameter*diameter;
            constexpr int midpoint = d3 / 2;

            Type ValVec[d3];
            const long i = blockDim.x * blockIdx.x + threadIdx.x;
            const long j = blockDim.y * blockIdx.y + threadIdx.y;
            const long k = blockDim.z * blockIdx.z + threadIdx.z;

            if (i >= N || j >= M || k >= Z)
                return; 
            
            int counter = 0;
            for(int i_m=-radius; i_m<=radius; i_m++)
            {
                int i1 = i + i_m;
                if ((i1 < 0) || (i1 >= N))
                    i1 = i;
                for(int j_m=-radius; j_m<=radius; j_m++)
                {
                    int j1 = j + j_m;
                    if ((j1 < 0) || (j1 >= M))
                        j1 = j;
                    for(int k_m=-radius; k_m<=radius; k_m++)
                    {
                        int k1 = k + k_m;
                        if ((k1 < 0) || (k1 >= Z))
                            k1 = k;
                        ValVec[counter] = in[i1 + N*j1 + N*M*k1];
                        counter++;
                    }
                }
            }

            /* do bubble sort here */
            for (int x = 0; x < d3 - 1; x++)
            {
                for(int y = 0; y < d3 - x - 1; y++)
                {
                    if (ValVec[y] > ValVec[y+1])
                    {
                        Type temp = ValVec[y];
                        ValVec[y] = ValVec[y+1];
                        ValVec[y+1] = temp;
                    }
                }
            }
            
            /* perform median filtration */
            if (dif == 0.0f)
                out[i + N*j + N*M*k] = ValVec[midpoint];
            else
            {
                /* perform dezingering */
                Type in_value = in[i + N*j + N*M*k];
                out[i + N*j + N*M*k] = fabsf(in_value - ValVec[midpoint]) >= dif ? ValVec[midpoint] : in_value;
            }
        
        }
    '''
    dz, dy, dx = data.shape
    # setting grid/block parameters
    block_x = 128
    block_y = 1
    block_z = 1
    block_dims = (block_x, block_y, block_z)
    grid_x = (dx + block_x - 1) // block_x
    grid_y = dy
    grid_z = dz
    grid_dims = (grid_x, grid_y, grid_z) 

    params = (data, out, dif, dz, dy, dx, dx*dy*dz)

    if kernel_size in [3, 5, 7, 9, 11, 13]:
        kernel_args = "median_general_kernel<{0}, {1}>".format(
            "float" if input_type == "float32" else "unsigned short",
            kernel_size
        )
    else:
        raise ValueError("Please select a correct kernel size: 3, 5, 7, 9, 11, 13")

    module = cp.RawModule(
        code=median_kernel,
        options=('-std=c++11',),
        name_expressions=[kernel_args]
    )
    
    median3d = module.get_function(kernel_args)
    median3d(grid_dims, block_dims, params)

    return out


def remove_outlier3d_cupy(
    data: cp.ndarray,
    kernel_size: int = 3,
    dif: float = 0.1
) -> cp.ndarray:
    """
    Selectively applies 3D median filter to a 3D array to remove outliers. Also called a dezinger.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array either float32 or uint16 data type.
    kernel_size : int, optional
        The size of the filter's kernel.
    dif : float, optional
        Expected difference value between outlier value and the 
        median value of the array.

    Returns
    -------
    ndarray
        Dezingered filtered 3D CuPy array either float32 or uint16 data type.

    Raises
    ------
    ValueError
        If the input array is not three dimensional.
    """                        
    return median_filter3d_cupy(
        data=data,
        kernel_size=kernel_size,
        dif=dif
    )


def inpainting_filter3d(
    data: np.ndarray,
    mask: np.ndarray,
    iter: int = 3,
    windowsize_half: int = 5,
    method_type: str = "random",
    ncore: int = 1
) -> np.ndarray:
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
