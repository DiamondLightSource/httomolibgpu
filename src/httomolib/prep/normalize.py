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
"""Modules for raw projection data normalization"""

import cupy as cp
from cupy import float32, log, mean, ndarray

__all__ = [
    'normalize_cupy',
    'normalize_raw_cuda',
]


def normalize_raw_cuda(
    data: ndarray,
    flats: ndarray,
    darks: ndarray,
    cutoff: float = 10.
) -> ndarray:
    """
    Normalize raw projection data using the flat and dark field projections.
    This is a raw CUDA kernel implementation with CuPy wrappers.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    flats : cp.ndarray
        3D flat field data as a CuPy array.
    darks : cp.ndarray
        3D dark field data as a CuPy array.
    cutoff : float, optional
        Permitted maximum value for the normalised data.

    Returns
    -------
    cp.ndarray
        Normalised 3D tomographic data as a CuPy array.
    """
    if data.ndim:
        raise ValueError("Input data must be a 3D stack of projections")
    dark0 = mean(darks, axis=0, dtype=float32)
    flat0 = mean(flats, axis=0, dtype=float32)
    out = cp.zeros(data.shape, dtype=float32)

    norm_kernel = cp.RawKernel(
        """extern "C" __global__ void normalize(const unsigned short* data,
           const float* flat,
           const float* dark,
           float* out, float eps, float cutoff, int A, int B)
           {
             int bid = blockIdx.x;
             int tx = threadIdx.x;
             int ty = threadIdx.y;
             data += bid * A * B;
             out += bid * A * B;
             
             for (int a = ty; a < A; a += blockDim.y)
    	     {
 	     #pragma unroll(4)
	     for (int b = tx; b < B; b += blockDim.x)
	        {
                float denom = flat[a * B + b] - dark[a * B + b];
                if (denom < eps)
                {
                  denom = eps;
                }
                float tmp = (float(data[a * B + b]) - dark[a * B + b]) / denom;
                if (tmp > cutoff)
                {
                  tmp = cutoff;
                }
                if (tmp <= 0.0)
                {
                  tmp = eps;
                }
	        out[a * B + b] = -log(tmp);
    	        }
	     }
           }""", "normalize")

    grids = (32, 32, 1)
    blocks = (data.shape[0], 1, 1)
    params = (data, flat0, dark0, out, float32(1e-6),
              float32(cutoff), data.shape[1], data.shape[2])
    norm_kernel(grids, blocks, params)

    return out


#: CuPy implementation with higher memory footprint than normalize_raw_cuda.
def normalize_cupy(
    data: ndarray,
    flats: ndarray,
    darks: ndarray,
    cutoff: float = 10.0,
    minus_log: bool = False
) -> ndarray:
    """
    Normalize raw projection data using the flat and dark field projections.    

    Parameters
    ----------
    data : ndarray
        3D stack of projections as a CuPy array.
    flats : ndarray
        2D or 3D flat field data as a CuPy array.
    darks : ndarray
        2D or 3D dark field data as a CuPy array.
    cutoff : float, optional
        Permitted maximum value for the normalised data.
    minus_log : bool, optional
        Apply negative log to the normalised data

    Returns
    -------
    ndarray
        Normalised 3D tomographic data as a CuPy array.
    """
    data = cp.asarray(data, dtype=cp.float32)
    eps = cp.float32(1e-6)

    if data.ndim != 3:
        raise ValueError("Input data must be a 3D stack of projections")

    if flats.ndim not in (2, 3):
        raise ValueError("Input flats must be 2D or 3D data only")

    if darks.ndim not in (2, 3):
        raise ValueError("Input darks must be 2D or 3D data only")

    if flats.ndim == 2:
        flats = flats[cp.newaxis, :, :]
    if darks.ndim == 2:
        darks = darks[cp.newaxis, :, :]

    darks = mean(darks, axis=0, dtype=float32)
    flats = mean(flats, axis=0, dtype=float32)

    # replicates tomopy implementation
    denom = (flats - darks)
    denom[denom < eps] = eps
    data = (data - darks) / denom
    data[data > cutoff] = cutoff

    return -log(data) if minus_log else data
