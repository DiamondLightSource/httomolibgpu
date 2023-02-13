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
"""Modules for stripes removal"""
import cupy as cp
import numpy as np
import nvtx
from cupy import abs, mean, ndarray
from cupyx.scipy.ndimage import median_filter

__all__ = [
    'remove_stripe_based_sorting_cupy',
    'remove_stripes_titarenko_cupy',
]

# TODO: port 'remove_all_stripe', 'remove_large_stripe' and 'remove_dead_stripe'
# from https://github.com/tomopy/tomopy/blob/master/source/tomopy/prep/stripe.py


## %%%%%%%%%%%%%%%%% remove_stripe_based_sorting_cupy %%%%%%%%%%%%%%%%%%%%%  ##
## Naive CuPy port of the NumPy implementation in TomoPy
@nvtx.annotate()
def remove_stripe_based_sorting_cupy(
    data: ndarray,
    size: int = 11,
    dim: int = 1,
    gpu_id : int = 0
) -> ndarray:
    """
    Remove full and partial stripe artifacts from sinogram using Nghia Vo's
    approach, algorithm 3 in Ref. [1]. Angular direction is along the axis 0.
    This algorithm works particularly well for removing partial stripes.

    Steps of the algorithm:
    1. Sort each column of the sinogram by its grayscale values.
    2. Apply a smoothing (median) filter on the sorted image along each row.
    3. Re-sort the smoothed image columns to the original rows to
       get the corrected sinogram.

    Parameters
    ----------
    data : cupy.ndarray
        3D tomographic data.
    size : int, optional
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.
    gpu_id : int, optional
        A GPU device index to perform operation on.      

    Returns
    -------
    cupy.ndarray
        Corrected 3D tomographic data.

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """
    cp.cuda.Device(gpu_id).use()

    matindex = _create_matindex(data.shape[2], data.shape[0])
    if size is None:
        if data.shape[2] > 2000:
            size = 21
        else:
            size = max(5, int(0.01 * data.shape[2]))

    for m in range(data.shape[1]):
        sino = data[:, m, :]
        data[:, m, :] = _rs_sort(sino, size, matindex, dim)

    return data


def _create_matindex(nrow, ncol):
    """
    Create a 2D array of indexes used for the sorting technique.
    """
    listindex = cp.arange(0.0, ncol, 1.0)
    matindex = cp.tile(listindex, (nrow, 1))
    return matindex


def _rs_sort(sinogram, size, matindex, dim):
    """
    Remove stripes using the sorting technique.
    """
    sinogram = cp.transpose(sinogram)
    matcomb = cp.asarray(cp.dstack((matindex, sinogram)))

    #: Sort each column of the sinogram by its grayscale values
    matsort = cp.asarray(
        [row[row[:, 1].argsort()] for row in matcomb])

    #: Now apply the median filter on the sorted image along each row
    if dim == 1:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, 1))
    else:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, size))

    #: step 3: re-sort the smoothed image columns to the original rows
    matsortback = cp.asarray(
        [row[row[:, 0].argsort()] for row in matsort])

    sino_corrected = matsortback[:, :, 1]

    return cp.transpose(sino_corrected)


## %%%%%%%%%%%%%%%%%%% remove_stripes_titarenko_cupy %%%%%%%%%%%%%%%%%%%%%%%  ##
@nvtx.annotate()
def remove_stripes_titarenko_cupy(
    data: ndarray,
    beta: float = 0.1,
    gpu_id : int = 0
) -> np.ndarray:
    """
    Removes stripes with the method of V. Titarenko (TomoCuPy implementation)

    Parameters
    ----------
    data : ndarray
        3D stack of projections as a CuPy array.
    beta : float, optional
        filter parameter, lower values increase the filter strength.
        Default is 0.1.
    gpu_id : int, optional
        A GPU device index to perform operation on.          

    Returns
    -------
    ndarray
        3D CuPy array of de-striped projections.
    """
    cp.cuda.Device(gpu_id).use()

    gamma = beta * ((1 - beta) / (1 + beta)) ** abs(
        cp.fft.fftfreq(data.shape[-1]) * data.shape[-1]
    )
    gamma[0] -= 1
    v = mean(data, axis=0)
    v = v - v[:, 0:1]
    v = cp.fft.irfft(cp.fft.rfft(v) * cp.fft.rfft(gamma))
    data[:] += v

    return data
