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
"""Modules for stripes removal using CuPy API"""
import cupy as cp
from cupy import abs, mean, ndarray
from cupyx.scipy.ndimage import median_filter


def remove_stripes_titarenko_cupy(data: ndarray,
                                  beta: float = 0.1) -> ndarray:
    """
    Removes stripes with the method of V. Titarenko (TomoCuPy implementation)

    Parameters
    ----------
    arr : ndarray
        3D stack of projections as a CuPy array.
    beta : float, optional
        filter parameter, lower values increase the filter strength.

    Returns
    -------
    ndarray
        3D CuPy array of de-striped projections.
    """
    gamma = beta * ((1 - beta) / (1 + beta)) ** abs(
        cp.fft.fftfreq(data.shape[-1]) * data.shape[-1]
    )
    gamma[0] -= 1
    v = mean(data, axis=0)
    v = v - v[:, 0:1]
    v = cp.fft.irfft(cp.fft.rfft(v) * cp.fft.rfft(gamma))
    data[:] += v

    return data


# Naive CuPy port of the NumPy implementation in TomoPy
def remove_stripe_based_sorting_cupy(tomo: ndarray,
                                     size: int = None,
                                     dim: int = 1) -> ndarray:
    """
    Remove full and partial stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (algorithm 3).
    Suitable for removing partial stripes.

    Parameters
    ----------
    tomo : cupy.ndarray
        3D tomographic data.
    size : int
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.

    Returns
    -------
    cupy.ndarray
        Corrected 3D tomographic data.
    """
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    if size is None:
        if tomo.shape[2] > 2000:
            size = 21
        else:
            size = max(5, int(0.01 * tomo.shape[2]))
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_sort(sino, size, matindex, dim)
    return tomo


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
    matsort = cp.asarray(
        [row[row[:, 1].argsort()] for row in matcomb])
    if dim == 1:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, 1))
    else:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, size))
    matsortback = cp.asarray(
        [row[row[:, 0].argsort()] for row in matsort])
    sino_corrected = matsortback[:, :, 1]
    return cp.transpose(sino_corrected)
