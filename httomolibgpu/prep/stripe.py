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
# ---------------------------------------------------------------------------
"""Modules for stripes removal"""

import numpy as np
from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from cupyx.scipy.ndimage import median_filter, binary_dilation, uniform_filter1d
else:
    median_filter = Mock()
    binary_dilation = Mock()
    uniform_filter1d = Mock()

from typing import Union

__all__ = [
    "remove_stripe_based_sorting",
    "remove_stripe_ti",
    "remove_all_stripe",
]


def remove_stripe_based_sorting(
    data: Union[cp.ndarray, np.ndarray],
    size: int = 11,
    dim: int = 1,
) -> Union[cp.ndarray, np.ndarray]:
    """
    Remove full and partial stripe artifacts from sinogram using Nghia Vo's
    approach, see :cite:`vo2018superior`. This algorithm works particularly
    well for removing partial stripes.

    Steps of the algorithm: 1. Sort each column of the sinogram by its grayscale values.
    2. Apply a smoothing (median) filter on the sorted image along each row. 3. Re-sort the smoothed image columns to the original rows to
    get the corrected sinogram.

    Parameters
    ----------
    data : ndarray
        3D tomographic data as a CuPy or NumPy array.
    size : int, optional
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data as a CuPy or NumPy array.

    """
    if size is None:
        if data.shape[2] > 2000:
            size = 21
        else:
            size = max(5, int(0.01 * data.shape[2]))

    for m in range(data.shape[1]):
        data[:, m, :] = _rs_sort(data[:, m, :], size, dim)

    return data


def _rs_sort(sinogram, size, dim):
    """
    Remove stripes using the sorting technique.
    """
    sinogram = cp.transpose(sinogram)

    #: Sort each column of the sinogram by its grayscale values
    #: Keep track of the sorting indices so we can reverse it below
    sortvals = cp.argsort(sinogram, axis=1)
    sortvals_reverse = cp.argsort(sortvals, axis=1)
    sino_sort = cp.take_along_axis(sinogram, sortvals, axis=1)

    #: Now apply the median filter on the sorted image along each row
    sino_sort = median_filter(sino_sort, (size, 1) if dim == 1 else (size, size))

    #: step 3: re-sort the smoothed image columns to the original rows
    sino_corrected = cp.take_along_axis(sino_sort, sortvals_reverse, axis=1)

    return cp.transpose(sino_corrected)


def remove_stripe_ti(
    data: Union[cp.ndarray, np.ndarray],
    beta: float = 0.1,
) -> Union[cp.ndarray, np.ndarray]:
    """
    Removes stripes with the method of V. Titarenko (TomoCuPy implementation).
    See :cite:`titarenko2010analytical`.

    Parameters
    ----------
    data : ndarray
        3D stack of projections as a CuPy array.
    beta : float, optional
        filter parameter, lower values increase the filter strength.
        Default is 0.1.

    Returns
    -------
    ndarray
        3D array of de-striped projections.
    """
    # TODO: detector dimensions must be even otherwise error
    gamma = beta * ((1 - beta) / (1 + beta)) ** cp.abs(
        cp.fft.fftfreq(data.shape[-1]) * data.shape[-1]
    )
    gamma[0] -= 1
    v = cp.mean(data, axis=0)
    v = v - v[:, 0:1]
    v = cp.fft.irfft(cp.fft.rfft(v) * cp.fft.rfft(gamma)).astype(data.dtype)
    data[:] += v
    return data


######## Optimized version for Vo-all ring removal in tomopy########
# This function is taken from TomoCuPy package
# *************************************************************************** #
#                  Copyright © 2022, UChicago Argonne, LLC                    #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy                             #
#                     By: Argonne National Laboratory                         #
#                                                                             #
#                           OPEN SOURCE LICENSE                               #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
#    this list of conditions and the following disclaimer.                    #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
#                                                                             #
# *************************************************************************** #
def remove_all_stripe(
    data: cp.ndarray,
    snr: float = 3.0,
    la_size: int = 61,
    sm_size: int = 21,
    dim: int = 1,
) -> cp.ndarray:
    """
    Remove all types of stripe artifacts from sinogram using Nghia Vo's
    approach, see :cite:`vo2018superior` (combination of algorithm 3,4,5, and 6).

    Parameters
    ----------
    data : ndarray
        3D tomographic data as a CuPy array.
    snr  : float, optional
        Ratio used to locate large stripes.
        Greater is less sensitive.
    la_size : int, optional
        Window size of the median filter to remove large stripes.
    sm_size : int, optional
        Window size of the median filter to remove small-to-medium stripes.
    dim : {1, 2}, optional
        Dimension of the window.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data as a CuPy or NumPy array.

    """
    matindex = _create_matindex(data.shape[2], data.shape[0])
    for m in range(data.shape[1]):
        sino = data[:, m, :]
        sino = _rs_dead(sino, snr, la_size, matindex)
        sino = _rs_sort2(sino, sm_size, matindex, dim)
        sino = cp.nan_to_num(sino)
        data[:, m, :] = sino
    return data


def _rs_sort2(sinogram, size, matindex, dim):
    """
    Remove stripes using the sorting technique.
    """
    sinogram = cp.transpose(sinogram)
    matcomb = cp.asarray(cp.dstack((matindex, sinogram)))

    # matsort = cp.asarray([row[row[:, 1].argsort()] for row in matcomb])
    ids = cp.argsort(matcomb[:, :, 1], axis=1)
    matsort = matcomb.copy()
    matsort[:, :, 0] = cp.take_along_axis(matsort[:, :, 0], ids, axis=1)
    matsort[:, :, 1] = cp.take_along_axis(matsort[:, :, 1], ids, axis=1)
    if dim == 1:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, 1))
    else:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, size))

    # matsortback = cp.asarray([row[row[:, 0].argsort()] for row in matsort])

    ids = cp.argsort(matsort[:, :, 0], axis=1)
    matsortback = matsort.copy()
    matsortback[:, :, 0] = cp.take_along_axis(matsortback[:, :, 0], ids, axis=1)
    matsortback[:, :, 1] = cp.take_along_axis(matsortback[:, :, 1], ids, axis=1)

    sino_corrected = matsortback[:, :, 1]
    return cp.transpose(sino_corrected)


def _mpolyfit(x, y):
    n = len(x)
    x_mean = cp.mean(x)
    y_mean = cp.mean(y)

    Sxy = cp.sum(x * y) - n * x_mean * y_mean
    Sxx = cp.sum(x * x) - n * x_mean * x_mean

    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _detect_stripe(listdata, snr):
    """
    Algorithm 4 in :cite:`Vo:18`. Used to locate stripes.
    """

    numdata = len(listdata)
    listsorted = cp.sort(listdata)[::-1]
    xlist = cp.arange(0, numdata, 1.0)
    ndrop = cp.int16(0.25 * numdata)
    # (_slope, _intercept) = cp.polyfit(xlist[ndrop:-ndrop - 1],
    #   listsorted[ndrop:-ndrop - 1], 1)
    (_slope, _intercept) = _mpolyfit(
        xlist[ndrop : -ndrop - 1], listsorted[ndrop : -ndrop - 1]
    )

    numt1 = _intercept + _slope * xlist[-1]
    noiselevel = cp.abs(numt1 - _intercept)
    noiselevel = cp.clip(noiselevel, 1e-6, None)
    val1 = cp.abs(listsorted[0] - _intercept) / noiselevel
    val2 = cp.abs(listsorted[-1] - numt1) / noiselevel
    listmask = cp.zeros_like(listdata)
    if val1 >= snr:
        upper_thresh = _intercept + noiselevel * snr * 0.5
        listmask[listdata > upper_thresh] = 1.0
    if val2 >= snr:
        lower_thresh = numt1 - noiselevel * snr * 0.5
        listmask[listdata <= lower_thresh] = 1.0
    return listmask


def _rs_large(sinogram, snr, size, matindex, drop_ratio=0.1, norm=True):
    """
    Remove large stripes.
    """
    drop_ratio = max(min(drop_ratio, 0.8), 0)  # = cp.clip(drop_ratio, 0.0, 0.8)
    (nrow, ncol) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    sinosort = cp.sort(sinogram, axis=0)
    sinosmooth = median_filter(sinosort, (1, size))
    list1 = cp.mean(sinosort[ndrop : nrow - ndrop], axis=0)
    list2 = cp.mean(sinosmooth[ndrop : nrow - ndrop], axis=0)
    # listfact = cp.divide(list1,
    #                      list2,
    #                      out=cp.ones_like(list1),
    #                      where=list2 != 0)

    listfact = list1 / list2

    # Locate stripes
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    matfact = cp.tile(listfact, (nrow, 1))
    # Normalize
    if norm is True:
        sinogram = sinogram / matfact
    sinogram1 = cp.transpose(sinogram)
    matcombine = cp.asarray(cp.dstack((matindex, sinogram1)))

    # matsort = cp.asarray([row[row[:, 1].argsort()] for row in matcombine])
    ids = cp.argsort(matcombine[:, :, 1], axis=1)
    matsort = matcombine.copy()
    matsort[:, :, 0] = cp.take_along_axis(matsort[:, :, 0], ids, axis=1)
    matsort[:, :, 1] = cp.take_along_axis(matsort[:, :, 1], ids, axis=1)

    matsort[:, :, 1] = cp.transpose(sinosmooth)
    # matsortback = cp.asarray([row[row[:, 0].argsort()] for row in matsort])
    ids = cp.argsort(matsort[:, :, 0], axis=1)
    matsortback = matsort.copy()
    matsortback[:, :, 0] = cp.take_along_axis(matsortback[:, :, 0], ids, axis=1)
    matsortback[:, :, 1] = cp.take_along_axis(matsortback[:, :, 1], ids, axis=1)

    sino_corrected = cp.transpose(matsortback[:, :, 1])
    listxmiss = cp.where(listmask > 0.0)[0]
    sinogram[:, listxmiss] = sino_corrected[:, listxmiss]
    return sinogram


def _rs_dead(sinogram, snr, size, matindex, norm=True):
    """
    Remove unresponsive and fluctuating stripes.
    """
    sinogram = cp.copy(sinogram)  # Make it mutable
    (nrow, _) = sinogram.shape
    # sinosmooth = cp.apply_along_axis(uniform_filter1d, 0, sinogram, 10)
    sinosmooth = uniform_filter1d(sinogram, 10, axis=0)

    listdiff = cp.sum(cp.abs(sinogram - sinosmooth), axis=0)
    listdiffbck = median_filter(listdiff, size)

    listfact = listdiff / listdiffbck

    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    listmask[0:2] = 0.0
    listmask[-2:] = 0.0
    listx = cp.where(listmask < 1.0)[0]
    listy = cp.arange(nrow)
    matz = sinogram[:, listx]

    listxmiss = cp.where(listmask > 0.0)[0]

    # finter = interpolate.interp2d(listx.get(), listy.get(), matz.get(), kind='linear')
    if len(listxmiss) > 0:
        # sinogram_c[:, listxmiss.get()] = finter(listxmiss.get(), listy.get())
        ids = cp.searchsorted(listx, listxmiss)
        sinogram[:, listxmiss] = matz[:, ids - 1] + (listxmiss - listx[ids - 1]) * (
            matz[:, ids] - matz[:, ids - 1]
        ) / (listx[ids] - listx[ids - 1])

    # Remove residual stripes
    if norm is True:
        sinogram = _rs_large(sinogram, snr, size, matindex)
    return sinogram


def _create_matindex(nrow, ncol):
    """
    Create a 2D array of indexes used for the sorting technique.
    """
    listindex = cp.arange(0.0, ncol, 1.0)
    matindex = cp.tile(listindex, (nrow, 1))
    return matindex.astype(np.float32)
