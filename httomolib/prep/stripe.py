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
from typing import Union

import cupy as cp
import numpy as np
import nvtx

__all__ = [
    "remove_stripe_based_sorting",
    "remove_stripe_ti",
]

# TODO: port 'remove_all_stripe', 'remove_large_stripe' and 'remove_dead_stripe'
# from https://github.com/tomopy/tomopy/blob/master/source/tomopy/prep/stripe.py


@nvtx.annotate()
def remove_stripe_based_sorting(
    data: Union[cp.ndarray, np.ndarray],
    size: int = 11,
    dim: int = 1,
) -> Union[cp.ndarray, np.ndarray]:
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

    References
    ----------
    .. [1] https://doi.org/10.1364/OE.26.028396
    """

    if size is None:
        if data.shape[2] > 2000:
            size = 21
        else:
            size = max(5, int(0.01 * data.shape[2]))

    for m in range(data.shape[1]):
        data[:, m, :] = _rs_sort(data[:, m, :], size, dim)

    return data


@nvtx.annotate()
def _rs_sort(sinogram, size, dim):
    """
    Remove stripes using the sorting technique.
    """
    xp = cp.get_array_module(sinogram)
    sinogram = xp.transpose(sinogram)

    #: Sort each column of the sinogram by its grayscale values
    #: Keep track of the sorting indices so we can reverse it below
    sortvals = xp.argsort(sinogram, axis=1)
    sortvals_reverse = xp.argsort(sortvals, axis=1)
    sino_sort = xp.take_along_axis(sinogram, sortvals, axis=1)

    #: Now apply the median filter on the sorted image along each row
    if xp.__name__ == "cupy":
        from cupyx.scipy.ndimage import median_filter
    else:
        from scipy.ndimage import median_filter

    sino_sort = median_filter(sino_sort, (size, 1) if dim == 1 else (size, size))

    #: step 3: re-sort the smoothed image columns to the original rows
    sino_corrected = xp.take_along_axis(sino_sort, sortvals_reverse, axis=1)

    return xp.transpose(sino_corrected)


@nvtx.annotate()
def remove_stripe_ti(
    data: Union[cp.ndarray, np.ndarray],
    beta: float = 0.1,
) -> Union[cp.ndarray, np.ndarray]:
    """
    Removes stripes with the method of V. Titarenko (TomoCuPy implementation)

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
    xp = cp.get_array_module(data)
    gamma = beta * ((1 - beta) / (1 + beta)) ** xp.abs(
        xp.fft.fftfreq(data.shape[-1]) * data.shape[-1]
    )
    gamma[0] -= 1
    v = xp.mean(data, axis=0)
    v = v - v[:, 0:1]
    v = xp.fft.irfft(xp.fft.rfft(v) * xp.fft.rfft(gamma)).astype(data.dtype)
    data[:] += v

    return data
