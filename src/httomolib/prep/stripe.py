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
from cupy import abs, mean, ndarray
from cupyx.scipy.ndimage import median_filter, binary_dilation, uniform_filter1d
from ._rgi import interpn

__all__ = [
    'remove_stripe_based_sorting_cupy',
    'remove_stripes_titarenko_cupy',
]

## %%%%%%%%%%%%%%%%%%%%% remove_all_stripe_cupy %%%%%%%%%%%%%%%%%%%%%%%%%  ##
## Naive CuPy port of the NumPy implementation in TomoPy
def remove_all_stripe_cupy(tomo: ndarray, snr: float=3, la_size: int=61,
                           sm_size: int=21, dim: int=1):
    """
    Remove all types of stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (combination of algorithm 3,4,5, and 6).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.

    snr  : float
        Ratio used to locate large stripes.
        Greater is less sensitive.

    la_size : int
        Window size of the median filter to remove large stripes.

    sm_size : int
        Window size of the median filter to remove small-to-medium stripes.

    dim : {1, 2}, optional
        Dimension of the window.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        sino = _rs_dead(sino, snr, la_size, matindex)
        sino = _rs_sort(sino, sm_size, matindex, dim)
        tomo[:, m, :] = sino
    return tomo


## %%%%%%%%%%%%%%%%%%%%% remove_large_stripe_cupy %%%%%%%%%%%%%%%%%%%%%%%%%  ##
## Naive CuPy port of the NumPy implementation in TomoPy
def remove_large_stripe_cupy(tomo: ndarray, snr: float=3, size: int=51,
                             drop_ratio: float=0.1, norm: bool=True) -> ndarray:
    """
    Remove large stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (algorithm 5).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.

    snr  : float, optional
        Ratio used to locate of large stripes.
        Greater is less sensitive.

    size : int, optional
        Window size of the median filter.

    drop_ratio : float, optional
        Ratio of pixels to be dropped, which is used to reduce the false
        detection of stripes.

    norm : bool, optional
        Apply normalization if True.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_large(sino, snr, size, matindex, drop_ratio, norm)

    return tomo


def _rs_large(sinogram: ndarray, snr: float, size: int, matindex: ndarray,
              drop_ratio: float=0.1, norm: bool=True) -> ndarray:
    drop_ratio = cp.clip(cp.asarray(drop_ratio, dtype=cp.float32), 0.0, 0.8)
    (nrow, _) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    # Note: CuPy's docs
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.sort.html refer
    # to the default option of `kind=None` being a stable algorithm. NumPy docs
    # https://numpy.org/doc/stable/reference/generated/numpy.sort.html#numpy.sort
    # on the other hand when given the default value `kind=None` uses quicksort.
    sinosort = cp.sort(sinogram, axis=0)
    sinosmooth = median_filter(sinosort, (1, size))
    list1 = mean(sinosort[ndrop:nrow - ndrop], axis=0)
    list2 = mean(sinosmooth[ndrop:nrow - ndrop], axis=0)
    # TODO: Using the `out` parameter in conjunction with the `where` parameter
    # could decrease memory usage via avoiding creating new array objects;
    # however, something isn't quite working with the value being passed for the
    # `where` parameter, requires a bit of investigation.
    listfact = cp.divide(list1, list2,
                         #out=cp.ones_like(list1),
                         #where=list2 != 0
                         )

    # Locate stripes
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    matfact = cp.tile(listfact, (nrow, 1))

    # Normalize
    if norm is True:
        sinogram = sinogram / matfact
    sinogram1 = cp.transpose(sinogram)
    matcombine = cp.asarray(cp.dstack((matindex, sinogram1)))
    matsort = cp.asarray(
        [row[row[:, 1].argsort()] for row in matcombine])
    matsort[:, :, 1] = cp.transpose(sinosmooth)
    matsortback = cp.asarray(
        [row[row[:, 0].argsort()] for row in matsort])
    sino_corrected = cp.transpose(matsortback[:, :, 1])
    # TODO: Taking into account NumPy docs for `np.where()`
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html#numpy.where
    # and CuPy docs for `cp.where()`
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.where.html?highlight=where,
    # a possibly better alternative is:
    # listxmiss = cp.asarray(listmask > 0.0).nonzero()[0]
    listxmiss = cp.where(listmask > 0.0)[0]
    sinogram[:, listxmiss] = sino_corrected[:, listxmiss]
    return sinogram


def _detect_stripe(listdata: ndarray, snr: float) -> ndarray:
    numdata = len(listdata)
    listsorted = cp.sort(listdata)[::-1]
    xlist = cp.arange(0, numdata, 1.0)
    ndrop = cp.int16(0.25 * numdata)
    (_slope, _intercept) = cp.polyfit(
        xlist[ndrop:-ndrop - 1], listsorted[ndrop:-ndrop - 1], 1)
    numt1 = _intercept + _slope * xlist[-1]
    noiselevel = abs(numt1 - _intercept)
    noiselevel = cp.clip(cp.asarray(noiselevel, dtype=cp.float32), 1e-6, None)
    val1 = abs(listsorted[0] - _intercept) / noiselevel
    val2 = abs(listsorted[-1] - numt1) / noiselevel
    listmask = cp.zeros_like(listdata)
    if (val1 >= snr):
        upper_thresh = _intercept + noiselevel * snr * 0.5
        listmask[listdata > upper_thresh] = 1.0
    if (val2 >= snr):
        lower_thresh = numt1 - noiselevel * snr * 0.5
        listmask[listdata <= lower_thresh] = 1.0
    return listmask


## %%%%%%%%%%%%%%%%%%%%% remove_dead_stripe_cupy %%%%%%%%%%%%%%%%%%%%%%%%%  ##
## Naive CuPy port of the NumPy implementation in TomoPy
def remove_dead_stripe_cupy(tomo: ndarray, snr: float=3, size: int=51,
                            norm: bool=True) -> ndarray:
    """
    Remove unresponsive and fluctuating stripe artifacts from sinogram using
    Nghia Vo's approach :cite:`Vo:18` (algorithm 6).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.

    snr  : float
        Ratio used to detect locations of large stripes.
        Greater is less sensitive.

    size : int
        Window size of the median filter.

    norm : bool, optional
        Remove residual stripes if True.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_dead(sino, snr, size, matindex, norm)
    return tomo


def _rs_dead(sinogram, snr, size, matindex, norm=True):
    """
    Remove unresponsive and fluctuating stripes.
    """
    (nrow, _) = sinogram.shape
    sinosmooth = cp.apply_along_axis(uniform_filter1d, 0, sinogram, 10)
    listdiff = cp.sum(abs(sinogram - sinosmooth), axis=0)
    listdiffbck = median_filter(listdiff, size)
    # TODO: Same situation as the analagous part in `_rs_large()`, see that
    # function's comment when using `cp.divide()`.
    listfact = cp.divide(listdiff, listdiffbck,
                         #out=np.ones_like(listdiff),
                         #where=listdiffbck != 0
                         )
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    listmask[0:2] = 0.0
    listmask[-2:] = 0.0
    # TODO: Same situation as the analagous part in `_rs_large()`, see that
    # function's comment when using `cp.where()`.
    listx = cp.where(listmask < 1.0)[0]
    listy = cp.arange(nrow)
    matz = sinogram[:, listx]
    points = (listy, listx)
    # Uses N-dimensional interpolation function copied from CuPy source code of
    # v12.0.0b3 pre-release
    finter = interpn(points, matz, tuple(cp.meshgrid(listy, listx)),
                     method='linear')
    # TODO: Same situation as the analagous part in `_rs_large()`, see that
    # function's comment when using `cp.where()`.
    listxmiss = cp.where(listmask > 0.0)[0]
    if len(listxmiss) > 0:
        sinogram[:, listxmiss] = finter(listxmiss, listy)
    # Remove residual stripes
    if norm is True:
        sinogram = _rs_large(sinogram, snr, size, matindex)
    return sinogram


## %%%%%%%%%%%%%%%%% remove_stripe_based_sorting_cupy %%%%%%%%%%%%%%%%%%%%%  ##
## Naive CuPy port of the NumPy implementation in TomoPy
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