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
from cupyx.scipy.ndimage import median_filter

__all__ = [
    'remove_stripe_based_sorting_cupy',
    'remove_stripes_titarenko_cupy',    
]
# TODO: port 'remove_all_stripe', 'remove_large_stripe' and 'remove_dead_stripe'
# from https://github.com/tomopy/tomopy/blob/master/source/tomopy/prep/stripe.py

## %%%%%%%%%%%%%%%%%remove_stripe_based_sorting_cupy%%%%%%%%%%%%%%%%%%%%%  ##
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
    
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##

## %%%%%%%%%%%%%%%%%%%remove_stripes_titarenko_cupy%%%%%%%%%%%%%%%%%%%%%%%  ##
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
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##


## %%%%%%%%%%%%%%%%%%%%%%%%detect_stripes%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def detect_stripes(data: ndarray,
                   search_window_dims: tuple = (1,9,1),
                   vert_window_size: float = 5,
                   gradient_gap: int = 2,
                   ncore: int = 1) -> ndarray: 
    """module to detect stripes in sinograms (2D) OR projection data (3D). 
    1. Taking first derrivative of the input in the direction orthogonal to stripes.
    2. Slide horizontal rectangular window orthogonal to stripes direction to accenuate outliers (stripes) using median.
    3. Slide the vertical thin (1 pixel) window to calculate a mean (further accenuates stripes).

    Args:
        data (ndarray): sinogram (2D) [angles x detectorsX] OR projection data (3D) [angles x detectorsY x detectorsX]
        search_window_dims (tuple, optional): searching rectangular window for weights calculation, 
        of a size (detectors_window_height, detectors_window_width, angles_window_depth). Defaults to (1,9,1).
        vert_window_size (float, optional): the half size of the vertical 1D window to calculate mean. Given in percents relative to the size of the angle dimension. Defaults to 5.
        gradient_gap (int, optional):  the gap in pixels with the neighbour while calculating a gradient (1 is the normal gradient). Defaults to 2.
        ncore (int, optional): _description_. Defaults to 1.

    Returns:
        ndarray: The associated weights (needed for thresholding)
    """       
    from larix.methods.misc import STRIPES_DETECT

    # calculate weights for stripes
    (stripe_weights, stats_vec) = STRIPES_DETECT(data, search_window_dims, vert_window_size, gradient_gap, ncore)
       
    return stripe_weights
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##

## %%%%%%%%%%%%%%%%%%%%%%%%merge_stripes%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def merge_stripes(data: ndarray,
                   stripe_width_max_perc: float = 5,
                   mask_dilate: int = 2,
                   threshold_stripes: float = 0.1,
                   ncore: int = 1) -> ndarray:
    """module to threshold the obtained stripe weights in sinograms (2D) OR projection data (3D) and merge stripes that are close to each other. 
    
    Args:
        data (ndarray): weigths for sinogram (2D) [angles x detectorsX] OR projection data (3D) [angles x detectorsY x detectorsX]
        stripe_width_max_perc (float, optional):  the maximum width of stripes in the data, given in percents relative to the size of the DetectorX. Defaults to 5.
        mask_dilate (int, optional): the number of pixels/voxels to dilate the obtained mask. Defaults to 2.
        threshold_stripes (float, optional): Threshold the obtained weights to get a binary mask, larger vaules are more sensitive to stripes. Defaults to 0.1.
        ncore (int, optional): _description_. Defaults to 1.

    Returns:
        ndarray: mask_stripe
    """       
    from larix.methods.misc import STRIPES_MERGE
    
    gradientX = _gradient(data, 2)    
    med_val = np.median(np.abs(gradientX).flatten(), axis=0)
    
    # we get a local stats here, needs to be adopted for global stats
    mask_stripe = np.zeros_like(data,dtype="uint8")    
    mask_stripe[data > med_val/threshold_stripes] = 1
    
    # merge stripes that are close to each other
    mask_stripe_merged = STRIPES_MERGE(mask_stripe, stripe_width_max_perc, mask_dilate, ncore)
    
    return mask_stripe_merged


def _gradient(data, axis):
    return np.gradient(data, axis=axis)
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
