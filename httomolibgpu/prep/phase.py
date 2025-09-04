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
"""Modules for phase retrieval and phase-contrast enhancement"""

import numpy as np
from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from cupyx.scipy.fft import fft2, ifft2, fftshift
else:
    fft2 = Mock()
    ifft2 = Mock()
    fftshift = Mock()

from numpy import float32
from typing import Tuple
import math

from httomolibgpu.misc.supp_func import data_checker

__all__ = [
    "paganin_filter_tomopy",
]

##-------------------------------------------------------------##
# Adaptation of retrieve_phase (Paganin filter) from TomoPy
def paganin_filter_tomopy(
    tomo: cp.ndarray,
    pixel_size: float = 1e-4,
    dist: float = 50.0,
    energy: float = 53.0,
    alpha: float = 1e-3,
) -> cp.ndarray:
    """
    Perform single-material phase retrieval from flats/darks corrected tomographic measurements. See
    :cite:`Paganin02` for a reference.

    Parameters
    ----------
    tomo : cp.ndarray
        3D array of f/d corrected tomographic projections.
    pixel_size : float, optional
        Detector pixel size in cm.
    dist : float, optional
        Propagation distance of the wavefront in cm.
    energy : float, optional
        Energy of incident wave in keV.
    alpha : float, optional
        Regularization parameter, the ratio of delta/beta. Larger values lead to more smoothing.

    Returns
    -------
    cp.ndarray
        The 3D array of Paganin phase-filtered projection images.
    """
    # Check the input data is valid
    if tomo.ndim != 3:
        raise ValueError(
            f"Invalid number of dimensions in data: {tomo.ndim},"
            " please provide a stack of 2D projections."
        )

    tomo = data_checker(tomo, verbosity=True, method_name="paganin_filter_tomopy")

    dz_orig, dy_orig, dx_orig = tomo.shape

    # Perform padding to the power of 2 as FFT is O(n*log(n)) complexity
    # TODO: adding other options of padding?
    padded_tomo, pad_tup = _pad_projections_to_second_power(tomo)

    dz, dy, dx = padded_tomo.shape

    # 3D FFT of tomo data
    padded_tomo = cp.asarray(padded_tomo, dtype=cp.complex64)
    fft_tomo = fft2(padded_tomo, axes=(-2, -1), overwrite_x=True)

    # Compute the reciprocal grid.
    w2 = _reciprocal_grid(pixel_size, (dy, dx))

    # Build filter in the Fourier space.
    phase_filter = fftshift(_paganin_filter_factor2(energy, dist, alpha, w2))
    phase_filter = phase_filter / phase_filter.max()  # normalisation

    # Filter projections
    fft_tomo *= phase_filter

    # Apply filter and take inverse FFT
    ifft_filtered_tomo = ifft2(fft_tomo, axes=(-2, -1), overwrite_x=True).real

    # slicing indices for cropping
    slc_indices = (
        slice(pad_tup[0][0], pad_tup[0][0] + dz_orig, 1),
        slice(pad_tup[1][0], pad_tup[1][0] + dy_orig, 1),
        slice(pad_tup[2][0], pad_tup[2][0] + dx_orig, 1),
    )

    # crop the padded filtered data:
    tomo = ifft_filtered_tomo[slc_indices].astype(cp.float32)

    # taking the negative log
    _log_kernel = cp.ElementwiseKernel(
        "C tomo",
        "C out",
        "out = -log(tomo)",
        name="log_kernel",
    )

    return _log_kernel(tomo)


def _shift_bit_length(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _calculate_pad_size(datashape: tuple) -> list:
    """Calculating the padding size

    Args:
        datashape (tuple): the shape of the 3D data

    Returns:
        list: the padded dimensions
    """
    pad_list = []
    for index, element in enumerate(datashape):
        if index == 0:
            pad_width = (0, 0)  # do not pad the slicing dim
        else:
            diff = _shift_bit_length(element + 1) - element
            if element % 2 == 0:
                pad_width_scalar = diff // 2
                pad_width = (pad_width_scalar, pad_width_scalar)
            else:
                # need an uneven padding for odd-number lengths
                left_pad = diff // 2
                right_pad = diff - left_pad
                pad_width = (left_pad, right_pad)

        pad_list.append(pad_width)

    return pad_list


def _pad_projections_to_second_power(
    tomo: cp.ndarray,
) -> Tuple[cp.ndarray, Tuple[int, int]]:
    """
    Performs padding of each projection to the next power of 2.
    If the shape is not even we also care of that before padding.

    Parameters
    ----------
    tomo : cp.ndarray
        3d projection data

    Returns
    -------
    Tuple consisting of:
    ndarray: padded 3d projection data
    tuple: a tuple with padding dimensions
    """
    full_shape_tomo = cp.shape(tomo)

    pad_list = _calculate_pad_size(full_shape_tomo)

    padded_tomo = cp.pad(tomo, tuple(pad_list), "edge")

    return padded_tomo, tuple(pad_list)


def _paganin_filter_factor2(energy, dist, alpha, w2):
    # Alpha represents the ratio of delta/beta.
    return 1 / (_wavelength(energy) * dist * w2 / (4 * math.pi) + alpha)


def _wavelength(energy: float) -> float:
    SPEED_OF_LIGHT = 299792458e2  # [cm/s]
    PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
    return 2 * math.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def _reciprocal_grid(pixel_size: float, shape_proj: tuple) -> cp.ndarray:
    """
    Calculate reciprocal grid.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    shape_proj : tuple
        Shape of the reciprocal grid along x and y axes.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    # Sampling in reciprocal space.
    indx = _reciprocal_coord(pixel_size, shape_proj[0])
    indy = _reciprocal_coord(pixel_size, shape_proj[1])
    indx_sq = cp.square(indx)
    indy_sq = cp.square(indy)

    return cp.add.outer(indx_sq, indy_sq)


def _reciprocal_coord(pixel_size: float, num_grid: int) -> cp.ndarray:
    """
    Calculate reciprocal grid coordinates for a given pixel size
    and discretization.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    num_grid : int
        Size of the reciprocal grid.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    n = num_grid - 1
    rc = cp.arange(-n, num_grid, 2, dtype=cp.float32)
    rc *= 2 * math.pi / (n * pixel_size)
    return rc