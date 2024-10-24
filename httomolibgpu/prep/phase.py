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
    from httomolibgpu.cuda_kernels import load_cuda_module
    from cupyx.scipy.fft import fft2, ifft2, fftshift
else:
    load_cuda_module = Mock()
    fft2 = Mock()
    ifft2 = Mock()
    fftshift = Mock()

from numpy import float32
from typing import Tuple
import math

__all__ = [
    "paganin_filter_savu",
    "paganin_filter_tomopy",
]


## %%%%%%%%%%%%%%%%%%%%%%% paganin_filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
#: CuPy implementation of Paganin filter from Savu
def paganin_filter_savu(
    data: cp.ndarray,
    ratio: float = 250.0,
    energy: float = 53.0,
    distance: float = 1.0,
    resolution: float = 1.28,
    pad_y: int = 100,
    pad_x: int = 100,
    pad_method: str = "edge",
    increment: float = 0.0,
) -> cp.ndarray:
    """
    Apply Paganin filter (for denoising or contrast enhancement) to
    projections.

    Parameters
    ----------
    data : cp.ndarray
        The stack of projections to filter.

    ratio : float, optional
        Ratio of delta/beta.

    energy : float, optional
        Beam energy in keV.

    distance : float, optional
        Distance from sample to detector in metres.

    resolution : float, optional
        Pixel size in microns.

    pad_y : int, optional
        Pad the top and bottom of projections.

    pad_x : int, optional
        Pad the left and right of projections.

    pad_method : str, optional
        Numpy pad method to use.

    increment : float, optional
        Increment all values by this amount before taking the log.

    Returns
    -------
    cp.ndarray
        The stack of filtered projections.
    """
    # Check the input data is valid
    if data.ndim != 3:
        raise ValueError(
            f"Invalid number of dimensions in data: {data.ndim},"
            " please provide a stack of 2D projections."
        )

    # Setup various values for the filter
    _, height, width = data.shape
    micron = 1e-6
    keV = 1000.0
    energy *= keV
    resolution *= micron
    wavelength = (1240.0 / energy) * 1e-9

    height1 = height + 2 * pad_y
    width1 = width + 2 * pad_x

    # Define the paganin filter, taking into account the padding that will be
    # applied to the projections (if any)

    # Using raw kernel her as indexing is direct and it avoids a lot of temporaries
    # and tiny kernels
    module = load_cuda_module("paganin_filter_gen")
    kernel = module.get_function("paganin_filter_gen")

    # Apply padding to all the 2D projections
    # Note: this takes considerable time on GPU...
    data = cp.pad(data, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode=pad_method)

    # Define array to hold result, which will not have the padding applied to it
    precond_kernel_float = cp.ElementwiseKernel(
        "T data",
        "T out",
        """
        if (isnan(data)) {
            out = T(0);
        } else if (isinf(data)) {
            out = data < 0.0 ? -3.402823e38f : 3.402823e38f;  // FLT_MAX, not available in cupy
        } else if (data == 0.0) {
            out = 1.0;
        } else {
            out = data;
        }
        """,
        name="paganin_precond_float",
        no_return=True,
    )
    precond_kernel_int = cp.ElementwiseKernel(
        "T data",
        "T out",
        """out = data == 0 ? 1 : data""",
        name="paganin_precond_int",
        no_return=True,
    )

    if data.dtype in (cp.float32, cp.float64):
        precond_kernel_float(data, data)
    else:
        precond_kernel_int(data, data)

    # avoid normalising in both directions - we include multiplier in the post_kernel
    data = cp.asarray(data, dtype=cp.complex64)
    data = fft2(data, axes=(-2, -1), overwrite_x=True, norm="backward")

    # prepare filter here, while the GPU is busy with the FFT
    filtercomplex = cp.empty((height1, width1), dtype=cp.complex64)
    bx = 16
    by = 8
    gx = (width1 + bx - 1) // bx
    gy = (height1 + by - 1) // by
    kernel(
        grid=(gx, gy, 1),
        block=(bx, by, 1),
        args=(
            cp.int32(width1),
            cp.int32(height1),
            cp.float32(resolution),
            cp.float32(wavelength),
            cp.float32(distance),
            cp.float32(ratio),
            filtercomplex,
        ),
    )
    data *= filtercomplex

    data = ifft2(data, axes=(-2, -1), overwrite_x=True, norm="forward")

    post_kernel = cp.ElementwiseKernel(
        "C pci1, raw float32 increment, raw float32 ratio, raw float32 fft_scale",
        "T out",
        "out = -0.5 * ratio * log(abs(pci1) * fft_scale + increment)",
        name="paganin_post_proc",
        no_return=True,
    )
    fft_scale = 1.0 / (data.shape[1] * data.shape[2])
    res = cp.empty((data.shape[0], height, width), dtype=cp.float32)
    post_kernel(
        data[:, pad_y : pad_y + height, pad_x : pad_x + width],
        np.float32(increment),
        np.float32(ratio),
        np.float32(fft_scale),
        res,
    )
    return res


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


##-------------------------------------------------------------##
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
