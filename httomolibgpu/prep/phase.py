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
"""Modules for phase retrieval and phase-contrast enhancement. For more detailed information, see :ref:`phase_contrast_module`."""

import numpy as np
from httomolibgpu import cupywrapper
from httomolibgpu.memory_estimator_helpers import _DeviceMemStack

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from cupyx.scipy.fft import fft2, ifft2, fftshift
    from cupyx.scipy.fftpack import get_fft_plan
else:
    fft2 = Mock()
    ifft2 = Mock()
    fftshift = Mock()

from numpy import float32
from typing import Optional, Tuple
import math

__all__ = [
    "paganin_filter",
    "paganin_filter_savu_legacy",
]


# This implementation originated from the TomoPy version. It has been modified to conform
# different unit standards and also control of the filter driven by 'delta/beta' ratio
# as opposed to 'alpha' in the TomoPy's implementation.
def paganin_filter(
    tomo: cp.ndarray,
    pixel_size: float = 1.28,
    distance: float = 1.0,
    energy: float = 53.0,
    ratio_delta_beta: float = 250,
    calc_peak_gpu_mem: bool = False,
) -> cp.ndarray:
    """
    Perform single-material phase retrieval from flats/darks corrected tomographic measurements. For more detailed information, see :ref:`phase_contrast_module`.
    Also see :cite:`Paganin02` and :cite:`paganin2020boosting` for references.

    Parameters
    ----------
    tomo : cp.ndarray
        3D array of f/d corrected tomographic projections.
    pixel_size : float
        Detector pixel size (resolution) in micron units.
    distance : float
        Propagation distance of the wavefront from sample to detector in metre units.
    energy : float
        Beam energy in keV.
    ratio_delta_beta : float
        The ratio of delta/beta, where delta is the phase shift and real part of the complex material refractive index and beta is the absorption.
    calc_peak_gpu_mem: bool
        Parameter to support memory estimation in HTTomo. Irrelevant to the method itself and can be ignored by user.

    Returns
    -------
    cp.ndarray
        The 3D array of Paganin phase-filtered projection images.
    """
    mem_stack = _DeviceMemStack() if calc_peak_gpu_mem else None
    # Check the input data is valid
    if not mem_stack and tomo.ndim != 3:
        raise ValueError(
            f"Invalid number of dimensions in data: {tomo.ndim},"
            " please provide a stack of 2D projections."
        )
    if mem_stack:
        mem_stack.malloc(np.prod(tomo) * np.float32().itemsize)
    dz_orig, dy_orig, dx_orig = tomo.shape if not mem_stack else tomo

    # Perform padding to the power of 2 as FFT is O(n*log(n)) complexity
    # TODO: adding other options of padding?
    padded_tomo, pad_tup = _pad_projections_to_second_power(tomo, mem_stack)

    dz, dy, dx = padded_tomo.shape if not mem_stack else padded_tomo

    # 3D FFT of tomo data
    if mem_stack:
        mem_stack.malloc(np.prod(padded_tomo) * np.complex64().itemsize)
        mem_stack.free(np.prod(padded_tomo) * np.float32().itemsize)
        fft_input = cp.empty(padded_tomo, dtype=cp.complex64)
    else:
        padded_tomo = cp.asarray(padded_tomo, dtype=cp.complex64)
        fft_input = padded_tomo

    fft_plan = get_fft_plan(fft_input, axes=(-2, -1))
    if mem_stack:
        mem_stack.malloc(fft_plan.work_area.mem.size)
        mem_stack.free(fft_plan.work_area.mem.size)
    else:
        with fft_plan:
            fft_tomo = fft2(padded_tomo, axes=(-2, -1), overwrite_x=True)
        del padded_tomo
    del fft_input
    del fft_plan

    # calculate alpha constant
    alpha = _calculate_alpha(energy, distance / 1e-6, ratio_delta_beta)

    # Compute the reciprocal grid
    indx = _reciprocal_coord(pixel_size, dy)
    indy = _reciprocal_coord(pixel_size, dx)

    if mem_stack:
        mem_stack.malloc(indx.size * indx.dtype.itemsize)  # cp.asarray(indx)
        mem_stack.malloc(indx.size * indx.dtype.itemsize)  # cp.square
        mem_stack.free(indx.size * indx.dtype.itemsize)  # cp.asarray(indx)
        mem_stack.malloc(indy.size * indy.dtype.itemsize)  # cp.asarray(indy)
        mem_stack.malloc(indy.size * indy.dtype.itemsize)  # cp.square
        mem_stack.free(indy.size * indy.dtype.itemsize)  # cp.asarray(indy)

        mem_stack.malloc(indx.size * indy.size * indx.dtype.itemsize)  # cp.add.outer
        mem_stack.free(indx.size * indx.dtype.itemsize)  # cp.square
        mem_stack.free(indy.size * indy.dtype.itemsize)  # cp.square
        mem_stack.malloc(indx.size * indy.size * indx.dtype.itemsize)  # phase_filter
        mem_stack.free(indx.size * indy.size * indx.dtype.itemsize)  # cp.add.outer
        mem_stack.free(indx.size * indy.size * indx.dtype.itemsize)  # phase_filter

    else:
        # Build Lorentzian-type filter
        phase_filter = fftshift(
            1.0
            / (
                1.0
                + alpha
                * (
                    cp.add.outer(
                        cp.square(cp.asarray(indx)), cp.square(cp.asarray(indy))
                    )
                )
            )
        )

        phase_filter = phase_filter / phase_filter.max()  # normalisation

        # Filter projections
        fft_tomo *= phase_filter
        del phase_filter

    # Apply filter and take inverse FFT
    ifft_input = (
        fft_tomo if not mem_stack else cp.empty(padded_tomo, dtype=cp.complex64)
    )
    ifft_plan = get_fft_plan(ifft_input, axes=(-2, -1))
    if mem_stack:
        mem_stack.malloc(ifft_plan.work_area.mem.size)
        mem_stack.free(ifft_plan.work_area.mem.size)
    else:
        with ifft_plan:
            ifft_filtered_tomo = ifft2(fft_tomo, axes=(-2, -1), overwrite_x=True).real
        del fft_tomo
    del ifft_plan
    del ifft_input

    # slicing indices for cropping
    slc_indices = (
        slice(pad_tup[0][0], pad_tup[0][0] + dz_orig, 1),
        slice(pad_tup[1][0], pad_tup[1][0] + dy_orig, 1),
        slice(pad_tup[2][0], pad_tup[2][0] + dx_orig, 1),
    )

    if mem_stack:
        mem_stack.malloc(np.prod(tomo) * np.float32().itemsize)  # astype(cp.float32)
        mem_stack.free(
            np.prod(padded_tomo) * np.complex64().itemsize
        )  # ifft_filtered_tomo
        mem_stack.malloc(
            np.prod(tomo) * np.float32().itemsize
        )  # return _log_kernel(tomo)
        return mem_stack.highwater

    # crop the padded filtered data:
    tomo = ifft_filtered_tomo[slc_indices].astype(cp.float32)
    del ifft_filtered_tomo

    # taking the negative log
    _log_kernel = cp.ElementwiseKernel(
        "C tomo",
        "C out",
        "out = -log(tomo)",
        name="log_kernel",
    )

    return _log_kernel(tomo)


def _calculate_alpha(energy, distance_micron, ratio_delta_beta):
    return (
        _wavelength_micron(energy) * distance_micron / (4 * math.pi)
    ) * ratio_delta_beta


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
    tomo: cp.ndarray, mem_stack: Optional[_DeviceMemStack]
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
    full_shape_tomo = cp.shape(tomo) if not mem_stack else tomo

    pad_list = _calculate_pad_size(full_shape_tomo)

    if mem_stack:
        padded_tomo = [
            sh + pad[0] + pad[1] for sh, pad in zip(full_shape_tomo, pad_list)
        ]
        mem_stack.malloc(np.prod(padded_tomo) * np.float32().itemsize)
    else:
        padded_tomo = cp.pad(tomo, tuple(pad_list), "edge")

    return padded_tomo, tuple(pad_list)


def _wavelength_micron(energy: float) -> float:
    SPEED_OF_LIGHT = 299792458e2 * 10000.0  # [microns/s]
    PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
    return 2 * math.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def _reciprocal_coord(pixel_size: float, num_grid: int) -> np.ndarray:
    """
    Calculate reciprocal grid coordinates for a given pixel size
    and discretization.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in microns.
    num_grid : int
        Size of the reciprocal grid.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    n = num_grid - 1
    rc = np.arange(-n, num_grid, 2, dtype=cp.float32)
    rc *= 2 * math.pi / (n * pixel_size)
    return rc


def paganin_filter_savu_legacy(
    tomo: cp.ndarray,
    pixel_size: float = 1.28,
    distance: float = 1.0,
    energy: float = 53.0,
    ratio_delta_beta: float = 250,
    calc_peak_gpu_mem: bool = False,
) -> cp.ndarray:
    """
    Perform single-material phase retrieval from flats/darks corrected tomographic measurements. For more detailed information, see :ref:`phase_contrast_module`.
    Also see :cite:`Paganin02` and :cite:`paganin2020boosting` for references. The ratio_delta_beta parameter here follows implementation in Savu software.
    The module will be retired in future in favour of paganin_filter. One can rescale parameter ratio_delta_beta / 4 to achieve the same effect in paganin_filter.

    Parameters
    ----------
    tomo : cp.ndarray
        3D array of f/d corrected tomographic projections.
    pixel_size : float
        Detector pixel size (resolution) in micron units.
    distance : float
        Propagation distance of the wavefront from sample to detector in metre units.
    energy : float
        Beam energy in keV.
    ratio_delta_beta : float
        The ratio of delta/beta, where delta is the phase shift and real part of the complex material refractive index and beta is the absorption.
    calc_peak_gpu_mem: bool
        Parameter to support memory estimation in HTTomo. Irrelevant to the method itself and can be ignored by user.

    Returns
    -------
    cp.ndarray
        The 3D array of Paganin phase-filtered projection images.
    """

    return paganin_filter(
        tomo,
        pixel_size,
        distance,
        energy,
        ratio_delta_beta / 4,
        calc_peak_gpu_mem=calc_peak_gpu_mem,
    )
