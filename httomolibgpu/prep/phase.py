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

import math
from typing import Tuple
import cupy as cp
from cupy import float32
import cupyx
import numpy as np
import nvtx

from httomolibgpu.cuda_kernels import load_cuda_module
from httomolibgpu.decorator import method_proj

__all__ = [
    "fresnel_filter",
    "paganin_filter",
    "paganin_filter2",
    "retrieve_phase",
]

# Define constants used in phase retrieval method
BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


def _calc_max_slices_fresnel(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    available_memory: int,
    **kwargs,
) -> Tuple[int, np.dtype, Tuple[int, int]]:
    height1, width1 = non_slice_dims_shape
    window_size = (height1 * width1) * np.float64().nbytes
    pad_width = min(150, int(0.1 * width1))
    padded_height = height1 + 2 * pad_width
    padded_width = width1 * 2 * pad_width
    in_slice_size = height1 * width1 * dtype.itemsize
    # float64 here as res is internally computed in double
    internal_slice_size = padded_height * padded_width * np.float64().nbytes
    out_slice_size = padded_height * padded_width * np.float32().nbytes
    slice_size = in_slice_size + out_slice_size + internal_slice_size
    # the internal data is computed using a for loop, so the temporaries won't have
    # a significant impact. But we add a safety margin for that
    safety = in_slice_size * 4

    available_memory -= window_size + safety
    return (available_memory // slice_size, float32(), non_slice_dims_shape)


# CuPy implementation of Fresnel filter ported from Savu
@method_proj(_calc_max_slices_fresnel)
def fresnel_filter(
    mat: cp.ndarray, pattern: str, ratio: float, apply_log: bool = True
) -> cp.ndarray:
    """
    Apply Fresnel filter.

    Parameters
    ----------
    mat : cp.ndarray
        The data to apply filtering to.

    pattern : str
        Choose 'PROJECTION' for filtering projection, otherwise, will be handled
        generically for other cases.

    ratio : float
        Control the strength of the filter. Greater is stronger.

    apply_log : bool, optional
        Apply negative log function to the data being filtered.

    Returns
    -------
    cp.ndarray
        The filtered data.
    """

    if mat.ndim == 2:
        mat = cp.expand_dims(mat, 0)

    if mat.ndim != 3:
        raise ValueError(
            f"Invalid number of dimensions in data: {mat.ndim},"
            " please provide a stack of 2D projections."
        )

    if apply_log is True:
        mat = -cp.log(mat)

    # Define window
    (depth1, height1, width1) = mat.shape[:3]
    window = _make_window(height1, width1, ratio, pattern)
    pad_width = min(150, int(0.1 * width1))

    # Regardless of working with projections or sinograms, the rows and columns
    # in the images to filter are in the same dimensions of the data: rows in
    # dimension 1, columns in dimension 2 (ie, for projection images, `nrow` is
    # the number of rows in a projection image, and for sinogram images, `nrow`
    # is the number of rows in a sinogram image).
    (_, nrow, ncol) = mat.shape

    # Define array to hold result. Note that, due to the padding applied, the
    # shape of the filtered images are different to the shape of the
    # original/unfiltered images.
    padded_height = mat.shape[1] + pad_width * 2
    res_height = min(nrow, padded_height - pad_width)
    padded_width = mat.shape[2] + pad_width * 2
    res_width = min(ncol, padded_width - pad_width)
    res = cp.zeros((mat.shape[0], res_height, res_width))

    # Loop over images and apply filter
    for i in range(mat.shape[0]):
        if pattern == "PROJECTION":
            top_drop = 10  # To remove the time stamp in some data
            mat_pad = cp.pad(
                mat[i][top_drop:],
                ((pad_width + top_drop, pad_width), (pad_width, pad_width)),
                mode="edge",
            )
            win_pad = cp.pad(window, pad_width, mode="edge")
            mat_dec = cp.fft.ifft2(cp.fft.fft2(mat_pad) / cp.fft.ifftshift(win_pad))
            mat_dec = cp.real(
                mat_dec[pad_width : pad_width + nrow, pad_width : pad_width + ncol]
            )
            res[i] = mat_dec
        else:
            mat_pad = cp.pad(mat[i], ((0, 0), (pad_width, pad_width)), mode="edge")
            win_pad = cp.pad(window, ((0, 0), (pad_width, pad_width)), mode="edge")
            mat_fft = cp.fft.fftshift(cp.fft.fft(mat_pad), axes=1) / win_pad
            mat_dec = cp.fft.ifft(cp.fft.ifftshift(mat_fft, axes=1))
            mat_dec = cp.real(mat_dec[:, pad_width : pad_width + ncol])
            res[i] = mat_dec

    if apply_log is True:
        res = cp.exp(-res)

    return cp.asarray(res, dtype=cp.float32)


def _make_window(height, width, ratio, pattern):
    center_hei = int(cp.ceil((height - 1) * 0.5))
    center_wid = int(cp.ceil((width - 1) * 0.5))
    if pattern == "PROJECTION":
        ulist = (1.0 * cp.arange(0, width) - center_wid) / width
        vlist = (1.0 * cp.arange(0, height) - center_hei) / height
        u, v = cp.meshgrid(ulist, vlist)
        win2d = 1.0 + ratio * (u**2 + v**2)
    else:
        ulist = (1.0 * cp.arange(0, width) - center_wid) / width
        win1d = 1.0 + ratio * ulist**2
        win2d = cp.tile(win1d, (height, 1))

    return win2d


def _calc_max_slices_paganin_filter(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    available_memory: int,
    **kwargs,
) -> Tuple[int, np.dtype, Tuple[int, int]]:
    pad_x = kwargs["pad_x"]
    pad_y = kwargs["pad_y"]
    input_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    in_slice_size = (
        (non_slice_dims_shape[0] + 2 * pad_y)
        * (non_slice_dims_shape[1] + 2 * pad_x)
        * dtype.itemsize
    )
    # FFT needs complex inputs, so copy to complex happens first
    complex_slice = in_slice_size / dtype.itemsize * np.complex64().nbytes
    fftplan_slice = complex_slice
    filter_size = complex_slice
    res_slice = np.prod(non_slice_dims_shape) * np.float32().nbytes
    slice_size = input_size + in_slice_size + complex_slice + fftplan_slice + res_slice
    available_memory -= filter_size
    return (int(available_memory // slice_size), float32(), non_slice_dims_shape)


## %%%%%%%%%%%%%%%%%%%%%%% paganin_filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
#: CuPy implementation of Paganin filter from Savu
@method_proj(_calc_max_slices_paganin_filter)
@nvtx.annotate()
def paganin_filter(
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
    data = cupyx.scipy.fft.fft2(data, axes=(-2, -1), overwrite_x=True, norm="backward")

    # prepare filter here, while the GPU is busy with the FFT
    filtercomplex = cp.empty((height1, width1), dtype=np.complex64)
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

    data = cupyx.scipy.fft.ifft2(data, axes=(-2, -1), overwrite_x=True, norm="forward")

    post_kernel = cp.ElementwiseKernel(
        "C pci1, raw float32 increment, raw float32 ratio, raw float32 fft_scale",
        "T out",
        "out = -0.5 * ratio * log(abs(pci1) * fft_scale + increment)",
        name="paganin_post_proc",
        no_return=True,
    )
    fft_scale = 1.0 / (data.shape[1] * data.shape[2])
    res = cp.empty((data.shape[0], height, width), dtype=np.float32)
    post_kernel(
        data[:, pad_y : pad_y + height, pad_x : pad_x + width],
        np.float32(increment),
        np.float32(ratio),
        np.float32(fft_scale),
        res,
    )

    return res


def _calc_max_slice_retrieve_phase(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    available_memory: int,
    **kwargs,
) -> Tuple[int, np.dtype, Tuple[int, int]]:
    dy, dz = non_slice_dims_shape
    pixel_size = kwargs["pixel_size"]
    energy = kwargs["energy"]
    wavelength = _wavelength(energy)
    dist = kwargs["dist"]
    py = _calc_pad_width(dy, pixel_size, wavelength, dist)
    pz = _calc_pad_width(dz, pixel_size, wavelength, dist)
    # x 2 for ifftshift
    grid_size = (dy + 2 * py) * (dz + 2 * pz) * np.float64().nbytes * 2
    prj_size = (dy + 2 * py) * (dz + 2 * pz) * np.float32().nbytes * 2
    prj_complex_size = int(prj_size * 1.85)
    fftplan_size = prj_complex_size
    prj_ret_size = prj_complex_size
    # the code goes through the projections line by line,
    # which doesn't need any significant memory compared to the slices
    # it also updates the inputs in-place

    available_memory -= (
        grid_size + prj_complex_size + prj_size + fftplan_size + prj_ret_size
    )
    slice_memory = np.prod(non_slice_dims_shape) * dtype.itemsize
    return (available_memory // slice_memory, dtype, non_slice_dims_shape)


## %%%%%%%%%%%%%%%%%%%%%%% retrieve_phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
#: CuPy implementation of retrieve_phase from TomoPy
@method_proj(_calc_max_slice_retrieve_phase)
@nvtx.annotate()
def retrieve_phase(
    tomo: cp.ndarray,
    pixel_size: float = 1e-4,
    dist: float = 50.0,
    energy: float = 20.0,
    alpha: float = 1e-3,
    pad: bool = True,
) -> cp.ndarray:
    """
    Perform single-step phase retrieval from phase-contrast measurements
    :cite:`Paganin:02`.

    Parameters
    ----------
    tomo : cp.ndarray
        3D tomographic data.
    pixel_size : float, optional
        Detector pixel size in cm.
    dist : float, optional
        Propagation distance of the wavefront in cm.
    energy : float, optional
        Energy of incident wave in keV.
    alpha : float, optional
        Regularization parameter.
    pad : bool, optional
        If True, extend the size of the projections by padding with zeros.

    Returns
    -------
    cp.ndarray
        Approximated 3D tomographic phase data.
    """

    # Check the input data is valid
    if tomo.ndim == 2:
        tomo = cp.expand_dims(tomo, 0)

    if tomo.ndim != 3:
        raise ValueError(
            f"Invalid number of dimensions in data: {tomo.ndim},"
            " please provide a stack of 2D projections."
        )

    # New dimensions and pad value after padding.
    py, pz, val = _calc_pad(tomo, pixel_size, dist, energy, pad)

    # Compute the reciprocal grid.
    _, dy, dz = tomo.shape
    w2 = _reciprocal_grid(pixel_size, (dy + 2 * py, dz + 2 * pz))

    # Filter in Fourier space.
    phase_filter = cp.fft.fftshift(_paganin_filter_factor(energy, dist, alpha, w2))

    prj = cp.full((dy + 2 * py, dz + 2 * pz), val, dtype=cp.float32)

    # Apply phase retrieval
    return _retrieve_phase(tomo, phase_filter, py, pz, prj, pad)


def _retrieve_phase(
    tomo: cp.ndarray,
    phase_filter: cp.ndarray,
    px: int,
    py: int,
    prj: cp.ndarray,
    pad: bool,
) -> cp.ndarray:
    _, dy, dz = tomo.shape
    num_projs = tomo.shape[0]
    normalized_phase_filter = phase_filter / phase_filter.max()
    for m in range(num_projs):
        prj[px : dy + px, py : dz + py] = tomo[m]
        prj[:px] = prj[px]
        prj[-px:] = prj[-px - 1]
        prj[:, :py] = prj[:, py][:, cp.newaxis]
        prj[:, -py:] = prj[:, -py - 1][:, cp.newaxis]
        # TomoPy has its own 2D FFT implementations
        # https://github.com/tomopy/tomopy/blob/master/source/tomopy/util/misc.py,
        # the NumPy equivalent in CuPy has been used as an alternative
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.fft.fft2.html#.
        fproj = cp.fft.fft2(prj)
        fproj *= normalized_phase_filter
        proj = cp.real(cp.fft.ifft2(fproj))
        if pad:
            proj = proj[px : dy + px, py : dz + py]
        tomo[m] = proj
    return tomo


def _calc_pad(
    tomo: cp.ndarray, pixel_size: float, dist: float, energy: float, pad: bool
) -> tuple[int, int, float]:
    """
    Calculate new dimensions and pad value after padding.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    pixel_size : float
        Detector pixel size in cm.
    dist : float
        Propagation distance of the wavefront in cm.
    energy : float
        Energy of incident wave in keV.
    pad : bool
        If True, extend the size of the projections by padding with zeros.

    Returns
    -------
    int
        Pad amount in projection axis.
    int
        Pad amount in sinogram axis.
    float
        Pad value.
    """
    _, dy, dz = tomo.shape
    wavelength = _wavelength(energy)
    py, pz, val = 0, 0, 0.0
    if pad:
        val = _calc_pad_val(tomo)
        py = _calc_pad_width(dy, pixel_size, wavelength, dist)
        pz = _calc_pad_width(dz, pixel_size, wavelength, dist)
    return py, pz, val


def _wavelength(energy: float) -> float:
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def _paganin_filter_factor(
    energy: float, dist: float, alpha: float, w2: cp.ndarray
) -> cp.ndarray:
    return 1 / (_wavelength(energy) * dist * w2 / (4 * PI) + alpha)


def _calc_pad_width(dim: int, pixel_size: float, wavelength: float, dist: float) -> int:
    pad_pix = cp.ceil(PI * wavelength * dist / pixel_size**2)
    return int((pow(2, cp.ceil(cp.log2(dim + pad_pix))) - dim) * 0.5)


def _calc_pad_val(tomo: cp.ndarray) -> float:
    return cp.mean((tomo[..., 0] + tomo[..., -1]) * 0.5)


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
    rc *= 2 * PI / (n * pixel_size)
    return rc


# Adaptation with some corrections of retrieve_phase (Paganin filter) from TomoPy
@nvtx.annotate()
def paganin_filter2(
    tomo: cp.ndarray,
    pixel_size: float = 1e-4,
    dist: float = 50.0,
    energy: float = 53.0,
    alpha: float = 1e-3,
) -> cp.ndarray:
    """
    Perform single-material phase retrieval from flats/darks corrected tomographic measurements
    :cite:`Paganin:02`.

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

    dz_orig, dy_orig, dx_orig = cp.shape(tomo)

    # Perform padding to the power of 2 as FFT is O(n*log(n)) complexity
    # NOTE: Need to convert to float32 as FFT produces complex128 array from uint16
    # TODO: adding other options of padding?
    padded_tomo, pad_tup = _pad_projections_to_second_power(tomo)

    dz, dy, dx = cp.shape(padded_tomo)

    # 3D FFT of tomo data
    padded_tomo = cp.asarray(padded_tomo, dtype=cp.complex64)
    fft_tomo = cupyx.scipy.fft.fft2(padded_tomo, axes=(-2, -1), overwrite_x=True)

    # Compute the reciprocal grid.
    w2 = _reciprocal_grid(pixel_size, (dy, dx))

    # Build filter in the Fourier space.
    phase_filter = cupyx.scipy.fft.fftshift(_paganin_filter_factor2(energy, dist, alpha, w2))
    phase_filter = phase_filter / phase_filter.max()  # normalisation

    # Apply filter and take inverse FFT
    ifft_filtered_tomo = (
        cupyx.scipy.fft.ifft2(phase_filter * fft_tomo, axes=(-2, -1), overwrite_x=True)
    ).real

    # slicing indices for cropping
    slc_indices = (
        slice(pad_tup[0][0], pad_tup[0][0] + dz_orig, 1),
        slice(pad_tup[1][0], pad_tup[1][0] + dy_orig, 1),
        slice(pad_tup[2][0], pad_tup[2][0] + dx_orig, 1),
    )

    # crop the padded filtered data:
    tomo = ifft_filtered_tomo[slc_indices].astype(cp.float32)

    # taking the negative log
    c = 4 * PI / _wavelength(energy)
    _log_kernel = cp.ElementwiseKernel(
        "C tomo, raw float32 c",
        "C out",
        "out = -log(tomo) * 1/c",
        name="log_kernel",
    )

    return _log_kernel(tomo, c)


def _shift_bit_length(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pad_projections_to_second_power(tomo: cp.ndarray) -> tuple[cp.ndarray, tuple]:
    """
    Performs padding of each projection to the next power of 2.
    If the shape is not even we also care of that before padding.

    Parameters
    ----------
    tomo : cp.ndarray
        3d projection data

    Returns
    -------
    ndarray: padded 3d projection data
    tuple: a tuple with padding dimensions
    """
    full_shape_tomo = cp.shape(tomo)

    pad_tup = []
    for index, element in enumerate(full_shape_tomo):
        if index == 0:
            pad_width = (0, 0)  # do not pad the slicing dim
        else:
            diff = _shift_bit_length(element + 1) - element
            if element % 2 == 0:
                pad_width = diff // 2
                pad_width = (pad_width, pad_width)
            else:
                # need an uneven padding for odd-number lengths
                left_pad = diff // 2
                right_pad = diff - left_pad
                pad_width = (left_pad, right_pad)

        pad_tup.append(pad_width)

    padded_tomo = cp.pad(tomo, tuple(pad_tup), "edge")

    return padded_tomo, pad_tup


def _paganin_filter_factor2(energy, dist, alpha, w2):
    # Alpha represents the ratio of delta/beta.
    return 1 / (1 + (dist * alpha * _wavelength(energy) * w2 / (4 * PI)))
