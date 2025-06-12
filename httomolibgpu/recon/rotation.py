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
"""Modules for finding the axis of rotation for 180 and 360 degrees scans"""

import numpy as np
from numpy.polynomial import Polynomial
from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from httomolibgpu.cuda_kernels import load_cuda_module
    from cupyx.scipy.ndimage import shift, gaussian_filter
    from skimage.registration import phase_cross_correlation
    from cupyx.scipy.fftpack import get_fft_plan
    from cupyx.scipy.fft import fft2, fftshift
else:
    load_cuda_module = Mock()
    shift = Mock()
    gaussian_filter = Mock()
    phase_cross_correlation = Mock()
    get_fft_plan = Mock()
    fft2 = Mock()
    fftshift = Mock()
    fft = Mock()
    rfft2 = Mock()

import math
from typing import List, Literal, Optional, Tuple, Union

from httomolibgpu.misc.supp_func import data_checker

__all__ = [
    "find_center_vo",
    "find_center_360",
    "find_center_pc",
]


def find_center_vo(
    data: cp.ndarray,
    ind: Optional[int] = None,
    average_radius: int = 0,
    cor_initialisation_value: Optional[float] = None,
    smin: int = -50,
    smax: int = 50,
    srad: float = 6.0,
    step: float = 0.5,
    ratio: float = 0.5,
    drop: int = 20,
) -> np.float32:
    """
    Find the rotation axis location (aka the centre of rotation) using Nghia Vo's method. See the paper
    :cite:`vo2014reliable`.

    Parameters
    ----------
    data : cp.ndarray
        3D [angles, detY, detX] tomographic data or a 2D [angles, detX] sinogram as a CuPy array.
    ind : int, optional
        Index of the slice to be used to estimate the CoR. If None is given, then the central sinogram will be extracted from the data array with a possible averaging, see .
    average_radius : int
        Averaging multiple sinograms around the ind-indexed sinogram to improve the signal-to-noise ratio. It is recommended to keep this parameter smaller than 10.
    cor_initialisation_value : float, optional
        The initial approximation for the centre of rotation. If the value is None, use the horizontal centre of the projection/sinogram image.
    smin : int
        Coarse search radius. Reference to the horizontal center of
        the sinogram.
    smax : int
        Coarse search radius. Reference to the horizontal center of
        the sinogram.
    srad : float
        Fine search radius.
    step : float
        Step of fine searching.
    ratio : float
        The ratio between the FOV of the camera and the size of object.
        It's used to generate the mask.
    drop : int
        Drop lines around vertical center of the mask.

    Returns
    -------
    float32
        Rotation axis location with a subpixel precision.
    """
    # if 2d sinogram is given it is extended into a 3D array along the vertical dimension
    if data.ndim == 2:
        data = cp.expand_dims(data, 1)
        ind = 0

    data = data_checker(data, verbosity=True, method_name="find_center_vo")

    angles_tot, detY_size, detX_size = data.shape

    if ind is None:
        ind = detY_size // 2  # middle slice index
        # averaging the data here to improve SNR
        if 2 * average_radius >= detY_size:
            # reduce the averaging radius
            average_radius = ind
        if ind > 0:
            _sino = cp.mean(
                data[:, ind - average_radius : ind + average_radius + 1, :], axis=1
            )
        else:
            _sino = data[:, ind, :]
    else:
        _sino = data[:, ind, :]

    if cor_initialisation_value is None:
        cor_initialisation_value = (detX_size - 1.0) / 2.0

    # downsampling ratios
    dsp_angle = 1
    dsp_detX = 1
    if detX_size > 2000:
        dsp_detX = 4
    if angles_tot > 2000:
        dsp_angle = 2

    start_cor = np.int16(np.floor(1.0 * (cor_initialisation_value + smin) / dsp_detX))
    stop_cor = np.int16(np.ceil(1.0 * (cor_initialisation_value + smax) / dsp_detX))
    fine_srange = max(srad, dsp_detX)
    off_set = 0.5 * dsp_detX if dsp_detX > 1 else 0.0

    # initiate denoising
    _sino_cs = gaussian_filter(_sino, (3, 1), mode="reflect")
    _sino_fs = gaussian_filter(_sino, (2, 2), mode="reflect")

    # Downsampling by averaging along a chosen dimension
    if dsp_angle > 1 or dsp_detX > 1:
        _sino_cs = _downsample(_sino_cs, dsp_angle, dsp_detX)

    init_cen = _search_coarse(_sino_cs, start_cor, stop_cor, ratio, drop)

    fine_cen = _search_fine(
        _sino_fs, fine_srange, step, float(init_cen) * dsp_detX + off_set, ratio, drop
    )
    cen_np = np.float32(cp.asnumpy(fine_cen))
    if cen_np == 0.0:
        return np.float32(cor_initialisation_value)
    else:
        return cen_np


def _search_coarse(sino, smin, smax, ratio, drop):
    (nrow, ncol) = sino.shape
    flip_sino = cp.ascontiguousarray(cp.fliplr(sino))
    comp_sino = cp.ascontiguousarray(cp.flipud(sino))

    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    cen_fliplr = (ncol - 1.0) / 2.0
    start_cor, stop_cor = np.sort((smin, smax))
    start_cor = np.int16(np.clip(start_cor, 0, ncol - 1))
    stop_cor = np.int16(np.clip(stop_cor, 0, ncol - 1))
    list_cor = cp.arange(start_cor, stop_cor + 1.0, dtype=cp.float32)
    list_shift = 2.0 * (list_cor - cen_fliplr)
    list_metric = cp.empty(list_shift.shape, dtype=cp.float32)

    sino_sino = cp.vstack((sino, flip_sino))
    for i, shift in enumerate(list_shift):
        _sino = sino_sino[nrow:]
        _sino[...] = cp.roll(flip_sino, int(shift), axis=1)
        if shift >= 0:
            _sino[:, :shift] = comp_sino[:, :shift]
        else:
            _sino[:, shift:] = comp_sino[:, shift:]
        list_metric[i] = cp.mean(cp.abs(fftshift(fft2(sino_sino))) * mask)

    minpos = cp.argmin(list_metric)
    if minpos == 0:
        print("WARNING!!!Global minimum is out of searching range")
        print(f"Please extend smin: {smin}")
    if minpos == len(list_metric) - 1:
        print("WARNING!!!Global minimum is out of searching range")
        print(f"Please extend smax: {smax}")
    cor = list_cor[minpos]
    return cor


def _search_fine(sino, srad, step, init_cen, ratio, drop):
    (nrow, ncol) = sino.shape

    flip_sino = cp.ascontiguousarray(cp.fliplr(sino))
    comp_sino = cp.ascontiguousarray(cp.flipud(sino))
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)

    cen_fliplr = (ncol - 1.0) / 2.0
    srad = np.clip(np.abs(srad), 1, ncol // 10 - 1)
    step = np.clip(np.abs(step), 0.1, 1.1)
    init_cen = np.clip(init_cen, srad, ncol - srad - 1)
    list_cor = init_cen + cp.arange(-srad, srad + step, step, dtype=cp.float32)
    list_shift = 2.0 * (list_cor - cen_fliplr)
    list_metric = cp.empty(list_shift.shape, dtype="float32")

    _calculate_metric(list_shift, sino, flip_sino, comp_sino, mask, out=list_metric)
    cor = list_cor[cp.argmin(list_metric)]
    return cor


def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    cen_row = int(math.ceil(nrow / 2.0) - 1)
    cen_col = int(math.ceil(ncol / 2.0) - 1)
    drop = min([drop, int(math.ceil(0.05 * nrow))])

    block_x = 128
    block_y = 1
    block_dims = (block_x, block_y)
    grid_x = (ncol + block_x - 1) // block_x
    grid_y = nrow
    grid_dims = (grid_x, grid_y)
    mask = cp.empty((nrow, ncol), dtype="float32")
    params = (
        ncol,
        nrow,
        cen_col,
        cen_row,
        cp.float32(du),
        cp.float32(dv),
        cp.float32(radius),
        cp.float32(drop),
        mask,
    )
    module = load_cuda_module("generate_mask")
    kernel = module.get_function("generate_mask_full")
    kernel(grid_dims, block_dims, params)
    return mask


def round_up(x: float) -> int:
    if x >= 0.0:
        return int(math.ceil(x))
    else:
        return int(math.floor(x))


def _get_available_gpu_memory() -> int:
    dev = cp.cuda.Device()
    # first, let's make some space
    cp.get_default_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    available_memory = dev.mem_info[0] + cp.get_default_memory_pool().free_bytes()
    return int(available_memory * 0.9)  # 10% safety margin


def _calculate_chunks(
    nshifts: int, shift_size: int, available_memory: Optional[int] = None
) -> List[int]:
    if available_memory is None:
        available_memory = _get_available_gpu_memory()

    available_memory -= shift_size
    freq_domain_size = (
        shift_size * 2  # it needs full (FFT), with complex64, so it's double
    )
    fft_plan_size = freq_domain_size
    size_per_shift = 2.5 * (fft_plan_size + freq_domain_size + shift_size)
    nshift_max = available_memory // size_per_shift
    assert nshift_max > 0, "Not enough memory to process"
    num_chunks = int(np.ceil(nshifts / nshift_max))
    chunk_size = int(np.ceil(nshifts / num_chunks))
    chunks = [chunk_size] * (num_chunks - 1)
    stop_idx = list(np.cumsum(chunks))
    stop_idx.append(nshifts)
    return stop_idx


def _calculate_metric(list_shift, sino, flip_sino, comp_sino, mask, out):
    # this tries to simplify - if shift_col is integer, no need to spline interpolate
    assert list_shift.dtype == cp.float32, "shifts must be single precision floats"
    assert sino.dtype == cp.float32, "sino must be float32"
    assert flip_sino.dtype == cp.float32, "flip_sino must be float32"
    assert comp_sino.dtype == cp.float32, "comp_sino must be float32"
    assert out.dtype == cp.float32, "out must be float32"
    assert flip_sino.flags["C_CONTIGUOUS"], "flip_sino must be C-contiguous"
    assert comp_sino.flags["C_CONTIGUOUS"], "comp_sino must be C-contiguous"
    assert list_shift.flags["C_CONTIGUOUS"], "list_shift must be C-contiguous"
    nshifts = list_shift.shape[0]
    na1 = sino.shape[0]
    na2 = flip_sino.shape[0]

    module = load_cuda_module("center_360_shifts")
    shift_whole_shifts = module.get_function("shift_whole_shifts")
    # note: we don't have to calculate the mean here, as we're only looking for minimum metric.
    # The sum is enough.
    masked_sum_abs_kernel = cp.ReductionKernel(
        in_params="complex64 x, float32 mask",  # input, complex + mask
        out_params="float32 out",  # output, real
        map_expr="abs(x) * mask",
        reduce_expr="a + b",
        post_map_expr="out = a",
        identity="0.0f",
        reduce_type="float",
        name="masked_sum_abs",
    )

    # determine how many shifts we can fit in the available memory
    # and iterate in chunks
    chunks = _calculate_chunks(
        nshifts, (na1 + na2) * flip_sino.shape[1] * cp.float32().nbytes
    )

    mat = cp.empty((chunks[0], na1 + na2, flip_sino.shape[1]), dtype=cp.float32)
    mat[:, :na1, :] = sino

    # explicitly create FFT plan here, so it's not cached and clearly re-used
    plan = get_fft_plan(mat, mat.shape[-2:], axes=(1, 2), value_type="C2C")

    for i, stop_idx in enumerate(chunks):
        if i > 0:
            # more than one iteration means we're tight on memory, so clear up freed blocks
            mat_freq = None
            cp.get_default_memory_pool().free_all_blocks()

        start_idx = 0 if i == 0 else chunks[i - 1]
        size = stop_idx - start_idx

        # first, handle the integer shifts without spline in a raw kernel,
        # and shift in the comp_sino one accordingly
        bx = 128
        gx = (comp_sino.shape[1] + bx - 1) // bx
        shift_whole_shifts(
            grid=(gx, na2, size),  ####
            block=(bx, 1, 1),
            args=(
                flip_sino,
                comp_sino,
                list_shift[start_idx:stop_idx],
                mat[:, na1:, :],
                comp_sino.shape[1],
                na1 + na2,
            ),
        )

        # now we can only look at the spline shifting, the rest is done
        list_shift_host = cp.asnumpy(list_shift[start_idx:stop_idx])
        for i in range(list_shift_host.shape[0]):
            shift_col = float(list_shift_host[i])
            if not shift_col.is_integer():
                shifted = shift(flip_sino, (0, shift_col), order=3, prefilter=True)
                shift_int = round_up(shift_col)
                if shift_int >= 0:
                    mat[i, na1:, shift_int:] = shifted[:, shift_int:]
                else:
                    mat[i, na1:, :shift_int] = shifted[:, :shift_int]

        # stack and transform
        # (we do the full sized mat FFT, even though the last chunk may be smaller, to
        # make sure we can re-use the same FFT plan as before)
        mat_freq = fftshift(fft2(mat, axes=(1, 2), norm=None, plan=plan), axes=(1, 2))

        masked_sum_abs_kernel(
            mat_freq[:size, :, :], mask, out=out[start_idx:stop_idx], axis=(1, 2)
        )


def _downsample(image, dsp_fact0, dsp_fact1):
    """Downsample an image by averaging.

    Parameters
    ----------
        image : 2D array.
        dsp_fact0 : downsampling factor along axis 0.
        dsp_fact1 : downsampling factor along axis 1.

    Returns
    ---------
        image_dsp : Downsampled image.
    """
    (height, width) = image.shape
    dsp_fact0 = cp.clip(cp.int16(dsp_fact0), 1, height // 2)
    dsp_fact1 = cp.clip(cp.int16(dsp_fact1), 1, width // 2)
    height_dsp = height // dsp_fact0
    width_dsp = width // dsp_fact1
    if dsp_fact0 == 1 and dsp_fact1 == 1:
        image_dsp = image
    else:
        image_dsp = image[0 : dsp_fact0 * height_dsp, 0 : dsp_fact1 * width_dsp]
        image_dsp = (
            image_dsp.reshape(height_dsp, dsp_fact0, width_dsp, dsp_fact1)
            .mean(-1)
            .mean(1)
        )
    return image_dsp


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # %%%%%%%%%%%%%%%%%%%%%%%%%find_center_360%%%%%%%%%%%%%%%%%%%%%%%%%
# --- Center of rotation (COR) estimation method ---#
def find_center_360(
    data: cp.ndarray,
    ind: Optional[int] = None,
    win_width: int = 10,
    side: Optional[Literal[0, 1]] = None,
    denoise: bool = True,
    norm: bool = False,
    use_overlap: bool = False,
) -> Tuple[float, float, Optional[Literal[0, 1]], float]:
    """
    Find the center-of-rotation (COR) in a 360-degree scan and also an offset
    to perform data transformation from 360 to 180 degrees scan. See :cite:`vo2021data`.

    Parameters
    ----------
    data : cp.ndarray
        3D tomographic data as a Cupy array.
    ind : int, optional
        Index of the slice to be used for estimate the CoR and the overlap.
    win_width : int, optional
        Window width used for finding the overlap area.
    side : {None, 0, 1}, optional
        Overlap size. Only there options: None, 0, or 1. "None" corresponds
        to fully automated determination. "0" corresponds to the left side.
        "1" corresponds to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalisation if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    cor : float
        Center-of-rotation.
    overlap : float
        Width of the overlap area between two halves of the sinogram.
    side : int
        Overlap side between two halves of the sinogram.
    overlap_position : float
        Position of the window in the first image giving the best
        correlation metric.
    """
    if data.ndim != 3:
        raise ValueError("A 3D array must be provided")

    data = data_checker(data, verbosity=True, method_name="find_center_360")

    # this method works with a 360-degree sinogram.
    if ind is None:
        _sino = data[:, 0, :]
    else:
        _sino = data[:, ind, :]

    (nrow, ncol) = _sino.shape
    nrow_180 = nrow // 2 + 1
    sino_top = _sino[0:nrow_180, :]
    sino_bot = cp.fliplr(_sino[-nrow_180:, :])
    (overlap, side, overlap_position) = _find_overlap(
        sino_top, sino_bot, win_width, side, denoise, norm, use_overlap
    )
    if side == 0:
        cor = overlap / 2.0 - 1.0
    else:
        cor = ncol - overlap / 2.0 - 1.0

    return cor, overlap, side, overlap_position


def _find_overlap(
    mat1, mat2, win_width, side=None, denoise=True, norm=False, use_overlap=False
):
    """
    Find the overlap area and overlap side between two images (Ref. [1]) where
    the overlap side referring to the first image.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 :  array_like
        2D array. Projection image or sinogram image.
    win_width : int
        Width of the searching window.
    side : {None, 0, 1}, optional
        Only there options: None, 0, or 1. "None" corresponding to fully
        automated determination. "0" corresponding to the left side. "1"
        corresponding to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    overlap : float
        Width of the overlap area between two images.
    side : int
        Overlap side between two images.
    overlap_position : float
        Position of the window in the first image giving the best
        correlation metric.

    """
    ncol1 = mat1.shape[1]
    ncol2 = mat2.shape[1]
    win_width = int(np.clip(win_width, 6, min(ncol1, ncol2) // 2))

    if side == 1:
        (list_metric, offset) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=side,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )
        overlap_position = _calculate_curvature(list_metric)[1]
        overlap_position += offset
        overlap = ncol1 - overlap_position + win_width // 2
    elif side == 0:
        (list_metric, offset) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=side,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )
        overlap_position = _calculate_curvature(list_metric)[1]
        overlap_position += offset
        overlap = overlap_position + win_width // 2
    else:
        (list_metric1, offset1) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=1,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )
        (list_metric2, offset2) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=0,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )

        (curvature1, overlap_position1) = _calculate_curvature(list_metric1)
        overlap_position1 += offset1
        (curvature2, overlap_position2) = _calculate_curvature(list_metric2)
        overlap_position2 += offset2

        if curvature1 > curvature2:
            side = 1
            overlap_position = overlap_position1
            overlap = ncol1 - overlap_position + win_width // 2
        else:
            side = 0
            overlap_position = overlap_position2
            overlap = overlap_position + win_width // 2

    return overlap, side, overlap_position


def _search_overlap(
    mat1, mat2, win_width, side, denoise=True, norm=False, use_overlap=False
):
    """
    Calculate the correlation metrics between a rectangular region, defined
    by the window width, on the utmost left/right side of image 2 and the
    same size region in image 1 where the region is slided across image 1.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 : array_like
        2D array. Projection image or sinogram image.
    win_width : int
        Width of the searching window.
    side : {0, 1}
        Only two options: 0 or 1. It is used to indicate the overlap side
        respects to image 1. "0" corresponds to the left side. "1" corresponds
        to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    list_metric : array_like
        1D array. List of the correlation metrics.
    offset : int
        Initial position of the searching window where the position
        corresponds to the center of the window.
    """

    if denoise is True:
        # note: the filtering makes the output contiguous
        mat1 = gaussian_filter(mat1, (2, 2), mode="reflect")
        mat2 = gaussian_filter(mat2, (2, 2), mode="reflect")
    else:
        mat1 = cp.ascontiguousarray(mat1, dtype=cp.float32)
        mat2 = cp.ascontiguousarray(mat2, dtype=cp.float32)

    (nrow1, ncol1) = mat1.shape
    (nrow2, ncol2) = mat2.shape

    if nrow1 != nrow2:
        raise ValueError("Two images are not at the same height!!!")

    win_width = int(np.clip(win_width, 6, min(ncol1, ncol2) // 2 - 1))
    offset = win_width // 2
    win_width = 2 * offset  # Make it even

    list_metric = _calc_metrics(mat1, mat2, win_width, side, use_overlap, norm)

    min_metric = cp.min(list_metric)
    if min_metric != 0.0:
        list_metric /= min_metric

    return list_metric, offset


def _calc_metrics(mat1, mat2, win_width, side, use_overlap, norm):
    assert mat1.dtype == cp.float32, "only float32 supported"
    assert mat2.dtype == cp.float32, "only float32 supported"
    assert mat1.shape[0] == mat2.shape[0]
    assert mat1.flags.c_contiguous, "only contiguos arrays supported"
    assert mat2.flags.c_contiguous, "only contiguos arrays supported"

    _calc_metrics_module = load_cuda_module(
        "calc_metrics",
        name_expressions=[
            "calc_metrics_kernel<false, false>",
            "calc_metrics_kernel<true, false>",
            "calc_metrics_kernel<false, true>",
            "calc_metrics_kernel<true, true>",
        ],
        options=("--maxrregcount=32",),
    )

    num_pos = mat1.shape[1] - win_width
    list_metric = cp.empty(num_pos, dtype=cp.float32)

    args = (
        mat1,
        np.int32(mat1.strides[0] / mat1.strides[1]),
        mat2,
        np.int32(mat2.strides[0] / mat2.strides[1]),
        np.int32(win_width),
        np.int32(mat1.shape[0]),
        np.int32(side),
        list_metric,
    )
    block = (128, 1, 1)
    grid = (1, np.int32(num_pos), 1)
    smem = block[0] * 4 * 6 if use_overlap else block[0] * 4 * 3
    bool2str = lambda x: "true" if x is True else "false"
    calc_metrics = _calc_metrics_module.get_function(
        f"calc_metrics_kernel<{bool2str(norm)}, {bool2str(use_overlap)}>"
    )
    calc_metrics(grid=grid, block=block, args=args, shared_mem=smem)

    return list_metric


def _calculate_curvature(list_metric):
    """
    Calculate the curvature of a fitted curve going through the minimum
    value of a metric list.

    Parameters
    ----------
    list_metric : array_like
        1D array. List of metrics.

    Returns
    -------
    curvature : float
        Quadratic coefficient of the parabola fitting.
    min_pos : float
        Position of the minimum value with sub-pixel accuracy.
    """
    radi = 2
    num_metric = list_metric.size
    min_metric_idx = int(cp.argmin(list_metric))
    min_pos = int(np.clip(min_metric_idx, radi, num_metric - radi - 1))

    # work mostly on CPU here - we have very small arrays here
    list1 = cp.asnumpy(list_metric[min_pos - radi : min_pos + radi + 1])
    if not all(map(np.isfinite, list1)):
        raise ValueError(
            "The list of metrics (list1) contains nan's or infs. Check your input data"
        )

    series1 = Polynomial.fit(np.arange(0, 2 * radi + 1), list1, deg=2)
    afact1 = series1.convert().coef[-1]

    list2 = cp.asnumpy(list_metric[min_pos - 1 : min_pos + 2])
    if not all(map(np.isfinite, list2)):
        raise ValueError(
            "The list of metrics (list2) contains nan's or infs. Check your input data"
        )

    series2 = Polynomial.fit(np.arange(min_pos - 1, min_pos + 2), list2, deg=2)
    afact2 = series2.convert().coef[-1]
    bfact2 = series2.convert().coef[-1 - 1]

    curvature = np.abs(afact1)
    if afact2 != 0.0:
        num = -bfact2 / (2 * afact2)
        if (num >= min_pos - 1) and (num <= min_pos + 1):
            min_pos = num

    return curvature, np.float32(min_pos)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


## %%%%%%%%%%%%%%%%%%%%%%find_center_pc%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def find_center_pc(
    proj1: cp.ndarray,
    proj2: cp.ndarray,
    tol: float = 0.5,
    rotc_guess: Union[float, Optional[str]] = None,
) -> np.float32:
    """
    Find rotation axis location by finding the offset between the first
    projection and a mirrored projection 180 degrees apart using
    phase correlation in Fourier space.
    The `phase_cross_correlation` function uses cross-correlation in Fourier
    space, optionally employing an upsampled matrix-multiplication DFT to
    achieve arbitrary subpixel precision. See :cite:`guizar2008efficient`.

    Parameters
    ----------
    proj1 : cp.ndarray
        Projection from the 0th degree angle.
    proj2 : cp.ndarray
        Projection from the 180th degree angle.
    tol : float, optional
        Subpixel accuracy. Defaults to 0.5.
    rotc_guess : float, optional
        Initial guess value for the rotation center. Defaults to None.

    Returns
    ----------
    np.float32
        Rotation axis location.
    """

    proj1 = data_checker(proj1, verbosity=True, method_name="find_center_pc")
    proj2 = data_checker(proj2, verbosity=True, method_name="find_center_pc")

    imgshift = 0.0 if rotc_guess is None else rotc_guess - (proj1.shape[1] - 1.0) / 2.0

    proj1 = shift(proj1, [0, -imgshift], mode="constant", cval=0)
    proj2 = shift(proj2, [0, -imgshift], mode="constant", cval=0)

    # create reflection of second projection
    proj2 = cp.fliplr(proj2)

    # do phase cross correlation between two images
    shiftr = phase_cross_correlation(
        reference_image=proj1.get(), moving_image=proj2.get(), upsample_factor=1.0 / tol
    )

    # Compute center of rotation as the center of first image and the
    # registered translation with the second image
    center = (proj1.shape[1] + shiftr[0][1] - 1.0) / 2.0

    return np.float32(center + imgshift)


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
