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
"""Modules for finding the axis of rotation"""

import math
from typing import Optional

import cupy as cp
import numpy as np
import nvtx
import scipy.ndimage as ndi
from cupy import ndarray
from cupyx.scipy.ndimage import gaussian_filter, shift
from scipy import stats

__all__ = [
    'find_center_vo_cupy',
    'find_center_360',
]


@nvtx.annotate()
def find_center_vo_cupy(
    data: ndarray,
    ind: int = None,
    smin: int = -50,
    smax: int = 50,
    srad: int = 6,
    step: int = 0.25,
    ratio: int = 0.5,
    drop: int = 20
) -> float:
    """
    Find rotation axis location using Nghia Vo's method. See the paper
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-16-19078&id=297315

    Parameters
    ----------
    data : ndarray
        3D tomographic data or a 2D sinogram as a CuPy array.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    smin : int, optional
        Coarse search radius. Reference to the horizontal center of
        the sinogram.
    smax : int, optional
        Coarse search radius. Reference to the horizontal center of
        the sinogram.        
    srad : float, optional
        Fine search radius.
    step : float, optional
        Step of fine searching.
    ratio : float, optional
        The ratio between the FOV of the camera and the size of object.
        It's used to generate the mask.
    drop : int, optional
        Drop lines around vertical center of the mask.

    Returns
    -------
    float
        Rotation axis location.
    """
    return _find_center_vo_gpu(data, ind, smin, smax, srad, step, ratio, drop)


def _find_center_vo_gpu(sino, ind, smin, smax, srad, step, ratio, drop):
    if sino.ndim == 2:
        sino = cp.expand_dims(sino, 1)
        ind = 0

    height = sino.shape[1]

    if ind is None:
        ind = height // 2
        if height > 10:
            _sino = cp.mean(sino[:, ind - 5: ind + 5, :], axis=1)
        else:
            _sino = sino[:, ind, :]
    else:
        _sino = sino[:, ind, :]

    with nvtx.annotate("gaussian_filter_1", color="green"):
        _sino_cs = gaussian_filter(_sino, (3, 1), mode="reflect")
    with nvtx.annotate("gaussian_filter_2", color="green"):
        _sino_fs = gaussian_filter(_sino, (2, 2), mode="reflect")
    
    if _sino.shape[0] * _sino.shape[1] > 4e6:
        # data is large, so downsample it before performing search for
        # centre of rotation
        _sino_coarse = _downsample(_sino_cs, 2, 1)
        init_cen = _search_coarse(_sino_coarse, smin / 4.0, smax / 4.0, ratio, drop)
        fine_cen = _search_fine(_sino_fs, srad, step, init_cen * 4.0, ratio, drop)
    else:
        init_cen = _search_coarse(_sino_cs, smin, smax, ratio, drop)
        fine_cen = _search_fine(_sino_fs, srad, step, init_cen, ratio, drop)

    return fine_cen.astype(cp.float32)


@nvtx.annotate()
def _search_coarse(sino, smin, smax, ratio, drop):
    (nrow, ncol) = sino.shape
    flip_sino = cp.ascontiguousarray(cp.fliplr(sino))
    comp_sino = cp.ascontiguousarray(cp.flipud(sino))
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)

    cen_fliplr = (ncol - 1.0) / 2.0
    smin_clip_val = max(min(smin + cen_fliplr, ncol - 1), 0)
    smin = smin_clip_val - cen_fliplr
    smax_clip_val = max(min(smax + cen_fliplr, ncol - 1), 0)
    smax = smax_clip_val - cen_fliplr
    start_cor = ncol // 2 + smin
    stop_cor = ncol // 2 + smax
    list_cor = cp.arange(start_cor, stop_cor + 0.5, 0.5, dtype=cp.float32)
    list_shift = 2.0 * (list_cor - cen_fliplr)
    list_metric = cp.empty(list_shift.shape, dtype=cp.float32)
    _calculate_metric(list_shift, sino, flip_sino, comp_sino, mask, list_metric)
    
    minpos = cp.argmin(list_metric)
    if minpos == 0:
        print("WARNING!!!Global minimum is out of searching range")
        print(f"Please extend smin: {smin}")
    if minpos == len(list_metric) - 1:
        print("WARNING!!!Global minimum is out of searching range")
        print(f"Please extend smax: {smax}")
    cor = list_cor[minpos]
    return cor


@nvtx.annotate()
def _search_fine(sino, srad, step, init_cen, ratio, drop):
    (nrow, ncol) = sino.shape

    flip_sino = cp.ascontiguousarray(cp.fliplr(sino))
    comp_sino = cp.ascontiguousarray(cp.flipud(sino))
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    
    cen_fliplr = (ncol - 1.0) / 2.0
    srad = max(min(abs(float(srad)), ncol/4.0), 1.0)
    step = max(min(abs(step), srad), 0.1)
    init_cen = max(min(init_cen, ncol - srad - 1), srad)
    list_cor = init_cen + cp.arange(-srad, srad + step, step, dtype=np.float32)
    list_shift = 2.0 * (list_cor - cen_fliplr)
    list_metric = cp.empty(list_shift.shape, dtype="float32")
    
    _calculate_metric(list_shift, sino, flip_sino, comp_sino, mask, out=list_metric)
    cor = list_cor[cp.argmin(list_metric)]
    return cor


GENERATE_MASK_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void generate_mask(const int ncol, const int nrow, const int cen_col,
                    const int cen_row, const float du, const float dv,
                    const float radius, const float drop, unsigned short* mask) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (j >= nrow || i >= ncol)
        return;
    
    int pos = __float2int_ru(((j - cen_row) * dv / radius) / du);
    int pos1 = -pos + cen_col;
    int pos2 = pos + cen_col;

    if (pos1 > pos2) {
        int temp = pos1;
        pos1 = pos2;
        pos2 = temp;
        if (pos1 >= ncol) {
            pos1 = ncol -1;
        }
        if (pos2 < 0) {
            pos2 = 0;
        }
    }
    else {
        if (pos1 < 0) {
            pos1 = 0;
        }
        if (pos2 >= ncol) {
            pos2 = ncol -1;
        }
    }

    short outval = (pos1 <= i && i <= pos2) ? 1 : 0;

    // mask[cen_row - drop: cen_row + drop + 1, :] = 0
    if (j >= cen_row - drop && j <= cen_row + drop) {
        outval = 0;
    }
    // mask[:, cen_col - 1: cen_col + 2] = 0
    if (i >= cen_col - 1 && i <= cen_col + 1) {
        outval = 0;
    }
    
    // write to ifft-shifted positions
    int ishift = (ncol + 1) / 2;
    int jshift = (nrow + 1) / 2;
    int outi = (i + ishift) % ncol;
    int outj = (j + jshift) % nrow;

    mask[outj * ncol + outi] = outval;
}
""",
    "generate_mask",
)


@nvtx.annotate()
def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    cen_row = int(math.ceil(nrow / 2.0) - 1)
    cen_col = int(math.ceil(ncol / 2.0) - 1)
    drop = min([drop, int(math.ceil(0.05 * nrow))])


    block_x = 16
    block_y = 8
    block_dims = (block_x, block_y)
    grid_x = (ncol + block_x - 1) // block_x
    grid_y = (nrow  + block_y - 1) // block_y
    grid_dims = (grid_x, grid_y)
    mask = cp.empty((nrow, ncol), dtype="uint16")
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
    GENERATE_MASK_KERNEL(grid_dims, block_dims, params)
    return mask


def round_up(x: float) -> int:
    if x >= 0.0:
        return int(math.ceil(x))
    else:
        return int(math.floor(x))


@nvtx.annotate()
def _calculate_metric(list_shift, sino1, sino2, sino3, mask, out):
    # this tries to simplify - if shift_col is integer, no need to spline interpolate
    assert list_shift.dtype == cp.float32, "shifts must be single precision floats"
    assert sino1.dtype == cp.float32, "sino1 must be float32"
    assert sino2.dtype == cp.float32, "sino1 must be float32"
    assert sino3.dtype == cp.float32, "sino1 must be float32"
    assert out.dtype == cp.float32, "sino1 must be float32"
    assert sino2.flags['C_CONTIGUOUS'], "sino2 must be C-contiguous"
    assert sino3.flags['C_CONTIGUOUS'], "sino3 must be C-contiguous"
    assert list_shift.flags['C_CONTIGUOUS'], "list_shift must be C-contiguous"
    nshifts = list_shift.shape[0]
    na1 = sino1.shape[0]
    na2 = sino2.shape[0]
    mat = cp.empty((nshifts, na1+na2, sino2.shape[1]), dtype=cp.float32)
    
    
    # first, handle the integer shifts without spline in a raw kernel, 
    # and shift in the sino3 one accordingly
    shift_whole_shifts = cp.RawKernel(r"""
        #include <cupy/complex.cuh>

        extern "C" __global__
        void shift_whole_shifts(const float* sino2, const float* sino3, 
                                const float* __restrict__ list_shift,
                                float* mat, int nx, int nymat) {
           int xid = threadIdx.x + blockIdx.x * blockDim.x;
           int yid = blockIdx.y;
           int zid = blockIdx.z;
           int ny = gridDim.y;
           
           if (xid >= nx) return;
           
           float shift_col = list_shift[zid];
           float int_part = 0.0;
           float frac_part = modf(shift_col, &int_part);
           if (abs(frac_part) > 1e-5f) {
                // we have a floating point shift, so we only roll in 
                // sino3, but we leave the rest for later using scipy
                int shift_int = shift_col >= 0.0 ? int(ceil(shift_col)) : int(floor(shift_col));
                if (shift_int >= 0 && xid < shift_int) {
                   mat[zid * nymat * nx + yid * nx + xid] = sino3[yid * nx + xid];
                } 
                if (shift_int < 0 && xid >= nx+shift_int) {
                   mat[zid * nymat * nx + yid * nx + xid] = sino3[yid * nx + xid];
                }
           } else {
                // we have an integer shift, so we can roll in directly
                // by indexing
                int shift_int = int(shift_col);
                if (shift_int >= 0) {
                    if (xid >= shift_int) {
                        mat[zid * nymat * nx + yid * nx + xid] = sino2[yid * nx + xid-shift_int];
                    } else {
                        mat[zid * nymat * nx + yid * nx + xid] = sino3[yid * nx + xid];
                    }
                } else {
                    if (xid < nx+shift_int) {
                        mat[zid * nymat * nx + yid * nx + xid] = sino2[yid * nx + xid-shift_int];
                    } else {
                        mat[zid * nymat * nx + yid * nx + xid] = sino3[yid * nx + xid];
                    }
                }
           }
        }
    """, name="shift_whole_shifts")

    bx = 128
    gx = (sino3.shape[1] + bx - 1) // bx
    shift_whole_shifts(grid=(gx, na2, list_shift.shape[0]),
                       block=(bx, 1, 1),
                       args=(sino2, sino3, list_shift, mat[:,na1:,:], sino3.shape[1], na1+na2))
    
    # now we can only look at the spline shifting, the rest is done
    list_shift_host = cp.asnumpy(list_shift)
    for i in range(list_shift_host.shape[0]):
        shift_col = float(list_shift_host[i])
        if not shift_col.is_integer():
            shifted = shift(sino2, (0, shift_col), order=3, prefilter=True)
            shift_int = round_up(shift_col)
            if shift_int >= 0:
                mat[i, na1:, shift_int:] = shifted[:, shift_int:]
            else:
                mat[i, na1:, :shift_int] = shifted[:, :shift_int]

        
    # stack and transform
    mat[:,:na1,:] = sino1
    cp.abs(cp.fft.fft2(mat, axes=(1,2)), out=mat)
    mat *= mask
    cp.mean(mat, axis=(1,2), out=out)




DOWNSAMPLE_SINO_KERNEL = cp.RawKernel(
    r"""
    extern "C" __global__
    void downsample_sino(float* sino, int dx, int dz, int level,
                         float* out) {
        // use shared memory to store the values used to "merge" columns of the
        // sinogram in the downsampling process
        extern __shared__ float downsampled_vals[];
        unsigned int binsize, i, j, k, orig_ind, out_ind, output_bin_no;
        i = blockDim.x * blockIdx.x + threadIdx.x;
        j = 0;
        k = blockDim.y * blockIdx.y + threadIdx.y;
        orig_ind = (k * dz) + i;
        binsize = 1 << level;
        unsigned int dz_downsampled = __float2uint_rd(
            fdividef(__uint2float_rd(dz), __uint2float_rd(binsize)));
        unsigned int i_downsampled = __float2uint_rd(
            fdividef(__uint2float_rd(i), __uint2float_rd(binsize)));
        if (orig_ind < dx * dz) {
            output_bin_no =  __float2uint_rd(
                fdividef(__uint2float_rd(i), __uint2float_rd(binsize))
            );
            out_ind = (k * dz_downsampled) + i_downsampled;
            downsampled_vals[threadIdx.y * 8 + threadIdx.x] = sino[orig_ind] / __uint2float_rd(binsize);
            // synchronise threads within thread-block so that it's guaranteed
            // that all the required values have been copied into shared memeory
            // to then sum and save in the downsampled output
            __syncthreads();
            // arbitrarily use the "beginning thread" in each "lot" of pixels
            // for downsampling to then save the desired value in the
            // downsampled output array
            if (i % 4 == 0) {
                out[out_ind] = downsampled_vals[threadIdx.y * 8 + threadIdx.x] +
                    downsampled_vals[threadIdx.y * 8 + threadIdx.x + 1] +
                    downsampled_vals[threadIdx.y * 8 + threadIdx.x + 2] +
                    downsampled_vals[threadIdx.y * 8 + threadIdx.x + 3];
            }
        }
    }
    """,
    "downsample_sino",
)


@nvtx.annotate()
def _downsample(sino, level, axis):
    assert sino.dtype == cp.float32, "single precision floating point input required"
    assert sino.flags['C_CONTIGUOUS'], "list_shift must be C-contiguous"

    dx, dz = sino.shape
    # Determine the new size, dim, of the downsampled dimension
    dim = int(sino.shape[axis] / math.pow(2, level))
    shape = [dx, dz]
    shape[axis] = dim
    downsampled_data = cp.empty(shape, dtype="float32")

    block_x = 8
    block_y = 8
    block_dims = (block_x, block_y)
    grid_x = (sino.shape[1] + block_x - 1) // block_x
    grid_y = (sino.shape[0] + block_y - 1) // block_y
    grid_dims = (grid_x, grid_y)
    # 8x8 thread-block, which means 16 "lots" of columns to downsample per
    # thread-block; 4 bytes per float, so allocate 16*6 = 64 bytes of shared
    # memeory per thread-block
    shared_mem_bytes = 64
    params = (sino, dx, dz, level, downsampled_data)
    DOWNSAMPLE_SINO_KERNEL(grid_dims, block_dims, params, shared_mem=shared_mem_bytes)
    return downsampled_data


#--- Center of rotation (COR) estimation method ---#
@nvtx.annotate()
def find_center_360(
    sino_360: np.ndarray,
    win_width: int = 10,
    side: set = None,
    denoise: bool = True,
    norm: bool = False,
    use_overlap: bool = False
) -> tuple[float, float, int, float]:
    """
    Find the center-of-rotation (COR) in a 360-degree scan with offset COR use
    the method presented in Ref. [1] by Nghia Vo.

    Parameters
    ----------
    sino_360 : ndarray
        2D array, a 360-degree sinogram.
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

    References
    ----------
    [1] : https://doi.org/10.1364/OE.418448
    """
    if sino_360.ndim != 2:
        raise ValueError("360-degree sinogram must be a 2D array.")

    (nrow, ncol) = sino_360.shape
    nrow_180 = nrow // 2 + 1
    sino_top = sino_360[0:nrow_180, :]
    sino_bot = np.fliplr(sino_360[-nrow_180:, :])
    (overlap, side, overlap_position) = _find_overlap(
        sino_top, sino_bot, win_width, side, denoise, norm, use_overlap)
    if side == 0:
        cor = overlap / 2.0 - 1.0
    else:
        cor = ncol - overlap / 2.0 - 1.0

    return np.float32(cor), np.float32(overlap), side, np.float32(overlap_position)


def _find_overlap(mat1, mat2, win_width, side=None, denoise=True, norm=False,
                 use_overlap=False):
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
    win_width = np.int16(np.clip(win_width, 6, min(ncol1, ncol2) // 2))

    if side == 1:
        (list_metric, offset) = _search_overlap(mat1, mat2, win_width, side,
                                               denoise, norm, use_overlap)
        overlap_position = _calculate_curvature(list_metric)[1]
        overlap_position += offset
        overlap = ncol1 - overlap_position + win_width // 2
    elif side == 0:
        (list_metric, offset) = _search_overlap(mat1, mat2, win_width, side,
                                               denoise, norm, use_overlap)
        overlap_position = _calculate_curvature(list_metric)[1]
        overlap_position += offset
        overlap = overlap_position + win_width // 2
    else:
        (list_metric1, offset1) = _search_overlap(mat1, mat2, win_width, 1,
                                                 norm, denoise, use_overlap)
        (list_metric2, offset2) = _search_overlap(mat1, mat2, win_width, 0,
                                                 norm, denoise, use_overlap)

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


def _search_overlap(mat1, mat2, win_width, side, denoise=True, norm=False,
                   use_overlap=False):
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
        mat1 = ndi.gaussian_filter(mat1, (2, 2), mode='reflect')
        mat2 = ndi.gaussian_filter(mat2, (2, 2), mode='reflect')
    (nrow1, ncol1) = mat1.shape
    (nrow2, ncol2) = mat2.shape

    if nrow1 != nrow2:
        raise ValueError("Two images are not at the same height!!!")

    win_width = np.int16(np.clip(win_width, 6, min(ncol1, ncol2) // 2 - 1))
    offset = win_width // 2
    win_width = 2 * offset  # Make it even
    ramp_down = np.linspace(1.0, 0.0, win_width)
    ramp_up = 1.0 - ramp_down
    wei_down = np.tile(ramp_down, (nrow1, 1))
    wei_up = np.tile(ramp_up, (nrow1, 1))

    if side == 1:
        mat2_roi = mat2[:, 0:win_width]
        mat2_roi_wei = mat2_roi * wei_up
    else:
        mat2_roi = mat2[:, ncol2 - win_width:]
        mat2_roi_wei = mat2_roi * wei_down

    list_mean2 = np.mean(np.abs(mat2_roi), axis=1)
    list_pos = np.arange(offset, ncol1 - offset)
    num_metric = len(list_pos)
    list_metric = np.ones(num_metric, dtype=np.float32)

    for i, pos in enumerate(list_pos):
        mat1_roi = mat1[:, pos - offset:pos + offset]
        if use_overlap is True:
            if side == 1:
                mat1_roi_wei = mat1_roi * wei_down
            else:
                mat1_roi_wei = mat1_roi * wei_up
        if norm is True:
            list_mean1 = np.mean(np.abs(mat1_roi), axis=1)
            list_fact = list_mean2 / list_mean1
            mat_fact = np.transpose(np.tile(list_fact, (win_width, 1)))
            mat1_roi = mat1_roi * mat_fact
            if use_overlap is True:
                mat1_roi_wei = mat1_roi_wei * mat_fact
        if use_overlap is True:
            mat_comb = mat1_roi_wei + mat2_roi_wei
            list_metric[i] = (_correlation_metric(mat1_roi, mat2_roi)
                              + _correlation_metric(mat1_roi, mat_comb)
                              + _correlation_metric(mat2_roi, mat_comb)) / 3.0
        else:
            list_metric[i] = _correlation_metric(mat1_roi, mat2_roi)
    min_metric = np.min(list_metric)
    if min_metric != 0.0:
        list_metric = list_metric / min_metric

    return list_metric, offset


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
    num_metric = len(list_metric)
    min_pos = np.clip(
        np.argmin(list_metric), radi, num_metric - radi - 1)

    list1 = list_metric[min_pos - radi:min_pos + radi + 1]
    afact1 = np.polyfit(np.arange(0, 2 * radi + 1), list1, 2)[0]
    list2 = list_metric[min_pos - 1:min_pos + 2]
    (afact2, bfact2, _) = np.polyfit(
        np.arange(min_pos - 1, min_pos + 2), list2, 2)

    curvature = np.abs(afact1)
    if afact2 != 0.0:
        num = - bfact2 / (2 * afact2)
        if (num >= min_pos - 1) and (num <= min_pos + 1):
            min_pos = num

    return curvature, np.float32(min_pos)


def _correlation_metric(mat1, mat2):
    """
    Calculate the correlation metric. Smaller metric corresponds to better
    correlation.

    Parameters
    ---------
    mat1 : array_like
    mat2 : array_like

    Returns
    -------
    float
        Correlation metric.
    """
    return np.abs(
        1.0 - stats.pearsonr(mat1.flatten('F'), mat2.flatten('F'))[0])
