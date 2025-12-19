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
"""Module for stripes removal"""

import numpy as np
import pywt
from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from cupyx.scipy.ndimage import median_filter, binary_dilation, uniform_filter1d
    from cupyx.scipy.fft import fft2, ifft2, fftshift
    from cupyx.scipy.fftpack import get_fft_plan
    from httomolibgpu.cuda_kernels import load_cuda_module
else:
    median_filter = Mock()
    binary_dilation = Mock()
    uniform_filter1d = Mock()
    fft2 = Mock()
    ifft2 = Mock()
    fftshift = Mock()


from typing import Optional, Tuple, Union

__all__ = [
    "remove_stripe_based_sorting",
    "remove_stripe_fw",
    "remove_stripe_ti",
    "remove_all_stripe",
    "raven_filter",
]


def remove_stripe_based_sorting(
    data: Union[cp.ndarray, np.ndarray],
    size: int = 11,
    dim: int = 1,
) -> Union[cp.ndarray, np.ndarray]:
    """
    Remove full and partial stripe artifacts from sinogram using Nghia Vo's
    approach, see :ref:`method_remove_stripe_based_sorting` and :cite:`vo2018superior`. This algorithm works particularly
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
    See :ref:`method_remove_stripe_ti` and :cite:`titarenko2010analytical`.

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

    _, _, dx_orig = data.shape
    if (dx_orig % 2) != 0:
        # if the horizontal detector size is odd, the data needs to be padded
        data = cp.pad(data, ((0, 0), (0, 0), (0, 1)), mode="edge")

    gamma = beta * ((1 - beta) / (1 + beta)) ** cp.abs(
        cp.fft.fftfreq(data.shape[-1]) * data.shape[-1]
    )
    gamma[0] -= 1
    v = cp.mean(data, axis=0)
    v = v - v[:, 0:1]
    v = cp.fft.irfft(cp.fft.rfft(v) * cp.fft.rfft(gamma)).astype(data.dtype)
    data[:] += v
    if (dx_orig % 2) != 0:
        # unpad
        return data[:, :, :-1]
    else:
        return data


###### Ring removal with wavelet filtering (adapted for cupy from pytroch_wavelet package https://pytorch-wavelets.readthedocs.io/)##########
# These functions are taken from TomoCuPy package
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


def _reflect(x: np.ndarray, minx: float, maxx: float) -> np.ndarray:
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


class _DeviceMemStack:
    def __init__(self) -> None:
        self.allocations = []
        self.current = 0
        self.highwater = 0

    def malloc(self, bytes):
        self.allocations.append(bytes)
        allocated = self._round_up(bytes)
        self.current += allocated
        self.highwater = max(self.current, self.highwater)

    def free(self, bytes):
        assert bytes in self.allocations
        self.allocations.remove(bytes)
        self.current -= self._round_up(bytes)
        assert self.current >= 0

    def _round_up(self, size):
        ALLOCATION_UNIT_SIZE = 512
        size = (size + ALLOCATION_UNIT_SIZE - 1) // ALLOCATION_UNIT_SIZE
        return size * ALLOCATION_UNIT_SIZE


def _mypad(
    x: cp.ndarray, pad: Tuple[int, int, int, int], mem_stack: Optional[_DeviceMemStack]
) -> cp.ndarray:
    """Function to do numpy like padding on Arrays. Only works for 2-D
    padding.

    Inputs:
        x (array): Array to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
    """
    # Vertical only
    if pad[0] == 0 and pad[1] == 0:
        m1, m2 = pad[2], pad[3]
        l = x.shape[-2] if not mem_stack else x[-2]
        xe = _reflect(np.arange(-m1, l + m2, dtype="int32"), -0.5, l - 0.5)
        if mem_stack:
            ret_shape = [x[0], x[1], xe.size, x[3]]
            mem_stack.malloc(np.prod(ret_shape) * np.float32().itemsize)
            return ret_shape
        return x[:, :, xe, :]
    # horizontal only
    elif pad[2] == 0 and pad[3] == 0:
        m1, m2 = pad[0], pad[1]
        l = x.shape[-1] if not mem_stack else x[-1]
        xe = _reflect(np.arange(-m1, l + m2, dtype="int32"), -0.5, l - 0.5)
        if mem_stack:
            ret_shape = [x[0], x[1], x[2], xe.size]
            mem_stack.malloc(np.prod(ret_shape) * np.float32().itemsize)
            return ret_shape
        return x[:, :, :, xe]


def _next_power_of_two(x: int, max_val: int = 128) -> int:
    n = 1
    while n < x and n < max_val:
        n *= 2
    return n


def _conv2d(
    x: cp.ndarray,
    w: np.ndarray,
    stride: Tuple[int, int],
    groups: int,
    mem_stack: Optional[_DeviceMemStack],
) -> cp.ndarray:
    """Convolution (equivalent pytorch.conv2d)"""
    b, ci, hi, wi = x.shape if not mem_stack else x
    co, _, hk, wk = w.shape
    ho = int(np.floor(1 + (hi - hk) / stride[0]))
    wo = int(np.floor(1 + (wi - wk) / stride[1]))
    out_shape = [b, co, ho, wo]
    if mem_stack:
        mem_stack.malloc(np.prod(out_shape) * np.float32().itemsize)
        return out_shape

    out = cp.zeros(out_shape, dtype="float32")
    w = cp.asarray(w)
    x = cp.expand_dims(x, axis=1)
    w = np.expand_dims(w, axis=0)
    symbol_names = [f"grouped_convolution_x<{wk}>", f"grouped_convolution_y<{hk}>"]
    module = load_cuda_module("remove_stripe_fw", name_expressions=symbol_names)
    dim_x = out.shape[-1]
    dim_y = out.shape[-2]
    dim_z = out.shape[0]
    in_stride_x = stride[1]
    in_stride_y = x.strides[-2] // x.dtype.itemsize
    in_stride_z = x.strides[0] // x.dtype.itemsize
    out_stride_z = out.strides[0] // x.dtype.itemsize
    out_stride_group = out.strides[1] // x.dtype.itemsize

    block_x = _next_power_of_two(dim_x)
    block_dim = (block_x, 1, 1)
    grid_x = (dim_x + block_x - 1) // block_x
    grid_dim = (grid_x, dim_y, dim_z)

    if groups == 1:
        grouped_convolution_kernel_x = module.get_function(symbol_names[0])
        grouped_convolution_kernel_x(
            grid_dim,
            block_dim,
            (
                dim_x,
                dim_y,
                dim_z,
                x,
                in_stride_x,
                in_stride_y,
                in_stride_z,
                out,
                out_stride_z,
                out_stride_group,
                w,
            ),
        )
        return out

    grouped_convolution_kernel_y = module.get_function(symbol_names[1])
    in_stride_group = x.strides[2] // x.dtype.itemsize
    grouped_convolution_kernel_y(
        grid_dim,
        block_dim,
        (
            dim_x,
            dim_y,
            dim_z,
            x,
            in_stride_x,
            in_stride_y,
            in_stride_z,
            in_stride_group,
            out,
            out_stride_z,
            out_stride_group,
            w,
        ),
    )
    del w
    return out


def _conv_transpose2d(
    x: cp.ndarray,
    w: np.ndarray,
    stride: Tuple[int, int],
    pad: Tuple[int, int],
    groups: int,
    mem_stack: Optional[_DeviceMemStack],
) -> cp.ndarray:
    """Transposed convolution (equivalent pytorch.conv_transpose2d)"""
    b, co, ho, wo = x.shape if not mem_stack else x
    co, ci, hk, wk = w.shape

    hi = (ho - 1) * stride[0] + hk
    wi = (wo - 1) * stride[1] + wk
    out_shape = [b, ci, hi, wi]
    if mem_stack:
        mem_stack.malloc(np.prod(out_shape) * np.float32().itemsize)
        mem_stack.malloc(w.size * np.float32().itemsize)
        if pad != 0:
            new_out_shape = [
                out_shape[0],
                out_shape[1],
                out_shape[2] - 2 * pad[0],
                out_shape[3] - 2 * pad[1],
            ]
            mem_stack.malloc(np.prod(new_out_shape) * np.float32().itemsize)
            mem_stack.free(np.prod(out_shape) * np.float32().itemsize)
            out_shape = new_out_shape
        mem_stack.free(w.size * np.float32().itemsize)
        return out_shape

    out = cp.zeros(out_shape, dtype="float32")
    w = cp.asarray(w)

    symbol_names = [
        f"transposed_convolution_x<{wk}>",
        f"transposed_convolution_y<{hk}>",
    ]
    module = load_cuda_module("remove_stripe_fw", name_expressions=symbol_names)
    dim_x = out.shape[-1]
    dim_y = out.shape[-2]
    dim_z = out.shape[0]
    in_dim_x = x.shape[-1]
    in_dim_y = x.shape[-2]
    in_stride_y = x.strides[-2] // x.dtype.itemsize
    in_stride_z = x.strides[0] // x.dtype.itemsize

    block_x = _next_power_of_two(dim_x)
    block_dim = (block_x, 1, 1)
    grid_x = (dim_x + block_x - 1) // block_x
    grid_dim = (grid_x, dim_y, dim_z)

    if wk > 1:
        transposed_convolution_kernel_x = module.get_function(symbol_names[0])
        transposed_convolution_kernel_x(
            grid_dim,
            block_dim,
            (dim_x, dim_y, dim_z, x, in_dim_x, in_stride_y, in_stride_z, w, out),
        )
    elif hk > 1:
        transposed_convolution_kernel_y = module.get_function(symbol_names[1])
        transposed_convolution_kernel_y(
            grid_dim,
            block_dim,
            (dim_x, dim_y, dim_z, x, in_dim_y, in_stride_y, in_stride_z, w, out),
        )
    else:
        assert False

    if pad != 0:
        out = out[:, :, pad[0] : out.shape[2] - pad[0], pad[1] : out.shape[3] - pad[1]]
    return cp.ascontiguousarray(out)


def _afb1d(
    x: cp.ndarray,
    h0: np.ndarray,
    h1: np.ndarray,
    dim: int,
    mem_stack: Optional[_DeviceMemStack],
) -> cp.ndarray:
    """1D analysis filter bank (along one dimension only) of an image

    Parameters
    ----------
    x (array): 4D input with the last two dimensions the spatial input
    h0 (array): 4D input for the lowpass filter. Should have shape (1, 1,
        h, 1) or (1, 1, 1, w)
    h1 (array): 4D input for the highpass filter. Should have shape (1, 1,
        h, 1) or (1, 1, 1, w)
    dim (int) - dimension of filtering. d=2 is for a vertical filter (called
        column filtering but filters across the rows). d=3 is for a
        horizontal filter, (called row filtering but filters across the
        columns).

    Returns
    -------
    lohi: lowpass and highpass subbands concatenated along the channel
        dimension
    """
    C = x.shape[1] if not mem_stack else x[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d] if not mem_stack else x[d]
    L = h0.size
    shape = [1, 1, 1, 1]
    shape[d] = L
    h = np.concatenate([h0.reshape(*shape), h1.reshape(*shape)] * C, axis=0)
    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode="symmetric")
    p = 2 * (outsize - 1) - N + L
    pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
    padded_x = _mypad(x, pad=pad, mem_stack=mem_stack)
    lohi = _conv2d(padded_x, h, stride=s, groups=C, mem_stack=mem_stack)
    if mem_stack:
        mem_stack.free(np.prod(padded_x) * np.float32().itemsize)
    del padded_x
    return lohi


def _sfb1d(
    lo: cp.ndarray,
    hi: cp.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    dim: int,
    mem_stack: Optional[_DeviceMemStack],
) -> cp.ndarray:
    """1D synthesis filter bank of an image Array"""

    C = lo.shape[1] if not mem_stack else lo[1]
    d = dim % 4
    L = g0.size
    shape = [1, 1, 1, 1]
    shape[d] = L
    s = (2, 1) if d == 2 else (1, 2)
    g0 = np.concatenate([g0.reshape(*shape)] * C, axis=0)
    g1 = np.concatenate([g1.reshape(*shape)] * C, axis=0)
    pad = (L - 2, 0) if d == 2 else (0, L - 2)
    y_lo = _conv_transpose2d(lo, g0, stride=s, pad=pad, groups=C, mem_stack=mem_stack)
    y_hi = _conv_transpose2d(hi, g1, stride=s, pad=pad, groups=C, mem_stack=mem_stack)
    if mem_stack:
        # Allocation of the sum
        mem_stack.malloc(np.prod(y_hi) * np.float32().itemsize)
        mem_stack.free(np.prod(y_lo) * np.float32().itemsize)
        mem_stack.free(np.prod(y_hi) * np.float32().itemsize)
        return y_lo
    return y_lo + y_hi


class _DWTForward:
    """Performs a 2d DWT Forward decomposition of an image

    Args:
        wave (str): Which wavelet to use.
    """

    def __init__(self, wave: str):
        super().__init__()

        wave = pywt.Wavelet(wave)
        h0_col, h1_col = wave.dec_lo, wave.dec_hi
        h0_row, h1_row = h0_col, h1_col

        self.h0_col = np.array(h0_col).astype("float32")[::-1].reshape((1, 1, -1, 1))
        self.h1_col = np.array(h1_col).astype("float32")[::-1].reshape((1, 1, -1, 1))
        self.h0_row = np.array(h0_row).astype("float32")[::-1].reshape((1, 1, 1, -1))
        self.h1_row = np.array(h1_row).astype("float32")[::-1].reshape((1, 1, 1, -1))

    def apply(
        self, x: cp.ndarray, mem_stack: Optional[_DeviceMemStack] = None
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Forward pass of the DWT.

        Args:
            x (array): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        # Do a multilevel transform
        # Do 1 level of the transform
        lohi = _afb1d(x, self.h0_row, self.h1_row, dim=3, mem_stack=mem_stack)
        y = _afb1d(lohi, self.h0_col, self.h1_col, dim=2, mem_stack=mem_stack)
        if mem_stack:
            y_shape = [y[0], np.prod(y) // y[0] // 4 // y[-2] // y[-1], 4, y[-2], y[-1]]
            x_shape = [y_shape[0], y_shape[1], y_shape[3], y_shape[4]]
            yh_shape = [y_shape[0], y_shape[1], y_shape[2] - 1, y_shape[3], y_shape[4]]

            mem_stack.free(np.prod(lohi) * np.float32().itemsize)
            mem_stack.malloc(np.prod(x_shape) * np.float32().itemsize)
            mem_stack.malloc(np.prod(yh_shape) * np.float32().itemsize)
            mem_stack.free(np.prod(y) * np.float32().itemsize)
            return x_shape, yh_shape
        del lohi
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        x = cp.ascontiguousarray(y[:, :, 0])
        yh = cp.ascontiguousarray(y[:, :, 1:])
        return (x, yh)


class _DWTInverse:
    """Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str): Which wavelet to use.
    """

    def __init__(self, wave: str):
        super().__init__()
        wave = pywt.Wavelet(wave)
        g0_col, g1_col = wave.rec_lo, wave.rec_hi
        g0_row, g1_row = g0_col, g1_col
        # Prepare the filters
        self.g0_col = np.array(g0_col).astype("float32").reshape((1, 1, -1, 1))
        self.g1_col = np.array(g1_col).astype("float32").reshape((1, 1, -1, 1))
        self.g0_row = np.array(g0_row).astype("float32").reshape((1, 1, 1, -1))
        self.g1_row = np.array(g1_row).astype("float32").reshape((1, 1, 1, -1))

    def apply(
        self,
        coeffs: Tuple[cp.ndarray, cp.ndarray],
        mem_stack: Optional[_DeviceMemStack] = None,
    ) -> cp.ndarray:
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass array of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass arrays of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        """
        yl, yh = coeffs
        lh = yh[:, :, 0, :, :] if not mem_stack else [yh[0], yh[1], yh[3], yh[4]]
        hl = yh[:, :, 1, :, :] if not mem_stack else [yh[0], yh[1], yh[3], yh[4]]
        hh = yh[:, :, 2, :, :] if not mem_stack else [yh[0], yh[1], yh[3], yh[4]]
        lo = _sfb1d(yl, lh, self.g0_col, self.g1_col, dim=2, mem_stack=mem_stack)
        hi = _sfb1d(hl, hh, self.g0_col, self.g1_col, dim=2, mem_stack=mem_stack)
        yl = _sfb1d(lo, hi, self.g0_row, self.g1_row, dim=3, mem_stack=mem_stack)
        if mem_stack:
            mem_stack.free(np.prod(lo) * np.float32().itemsize)
            mem_stack.free(np.prod(hi) * np.float32().itemsize)
        del lo
        del hi
        return yl


def _repair_memory_fragmentation_if_needed(fragmentation_threshold: float = 0.2):
    pool = cp.get_default_memory_pool()
    total = pool.total_bytes()
    if (total / pool.used_bytes()) - 1 > fragmentation_threshold:
        pool.free_all_blocks()


def remove_stripe_fw(
    data: cp.ndarray,
    sigma: float = 2,
    wname: str = "db5",
    level: Optional[int] = None,
    calc_peak_gpu_mem: bool = False,
) -> cp.ndarray:
    """
    Remove horizontal stripes from sinogram using the Fourier-Wavelet (FW) based method :cite:`munch2009stripe`. The original source code
    taken from TomoCupy and NABU packages.

    Parameters
    ----------
    data : ndarray
        3D tomographic data as a CuPy array.
    sigma : float
        Damping parameter in Fourier space.
    wname : str
        Type of the wavelet filter: select from 'db5', 'db7', 'haar', 'sym5', 'sym16' 'bior4.4'.
    level : int, optional
        Number of discrete wavelet transform levels.
    calc_peak_gpu_mem: str:
        Parameter to support memory estimation in HTTomo. Irrelevant to the method itself and can be ignored by user.

    Returns
    -------
    ndarray
        Stripe-corrected 3D tomographic data as a CuPy array.
    """

    if level is None:
        if calc_peak_gpu_mem:
            size = np.max(data)  # data is a tuple in this case
        else:
            size = np.max(data.shape)
        level = int(np.ceil(np.log2(size)))

    [nproj, nz, ni] = data.shape if not calc_peak_gpu_mem else data

    nproj_pad = nproj + nproj // 8

    # Accepts all wave types available to PyWavelets
    xfm = _DWTForward(wave=wname)
    ifm = _DWTInverse(wave=wname)

    # Wavelet decomposition.
    cc = []
    sli_shape = [nz, 1, nproj_pad, ni]

    if calc_peak_gpu_mem:
        mem_stack = _DeviceMemStack()
        # A data copy is assumed when invoking the function
        mem_stack.malloc(np.prod(data) * np.float32().itemsize)
        mem_stack.malloc(np.prod(sli_shape) * np.float32().itemsize)
        cc = []
        fcV_bytes = None
        for k in range(level):
            new_sli_shape, c = xfm.apply(sli_shape, mem_stack)
            mem_stack.free(np.prod(sli_shape) * np.float32().itemsize)
            sli_shape = new_sli_shape
            cc.append(c)

            if fcV_bytes:
                mem_stack.free(fcV_bytes)
            fcV_shape = [c[0], c[3], c[4]]
            fcV_bytes = np.prod(fcV_shape) * np.complex64().itemsize
            mem_stack.malloc(fcV_bytes)

            # For the FFT
            mem_stack.malloc(2 * np.prod(fcV_shape) * np.float32().itemsize)
            mem_stack.malloc(2 * fcV_bytes)

            fft_dummy = cp.empty(fcV_shape, dtype="float32")
            fft_plan = get_fft_plan(fft_dummy)
            fft_plan_size = fft_plan.work_area.mem.size
            del fft_dummy
            del fft_plan
            mem_stack.malloc(fft_plan_size)
            mem_stack.free(2 * np.prod(fcV_shape) * np.float32().itemsize)
            mem_stack.free(fft_plan_size)
            mem_stack.free(2 * fcV_bytes)

            # The rest of the iteration doesn't contribute to the peak
        # NOTE: The last iteration of fcV is "leaked"

        for k in range(level)[::-1]:
            new_sli_shape = [sli_shape[0], sli_shape[1], cc[k][-2], cc[k][-1]]
            new_sli_shape = ifm.apply((new_sli_shape, cc[k]), mem_stack)
            mem_stack.free(np.prod(sli_shape) * np.float32().itemsize)
            sli_shape = new_sli_shape

        mem_stack.malloc(np.prod(data) * np.float32().itemsize)
        for c in cc:
            mem_stack.free(np.prod(c) * np.float32().itemsize)
        mem_stack.free(np.prod(sli_shape) * np.float32().itemsize)
        return int(mem_stack.highwater * 1.1)

    sli = cp.zeros(sli_shape, dtype="float32")
    sli[:, 0, (nproj_pad - nproj) // 2 : (nproj_pad + nproj) // 2] = data.swapaxes(0, 1)
    for k in range(level):
        sli, c = xfm.apply(sli)
        cc.append(c)
        # FFT
        fft_in = cp.ascontiguousarray(cc[k][:, 0, 1])
        fft_plan = get_fft_plan(fft_in, axes=1)
        with fft_plan:
            fcV = cp.fft.fft(fft_in, axis=1)
        del fft_plan
        del fft_in
        _, my, mx = fcV.shape
        # Damping of ring artifact information.
        y_hat = np.fft.ifftshift((np.arange(-my, my, 2) + 1) / 2)
        damp = -np.expm1(-(y_hat**2) / (2 * sigma**2))
        fcV *= cp.tile(damp, (mx, 1)).swapaxes(0, 1)
        # Inverse FFT.
        ifft_in = cp.ascontiguousarray(fcV)
        ifft_plan = get_fft_plan(ifft_in, axes=1)
        with ifft_plan:
            cc[k][:, 0, 1] = cp.fft.ifft(ifft_in, my, axis=1).real
        del ifft_plan
        del ifft_in
        _repair_memory_fragmentation_if_needed()

    # Wavelet reconstruction.
    for k in range(level)[::-1]:
        shape0 = cc[k][0, 0, 1].shape
        sli = sli[:, :, : shape0[0], : shape0[1]]
        sli = ifm.apply((sli, cc[k]))
        _repair_memory_fragmentation_if_needed()

    data = sli[:, 0, (nproj_pad - nproj) // 2 : (nproj_pad + nproj) // 2, :ni]
    data = data.swapaxes(0, 1)
    return cp.ascontiguousarray(data)


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
    approach, see :ref:`method_remove_all_stripe` and :cite:`vo2018superior` (combination of algorithm 3,4,5, and 6).

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
        sino = _rs_sort(sino, sm_size, dim)
        sino = cp.nan_to_num(sino)
        data[:, m, :] = sino
    return data


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

    ids = cp.argsort(matcombine[:, :, 1], axis=1)
    matsort = matcombine.copy()
    matsort[:, :, 0] = cp.take_along_axis(matsort[:, :, 0], ids, axis=1)
    matsort[:, :, 1] = cp.take_along_axis(matsort[:, :, 1], ids, axis=1)

    matsort[:, :, 1] = cp.transpose(sinosmooth)
    ids = cp.argsort(matsort[:, :, 0], axis=1)
    matsortback = matsort.copy()
    matsortback[:, :, 0] = cp.take_along_axis(matsortback[:, :, 0], ids, axis=1)
    matsortback[:, :, 1] = cp.take_along_axis(matsortback[:, :, 1], ids, axis=1)

    sino_corrected = cp.transpose(matsortback[:, :, 1])
    listxmiss = cp.where(listmask > 0.0)[0]
    sinogram[:, listxmiss] = sino_corrected[:, listxmiss]
    return sinogram


def _rs_dead(sinogram, snr, size, matindex, norm=True):
    """remove unresponsive and fluctuating stripes"""
    sinogram = cp.copy(sinogram)  # Make it mutable
    (nrow, _) = sinogram.shape
    sinosmooth = uniform_filter1d(sinogram, 10, axis=0)

    listdiff = cp.sum(cp.abs(sinogram - sinosmooth), axis=0)
    listdiffbck = median_filter(listdiff, size)

    listfact = listdiff / listdiffbck

    listmask = _detect_stripe(listfact, snr)
    del listfact
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    listmask[0:2] = 0.0
    listmask[-2:] = 0.0

    listx = cp.where(listmask < 1.0)[0]
    listxmiss = cp.where(listmask > 0.0)[0]
    del listmask

    if len(listxmiss) > 0:
        ids = cp.searchsorted(listx, listxmiss)
        weights = (listxmiss - listx[ids - 1]) / (listx[ids] - listx[ids - 1])
        # direct interpolation without making an extra copy
        sinogram[:, listxmiss] = sinogram[:, listx[ids - 1]] + weights * (
            sinogram[:, listx[ids]] - sinogram[:, listx[ids - 1]]
        )

    # Remove residual stripes
    if norm is True:
        sinogram = _rs_large(sinogram, snr, size, matindex)
    return sinogram


def raven_filter(
    data: cp.ndarray,
    pad_y: int = 20,
    pad_x: int = 20,
    pad_method: str = "edge",
    uvalue: int = 20,
    nvalue: int = 4,
    vvalue: int = 2,
) -> cp.ndarray:
    """
    Applies FFT-based Raven filter :cite:`raven1998numerical` to a 3D CuPy array. For more detailed information, see :ref:`method_raven_filter`.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array either float32 or uint16 data type.

    pad_y : int, optional
        Pad the top and bottom of projections.

    pad_x : int, optional
        Pad the left and right of projections.

    pad_method : str, optional
        Numpy pad method to use.

    uvalue : int, optional
        Cut-off frequency. To control the strength of filter, e.g., strong=10, moderate=20, weak=50.

    nvalue : int, optional
        The shape of filter.

    vvalue : int, optional
        Number of image-rows around the zero-frequency to be applied the filter.

    Returns
    -------
    cp.ndarray
        Raven filtered 3D CuPy array in float32 data type.

    Raises
    ------
    ValueError
        If the input array is not three dimensional.
    """
    if data.dtype != cp.float32:
        raise ValueError("The input data should be float32 data type")

    # Padding of the sinogram
    data = cp.pad(data, ((pad_y, pad_y), (0, 0), (pad_x, pad_x)), mode=pad_method)

    # FFT and shift of sinogram
    fft_data = fft2(data, axes=(0, 2), overwrite_x=True)
    fft_data_shifted = fftshift(fft_data, axes=(0, 2))

    # Calculation type
    calc_type = fft_data_shifted.dtype

    # Setup various values for the filter
    height, images, width = data.shape

    # Set the input type of the kernel
    kernel_args = "raven_filter<{0}>".format(
        "float" if calc_type == "complex64" else "double"
    )

    # setting grid/block parameters
    block_x = 128
    block_dims = (block_x, 1, 1)
    grid_x = (width + block_x - 1) // block_x
    grid_y = images
    grid_z = height
    grid_dims = (grid_x, grid_y, grid_z)
    params = (fft_data_shifted, fft_data, width, images, height, uvalue, nvalue, vvalue)

    raven_module = load_cuda_module("raven_filter", name_expressions=[kernel_args])
    raven_filt = raven_module.get_function(kernel_args)

    raven_filt(grid_dims, block_dims, params)
    del fft_data_shifted

    # raven_filt already doing ifftshifting
    data = ifft2(fft_data, axes=(0, 2), overwrite_x=True)

    # Removing padding
    data = data[pad_y : height - pad_y, :, pad_x : width - pad_x].real

    return cp.require(data, requirements="C")


def _create_matindex(nrow, ncol):
    """
    Create a 2D array of indexes used for the sorting technique.
    """
    listindex = cp.arange(0.0, ncol, 1.0)
    matindex = cp.tile(listindex, (nrow, 1))
    return matindex.astype(np.float32)
