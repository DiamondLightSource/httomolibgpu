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
# Created Date: 21/October/2022
# ---------------------------------------------------------------------------
""" Module for data correction. For more detailed information see :ref:`data_correction_module`.

"""

import numpy as np
from typing import Union

from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from numpy import float32
from unittest.mock import Mock

if cupy_run:
    from httomolibgpu.cuda_kernels import load_cuda_module
    from cupyx.scipy.fft import fft2, ifft2, fftshift, ifftshift
else:
    load_cuda_module = Mock()
    fft2 = Mock()
    ifft2 = Mock()
    fftshift = Mock()
    ifftshift = Mock()


__all__ = [
    "raven_filter",
]


def raven_filter_savu(
    data: cp.ndarray,
    kernel_size: int = 3,
    pad_y: int = 100,
    pad_x: int = 100,
    pad_method: str = "edge",
) -> cp.ndarray:
    """
    Applies raven filter to a 3D CuPy array. For more detailed information, see :ref:`method_raven_filter`.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array either float32 or uint16 data type.
    kernel_size : int, optional
        The size of the filter's kernel (a diameter).

    pad_y : int, optional
        Pad the top and bottom of projections.

    pad_x : int, optional
        Pad the left and right of projections.

    pad_method : str, optional
        Numpy pad method to use.

    Returns
    -------
    ndarray
        Raven filtered 3D CuPy array either float32 or uint16 data type.

    Raises
    ------
    ValueError
        If the input array is not three dimensional.
    """
    input_type = data.dtype

    if input_type not in ["float32", "uint16"]:
        raise ValueError("The input data should be either float32 or uint16 data type")

    if data.ndim == 3:
        if 0 in data.shape:
            raise ValueError("The length of one of dimensions is equal to zero")
    else:
        raise ValueError("The input array must be a 3D array")

    if kernel_size not in [3, 5, 7, 9, 11, 13]:
        raise ValueError("Please select a correct kernel size: 3, 5, 7, 9, 11, 13")


    dz_orig, dy_orig, dx_orig = data.shape

    padded_data, pad_tup = cp.pad(data, fftpad, "edge")
    dz, dy, dx = padded_data.shape

    # 3D FFT of data
    padded_data = cp.pad(data, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode=pad_method)
    fft_data = fft2(padded_data, axes=(-2, -1), overwrite_x=True)
    fft_data_shifted = fftshift(fft_data)

    # Setup various values for the filter
    _, height, width = data.shape

    height1 = height + 2 * pad_y
    width1 = width + 2 * pad_x

    # raven
    kernel_args = "raven_general_kernel3d<{0}, {1}>".format(
        "float" if input_type == "float32" else "unsigned short", kernel_size
    )
    block_x = 128
    # setting grid/block parameters
    block_dims = (block_x, 1, 1)
    grid_x = (dx + block_x - 1) // block_x
    grid_y = dy
    grid_z = dz
    grid_dims = (grid_x, grid_y, grid_z)
    params = (fft_data_shifted, dz, dy, dx)

    raven_module = load_cuda_module("raven_kernel", name_expressions=[kernel_args])
    raven_filt = raven_module.get_function(kernel_args)

    raven_filt(grid_dims, block_dims, params)

    fft_data = fftshift(fft_data_shifted)

    data = ifft2(fft_data, axes=(-2, -1), overwrite_x=True, norm="forward")

    return data
