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


def raven_filter(
    data: cp.ndarray,
    uvalue: int = 20,
    nvalue: int = 4,
    vvalue: int = 2,
    pad_y: int = 20,
    pad_x: int = 20,
    pad_method: str = "edge",
) -> cp.ndarray:
    """
    Applies raven filter to a 3D CuPy array. For more detailed information, see :ref:`method_raven_filter`.

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
        The shape of filter.

    nvalue : int, optional
        The shape of filter.

    vvalue : int, optional
        The number of rows to be applied the filter

    Returns
    -------
    ndarray
        Raven filtered 3D CuPy array in float32 data type.

    Raises
    ------
    ValueError
        If the input array is not three dimensional.
    """
    input_type = data.dtype

    if input_type not in ["float32"]:
        raise ValueError("The input data should be float32")

    # if data.ndim != 3:
    #     raise ValueError("only 3D data is supported")

    # Padding of the data
    padded_data = cp.pad(data, ((pad_y, pad_y), (pad_x, pad_x)), mode=pad_method)

    # FFT and shift of data
    fft_data = fft2(padded_data, axes=(-2, -1), overwrite_x=True)
    fft_data_shifted = fftshift(fft_data)

    # Setup various values for the filter
    height, width = data.shape

    height1 = height + 2 * pad_y
    width1 = width + 2 * pad_x

    # setting grid/block parameters
    block_x = 128
    block_dims = (block_x, 1, 1)
    grid_x = (width1 + block_x - 1) // block_x
    grid_y = height1
    grid_dims = (grid_x, grid_y, 1)
    params = (fft_data_shifted, fft_data, width1, height1, uvalue, nvalue, vvalue)

    raven_module = load_cuda_module("raven_filter")
    raven_filt = raven_module.get_function("raven_filter")
    
    raven_filt(grid_dims, block_dims, params)
    
    # raven_fil already doing ifftshifting
    # fft_data = ifftshift(fft_data_shifted)
    data = ifft2(fft_data, axes=(-2, -1), overwrite_x=True)

    return data
