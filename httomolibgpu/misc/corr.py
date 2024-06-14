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
""" Module for data correction """

import numpy as np
from httomolibgpu import cupywrapper

cp = cupywrapper.cp

nvtx = cupywrapper.nvtx
from numpy import float32

__all__ = [
    "median_filter",
    "remove_outlier",
]


def median_filter(
    data: cp.ndarray,
    kernel_size: int = 3,
    axis: int = 0,
    dif: float = 0.0,
) -> cp.ndarray:
    """
    Apply 2D or 3D median or dezinger (when dif>0) filter to a 3D array.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array either float32 or uint16 data type.
    kernel_size : int, optional
        The size of the filter's kernel (a diameter).
    axis (int, optional):
        Axis along which the 2D filter kernel should be applied. If set to None, then the kernel is 3D.
    dif : float, optional
        Expected difference value between outlier value and the
        median value of the array, leave equal to 0 for classical median.

    Returns
    -------
    ndarray
        Median filtered 3D CuPy array either float32 or uint16 data type.

    Raises
    ------
    ValueError
        If the input array is not three dimensional.
    """
    if cupywrapper.cupy_run:
        return __median_filter(data, kernel_size, axis, dif)
    else:
        print("median_filter won't be executed because CuPy is not installed")
        return data


@nvtx.annotate()
def __median_filter(
    data: cp.ndarray,
    kernel_size: int = 3,
    axis: int = 0,
    dif: float = 0.0,
) -> cp.ndarray:
    try:
        from cucim.skimage.filters import median
        from cucim.skimage.morphology import disk
    except ImportError:
        print(
            "Cucim library of Rapidsai is a required dependency for median_filter and remove_outlier modules, please install"
        )
    from httomolibgpu.cuda_kernels import load_cuda_module

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

    if axis not in [0, 1, 2, None]:
        raise ValueError("The axis should be 0,1,2 or None for full 3d processing")

    dz, dy, dx = data.shape
    output = cp.empty(data.shape, dtype=input_type, order="C")

    if axis == 0:
        for j in range(dz):
            median(data[j, :, :], footprint=disk(kernel_size // 2), out=output[j, :, :])
    elif axis == 1:
        for j in range(dy):
            median(data[:, j, :], footprint=disk(kernel_size // 2), out=output[:, j, :])
    elif axis == 2:
        for j in range(dx):
            median(data[:, :, j], footprint=disk(kernel_size // 2), out=output[:, :, j])
    else:
        # 3d median or dezinger
        kernel_args = "median_general_kernel3d<{0}, {1}>".format(
            "float" if input_type == "float32" else "unsigned short", kernel_size
        )
        block_x = 128
        # setting grid/block parameters
        block_dims = (block_x, 1, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_z = dz
        grid_dims = (grid_x, grid_y, grid_z)
        params = (data, output, dif, dz, dy, dx)

        median_module = load_cuda_module(
            "median_kernel", name_expressions=[kernel_args]
        )
        median_filt = median_module.get_function(kernel_args)

        median_filt(grid_dims, block_dims, params)

    if axis is not None and dif > 0:
        # 2d dezingering enabled
        kernel_name = "thresholding"
        kernel = r"""
            float dif_curr = abs(float(data) - float(output));
            if (dif_curr < dif) {
                output = data;
            }
            """
        thresholding_kernel = cp.ElementwiseKernel(
            "T data, raw float32 dif",
            "T output",
            kernel,
            kernel_name,
            options=("-std=c++11",),
            no_return=True,
        )
        thresholding_kernel(data, float32(dif), output)
    return output


def remove_outlier(
    data: cp.ndarray, kernel_size: int = 3, axis: int = 0, dif: float = 0.1
) -> cp.ndarray:
    """
    Selectively applies 3D median filter to a 3D array to remove outliers. Also called a dezinger.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array either float32 or uint16 data type.
    kernel_size : int, optional
        The size of the filter's kernel (a diameter).
    axis (int, optional):
        Axis along which the 2D filter kernel should be applied. If set to None, then the kernel is 3D.
    dif : float, optional
        Expected difference value between outlier value and the
        median value of the array.

    Returns
    -------
    ndarray
        Dezingered filtered 3D CuPy array either float32 or uint16 data type.

    Raises
    ------
    ValueError
        Threshold value (dif) must be positive and nonzero.
    """

    if dif <= 0.0:
        raise ValueError("Threshold value (dif) must be positive and nonzero.")

    if cupywrapper.cupy_run:
        return __median_filter(data, kernel_size, axis, dif)
    else:
        print("remove_outlier won't be executed because CuPy is not installed")
        return data
