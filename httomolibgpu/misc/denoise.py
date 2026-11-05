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
# Created Date: 18/December/2024
# ---------------------------------------------------------------------------
"""Module for data denoising. For more detailed information see :ref:`data_denoising_module`:

* :mod:`httomolibgpu.misc.denoise.total_variation_ROF`

* :mod:`httomolibgpu.misc.denoise.total_variation_PD`

* :mod:`httomolibgpu.misc.denoise.non_local_means`

"""

from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from tomobar.regularisersCuPy import ROF_TV_cupy, PD_TV_cupy
    from httomolibgpu.cuda_kernels import load_cuda_module
else:
    ROF_TV = Mock()
    PD_TV = Mock()
    load_cuda_module = Mock()

from typing import Union
from math import isfinite
from numbers import Integral, Real

import numpy as np

from httomolibgpu.misc.utils import (
    __check_variable_type,
    __check_if_data_3D_array,
    __check_if_data_correct_type,
)

__all__ = [
    "non_local_means",
    "total_variation_ROF",
    "total_variation_PD",
]


def non_local_means(
    data: cp.ndarray,
    h: float = 0.1,
    search_radius: int = 5,
    patch_radius: int = 2,
    num_neighbours: int = 0,
) -> cp.ndarray:
    """Denoise images with Gaussian-weighted non-local means on the GPU.

    A 3D input is treated as independent 2D images along axis 0. This uses
    the patch-distance model of CCPi's PatchSelect_GPU, followed by a
    normalised weighted average. See :ref:`method_non_local_means`.

    Parameters
    ----------
    data : cp.ndarray
        Non-empty float32 CuPy image (y, x) or stack (n, y, x). Values must
        be finite. Non-contiguous inputs are supported and are not modified.
    h : float
        Positive, finite smoothing strength in input intensity units.
        Larger values give more smoothing. No intensity rescaling is applied.
        The Gaussian-weighted squared patch differences are summed, not
        averaged, so the appropriate h also depends on patch_radius.
    search_radius : int
        Non-negative half-width of the square search window, in pixels.
    patch_radius : int
        Non-negative half-width of the comparison patch. Its Gaussian has
        sigma equal to patch_radius. Zero compares individual pixels.
    num_neighbours : int
        Number of strongest non-self matches to average, plus the centre
        pixel with weight one. Zero uses all matches. A positive value must
        not exceed (2 * search_radius + 1)**2 - 1. Near image boundaries,
        fewer neighbours may be available.

    Returns
    -------
    cp.ndarray
        New C-contiguous float32 array with the input shape and device.

    Notes
    -----
    Candidate centres outside the image are excluded. Patch comparisons use
    only offsets valid in both patches, with Gaussian weights attached to
    their original offsets; the truncated weights are not renormalised.
    Identical patches remain valid matches. These boundary and zero-distance
    rules intentionally differ from the linked CCPi implementation.

    The kernel runs on the input device and current CuPy stream. It retains
    no full neighbour graph: workspace consists of a contiguous input copy
    if needed, the output, Gaussian weights and per-thread selection storage.
    Runtime grows with search-window area times patch area; requesting many
    neighbours also increases per-thread storage and selection cost.

    Raises
    ------
    TypeError
        If data is not a CuPy array, or a parameter has an invalid type.
    ValueError
        If shape, dtype, or a parameter value is invalid.
    """
    if not cupy_run:
        raise RuntimeError("non_local_means requires CuPy and an accessible CUDA GPU")
    if not isinstance(data, cp.ndarray):
        raise TypeError("data must be a CuPy array")
    if data.ndim not in (2, 3) or any(size == 0 for size in data.shape):
        raise ValueError("data must be a non-empty 2D image or 3D stack")
    if data.dtype != cp.float32:
        raise ValueError("data must have dtype float32")
    if isinstance(h, bool) or not isinstance(h, Real):
        raise TypeError("h must be a real number")
    h = float(h)
    if not isfinite(h) or h <= 0 or not 0 < h * h < float("inf"):
        raise ValueError("h must be positive with a finite, non-zero square")
    for name, value in (
        ("search_radius", search_radius),
        ("patch_radius", patch_radius),
        ("num_neighbours", num_neighbours),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 0 or value > np.iinfo(np.int32).max:
            raise ValueError(f"{name} must be a non-negative int32 value")
    if num_neighbours > (2 * int(search_radius) + 1) ** 2 - 1:
        raise ValueError("num_neighbours exceeds the number of non-self candidates")
    height, width = data.shape[-2:]
    if max(height, width) > np.iinfo(np.int32).max:
        raise ValueError("image dimensions must fit int32")

    # Offsets beyond the image extent can never contribute to any patch.
    patch_extent = min(int(patch_radius), max(height, width) - 1)
    search_extent = min(int(search_radius), max(height, width) - 1)
    neighbours = min(int(num_neighbours), height * width - 1)
    with data.device:
        source = cp.ascontiguousarray(data)
        output = cp.empty(data.shape, dtype=cp.float32)
        offsets = cp.arange(-patch_extent, patch_extent + 1, dtype=cp.float32)
        if patch_radius:
            gaussian = cp.exp(-(offsets**2) / (2.0 * float(patch_radius) ** 2))
        else:
            gaussian = cp.ones(1, dtype=cp.float32)
        kernel_name = f"non_local_means_kernel<{neighbours}>"
        module = load_cuda_module("non_local_means", name_expressions=[kernel_name])
        kernel = module.get_function(kernel_name)
        block = (128,)
        grid = ((data.size + block[0] - 1) // block[0],)
        kernel(
            grid,
            block,
            (
                source,
                output,
                gaussian,
                np.int64(data.size),
                np.int32(height),
                np.int32(width),
                np.int32(search_extent),
                np.int32(patch_extent),
                np.float64(h * h),
            ),
        )
    return output


def total_variation_ROF(
    data: cp.ndarray,
    regularisation_parameter: Union[float, int] = 1e-05,
    iterations: int = 3000,
    time_marching_parameter: float = 0.001,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """
    Total Variation using Rudin-Osher-Fatemi (ROF) :cite:`rudin1992nonlinear` explicit iteration scheme to perform edge-preserving image denoising.
    This is a gradient-based algorithm for a smoothed TV term which requires a small time marching parameter and a significant number of iterations.
    See more in :ref:`method_total_variation_ROF`.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array of float32 data type.
    regularisation_parameter : float
        Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
    iterations : int
        The number of iterations. Defaults to 3000.
    time_marching_parameter : float
        Time marching parameter, needs to be small to ensure convergence. Defaults to 0.001.
    gpu_id : int
        GPU device index to perform processing on. Defaults to 0.
    half_precision : bool
        Perform faster computation in half-precision with a very minimal sacrifice in quality. Defaults to False.

    Returns
    -------
    ndarray
        TV-ROF filtered 3D CuPy array in float32 data type.

    Raises
    ------
    ValueError
        If the input array is not float32 data type.
    """
    ### Data and parameters checks ###
    methods_name = "total_variation_ROF"
    __check_if_data_3D_array(data, methods_name)
    __check_if_data_correct_type(
        data, accepted_type=["float32"], methods_name=methods_name
    )
    __check_variable_type(
        regularisation_parameter,
        [float, int],
        "regularisation_parameter",
        [],
        methods_name,
    )
    __check_variable_type(iterations, [int], "iterations", [], methods_name)
    __check_variable_type(
        time_marching_parameter, [float], "time_marching_parameter", [], methods_name
    )
    __check_variable_type(gpu_id, [int], "gpu_id", [], methods_name)
    __check_variable_type(half_precision, [bool], "half_precision", [], methods_name)
    ###################################

    return ROF_TV_cupy(
        data,
        regularisation_parameter,
        iterations,
        time_marching_parameter,
        gpu_id,
        half_precision,
    )


def total_variation_PD(
    data: cp.ndarray,
    regularisation_parameter: Union[float, int] = 1e-05,
    iterations: int = 1000,
    isotropic: bool = True,
    nonnegativity: bool = False,
    lipschitz_const: Union[float, int] = 8.0,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """
    Primal Dual algorithm for non-smooth convex Total Variation functional :cite:`chan1999nonlinear`. See more in :ref:`method_total_variation_PD`.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array of float32 data type.
    regularisation_parameter : float, int
        Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
    iterations : int
        The number of iterations. Defaults to 1000.
    isotropic : bool
        Choose between isotropic or anisotropic TV norm. Defaults to isotropic.
    nonnegativity  : bool
        Enable non-negativity in iterations. Defaults to False.
    lipschitz_const : float,int
        Lipschitz constant to control convergence. Defaults to 8.
    gpu_id : int
        GPU device index to perform processing on. Defaults to 0.
    half_precision : bool
        Perform faster computation in half-precision with a very minimal sacrifice in quality. Defaults to False.

    Returns
    -------
    ndarray
        TV-PD filtered 3D CuPy array in float32 data type.

    Raises
    ------
    ValueError
        If the input array is not float32 data type.
    """

    ### Data and parameters checks ###
    methods_name = "total_variation_PD"
    __check_if_data_3D_array(data, methods_name)
    __check_if_data_correct_type(
        data, accepted_type=["float32"], methods_name=methods_name
    )
    __check_variable_type(
        regularisation_parameter,
        [float, int],
        "regularisation_parameter",
        [],
        methods_name,
    )
    __check_variable_type(iterations, [int], "iterations", [], methods_name)
    __check_variable_type(isotropic, [bool], "isotropic", [], methods_name)
    __check_variable_type(nonnegativity, [bool], "nonnegativity", [], methods_name)
    __check_variable_type(
        lipschitz_const, [float, int], "lipschitz_const", [], methods_name
    )
    __check_variable_type(gpu_id, [int], "gpu_id", [], methods_name)
    __check_variable_type(half_precision, [bool], "half_precision", [], methods_name)
    ###################################

    methodTV = 0
    if not isotropic:
        methodTV = 1

    nonneg = 0
    if nonnegativity:
        nonneg = 1

    return PD_TV_cupy(
        data,
        regularisation_parameter,
        iterations,
        methodTV,
        nonneg,
        lipschitz_const,
        gpu_id,
        half_precision,
    )
