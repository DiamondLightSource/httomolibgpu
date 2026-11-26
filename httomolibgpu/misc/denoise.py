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

* :func:`~httomolibgpu.misc.denoise.total_variation_ROF`

* :func:`~httomolibgpu.misc.denoise.total_variation_PD`

* :func:`~httomolibgpu.misc.denoise.total_variation_SB`

* :func:`~httomolibgpu.misc.denoise.total_variation_FGP`

* :func:`~httomolibgpu.misc.denoise.total_generalised_variation`

* :func:`~httomolibgpu.misc.denoise.nonlinear_diffusion`

* :func:`~httomolibgpu.misc.denoise.non_local_means`

"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Union

import numpy as np

from httomolibgpu import cupywrapper
from httomolibgpu.cuda_kernels import load_cuda_module
from httomolibgpu.misc.utils import (
    __check_if_data_3D_array,
    __check_if_data_correct_type,
    __check_if_data_ndim,
    __check_if_positive_nonzero,
    __check_variable_type,
)

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run


if cupy_run:
    from tomobar.regularisers import (
        FGP_TV_cupy,
        NDF_cupy,
        PD_TV_cupy,
        ROF_TV_cupy,
        SB_TV_cupy,
        TGV_cupy,
    )
else:
    ROF_TV_cupy = cupywrapper._UnavailableGPU("ROF_TV_cupy")
    PD_TV_cupy = cupywrapper._UnavailableGPU("PD_TV_cupy")
    SB_TV_cupy = cupywrapper._UnavailableGPU("SB_TV_cupy")
    FGP_TV_cupy = cupywrapper._UnavailableGPU("FGP_TV_cupy")
    TGV_cupy = cupywrapper._UnavailableGPU("TGV_cupy")
    NDF_cupy = cupywrapper._UnavailableGPU("NDF_cupy")


__all__ = [
    "non_local_means",
    "total_variation_ROF",
    "total_variation_PD",
    "total_variation_SB",
    "total_variation_FGP",
    "total_generalised_variation",
    "nonlinear_diffusion",
]


def _validate_non_local_means(
    data, h, search_radius, patch_radius, num_neighbours, full_3d
):
    """Validate NLM inputs using the common data and parameter checks."""
    methods_name = "non_local_means"
    if not cupy_run:
        raise RuntimeError("non_local_means requires CuPy and an accessible CUDA GPU")
    if not isinstance(data, cp.ndarray):
        raise TypeError("data must be a CuPy array")
    __check_if_data_ndim(data, (2, 3), methods_name)
    __check_if_data_correct_type(data, ["float32"], methods_name)
    __check_variable_type(
        full_3d, [bool], "full_3d", [], methods_name, type_error=TypeError
    )
    volume_mode = full_3d and data.ndim == 3
    spatial_shape = data.shape if volume_mode else data.shape[-2:]
    __check_variable_type(h, [Real], "h", [], methods_name, type_error=TypeError)
    h = float(h)
    __check_if_positive_nonzero(h, "h", True, True, methods_name)
    if not isfinite(h * h) or h * h == 0:
        raise ValueError("h must have a finite, non-zero square")
    for name, value in (
        ("search_radius", search_radius),
        ("patch_radius", patch_radius),
        ("num_neighbours", num_neighbours),
    ):
        __check_variable_type(
            value, [Integral], name, [], methods_name, type_error=TypeError
        )
        __check_if_positive_nonzero(value, name, True, False, methods_name)
        if value > np.iinfo(np.int32).max:
            raise ValueError(f"{name} must fit int32")
    if num_neighbours > (2 * int(search_radius) + 1) ** len(spatial_shape) - 1:
        raise ValueError("num_neighbours exceeds the number of non-self candidates")
    height, width = data.shape[-2:]
    depth = data.shape[0] if data.ndim == 3 else 1
    if max(depth, height, width) > np.iinfo(np.int32).max:
        raise ValueError("image dimensions must fit int32")

    return h, volume_mode, spatial_shape


def non_local_means(
    data: cp.ndarray,
    h: float = 0.01,
    search_radius: int = 2,
    patch_radius: int = 1,
    num_neighbours: int = 0,
    full_3d: bool = True,
) -> cp.ndarray:
    """GPU non-local means denoising. See :ref:`method_non_local_means`.

    Parameters
    ----------
    data : cp.ndarray
        CuPy float32 image (Y, X) or volume (Z, Y, X); input is not modified.
    h : float
        Positive finite smoothing strength in intensity units. Patch distances
        are summed, not averaged; retune when changing patch size or dimension.
    search_radius : int
        Nonnegative search-window half-width; zero returns a copy.
    patch_radius : int
        Nonnegative patch half-width and Gaussian weighting sigma; zero compares pixels.
    num_neighbours : int
        Strongest non-self matches to average; 0 uses all. Positive values must
        not exceed (2 * search_radius + 1)**D - 1 for processing dimension D.
    full_3d : bool
        True uses cubic volume patches; False filters each Z slice independently.
        A 2D input always uses 2D processing.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 denoised array, same shape and device as input.

    Raises
    ------
    TypeError
        If data is not a CuPy array, or a parameter has an invalid type.
    ValueError
        If shape, dtype, or a parameter value is invalid.
    """
    h, volume_mode, spatial_shape = _validate_non_local_means(
        data, h, search_radius, patch_radius, num_neighbours, full_3d
    )
    height, width = data.shape[-2:]
    depth = data.shape[0] if data.ndim == 3 else 1

    # Offsets beyond the image extent can never contribute to any patch.
    patch_extent = min(int(patch_radius), max(spatial_shape) - 1)
    search_extent = min(int(search_radius), max(spatial_shape) - 1)
    max_candidates = 1
    for size in spatial_shape:
        max_candidates *= min(size, 2 * search_extent + 1)
    neighbours = min(int(num_neighbours), max_candidates - 1)
    with data.device:
        source = cp.ascontiguousarray(data)
        output = cp.empty(data.shape, dtype=cp.float32)
        offsets = cp.arange(-patch_extent, patch_extent + 1, dtype=cp.float32)
        if patch_radius:
            gaussian = cp.exp(-(offsets**2) / (2.0 * float(patch_radius) ** 2))
        else:
            gaussian = cp.ones(1, dtype=cp.float32)
        kernel_name = f"non_local_means_kernel<{neighbours}, {int(volume_mode)}>"
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
                np.int32(depth),
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
    """Smoothed ROF-TV denoising. See :ref:`method_total_variation_ROF`.

    Parameters
    ----------
    data : cp.ndarray
        Finite float32 CuPy volume (Z, Y, X); use (1, Y, X) for a 2D image.
        Input is not modified.
    regularisation_parameter : float
        Nonnegative smoothing strength; larger values smooth more. Zero disables it.
    iterations : int
        Positive fixed iteration count.
    time_marching_parameter : float
        Positive explicit time step; smaller values may need more iterations.
    gpu_id : int
        CUDA device index; input must reside on this device.
    half_precision : bool
        Use float16 auxiliary fields; image and output remain float32.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 denoised array, same shape and device as input.

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
    """Primal-dual TV denoising. See :ref:`method_total_variation_PD`.

    Parameters
    ----------
    data : cp.ndarray
        Finite float32 CuPy volume (Z, Y, X); use (1, Y, X) for a 2D image.
        Input is not modified.
    regularisation_parameter : float, int
        Nonnegative smoothing strength; larger values smooth more. Zero disables it.
    iterations : int
        Positive fixed iteration count.
    isotropic : bool
        True: isotropic TV; False: anisotropic TV.
    nonnegativity : bool
        Project image values to >= 0, including at zero strength.
    lipschitz_const : float,int
        Positive squared gradient-operator bound; minimum 4 * spatial_ndim.
    gpu_id : int
        CUDA device index; input must reside on this device.
    half_precision : bool
        Use float16 auxiliary fields; image and output remain float32.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 denoised array, same shape and device as input.

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


def _check_tomobar_denoiser(
    data,
    method,
    regularisation_parameter,
    iterations,
    nonnegativity,
    gpu_id,
    half_precision,
    parameters=(),
):
    """Share the PD-TV wrapper contract; ToMoBAR validates parameter ranges."""
    cupywrapper.require_gpu(method, device_index=gpu_id)
    if not isinstance(data, cp.ndarray):
        raise TypeError("data must be a CuPy array")
    __check_if_data_3D_array(data, method)
    __check_if_data_correct_type(data, ["float32"], method)
    for value, types, name in (
        (regularisation_parameter, [float, int], "regularisation_parameter"),
        (iterations, [int], "iterations"),
        (nonnegativity, [bool], "nonnegativity"),
        (gpu_id, [int], "gpu_id"),
        (half_precision, [bool], "half_precision"),
        *parameters,
    ):
        __check_variable_type(value, types, name, [], method)


def total_variation_SB(
    data: cp.ndarray,
    regularisation_parameter: Union[float, int] = 0.001,
    iterations: int = 150,
    isotropic: bool = True,
    nonnegativity: bool = False,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Split Bregman TV denoising. See :ref:`method_total_variation_SB`.

    Parameters
    ----------
    data : cupy.ndarray
        Finite float32 CuPy volume (Z, Y, X); use (1, Y, X) for a 2D image.
        Input is not modified.
    regularisation_parameter : float or int
        Nonnegative smoothing strength; larger values smooth more. Zero disables it.
    iterations : int
        Positive fixed iteration count.
    isotropic : bool
        True: isotropic TV; False: anisotropic TV.
    nonnegativity : bool
        Project image values to >= 0, including at zero strength.
    gpu_id : int
        CUDA device index; input must reside on this device.
    half_precision : bool
        Use float16 auxiliary fields; image and output remain float32.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 denoised array, same shape and device as input.
    """
    _check_tomobar_denoiser(
        data,
        "total_variation_SB",
        regularisation_parameter,
        iterations,
        nonnegativity,
        gpu_id,
        half_precision,
        ((isotropic, [bool], "isotropic"),),
    )
    return SB_TV_cupy(
        data,
        regularisation_parameter=regularisation_parameter,
        iterations=iterations,
        methodTV=0 if isotropic else 1,
        nonneg=int(nonnegativity),
        gpu_id=gpu_id,
        half_precision=half_precision,
    )


def total_variation_FGP(
    data: cp.ndarray,
    regularisation_parameter: Union[float, int] = 0.001,
    iterations: int = 150,
    isotropic: bool = True,
    nonnegativity: bool = False,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Fast Gradient Projection TV denoising. See :ref:`method_total_variation_FGP`.

    Parameters
    ----------
    data : cupy.ndarray
        Finite float32 CuPy volume (Z, Y, X); use (1, Y, X) for a 2D image.
        Input is not modified.
    regularisation_parameter : float or int
        Nonnegative smoothing strength; larger values smooth more. Zero disables it.
    iterations : int
        Positive fixed iteration count.
    isotropic : bool
        True: isotropic TV; False: anisotropic TV.
    nonnegativity : bool
        Project image values to >= 0, including at zero strength.
    gpu_id : int
        CUDA device index; input must reside on this device.
    half_precision : bool
        Use float16 auxiliary fields; image and output remain float32.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 denoised array, same shape and device as input.
    """
    _check_tomobar_denoiser(
        data,
        "total_variation_FGP",
        regularisation_parameter,
        iterations,
        nonnegativity,
        gpu_id,
        half_precision,
        ((isotropic, [bool], "isotropic"),),
    )
    return FGP_TV_cupy(
        data,
        regularisation_parameter=regularisation_parameter,
        iterations=iterations,
        methodTV=0 if isotropic else 1,
        nonneg=int(nonnegativity),
        gpu_id=gpu_id,
        half_precision=half_precision,
    )


def total_generalised_variation(
    data: cp.ndarray,
    regularisation_parameter: Union[float, int] = 0.001,
    iterations: int = 150,
    alpha1: float = 1.0,
    alpha0: float = 2.0,
    lipschitz_const: float = 12.0,
    nonnegativity: bool = False,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Second-order TGV denoising. See :ref:`method_total_generalised_variation`.

    Parameters
    ----------
    data : cupy.ndarray
        Finite float32 CuPy volume (Z, Y, X); use (1, Y, X) for a 2D image.
        Input is not modified.
    regularisation_parameter : float or int
        Nonnegative smoothing strength; larger values smooth more. Zero disables it.
    iterations : int
        Positive fixed iteration count.
    alpha1 : float
        Positive first-order TGV weight.
    alpha0 : float
        Positive second-order TGV weight.
    lipschitz_const : float
        Positive squared derivative-operator bound; minimum 8 * spatial_ndim + 2.
    nonnegativity : bool
        Project image values to >= 0, including at zero strength.
    gpu_id : int
        CUDA device index; input must reside on this device.
    half_precision : bool
        Use float16 auxiliary fields; image and output remain float32.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 denoised array, same shape and device as input.
    """
    _check_tomobar_denoiser(
        data,
        "total_generalised_variation",
        regularisation_parameter,
        iterations,
        nonnegativity,
        gpu_id,
        half_precision,
        (
            (alpha1, [float, int], "alpha1"),
            (alpha0, [float, int], "alpha0"),
            (lipschitz_const, [float, int], "lipschitz_const"),
        ),
    )
    return TGV_cupy(
        data,
        regularisation_parameter=regularisation_parameter,
        iterations=iterations,
        alpha1=alpha1,
        alpha0=alpha0,
        lipschitz_const=lipschitz_const,
        nonneg=int(nonnegativity),
        gpu_id=gpu_id,
        half_precision=half_precision,
    )


def nonlinear_diffusion(
    data: cp.ndarray,
    regularisation_parameter: Union[float, int] = 0.001,
    iterations: int = 150,
    sigma: float = 0.01,
    time_marching_parameter: float = 0.005,
    penalty_type: int = 1,
    nonnegativity: bool = False,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Edge-preserving diffusion denoising. See :ref:`method_nonlinear_diffusion`.

    Parameters
    ----------
    data : cupy.ndarray
        Finite float32 CuPy volume (Z, Y, X); use (1, Y, X) for a 2D image.
        Input is not modified.
    regularisation_parameter : float or int
        Nonnegative smoothing strength; larger values smooth more. Zero disables it.
    iterations : int
        Positive fixed iteration count.
    sigma : float
        Nonnegative edge parameter; effective threshold is sigma / sqrt(2).
        Zero selects linear diffusion.
    time_marching_parameter : float
        Positive requested time step; ToMoBAR caps it internally for stability.
    penalty_type : int
        1 Huber, 2 Perona-Malik, 3 Tukey, 4 truncated quadratic,
        5 truncated Huber. Types 2--5 are nonconvex.
    nonnegativity : bool
        Project image values to >= 0, including at zero strength.
    gpu_id : int
        CUDA device index; input must reside on this device.
    half_precision : bool
        Must be False; NDF supports float32 only.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 denoised array, same shape and device as input.
    """
    _check_tomobar_denoiser(
        data,
        "nonlinear_diffusion",
        regularisation_parameter,
        iterations,
        nonnegativity,
        gpu_id,
        half_precision,
        (
            (sigma, [float, int], "sigma"),
            (time_marching_parameter, [float, int], "time_marching_parameter"),
            (penalty_type, [int], "penalty_type"),
        ),
    )
    return NDF_cupy(
        data,
        regularisation_parameter=regularisation_parameter,
        iterations=iterations,
        sigma=sigma,
        time_marching_parameter=time_marching_parameter,
        penalty_type=penalty_type,
        nonneg=int(nonnegativity),
        gpu_id=gpu_id,
        half_precision=half_precision,
    )
