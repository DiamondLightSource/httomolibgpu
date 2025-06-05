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
"""Module for data denoising. For more detailed information see :ref:`data_denoising_module`."""

import numpy as np
from typing import Union, Optional

from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

from httomolibgpu.misc.supp_func import data_checker

if cupy_run:
    from ccpi.filters.regularisersCuPy import ROF_TV, PD_TV
else:
    ROF_TV = Mock()
    PD_TV = Mock()


__all__ = [
    "total_variation_ROF",
    "total_variation_PD",
]


def total_variation_ROF(
    data: cp.ndarray,
    regularisation_parameter: Optional[float] = 1e-05,
    iterations: Optional[int] = 3000,
    time_marching_parameter: Optional[float] = 0.001,
    gpu_id: Optional[int] = 0,
) -> cp.ndarray:
    """
    Total Variation using Rudin-Osher-Fatemi (ROF) :cite:`rudin1992nonlinear` explicit iteration scheme to perform edge-preserving image denoising.
    This is a gradient-based algorithm for a smoothed TV term which requires a small time marching parameter and a significant number of iterations.
    See more in :ref:`method_total_variation_ROF`.


    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array of float32 data type.
    regularisation_parameter : float, optional
        Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
    iterations : int, optional
        The number of iterations. Defaults to 3000.
    time_marching_parameter : float, optional
        Time marching parameter, needs to be small to ensure convergence. Defaults to 0.001.
    gpu_id : int, optional
        GPU device index to perform processing on. Defaults to 0.

    Returns
    -------
    ndarray
        TV-ROF filtered 3D CuPy array in float32 data type.

    Raises
    ------
    ValueError
        If the input array is not float32 data type.
    """

    data = data_checker(data, verbosity=True, method_name="total_variation_ROF")

    return ROF_TV(
        data, regularisation_parameter, iterations, time_marching_parameter, gpu_id
    )


def total_variation_PD(
    data: cp.ndarray,
    regularisation_parameter: Optional[float] = 1e-05,
    iterations: Optional[int] = 1000,
    isotropic: Optional[bool] = True,
    nonnegativity: Optional[bool] = False,
    lipschitz_const: Optional[float] = 8.0,
    gpu_id: Optional[int] = 0,
) -> cp.ndarray:
    """
    Primal Dual algorithm for non-smooth convex Total Variation functional :cite:`chan1999nonlinear`. See more in :ref:`method_total_variation_PD`.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array of float32 data type.
    regularisation_parameter : float
        Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
    iterations : int
        The number of iterations. Defaults to 1000.
    isotropic : bool
        Choose between isotropic or anisotropic TV norm. Defaults to isotropic.
    nonnegativity  : bool
        Enable non-negativity in iterations. Defaults to False.
    lipschitz_const : float
        Lipschitz constant to control convergence. Defaults to 8.
    gpu_id : int
        GPU device index to perform processing on. Defaults to 0.

    Returns
    -------
    ndarray
        TV-PD filtered 3D CuPy array in float32 data type.

    Raises
    ------
    ValueError
        If the input array is not float32 data type.
    """

    data_checker(data, verbosity=True, method_name="total_variation_PD")

    methodTV = 0
    if not isotropic:
        methodTV = 1

    nonneg = 0
    if nonnegativity:
        nonneg = 1

    return PD_TV(
        data,
        regularisation_parameter,
        iterations,
        methodTV,
        nonneg,
        lipschitz_const,
        gpu_id,
    )
