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
""" Module for data denoising. For more detailed information see :ref:`data_denoising_module`.
"""

import numpy as np
from typing import Union, Optional

from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from numpy import float32
from unittest.mock import Mock


from ccpi.filters.regularisersCuPy import ROF_TV, PD_TV

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
    Total Variation using Rudin-Osher-Fatemi (ROF) explicit iteration scheme to perform edge-preserving image denoising.
    This is a gradient-based algorithm for a smoothed TV term which requires a small time marching parameter and a significant number of iterations.
        

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

    return ROF_TV(data, regularisation_parameter, iterations, time_marching_parameter, gpu_id)


def total_variation_PD(
    data: cp.ndarray,
    regularisation_parameter: Optional[float] = 1e-05,
    iterations: Optional[int] = 1000,
    methodTV: Optional[int] = 0,
    nonneg: Optional[int] = 0,
    lipschitz_const: Optional[float] = 8.0,
    gpu_id: Optional[int] = 0,
) -> cp.ndarray:
    """
    Primal Dual algorithm for non-smooth convex Total Variation functional.         

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy 3D array of float32 data type.
    regularisation_parameter : float, optional
        Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
    iterations : int, optional
        The number of iterations. Defaults to 1000.
    methodTV : int, optional
        Choose between isotropic (0) or anisotropic (1) case for the TV norm. Defaults to isotropic (0).
    nonneg  : int, optional
        Enable non-negativity in updates by selecting 1. Defaults to 0.
    lipschitz_const : float, optional
        Lipschitz constant to control convergence. Defaults to 8.
    gpu_id : int, optional
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

    return PD_TV(data, regularisation_parameter, iterations, methodTV, nonneg, lipschitz_const, gpu_id)