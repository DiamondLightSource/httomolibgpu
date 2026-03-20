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
"""Modules for raw projection data normalization"""

from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from cupy import mean
else:
    mean = Mock()

from typing import Union
from numpy import float32
from httomolibgpu.misc.utils import (
    __check_variable_type,
    __check_if_data_3D_array,
    __check_if_data_correct_type,
)

__all__ = ["dark_flat_field_correction", "minus_log"]


def dark_flat_field_correction(
    data: cp.ndarray,
    flats: cp.ndarray,
    darks: cp.ndarray,
    flats_multiplier: Union[float, int] = 1.0,
    darks_multiplier: Union[float, int] = 1.0,
    cutoff: Union[float, int] = 10.0,
) -> cp.ndarray:
    """
    Perform dark/flat field correction of raw projection data.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    flats : cp.ndarray
        3D flat field data as a CuPy array.
    darks : cp.ndarray
        3D dark field data as a CuPy array.
    flats_multiplier: float,int
        A multiplier to apply to flats, can work as an intensity compensation constant.
    darks_multiplier: float,int
        A multiplier to apply to darks, can work as an intensity compensation constant.
    cutoff : float,int
        Permitted maximum value for the normalised data.

            Returns
    -------
    cp.ndarray
        dark/flat field corrected 3D tomographic data as a CuPy array.
    """
    ### Data and parameters checks ###
    methods_name = "dark_flat_field_correction"
    __check_if_data_3D_array(data, methods_name)
    __check_if_data_correct_type(
        data, accepted_type=["float32", "uint16"], methods_name=methods_name
    )
    __check_if_data_correct_type(
        flats, accepted_type=["float32", "uint16"], methods_name=methods_name
    )
    __check_if_data_correct_type(
        darks, accepted_type=["float32", "uint16"], methods_name=methods_name
    )
    __check_variable_type(
        flats_multiplier, [int, float], "flats_multiplier", [], methods_name
    )
    __check_variable_type(
        darks_multiplier, [int, float], "darks_multiplier", [], methods_name
    )
    __check_variable_type(cutoff, [int, float], "cutoff", [], methods_name)
    ###################################

    _check_valid_input_flats_darks(flats, darks)

    dark0 = cp.empty(darks.shape[1:], dtype=float32)
    flat0 = cp.empty(flats.shape[1:], dtype=float32)
    out = cp.empty(data.shape, dtype=float32)
    mean(darks, axis=0, dtype=float32, out=dark0)
    mean(flats, axis=0, dtype=float32, out=flat0)

    dark0 *= darks_multiplier
    flat0 *= flats_multiplier

    kernel_name = "normalisation"
    kernel = r"""
        float denom = float(flats) - float(darks);
        if (denom < eps) {
            denom = eps;
        }
        float v = (float(data) - float(darks))/denom;
        """
    kernel += "if (v > cutoff) v = cutoff;\n"
    kernel += "if (v < -cutoff) v = cutoff;\n"
    kernel += "out = v;\n"

    normalisation_kernel = cp.ElementwiseKernel(
        "T data, U flats, U darks, raw float32 cutoff",
        "float32 out",
        kernel,
        kernel_name,
        options=("-std=c++17",),
        loop_prep="constexpr float eps = 1.0e-07;",
        no_return=True,
    )

    normalisation_kernel(data, flat0, dark0, float32(cutoff), out)

    return out


def minus_log(data: cp.ndarray) -> cp.ndarray:
    """
    Apply -log(data) operation

    Parameters
    ----------
    data : cp.ndarray
        Data as a CuPy array.

    Returns
    -------
    cp.ndarray
        data after -log(data)
    """

    return -cp.log(data)


def _check_valid_input_flats_darks(flats, darks) -> None:
    """Helper function to check the validity of darks and flats"""
    if flats.ndim not in (2, 3):
        raise ValueError("Input flats must be 2D or 3D data only")

    if darks.ndim not in (2, 3):
        raise ValueError("Input darks must be 2D or 3D data only")

    if flats.ndim == 2:
        flats = flats[cp.newaxis, :, :]
    if darks.ndim == 2:
        darks = darks[cp.newaxis, :, :]
