#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2023 Diamond Light Source Ltd.
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
# Created Date: 5 August 2026
# ---------------------------------------------------------------------------
"""Module for data type morphing functions"""

import numpy as np
from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from typing import Optional

from httomolibgpu.misc.utils import (
    __check_variable_type,
    __check_if_data_3D_array,
    __check_if_data_correct_type,
    __check_if_positive_nonzero,
)

__all__ = [
    "seam_blend_stitched_data",
]


def seam_blend_stitched_data(
    data: cp.ndarray,
    overlap: int,
    seam_index: int = 2560,
    shift_seam_index: Optional[int] = None,
) -> cp.ndarray:
    """
    Function blends the seam present in the stitched projection data using the overlap value provided.
    Used in HTTomo for seamless stitching of datasets coming from two different (PCO) cameras.

    Parameters
    ----------
    data : cp.ndarray
        3d CuPy array of the stitched data, assuming the following axis ["angles", "detY", "detX"].
    overlap : int
        Overlap between the LEFT and the RIGHT images of the stitched data. Usually known from the experiment.
    seam_index : int
        The horizontal index of the seam in the stitched data. Normally equals to the width of one frame/projection before the stitching.
    shift_seam_index : optional, int
        performs a shift of the 'seam index' that is introduced by the data cropping. This is an HTTomo related feature and should be ignored by users.
    Raises
    ----------
        ValueError: When data is not 3D.

    Returns
    ----------
        cp.ndarray: stitched data CuPy array without the seam.
    """
    ### Data and parameters checks ###
    methods_name = "seam_blend_stitched_data"
    __check_if_data_3D_array(data, methods_name)
    __check_if_data_correct_type(
        data,
        accepted_type=["float64", "float32", "uint8", "uint16", "uint32"],
        methods_name=methods_name,
    )
    __check_if_positive_nonzero(
        seam_index,
        "seam_index",
        True,
        True,
        methods_name,
    )
    __check_variable_type(seam_index, [int], "seam_index", [], methods_name)
    __check_variable_type(
        shift_seam_index, [int, type(None)], "shift_seam_index", [], methods_name
    )
    ###################################

    if shift_seam_index is None:
        shift_seam_index = 0

    seam_index -= shift_seam_index

    angles_dim, detY, detX = data.shape
    if seam_index < 0 or seam_index >= detX:
        err_str = f"Seam index '{seam_index}' cannot be negative or larger than the horizontal dimension of the data '{detX}'. Check 'shift_seam_index'."
        raise ValueError(err_str)

    ramp = np.linspace(0, 1, overlap)

    blended_data = cp.empty((angles_dim, detY, detX - overlap), dtype=data.dtype)

    blended_data[:, :, 0 : seam_index - overlap] = data[
        :, :, 0 : seam_index - overlap
    ]  # left side
    blended_data[:, :, seam_index::] = data[:, :, seam_index + overlap :]  # right side
    blended_data[:, :, seam_index - overlap : seam_index] = (
        data[:, :, seam_index - overlap : seam_index] * ramp[::-1]
        + data[:, :, seam_index : seam_index + overlap] * ramp
    )  # overlap area

    return blended_data
