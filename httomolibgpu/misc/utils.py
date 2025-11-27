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
# Created Date: 02/June/2025
# ---------------------------------------------------------------------------
"""Various utilities for data inspection and correction"""

from httomolibgpu import cupywrapper
from typing import Optional

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from httomolibgpu.cuda_kernels import load_cuda_module
else:
    load_cuda_module = Mock()


__all__ = [
    "data_checker",
]


def data_checker(
    data: cp.ndarray,
    infsnans_correct: bool = True,
    zeros_warning: bool = False,
    data_to_method_name: Optional[str] = None,
    verbosity: bool = True,
) -> cp.ndarray:
    """Function that performs checks on input data to ensure its validity, performs corrections and prints the warnings.
    Currently it checks for the presence of Infs and NaNs in the data and corrects them.

    Parameters
    ----------
    data : cp.ndarray
        CuPy array either float32 or uint16 data type.
    infsnans_correct: bool
        Perform correction of NaNs and Infs if they are present in the data.
    zeros_warning: bool
        Count the number of zeros in the data and produce a warning if more half of the data are zeros.
    verbosity : bool
        Print the warnings.
    data_to_method_name : str, optional.
        Method's name the output of which is tested. This is tailored for printing purposes when the method runs in HTTomo.

    Returns
    -------
    cp.ndarray
        Returns corrected CuPy array.
    """
    if data.dtype not in ["uint16", "float32"]:
        raise ValueError(
            "The input data of `uint16` and `float32` data types is accepted only."
        )

    if infsnans_correct:
        data = __naninfs_check(
            data, verbosity=verbosity, method_name=data_to_method_name
        )
    # TODO
    # if zeros_warning:
    # __zeros_check(data, verbosity=verbosity, percentage_threshold = 50, method_name=data_to_method_name)

    return data


def __naninfs_check(
    data: cp.ndarray,
    verbosity: bool = True,
    method_name: Optional[str] = None,
) -> cp.ndarray:
    """
    This function finds NaN's, +-Inf's in the input data and then prints the warnings and correct the data if correction is enabled.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy or Numpy array either float32 or uint16 data type.
    verbosity : bool
        If enabled, then the printing of the warning happens when data contains infs or nans
    method_name : str, optional.
        Method's name for which the output data is tested.

    Returns
    -------
    ndarray
        Uncorrected or corrected (nans and infs converted to zeros) input array.
    """
    present_nans_infs_b = False

    input_type = data.dtype
    if len(data.shape) == 2:
        dy, dx = data.shape
        dz = 1
    else:
        dz, dy, dx = data.shape

    present_nans_infs = cp.zeros(shape=(1)).astype(cp.uint8)

    block_x = 128
    # setting grid/block parameters
    block_dims = (block_x, 1, 1)
    grid_x = (dx + block_x - 1) // block_x
    grid_y = dy
    grid_z = dz
    grid_dims = (grid_x, grid_y, grid_z)
    params = (data, dz, dy, dx, present_nans_infs)

    kernel_args = "remove_nan_inf<{0}>".format(
        "float" if input_type == "float32" else "unsigned short"
    )

    module = load_cuda_module("remove_nan_inf", name_expressions=[kernel_args])
    remove_nan_inf_kernel = module.get_function(kernel_args)
    remove_nan_inf_kernel(grid_dims, block_dims, params)

    if present_nans_infs[0].get() == 1:
        present_nans_infs_b = True

    if present_nans_infs_b:
        if verbosity:
            print(
                "Warning! Output data of the \033[31m{}\033[0m method contains Inf's or/and NaN's. Corrected to zeros.".format(
                    method_name
                )
            )
    return data
