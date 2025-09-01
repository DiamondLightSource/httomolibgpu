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
"""This is a collection of supplementary functions (utils) to perform various data checks"""

from httomolibgpu import cupywrapper
from typing import Optional

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

import numpy as np

from unittest.mock import Mock

if cupy_run:
    from httomolibgpu.cuda_kernels import load_cuda_module
else:
    load_cuda_module = Mock()


def _naninfs_check(
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
        Method's name for which input data is tested.

    Returns
    -------
    ndarray
        Uncorrected or corrected (nans and infs converted to zeros) input array.
    """
    present_nans_infs_b = False

    if cupy_run:
        xp = cp.get_array_module(data)
    else:
        import numpy as xp

    if xp.__name__ == "cupy":
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
    else:
        if not np.all(np.isfinite(data)):
            present_nans_infs_b = True
            np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if present_nans_infs_b:
        if verbosity:
            print(
                f"Warning!!! Input data to method: {method_name} contains Inf's or/and NaN's. This will be corrected but it is recommended to check the validity of input to the method."
            )

    return data


def _zeros_check(
    data: cp.ndarray,
    verbosity: bool = True,
    percentage_threshold: float = 50,
    method_name: Optional[str] = None,
) -> bool:
    """
    This function finds all zeros present in the data. If the amount of zeros is larger than percentage_threshold it prints the warning.

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy or Numpy array.
    verbosity : bool
        If enabled, then the printing of the warning happens when data contains infs or nans.
    percentage_threshold: float:
        If the number of zeros in input data is more than the percentage of all data points, then print the data warning
    method_name : str, optional.
        Method's name for which input data is tested.

    Returns
    -------
    bool
        True if the data contains too many zeros
    """
    if cupy_run:
        xp = cp.get_array_module(data)
    else:
        import numpy as xp

    nonzero_elements_total = 1
    for tot_elements_mult in data.shape:
        nonzero_elements_total *= tot_elements_mult

    warning_zeros = False
    zero_elements_total = nonzero_elements_total - int(xp.count_nonzero(data))

    if (zero_elements_total / nonzero_elements_total) * 100 >= percentage_threshold:
        warning_zeros = True
        if verbosity:
            print(
                f"Warning!!! Input data to method: {method_name} contains more than {percentage_threshold} percent of zeros."
            )

    return warning_zeros


def data_checker(
    data: cp.ndarray,
    verbosity: bool = True,
    method_name: Optional[str] = None,
) -> bool:
    """
    Function that performs the variety of checks on input data, in some cases also correct the data and prints warnings.
    Currently it checks for: the presence of infs and nans in data.

    Parameters
    ----------
    data : xp.ndarray
        Input CuPy or Numpy array either float32 or uint16 data type.
    verbosity : bool
        If enabled, then the printing of the warning happens when data contains infs or nans.
    method_name : str, optional.
        Method's name for which input data is tested.

    Returns
    -------
    cp.ndarray
        Returns corrected or not data array.
    """

    data = _naninfs_check(data, verbosity=verbosity, method_name=method_name)

    # ! The number of zero elements check is currently switched off as it requires sorting or AtomicAdd, which makes it inefficient on GPUs. !
    # _zeros_check(
    #     data,
    #     verbosity=verbosity,
    #     percentage_threshold=50,
    #     method_name=method_name,
    # )

    return data
