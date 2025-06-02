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


def _naninfs_check(
    data: cp.ndarray,
    correction: bool = True,
    verbosity: bool = True,
    method_name: Optional[str] = None,
) -> cp.ndarray:
    """
    Function finds NaN's, +-Inf's in the input data and then prints the warning and correct the data

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy array either float32 or uint16 data type.
    correction : bool
        If correction is enabled then Inf's and NaN's will be replaced by zeros.
    verbosity : bool
        If enabled, then the printing of the warning happens when data contains infs or nans
    method_name : str, optional.
        Method's name for which input data is tested.

    Returns
    -------
    ndarray
        Corrected (or not) CuPy array.
    """
    if cupy_run:
        xp = cp.get_array_module(data)
    else:
        import numpy as xp

    if not xp.all(xp.isfinite(data)):
        if verbosity:
            print(
                f"Warning!!! Input data to method: {method_name} contains Inf's or/and NaN's."
            )
        if correction:
            print(
                "Inf's or/and NaN's will be corrected to finite integers (zeros). It is advisable to check the correctness of the input."
            )
            xp.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def _zeros_check(
    data: cp.ndarray,
    verbosity: bool = True,
    percentage_threshold: float = 50,
    method_name: Optional[str] = None,
) -> bool:
    """
    Function finds NaN's, +-Inf's in the input data and then prints the warning and correct the data

    Parameters
    ----------
    data : cp.ndarray
        Input CuPy array either float32 or uint16 data type.
    verbosity : bool
        If enabled, then the printing of the warning happens when data contains infs or nans
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

    warning_zeros = False
    zero_elements_total = int(xp.count_nonzero(data == 0))
    nonzero_elements_total = len(data.flatten())
    if (zero_elements_total / nonzero_elements_total) * 100 >= percentage_threshold:
        warning_zeros = True
        if verbosity:
            print(
                f"Warning!!! Input data to method: {method_name} contains more than {percentage_threshold} percent of zeros."
            )

    return warning_zeros
