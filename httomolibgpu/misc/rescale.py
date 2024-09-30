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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ecpress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Tomography Team at DLS <scientificsoftware@diamond.ac.uk>
# Created Date: 1 March 2024
# ---------------------------------------------------------------------------
""" Module for data rescaling. For more detailed information see :ref:`data_rescale_module`.

"""

import numpy as np
from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from typing import Literal, Optional, Tuple, Union

__all__ = [
    "rescale_to_int",
]


def rescale_to_int(
    data: Union[np.ndarray, cp.ndarray],
    perc_range_min: float = 0.0,
    perc_range_max: float = 100.0,
    bits: Literal[8, 16, 32] = 8,
    glob_stats: Optional[Tuple[float, float, float, int]] = None,
) -> Union[np.ndarray, cp.ndarray]:
    """
    Rescales the data given as float32 type and converts it into the range of an unsigned integer type
    with the given number of bits. For more detailed information and examples, see :ref:`method_rescale_to_int`.

    Parameters
    ----------
    data : Union[np.ndarray, cp.ndarray]
        Input data as a numpy or cupy array (the function is cpu-gpu agnostic)
    perc_range_min: float, optional
        The lower cutoff point in the input data, in percent of the data range (defaults to 0).
        The lower bound is computed as min + perc_range_min/100*(max-min)
    perc_range_max: float, optional
        The upper cutoff point in the input data, in percent of the data range (defaults to 100).
        The upper bound is computed as min + perc_range_max/100*(max-min)
    bits: Literal[8, 16, 32], optional
        The number of bits in the output integer range (defaults to 8).
        Allowed values are:
        - 8 -> uint8
        - 16 -> uint16
        - 32 -> uint32
    glob_stats: tuple, optional
        Global statistics of the full dataset (beyond the data passed into this call).
        It's a tuple with (min, max, sum, num_items). If not given, the min/max is
        computed from the given data.

    Returns
    -------
    Union[np.ndarray, cp.ndarray]
        The original data, clipped to the range specified with the perc_range_min and
        perc_range_max, and scaled to the full range of the output integer type
    """
    if bits == 8:
        output_dtype: Union[type[np.uint8], type[np.uint16], type[np.uint32]] = np.uint8
    elif bits == 16:
        output_dtype = np.uint16
    else:
        output_dtype = np.uint32

    if cupy_run:
        xp = cp.get_array_module(data)
    else:
        import numpy as xp

    # get the min and max integer values of the output type
    output_min = xp.iinfo(output_dtype).min
    output_max = xp.iinfo(output_dtype).max

    if not isinstance(glob_stats, tuple):
        min_value = float(xp.min(data))
        max_value = float(xp.max(data))
    else:
        min_value = glob_stats[0]
        max_value = glob_stats[1]

    range_intensity = max_value - min_value
    input_min = (perc_range_min * (range_intensity) / 100) + min_value
    input_max = (perc_range_max * (range_intensity) / 100) + min_value

    if (input_max - input_min) != 0.0:
        factor = xp.float32((output_max - output_min) / (input_max - input_min))
    else:
        factor = 1.0

    res = xp.empty(data.shape, dtype=output_dtype)
    if xp.__name__ == "numpy":
        if input_max == pow(2, 32):
            input_max -= 1
        data[np.logical_not(np.isfinite(data))] = 0
        res = np.copy(data.astype(float))
        res[data.astype(float) < input_min] = int(input_min)
        res[data.astype(float) > input_max] = int(input_max)
        res -= input_min
        res *= factor
        res = output_dtype(res)
    else:
        rescale_kernel = cp.ElementwiseKernel(
            "T x, raw T input_min, raw T input_max, raw T factor",
            "O out",
            """
            T x_clean = isnan(x) || isinf(x) ? T(0) : x;
            T x_clipped = x_clean < input_min ? input_min : (x_clean > input_max ? input_max : x_clean);
            T x_rebased = x_clipped - input_min;
            out = O(x_rebased * factor);
            """,
            "rescale_to_int",
        )
        rescale_kernel(data, input_min, input_max, factor, res)
    return res
