#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2026 Diamond Light Source Ltd.
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
# Created Date: 22 April 2026
# ---------------------------------------------------------------------------

from typing import Tuple
import cupy as cp


def argsort_with_reverse(
    data: cp.ndarray, axis: int = -1
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute sorting indices for an 1D or 2D array, and efficiently compute the indices to revert the sort.
    """
    dim = len(data.shape)
    if 1 <= dim <= 2:
        pass
    else:
        raise ValueError("only 1D and 2D arrays are supported")
    if axis < 0:
        axis = dim + axis
    if axis >= dim:
        raise ValueError("invalid axis")
    sort_indices = cp.argsort(data, axis=axis)
    reverse_sort_indices = cp.empty_like(sort_indices)
    if dim == 1:
        reverse_sort_indices[sort_indices] = cp.arange(0, data.size)
    elif axis == 0:  # sort rows
        nrows, ncols = data.shape
        cols = cp.arange(ncols)[None, :]
        reverse_sort_indices[sort_indices, cols] = cp.arange(nrows)[:, None]
    else:  # sort columns
        nrows, ncols = data.shape
        rows = cp.arange(nrows)[:, None]
        reverse_sort_indices[rows, sort_indices] = cp.arange(ncols)

    return sort_indices, reverse_sort_indices
