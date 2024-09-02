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
# Created Date: 23 March 2023
# ---------------------------------------------------------------------------
"""Module for data type morphing functions"""

import numpy as np
from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from cupyx.scipy.interpolate import interpn
else:
    interpn = Mock()

from typing import Literal

__all__ = [
    "sino_360_to_180",
    "data_resampler",
]


def sino_360_to_180(
    data: cp.ndarray, overlap: int = 0, rotation: Literal["left", "right"] = "left"
) -> cp.ndarray:
    """
    Converts 0-360 degrees sinogram to a 0-180 sinogram.
    If the number of projections in the input data is odd, the last projection
    will be discarded. See :cite:`vo2021data`.

    Parameters
    ----------
    data : cp.ndarray
        Input 3D data.
    overlap : scalar, optional
        Overlapping number of pixels.
    rotation : string, optional
        'left' if rotation center is close to the left of the
        field-of-view, 'right' otherwise.
    Returns
    -------
    cp.ndarray
        Output 3D data.
    """
    if data.ndim != 3:
        raise ValueError("only 3D data is supported")

    dx, dy, dz = data.shape

    overlap = int(np.round(overlap))
    if overlap >= dz:
        raise ValueError("overlap must be less than data.shape[2]")
    if overlap < 0:
        raise ValueError("only positive overlaps are allowed.")

    if rotation not in ["left", "right"]:
        raise ValueError('rotation parameter must be either "left" or "right"')

    n = dx // 2

    out = cp.empty((n, dy, 2 * dz - overlap), dtype=data.dtype)

    if rotation == "left":
        weights = cp.linspace(0, 1.0, overlap, dtype=cp.float32)
        out[:, :, -dz + overlap :] = data[:n, :, overlap:]
        out[:, :, : dz - overlap] = data[n : 2 * n, :, overlap:][:, :, ::-1]
        out[:, :, dz - overlap : dz] = (
            weights * data[:n, :, :overlap]
            + (weights * data[n : 2 * n, :, :overlap])[:, :, ::-1]
        )
    if rotation == "right":
        weights = cp.linspace(1.0, 0, overlap, dtype=cp.float32)
        out[:, :, : dz - overlap] = data[:n, :, :-overlap]
        out[:, :, -dz + overlap :] = data[n : 2 * n, :, :-overlap][:, :, ::-1]
        out[:, :, dz - overlap : dz] = (
            weights * data[:n, :, -overlap:]
            + (weights * data[n : 2 * n, :, -overlap:])[:, :, ::-1]
        )

    return out


def data_resampler(
    data: cp.ndarray, newshape: list, axis: int = 1, interpolation: str = "linear"
) -> cp.ndarray:
    """Down/Up-resampler of the input data implemented through interpn function.
       Please note that the method will leave the specified axis
       dimension unchanged, e.g. (128,128,128) -> (128,256,256) for axis = 0 and
       newshape = [256,256].

    Args:
        data (cp.ndarray): 3d cupy array.
        newshape (list): 2d list that defines the 2D slice shape of new shape data.
        axis (int, optional): Axis along which the scaling is applied. Defaults to 1.
        interpolation (str, optional): Selection of interpolation method. Defaults to 'linear'.

    Raises:
        ValueError: When data is not 3D

    Returns:
        cp.ndarray: Up/Down-scaled 3D cupy array
    """
    if data.ndim != 3:
        raise ValueError("only 3D data is supported")

    N, M, Z = cp.shape(data)

    if axis == 0:
        xaxis = cp.arange(M) - M / 2
        yaxis = cp.arange(Z) - Z / 2
        step_x = M / newshape[0]
        step_y = Z / newshape[1]
        scaled_data = cp.empty((N, newshape[0], newshape[1]), dtype=cp.float32)
    elif axis == 1:
        xaxis = cp.arange(N) - N / 2
        yaxis = cp.arange(Z) - Z / 2
        step_x = N / newshape[0]
        step_y = Z / newshape[1]
        scaled_data = cp.empty((newshape[0], M, newshape[1]), dtype=cp.float32)
    elif axis == 2:
        xaxis = cp.arange(N) - N / 2
        yaxis = cp.arange(M) - M / 2
        step_x = N / newshape[0]
        step_y = M / newshape[1]
        scaled_data = cp.empty((newshape[0], newshape[1], Z), dtype=cp.float32)
    else:
        raise ValueError("Only 0,1,2 values for axes are supported")

    points = (xaxis, yaxis)

    scale_x = 2 / step_x
    scale_y = 2 / step_y

    y1 = np.linspace(
        -newshape[0] / scale_x,
        newshape[0] / scale_x - step_x,
        num=newshape[0],
        endpoint=False,
    ).astype(np.float32)
    x1 = np.linspace(
        -newshape[1] / scale_y,
        newshape[1] / scale_y - step_y,
        num=newshape[1],
        endpoint=False,
    ).astype(np.float32)

    xi_mesh = np.meshgrid(x1, y1)
    xi = np.empty((2, newshape[0], newshape[1]), dtype=np.float32)
    xi[0, :, :] = xi_mesh[1]
    xi[1, :, :] = xi_mesh[0]
    xi_size = xi.size
    xi = np.rollaxis(xi, 0, 3)
    xi = np.reshape(xi, [xi_size // 2, 2])
    xi = cp.asarray(xi, dtype=cp.float32, order="C")

    if axis == 0:
        for j in range(N):
            res = interpn(
                points,
                data[j, :, :],
                xi,
                method=interpolation,
                bounds_error=False,
                fill_value=0.0,
            )
            scaled_data[j, :, :] = cp.reshape(
                res, [newshape[0], newshape[1]], order="C"
            )
    elif axis == 1:
        for j in range(M):
            res = interpn(
                points,
                data[:, j, :],
                xi,
                method=interpolation,
                bounds_error=False,
                fill_value=0.0,
            )
            scaled_data[:, j, :] = cp.reshape(
                res, [newshape[0], newshape[1]], order="C"
            )
    else:
        for j in range(Z):
            res = interpn(
                points,
                data[:, :, j],
                xi,
                method=interpolation,
                bounds_error=False,
                fill_value=0.0,
            )
            scaled_data[:, :, j] = cp.reshape(
                res, [newshape[0], newshape[1]], order="C"
            )

    return scaled_data
