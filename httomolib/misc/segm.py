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
# Created By  : <scientificsoftware@diamond.ac.uk>
# Created Date: 25/October/2022
# ---------------------------------------------------------------------------
"""Modules for data segmentation and thresholding"""

import numpy as np
from numpy import ndarray
from skimage import filters
from httomolib.decorator import method_all

__all__ = [
    "binary_thresholding",
]

@method_all(cpuonly=True)
def binary_thresholding(
    data: ndarray,
    val_intensity: float = 0.1,
    otsu: bool = False,
    foreground: bool = True,
    axis: int = 1,
) -> ndarray:
    """
    Performs binary thresholding to the input data

    Parameters
    ----------
    data : ndarray
        Input array.
    val_intensity: float, optional
        The grayscale intensity value that defines the binary threshold.
        Defaults to 0.1
    otsu: bool, optional
        If set to True, val_intensity will be overwritten by Otsu method.
    foreground : bool, optional
        Get the foreground, otherwise background.
    axis : int, optional
        Specify the axis to use to slice the data (if data is the 3D array).

    Returns
    -------
    ndarray
        A binary mask of the input data.
    """

    # initialising output mask
    mask = np.zeros(np.shape(data), dtype=np.uint8)

    data_full_shape = np.shape(data)
    if data.ndim == 3:
        slice_dim_size = data_full_shape[axis]
        for _ in range(slice_dim_size):
            _get_mask(data, mask, val_intensity, otsu, foreground)
    else:
        _get_mask(data, mask, val_intensity, otsu, foreground)

    return mask


def _get_mask(data, mask, val_intensity, otsu, foreground):
    """Helper function to get the data binary segmented into a mask"""
    if otsu:
        # get the intensity value based on Otsu
        val_intensity = filters.threshold_otsu(data)

    mask[data > val_intensity if foreground else data <= val_intensity] = 1
    return mask
