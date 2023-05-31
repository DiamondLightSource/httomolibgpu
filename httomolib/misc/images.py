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
# Created Date: 27/October/2022
# ---------------------------------------------------------------------------
""" Module for loading/saving images """

import os
from typing import Optional

import numpy as np
from numpy import ndarray
from PIL import Image
from skimage import exposure
from httomolib.decorator import method_all

__all__ = [
    "save_to_images",
]


@method_all(cpuonly=True)
def save_to_images(
    data: ndarray,
    out_dir: str,
    subfolder_name: str = "images",
    axis: int = 0,
    file_format: str = "tif",
    bits: int = 8,
    perc_range_min: float = 0.0,
    perc_range_max: float = 100.0,
    jpeg_quality: int = 95,
    glob_stats: Optional[tuple] = None,
    comm_rank: int = 0,
):
    """
    Saves data as 2D images.

    Parameters
    ----------
    data : np.ndarray
        Required input NumPy ndarray.
    out_dir : str
        The main output directory for images.
    subfolder_name : str, optional
        Subfolder name within the main output directory.
        Defaults to 'images'.
    axis : int, optional
        Specify the axis to use to slice the data (if `data` is a 3D array).
    file_format : str, optional
        Specify the file format to use, e.g. "png", "jpeg", or "tif".
        Defaults to "tif".
    bits : int, optional
        Specify the number of bits to use (8, 16, or 32-bit).
    perc_range_min: float, optional
        Using `np.percentile` to scale data in percentage range.
        Defaults to 0.0
    perc_range_max: float, optional
        Using `np.percentile` to scale data in percentage range.
        Defaults to 100.0
    jpeg_quality : int, optional
        Specify the quality of the jpeg image.
    glob_stats: tuple, optional
        Global statistics of the input data in a tuple format: (min, max, mean, std_var).
        If None, then it will be calculated.
    comm_rank: int, optional
        comm.rank integer extracted from the MPI communicator for parallel run.
    """

    if bits not in [8, 16, 32]:
        bits = 32
        print(
            "The selected bit type %s is not available, "
            "resetting to 32 bit \n" % str(bits)
        )

    # create the output folder
    subsubfolder_name = f"images{str(bits)}bit_{str(file_format)}"
    path_to_images_dir = os.path.join(out_dir, subfolder_name, subsubfolder_name)
    try:
        os.makedirs(path_to_images_dir)
    except OSError:
        if not os.path.isdir(path_to_images_dir):
            raise

    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)
    
    if glob_stats is None:
        min_percentile = np.nanpercentile(data, perc_range_min)
        max_percentile = np.nanpercentile(data, perc_range_max)
    else:
        # calculate the range here based on global max and min
        range_intensity = glob_stats[1] - glob_stats[0]
        min_percentile = (perc_range_min * (range_intensity) / 100) + glob_stats[0]
        max_percentile = (perc_range_max * (range_intensity) / 100) + glob_stats[0]
    
    if data.ndim == 3:
        slice_dim_size = np.shape(data)[axis]
        for idx in range(slice_dim_size):
            filename = f"{idx + comm_rank*slice_dim_size:05d}.{file_format}"
            filepath = os.path.join(path_to_images_dir, f"{filename}")
            _save_single_img(
                data.take(indices=idx, axis=axis),
                min_percentile,
                max_percentile,
                bits,
                jpeg_quality,
                filepath,
            )
    else:
        filename = f"{1:05d}.{file_format}"
        filepath = os.path.join(path_to_images_dir, f"{filename}")
        _save_single_img(data, min_percentile, max_percentile, bits, jpeg_quality, filepath)

def _save_single_img(array2d,
                     min_percentile,
                     max_percentile,
                     bits, 
                     jpeg_quality, 
                     path_to_out_file):
    """Rescales to the bit chosen and saves the image."""
    if bits == 8:
        array2d = exposure.rescale_intensity(
            array2d, in_range=(min_percentile, max_percentile), out_range=(0, 255)
        ).astype(np.uint8)

    elif bits == 16:
        array2d = exposure.rescale_intensity(
            array2d, in_range=(min_percentile, max_percentile), out_range=(0, 65535)
        ).astype(np.uint16)

    else:
        array2d = exposure.rescale_intensity(
            array2d, in_range=(min_percentile, max_percentile), out_range=(min_percentile, max_percentile)
        ).astype(np.uint32)

    img = Image.fromarray(array2d)
    img.save(path_to_out_file, quality=jpeg_quality)
