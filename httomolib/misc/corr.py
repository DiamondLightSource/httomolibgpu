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
# Created Date: 21/October/2022
# ---------------------------------------------------------------------------
""" Modules for data correction """

from typing import Tuple
import numpy as np

from httomolib.decorator import method_all

__all__ = [
    "inpainting_filter3d",
]

@method_all(cpuonly=True)
def inpainting_filter3d(
    data: np.ndarray,
    mask: np.ndarray,
    iter: int = 3,
    windowsize_half: int = 5,
    method_type: str = "random",
    ncore: int = 1,
) -> np.ndarray:
    """
    Inpainting filter for 3D data, taken from the Larix toolbox
    (C - implementation).

    A morphological inpainting scheme which progresses from the
    edge of the mask inwards. It acts like a diffusion-type process
    but significantly faster in convergence.

    Parameters
    ----------
    data : ndarray
        Input array.
    mask : ndarray
        Input binary mask (uint8) the same size as data,
        integer 1 will define the inpainting area.
    iter : int, optional
        An additional number of iterations to run after the region
        has been inpainted (smoothing effect).
    windowsize_half : int, optional
        Half-window size of the searching window (neighbourhood window).
    method_type : str, optional
        Method type to select for a value in the neighbourhood: mean, median,
        or random. Defaults to "random".
    ncore : int, optional
        The number of CPU cores to use.

    Returns
    -------
    ndarray
        Inpainted array.
    """

    from larix.methods.misc import INPAINT_EUCL_WEIGHTED

    return INPAINT_EUCL_WEIGHTED(data, mask, iter, windowsize_half, method_type, ncore)
