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
# Created Date: 01 November 2022
# version ='0.1'
# ---------------------------------------------------------------------------
"""Modules for tomographic reconstruction"""

from typing import Optional

#import cupy as cp
#from cupy import ndarray, swapaxes
import numpy as np
from numpy import ndarray

__all__ = [
    'reconstruct_tomobar',
]

def reconstruct_tomobar(
    data: ndarray,
    angles: ndarray,
    center: float = None,
    objsize: int = None,
    algorithm: str = 'FBP3D',
    gpu_id : int = 0
    ) -> ndarray:
    """
    Perform reconstruction using ToMoBAR wrappers around ASTRA toolbox. 

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : cp.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    objsize : int, optional
        The size in pixels of the reconstructed object.
    algorithm : str, optional
        The name of the reconstruction method.
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The reconstructed volumes a CuPy array.
    """
    from tomobar.methodsDIR import RecToolsDIR    
    
    if center is None:
        center = 0.0
    if objsize is None:
        objsize = data.shape[2]
    # set parameters and initiate a TomoBar class object for direct reconstruction
    RectoolsDIR = RecToolsDIR(DetectorsDimH=data.shape[2],  # DetectorsDimH # detector dimension (horizontal)
                              DetectorsDimV=data.shape[1],  # DetectorsDimV # detector dimension (vertical) for 3D case only
                              CenterRotOffset=data.shape[2] * 0.5 - center,  # The center of rotation combined with the shift offsets
                              AnglesVec=-angles,  # the vector of angles in radians
                              ObjSize=objsize,  # a scalar to define the reconstructed object dimensions
                              device_projector=gpu_id)
        
    return RectoolsDIR.FBP(np.swapaxes(data, 0, 1))  # perform 3D FBP as 3D BP with Astra and then filtering on CPU