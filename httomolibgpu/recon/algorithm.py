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
# ---------------------------------------------------------------------------
"""Module for tomographic reconstruction"""

from typing import Optional, Tuple, Union

import cupy as cp
from cupy import float32, complex64
import cupyx
import numpy as np
import nvtx

from httomolibgpu.cuda_kernels import load_cuda_module

__all__ = [
    "FBP",
    "SIRT",
    "CGLS",
]

def _apply_circular_mask(data, recon_mask_radius):
    
    recon_size = data.shape[1]
    Y, X = cp.ogrid[:recon_size, :recon_size]
    half_size = recon_size//2
    dist_from_center = cp.sqrt((X - half_size)**2 + (Y-half_size)**2)
    if recon_mask_radius <= 1.0:
        mask = dist_from_center <= half_size - abs(half_size - half_size/recon_mask_radius)
    else:
        mask = dist_from_center <= half_size + abs(half_size - half_size/recon_mask_radius)    
    return data*mask


## %%%%%%%%%%%%%%%%%%%%%%% FBP reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
@nvtx.annotate()
def FBP(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = None,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform Filtered Backprojection (FBP) reconstruction using ASTRA toolbox and ToMoBAR wrappers.
    This is a 3D recon from a CuPy array and a custom built filter.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels. 
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: int, optional
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artefacts. The values outside the diameter will be set to zero. 
        None by default, to see the effect of the mask try setting the value in the range [0.7-1.0]. 
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The FBP reconstructed volume as a CuPy array.
    """
    from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
    RecToolsCP = RecToolsDIRCuPy(DetectorsDimH=data.shape[2],  # Horizontal detector dimension
                                 DetectorsDimV=data.shape[1],  # Vertical detector dimension (3D case)
                                 CenterRotOffset=data.shape[2] / 2 - center - 0.5,  # Center of Rotation scalar or a vector
                                 AnglesVec=-angles,  # A vector of projection angles in radians
                                 ObjSize=recon_size,  # Reconstructed object dimensions (scalar)
                                 device_projector=gpu_id,
                                 )
    reconstruction = RecToolsCP.FBP3D(data)
    cp._default_memory_pool.free_all_blocks()
    # perform masking to the result of reconstruction if needed 
    if recon_mask_radius is not None:
        reconstruction = _apply_circular_mask(reconstruction, recon_mask_radius)
    return cp.swapaxes(reconstruction,0,1)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##

## %%%%%%%%%%%%%%%%%%%%%%% SIRT reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
@nvtx.annotate()
def SIRT(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,    
    iterations: Optional[int] = 300,
    nonnegativity: Optional[bool] = True,
    recon_mask_radius: Optional[float] = None,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform Simultaneous Iterative Recostruction Technique (SIRT) using ASTRA toolbox and ToMoBAR wrappers.
    This is 3D recon directly from a CuPy array while using ASTRA GPUlink capability.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels. 
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    iterations : int, optional
        The number of SIRT iterations.
    nonnegativity : bool, optional
        Impose nonnegativity constraint on reconstructed image.
    recon_mask_radius: int, optional
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artefacts. The values outside the diameter will be set to zero. 
        None by default, to see the effect of the mask try setting the value in the range [0.7-1.0].
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The SIRT reconstructed volume as a CuPy array.
    """
    from tomobar.methodsIR_CuPy import RecToolsIRCuPy

    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
        
    RecToolsCP = RecToolsIRCuPy(DetectorsDimH=data.shape[2],  # Horizontal detector dimension
                                 DetectorsDimV=data.shape[1],  # Vertical detector dimension (3D case)
                                 CenterRotOffset=data.shape[2] / 2 - center - 0.5,  # Center of Rotation scalar or a vector
                                 AnglesVec=-angles,  # A vector of projection angles in radians
                                 ObjSize=recon_size,  # Reconstructed object dimensions (scalar)
                                 device_projector=gpu_id,
                                 )
    _data_ = {"projection_norm_data": data}  # data dictionary
    _algorithm_ = {"iterations": iterations, "nonnegativity": nonnegativity}
    reconstruction = RecToolsCP.SIRT(_data_, _algorithm_)
    cp._default_memory_pool.free_all_blocks()
    # perform masking to the result of reconstruction if needed 
    if recon_mask_radius is not None:
        reconstruction = _apply_circular_mask(reconstruction, recon_mask_radius)    
    return cp.swapaxes(reconstruction,0,1)

## %%%%%%%%%%%%%%%%%%%%%%% CGLS reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
@nvtx.annotate()
def CGLS(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,
    iterations: Optional[int] = 20,
    nonnegativity: Optional[bool] = True,
    recon_mask_radius: Optional[float] = None,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform Congugate Gradient Least Squares (CGLS) using ASTRA toolbox and ToMoBAR wrappers.
    This is 3D recon directly from a CuPy array while using ASTRA GPUlink capability.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels. 
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    iterations : int, optional
        The number of CGLS iterations.
    nonnegativity : bool, optional
        Impose nonnegativity constraint on reconstructed image.
    recon_mask_radius: int, optional
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artefacts. The values outside the diameter will be set to zero. 
        None by default, to see the effect of the mask try setting the value in the range [0.7-1.0].        
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The CGLS reconstructed volume as a CuPy array.
    """
    from tomobar.methodsIR_CuPy import RecToolsIRCuPy

    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
        
    RecToolsCP = RecToolsIRCuPy(DetectorsDimH=data.shape[2],  # Horizontal detector dimension
                                 DetectorsDimV=data.shape[1],  # Vertical detector dimension (3D case)
                                 CenterRotOffset=data.shape[2] / 2 - center - 0.5,  # Center of Rotation scalar or a vector
                                 AnglesVec=-angles,  # A vector of projection angles in radians
                                 ObjSize=recon_size,  # Reconstructed object dimensions (scalar)
                                 device_projector=gpu_id,
                                 )
    _data_ = {"projection_norm_data": data}  # data dictionary
    _algorithm_ = {"iterations": iterations, "nonnegativity": nonnegativity}
    reconstruction = RecToolsCP.CGLS(_data_, _algorithm_)
    cp._default_memory_pool.free_all_blocks()
    # perform masking to the result of reconstruction if needed 
    if recon_mask_radius is not None:
        reconstruction = _apply_circular_mask(reconstruction, recon_mask_radius)
    return cp.swapaxes(reconstruction,0,1)
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##