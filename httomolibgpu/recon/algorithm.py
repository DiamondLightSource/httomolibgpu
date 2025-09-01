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
# Changes relative to ToMoBAR 2024.01 version
# ---------------------------------------------------------------------------
"""Module for tomographic reconstruction. For more detailed information see :ref:`image_reconstruction_module`"""

import numpy as np
from httomolibgpu import cupywrapper

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run

from unittest.mock import Mock

if cupy_run:
    from tomobar.methodsDIR import RecToolsDIR
    from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy
    from tomobar.methodsIR_CuPy import RecToolsIRCuPy
else:
    RecToolsDIR = Mock()
    RecToolsDIRCuPy = Mock()
    RecToolsIRCuPy = Mock()

from numpy import float32, complex64
from typing import Optional, Type

from httomolibgpu.misc.supp_func import data_checker


__all__ = [
    "FBP2d_astra",
    "FBP3d_tomobar",
    "LPRec3d_tomobar",
    "SIRT3d_tomobar",
    "CGLS3d_tomobar",
]

input_data_axis_labels = ["angles", "detY", "detX"]  # set the labels of the input data


## %%%%%%%%%%%%%%%%%%%%%%% FBP2d_astra reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def FBP2d_astra(
    data: np.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: int = 0,
    filter_type: str = "ram-lak",
    filter_parameter: Optional[float] = None,
    filter_d: Optional[float] = None,
    recon_size: Optional[int] = None,
    recon_mask_radius: float = 0.95,
    neglog: bool = False,
    gpu_id: int = 0,
) -> np.ndarray:
    """
    Perform Filtered Backprojection (FBP) reconstruction slice-by-slice (2d) using ASTRA toolbox :cite:`van2016fast` and
    ToMoBAR :cite:`kazantsev2020tomographic` wrappers.
    This is a 2D recon using ASTRA's API for the FBP_CUDA method, see more in :ref:`method_FBP2d_astra`.

    Parameters
    ----------
    data : np.ndarray
        Projection data as a 3d numpy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    detector_pad : int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction.
    filter_type: str
        Type of projection filter, see ASTRA's API for all available options for filters.
    filter_parameter: float, optional
        Parameter value for the 'tukey', 'gaussian', 'blackman' and 'kaiser' filter types.
    filter_d: float, optional
        D parameter value for 'shepp-logan', 'cosine', 'hamming' and 'hann' filter types.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: float
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the given diameter will be set to zero.
        It is recommended to keep the value in the range [0.7-1.0].
    neglog: bool
        Take negative logarithm on input data to convert to attenuation coefficient or a density of the scanned object. Defaults to False,
        assuming that the negative log is taken either in normalisation procedure on with Paganin filter application.
    gpu_id : int
        A GPU device index to perform operation on.

    Returns
    -------
    np.ndarray
        The FBP reconstructed volume as a numpy array.
    """
    data = data_checker(data, verbosity=True, method_name="FBP2d_astra")

    data_shape = np.shape(data)
    if recon_size is None:
        recon_size = data_shape[2]

    RecTools = _instantiate_direct_recon2d_class(
        data, angles, center, detector_pad, recon_size, gpu_id
    )

    detY_size = data_shape[1]
    reconstruction = np.empty(
        (recon_size, detY_size, recon_size), dtype=np.float32(), order="C"
    )
    _take_neg_log_np(data) if neglog else data

    # loop over detY slices
    for slice_index in range(0, detY_size):
        reconstruction[:, slice_index, :] = np.flipud(
            RecTools.FBP(
                data[:, slice_index, :],
                filter_type=filter_type,
                filter_parameter=filter_parameter,
                filter_d=filter_d,
                recon_mask_radius=recon_mask_radius,
            )
        )
    return reconstruction


## %%%%%%%%%%%%%%%%%%%%%%% FBP reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def FBP3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: int = 0,
    filter_freq_cutoff: float = 0.35,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = 0.95,
    neglog: bool = False,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform Filtered Backprojection (FBP) reconstruction using ASTRA toolbox :cite:`van2016fast` and
    ToMoBAR :cite:`kazantsev2020tomographic` wrappers.
    This is a 3D recon from the CuPy array directly and using a custom built SINC filter for filtration in Fourier space, 
    see more in :ref:`method_FBP3d_tomobar`.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a 3d CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    detector_pad : int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction.
    filter_freq_cutoff : float
        Cutoff frequency parameter for the SINC filter, the lower values may produce better contrast but noisy reconstruction. The filter change will also affect the dynamic range of the reconstructed image. 
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: float, optional
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the given diameter will be set to zero.
        It is recommended to keep the value in the range [0.7-1.0].
    neglog: bool
        Take negative logarithm on input data to convert to attenuation coefficient or a density of the scanned object. Defaults to False,
        assuming that the negative log is taken either in normalisation procedure on with Paganin filter application.
    gpu_id : int
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        FBP reconstructed volume as a CuPy array.
    """
    data = data_checker(data, verbosity=True, method_name="FBP3d_tomobar")

    RecToolsCP = _instantiate_direct_recon_class(
        data, angles, center, detector_pad, recon_size, gpu_id
    )

    reconstruction = RecToolsCP.FBP(
        _take_neg_log(data) if neglog else data,
        cutoff_freq=filter_freq_cutoff,
        recon_mask_radius=recon_mask_radius,
        data_axes_labels_order=input_data_axis_labels,
    )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% LPRec  %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def LPRec3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: int = 0,
    filter_type: str = "shepp",
    filter_freq_cutoff: float = 1.0,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = 0.95,
    neglog: bool = False,
) -> cp.ndarray:
    """
    Fourier direct inversion in 3D on unequally spaced (also called as Log-Polar) grids using
    CuPy array as an input. This implementation follows V. Nikitin's CUDA-C implementation and TomoCuPy package.
    :cite:`andersson2016fast`, see more in :ref:`method_LPRec3d_tomobar`.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a 3d CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    detector_pad : int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction.
    filter_type : str
        Filter type, the accepted strings are: none, ramp, shepp, cosine, cosine2, hamming, hann, parzen.
    filter_freq_cutoff : float
        Cutoff frequency parameter for a filter.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: float, optional
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the given diameter will be set to zero.
        It is recommended to keep the value in the range [0.7-1.0].
    neglog: bool
        Take negative logarithm on input data to convert to attenuation coefficient or a density of the scanned object. Defaults to False,
        assuming that the negative log is taken either in normalisation procedure on with Paganin filter application.

    Returns
    -------
    cp.ndarray
        The Log-polar Fourier reconstructed volume as a CuPy array.
    """

    data = data_checker(data, verbosity=True, method_name="LPRec3d_tomobar")

    RecToolsCP = _instantiate_direct_recon_class(
        data, angles, center, detector_pad, recon_size, 0
    )

    reconstruction = RecToolsCP.FOURIER_INV(
        _take_neg_log(data) if neglog else data,
        recon_mask_radius=recon_mask_radius,
        data_axes_labels_order=input_data_axis_labels,
        filter_type=filter_type,
        cutoff_freq=filter_freq_cutoff,
    )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% SIRT reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def SIRT3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: int = 0,
    recon_size: Optional[int] = None,
    iterations: Optional[int] = 300,
    nonnegativity: Optional[bool] = True,
    neglog: bool = False,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform Simultaneous Iterative Recostruction Technique (SIRT) using ASTRA toolbox :cite:`van2016fast` and
    ToMoBAR :cite:`kazantsev2020tomographic` wrappers.
    This is 3D recon directly from a CuPy array while using ASTRA GPUlink capability to avoid host-device
    transactions for projection and backprojection.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    detector_pad : int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    iterations : int, optional
        The number of SIRT iterations.
    nonnegativity : bool, optional
        Impose nonnegativity constraint on reconstructed image.
    neglog: bool
        Take negative logarithm on input data to convert to attenuation coefficient or a density of the scanned object. Defaults to False,
        assuming that the negative log is taken either in normalisation procedure on with Paganin filter application.
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The SIRT reconstructed volume as a CuPy array.
    """
    data = data_checker(data, verbosity=True, method_name="SIRT3d_tomobar")

    RecToolsCP = _instantiate_iterative_recon_class(
        data,
        angles,
        center,
        detector_pad,
        recon_size,
        gpu_id,
        datafidelity="LS",
    )

    _data_ = {
        "projection_norm_data": _take_neg_log(data) if neglog else data,
        "data_axes_labels_order": input_data_axis_labels,
    }  # data dictionary
    _algorithm_ = {
        "iterations": iterations,
        "nonnegativity": nonnegativity,
    }
    reconstruction = RecToolsCP.SIRT(_data_, _algorithm_)
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% CGLS reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def CGLS3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: int = 0,
    recon_size: Optional[int] = None,
    iterations: Optional[int] = 20,
    nonnegativity: Optional[bool] = True,
    neglog: bool = False,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform Conjugate Gradient Least Squares (CGLS) using ASTRA toolbox :cite:`van2016fast` and
    ToMoBAR :cite:`kazantsev2020tomographic` wrappers.
    This is 3D recon directly from a CuPy array while using ASTRA GPUlink capability to avoid host-device
    transactions for projection and backprojection.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    detector_pad : int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    iterations : int, optional
        The number of CGLS iterations.
    nonnegativity : bool, optional
        Impose nonnegativity constraint on reconstructed image.
    neglog: bool
        Take negative logarithm on input data to convert to attenuation coefficient or a density of the scanned object. Defaults to False,
        assuming that the negative log is taken either in normalisation procedure on with Paganin filter application.
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The CGLS reconstructed volume as a CuPy array.
    """
    data = data_checker(data, verbosity=True, method_name="CGLS3d_tomobar")

    RecToolsCP = _instantiate_iterative_recon_class(
        data, angles, center, detector_pad, recon_size, gpu_id, datafidelity="LS"
    )

    _data_ = {
        "projection_norm_data": _take_neg_log(data) if neglog else data,
        "data_axes_labels_order": input_data_axis_labels,
    }  # data dictionary
    _algorithm_ = {"iterations": iterations, "nonnegativity": nonnegativity}
    reconstruction = RecToolsCP.CGLS(_data_, _algorithm_)
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def _instantiate_direct_recon_class(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: int = 0,
    recon_size: Optional[int] = None,
    gpu_id: int = 0,
) -> Type:
    """instantiate ToMoBAR's direct recon class

    Args:
        data (cp.ndarray): data array
        angles (np.ndarray): angles
        center (Optional[float], optional): center of recon. Defaults to None.
        detector_pad (int): Detector width padding. Defaults to 0.
        recon_size (Optional[int], optional): recon_size. Defaults to None.
        gpu_id (int, optional): gpu ID. Defaults to 0.

    Returns:
        Type[RecToolsDIRCuPy]: an instance of the direct recon class
    """
    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=data.shape[2],  # Horizontal detector dimension
        DetectorsDimH_pad=detector_pad,  # padding for horizontal detector
        DetectorsDimV=data.shape[1],  # Vertical detector dimension (3D case)
        CenterRotOffset=data.shape[2] / 2
        - center
        - 0.5,  # Center of Rotation scalar or a vector
        AnglesVec=-angles,  # A vector of projection angles in radians
        ObjSize=recon_size,  # Reconstructed object dimensions (scalar)
        device_projector=gpu_id,
    )
    return RecToolsCP


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def _instantiate_direct_recon2d_class(
    data: np.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: int = 0,
    recon_size: Optional[int] = None,
    gpu_id: int = 0,
) -> Type:
    """instantiate ToMoBAR's direct recon class for 2d reconstruction

    Args:
        data (cp.ndarray): data array
        angles (np.ndarray): angles
        center (Optional[float], optional): center of recon. Defaults to None.
        detector_pad (int): Detector width padding. Defaults to 0.
        recon_size (Optional[int], optional): recon_size. Defaults to None.
        gpu_id (int, optional): gpu ID. Defaults to 0.

    Returns:
        Type[RecToolsDIR]: an instance of the direct recon class
    """
    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
    RecTools = RecToolsDIR(
        DetectorsDimH=data.shape[2],  # Horizontal detector dimension
        DetectorsDimH_pad=detector_pad,  # padding for horizontal detector
        DetectorsDimV=None,  # 2d case
        CenterRotOffset=data.shape[2] / 2
        - center
        - 0.5,  # Center of Rotation scalar or a vector
        AnglesVec=-angles,  # A vector of projection angles in radians
        ObjSize=recon_size,  # Reconstructed object dimensions (scalar)
        device_projector=gpu_id,
    )
    return RecTools


def _instantiate_iterative_recon_class(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: int = 0,
    recon_size: Optional[int] = None,
    gpu_id: int = 0,
    datafidelity: str = "LS",
) -> Type:
    """instantiate ToMoBAR's iterative recon class

    Args:
        data (cp.ndarray): data array
        angles (np.ndarray): angles
        center (Optional[float], optional): center of recon. Defaults to None.
        detector_pad (int): Detector width padding. Defaults to 0.
        recon_size (Optional[int], optional): recon_size. Defaults to None.
        datafidelity (str, optional): Data fidelity
        gpu_id (int, optional): gpu ID. Defaults to 0.

    Returns:
        Type[RecToolsIRCuPy]: an instance of the iterative class
    """
    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
    RecToolsCP = RecToolsIRCuPy(
        DetectorsDimH=data.shape[2],  # Horizontal detector dimension
        DetectorsDimH_pad=detector_pad,  # padding for horizontal detector
        DetectorsDimV=data.shape[1],  # Vertical detector dimension (3D case)
        CenterRotOffset=data.shape[2] / 2
        - center
        - 0.5,  # Center of Rotation scalar or a vector
        AnglesVec=-angles,  # A vector of projection angles in radians
        ObjSize=recon_size,  # Reconstructed object dimensions (scalar)
        datafidelity=datafidelity,
        device_projector=gpu_id,
    )
    return RecToolsCP


def _take_neg_log(data: cp.ndarray) -> cp.ndarray:
    """Taking negative log"""
    data[data <= 0] = 1
    data = -cp.log(data)
    data[cp.isnan(data)] = 6.0
    data[cp.isinf(data)] = 0
    return data


def _take_neg_log_np(data: np.ndarray) -> np.ndarray:
    """Taking negative log"""
    data[data <= 0] = 1
    data = -np.log(data)
    data[np.isnan(data)] = 6.0
    data[np.isinf(data)] = 0
    return data
