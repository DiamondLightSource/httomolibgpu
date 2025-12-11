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
"""Module for tomographic reconstruction. For more detailed information, see :ref:`image_reconstruction_module`"""

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

from numpy import float32
from typing import Optional, Type, Union


__all__ = [
    "FBP2d_astra",
    "FBP3d_tomobar",
    "LPRec3d_tomobar",
    "SIRT3d_tomobar",
    "CGLS3d_tomobar",
    "FISTA3d_tomobar",
]

input_data_axis_labels = ["angles", "detY", "detX"]  # set the labels of the input data


## %%%%%%%%%%%%%%%%%%%%%%% FBP2d_astra reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def FBP2d_astra(
    data: np.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    filter_type: str = "ram-lak",
    filter_parameter: Optional[float] = None,
    filter_d: Optional[float] = None,
    recon_size: Optional[int] = None,
    recon_mask_radius: float = 0.95,
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
    detector_pad : bool, int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction. Set to True to perform
        an automated padding or specify a certain value as an integer.
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
        To implement the cropping one can use the range [0.7-1.0] or set to 2.0 when no cropping required.
    gpu_id : int
        A GPU device index to perform operation on.

    Returns
    -------
    np.ndarray
        The FBP reconstructed volume as a numpy array.
    """

    data_shape = np.shape(data)
    if recon_size is None:
        recon_size = data_shape[2]

    RecTools = _instantiate_direct_recon2d_class(
        data, angles, center, detector_pad, recon_size, gpu_id
    )

    detY_size = data_shape[1]
    reconstruction = np.empty(
        (recon_size, detY_size, recon_size), dtype=float32, order="C"
    )
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
    detector_pad: Union[bool, int] = False,
    filter_freq_cutoff: float = 0.35,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = 0.95,
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
    detector_pad : bool, int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction. Set to True to perform
        an automated padding or specify a certain value as an integer.
    filter_freq_cutoff : float
        Cutoff frequency parameter for the SINC filter, the lower values may produce better contrast but noisy reconstruction. The filter change will also affect the dynamic range of the reconstructed image.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: float, optional
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the given diameter will be set to zero.
        To implement the cropping one can use the range [0.7-1.0] or set to 2.0 when no cropping required.
    gpu_id : int
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        FBP reconstructed volume as a CuPy array.
    """

    RecToolsCP = _instantiate_direct_recon_class(
        data, angles, center, detector_pad, recon_size, gpu_id
    )

    reconstruction = RecToolsCP.FBP(
        data,
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
    detector_pad: Union[bool, int] = False,
    filter_type: str = "shepp",
    filter_freq_cutoff: float = 1.0,
    recon_size: Optional[int] = None,
    recon_mask_radius: float = 0.95,
    power_of_2_oversampling: Optional[bool] = True,
    power_of_2_cropping: Optional[bool] = False,
    min_mem_usage_filter: Optional[bool] = True,
    min_mem_usage_ifft2: Optional[bool] = True,
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
    detector_pad : bool, int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction. Set to True to perform
        an automated padding or specify a certain value as an integer.
    filter_type : str
        Filter type, the accepted strings are: none, ramp, shepp, cosine, cosine2, hamming, hann, parzen.
    filter_freq_cutoff : float
        Cutoff frequency parameter for a filter.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: float
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the given diameter will be set to zero.
        To implement the cropping one can use the range [0.7-1.0] or set to 2.0 when no cropping required.

    Returns
    -------
    cp.ndarray
        The Log-polar Fourier reconstructed volume as a CuPy array.
    """

    RecToolsCP = _instantiate_direct_recon_class(
        data, angles, center, detector_pad, recon_size, 0
    )

    reconstruction = RecToolsCP.FOURIER_INV(
        data,
        recon_mask_radius=recon_mask_radius,
        data_axes_labels_order=input_data_axis_labels,
        filter_type=filter_type,
        cutoff_freq=filter_freq_cutoff,
        power_of_2_oversampling=power_of_2_oversampling,
        power_of_2_cropping=power_of_2_cropping,
        min_mem_usage_filter=min_mem_usage_filter,
        min_mem_usage_ifft2=min_mem_usage_ifft2,
    )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% SIRT reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def SIRT3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    recon_mask_radius: float = 0.95,
    iterations: int = 300,
    nonnegativity: bool = True,
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
    detector_pad : bool, int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction. Set to True to perform
        an automated padding or specify a certain value as an integer.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: float
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the given diameter will be set to zero.
        To implement the cropping one can use the range [0.7-1.0] or set to 2.0 when no cropping required.
    iterations : int
        The number of SIRT iterations.
    nonnegativity : bool
        Impose nonnegativity constraint on the reconstructed image.
    gpu_id : int
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The SIRT reconstructed volume as a CuPy array.
    """

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
        "projection_norm_data": data,
        "data_axes_labels_order": input_data_axis_labels,
    }  # data dictionary
    _algorithm_ = {
        "iterations": iterations,
        "nonnegativity": nonnegativity,
        "recon_mask_radius": recon_mask_radius,
    }
    reconstruction = RecToolsCP.SIRT(_data_, _algorithm_)
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% CGLS reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def CGLS3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    recon_mask_radius: float = 0.95,
    iterations: int = 20,
    nonnegativity: bool = True,
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
    detector_pad : bool, int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction. Set to True to perform
        an automated padding or specify a certain value as an integer.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: float
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the given diameter will be set to zero.
        To implement the cropping one can use the range [0.7-1.0] or set to 2.0 when no cropping required.
    iterations : int
        The number of CGLS iterations.
    nonnegativity : bool
        Impose nonnegativity constraint on reconstructed image.
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The CGLS reconstructed volume as a CuPy array.
    """

    RecToolsCP = _instantiate_iterative_recon_class(
        data, angles, center, detector_pad, recon_size, gpu_id, datafidelity="LS"
    )

    _data_ = {
        "projection_norm_data": data,
        "data_axes_labels_order": input_data_axis_labels,
    }  # data dictionary
    _algorithm_ = {
        "iterations": iterations,
        "nonnegativity": nonnegativity,
        "recon_mask_radius": recon_mask_radius,
    }
    reconstruction = RecToolsCP.CGLS(_data_, _algorithm_)
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% FISTA reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def FISTA3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    recon_mask_radius: float = 0.95,
    iterations: int = 20,
    subsets_number: int = 6,
    regularisation_type: str = "PD_TV",
    regularisation_parameter: float = 0.000001,
    regularisation_iterations: int = 50,
    regularisation_half_precision: bool = True,
    nonnegativity: bool = True,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    A Fast Iterative Shrinkage-Thresholding Algorithm :cite:`beck2009fast` with various types of regularisation or
    denoising operations :cite:`kazantsev2019ccpi` (currently accepts ROF_TV and PD_TV regularisations only).

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    detector_pad : bool, int
        Detector width padding with edge values to remove circle/arc type artifacts in the reconstruction. Set to True to perform
        an automated padding or specify a certain value as an integer.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: float
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the given diameter will be set to zero.
        To implement the cropping one can use the range [0.7-1.0] or set to 2.0 when no cropping required.
    iterations : int
        The number of FISTA algorithm iterations.
    subsets_number: int
        The number of the ordered subsets to accelerate convergence. Keep the value bellow 10 to avoid divergence.
    regularisation_type: str
        A method to use for regularisation. Currently PD_TV and ROF_TV are available.
    regularisation_parameter: float
        The main regularisation parameter to control the amount of smoothing/noise removal. Larger values lead to stronger smoothing.
    regularisation_iterations: int
        The number of iterations for regularisers (aka INNER iterations).
    regularisation_half_precision: bool
        Perform faster regularisation computation in half-precision with a very minimal sacrifice in quality.
    nonnegativity : bool
        Impose nonnegativity constraint on the reconstructed image.
    gpu_id : int
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The FISTA reconstructed volume as a CuPy array.
    """
    RecToolsCP = _instantiate_iterative_recon_class(
        data, angles, center, detector_pad, recon_size, gpu_id, datafidelity="LS"
    )

    _data_ = {
        "projection_norm_data": data,
        "OS_number": subsets_number,
        "data_axes_labels_order": input_data_axis_labels,
    }
    lc = RecToolsCP.powermethod(_data_)  # calculate Lipschitz constant (run once)

    _algorithm_ = {
        "iterations": iterations,
        "lipschitz_const": lc.get(),
        "nonnegativity": nonnegativity,
        "recon_mask_radius": recon_mask_radius,
    }

    _regularisation_ = {
        "method": regularisation_type,  # Selected regularisation method
        "regul_param": regularisation_parameter,  # Regularisation parameter
        "iterations": regularisation_iterations,  # The number of regularisation iterations
        "half_precision": regularisation_half_precision,  # enabling half-precision calculation
    }

    reconstruction = RecToolsCP.FISTA(_data_, _algorithm_, _regularisation_)
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def _instantiate_direct_recon_class(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    gpu_id: int = 0,
) -> Type:
    """instantiate ToMoBAR's direct recon class

    Args:
        data (cp.ndarray): data array
        angles (np.ndarray): angles
        center (Optional[float], optional): center of recon. Defaults to None.
        detector_pad : (Union[bool, int]) : Detector width padding. Defaults to False.
        recon_size (Optional[int], optional): recon_size. Defaults to None.
        gpu_id (int, optional): gpu ID. Defaults to 0.

    Returns:
        Type[RecToolsDIRCuPy]: an instance of the direct recon class
    """
    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
    if detector_pad is True:
        detector_pad = __estimate_detectorHoriz_padding(data.shape[2])
    elif detector_pad is False:
        detector_pad = 0
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
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    gpu_id: int = 0,
) -> Type:
    """instantiate ToMoBAR's direct recon class for 2d reconstruction

    Args:
        data (cp.ndarray): data array
        angles (np.ndarray): angles
        center (Optional[float], optional): center of recon. Defaults to None.
        detector_pad : (Union[bool, int]) : Detector width padding. Defaults to False.
        recon_size (Optional[int], optional): recon_size. Defaults to None.
        gpu_id (int, optional): gpu ID. Defaults to 0.

    Returns:
        Type[RecToolsDIR]: an instance of the direct recon class
    """
    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
    if detector_pad is True:
        detector_pad = __estimate_detectorHoriz_padding(data.shape[2])
    elif detector_pad is False:
        detector_pad = 0
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
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    gpu_id: int = 0,
    datafidelity: str = "LS",
) -> Type:
    """instantiate ToMoBAR's iterative recon class

    Args:
        data (cp.ndarray): data array
        angles (np.ndarray): angles
        center (Optional[float], optional): center of recon. Defaults to None.
        detector_pad : (Union[bool, int]) : Detector width padding. Defaults to False.
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
    if detector_pad is True:
        detector_pad = __estimate_detectorHoriz_padding(data.shape[2])
    elif detector_pad is False:
        detector_pad = 0
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


def __estimate_detectorHoriz_padding(detX_size) -> int:
    det_half = detX_size // 2
    padded_value_exact = int(np.sqrt(2 * (det_half**2))) - det_half
    padded_add_margin = padded_value_exact // 2
    return padded_value_exact + padded_add_margin
