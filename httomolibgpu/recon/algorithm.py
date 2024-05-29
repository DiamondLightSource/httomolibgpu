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
"""Module for tomographic reconstruction"""

import numpy as np
from httomolibgpu import cupywrapper

cp = cupywrapper.cp

nvtx = cupywrapper.nvtx
from numpy import float32, complex64
from typing import Optional, Type

__all__ = [
    "FBP",
    "LPRec",
    "SIRT",
    "CGLS",
    "FBP_CIL",
]

input_data_axis_labels = ["angles", "detY", "detX"]  # set the labels of the input data


## %%%%%%%%%%%%%%%%%%%%%%% FBP reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def FBP(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    filter_freq_cutoff: Optional[float] = 0.6,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = None,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform Filtered Backprojection (FBP) reconstruction using ASTRA toolbox :cite:`van2016fast` and
    ToMoBAR :cite:`kazantsev2020tomographic` wrappers.
    This is a 3D recon from a CuPy array directly and a custom built filter.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    filter_freq_cutoff : float, optional
        Cutoff frequency parameter for the sinc filter, the lowest values produce more crispy but noisy reconstruction.
    recon_size : int, optional
        The [recon_size, recon_size] shape of the reconstructed slice in pixels.
        By default (None), the reconstructed size will be the dimension of the horizontal detector.
    recon_mask_radius: float, optional
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the diameter will be set to zero.
        None by default, to see the effect of the mask try setting the value in the range [0.7-1.0].
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The FBP reconstructed volume as a CuPy array.
    """
    if cupywrapper.cupy_run:
        return __FBP(
            data,
            angles,
            center,
            filter_freq_cutoff,
            recon_size,
            recon_mask_radius,
            gpu_id,
        )
    else:
        print("FBP won't be executed because CuPy is not installed")
        return data


@nvtx.annotate()
def __FBP(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    filter_freq_cutoff: Optional[float] = 0.6,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = None,
    gpu_id: int = 0,
) -> cp.ndarray:

    RecToolsCP = _instantiate_direct_recon_class(
        data, angles, center, recon_size, gpu_id
    )

    reconstruction = RecToolsCP.FBP(
        data,
        cutoff_freq=filter_freq_cutoff,
        recon_mask_radius=recon_mask_radius,
        data_axes_labels_order=input_data_axis_labels,
    )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


def FBP_CIL(
    data: np.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    objsize: Optional[int] = None,
    gpu_id: int = 0,
    filter: str = "ram-lak",
    num_slices: int = None,
) -> np.ndarray:
    """
    Perform Filtered Backprojection (FBP) reconstruction using CIL.

    Parameters
    ----------
    data : np.ndarray
        Projection data as a NumPy array.
    objsize : int, optional
        The size in pixels of the reconstructed object.
    gpu_id : int, optional
        A GPU device index to perform operation on.
    filter : str, optional
        The Filter of the FBP algorithm, default 'ram-lak'
    num_slices : int, optional
        The number of slices to process in each chunk.


    Returns
    -------
    np.ndarray
        The FBP reconstructed volume as a NumPy array.
    """
    data = np.swapaxes(data, 0, 1)
    if center is None:
        center = data.shape[2] / 2  # making a crude guess

    from cil.framework import AcquisitionGeometry, ImageGeometry, AcquisitionData
    from cil.recon import FBP

    if num_slices is None:
        num_slices = data.shape[0]

    # create acquisition geometry
    if len(data.shape) > 2:
        det_y, num_angles, det_x = data.shape
        # this will depend on the assumptions of the data. Is beginning or middle of pixel?
        panel_centre = (det_x - 1) / 2
        ag = AcquisitionGeometry.create_Parallel3D(
            rotation_axis_position=(center - panel_centre, 0, 0)
        )
        ag.set_panel(num_pixels=(det_x, det_y), pixel_size=(1.0, 1.0))
        ag.dimension_labels = [
            AcquisitionGeometry.VERTICAL,
            AcquisitionGeometry.ANGLE,
            AcquisitionGeometry.HORIZONTAL,
        ]

        if objsize is not None:
            ig = ImageGeometry(
                voxel_num_x=objsize,
                voxel_num_y=objsize,
                voxel_num_z=det_y,
                voxel_size_x=1.0,
                voxel_size_y=1.0,
                voxel_size_z=1.0,
            )

    else:
        num_angles, det_x = data.shape
        # this will depend on the assumptions of the data. Is beginning or middle of pixel?
        panel_centre = (det_x - 1) // 2
        ag = AcquisitionGeometry.create_Parallel2D(
            rotation_axis_position=(center - panel_centre, 0)
        )
        ag.set_panel(num_pixels=det_x, pixel_size=1.0)
        ag.dimension_labels = [
            AcquisitionGeometry.ANGLE,
            AcquisitionGeometry.HORIZONTAL,
        ]

        if objsize is not None:
            ig = ImageGeometry(
                voxel_num_x=objsize,
                voxel_num_y=objsize,
                voxel_size_x=1.0,
                voxel_size_y=1.0,
            )

    ag.set_angles(-angles, angle_unit="radian")

    if objsize is None:
        ig = ag.get_ImageGeometry()

    # create acquisition data
    adata = AcquisitionData(data, geometry=ag, deep_copy=False)

    # make sure the data is in the correct order
    adata.reorder(order="astra")

    # create the FBP recon with backend ASTRA
    fbp = FBP(input=adata, image_geometry=ig, filter=filter, backend="astra")

    # if num_slices is not None:
    #    fbp.set_split_processing(slices_per_chunk=num_slices)

    # if gpu_id is not None:
    #     import astra

    #     astra.set_gpu_index(gpu_id)

    reconstruction = fbp.run(verbose=0)
    reconstruction.reorder(
        [ImageGeometry.HORIZONTAL_X, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y]
    )

    # return numpy array
    return reconstruction.as_array()


## %%%%%%%%%%%%%%%%%%%%%%% LPRec  %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def LPRec(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = None,
) -> cp.ndarray:
    """
    Fourier direct inversion in 3D on unequally spaced (also called as Log-Polar) grids using
    CuPy array as an input. This implementation follows V. Nikitin's CUDA-C implementation and TomoCuPy package.
    :cite:`andersson2016fast`.

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
    recon_mask_radius: float, optional
        The radius of the circular mask that applies to the reconstructed slice in order to crop
        out some undesirable artifacts. The values outside the diameter will be set to zero.
        None by default, to see the effect of the mask try setting the value in the range [0.7-1.0].

    Returns
    -------
    cp.ndarray
        The Log-polar Fourier reconstructed volume as a CuPy array.
    """
    if cupywrapper.cupy_run:
        return __LPRec(
            data,
            angles,
            center,
            recon_size,
            recon_mask_radius,
        )
    else:
        print("LPRec won't be executed because CuPy is not installed")
        return data


def __LPRec(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = None,
) -> cp.ndarray:

    RecToolsCP = _instantiate_direct_recon_class(data, angles, center, recon_size, 0)

    reconstruction = RecToolsCP.FOURIER_INV(
        data,
        recon_mask_radius=recon_mask_radius,
        data_axes_labels_order=input_data_axis_labels,
    )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")



## %%%%%%%%%%%%%%%%%%%%%%% SIRT reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def SIRT(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,
    iterations: Optional[int] = 300,
    nonnegativity: Optional[bool] = True,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform Simultaneous Iterative Recostruction Technique (SIRT) using ASTRA toolbox :cite:`van2016fast` and
    ToMoBAR :cite:`kazantsev2020tomographic` wrappers.
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
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The SIRT reconstructed volume as a CuPy array.
    """
    if cupywrapper.cupy_run:
        return __SIRT(
            data,
            angles,
            center,
            recon_size,
            iterations,
            nonnegativity,
            gpu_id,
        )
    else:
        print("SIRT won't be executed because CuPy is not installed")
        return data


@nvtx.annotate()
def __SIRT(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,
    iterations: Optional[int] = 300,
    nonnegativity: Optional[bool] = True,
    gpu_id: int = 0,
) -> cp.ndarray:

    RecToolsCP = _instantiate_iterative_recon_class(
        data, angles, center, recon_size, gpu_id, datafidelity="LS"
    )

    _data_ = {
        "projection_norm_data": data,
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
def CGLS(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,
    iterations: Optional[int] = 20,
    nonnegativity: Optional[bool] = True,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform Congugate Gradient Least Squares (CGLS) using ASTRA toolbox :cite:`van2016fast` and
    ToMoBAR :cite:`kazantsev2020tomographic` wrappers.
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
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The CGLS reconstructed volume as a CuPy array.
    """
    if cupywrapper.cupy_run:
        return __CGLS(
            data,
            angles,
            center,
            recon_size,
            iterations,
            nonnegativity,
            gpu_id,
        )
    else:
        print("CGLS won't be executed because CuPy is not installed")
        return data


@nvtx.annotate()
def __CGLS(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,
    iterations: Optional[int] = 20,
    nonnegativity: Optional[bool] = True,
    gpu_id: int = 0,
) -> cp.ndarray:

    RecToolsCP = _instantiate_iterative_recon_class(
        data, angles, center, recon_size, gpu_id, datafidelity="LS"
    )

    _data_ = {
        "projection_norm_data": data,
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
    recon_size: Optional[int] = None,
    gpu_id: int = 0,
) -> Type:
    """instantiate ToMoBAR's direct recon class

    Args:
        data (cp.ndarray): data array
        angles (np.ndarray): angles
        center (Optional[float], optional): center of recon. Defaults to None.
        recon_size (Optional[int], optional): recon_size. Defaults to None.
        gpu_id (int, optional): gpu ID. Defaults to 0.

    Returns:
        Type[RecToolsDIRCuPy]: an instance of the direct recon class
    """
    from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=data.shape[2],  # Horizontal detector dimension
        DetectorsDimV=data.shape[1],  # Vertical detector dimension (3D case)
        CenterRotOffset=data.shape[2] / 2
        - center
        - 0.5,  # Center of Rotation scalar or a vector
        AnglesVec=-angles,  # A vector of projection angles in radians
        ObjSize=recon_size,  # Reconstructed object dimensions (scalar)
        device_projector=gpu_id,
    )
    return RecToolsCP


def _instantiate_iterative_recon_class(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    recon_size: Optional[int] = None,
    gpu_id: int = 0,
    datafidelity: str = "LS",
) -> Type:
    """instantiate ToMoBAR's iterative recon class

    Args:
        data (cp.ndarray): data array
        angles (np.ndarray): angles
        center (Optional[float], optional): center of recon. Defaults to None.
        recon_size (Optional[int], optional): recon_size. Defaults to None.
        datafidelity (str, optional): Data fidelity
        gpu_id (int, optional): gpu ID. Defaults to 0.

    Returns:
        Type[RecToolsIRCuPy]: an instance of the iterative class
    """
    from tomobar.methodsIR_CuPy import RecToolsIRCuPy

    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if recon_size is None:
        recon_size = data.shape[2]
    RecToolsCP = RecToolsIRCuPy(
        DetectorsDimH=data.shape[2],  # Horizontal detector dimension
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
