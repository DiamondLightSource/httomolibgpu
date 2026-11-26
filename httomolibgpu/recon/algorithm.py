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
# Changes relative to ToMoBAR 2026.04 version
# ---------------------------------------------------------------------------
"""Module for tomographic reconstruction. For more detailed information, see :ref:`image_reconstruction_module`

* :func:`~httomolibgpu.recon.algorithm.FBP2d_astra`

* :func:`~httomolibgpu.recon.algorithm.FBP3d_tomobar`

* :func:`~httomolibgpu.recon.algorithm.LPRec3d_tomobar`

* :func:`~httomolibgpu.recon.algorithm.SIRT3d_tomobar`

* :func:`~httomolibgpu.recon.algorithm.CGLS3d_tomobar`

* :func:`~httomolibgpu.recon.algorithm.FISTA3d_tomobar`

* :func:`~httomolibgpu.recon.algorithm.ADMM3d_tomobar`

* :func:`~httomolibgpu.recon.algorithm.OSEM3d_tomobar`

* :func:`~httomolibgpu.recon.algorithm.iterative_alignment`

"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Optional, Tuple, Type, Union

from numpy import float32
import numpy as np

from httomolibgpu import cupywrapper
from httomolibgpu.misc.utils import (
    __check_if_data_3D_array,
    __check_if_data_correct_type,
    __check_variable_type,
)

cp = cupywrapper.cp
cupy_run = cupywrapper.cupy_run


if cupy_run:
    from tomobar.geometry import (
        Parallel2DGeometry,
        Parallel3DGeometry,
        Volume2DGeometry,
        Volume3DGeometry,
    )
    from tomobar.options import DataFidelity, Diagnostics, Regularisation
    from tomobar.reconstruction.direct import DirectReconstructor
    from tomobar.reconstruction.iterative import IterativeReconstructor
    from tomobar.reconstruction.memory import DeviceMemStack
else:
    DeviceMemStack = cupywrapper._UnavailableGPU("DeviceMemStack")
    Parallel2DGeometry = cupywrapper._UnavailableGPU("Parallel2DGeometry")
    Parallel3DGeometry = cupywrapper._UnavailableGPU("Parallel3DGeometry")
    Volume2DGeometry = cupywrapper._UnavailableGPU("Volume2DGeometry")
    Volume3DGeometry = cupywrapper._UnavailableGPU("Volume3DGeometry")
    DirectReconstructor = cupywrapper._UnavailableGPU("DirectReconstructor")
    IterativeReconstructor = cupywrapper._UnavailableGPU("IterativeReconstructor")
    DataFidelity = cupywrapper._UnavailableGPU("DataFidelity")
    Regularisation = cupywrapper._UnavailableGPU("Regularisation")
    Diagnostics = cupywrapper._UnavailableGPU("Diagnostics")


__all__ = [
    "FBP2d_astra",
    "FBP3d_tomobar",
    "LPRec3d_tomobar",
    "SIRT3d_tomobar",
    "CGLS3d_tomobar",
    "FISTA3d_tomobar",
    "ADMM3d_tomobar",
    "OSEM3d_tomobar",
    "iterative_alignment",
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
    recon_mask_radius: Optional[float] = 0.95,
    gpu_id: int = 0,
) -> np.ndarray:
    """Slice-wise ASTRA FBP reconstruction. See :ref:`method_FBP2d_astra`.

    Parameters
    ----------
    data : np.ndarray
        NumPy projections, shape (angles, detector rows, detector columns);
        float32 or uint16. Supply preprocessed line integrals.
    angles : np.ndarray
        NumPy projection angles in radians, one per input view.
    center : float, optional
        Horizontal CoR in pixels before padding; None uses detector width // 2.
    detector_pad : bool, int
        Edge padding per detector side: False disables, True estimates,
        an integer specifies pixels. Does not itself enlarge the output.
    filter_type : str
        ASTRA projection filter; supported names are validated by this wrapper.
    filter_parameter : float, optional
        Filter parameter for tukey, gaussian, blackman or kaiser.
    filter_d : float, optional
        D parameter for shepp-logan, cosine, hamming or hann.
    recon_size : int, optional
        Slice width and height in pixels; None uses unpadded detector width.
    recon_mask_radius : float, optional
        Positive circular-mask parameter: 1 is the inscribed circle, smaller
        values mask more, None disables. Output shape is unchanged.
    gpu_id : int
        CUDA device for reconstruction; NumPy slices are transferred to it.

    Returns
    -------
    numpy.ndarray
        C-contiguous float32 reconstruction, shape (recon_size, detector rows,
        recon_size). See recon_size for sizing rules.
    """
    ### Data and parameters checks ###
    methods_name = "FBP2d_astra"
    __common_data_parameters_check(
        data,
        angles,
        methods_name,
        center,
        detector_pad,
        recon_size,
        recon_mask_radius,
        gpu_id,
    )
    __check_variable_type(
        filter_type,
        [str],
        "filter_type",
        [
            "none",
            "ram-lak",
            "cosine",
            "shepp-logan",
            "tukey",
            "gaussian",
            "blackman",
            "kaiser",
            "lanczos",
            "hann",
        ],
        methods_name,
    )
    __check_variable_type(
        filter_parameter, [float, int, type(None)], "filter_parameter", [], methods_name
    )
    __check_variable_type(
        filter_d, [float, int, type(None)], "filter_d", [], methods_name
    )
    ###################################

    data_shape = np.shape(data)
    if recon_size is None:
        recon_size = data_shape[2]

    detY_size = data_shape[1]
    reconstruction = np.empty(
        (recon_size, detY_size, recon_size), dtype=float32, order="C"
    )
    # Reuse one projector across slices and release its resources on exit.
    with _instantiate_direct_recon2d_class(
        data, angles, center, detector_pad, recon_size, gpu_id
    ) as reconstructor:
        for slice_index in range(detY_size):
            # Retain the public wrapper's historical vertical orientation.
            reconstruction[:, slice_index, :] = np.flipud(
                reconstructor.FBP(
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
    """GPU 3D filtered backprojection. See :ref:`method_FBP3d_tomobar`.

    Parameters
    ----------
    data : cp.ndarray
        Projections in (angles, detector rows, detector columns) order; float32
        or uint16. Transmission methods expect preprocessed line integrals.
    angles : np.ndarray
        NumPy projection angles in radians, one per input view.
    center : float, optional
        Horizontal CoR in pixels before padding; None uses detector width // 2.
    detector_pad : bool, int
        Edge padding per detector side: False disables, True estimates,
        an integer specifies pixels. Does not itself enlarge the output.
    filter_freq_cutoff : float
        Positive SINC filter shape parameter; affects noise and intensity scale.
    recon_size : int, optional
        Slice width and height in pixels; None uses unpadded detector width.
    recon_mask_radius : float, optional
        Positive circular-mask parameter: 1 is the inscribed circle, smaller
        values mask more, None disables. Output shape is unchanged.
    gpu_id : int
        CUDA device index. CuPy input must reside on this device.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 reconstruction, shape (recon_size, detector rows,
        recon_size). See recon_size for sizing rules.
    """
    ### Data and parameters checks ###
    methods_name = "FBP3d_tomobar"
    __common_data_parameters_check(
        data,
        angles,
        methods_name,
        center,
        detector_pad,
        recon_size,
        recon_mask_radius,
        gpu_id,
    )
    __check_variable_type(
        filter_freq_cutoff, [float, int], "filter_freq_cutoff", [], methods_name
    )
    ###################################

    scan, grid, detector_pad = _build_reconstruction_geometry(
        data.shape, angles, center, detector_pad, recon_size
    )
    with DirectReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=detector_pad,
        device_index=gpu_id,
    ) as reconstructor:
        reconstruction = reconstructor.FBP(
            data,
            filter_type="SINC",
            cutoff_freq=filter_freq_cutoff,
            recon_mask_radius=recon_mask_radius,
            input_axes=input_data_axis_labels,
        )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% LPRec  %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def LPRec3d_tomobar(
    data: cp.ndarray | Tuple[int, int, int],
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
    gpu_id: int = 0,
    **kwargs,
) -> cp.ndarray:
    """GPU log-polar Fourier reconstruction. See :ref:`method_LPRec3d_tomobar`.

    Parameters
    ----------
    data : cp.ndarray or tuple[int, int, int]
        CuPy projections in (angles, detector rows, detector columns) order;
        float32 or uint16. A shape tuple enables DeviceMemStack memory estimation.
    angles : np.ndarray
        NumPy projection angles in radians, one per input view.
    center : float, optional
        Horizontal CoR in pixels before padding; None uses detector width // 2.
    detector_pad : bool, int
        Edge padding per detector side: False disables, True estimates,
        an integer specifies pixels. Does not itself enlarge the output.
    filter_type : str
        Fourier filter: none, ramp, shepp, cosine, cosine2, hamming, hann or parzen.
    filter_freq_cutoff : float
        Positive filter-response amplitude multiplier; scales intensities.
    recon_size : int, optional
        Slice width and height in pixels; None uses unpadded detector width.
    recon_mask_radius : float, optional
        Positive circular-mask parameter: 1 is the inscribed circle, smaller
        values mask more, None disables. Output shape is unchanged.
    power_of_2_oversampling : bool, optional
        Choose a power-of-two filtering FFT width; False uses oversampling_level.
    power_of_2_cropping : bool, optional
        Round working detector width up to a nearby power of two (within 10%).
        Returned volume shape is unchanged.
    min_mem_usage_filter : bool, optional
        Filter one detector row at a time; False uses row batches.
    min_mem_usage_ifft2 : bool, optional
        Transform one packed slice (two detector rows) at a time; False batches.
    gpu_id : int, optional
        CUDA device index. CuPy input must reside on this device.
    legacy_scaling : bool, optional
        Forwarded through kwargs. Defaults to True to preserve historical amplitude
        and odd-width cropping. False selects TomoBAR quantitative normalisation.

    **kwargs : dict, optional
        Additional :class:`tomobar.options.FourierOptions` fields.

    Returns
    -------
    cupy.ndarray or float
        C-contiguous float32 volume (Y, detector row, X), or estimated peak
        bytes in a DeviceMemStack context. Legacy odd-width cropping is retained
        when legacy_scaling=True.
    """
    ### Data and parameters checks ###
    methods_name = "LPRec3d_tomobar"
    __common_data_parameters_check(
        data,
        angles,
        methods_name,
        center,
        detector_pad,
        recon_size,
        recon_mask_radius,
        gpu_id=gpu_id,
        mem_stack=DeviceMemStack.instance(),
    )
    __check_variable_type(
        filter_type,
        [str],
        "filter_type",
        ["none", "ramp", "shepp", "cosine", "cosine2", "hamming", "hann", "parzen"],
        methods_name,
    )
    __check_variable_type(
        filter_freq_cutoff, [float, int], "filter_freq_cutoff", [], methods_name
    )
    ###################################

    data_shape = data if isinstance(data, tuple) else data.shape
    scan, grid, detector_pad = _build_reconstruction_geometry(
        data_shape, angles, center, detector_pad, recon_size
    )
    # Keep this wrapper's historical amplitude and odd-width crop convention.
    kwargs.setdefault("legacy_scaling", True)
    # Keep planning, FFT workspace estimation, cache cleanup and the final
    # contiguous transpose on the requested GPU, not the caller's active GPU.
    with cp.cuda.Device(gpu_id):
        with DirectReconstructor(
            projection_geometry=scan,
            volume_geometry=grid,
            detector_padding=detector_pad,
            device_index=gpu_id,
        ) as reconstructor:
            reconstruction = reconstructor.FOURIER_INV(
                data,
                recon_mask_radius=recon_mask_radius,
                input_axes=input_data_axis_labels,
                filter_type=filter_type,
                cutoff_freq=filter_freq_cutoff,
                power_of_2_oversampling=power_of_2_oversampling,
                power_of_2_cropping=power_of_2_cropping,
                min_mem_usage_filter=min_mem_usage_filter,
                min_mem_usage_ifft2=min_mem_usage_ifft2,
                **kwargs,
            )
        cp._default_memory_pool.free_all_blocks()

        mem_stack = DeviceMemStack.instance()
        if mem_stack:
            return mem_stack.highwater * 1.00625

        return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% SIRT reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def SIRT3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = 0.95,
    iterations: int = 300,
    nonnegativity: bool = True,
    gpu_id: int = 0,
) -> cp.ndarray:
    """GPU SIRT reconstruction. See :ref:`method_SIRT3d_tomobar`.

    Parameters
    ----------
    data : cp.ndarray
        Projections in (angles, detector rows, detector columns) order; float32
        or uint16. Transmission methods expect preprocessed line integrals.
    angles : np.ndarray
        NumPy projection angles in radians, one per input view.
    center : float, optional
        Horizontal CoR in pixels before padding; None uses detector width // 2.
    detector_pad : bool, int
        Edge padding per detector side: False disables, True estimates,
        an integer specifies pixels. Does not itself enlarge the output.
    recon_size : int, optional
        Slice width and height in pixels; None uses unpadded detector width.
    recon_mask_radius : float, optional
        Positive circular-mask parameter: 1 is the inscribed circle, smaller
        values mask more, None disables. Output shape is unchanged.
    iterations : int
        Number of SIRT iterations.
    nonnegativity : bool
        Constrain reconstructed values to be nonnegative.
    gpu_id : int
        CUDA device index. CuPy input must reside on this device.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 reconstruction, shape (recon_size, detector rows,
        recon_size). See recon_size for sizing rules.
    """
    ### Data and parameters checks ###
    methods_name = "SIRT3d_tomobar"
    __common_data_parameters_check(
        data,
        angles,
        methods_name,
        center,
        detector_pad,
        recon_size,
        recon_mask_radius,
        gpu_id,
    )
    __common_iterative_basic_parameters_check(methods_name, iterations, nonnegativity)
    ###################################

    scan, grid, detector_pad = _build_reconstruction_geometry(
        data.shape, angles, center, detector_pad, recon_size
    )
    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=detector_pad,
        device_index=gpu_id,
        backend="astra",
        num_subsets=1,
    ) as reconstructor:
        reconstruction = reconstructor.SIRT(
            data,
            input_axes=input_data_axis_labels,
            iterations=iterations,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
        )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% CGLS reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def CGLS3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = 0.95,
    iterations: int = 20,
    nonnegativity: bool = True,
    gpu_id: int = 0,
) -> cp.ndarray:
    """GPU CGLS reconstruction. See :ref:`method_CGLS3d_tomobar`.

    Parameters
    ----------
    data : cp.ndarray
        Projections in (angles, detector rows, detector columns) order; float32
        or uint16. Transmission methods expect preprocessed line integrals.
    angles : np.ndarray
        NumPy projection angles in radians, one per input view.
    center : float, optional
        Horizontal CoR in pixels before padding; None uses detector width // 2.
    detector_pad : bool, int
        Edge padding per detector side: False disables, True estimates,
        an integer specifies pixels. Does not itself enlarge the output.
    recon_size : int, optional
        Slice width and height in pixels; None uses unpadded detector width.
    recon_mask_radius : float, optional
        Positive circular-mask parameter: 1 is the inscribed circle, smaller
        values mask more, None disables. Output shape is unchanged.
    iterations : int
        Number of CGLS iterations.
    nonnegativity : bool
        Constrain reconstructed values to be nonnegative.
    gpu_id : int, optional
        CUDA device index. CuPy input must reside on this device.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 reconstruction, shape (recon_size, detector rows,
        recon_size). See recon_size for sizing rules.
    """
    ### Data and parameters checks ###
    methods_name = "CGLS3d_tomobar"
    __common_data_parameters_check(
        data,
        angles,
        methods_name,
        center,
        detector_pad,
        recon_size,
        recon_mask_radius,
        gpu_id,
    )
    __common_iterative_basic_parameters_check(methods_name, iterations, nonnegativity)
    ###################################

    scan, grid, detector_pad = _build_reconstruction_geometry(
        data.shape, angles, center, detector_pad, recon_size
    )
    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=detector_pad,
        device_index=gpu_id,
        backend="astra",
        num_subsets=1,
    ) as reconstructor:
        reconstruction = reconstructor.CGLS(
            data,
            input_axes=input_data_axis_labels,
            iterations=iterations,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
        )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% FISTA reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def FISTA3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = 0.95,
    iterations: int = 20,
    subsets_number: int = 6,
    data_fidelity: str = "LS",
    regularisation_type: Literal[
        "ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV", "NDF"
    ] = "PD_TV",
    regularisation_parameter: float = 0.000001,
    regularisation_iterations: int = 50,
    regularisation_half_precision: bool = True,
    nonnegativity: bool = True,
    gpu_id: int = 0,
    *,
    regularisation_options: Optional[dict] = None,
) -> cp.ndarray:
    """Regularised GPU FISTA reconstruction. See :ref:`method_FISTA3d_tomobar`.

    Parameters
    ----------
    data : cp.ndarray
        Projections in (angles, detector rows, detector columns) order; float32
        or uint16. Transmission methods expect preprocessed line integrals.
    angles : np.ndarray
        NumPy projection angles in radians, one per input view.
    center : float, optional
        Horizontal CoR in pixels before padding; None uses detector width // 2.
    detector_pad : bool, int
        Edge padding per detector side: False disables, True estimates,
        an integer specifies pixels. Does not itself enlarge the output.
    recon_size : int, optional
        Slice width and height in pixels; None uses unpadded detector width.
    recon_mask_radius : float, optional
        Positive circular-mask parameter: 1 is the inscribed circle, smaller
        values mask more, None disables. Output shape is unchanged.
    iterations : int
        Outer FISTA iterations; each visits all subsets.
    subsets_number : int
        Interleaved subset count; 1 uses full data. More subsets can destabilise
        convergence.
    data_fidelity : str
        "LS" only; this wrapper does not accept measurement weights.
    regularisation_type : str
        "ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV" or "NDF".
        See :ref:`iterative_regularisation`.
    regularisation_parameter : float
        Per-proximal-step smoothing strength; zero disables regularisation.
    regularisation_iterations : int
        Inner regulariser iterations per proximal update.
    regularisation_half_precision : bool
        Use float16 auxiliary fields; output stays float32. NDF requires False.
    nonnegativity : bool
        Constrain reconstructed values to be nonnegative.
    gpu_id : int
        CUDA device index. CuPy input must reside on this device.
    regularisation_options : dict, optional
        Native method-specific settings; None uses defaults.
        See :ref:`iterative_regularisation` for supported keys and examples.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 reconstruction, shape (recon_size, detector rows,
        recon_size). See recon_size for sizing rules.
    """
    ### Data and parameters checks ###
    methods_name = "FISTA3d_tomobar"
    __common_data_parameters_check(
        data,
        angles,
        methods_name,
        center,
        detector_pad,
        recon_size,
        recon_mask_radius,
        gpu_id,
    )
    __common_iterative_basic_parameters_check(methods_name, iterations, nonnegativity)
    __common_iterative_parameters_check(
        methods_name,
        subsets_number,
        regularisation_type,
        regularisation_parameter,
        regularisation_iterations,
        regularisation_half_precision,
    )
    ###################################

    # This wrapper has no measurement-weight argument, so only LS is supported.
    __check_variable_type(data_fidelity, [str], "data_fidelity", ["LS"], methods_name)
    fidelity = DataFidelity(method=data_fidelity)
    regularisation = _build_regularisation(
        regularisation_type,
        regularisation_parameter,
        regularisation_iterations,
        regularisation_half_precision,
        regularisation_options,
    )
    scan, grid, detector_pad = _build_reconstruction_geometry(
        data.shape, angles, center, detector_pad, recon_size
    )
    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=detector_pad,
        device_index=gpu_id,
        backend="astra",
        num_subsets=subsets_number,
    ) as reconstructor:
        lc = reconstructor.powermethod()
        # The old API passed regul_param directly to every proximal step.
        # The new API divides objective strength by num_subsets * lc internally.
        regularisation = replace(
            regularisation,
            strength=regularisation_parameter * reconstructor.num_subsets * lc,
        )
        reconstruction = reconstructor.FISTA(
            data,
            input_axes=input_data_axis_labels,
            iterations=iterations,
            lipschitz_constant=lc,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
            fidelity=fidelity,
            regularisation=regularisation,
        )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% ADMM reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def ADMM3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = 0.95,
    iterations: int = 3,
    data_fidelity: str = "LS",
    initialisation: Literal["FBP", "CGLS", "SIRT", None] = None,
    rho: float = 1.0,
    relaxation: float = 1.7,
    regularisation_type: Literal[
        "ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV", "NDF"
    ] = "PD_TV",
    regularisation_parameter: float = 0.0025,
    regularisation_iterations: int = 40,
    regularisation_half_precision: bool = True,
    nonnegativity: bool = False,
    gpu_id: int = 0,
    *,
    data_solver: Literal["linearised", "cg"] = "cg",
    cg_iterations: int = 5,
    cg_tolerance: float = 1e-4,
    lipschitz_constant: Optional[float] = None,
    weights: Optional[Union[np.ndarray, cp.ndarray]] = None,
    beta_swls: float = 0.1,
    regularisation_options: Optional[dict] = None,
) -> cp.ndarray:
    """Full-data GPU ADMM reconstruction. See :ref:`method_ADMM3d_tomobar`.

    Parameters
    ----------
    data : cp.ndarray
        Projections in (angles, detector rows, detector columns) order; float32
        or uint16. Transmission methods expect preprocessed line integrals.
    angles : np.ndarray
        NumPy projection angles in radians, one per input view.
    center : float, optional
        Horizontal CoR in pixels before padding; None uses detector width // 2.
    detector_pad : bool, int
        Edge padding per detector side: False disables, True estimates,
        an integer specifies pixels. Does not itself enlarge the output.
    recon_size : int, optional
        Slice width/height; None uses unpadded detector width. With a warm start
        and nonzero padding, padded detector width overrides this value.
    recon_mask_radius : float, optional
        Positive circular-mask parameter: 1 is the inscribed circle, smaller
        values mask more, None disables. Output shape is unchanged.
    iterations : int
        Number of full-data ADMM iterations.
    data_fidelity : str
        "LS", "PWLS" or "SWLS"; weighted models require weights.
    initialisation : str, optional
        None starts from zeros; FBP uses SINC cutoff 0.35; CGLS/SIRT use 15/150
        nonnegative iterations. Warm starts with padding change output size.
    rho : float
        Positive ADMM penalty; proximal strength is regularisation_parameter / rho.
    relaxation : float
        ADMM relaxation, strictly between 0 and 2.
    regularisation_type : str
        "ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV" or "NDF".
        See :ref:`iterative_regularisation`.
    regularisation_parameter : float
        Objective regularisation strength; zero disables regularisation.
    regularisation_iterations : int
        Inner regulariser iterations per proximal update.
    regularisation_half_precision : bool
        Use float16 auxiliary fields; output stays float32. NDF requires False.
    nonnegativity : bool
        Constrain reconstructed values to be nonnegative.
    gpu_id : int
        CUDA device index. CuPy input must reside on this device.
    data_solver : str
        "cg" approximately solves the quadratic subproblem; "linearised" uses
        one full-data gradient step.
    cg_iterations : int
        Positive maximum CG steps per outer iteration; CG only.
    cg_tolerance : float
        Relative CG residual tolerance in [0, 1); zero disables relative stopping.
    lipschitz_constant : float, optional
        Positive full-data gradient bound for linearised updates; None estimates
        it with the power method. Unused by CG.
    weights : numpy.ndarray or cupy.ndarray, optional
        Finite nonnegative weights, same shape/order as data, with a positive
        value. Required for PWLS/SWLS, rejected for LS; a copy is normalised to max 1.
    beta_swls : float
        Nonnegative SWLS stripe penalty; larger values approach PWLS.
    regularisation_options : dict, optional
        Native method-specific settings; None uses defaults.
        See :ref:`iterative_regularisation` for supported keys and examples.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 reconstruction, shape (recon_size, detector rows,
        recon_size). See recon_size for sizing rules.
    """
    ### Data and parameters checks ###
    methods_name = "ADMM3d_tomobar"
    __common_data_parameters_check(
        data,
        angles,
        methods_name,
        center,
        detector_pad,
        recon_size,
        recon_mask_radius,
        gpu_id,
    )
    __common_iterative_basic_parameters_check(methods_name, iterations, nonnegativity)
    __common_iterative_parameters_check(
        methods_name,
        1,
        regularisation_type,
        regularisation_parameter,
        regularisation_iterations,
        regularisation_half_precision,
    )
    ###################################
    __check_variable_type(
        initialisation,
        [str, type(None)],
        "initialisation",
        ["FBP", "CGLS", "SIRT"],
        methods_name,
    )
    __check_variable_type(rho, [float, int], "rho", [], methods_name)
    __check_variable_type(relaxation, [float, int], "relaxation", [], methods_name)
    __check_variable_type(
        data_solver, [str], "data_solver", ["linearised", "cg"], methods_name
    )
    __check_variable_type(cg_iterations, [int], "cg_iterations", [], methods_name)
    __check_variable_type(cg_tolerance, [float, int], "cg_tolerance", [], methods_name)
    __check_variable_type(
        lipschitz_constant,
        [float, int, type(None)],
        "lipschitz_constant",
        [],
        methods_name,
    )
    __check_variable_type(
        data_fidelity, [str], "data_fidelity", ["LS", "PWLS", "SWLS"], methods_name
    )
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    if not np.isfinite(relaxation) or not 0 < relaxation < 2:
        raise ValueError("relaxation must be between 0 and 2")
    if cg_iterations < 1:
        raise ValueError("cg_iterations must be positive")
    if not np.isfinite(cg_tolerance) or not 0 <= cg_tolerance < 1:
        raise ValueError("cg_tolerance must be finite and in [0, 1)")
    if lipschitz_constant is not None and (
        not np.isfinite(lipschitz_constant) or lipschitz_constant <= 0
    ):
        raise ValueError("lipschitz_constant must be finite and positive")
    fidelity = DataFidelity(method=data_fidelity, beta_swls=beta_swls)
    if data_fidelity == "LS":
        if weights is not None:
            raise ValueError("weights are only supported for PWLS and SWLS")
    elif weights is None:
        raise ValueError("weights are required for PWLS and SWLS")
    if weights is not None:
        if not isinstance(weights, (np.ndarray, cp.ndarray)):
            raise TypeError("weights must be a NumPy or CuPy array")
        if weights.shape != data.shape:
            raise ValueError("weights must match the data shape and axis order")
    # ADMM applies strength / rho internally, as the previous wrapper did.
    regularisation = _build_regularisation(
        regularisation_type,
        regularisation_parameter,
        regularisation_iterations,
        regularisation_half_precision,
        regularisation_options,
    )
    detector_width = data.shape[2]
    detector_pad = _resolve_detector_padding(detector_width, detector_pad)
    # ADMM owns this compatibility policy; it is not part of common geometry.
    if initialisation is not None and detector_pad > 0:
        recon_size = detector_width + 2 * detector_pad

    scan, grid, detector_pad = _build_reconstruction_geometry(
        data.shape, angles, center, detector_pad, recon_size
    )
    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=detector_pad,
        device_index=gpu_id,
        backend="astra",
        num_subsets=1,
    ) as reconstructor:
        initialisation_vol = None
        if initialisation == "FBP":
            with DirectReconstructor(
                projection_geometry=scan,
                volume_geometry=grid,
                detector_padding=detector_pad,
                device_index=gpu_id,
            ) as direct:
                initialisation_vol = direct.FBP(
                    data,
                    input_axes=input_data_axis_labels,
                    filter_type="SINC",
                    cutoff_freq=0.35,
                    recon_mask_radius=recon_mask_radius,
                )
        elif initialisation in ("CGLS", "SIRT"):
            method = getattr(reconstructor, initialisation)
            initialisation_vol = method(
                data,
                input_axes=input_data_axis_labels,
                iterations=15 if initialisation == "CGLS" else 150,
                nonnegativity=True,
                recon_mask_radius=recon_mask_radius,
            )
        reconstruction = reconstructor.ADMM(
            data,
            input_axes=input_data_axis_labels,
            initial=initialisation_vol,
            iterations=iterations,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
            rho=rho,
            relaxation=relaxation,
            data_solver=data_solver,
            cg_iterations=cg_iterations,
            cg_tolerance=cg_tolerance,
            lipschitz_constant=lipschitz_constant,
            weights=weights,
            diagnostics=Diagnostics(residual_interval=None),
            fidelity=fidelity,
            regularisation=regularisation,
        )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


## %%%%%%%%%%%%%%%%%%%%%%% OSEM reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def OSEM3d_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = 0.95,
    iterations: int = 20,
    subsets_number: int = 12,
    regularisation_type: Literal[
        "ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV", "NDF"
    ] = "PD_TV",
    regularisation_parameter: float = 1.0,
    regularisation_iterations: int = 30,
    regularisation_half_precision: bool = True,
    nonnegativity: bool = True,
    gpu_id: int = 0,
    *,
    regularisation_options: Optional[dict] = None,
) -> cp.ndarray:
    """Regularised GPU ordered-subsets EM for nonnegative emission measurements.

    Parameters
    ----------
    data : cp.ndarray
        Nonnegative emission projections, shape (angles, detector rows, detector
        columns); float32 or uint16.
    angles : np.ndarray
        NumPy projection angles in radians, one per input view.
    center : float, optional
        Horizontal CoR in pixels before padding; None uses detector width // 2.
    detector_pad : bool, int
        Edge padding per detector side: False disables, True estimates,
        an integer specifies pixels. Does not itself enlarge the output.
    recon_size : int, optional
        Slice width and height in pixels; None uses unpadded detector width.
    recon_mask_radius : float, optional
        Positive circular-mask parameter: 1 is the inscribed circle, smaller
        values mask more, None disables. Output shape is unchanged.
    iterations : int
        Outer iterations; each visits all interleaved subsets.
    subsets_number : int
        Interleaved subset count; 1 gives MLEM.
    regularisation_type : str
        "ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV" or "NDF".
        See :ref:`iterative_regularisation`.
    regularisation_parameter : float
        Smoothing strength per subset proximal update; zero disables it.
    regularisation_iterations : int
        Inner regulariser iterations per proximal update.
    regularisation_half_precision : bool
        Use float16 auxiliary fields; output stays float32. NDF requires False.
    nonnegativity : bool
        Constrain reconstructed values to be nonnegative.
    gpu_id : int
        CUDA device index. CuPy input must reside on this device.
    regularisation_options : dict, optional
        Native method-specific settings; None uses defaults.
        See :ref:`iterative_regularisation` for supported keys and examples.

    Returns
    -------
    cupy.ndarray
        C-contiguous float32 reconstruction, shape (recon_size, detector rows,
        recon_size). See recon_size for sizing rules.
    """
    ### Data and parameters checks ###
    methods_name = "OSEM3d_tomobar"
    __common_data_parameters_check(
        data,
        angles,
        methods_name,
        center,
        detector_pad,
        recon_size,
        recon_mask_radius,
        gpu_id,
    )
    __common_iterative_basic_parameters_check(methods_name, iterations, nonnegativity)
    __common_iterative_parameters_check(
        methods_name,
        subsets_number,
        regularisation_type,
        regularisation_parameter,
        regularisation_iterations,
        regularisation_half_precision,
    )
    ###################################

    regularisation = _build_regularisation(
        regularisation_type,
        regularisation_parameter,
        regularisation_iterations,
        regularisation_half_precision,
        regularisation_options,
    )
    scan, grid, detector_pad = _build_reconstruction_geometry(
        data.shape, angles, center, detector_pad, recon_size
    )
    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=detector_pad,
        device_index=gpu_id,
        backend="astra",
        num_subsets=subsets_number,
    ) as reconstructor:
        reconstruction = reconstructor.OSEM(
            data,
            input_axes=input_data_axis_labels,
            iterations=iterations,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
            regularisation=regularisation,
        )
    cp._default_memory_pool.free_all_blocks()
    return cp.require(cp.swapaxes(reconstruction, 0, 1), requirements="C")


def iterative_alignment(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    recon_mask_radius: Optional[float] = 0.95,
    gpu_id: int = 0,
    *,
    iterations: int = 20,
    reconstruction_method: Literal[
        "Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"
    ] = "SIRT",
    reconstruction_iterations: int = 1,
    final_reconstruction_iterations: Optional[int] = None,
    subsets_number: int = 1,
    initial: Optional[Union[np.ndarray, cp.ndarray]] = None,
    detector_shifts: Optional[np.ndarray] = None,
    solver_options: Optional[dict] = None,
    regularisation: Optional["Regularisation"] = None,
    upsample_factor: int = 10,
    normalization: Optional[Literal["phase"]] = None,
    registration_backend: Literal["httomolibgpu", "skimage"] = "httomolibgpu",
    taper: bool = True,
    relaxation: float = 1.0,
    max_shift: Optional[float] = None,
    tolerance: float = 0.05,
    reference_view: Optional[int] = None,
    return_diagnostics: bool = False,
) -> Union[Tuple[cp.ndarray, np.ndarray], Tuple[cp.ndarray, np.ndarray, dict]]:
    """Iterative reconstruction/reprojection alignment. See :ref:`method_iterative_alignment`.

    Parameters
    ----------
    data : cupy.ndarray
        Finite CuPy projections, shape (angles, detector rows, detector columns);
        float32 or uint16, with at least three rows/columns. Input is unchanged.
    angles : numpy.ndarray
        NumPy projection angles in radians, one per input view.
    center : float, optional
        Fixed horizontal CoR in pixels; None uses detector width // 2. Not optimised.
    detector_pad : bool or int, optional
        Edge padding per detector side: False disables, True estimates,
        an integer specifies pixels. Does not itself enlarge the output.
    recon_size : int, optional
        Slice width and height in pixels; None uses unpadded detector width.
    recon_mask_radius : float, optional
        Positive output mask parameter; 1 is inscribed, None disables.
        Registration uses full-grid images before masking.
    gpu_id : int, optional
        CUDA device index. CuPy input must reside on this device.
    iterations : int, optional
        Positive maximum reconstruction/registration cycles.
    reconstruction_method : str, optional
        Inner solver: Landweber, SIRT, CGLS, FISTA, ADMM or OSEM.
    reconstruction_iterations : int, optional
        Positive inner iterations per alignment cycle; subset solvers visit
        all subsets per iteration.
    final_reconstruction_iterations : int, optional
        Positive iterations on final geometry; None uses reconstruction_iterations.
    subsets_number : int, optional
        Subset count; Landweber/SIRT/CGLS require 1. ADMM uses all views.
    initial : numpy.ndarray or cupy.ndarray, optional
        NumPy/CuPy warm start in (Y, detector row, X) order. None uses solver
        defaults: ones for SIRT/OSEM, zeros otherwise.
    detector_shifts : numpy.ndarray, optional
        Initial NumPy offsets, shape (views, 2), horizontal then vertical pixels.
        Geometry offsets have the opposite sign to measured image motion.
    solver_options : dict, optional
        Native inner-solver options; nonnegativity defaults to True. Weights
        use data axis order. See :ref:`method_iterative_alignment`.
    regularisation : tomobar.options.Regularisation, optional
        Native ToMoBAR Regularisation object; alternatively set in solver_options,
        but not both. Uses native objective-strength scaling.
    upsample_factor : int, optional
        Positive registration upsampling; resolution is about 1 / upsample_factor pixels.
    normalization : {None, "phase"}, optional
        None: unnormalised correlation; "phase": phase-only correlation.
    registration_backend : {"httomolibgpu", "skimage"}, optional
        "httomolibgpu": GPU registration; "skimage": CPU with per-view transfers.
    taper : bool, optional
        Apply a cosine edge taper for registration.
    relaxation : float, optional
        Shift-update damping in (0, 1]; distinct from inner ADMM relaxation.
    max_shift : float, optional
        Positive per-cycle component bound in pixels; None applies no bound.
    tolerance : float, optional
        Nonnegative RMS update stopping threshold in pixels; not an accuracy guarantee.
    reference_view : int, optional
        Fix this view's shifts; None preserves mean shifts across usable views.
    return_diagnostics : bool, optional
        Also return a diagnostics dictionary when True; otherwise return
        (volume, shifts). See :ref:`method_iterative_alignment`.

    Returns
    -------
    volume : cupy.ndarray
        C-contiguous float32 reconstruction in (Y, detector row, X) order.
    shifts : numpy.ndarray
        Absolute detector offsets, shape (views, 2), horizontal then vertical
        pixels; excludes scalar CoR offset.
    diagnostics : dict, optional
        Only when return_diagnostics=True: shift_history, convergence, residuals,
        valid_views, converged, geometry. See :ref:`method_iterative_alignment`.
    """
    __common_data_parameters_check(
        data,
        angles,
        "iterative_alignment",
        center,
        detector_pad,
        recon_size,
        recon_mask_radius,
        gpu_id,
    )
    __check_variable_type(
        return_diagnostics, [bool], "return_diagnostics", [], "iterative_alignment"
    )
    if not isinstance(data, cp.ndarray):
        raise TypeError("data must be a CuPy array")
    scan, grid, padding = _build_reconstruction_geometry(
        data.shape, angles, center, detector_pad, recon_size
    )
    if detector_shifts is not None:
        scan = replace(scan, detector_shifts=detector_shifts)
    if initial is not None:
        if not isinstance(initial, (np.ndarray, cp.ndarray)):
            raise TypeError("initial must be a NumPy or CuPy array")
        expected_shape = (grid.shape[1], grid.shape[0], grid.shape[2])
        if initial.shape != expected_shape:
            raise ValueError(
                f"initial must have shape {expected_shape} in (Y, detector row, X) order"
            )
        initial = initial.swapaxes(0, 1)
    with cp.cuda.Device(gpu_id):
        with IterativeReconstructor(
            projection_geometry=scan,
            volume_geometry=grid,
            detector_padding=padding,
            device_index=gpu_id,
            backend="astra",
            num_subsets=subsets_number,
        ) as reconstructor:
            result = reconstructor.align(
                data,
                input_axes=input_data_axis_labels,
                iterations=iterations,
                reconstruction_method=reconstruction_method,
                reconstruction_iterations=reconstruction_iterations,
                final_reconstruction_iterations=final_reconstruction_iterations,
                initial=initial,
                solver_options=solver_options,
                regularisation=regularisation,
                upsample_factor=upsample_factor,
                normalization=normalization,
                registration_backend=registration_backend,
                taper=taper,
                relaxation=relaxation,
                max_shift=max_shift,
                tolerance=tolerance,
                reference_view=reference_view,
                recon_mask_radius=recon_mask_radius,
            )
        image = cp.require(cp.swapaxes(result.image, 0, 1), requirements="C")
    if return_diagnostics:
        return (
            image,
            result.shifts,
            {
                "shift_history": result.shift_history,
                "convergence": result.convergence,
                "residuals": result.residuals,
                "valid_views": result.valid_views,
                "converged": result.converged,
                "geometry": result.geometry,
            },
        )
    return image, result.shifts


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
def _resolve_detector_padding(
    detector_width: int, detector_pad: Union[bool, int]
) -> int:
    """Resolve automatic padding while retaining an explicit pixel count."""
    if detector_pad is True:
        return __estimate_detectorHoriz_padding(detector_width)
    if detector_pad is False:
        return 0
    return detector_pad


def _build_reconstruction_geometry(
    data_shape: Tuple[int, int, int],
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    *,
    dimensions: Literal[2, 3] = 3,
) -> tuple:
    """Build acquisition and volume geometry without allocating GPU resources.

    Input shape is always (angles, detector rows, detector columns). Preserve
    the wrappers' angle sign, pixel-centre offset and default integer centre.
    Algorithms choose the working reconstruction size before calling this helper;
    solver settings, output cropping and resource lifetime remain with callers.
    """
    detector_rows, detector_width = data_shape[1:]
    if center is None:
        center = detector_width // 2
    if recon_size is None:
        recon_size = detector_width
    padding = _resolve_detector_padding(detector_width, detector_pad)
    geometry = dict(
        angles=-angles,
        centre_of_rotation=detector_width / 2 - center - 0.5,
    )
    if dimensions == 3:
        scan = Parallel3DGeometry.tomography(
            detector_shape=(detector_rows, detector_width), **geometry
        )
        grid = Volume3DGeometry(shape=(detector_rows, recon_size, recon_size))
    elif dimensions == 2:
        scan = Parallel2DGeometry(detector_shape=(detector_width,), **geometry)
        grid = Volume2DGeometry(shape=(recon_size, recon_size))
    else:
        raise ValueError("dimensions must be 2 or 3")
    return scan, grid, padding


def _instantiate_direct_recon2d_class(
    data: np.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    detector_pad: Union[bool, int] = False,
    recon_size: Optional[int] = None,
    gpu_id: int = 0,
) -> Type:
    """Create a geometry-based 2D reconstructor; the caller must close it."""
    scan, grid, detector_pad = _build_reconstruction_geometry(
        data.shape, angles, center, detector_pad, recon_size, dimensions=2
    )
    return DirectReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=detector_pad,
        device_index=gpu_id,
    )


def __estimate_detectorHoriz_padding(detX_size) -> int:
    det_half = detX_size // 2
    padded_value_exact = int(np.sqrt(2 * (det_half**2))) - det_half
    padded_add_margin = padded_value_exact // 2
    return padded_value_exact + padded_add_margin


def __common_data_parameters_check(
    data,
    angles,
    methods_name,
    center,
    detector_pad,
    recon_size,
    recon_mask_radius,
    gpu_id,
    mem_stack=None,
):
    ### Data and parameters checks ###
    if mem_stack is None:
        cupywrapper.require_gpu(methods_name, device_index=gpu_id)
        __check_if_data_3D_array(data, methods_name)
        __check_if_data_correct_type(
            data, accepted_type=["float32", "uint16"], methods_name=methods_name
        )
        if len(angles) != data.shape[0]:
            err_str = f"The angles length {len(angles)} is not equal to the input data angles dimension {data.shape[0]} for method '{methods_name}'."
            raise ValueError(err_str)
    __check_variable_type(center, [float, int, type(None)], "center", [], methods_name)
    __check_variable_type(detector_pad, [bool, int], "detector_pad", [], methods_name)
    __check_variable_type(recon_size, [int, type(None)], "recon_size", [], methods_name)
    __check_variable_type(
        recon_mask_radius,
        [float, int, type(None)],
        "recon_mask_radius",
        [],
        methods_name,
    )
    __check_variable_type(gpu_id, [int], "gpu_id", [], methods_name)
    ###################################


def __common_iterative_basic_parameters_check(methods_name, iterations, nonnegativity):
    __check_variable_type(iterations, [int], "iterations", [], methods_name)
    __check_variable_type(nonnegativity, [bool], "nonnegativity", [], methods_name)


def _build_regularisation(method, strength, iterations, half_precision, options):
    """Validate native inner controls without overriding wrapper strength scaling."""
    if options is None:
        options = {}
    elif not isinstance(options, dict):
        raise TypeError("regularisation_options must be a dictionary or None")
    else:
        options = options.copy()
    allowed = {
        "time_step",
        "lipschitz_constant",
        "tv_type",
        "alpha1",
        "alpha0",
        "sigma",
        "penalty_type",
    }
    unsupported = options.keys() - allowed
    if unsupported:
        raise ValueError(
            "Unsupported regularisation_options keys: "
            + ", ".join(sorted(map(str, unsupported)))
            + ". Use the dedicated wrapper parameters for method, strength, iterations and half precision."
        )
    return Regularisation(
        method=method,
        strength=strength,
        iterations=iterations,
        half_precision=half_precision,
        **options,
    )


def __common_iterative_parameters_check(
    methods_name,
    subsets_number,
    regularisation_type,
    regularisation_parameter,
    regularisation_iterations,
    regularisation_half_precision,
):
    __check_variable_type(subsets_number, [int], "subsets_number", [], methods_name)
    __check_variable_type(
        regularisation_type,
        [str],
        "regularisation_type",
        ["ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV", "NDF"],
        methods_name,
    )
    __check_variable_type(
        regularisation_parameter,
        [float, int],
        "regularisation_parameter",
        [],
        methods_name,
    )
    __check_variable_type(
        regularisation_iterations, [int], "regularisation_iterations", [], methods_name
    )
    __check_variable_type(
        regularisation_half_precision,
        [bool],
        "regularisation_half_precision",
        [],
        methods_name,
    )
