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

from typing import Optional, Tuple

import cupy as cp
import cupyx
import numpy as np
import nvtx
from httomolib.decorator import method_sino

from httomolib.cuda_kernels import load_cuda_module

__all__ = [
    "reconstruct_tomobar",
    "reconstruct_tomopy_astra",
]


def _calc_max_slices_reconstruct_tomobar(
    other_dims: Tuple[int, int], dtype: np.dtype, available_memory: int, **kwargs
) -> Tuple[int, np.dtype]:
    # we first run filtersync, and calc the memory for that - how com it's 
    DetectorsLengthH = other_dims[1]
    in_slice_size = np.prod(other_dims) * dtype.itemsize
    filter_size = (DetectorsLengthH//2+1) * np.float32().itemsize
    freq_slice = other_dims[0] * (DetectorsLengthH//2+1) * np.complex64().itemsize
    fftplan_size = freq_slice * 2
    swapaxis_size = in_slice_size
    # can only guess what the astra toolbox uses re memory...
    astra_size = in_slice_size * 2

    available_memory -= filter_size
    return available_memory // (in_slice_size + freq_slice + fftplan_size + swapaxis_size + astra_size), np.float32()


## %%%%%%%%%%%%%%%%%%%%%%% ToMoBAR reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
@method_sino(_calc_max_slices_reconstruct_tomobar)
@nvtx.annotate()
def reconstruct_tomobar(
    data: cp.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    objsize: Optional[int] = None,
    gpu_id: int = 0,
) -> cp.ndarray:
    """
    Perform reconstruction using ToMoBAR wrappers around ASTRA toolbox.
    This is a 3D recon using 3D astra geometry routines.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    objsize : int, optional
        The size in pixels of the reconstructed object.
    gpu_id : int, optional
        A GPU device index to perform operation on.

    Returns
    -------
    cp.ndarray
        The reconstructed volume as a CuPy array.
    """
    from tomobar.supp.astraOP import AstraTools3D

    if center is None:
        center = data.shape[2] // 2  # making a crude guess
    if objsize is None:
        objsize = data.shape[2]

    # Perform filtering of the data on the GPU and then pass a pointer to CuPy array to do backprojection.
    # initiate a 3D ASTRA class object
    Atools = AstraTools3D(
        DetectorsDimH=data.shape[2],  # DetectorsDimH # detector dimension (horizontal)
        # DetectorsDimV: detector dimension (vertical) for 3D case only
        DetectorsDimV=data.shape[1],
        AnglesVec=-angles,  # the vector of angles in radians
        # The center of rotation combined with the shift offsets
        CenterRotOffset=data.shape[2] / 2 - center - 0.5,
        ObjSize=objsize,  # a scalar to define the reconstructed object dimensions
        OS_number=1,  # OS recon disabled
        device_projector="gpu",
        GPUdevice_index=gpu_id,
    )
    # filter the data on the GPU and keep the result there
    data = _filtersinc3D_cupy(data)

    # the Astra toolbox requires C-contiguous arrays, and swap axes seem to return a sliced view which
    # is neither C nor F contiguous.
    # So we have to make it C-contiguous first
    data = cp.ascontiguousarray(cp.swapaxes(data, 0, 1))

    # backproject the filtered data while keeping data on the GPU
    reconstruction = Atools.backprojCuPy(data)
    cp._default_memory_pool.free_all_blocks()

    return reconstruction


@nvtx.annotate()
def _filtersinc3D_cupy(projection3D):
    """applies a filter to 3D projection data

    Args:
        projection3D (ndarray): projection data must be a CuPy array.

    Returns:
        ndarray: a CuPy array of filtered projection data.
    """
    (projectionsNum, DetectorsLengthV, DetectorsLengthH) = cp.shape(projection3D)

    # prepearing a ramp-like filter to apply to every projection
    module = load_cuda_module("generate_filtersync")
    filter_prep = module.get_function("generate_filtersinc")

    # Use real FFT to save space and time
    proj_f = cupyx.scipy.fft.rfft(projection3D, axis=-1, norm="backward", overwrite_x=True)

    # generating the filter here so we can schedule/allocate while FFT is keeping the GPU busy
    a = 1.1
    f = cp.empty((1, 1, DetectorsLengthH//2+1), dtype=np.float32)
    bx = 256
    # because FFT is linear, we can apply the FFT scaling + multiplier in the filter
    multiplier = 1.0 / projectionsNum / DetectorsLengthH
    filter_prep(
        grid=(1, 1, 1),
        block=(bx, 1, 1),
        args=(cp.float32(a), f, np.int32(DetectorsLengthH), np.float32(multiplier)),
        shared_mem=bx * 4,
    )

    # actual filtering
    proj_f *= f
    
    return cupyx.scipy.fft.irfft(proj_f, projection3D.shape[2], axis=-1, norm="forward", overwrite_x=True)
    


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##

def _calc_max_slices_reconstruct_tompy_astra(
    other_dims: Tuple[int, int], dtype: np.dtype, available_memory: int, **kwargs
) -> Tuple[int, np.dtype]:
    algorithm = kwargs['algorithm']
    # we don't know how Astra uses the memory - we can only guess
    if algorithm  == 'FBP_CUDA':
        slice_mem = np.prod(other_dims) * dtype.itemsize * 4
        return available_memory // slice_mem, dtype

    # no GPU used, we're not really limiting this
    return available_memory // np.prod(other_dims) // dtype.itemsize, dtype


## %%%%%%%%%%%%%%%%%%%%%%% Tomopy/ASTRA reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%  ##
@method_sino(_calc_max_slices_reconstruct_tompy_astra)
@nvtx.annotate()
def reconstruct_tomopy_astra(
    data: np.ndarray,
    angles: np.ndarray,
    center: Optional[float] = None,
    algorithm: str = "FBP_CUDA",
    iterations: int = 1,
    proj_type: str = "cuda",
    gpu_id: int = 0,
    ncore: int = 1,
) -> np.ndarray:
    """
    Perform reconstruction using tomopy with wrappers around ASTRA toolbox.
    This is a 3D recon using 2D (slice-by-slice) astra geometry routines.

    Parameters
    ----------
    data : np.ndarray
        Projection data as a numpy array.
    angles : np.ndarray
        An array of angles given in radians.
    center : float, optional
        The center of rotation (CoR).
    algorithm : str, optional
        The name of the reconstruction method, see available ASTRA methods.
    iterations : int, optional
        The number of iterations if the iterative algorithm is chosen.
    proj_type : str, optional
        Define projector type, e.g., "cuda" for "FBP_CUDA" algorithm
        or "linear" for "FBP" (CPU) algorithm, see more available ASTRA projectors.
    gpu_id : int, optional
        A GPU device index to perform operation on.
    ncore : int, optional
        Number of cores that will be assigned to jobs.

    Returns
    -------
    np.ndarray
        The reconstructed volume.
    """
    from tomopy import astra, recon

    reconstruction = recon(
        data,
        theta=angles,
        center=center,
        algorithm=astra,
        options={
            "method": algorithm,
            "proj_type": proj_type,
            "gpu_list": [gpu_id],
            "num_iter": iterations,
        },
        ncore=ncore,
    )

    return reconstruction


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
