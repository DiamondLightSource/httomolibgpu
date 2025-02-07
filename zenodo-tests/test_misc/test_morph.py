import time
import cupy as cp
import numpy as np
from cupy.cuda import nvtx
import pytest
from numpy.testing import assert_allclose
from httomolibgpu.misc.morph import sino_360_to_180
from httomolibgpu.prep.normalize import normalize
from conftest import force_clean_gpu_memory


def test_sino_360_to_180_i13_dataset1(i13_dataset1):
    projdata = i13_dataset1[0]
    flats = i13_dataset1[2]
    darks = i13_dataset1[3]
    del i13_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    stiched_data_180degrees = sino_360_to_180(
        data_normalised, overlap=473.822265625, rotation="right"
    )
    stiched_data_180degrees = stiched_data_180degrees.get()

    assert_allclose(np.sum(stiched_data_180degrees), 28512826.0, rtol=1e-07, atol=1e-6)
    assert stiched_data_180degrees.shape == (3000, 10, 4646)
    assert stiched_data_180degrees.dtype == np.float32
    assert stiched_data_180degrees.flags.c_contiguous


def test_sino_360_to_180_i13_dataset3(i13_dataset3):
    projdata = i13_dataset3[0]
    flats = i13_dataset3[2]
    darks = i13_dataset3[3]
    del i13_dataset3

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    stiched_data_180degrees = sino_360_to_180(
        data_normalised, overlap=438.173828, rotation="left"
    )
    stiched_data_180degrees = stiched_data_180degrees.get()

    assert_allclose(np.sum(stiched_data_180degrees), 24865750.0, rtol=1e-07, atol=1e-6)
    assert stiched_data_180degrees.shape == (3000, 3, 4682)
    assert stiched_data_180degrees.dtype == np.float32
    assert stiched_data_180degrees.flags.c_contiguous
