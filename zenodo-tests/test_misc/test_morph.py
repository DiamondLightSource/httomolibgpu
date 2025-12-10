import cupy as cp
import numpy as np
from httomolibgpu.misc.morph import sino_360_to_180
from httomolibgpu.prep.normalize import dark_flat_field_correction
from conftest import force_clean_gpu_memory


def test_sino_360_to_180_i13_dataset1(i13_dataset1):
    projdata = i13_dataset1[0]
    flats = i13_dataset1[2]
    darks = i13_dataset1[3]
    del i13_dataset1

    data_normalised = dark_flat_field_correction(projdata, flats, darks, cutoff=10)

    del flats, darks, projdata
    force_clean_gpu_memory()

    stiched_data_180degrees = sino_360_to_180(
        data_normalised, overlap=473.822265625, side="right"
    )
    stiched_data_180degrees = cp.asnumpy(stiched_data_180degrees)

    assert stiched_data_180degrees.shape == (3000, 10, 4646)
    assert stiched_data_180degrees.dtype == np.float32
    assert stiched_data_180degrees.flags.c_contiguous


def test_sino_360_to_180_i13_dataset3(i13_dataset3):
    projdata = i13_dataset3[0]
    flats = i13_dataset3[2]
    darks = i13_dataset3[3]
    del i13_dataset3

    data_normalised = dark_flat_field_correction(projdata, flats, darks, cutoff=10)
    del flats, darks, projdata
    force_clean_gpu_memory()

    stiched_data_180degrees = sino_360_to_180(
        data_normalised, overlap=438.173828, side="left"
    )
    stiched_data_180degrees = cp.asnumpy(stiched_data_180degrees)

    assert stiched_data_180degrees.shape == (3000, 3, 4682)
    assert stiched_data_180degrees.dtype == np.float32
    assert stiched_data_180degrees.flags.c_contiguous


def test_sino_360_to_180_i12_dataset5(i12_dataset5):
    projdata = i12_dataset5[0]
    flats = i12_dataset5[2]
    darks = i12_dataset5[3]
    del i12_dataset5

    data_normalised = dark_flat_field_correction(projdata, flats, darks, cutoff=10)
    del flats, darks, projdata
    force_clean_gpu_memory()

    stiched_data_180degrees = sino_360_to_180(
        data_normalised, overlap=186.66, side="left"
    )
    stiched_data_180degrees = cp.asnumpy(stiched_data_180degrees)

    assert stiched_data_180degrees.shape == (1800, 15, 4933)
    assert stiched_data_180degrees.dtype == np.float32
    assert stiched_data_180degrees.flags.c_contiguous
