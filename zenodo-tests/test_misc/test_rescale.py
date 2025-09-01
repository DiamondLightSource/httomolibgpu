import cupy as cp
import numpy as np
from numpy.testing import assert_allclose

from httomolibgpu.misc.rescale import rescale_to_int
from httomolib.misc.rescale import rescale_to_int as rescale_to_int_cpu

from httomolibgpu.prep.normalize import normalize
from conftest import force_clean_gpu_memory


def test_rescale_to_int_cpu_vs_gpu_16bit_i12_dataset1(i12_dataset1: tuple):
    projdata = i12_dataset1[0]
    angles = i12_dataset1[1]
    flats = i12_dataset1[2]
    darks = i12_dataset1[3]
    del i12_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata, angles
    force_clean_gpu_memory()

    rescaled_data_gpu = rescale_to_int(
        data_normalised, perc_range_min=10, perc_range_max=90, bits=16, glob_stats=None
    )
    rescaled_data_cpu = rescale_to_int_cpu(
        cp.asnumpy(data_normalised, order="C"),
        perc_range_min=10,
        perc_range_max=90,
        bits=16,
        glob_stats=None,
    )

    rescaled_data_gpu_np = cp.asnumpy(rescaled_data_gpu, order="C")
    residual_data = np.sum(rescaled_data_cpu - rescaled_data_gpu_np)

    assert_allclose(residual_data, 0.0, rtol=1e-07, atol=1e-6)

    assert rescaled_data_gpu_np.shape == (1801, 50, 2560)
    assert rescaled_data_gpu_np.dtype == np.uint16
    assert rescaled_data_gpu_np.flags.c_contiguous
    assert rescaled_data_cpu.shape == (1801, 50, 2560)
    assert rescaled_data_cpu.dtype == np.uint16
    assert rescaled_data_cpu.flags.c_contiguous


def test_rescale_to_int_cpu_vs_gpu_8bit_i12_dataset1(i12_dataset1: tuple):
    projdata = i12_dataset1[0]
    angles = i12_dataset1[1]
    flats = i12_dataset1[2]
    darks = i12_dataset1[3]
    del i12_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata, angles
    force_clean_gpu_memory()

    rescaled_data_gpu = rescale_to_int(
        data_normalised, perc_range_min=10, perc_range_max=90, bits=8, glob_stats=None
    )
    rescaled_data_cpu = rescale_to_int_cpu(
        cp.asnumpy(data_normalised, order="C"),
        perc_range_min=10,
        perc_range_max=90,
        bits=8,
        glob_stats=None,
    )

    rescaled_data_gpu_np = cp.asnumpy(rescaled_data_gpu, order="C")
    residual_data = np.sum(rescaled_data_cpu - rescaled_data_gpu_np)

    assert_allclose(residual_data, 0.0, rtol=1e-07, atol=1e-6)

    assert rescaled_data_gpu_np.shape == (1801, 50, 2560)
    assert rescaled_data_gpu_np.dtype == np.uint8
    assert rescaled_data_gpu_np.flags.c_contiguous
    assert rescaled_data_cpu.shape == (1801, 50, 2560)
    assert rescaled_data_cpu.dtype == np.uint8
    assert rescaled_data_cpu.flags.c_contiguous
