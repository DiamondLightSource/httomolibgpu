import cupy as cp
import numpy as np
import pytest
from cupy.cuda import nvtx
import time
from math import isclose

from httomolibgpu.prep.normalize import normalize
from httomolibgpu.recon.algorithm import (
    FBP2d_astra,
    FBP3d_tomobar,
    LPRec3d_tomobar,
)
from httomolibgpu.misc.morph import sino_360_to_180
from numpy.testing import assert_allclose
import time
import pytest
from cupy.cuda import nvtx
from conftest import force_clean_gpu_memory


def test_reconstruct_FBP2d_astra_i12_dataset1(i12_dataset1):
    force_clean_gpu_memory()
    projdata = i12_dataset1[0]
    angles = i12_dataset1[1]
    flats = i12_dataset1[2]
    darks = i12_dataset1[3]
    del i12_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    recon_data = FBP2d_astra(
        cp.asnumpy(data_normalised),
        np.deg2rad(angles),
        center=1253.75,
        filter_type="shepp-logan",
        filter_parameter=None,
        filter_d=2.0,
        recon_mask_radius=0.9,
    )
    assert recon_data.flags.c_contiguous
    assert_allclose(np.sum(recon_data), 84672.84, atol=1e-2)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 50, 2560)


def test_reconstruct_FBP3d_tomobar_i12_dataset1(i12_dataset1):
    force_clean_gpu_memory()
    projdata = i12_dataset1[0]
    angles = i12_dataset1[1]
    flats = i12_dataset1[2]
    darks = i12_dataset1[3]
    del i12_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    recon_data = FBP3d_tomobar(
        data_normalised,
        np.deg2rad(angles),
        center=1253.75,
        filter_freq_cutoff=0.35,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.sum(recon_data), 46569.39, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 50, 2560)


def test_reconstruct_LPRec3d_tomobar_i12_dataset1(i12_dataset1):
    force_clean_gpu_memory()
    projdata = i12_dataset1[0]
    angles = i12_dataset1[1]
    flats = i12_dataset1[2]
    darks = i12_dataset1[3]
    del i12_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    data_normalised_cut = data_normalised[:, 5:8, :]
    del flats, darks, projdata, data_normalised
    force_clean_gpu_memory()

    recon_data = LPRec3d_tomobar(
        data_normalised_cut,
        np.deg2rad(angles),
        center=1253.75,
        filter_type="shepp",
        filter_freq_cutoff=1.0,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert isclose(np.sum(recon_data), 8973.755, abs_tol=10**-3)
    assert pytest.approx(np.max(recon_data), rel=1e-3) == 0.00640
    assert pytest.approx(np.min(recon_data), rel=1e-3) == -0.00617
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 3, 2560)


def test_reconstruct_LPRec_tomobar_i13_dataset1(i13_dataset1):
    force_clean_gpu_memory()
    projdata = i13_dataset1[0]
    angles = i13_dataset1[1]
    flats = i13_dataset1[2]
    darks = i13_dataset1[3]
    del i13_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    stiched_data_180degrees = sino_360_to_180(
        data_normalised[:, 2:3, :], overlap=473.822265625, side="right"
    )
    del data_normalised
    force_clean_gpu_memory()

    # GPU archetictures older than 5.3 wont accept the data larger than
    # (4096, 4096, 4096), while the newer ones can accept (16384 x 16384 x 16384)

    # recon_data = FBP3d_tomobar(
    #     stiched_data_180degrees,
    #     np.deg2rad(angles[0:3000]),
    #     center=2322,
    #     filter_freq_cutoff=0.35,
    # )
    recon_data = LPRec3d_tomobar(
        data=stiched_data_180degrees,
        angles=np.deg2rad(angles[0:3000]),
        center=2322.08,
        filter_type="shepp",
        filter_freq_cutoff=1.0,
    )

    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert isclose(np.sum(recon_data), 1241.859, abs_tol=10**-3)
    assert pytest.approx(np.max(recon_data), rel=1e-3) == 0.00823
    assert pytest.approx(np.min(recon_data), rel=1e-3) == -0.00656
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (4646, 1, 4646)


@pytest.mark.perf
def test_FBP3d_tomobar_performance_i13_dataset2(i13_dataset2):
    force_clean_gpu_memory()
    dev = cp.cuda.Device()
    projdata = i13_dataset2[0]
    angles = i13_dataset2[1]
    flats = i13_dataset2[2]
    darks = i13_dataset2[3]
    del i13_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    # cold run first
    FBP3d_tomobar(
        data_normalised,
        np.deg2rad(angles),
        center=1253.75,
        filter_freq_cutoff=0.35,
    )
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        FBP3d_tomobar(
            data_normalised,
            np.deg2rad(angles),
            center=1286.25,
            filter_freq_cutoff=0.35,
        )
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_reconstruct_LPRec3d_tomobar_i13_dataset2(i13_dataset2):
    force_clean_gpu_memory()
    projdata = i13_dataset2[0]
    angles = i13_dataset2[1]
    flats = i13_dataset2[2]
    darks = i13_dataset2[3]
    del i13_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    recon_data = LPRec3d_tomobar(
        data=data_normalised,
        angles=np.deg2rad(angles),
        center=1286.25,
        filter_type="shepp",
        filter_freq_cutoff=1.0,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()

    assert isclose(np.sum(recon_data), 4095.6257, abs_tol=10**-3)
    assert pytest.approx(np.max(recon_data), rel=1e-3) == 0.0105672
    assert pytest.approx(np.min(recon_data), rel=1e-3) == -0.00839
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 10, 2560)


@pytest.mark.perf
def test_LPRec3d_tomobar_performance_i13_dataset2(i13_dataset2):
    dev = cp.cuda.Device()
    projdata = i13_dataset2[0]
    angles = i13_dataset2[1]
    flats = i13_dataset2[2]
    darks = i13_dataset2[3]
    del i13_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    # cold run first
    LPRec3d_tomobar(
        data=data_normalised,
        angles=np.deg2rad(angles),
        center=1286.25,
    )
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        LPRec3d_tomobar(
            data=data_normalised,
            angles=np.deg2rad(angles),
            center=1286.25,
        )
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_reconstruct_FBP3d_tomobar_i13_dataset3(i13_dataset3):
    force_clean_gpu_memory()
    projdata = i13_dataset3[0]
    angles = i13_dataset3[1]
    flats = i13_dataset3[2]
    darks = i13_dataset3[3]
    del i13_dataset3

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    stiched_data_180degrees = sino_360_to_180(
        data_normalised, overlap=438.173828, side="left"
    )
    force_clean_gpu_memory()

    # GPU archetictures older than 5.3 wont accept the data larger than
    # (4096, 4096, 4096), while the newer ones can accept (16384 x 16384 x 16384)

    recon_data = FBP3d_tomobar(
        stiched_data_180degrees,
        np.deg2rad(angles[0:3000]),
        center=2339,
        filter_freq_cutoff=0.35,
    )

    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()

    assert recon_data.dtype == np.float32
    assert recon_data.shape == (4682, 3, 4682)


def test_reconstruct_FBP3d_tomobar_i12_dataset5(i12_dataset5):
    force_clean_gpu_memory()
    projdata = i12_dataset5[0]
    angles = i12_dataset5[1]
    flats = i12_dataset5[2]
    darks = i12_dataset5[3]
    del i12_dataset5

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    stiched_data_180degrees = sino_360_to_180(
        data_normalised, overlap=186.66, side="left"
    )
    force_clean_gpu_memory()

    recon_data = FBP3d_tomobar(
        stiched_data_180degrees,
        np.deg2rad(angles[0:1800]),
        center=2466,
        filter_freq_cutoff=0.35,
    )

    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()

    assert recon_data.dtype == np.float32
    assert recon_data.shape == (4933, 15, 4933)
