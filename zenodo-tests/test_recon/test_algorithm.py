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
    CGLS3d_tomobar,
    SIRT3d_tomobar,
    FISTA3d_tomobar,
)
from httomolibgpu.misc.morph import sino_360_to_180
from numpy.testing import assert_allclose
import time
import pytest
from cupy.cuda import nvtx
from conftest import force_clean_gpu_memory


def test_reconstruct_FBP2d_astra_i12_dataset1(i12_dataset1: tuple):
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
        detector_pad=0,
        filter_type="shepp-logan",
        filter_parameter=None,
        filter_d=2.0,
        recon_mask_radius=0.9,
    )
    assert recon_data.flags.c_contiguous
    assert_allclose(np.sum(recon_data), 84672.84, atol=1e-2)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 50, 2560)


def test_reconstruct_FBP3d_tomobar_i12_dataset1(i12_dataset1: tuple):
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
        detector_pad=0,
        filter_freq_cutoff=0.35,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.sum(recon_data), 46569.39, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 50, 2560)


def test_reconstruct_FBP3d_tomobar_i12_dataset1_pad(i12_dataset1: tuple):
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
        data_normalised[:, 10:20, :],
        np.deg2rad(angles),
        center=1253.75,
        detector_pad=150,
        filter_freq_cutoff=0.35,
        recon_mask_radius=2.0,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.sum(recon_data), 9864.915, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 10, 2560)


def test_reconstruct_FBP3d_tomobar_i12_dataset1_autopad(i12_dataset1: tuple):
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
        data_normalised[:, 10:15, :],
        np.deg2rad(angles),
        center=1253.75,
        detector_pad=True,
        filter_freq_cutoff=0.35,
        recon_mask_radius=2.0,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.sum(recon_data), 6669.1274, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 5, 2560)


def test_reconstruct_LPRec3d_tomobar_i12_dataset1(i12_dataset1: tuple):
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
        detector_pad=0,
        filter_type="shepp",
        filter_freq_cutoff=1.0,
        recon_mask_radius=2.0,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert isclose(np.sum(recon_data), 9628.818, abs_tol=10**-3)
    assert pytest.approx(np.max(recon_data), rel=1e-3) == 0.006367563270032406
    assert pytest.approx(np.min(recon_data), rel=1e-3) == -0.0062076798
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 3, 2560)


def test_reconstruct_LPRec3d_tomobar_i12_dataset1_autopad(i12_dataset1: tuple):
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
        detector_pad=True,
        filter_type="shepp",
        filter_freq_cutoff=1.0,
        recon_mask_radius=2.0,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert int(np.sum(recon_data)) == 9147
    assert pytest.approx(np.max(recon_data), rel=1e-3) == 0.0063818083
    assert pytest.approx(np.min(recon_data), rel=1e-3) == -0.006200762
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 3, 2560)


def test_reconstruct_LPRec_tomobar_i13_dataset1(i13_dataset1: tuple):
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
        detector_pad=False,
        filter_type="shepp",
        filter_freq_cutoff=1.0,
        recon_mask_radius=2.0,
    )

    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert int(np.sum(recon_data)) == 1149
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (4646, 1, 4646)


@pytest.mark.perf
def test_FBP3d_tomobar_performance_i13_dataset2(i13_dataset2: tuple):
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
        detector_pad=0,
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
            detector_pad=0,
            filter_freq_cutoff=0.35,
        )
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_reconstruct_LPRec3d_tomobar_i13_dataset2(i13_dataset2: tuple):
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
        detector_pad=False,
        filter_type="shepp",
        filter_freq_cutoff=1.0,
        recon_mask_radius=2.0,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert int(np.sum(recon_data)) == 3448
    assert pytest.approx(np.max(recon_data), rel=1e-3) == 0.010543354
    assert pytest.approx(np.min(recon_data), rel=1e-3) == -0.008385599
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 10, 2560)


@pytest.mark.perf
def test_LPRec3d_tomobar_performance_i13_dataset2(i13_dataset2: tuple):
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
        detector_pad=0,
    )
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        LPRec3d_tomobar(
            data=data_normalised,
            angles=np.deg2rad(angles),
            center=1286.25,
            detector_pad=0,
        )
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_reconstruct_FBP3d_tomobar_i13_dataset3(i13_dataset3: tuple):
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
        detector_pad=0,
        filter_freq_cutoff=0.35,
    )

    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()

    assert recon_data.dtype == np.float32
    assert recon_data.shape == (4682, 3, 4682)


def test_reconstruct_FBP3d_tomobar_i12_dataset5(i12_dataset5: tuple):
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
        detector_pad=0,
        filter_freq_cutoff=0.35,
    )

    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()

    assert recon_data.dtype == np.float32
    assert recon_data.shape == (4933, 15, 4933)


def test_reconstruct_LPRec3d_tomobar_k11_dataset2(k11_dataset2: tuple):
    force_clean_gpu_memory()
    projdata = k11_dataset2[0]
    angles = k11_dataset2[1]
    flats = k11_dataset2[2]
    darks = k11_dataset2[3]
    del k11_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    recon_data = LPRec3d_tomobar(
        data=data_normalised,
        angles=np.deg2rad(angles),
        center=1280.25,
        detector_pad=False,
        filter_type="shepp",
        filter_freq_cutoff=1.0,
        recon_mask_radius=2.0,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert isclose(np.sum(recon_data), 10865.341, abs_tol=10**-3)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 25, 2560)


def test_reconstruct_CGLS3d_tomobar_k11_dataset2(k11_dataset2: tuple):
    force_clean_gpu_memory()
    projdata = k11_dataset2[0]
    angles = k11_dataset2[1]
    flats = k11_dataset2[2]
    darks = k11_dataset2[3]
    del k11_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    recon_data = CGLS3d_tomobar(
        data=data_normalised[:, 5:10, :],
        angles=np.deg2rad(angles),
        center=1280.25,
        detector_pad=False,
        recon_mask_radius=2.0,
        iterations=15,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert isclose(np.sum(recon_data), 6392.362, abs_tol=10**-3)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 5, 2560)


def test_reconstruct_SIRT3d_tomobar_k11_dataset2(k11_dataset2: tuple):
    force_clean_gpu_memory()
    projdata = k11_dataset2[0]
    angles = k11_dataset2[1]
    flats = k11_dataset2[2]
    darks = k11_dataset2[3]
    del k11_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    recon_data = SIRT3d_tomobar(
        data=data_normalised[:, 5:10, :],
        angles=np.deg2rad(angles),
        center=1280.25,
        detector_pad=False,
        recon_mask_radius=2.0,
        iterations=15,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert isclose(np.sum(recon_data), 2564.5596, abs_tol=10**-3)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 5, 2560)


def test_reconstruct_FISTA3d_tomobar_autopad_k11_dataset2(k11_dataset2: tuple):
    force_clean_gpu_memory()
    projdata = k11_dataset2[0]
    angles = k11_dataset2[1]
    flats = k11_dataset2[2]
    darks = k11_dataset2[3]
    del k11_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    recon_data = FISTA3d_tomobar(
        data=data_normalised[:, 5:10, :],
        angles=np.deg2rad(angles),
        center=1280.25,
        detector_pad=True,
        recon_mask_radius=2.0,
        iterations=7,
        regularisation_parameter=0.0000005,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert isclose(np.sum(recon_data), 1518.5251, abs_tol=10**-3)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 5, 2560)
