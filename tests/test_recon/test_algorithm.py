import cupy as cp
import numpy as np
from httomolibgpu.prep.normalize import normalize as normalize_cupy
from httomolibgpu.recon.algorithm import (
    FBP,
    LPRec,
    SIRT,
    CGLS,
)
from numpy.testing import assert_allclose
import time
import pytest
from cupy.cuda import nvtx


def test_reconstruct_FBP_1(data, flats, darks, ensure_clean_memory):
    recon_data = FBP(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True),
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
        filter_freq_cutoff=1.1,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.000798, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.102106, rtol=1e-05)
    assert_allclose(np.std(recon_data), 0.006293, rtol=1e-07, atol=1e-6)
    assert_allclose(np.median(recon_data), -0.000555, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (160, 128, 160)


def test_reconstruct_FBP_2(data, flats, darks, ensure_clean_memory):
    recon_data = FBP(
        normalize_cupy(data, flats, darks, cutoff=20.5, minus_log=False),
        np.linspace(5.0 * np.pi / 360.0, 180.0 * np.pi / 360.0, data.shape[0]),
        15.5,
        filter_freq_cutoff=1.1,
    )

    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), -0.00015, rtol=1e-07, atol=1e-6)
    assert_allclose(
        np.mean(recon_data, axis=(0, 2)).sum(), -0.019142, rtol=1e-06, atol=1e-5
    )
    assert_allclose(np.std(recon_data), 0.003561, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32


def test_reconstruct_FBP_3(data, flats, darks, ensure_clean_memory):
    recon_data = FBP(
        normalize_cupy(data, flats, darks, cutoff=20.5, minus_log=False),
        np.linspace(5.0 * np.pi / 360.0, 180.0 * np.pi / 360.0, data.shape[0]),
        79,  # center
        1.1,  # filter_freq_cutoff
        210,  # recon_size
        0.9,  # recon_mask_radius
    )

    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), -0.000252, atol=1e-6)
    assert_allclose(
        np.mean(recon_data, axis=(0, 2)).sum(), -0.03229, rtol=1e-06, atol=1e-5
    )
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (210, 128, 210)


def test_reconstruct_LPREC_1(data, flats, darks, ensure_clean_memory):
    recon_data = LPRec(
        data=normalize_cupy(data, flats, darks, cutoff=10, minus_log=True),
        angles=np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        recon_size=130,
        recon_mask_radius=0.95,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.0035118104, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.44951183, rtol=1e-05)
    assert_allclose(np.max(recon_data), 0.058334317, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (130, 128, 130)


def test_reconstruct_SIRT(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    recon_data = SIRT(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True),
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
        recon_size=objrecon_size,
        iterations=10,
        nonnegativity=True,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.0018447536, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.23612846, rtol=1e-05)
    assert recon_data.dtype == np.float32


def test_reconstruct_CGLS(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    recon_data = CGLS(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True),
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
        recon_size=objrecon_size,
        iterations=5,
        nonnegativity=True,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.0021818762, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.279187, rtol=1e-03)
    assert recon_data.dtype == np.float32


@pytest.mark.perf
def test_FBP_performance(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)
    angles = np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0])
    cor = 79.5
    filter_freq_cutoff = 1.1

    # cold run first
    FBP(data, angles, cor, filter_freq_cutoff)
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        FBP(data, angles, cor)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
