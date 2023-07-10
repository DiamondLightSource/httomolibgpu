import cupy as cp
import numpy as np
from httomolibgpu.prep.normalize import normalize as normalize_cupy
from httomolibgpu.recon.algorithm import (
    FBP,
    SIRT,
    CGLS,
)
from numpy.testing import assert_allclose
import time
import pytest
from cupy.cuda import nvtx

from tests import MaxMemoryHook


@cp.testing.gpu
def test_reconstruct_FBP_1(data, flats, darks, ensure_clean_memory):
    recon_data = FBP(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True),
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
    )
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.000798, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(1, 2)).sum(), 0.102106, rtol=1e-05)
    assert_allclose(np.std(recon_data), 0.006293, rtol=1e-07, atol=1e-6)
    assert_allclose(np.median(recon_data), -0.000555, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32


@cp.testing.gpu
def test_reconstruct_FBP_2(data, flats, darks, ensure_clean_memory):
    recon_data = FBP(
        normalize_cupy(data, flats, darks, cutoff=20.5, minus_log=False),
        np.linspace(5.0 * np.pi / 360.0, 180.0 * np.pi / 360.0, data.shape[0]),
        15.5,
    )

    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), -0.00015, rtol=1e-07, atol=1e-6)
    assert_allclose(
        np.mean(recon_data, axis=(1, 2)).sum(), -0.019142, rtol=1e-06, atol=1e-5
    )
    assert_allclose(np.std(recon_data), 0.003561, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32


@cp.testing.gpu
def test_reconstruct_FBP_hook(data, flats, darks, ensure_clean_memory):
    normalized = normalize_cupy(data, flats, darks, cutoff=10, minus_log=True)

    cp.get_default_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()

    objrecon_size = data.shape[2]
    hook = MaxMemoryHook(normalized.size * normalized.itemsize)
    with hook:
        recon_data = FBP(
            normalized,
            np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
            79.5,
            objsize=objrecon_size,
        )

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = data.shape[1]
    estimated_slices, dtype_out, output_dims = FBP.meta.calc_max_slices(1,
                                                       (data.shape[0], data.shape[2]),
                                                       normalized.dtype,
                                                       max_mem,
                                                       objsize=objrecon_size)
    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8

    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.00079770206, rtol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(1, 2)).sum(), 0.10210582, rtol=1e-6)


@cp.testing.gpu
def test_reconstruct_SIRT(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    recon_data = SIRT(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True),
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
        objsize=objrecon_size,
        iterations=10,
        nonnegativity=True,
    )
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.0018447536, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(1, 2)).sum(), 0.23612846, rtol=1e-05)
    assert recon_data.dtype == np.float32


@cp.testing.gpu
def test_reconstruct_SIRT_hook(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    normalized = normalize_cupy(data, flats, darks, cutoff=10, minus_log=True)

    cp.get_default_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()

    objrecon_size = data.shape[2]
    hook = MaxMemoryHook(normalized.size * normalized.itemsize)
    with hook:
        recon_data = SIRT(
            normalized,
            np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
            79.5,
            objsize=objrecon_size,
            iterations=2,
            nonnegativity=True,
        )

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = data.shape[1]
    estimated_slices, dtype_out, output_dims = SIRT.meta.calc_max_slices(1,
                                                       (data.shape[0], data.shape[2]),
                                                       normalized.dtype,
                                                       max_mem,
                                                       objsize=objrecon_size)
    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8


@cp.testing.gpu
def test_reconstruct_SIRT_hook2(ensure_clean_memory):
    np.random.seed(12345)
    data_host = np.random.random_sample(size=(1801, 10, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)

    objrecon_size = data.shape[2]
    cp.get_default_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()

    objrecon_size = data.shape[2]
    hook = MaxMemoryHook(data.size * data.itemsize)
    with hook:
        recon_data = SIRT(
            data,
            np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
            1200,
            objsize=objrecon_size,
            iterations=2,
            nonnegativity=True,
        )

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = data.shape[1]
    estimated_slices, dtype_out, output_dims = SIRT.meta.calc_max_slices(1,
                                                       (data.shape[0], data.shape[2]),
                                                       data.dtype,
                                                       max_mem,
                                                       objsize=objrecon_size)
    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8


@cp.testing.gpu
def test_reconstruct_CGLS(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    recon_data = CGLS(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True),
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
        objsize=objrecon_size,
        iterations=5,
        nonnegativity=True,
    )
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.0021818762, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(1, 2)).sum(), 0.279187, rtol=1e-03)
    assert recon_data.dtype == np.float32


@cp.testing.gpu
def test_reconstruct_CGLS_hook(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    normalized = normalize_cupy(data, flats, darks, cutoff=10, minus_log=True)

    cp.get_default_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()

    objrecon_size = data.shape[2]
    hook = MaxMemoryHook(normalized.size * normalized.itemsize)
    with hook:
        recon_data = CGLS(
            normalized,
            np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
            79.5,
            objsize=objrecon_size,
            iterations=2,
            nonnegativity=True,
        )

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = data.shape[1]
    estimated_slices, dtype_out, output_dims = CGLS.meta.calc_max_slices(1,
                                                       (data.shape[0], data.shape[2]),                                                       
                                                       normalized.dtype,
                                                       max_mem,
                                                       objsize=objrecon_size)

    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8


@cp.testing.gpu
def test_reconstruct_CGLS_hook2(ensure_clean_memory):
    np.random.seed(12345)
    data_host = np.random.random_sample(size=(1801, 10, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)

    objrecon_size = data.shape[2]
    cp.get_default_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()

    objrecon_size = data.shape[2]
    hook = MaxMemoryHook(data.size * data.itemsize)
    with hook:
        recon_data = CGLS(
            data,
            np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
            1200,
            objsize=objrecon_size,
            iterations=2,
            nonnegativity=True,
        )

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = data.shape[1]
    estimated_slices, dtype_out, output_dims = CGLS.meta.calc_max_slices(1,
                                                       (data.shape[0], data.shape[2]),                                                       
                                                       data.dtype,
                                                       max_mem,
                                                       objsize=objrecon_size)
    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8


@cp.testing.gpu
@pytest.mark.perf
def test_FBP_performance(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)
    angles = np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0])
    cor = 79.5

    # cold run first
    FBP(data, angles, cor)
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        FBP(data, angles, cor)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
