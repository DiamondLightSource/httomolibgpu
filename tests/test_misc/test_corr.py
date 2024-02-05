import time

import cupy as cp
import numpy as np
import pytest
import scipy
from cupy.cuda import nvtx
from cupyx.scipy.ndimage import median_filter as median_filter_cupy
from httomolibgpu.misc.corr import (
    median_filter3d,
    remove_outlier3d,
)
from numpy.testing import assert_allclose, assert_equal

eps = 1e-6


def test_median_filter3d_vs_scipy_on_arange(ensure_clean_memory):
    mat = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    assert_equal(
        scipy.ndimage.median_filter(np.float32(mat), size=3),
        median_filter3d(cp.asarray(mat, dtype=cp.float32), kernel_size=3).get(),
    )


def test_median_filter3d_vs_scipy(host_data, ensure_clean_memory):
    assert_equal(
        scipy.ndimage.median_filter(np.float32(host_data), size=3),
        median_filter3d(cp.asarray(host_data, dtype=cp.float32), kernel_size=3).get(),
    )


@pytest.mark.perf
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_median_filter3d_benchmark(
    host_data, ensure_clean_memory, kernel_size, benchmark
):
    benchmark(
        median_filter3d,
        cp.asarray(host_data, dtype=cp.float32),
        kernel_size=kernel_size,
    )


@pytest.mark.perf
@pytest.mark.parametrize("size", [3, 5])
def test_scipy_median_filter_benchmark(data, ensure_clean_memory, benchmark, size):
    benchmark(median_filter_cupy, data.astype(cp.float32), size=size)


def test_median_filter3d_1D_raises(ensure_clean_memory):
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        median_filter3d(_data, kernel_size=3)


def test_median_filter3d_zero_dim(ensure_clean_memory):
    _data = cp.ones(shape=(10, 10, 0)) * 100
    with pytest.raises(ValueError):
        median_filter3d(_data, kernel_size=3)


def test_median_filter3d_even_kernel_size(data):
    with pytest.raises(ValueError):
        median_filter3d(data, kernel_size=4)


def test_median_filter3d_wrong_dtype(data):
    with pytest.raises(ValueError):
        median_filter3d(data.astype(cp.float64), kernel_size=3)


def test_median_filter3d(data):
    filtered_data = median_filter3d(data, kernel_size=3).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), 808.753494, rtol=eps)
    assert_allclose(np.mean(filtered_data, axis=(1, 2)).sum(), 145575.628906)
    assert_allclose(np.max(filtered_data), 1028.0)
    assert_allclose(np.min(filtered_data), 89.0)

    assert filtered_data.dtype == np.uint16
    assert filtered_data.flags.c_contiguous

    assert (
        median_filter3d(data.astype(cp.float32), kernel_size=5, dif=1.5).get().dtype
        == np.float32
    )


@pytest.mark.perf
def test_median_filter3d_performance(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.random_sample(size=(450, 2160, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=cp.float32)

    # warm up
    median_filter3d(data, kernel_size=3)
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        median_filter3d(data, kernel_size=3)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_remove_outlier3d(data):
    filtered_data = remove_outlier3d(data, kernel_size=3, dif=1.5).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), 808.753494, rtol=eps)
    assert_allclose(np.mean(filtered_data, axis=(1, 2)).sum(), 145575.628906)
    assert_allclose(np.median(filtered_data), 976.0)
    assert_allclose(np.median(filtered_data, axis=(1, 2)).sum(), 175741.5)

    assert filtered_data.dtype == np.uint16

    assert (
        remove_outlier3d(data.astype(cp.float32), kernel_size=5, dif=1.5).get().dtype
        == np.float32
    )
    assert filtered_data.flags.c_contiguous
