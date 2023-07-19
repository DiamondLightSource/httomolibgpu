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
from httomolibgpu import method_registry
from numpy.testing import assert_allclose, assert_equal

eps = 1e-6


@cp.testing.gpu
def test_median_filter3d_vs_scipy_on_arange(ensure_clean_memory):
    mat = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    assert_equal(
        scipy.ndimage.median_filter(np.float32(mat), size=3),
        median_filter3d(cp.asarray(mat, dtype=cp.float32), kernel_size=3).get(),
    )


@cp.testing.gpu
def test_median_filter3d_vs_scipy(host_data, ensure_clean_memory):
    assert_equal(
        scipy.ndimage.median_filter(np.float32(host_data), size=3),
        median_filter3d(cp.asarray(host_data, dtype=cp.float32), kernel_size=3).get(),
    )


@pytest.mark.perf
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_median_filter3d_benchmark(host_data, ensure_clean_memory, kernel_size, benchmark):
    benchmark(median_filter3d, cp.asarray(host_data, dtype=cp.float32), kernel_size=kernel_size)


@pytest.mark.perf
@pytest.mark.parametrize("size", [3, 5])
def test_scipy_median_filter_benchmark(data, ensure_clean_memory, benchmark, size):
    benchmark(median_filter_cupy, data.astype(cp.float32), size=size)


@cp.testing.gpu
def test_median_filter3d_1D_raises(ensure_clean_memory):
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        median_filter3d(_data, kernel_size=3)


@cp.testing.gpu
def test_median_filter3d_zero_dim(ensure_clean_memory):
    _data = cp.ones(shape=(10, 10, 0)) * 100
    with pytest.raises(ValueError):
        median_filter3d(_data, kernel_size=3)


@cp.testing.gpu
def test_median_filter3d_even_kernel_size(data):
    with pytest.raises(ValueError):
        median_filter3d(data, kernel_size=4)


@cp.testing.gpu
def test_median_filter3d_wrong_dtype(data):
    with pytest.raises(ValueError):
        median_filter3d(data.astype(cp.float64), kernel_size=3)


@cp.testing.gpu
def test_median_filter3d(data):
    filtered_data = median_filter3d(data, kernel_size=3).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), 808.753494, rtol=eps)
    assert_allclose(np.mean(filtered_data, axis=(1, 2)).sum(), 145575.628906)
    assert_allclose(np.max(filtered_data), 1028.0)
    assert_allclose(np.min(filtered_data), 89.0)

    assert filtered_data.dtype == np.uint16

    assert (
        median_filter3d(data.astype(cp.float32), kernel_size=5, dif=1.5).get().dtype
        == np.float32
    )


@cp.testing.gpu
def test_median_filter3d_memory_calc():
    dy = 5
    dx = 2560
    available_memory = (dx*dy*200 + 42) * 2
    args=dict(kernel_size=3, dif=1.5)

    assert 'median_filter3d' in method_registry['httomolibgpu']['misc']['corr']
    assert median_filter3d.meta.calc_max_slices(0, 
                                                (dy, dx),
                                                np.uint16(), available_memory, **args) == (100, np.uint16(), (dy, dx))
    assert median_filter3d.meta.calc_max_slices(0, 
                                                (dy, dx),
                                                np.float32(), available_memory, **args) == (50, np.float32(), (dy, dx))
    assert median_filter3d.meta.calc_max_slices(1, 
                                                (dy, dx),
                                                np.uint16(), available_memory, **args) == (100, np.uint16(), (dy, dx))
    assert median_filter3d.meta.calc_max_slices(1, 
                                                (dy, dx),
                                                np.float32(), available_memory, **args) == (50, np.float32(), (dy, dx))


@cp.testing.gpu
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


@cp.testing.gpu
def test_remove_outlier3d(data):
    filtered_data = remove_outlier3d(data, kernel_size=9, dif=1.5).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), 809.023452, rtol=eps)
    assert_allclose(np.mean(filtered_data, axis=(1, 2)).sum(), 145624.221436)
    assert_allclose(np.median(filtered_data), 980.0)
    assert_allclose(np.median(filtered_data, axis=(1, 2)).sum(), 176400.0)

    assert filtered_data.dtype == np.uint16

    assert (
        remove_outlier3d(data.astype(cp.float32), kernel_size=5, dif=1.5).get().dtype
        == np.float32
    )
