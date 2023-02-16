import time

import cupy as cp
import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_equal
from httomolib.prep.normalize import normalize_cupy, normalize_raw_cuda
from cupy.cuda import nvtx


@cp.testing.gpu
def test_normalize_cupy_1D_raises(data, flats, darks, ensure_clean_memory):
    _data_1d = cp.ones(10)

    #: data cannot be a 1D array
    with pytest.raises(ValueError):
        normalize_cupy(_data_1d, flats, darks)

    #: flats cannot be a 1D array
    with pytest.raises(ValueError):
        normalize_cupy(data, _data_1d, darks)


@cp.testing.gpu
def test_normalize_raw_cuda_1D_raises(data, flats, darks, ensure_clean_memory):
    _data_1d = cp.ones(10)

    #: data cannot be a 1D array
    with pytest.raises(ValueError):
        normalize_raw_cuda(_data_1d, flats, darks)

    with pytest.raises(ValueError):
        normalize_raw_cuda(data, _data_1d, _data_1d)


@cp.testing.gpu
def test_normalize_cupy(data, flats, darks, ensure_clean_memory):
    #--- testing normalize_cupy  ---#
    data_normalize = normalize_cupy(cp.copy(data), flats, darks, minus_log=True).get()

    assert data_normalize.dtype == np.float32

    assert_allclose(np.mean(data_normalize), 0.2892469, rtol=1e-06)
    assert_allclose(np.mean(data_normalize, axis=(1, 2)).sum(), 52.064465, rtol=1e-06)
    assert_allclose(np.median(data_normalize), 0.01723744, rtol=1e-06)
    assert_allclose(np.std(data_normalize), 0.524382, rtol=1e-06)


@cp.testing.gpu
def test_normalize_raw_cuda(data, flats, darks, ensure_clean_memory):
    #--- testing normalize_raw_cuda  ---#
    data_normalize = normalize_raw_cuda(data, flats, darks, minus_log=True).get()

    assert data_normalize.dtype == np.float32

    assert_allclose(np.mean(data_normalize), 0.2892469, rtol=1e-06)
    assert_allclose(np.mean(data_normalize, axis=(1, 2)).sum(), 52.064465, rtol=1e-06)
    assert_allclose(np.median(data_normalize), 0.01723744, rtol=1e-06)
    assert_allclose(np.std(data_normalize), 0.524382, rtol=1e-06)


@cp.testing.gpu
def test_normalize_raw_cuda_vs_normalize_cupy(data, ensure_clean_memory):
    flats = cp.ones(shape=data.shape) + 100
    darks = cp.random.randint(low=64, high=117, size=data.shape, dtype=cp.uint16)
    assert_equal(
        normalize_cupy(cp.copy(data), flats, darks, minus_log=True, nonnegativity=True).get(),
        normalize_raw_cuda(data, flats, darks, minus_log=True, nonnegativity=True).get()
    )


@cp.testing.gpu
@pytest.mark.perf
def test_normalize_cupy_performance(ensure_clean_memory):
    # Note: low/high and size values taken from sample2_medium.yaml real run
    data_host = np.random.randint(low=7515, high=37624, size=(1801, 5, 2560), dtype=np.uint16)
    flats_host = np.random.randint(low=43397, high=52324, size=(50, 5, 2560), dtype=np.uint16)
    darks_host = np.random.randint(low=64, high=117, size=(20, 5,2560), dtype=np.uint16)
    data = cp.asarray(data_host)
    flats = cp.asarray(flats_host)
    darks = cp.asarray(darks_host)
    
    # run code and time it
    # do a cold run for warmup
    normalize_cupy(cp.copy(data), flats, darks, cutoff=10.0, minus_log=True, nonnegativity=False, remove_nans=False)
    dev = cp.cuda.Device()
    dev.synchronize()
    start = time.perf_counter_ns()
    nvtx.RangePush('Core')
    for _ in range(10):
        normalize_cupy(cp.copy(data), flats, darks, cutoff=10.0, minus_log=True, nonnegativity=False, remove_nans=False)
    nvtx.RangePop()
    dev.synchronize()
    end = time.perf_counter_ns()
    duration_ms = float(end - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


@cp.testing.gpu
@pytest.mark.perf
def test_normalize_raw_cuda_performance(ensure_clean_memory):
    data = cp.random.randint(low=7515, high=37624, size=(1801, 5, 2560), dtype=cp.uint16)
    flats = cp.random.randint(low=43397, high=52324, size=(50, 5, 2560), dtype=cp.uint16)
    darks = cp.random.randint(low=64, high=117, size=(20, 5,2560), dtype=cp.uint16)

    # run code and time it
    # do a cold run for warmup
    normalize_raw_cuda(data, flats, darks, cutoff=10.0, minus_log=True, nonnegativity=True)
    dev = cp.cuda.Device()
    dev.synchronize()
    start = time.perf_counter_ns()
    nvtx.RangePush('Core')
    for _ in range(10):
        normalize_raw_cuda(data, flats, darks, cutoff=10.0, minus_log=True, nonnegativity=True)
    nvtx.RangePop()
    dev.synchronize()
    end = time.perf_counter_ns()
    duration_ms = float(end - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
