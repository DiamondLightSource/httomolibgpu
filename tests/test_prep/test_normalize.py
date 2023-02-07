import time

import cupy as cp
import numpy as np
import pytest
from cupy.cuda import nvtx
from httomolib.prep.normalize import normalize_cupy
from numpy.testing import assert_allclose


@cp.testing.gpu
def test_normalize_1D_raises(data, flats, darks):
    _data_1d = cp.ones(10)

    #: data cannot be a 1D array
    with pytest.raises(ValueError):
        normalize_cupy(_data_1d, flats, darks)

    #: flats cannot be a 1D array
    with pytest.raises(ValueError):
        normalize_cupy(data, _data_1d, darks)


@cp.testing.gpu
def test_normalize(data, flats, darks):
    #--- testing normalize_cupy  ---#
    data_normalize = cp.asnumpy(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True))

    assert data_normalize.dtype == np.float32

    assert_allclose(np.min(data_normalize), -0.16163824, rtol=1e-06)
    assert_allclose(np.max(data_normalize), 2.7530956, rtol=1e-06)

@cp.testing.gpu
@pytest.mark.perf
def test_normalize_performance(ensure_clean_memory):
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
