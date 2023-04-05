import time

import cupy as cp
import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_equal
from httomolib.prep.normalize import normalize
from httomolib import method_registry
from cupy.cuda import nvtx

from tests import MaxMemoryHook


@cp.testing.gpu
def test_normalize_1D_raises(data, flats, darks, ensure_clean_memory):
    _data_1d = cp.ones(10)

    #: data cannot be a 1D array
    with pytest.raises(ValueError):
        normalize(_data_1d, flats, darks)

    #: flats cannot be a 1D array
    with pytest.raises(ValueError):
        normalize(data, _data_1d, darks)


@cp.testing.gpu
def test_normalize(data, flats, darks, ensure_clean_memory):
    # --- testing normalize  ---#
    hook = MaxMemoryHook()
    with hook:
        data_normalize = normalize(cp.copy(data), flats, darks, minus_log=True).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = data.shape[0]
    estimated_slices = normalize.meta.calc_max_slices(0, (data.shape[1], data.shape[2]), data.dtype, max_mem,
                                                      flats=flats, darks=darks)
    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8 

    assert data_normalize.dtype == np.float32

    assert_allclose(np.mean(data_normalize), 0.2892469, rtol=1e-06)
    assert_allclose(np.mean(data_normalize, axis=(1, 2)).sum(), 52.064465, rtol=1e-06)
    assert_allclose(np.median(data_normalize), 0.01723744, rtol=1e-06)
    assert_allclose(np.std(data_normalize), 0.524382, rtol=1e-06)


@cp.testing.gpu
@pytest.mark.perf
def test_normalize_performance(ensure_clean_memory):
    # Note: low/high and size values taken from sample2_medium.yaml real run
    data_host = np.random.randint(
        low=7515, high=37624, size=(1801, 5, 2560), dtype=np.uint16
    )
    flats_host = np.random.randint(
        low=43397, high=52324, size=(50, 5, 2560), dtype=np.uint16
    )
    darks_host = np.random.randint(
        low=64, high=117, size=(20, 5, 2560), dtype=np.uint16
    )
    data = cp.asarray(data_host)
    flats = cp.asarray(flats_host)
    darks = cp.asarray(darks_host)

    # run code and time it
    # do a cold run for warmup
    normalize(
        cp.copy(data),
        flats,
        darks,
        cutoff=10.0,
        minus_log=True,
        nonnegativity=False,
        remove_nans=False,
    )
    dev = cp.cuda.Device()
    dev.synchronize()
    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        normalize(
            cp.copy(data),
            flats,
            darks,
            cutoff=10.0,
            minus_log=True,
            nonnegativity=False,
            remove_nans=False,
        )
    nvtx.RangePop()
    dev.synchronize()
    end = time.perf_counter_ns()
    duration_ms = float(end - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_normalize_memory_calc():
    dy = 5
    dx = 2560
    # space for 200 slices in total, incl. flats/darks means, but not the 
    # flats/darks themselves (we assume they have been pre-allocated already and
    # excluded from available memory)
    available_memory = (dx*dy*200 + 42) * 4  

    # 2 slices for means are needed -> 198 left
    flats = cp.empty((50, dy, dx), dtype=np.uint16)
    darks = cp.empty((30, dy, dx), dtype=np.uint16)
    
    args = dict(flats=flats, 
                darks=darks, 
                cutoff=10.0,
                minus_log=True,
                nonnegativity=False,
                remove_nans=False)
    
    assert 'normalize' in method_registry['httomolib']['prep']['normalize']

    # with 4-byte inputs and outputs, we can fit 198 slices between input / output
    assert normalize.meta.calc_max_slices(0, (dy, dx), np.float32(), available_memory, **args) == 99

    # with 2-byte inputs, 4-byte outputs, we can fit 132 (input takes only half the space)
    assert normalize.meta.calc_max_slices(0, (dy, dx), np.uint16(), available_memory, **args) == 132
    