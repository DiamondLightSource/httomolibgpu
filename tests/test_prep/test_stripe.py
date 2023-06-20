import time
import cupy as cp
from cupy.cuda import nvtx
import numpy as np
import pytest
from httomolibgpu.prep.normalize import normalize
from httomolibgpu.prep.stripe import (
    remove_stripe_based_sorting,
    remove_stripe_ti,
)
from httomolibgpu import method_registry
from numpy.testing import assert_allclose

from tests import MaxMemoryHook


@cp.testing.gpu
def test_remove_stripe_ti_on_data(data, flats, darks):
    # --- testing the CuPy implementation from TomoCupy ---#
    data = normalize(data, flats, darks, cutoff=10, minus_log=True)
    
    data_after_stripe_removal = remove_stripe_ti(cp.copy(data)).get()

    assert_allclose(np.mean(data_after_stripe_removal), 0.28924704, rtol=1e-05)
    assert_allclose(
        np.mean(data_after_stripe_removal, axis=(1, 2)).sum(), 52.064457, rtol=1e-06
    )
    assert_allclose(np.median(data_after_stripe_removal), 0.026177486, rtol=1e-05)
    assert_allclose(np.max(data_after_stripe_removal), 2.715983, rtol=1e-05)
    
    data = None  #: free up GPU memory
    # make sure the output is float32
    assert data_after_stripe_removal.dtype == np.float32

@cp.testing.gpu
def test_remove_stripe_ti_on_data_meta(data, flats, darks):
    # --- testing the CuPy implementation from TomoCupy ---#
    data = normalize(data, flats, darks, cutoff=10, minus_log=True)
    
    hook = MaxMemoryHook()
    with hook:
        data_after_stripe_removal = remove_stripe_ti(cp.copy(data)).get()

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = data.shape[1]
    estimated_slices, dtype_out, output_dims = remove_stripe_ti.meta.calc_max_slices(1,
                                                                (data.shape[0], data.shape[2]),
                                                                 data.dtype, max_mem)
    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8 

    data = None  #: free up GPU memory   
    assert remove_stripe_ti.meta.pattern == 'sinogram'
    assert 'remove_stripe_ti' in method_registry['httomolibgpu']['prep']['stripe']


def test_remove_stripe_ti_on_flats(host_flats):
    #: testing that numpy uint16 arrays can be passed
    corrected_data = remove_stripe_ti(np.copy(host_flats))
    assert_allclose(np.mean(corrected_data), 976.558447, rtol=1e-7)
    assert_allclose(np.mean(corrected_data, axis=(1, 2)).sum(), 19531.168945, rtol=1e-7)
    assert_allclose(np.median(corrected_data), 976.0, rtol=1e-7)


@cp.testing.gpu
def test_remove_stripe_ti_numpy_vs_cupy_on_random_data():
    host_data = np.random.random_sample(size=(181, 5, 256)).astype(np.float32) * 2.0
    corrected_host_data = remove_stripe_ti(np.copy(host_data))
    corrected_data = remove_stripe_ti(
        cp.copy(cp.asarray(host_data, dtype=cp.float32))
    ).get()

    assert_allclose(np.sum(corrected_data), np.sum(corrected_host_data), rtol=1e-6)
    assert_allclose(
        np.median(corrected_data), np.median(corrected_host_data), rtol=1e-6
    )


@cp.testing.gpu
def test_stripe_removal_sorting_cupy(data, flats, darks):
    # --- testing the CuPy port of TomoPy's implementation ---#
    data = normalize(data, flats, darks, cutoff=10, minus_log=True)
    corrected_data = remove_stripe_based_sorting(data).get()    
   
    data = None  #: free up GPU memory
    assert_allclose(np.mean(corrected_data), 0.288198, rtol=1e-06)
    assert_allclose(np.mean(corrected_data, axis=(1, 2)).sum(), 51.87565, rtol=1e-06)
    assert_allclose(np.sum(corrected_data), 1062413.6, rtol=1e-06)

    # make sure the output is float32
    assert corrected_data.dtype == np.float32

@cp.testing.gpu
def test_stripe_removal_sorting_cupy_meta(data, flats, darks):
    # --- testing the CuPy port of TomoPy's implementation ---#
    data = normalize(data, flats, darks, cutoff=10, minus_log=True)
    hook = MaxMemoryHook(data.size * data.itemsize)
    with hook:
        corrected_data = remove_stripe_based_sorting(data).get()
    
    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = data.shape[1]
    estimated_slices, dtype_out, output_dims = remove_stripe_based_sorting.meta.calc_max_slices(1,
                                                                           (data.shape[0], data.shape[2]),                                                                           
                                                                           data.dtype, max_mem)
    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8 
    
    data = None  #: free up GPU memory
    assert remove_stripe_based_sorting.meta.pattern == 'sinogram'
    assert 'remove_stripe_based_sorting' in method_registry['httomolibgpu']['prep']['stripe']


@cp.testing.gpu
@cp.testing.numpy_cupy_allclose(rtol=1e-6)
def test_stripe_removal_sorting_numpy_vs_cupy_on_random_data(ensure_clean_memory, xp):
    np.random.seed(12345)
    data = np.random.random_sample(size=(181, 5, 256)).astype(np.float32) * 2.0 + 0.001
    data = xp.asarray(data)
    return xp.asarray(remove_stripe_based_sorting(data))


@cp.testing.gpu
@pytest.mark.perf
def test_stripe_removal_sorting_cupy_performance(ensure_clean_memory):
    data_host = (
        np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    )
    data = cp.asarray(data_host, dtype=np.float32)

    # do a cold run first
    remove_stripe_based_sorting(cp.copy(data))

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        # have to take copy, as data is modified in-place
        remove_stripe_based_sorting(cp.copy(data))
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


@cp.testing.gpu
@pytest.mark.perf
def test_remove_stripe_ti_performance(ensure_clean_memory):
    data_host = (
        np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    )
    data = cp.asarray(data_host, dtype=np.float32)

    # do a cold run first
    remove_stripe_ti(cp.copy(data))

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        # have to take copy, as data is modified in-place
        remove_stripe_ti(cp.copy(data))
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
