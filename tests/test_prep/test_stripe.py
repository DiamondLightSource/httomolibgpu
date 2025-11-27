import time
import cupy as cp
from cupy.cuda import nvtx
import numpy as np
import pytest

from httomolibgpu.prep.normalize import dark_flat_field_correction, minus_log
from httomolibgpu.prep.stripe import (
    remove_stripe_based_sorting,
    remove_stripe_ti,
    remove_stripe_fw,
    remove_all_stripe,
    raven_filter,
)
from numpy.testing import assert_allclose


def test_remove_stripe_ti_on_data(data, flats, darks):
    # --- testing the CuPy implementation from TomoCupy ---#
    data_norm = dark_flat_field_correction(cp.copy(data), flats, darks, cutoff=10)
    data_norm = minus_log(data_norm)

    data_after_stripe_removal = cp.asnumpy(remove_stripe_ti(data_norm))

    assert_allclose(np.mean(data_after_stripe_removal), 0.28924704, rtol=1e-05)
    assert_allclose(
        np.mean(data_after_stripe_removal, axis=(1, 2)).sum(), 52.064457, rtol=1e-06
    )
    assert_allclose(np.median(data_after_stripe_removal), 0.026177486, rtol=1e-05)
    assert_allclose(np.max(data_after_stripe_removal), 2.715983, rtol=1e-05)
    assert data_after_stripe_removal.flags.c_contiguous

    data = None  #: free up GPU memory
    # make sure the output is float32
    assert data_after_stripe_removal.dtype == np.float32


class MaxMemoryHook(cp.cuda.MemoryHook):
    def __init__(self, initial=0):
        self.max_mem = initial
        self.current = initial

    def malloc_postprocess(
        self, device_id: int, size: int, mem_size: int, mem_ptr: int, pmem_id: int
    ):
        self.current += mem_size
        self.max_mem = max(self.max_mem, self.current)

    def free_postprocess(
        self, device_id: int, mem_size: int, mem_ptr: int, pmem_id: int
    ):
        self.current -= mem_size


def test_remove_stripe_fw_on_data(data, flats, darks):
    # --- testing the CuPy implementation from TomoCupy ---#
    data_norm = dark_flat_field_correction(data, flats, darks, cutoff=10)
    data_norm = minus_log(data_norm)

    data_after_stripe_removal = remove_stripe_fw(cp.copy(data_norm)).get()

    assert_allclose(np.mean(data_after_stripe_removal), 0.279236, rtol=1e-05)
    assert_allclose(
        np.mean(data_after_stripe_removal, axis=(1, 2)).sum(), 50.2624, rtol=1e-06
    )
    assert_allclose(np.median(data_after_stripe_removal), 0.079203, rtol=1e-05)
    assert_allclose(np.max(data_after_stripe_removal), 2.442347, rtol=1e-05)
    assert data_after_stripe_removal.flags.c_contiguous

    data = None  #: free up GPU memory
    # make sure the output is float32
    assert data_after_stripe_removal.dtype == np.float32


@pytest.fixture
def ensure_clean_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    yield None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()


@pytest.mark.parametrize("wname", ["haar", "db4", "sym5", "sym16", "bior4.4"])
@pytest.mark.parametrize("slices", [55, 80])
@pytest.mark.parametrize("level", [1, 3, 7, 11])
@pytest.mark.parametrize("dim_x", [128, 140])
def test_remove_stripe_fw_calc_mem(slices, level, dim_x, wname, ensure_clean_memory):
    dim_y = 159
    data = cp.random.random_sample((slices, dim_x, dim_y), dtype=np.float32)
    hook = MaxMemoryHook()
    with hook:
        remove_stripe_fw(cp.copy(data), level=level)
    actual_mem_peak = hook.max_mem

    hook = MaxMemoryHook()
    with hook:
        estimated_mem_peak = remove_stripe_fw(
            data.shape, level=level, wname=wname, calc_peak_gpu_mem=True
        )
    assert hook.max_mem == 0

    assert actual_mem_peak * 0.99 <= estimated_mem_peak
    assert estimated_mem_peak <= actual_mem_peak * 1.01


@pytest.mark.parametrize("angles", [180, 181])
@pytest.mark.parametrize("det_x", [11, 18])
@pytest.mark.parametrize("det_y", [5, 7, 8])
def test_remove_stripe_ti_dims_change(angles, det_y, det_x):
    data = cp.random.random_sample(size=(angles, det_y, det_x)).astype(cp.float32) * 2.0
    corrected_data = remove_stripe_ti(data.copy())
    assert corrected_data.shape == (angles, det_y, det_x)


def test_stripe_removal_sorting_cupy(data, flats, darks):
    # --- testing the CuPy port of TomoPy's implementation ---#
    data_norm = dark_flat_field_correction(cp.copy(data), flats, darks, cutoff=10)
    data_norm = minus_log(data_norm)

    corrected_data = cp.asnumpy(remove_stripe_based_sorting(data_norm))

    data = None  #: free up GPU memory
    assert_allclose(np.mean(corrected_data), 0.288198, rtol=1e-06)
    assert_allclose(np.mean(corrected_data, axis=(1, 2)).sum(), 51.87565, rtol=1e-06)
    assert_allclose(np.sum(corrected_data), 1062413.6, rtol=1e-06)

    # make sure the output is float32
    assert corrected_data.dtype == np.float32
    assert corrected_data.flags.c_contiguous


def test_stripe_raven_cupy(data, flats, darks):
    data_norm = dark_flat_field_correction(cp.copy(data), flats, darks, cutoff=10)
    data_norm = minus_log(data_norm)

    data_after_raven_gpu = cp.asnumpy(raven_filter(data_norm))

    data = None  #: free up GPU memory
    assert_allclose(np.mean(data_after_raven_gpu), 0.2892464, rtol=1e-06)

    # make sure the output is float32
    assert data_after_raven_gpu.dtype == np.float32
    assert data_after_raven_gpu.shape == (180, 128, 160)


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


@pytest.mark.perf
def test_raven_filter_performance(ensure_clean_memory):
    data_host = (
        np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    )
    data = cp.asarray(data_host, dtype=np.float32)

    # do a cold run first
    raven_filter(cp.copy(data))

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        # have to take copy, as data is modified in-place
        raven_filter(cp.copy(data))
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_remove_all_stripe_on_data(data, flats, darks):
    # --- testing the CuPy implementation from TomoCupy ---#
    data_norm = dark_flat_field_correction(cp.copy(data), flats, darks, cutoff=10)
    data_norm = minus_log(data_norm)

    data_after_stripe_removal = cp.asnumpy(remove_all_stripe(data_norm))

    assert_allclose(np.mean(data_after_stripe_removal), 0.266914, rtol=1e-05)
    assert_allclose(
        np.mean(data_after_stripe_removal, axis=(1, 2)).sum(), 48.04459, rtol=1e-06
    )
    assert_allclose(np.median(data_after_stripe_removal), 0.015338, rtol=1e-04)
    assert_allclose(np.max(data_after_stripe_removal), 2.298123, rtol=1e-05)
    assert_allclose(
        np.median(data_after_stripe_removal, axis=(1, 2)).sum(), 2.788046, rtol=1e-6
    )
    assert_allclose(
        np.median(data_after_stripe_removal, axis=(0, 1)).sum(), 28.661312, rtol=1e-6
    )

    data = None  #: free up GPU memory
    # make sure the output is float32
    assert data_after_stripe_removal.dtype == np.float32
    assert data_after_stripe_removal.flags.c_contiguous


@pytest.mark.perf
def test_remove_all_stripe_performance(ensure_clean_memory):
    data_host = (
        np.random.random_sample(size=(1801, 100, 2560)).astype(np.float32) * 2.0 + 0.001
    )
    data = cp.asarray(data_host, dtype=np.float32)
    remove_all_stripe(cp.copy(data))
    dev = cp.cuda.Device()
    dev.synchronize()
    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        # have to take copy, as data is modified in-place
        remove_all_stripe(cp.copy(data))
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
