import time
import cupy as cp
from cupy.cuda import nvtx
import numpy as np
import pytest
from httomolib.prep.normalize import normalize_cupy
from httomolib.prep.stripe import (
    remove_stripe_based_sorting_cupy,
    remove_stripe_ti,
)
from numpy.testing import assert_allclose


@cp.testing.gpu
def test_remove_stripe_ti_on_data(data, flats, darks):
    # --- testing the CuPy implementation from TomoCupy ---#
    data = normalize_cupy(data, flats, darks, cutoff=10, minus_log=True)
    data_after_stripe_removal = remove_stripe_ti(data).get()

    data = None  #: free up GPU memory
    assert_allclose(np.mean(data_after_stripe_removal), 0.28924704, rtol=1e-05)
    assert_allclose(np.mean(data_after_stripe_removal, axis=(1, 2)).sum(),
        52.064457, rtol=1e-06)
    assert_allclose(np.median(data_after_stripe_removal), 0.026177486, rtol=1e-05)
    assert_allclose(np.max(data_after_stripe_removal), 2.715983, rtol=1e-05)

    # make sure the output is float32
    assert data_after_stripe_removal.dtype == np.float32


def test_remove_stripe_ti_on_flats(host_flats):
    #: testing that numpy uint16 arrays can be passed
    corrected_data = remove_stripe_ti(host_flats)
    assert_allclose(np.mean(corrected_data), 976.558447, rtol=1e-7)
    assert_allclose(np.mean(corrected_data, axis=(1, 2)).sum(),
        19531.168945, rtol=1e-7)
    assert_allclose(np.median(corrected_data), 976., rtol=1e-7)


@cp.testing.gpu
def test_stripe_removal_sorting_cupy(data, flats, darks):
    # --- testing the CuPy port of TomoPy's implementation ---#
    data = normalize_cupy(data, flats, darks, cutoff=10, minus_log=True)
    corrected_data = remove_stripe_based_sorting_cupy(data).get()

    data = None  #: free up GPU memory
    assert_allclose(np.mean(corrected_data), 0.288198, rtol=1e-06)
    assert_allclose(np.max(corrected_data), 2.5242403, rtol=1e-07)
    assert_allclose(np.min(corrected_data), -0.10906063, rtol=1e-07)

    # make sure the output is float32
    assert corrected_data.dtype == np.float32


@cp.testing.gpu
@pytest.mark.perf
def test_stripe_removal_sorting_cupy_performance():
    data_host = np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    data = cp.asarray(data_host, dtype=np.float32)

    # do a cold run first
    remove_stripe_based_sorting_cupy(cp.copy(data))

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        # have to take copy, as data is modified in-place
        remove_stripe_based_sorting_cupy(cp.copy(data))
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
