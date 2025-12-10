import time

import cupy as cp
import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_equal
from httomolibgpu.prep.normalize import dark_flat_field_correction, minus_log

from cupy.cuda import nvtx


def test_dark_flat_field_correction_1D_raises(data, flats, darks, ensure_clean_memory):
    _data_1d = cp.ones(10)

    #: data cannot be a 1D array
    with pytest.raises(ValueError):
        dark_flat_field_correction(_data_1d, flats, darks)

    #: flats cannot be a 1D array
    with pytest.raises(ValueError):
        dark_flat_field_correction(data, _data_1d, darks)


def test_dark_flat_field_minus_log_correction(data, flats, darks, ensure_clean_memory):
    # --- testing dark_flat_field_correction with minus_log ---#
    data_normalize = dark_flat_field_correction(cp.copy(data), flats, darks)
    data_normalize = cp.asnumpy(minus_log(data_normalize))

    assert data_normalize.dtype == np.float32

    assert_allclose(np.mean(data_normalize), 0.2892469, rtol=1e-06)
    assert_allclose(np.mean(data_normalize, axis=(1, 2)).sum(), 52.064465, rtol=1e-06)
    assert_allclose(np.median(data_normalize), 0.01723744, rtol=1e-06)
    assert_allclose(np.std(data_normalize), 0.524382, rtol=1e-06)
    assert data_normalize.flags.c_contiguous


def test_dark_flat_field_minus_log_scale(data, flats, darks, ensure_clean_memory):
    # --- testing dark_flat_field_correction with minus_log and scaling  ---#
    data_normalize = dark_flat_field_correction(
        cp.copy(data),
        flats,
        darks,
        flats_multiplier=10.0,
        darks_multiplier=10.0,
    )
    data_normalize = cp.asnumpy(minus_log(data_normalize))

    assert data_normalize.dtype == np.float32

    assert_allclose(np.mean(data_normalize), 2.5918312, rtol=1e-06)
    assert_allclose(np.mean(data_normalize, axis=(1, 2)).sum(), 466.5298, rtol=1e-06)
    assert_allclose(np.median(data_normalize), 2.3198225, rtol=1e-06)
    assert_allclose(np.std(data_normalize), 0.5243825, rtol=1e-06)
    assert data_normalize.flags.c_contiguous


@pytest.mark.perf
def test_dark_flat_field_performance(ensure_clean_memory):
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
    dark_flat_field_correction(
        cp.copy(data),
        flats,
        darks,
        cutoff=10.0,
    )
    dev = cp.cuda.Device()
    dev.synchronize()
    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        dark_flat_field_correction(
            cp.copy(data),
            flats,
            darks,
            cutoff=10.0,
        )
    nvtx.RangePop()
    dev.synchronize()
    end = time.perf_counter_ns()
    duration_ms = float(end - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
