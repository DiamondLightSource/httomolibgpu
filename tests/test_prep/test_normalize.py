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


def test_dark_flat_field_upper_bound(data, flats, darks, ensure_clean_memory):
    # --- testing dark_flat_field_correction with minus_log ---#
    data_normalize = cp.asnumpy(
        dark_flat_field_correction(
            cp.copy(data),
            flats,
            darks,
            flats_multiplier=1,
            darks_multiplier=1,
            upper_bound=0.8,
            lower_bound=None,
        )
    )

    assert data_normalize.dtype == np.float32
    assert_allclose(np.mean(data_normalize), 0.6904512, rtol=1e-06)
    assert_allclose(np.mean(data_normalize, axis=(1, 2)).sum(), 124.28119, rtol=1e-06)
    assert data_normalize.flags.c_contiguous


def test_dark_flat_field_lower_bound(data, flats, darks, ensure_clean_memory):
    # --- testing dark_flat_field_correction with minus_log ---#
    data_normalize = cp.asnumpy(
        dark_flat_field_correction(
            cp.copy(data),
            flats,
            darks,
            flats_multiplier=1,
            darks_multiplier=1,
            upper_bound=None,
            lower_bound=0.1,
        )
    )

    assert data_normalize.dtype == np.float32
    assert_allclose(np.mean(data_normalize), 0.8285064, rtol=1e-06)
    assert_allclose(np.mean(data_normalize, axis=(1, 2)).sum(), 149.13116, rtol=1e-06)
    assert data_normalize.flags.c_contiguous


def test_dark_flat_field_upper_and_lower_bound(data, flats, darks, ensure_clean_memory):
    # --- testing dark_flat_field_correction with minus_log ---#
    data_normalize = cp.asnumpy(
        dark_flat_field_correction(
            cp.copy(data),
            flats,
            darks,
            flats_multiplier=1,
            darks_multiplier=1,
            upper_bound=0.7,
            lower_bound=0.2,
        )
    )

    assert data_normalize.dtype == np.float32
    assert_allclose(np.mean(data_normalize), 0.6218485, rtol=1e-06)
    assert_allclose(np.mean(data_normalize, axis=(1, 2)).sum(), 111.93274, rtol=1e-06)
    assert data_normalize.flags.c_contiguous


@pytest.mark.parametrize(
    "clipping_warning", [None, False, True], ids=["default", "off", "on"]
)
@pytest.mark.parametrize(
    "clipped_count", [1, 2, 3], ids=["25_percent", "50_percent", "75_percent"]
)
@pytest.mark.parametrize("bound", ["upper_bound", "lower_bound"])
def test_dark_flat_field_clipping_warning(
    clipping_warning, clipped_count, bound, capsys, ensure_clean_memory
):
    # Unit flats and zero darks leave the input unchanged before clipping.
    clipped_value, other_value = (
        (0.75, 0.25) if bound == "upper_bound" else (0.25, 0.75)
    )
    data_host = np.array(
        [clipped_value] * clipped_count + [other_value] * (4 - clipped_count),
        dtype=np.float32,
    ).reshape(1, 2, 2)
    flats = cp.ones((1, 2, 2), dtype=cp.float32)
    darks = cp.zeros((1, 2, 2), dtype=cp.float32)
    kwargs = {bound: 0.5}
    if clipping_warning is not None:
        kwargs["clipping_warning"] = clipping_warning

    result = dark_flat_field_correction(cp.asarray(data_host), flats, darks, **kwargs)

    captured = capsys.readouterr()
    if clipping_warning and clipped_count >= 2:
        assert captured.out == (
            "Warning! The output data of 'dark_flat_field_correction' method contains "
            f"\033[31m{clipped_count * 25}\033[0m percent of '{bound}' clipped data.\n"
        )
    else:
        assert captured.out == ""
    assert captured.err == ""
    expected = (
        np.minimum(data_host, 0.5)
        if bound == "upper_bound"
        else np.maximum(data_host, 0.5)
    )
    assert_allclose(cp.asnumpy(result), expected)


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
