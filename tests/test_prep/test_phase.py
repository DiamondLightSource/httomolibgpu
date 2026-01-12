import time

import cupy as cp
import numpy as np
import pytest
from cupy.cuda import nvtx
from httomolibgpu.prep.phase import paganin_filter
from numpy.testing import assert_allclose

from ..conftest import MaxMemoryHook

eps = 1e-6


# paganin filter
def test_paganin_filter_1D_raises(ensure_clean_memory):
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        paganin_filter(_data)

    _data = None  #: free up GPU memory


def test_paganin_filter(data):
    # --- testing the Paganin filter on tomo_standard ---#
    filtered_data = paganin_filter(data).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), -6.725061, rtol=eps)
    assert_allclose(np.max(filtered_data), -6.367116, rtol=eps)

    #: make sure the output is float32
    assert filtered_data.dtype == np.float32
    assert filtered_data.flags.c_contiguous


def test_paganin_filter_energy100(data):
    filtered_data = paganin_filter(data, energy=100.0).get()

    assert_allclose(np.mean(filtered_data), -6.7109337, rtol=1e-05)
    assert_allclose(np.min(filtered_data), -6.9103956, rtol=eps)

    assert filtered_data.ndim == 3
    assert filtered_data.dtype == np.float32


def test_paganin_filter_dist3(data):
    filtered_data = paganin_filter(data, distance=3.0, ratio_delta_beta=500).get()

    assert_allclose(np.sum(np.mean(filtered_data, axis=(1, 2))), -1214.3943, rtol=1e-6)
    assert_allclose(np.sum(filtered_data), -24870786.0, rtol=1e-6)


@pytest.mark.parametrize(
    "test_case",
    [
        ("next_power_of_2", None, -6.725061, -6.367116),
        ("next_fast_length", None, -6.677313, -6.096187),
        ("use_pad_x_y", (0, 0), -6.677313, -6.096187),
        ("use_pad_x_y", (80, 80), -6.73193, -6.405338),
        ("use_pad_x_y", (45, 75), -6.726483, -6.37466),
    ],
)
def test_paganin_filter_padding_options(data, test_case):
    # --- testing the Paganin filter on tomo_standard ---#
    padding_method, pad_x_y, test_mean, test_max = test_case
    filtered_data = paganin_filter(
        data,
        calculate_padding_value_method=padding_method,
        pad_x_y=pad_x_y,
    ).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), test_mean, rtol=eps)
    assert_allclose(np.max(filtered_data), test_max, rtol=eps)

    #: make sure the output is float32
    assert filtered_data.dtype == np.float32
    assert filtered_data.flags.c_contiguous


@pytest.mark.perf
def test_paganin_filter_performance(ensure_clean_memory):
    # Note: low/high and size values taken from sample2_medium.yaml real run

    # this test needs ~20GB of memory with 1801 - we'll divide depending on GPU memory
    dev = cp.cuda.Device()
    mem_80percent = 0.8 * dev.mem_info[0]
    size = 1801
    required_mem = 20 * 1024 * 1024 * 1024
    if mem_80percent < required_mem:
        size = int(np.ceil(size / required_mem * mem_80percent))
        print(f"Using smaller size of ({size}, 5, 2560) due to memory restrictions")

    data_host = np.random.random_sample(size=(size, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)

    # run code and time it
    # cold run first
    paganin_filter(data)
    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        paganin_filter(data)

    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


@pytest.mark.parametrize("slices", [3, 7, 32, 61, 109, 120, 150])
@pytest.mark.parametrize("dim_x", [128, 140])
@pytest.mark.parametrize(
    "padding",
    [("next_power_of_2", None), ("next_fast_length", None), ("use_pad_x_y", (45, 45))],
)
def test_paganin_filter_calc_mem(slices, dim_x, padding, ensure_clean_memory):
    dim_y = 159
    padding_method, pad_x_y = padding

    data = cp.random.random_sample((slices, dim_x, dim_y), dtype=np.float32)
    hook = MaxMemoryHook()
    with hook:
        paganin_filter(
            cp.copy(data),
            calculate_padding_value_method=padding_method,
            pad_x_y=pad_x_y,
        )
    actual_mem_peak = hook.max_mem

    try:
        estimated_mem_peak = paganin_filter(
            data.shape,
            calculate_padding_value_method=padding_method,
            pad_x_y=pad_x_y,
            calc_peak_gpu_mem=True,
        )
    except cp.cuda.memory.OutOfMemoryError:
        pytest.skip("Not enough GPU memory to estimate memory peak")

    assert actual_mem_peak == estimated_mem_peak


@pytest.mark.parametrize("slices", [38, 177, 268, 320, 490, 607, 803, 859, 902, 951])
@pytest.mark.parametrize("dims", [(900, 1280), (1801, 1540), (1801, 2560)])
@pytest.mark.parametrize(
    "padding",
    [
        ("next_power_of_2", None),
        ("next_fast_length", None),
        ("use_pad_x_y", (145, 122)),
    ],
)
def test_paganin_filter_calc_mem_big(slices, dims, padding, ensure_clean_memory):
    dim_y, dim_x = dims
    data_shape = (slices, dim_x, dim_y)
    padding_method, pad_x_y = padding
    try:
        estimated_mem_peak = paganin_filter(
            data_shape,
            calculate_padding_value_method=padding_method,
            pad_x_y=pad_x_y,
            calc_peak_gpu_mem=True,
        )
    except cp.cuda.memory.OutOfMemoryError:
        pytest.skip("Not enough GPU memory to estimate memory peak")
    except cp.cuda.cufft.CuFFTError as cufft_error:
        if cufft_error.result == 8:  # CUFFT_INVALID_SIZE
            pytest.skip("Not usable FFT size")
        else:
            raise
    av_mem = cp.cuda.Device().mem_info[0]
    if av_mem < estimated_mem_peak:
        pytest.skip("Not enough GPU memory to run this test")

    hook = MaxMemoryHook()
    with hook:
        data = cp.random.random_sample(data_shape, dtype=np.float32)
        paganin_filter(
            data,
            calculate_padding_value_method=padding_method,
            pad_x_y=pad_x_y,
        )
    actual_mem_peak = hook.max_mem

    assert actual_mem_peak == estimated_mem_peak
