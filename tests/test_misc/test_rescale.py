import time
from typing import Literal
import numpy as np
import cupy as cp
import pytest
from cupy.cuda import nvtx

from httomolibgpu.misc.rescale import rescale_to_int


def test_rescale_no_change():
    data = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8).astype(np.float32)
    data_dev = cp.asarray(data)
    res_dev = rescale_to_int(
        data_dev, bits=8, glob_stats=(0.0, 255.0, 100.0, data.size)
    )

    res = cp.asnumpy(res_dev).astype(np.float32)

    assert res_dev.dtype == np.uint8
    np.testing.assert_array_equal(res, data)


@pytest.mark.parametrize("bits", [8, 16, 32])
def test_rescale_no_change_no_stats(bits: Literal[8, 16, 32]):
    data = np.ones((30, 50), dtype=np.float32)
    data[0, 0] = 0.0
    data[13, 1] = (2**bits) - 1
    data_dev = cp.asarray(data)
    res_dev = rescale_to_int(data_dev, bits=bits)

    res_dev_float32 = cp.asnumpy(res_dev).astype(np.float32)

    assert res_dev.dtype.itemsize == bits // 8
    np.testing.assert_array_equal(res_dev_float32, data)


def test_rescale_double():
    data = np.ones((30, 50), dtype=np.float32)

    data_dev = cp.asarray(data)
    res_dev = rescale_to_int(data_dev, bits=8, glob_stats=(0, 2, 100, data.size))

    res = cp.asnumpy(res_dev).astype(np.float32)

    np.testing.assert_array_almost_equal(res, 127.0)


def test_rescale_handles_nan_inf():
    data = np.ones((30, 50), dtype=np.float32)
    data[0, 0] = float("nan")
    data[0, 1] = float("inf")
    data[0, 2] = float("-inf")

    data_dev = cp.asarray(data)
    res_dev = rescale_to_int(data_dev, bits=8, glob_stats=(0, 2, 100, data.size))

    res = cp.asnumpy(res_dev).astype(np.float32)

    np.testing.assert_array_equal(res[0, 0:3], 0.0)


def test_rescale_double_offset():
    data = np.ones((30, 50), dtype=np.float32) + 10

    data_dev = cp.asarray(data)
    res_dev = rescale_to_int(data_dev, bits=8, glob_stats=(10, 12, 100, data.size))

    res = cp.asnumpy(res_dev).astype(np.float32)

    np.testing.assert_array_almost_equal(res, 127.0)


@pytest.mark.parametrize("bits", [8, 16])
def test_rescale_double_offset_min_percentage(bits: Literal[8, 16, 32]):
    data = np.ones((30, 50), dtype=np.float32) + 15
    data[0, 0] = 10
    data[0, 1] = 20

    data_dev = cp.asarray(data)
    res_dev = rescale_to_int(
        data_dev,
        bits=bits,
        glob_stats=(10, 20, 100, data.size),
        perc_range_min=10.0,
        perc_range_max=90.0,
    )

    res = cp.asnumpy(res_dev).astype(np.float32)

    max = (2**bits) - 1
    num = int(5 / 8 * max)
    # note: with 32bit, the float type actually overflows and the result is not full precision
    np.testing.assert_array_almost_equal(res[1:, :], num)
    assert res[0, 0] == 0.0
    assert res[0, 1] == max


def test_tomo_data_scale(data):
    data_cpu = data.get()
    res_dev = rescale_to_int(
        data.astype(cp.float32), perc_range_min=10, perc_range_max=90, bits=8
    )
    res = res_dev.get()
    assert res_dev.dtype == np.uint8


@pytest.mark.perf
def test_rescale_performance():
    data = cp.random.random((1801, 400, 2560), dtype=np.float32) * 500 + 20
    in_min = float(cp.min(data))
    in_max = float(cp.max(data))

    # do a cold run first
    rescale_to_int(
        data,
        bits=8,
        glob_stats=(in_min, in_max, 100.0, data.size),
        perc_range_max=95.0,
        perc_range_min=5.0,
    )

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(20):
        rescale_to_int(
            data,
            bits=8,
            glob_stats=(in_min, in_max, 100.0, data.size),
            perc_range_max=95.0,
            perc_range_min=5.0,
        )
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 20

    assert "performance in ms" == duration_ms
