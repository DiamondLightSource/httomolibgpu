import cupy as cp
import numpy as np

from httomolibgpu.misc.utils import (
    __naninfs_check,
    data_checker,
)
from numpy.testing import assert_equal


def test_naninfs_check1():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.float32) * 100
    _data_output = __naninfs_check(_data_input.copy())

    assert_equal(
        _data_input.get(),
        _data_output.get(),
    )
    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_naninfs_check2():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.float32) * 100
    _data_input[1, 1, 1] = -cp.inf
    _data_input[1, 1, 2] = cp.inf
    _data_input[1, 1, 3] = cp.nan
    _data_output = __naninfs_check(_data_input.copy())

    assert_equal(
        _data_output[1, 1, 1].get(),
        0,
    )
    assert_equal(
        _data_output[1, 1, 2].get(),
        0,
    )
    assert_equal(
        _data_output[1, 1, 3].get(),
        0,
    )
    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_naninfs_check3():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.uint16) * 100
    _data_output = __naninfs_check(_data_input.copy())

    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_naninfs_check4():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.uint8) * 100
    _data_output = __naninfs_check(_data_input.copy())

    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_data_infsnans_checker():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.float32) * 100.0
    _data_input[1, 1, 1] = -cp.inf
    _data_input[1, 1, 2] = cp.inf
    _data_input[1, 1, 3] = cp.nan

    _data_output = data_checker(_data_input.copy())

    assert_equal(
        _data_output[1, 1, 1].get(),
        0.0,
    )
    assert_equal(
        _data_output[1, 1, 2].get(),
        0.0,
    )
    assert_equal(
        _data_output[1, 1, 3].get(),
        0.0,
    )
    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)
