import cupy as cp
import numpy as np

from httomolibgpu.misc.supp_func import (
    _naninfs_check,
    _zeros_check,
    data_checker,
)
from numpy.testing import assert_equal


def test_naninfs_check1():
    _data_input = cp.ones(shape=(10, 10, 10)) * 100
    _data_output = _naninfs_check(_data_input.copy())

    assert_equal(
        _data_input.get(),
        _data_output.get(),
    )
    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_naninfs_check1_numpy():
    _data_input = np.ones(shape=(10, 10, 10)) * 100
    _data_output = _naninfs_check(_data_input.copy())

    assert_equal(
        _data_input,
        _data_output,
    )
    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_naninfs_check2():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.float32) * 100
    _data_input[1, 1, 1] = -cp.inf
    _data_input[1, 1, 2] = cp.inf
    _data_input[1, 1, 3] = cp.nan
    _data_output = _naninfs_check(_data_input.copy())

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


def test_naninfs_check2_numpy():
    _data_input = np.ones(shape=(10, 10, 10), dtype=np.float32) * 100
    _data_input[1, 1, 1] = -np.inf
    _data_input[1, 1, 2] = np.inf
    _data_input[1, 1, 3] = np.nan
    _data_output = _naninfs_check(_data_input.copy())

    assert_equal(
        _data_output[1, 1, 1],
        0,
    )
    assert_equal(
        _data_output[1, 1, 2],
        0,
    )
    assert_equal(
        _data_output[1, 1, 3],
        0,
    )
    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_naninfs_check3():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.uint16) * 100
    _data_output = _naninfs_check(_data_input.copy())

    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_naninfs_check3_numpy():
    _data_input = np.ones(shape=(10, 10, 10), dtype=np.uint16) * 100
    _data_output = _naninfs_check(_data_input.copy())

    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_naninfs_check4_numpy():
    _data_input = np.ones(shape=(10, 10, 10), dtype=np.uint8) * 100
    _data_output = _naninfs_check(_data_input.copy())

    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_zeros_check1():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.float32) * 100
    warning_zeros = _zeros_check(_data_input.copy())

    assert warning_zeros == False


def test_zeros_check1_numpy():
    _data_input = np.ones(shape=(10, 10, 10), dtype=np.float32) * 100
    warning_zeros = _zeros_check(_data_input.copy())

    assert warning_zeros == False


def test_zeros_check2():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.float32) * 100
    _data_input[2:7, :, :] = 0.0
    warning_zeros = _zeros_check(_data_input.copy())

    assert warning_zeros == True


def test_zeros_check2_numpy():
    _data_input = np.ones(shape=(10, 10, 10), dtype=np.float32)
    _data_input[2:7, :, :] = 0.0
    warning_zeros = _zeros_check(_data_input.copy())

    assert warning_zeros == True


def test_zeros_check3():
    _data_input = cp.ones(shape=(10, 10, 10)).astype(cp.float32) * 100
    _data_input[3:7, :, :] = 0.0
    warning_zeros = _zeros_check(_data_input.copy())

    assert warning_zeros == False


def test_zeros_check3_numpy():
    _data_input = np.ones(shape=(10, 10, 10), dtype=np.float32)
    _data_input[3:7, :, :] = 0.0
    warning_zeros = _zeros_check(_data_input.copy())

    assert warning_zeros == False


def test_data_checker_numpy():
    _data_input = np.ones(shape=(10, 10, 10)).astype(np.float32)
    _data_input[1, 1, 1] = -np.inf
    _data_input[1, 1, 2] = np.inf
    _data_input[1, 1, 3] = np.nan

    _data_output = data_checker(_data_input.copy())

    assert_equal(
        _data_output[1, 1, 1],
        0,
    )
    assert_equal(
        _data_output[1, 1, 2],
        0,
    )
    assert_equal(
        _data_output[1, 1, 3],
        0,
    )
    assert _data_output.dtype == _data_input.dtype
    assert _data_output.shape == (10, 10, 10)


def test_data_checker():
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
