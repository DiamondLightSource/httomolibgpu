import cupy as cp
import numpy as np
import pytest
from numbers import Integral, Real

from httomolibgpu.misc.utils import (
    __naninfs_check,
    data_checker,
    __check_variable_type,
    __check_if_data_ndim,
    __check_if_data_3D_array,
)
from numpy.testing import assert_equal


@pytest.mark.parametrize("value", [1, np.int32(1), np.uint64(1)])
def test_variable_type_accepts_integral_scalars(value):
    __check_variable_type(value, [Integral], "radius", [], "example")


@pytest.mark.parametrize("value", [1, 0.1, np.float32(0.1), np.float64(0.1)])
def test_variable_type_accepts_real_scalars(value):
    __check_variable_type(value, [Real], "strength", [], "example")


@pytest.mark.parametrize("numeric_type", [Integral, Real])
@pytest.mark.parametrize("value", [True, np.bool_(True), "1"])
def test_numeric_type_checks_reject_booleans_and_strings(value, numeric_type):
    with pytest.raises(TypeError):
        __check_variable_type(
            value, [numeric_type], "parameter", [], "example", type_error=TypeError
        )


def test_variable_type_keeps_existing_value_error_by_default():
    with pytest.raises(ValueError):
        __check_variable_type("1", [int], "radius", [], "example")


def test_variable_type_does_not_change_allowed_types():
    allowed_types = [int, float]
    __check_variable_type(np.int32(1), allowed_types, "radius", [], "example")
    assert allowed_types == [int, float]


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
def test_data_dimensions_accept_images_and_volumes(shape):
    __check_if_data_ndim(np.empty(shape), (2, 3), "example")


@pytest.mark.parametrize("shape", [(4,), (1, 2, 3, 4), (0, 3), (2, 0, 3)])
def test_data_dimensions_reject_empty_or_unsupported_arrays(shape):
    with pytest.raises(ValueError, match="non-empty 2D or 3D array"):
        __check_if_data_ndim(np.empty(shape), (2, 3), "example")


def test_existing_3d_check_still_rejects_a_2d_image():
    with pytest.raises(ValueError, match="3D array"):
        __check_if_data_3D_array(np.empty((2, 3)), "example")


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
