import cupy as cp
import numpy as np
import pytest
from httomolibgpu.misc.denoise import (
    total_variation_ROF,
    total_variation_PD,
)
from numpy.testing import assert_allclose, assert_equal

eps = 1e-6


def test_tv_rof_wrong_dtype(data):
    with pytest.raises(ValueError):
        total_variation_ROF(data)


def test_tv_ROF(data):
    filtered_data = total_variation_ROF(
        data.astype(cp.float32), regularisation_parameter=5, iterations=200
    ).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), 809.04994, rtol=eps)
    assert_allclose(np.max(filtered_data), 1130.5753)
    assert_allclose(np.min(filtered_data), 67.425545)

    assert filtered_data.dtype == np.float32
    assert filtered_data.flags.c_contiguous


def test_tv_pd_wrong_dtype(data):
    with pytest.raises(ValueError):
        total_variation_PD(data)


def test_tv_PD(data):
    filtered_data = total_variation_PD(
        data.astype(cp.float32), regularisation_parameter=5, iterations=200
    ).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), 809.04987, rtol=eps)
    assert_allclose(np.max(filtered_data), 1112.6956)
    assert_allclose(np.min(filtered_data), 77.97832)

    assert filtered_data.dtype == np.float32
    assert filtered_data.flags.c_contiguous
