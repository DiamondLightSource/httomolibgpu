import cupy as cp
import numpy as np
import pytest
import scipy
from httomolib.misc.corr import (
    inpainting_filter3d,
    median_filter3d_cupy,
    remove_outlier3d_cupy,
)
from numpy.testing import assert_allclose, assert_equal

eps = 1e-6


def test_inpainting_filter3d(host_data):
    mask = np.zeros(shape=host_data.shape)
    filtered_data = inpainting_filter3d(host_data, mask)
    
    assert_allclose(np.min(filtered_data), 62.0)
    assert_allclose(np.max(filtered_data), 1136.0)
    assert_allclose(np.mean(filtered_data), 809.04987, rtol=eps)

    #: make sure the output is float32
    assert filtered_data.dtype == np.float32


@cp.testing.gpu
def test_median_filter3d_cupy_vs_scipy_on_arange(ensure_clean_memory):
    mat = np.arange(4*5*6).reshape(4, 5, 6)
    assert_equal(
        scipy.ndimage.median_filter(np.float32(mat), size=3),
        median_filter3d_cupy(cp.asarray(mat, dtype=cp.float32), kernel_size=3).get()
    )


@cp.testing.gpu
def test_median_filter3d_cupy_vs_scipy(host_data, ensure_clean_memory):
    assert_equal(
        scipy.ndimage.median_filter(np.float32(host_data), size=3),
        median_filter3d_cupy(cp.asarray(host_data, dtype=cp.float32), kernel_size=3).get()
    )


@cp.testing.gpu
def test_median_filter3d_cupy_1D_raises(ensure_clean_memory):
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        median_filter3d_cupy(_data, kernel_size=3)


@cp.testing.gpu
def test_median_filter3d_cupy_zero_dim(ensure_clean_memory):
    _data = cp.ones(shape=(10, 10, 0)) * 100
    with pytest.raises(ValueError):
        median_filter3d_cupy(_data, kernel_size=3)


@cp.testing.gpu
def test_median_filter3d_cupy_even_kernel_size(data):
    with pytest.raises(ValueError):
        median_filter3d_cupy(data, kernel_size=4)


@cp.testing.gpu
def test_median_filter3d_cupy_wrong_dtype(data):
    with pytest.raises(ValueError):
        median_filter3d_cupy(data.astype(cp.float64), kernel_size=3)


@cp.testing.gpu
def test_median_filter3d_cupy(data):
    filtered_data = median_filter3d_cupy(data, kernel_size=3).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), 808.753494, rtol=eps)
    assert_allclose(np.max(filtered_data), 1028.0)
    assert_allclose(np.min(filtered_data), 89.0)

    assert filtered_data.dtype == np.uint16

    assert median_filter3d_cupy(
        data.astype(cp.float32), kernel_size=5, dif=1.5).get().dtype == np.float32
