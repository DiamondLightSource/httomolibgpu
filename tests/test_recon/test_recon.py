import cupy as cp
import numpy as np
import pytest
from httomolib.prep.normalize import normalize
from httomolib.recon.rotation import find_center_360, find_center_vo
from numpy.testing import assert_allclose


@cp.testing.gpu
def test_find_center_vo(data, flats, darks):
    data = normalize(data, flats, darks)

    #--- testing the center of rotation on tomo_standard ---#
    cor = find_center_vo(data)

    data = None #: free up GPU memory
    assert_allclose(cor, 79.5)

    #: Check that we only get a float32 output
    assert cor.dtype == np.float32


@cp.testing.gpu
def test_find_center_vo_ones(ensure_clean_memory):
    mat = cp.ones(shape=(103, 450, 230), dtype=cp.float32)
    cor = find_center_vo(mat)

    assert_allclose(cor, 59.0)
    mat = None #: free up GPU memory


def test_find_center_360_ones():
    mat = cp.ones(shape=(100, 100, 100), dtype=cp.float32)
    (cor, overlap, side, overlap_position) = find_center_360(mat)

    assert_allclose(cor, 5.0)
    assert_allclose(overlap, 12.0)
    assert side == 0
    assert_allclose(overlap_position, 7.0)


def test_find_center_360_data(data):
    eps = 1e-5
    data  = data.astype(cp.float32)
    (cor, overlap, side, overlap_pos) = find_center_360(data)
    assert_allclose(cor, 143.1929, rtol=eps)
    assert_allclose(overlap, 31.61421, rtol=eps)
    assert side == 1
    assert_allclose(overlap_pos, 133.38579, rtol=eps)

    #: Check that we only get a float32 output
    assert cor.dtype == np.float32
    assert overlap.dtype == np.float32


def test_find_center_360_1D_raises(data):
    #: 360-degree sinogram must be a 3d array

    with pytest.raises(ValueError):
        find_center_360(data[:, 10, :])
    with pytest.raises(ValueError):
        find_center_360(np.ones(10))