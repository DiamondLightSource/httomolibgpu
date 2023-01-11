import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose

from httomolib.prep.normalize import normalize_cupy
from httomolib.recon.rotation import find_center_360, find_center_vo_cupy

in_file = 'tests/test_data/tomo_standard.npz'
datafile = np.load(in_file)
# keys: data, flats, darks, angles, angles_total, detector_y, detector_x
host_data = datafile['data']
host_flats = datafile['flats']
host_darks = datafile['darks']


def test_find_center_vo_cupy():
    data = cp.asarray(host_data)
    flats = cp.asarray(host_flats)
    darks = cp.asarray(host_darks)
    data = normalize_cupy(data, flats, darks)

    #--- testing the center of rotation on tomo_standard ---#
    cor = find_center_vo_cupy(data)
    for _ in range(10):
        assert_allclose(cor, 79.5)

    # free up GPU memory by no longer referencing the variables
    data, flats, darks = None, None, None
    cp._default_memory_pool.free_all_blocks()


def test_find_center_360():
    mat = np.ones(shape=(100, 100, 100))
    (cor, overlap, side, overlap_position) = find_center_360(mat[:, 2, :])
    for _ in range(5):
        assert_allclose(cor, 5.0)
        assert_allclose(overlap, 12.0)
        assert side == 0
        assert_allclose(overlap_position, 7.0)

    eps = 1e-5
    (cor, overlap, side, overlap_pos) = find_center_360(host_data[:, 10, :])
    assert_allclose(cor, 118.56627, rtol=eps)
    assert_allclose(overlap, 80.86746, rtol=eps)
    assert side == 1
    assert_allclose(overlap_pos, 84.13254, rtol=eps)

    #: Check that we only get a float32 output
    assert cor.dtype == np.float32
    assert overlap.dtype == np.float32

    #: 360-degree sinogram must be a 2d array
    pytest.raises(ValueError, lambda: find_center_360(host_data[:, 10, 10]))
    pytest.raises(ValueError, lambda: find_center_360(np.ones(10)))
