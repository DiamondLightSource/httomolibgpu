import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose

from httomolib.prep.normalize import normalize_cupy, normalize_raw_cuda


def test_normalize():
    # testing cupy implementation for normalization

    in_file = 'tests/test_data/tomo_standard.npz'
    datafile = np.load(in_file)
    # keys: data, flats, darks, angles, angles_total, detector_y, detector_x
    host_data = datafile['data']
    host_flats = datafile['flats']
    host_darks = datafile['darks']

    data = cp.asarray(host_data)
    flats = cp.asarray(host_flats)
    darks = cp.asarray(host_darks)

    data_min = cp.array(-0.16163824, dtype=cp.float32)
    data_max = cp.array(2.7530956, dtype=cp.float32)

    _data_1d = cp.ones(10)
    #: data cannot be a 1D array
    pytest.raises(ValueError, lambda: normalize_cupy(_data_1d, flats, darks))
    #: flats cannot be a 1D array
    pytest.raises(ValueError, lambda: normalize_cupy(data, _data_1d, darks))

    #--- testing normalize_cupy  ---#
    data_normalize_cupy = normalize_cupy(data, flats, darks, cutoff=10, minus_log=True)
    for _ in range(10):
        assert_allclose(cp.min(data_normalize_cupy), data_min, rtol=1e-06)
        assert_allclose(cp.max(data_normalize_cupy), data_max, rtol=1e-06)

    #--- testing normalize_raw_cuda  ---#
    # TODO: fix the normalize_raw_cuda test failing at max assert
    #data_normalize_raw_cuda = normalize_raw_cuda(data, flats, darks, cutoff=10, minus_log=True)
    #for _ in range(10):
        #assert_allclose(cp.min(data_normalize_raw_cuda), data_min, rtol=1e-05)
        #assert_allclose(cp.max(data_normalize_raw_cuda), data_max, rtol=1e-06)

    #: free up GPU memory by no longer referencing the variables
    data_normalize_cupy = flats = darks = data_min = data_max = _data_1d = None
    cp._default_memory_pool.free_all_blocks()
