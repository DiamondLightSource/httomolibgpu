import cupy as cp
import numpy as np
from cupy.testing import assert_allclose

from httomolib.prep.normalize import normalize_raw_cuda, normalize_cupy


def test_normalize_raw_cuda():
    # testing cupy implementation for normalization

    in_file = 'data/tomo_standard.npz'
    datafile = np.load(in_file) #keys: data, flats, darks, angles, angles_total, detector_y, detector_x
    host_data = datafile['data']
    host_flats = datafile['flats']
    host_darks = datafile['darks']
    
    data = cp.asarray(host_data)
    flats = cp.asarray(host_flats)
    darks = cp.asarray(host_darks)
    data = normalize_raw_cuda(data, flats, darks)

    data_min = cp.array(-0.16163824, dtype=cp.float32)
    data_max = cp.array(2.7530956, dtype=cp.float32)

    for _ in range(10):
        assert_allclose(cp.min(data), data_min, rtol=1e-05)
        assert_allclose(cp.max(data), data_max, rtol=1e-05)

    # free up GPU memory by no longer referencing the variables
    data, flats, darks, data_min, data_max = None, None, None, None, None
    cp._default_memory_pool.free_all_blocks()

def test_normalize_cupy():
    # testing cupy implementation for normalization

    in_file = 'data/tomo_standard.npz'
    datafile = np.load(in_file) #keys: data, flats, darks, angles, angles_total, detector_y, detector_x
    host_data = datafile['data']
    host_flats = datafile['flats']
    host_darks = datafile['darks']
    
    data = cp.asarray(host_data)
    flats = cp.asarray(host_flats)
    darks = cp.asarray(host_darks)
    data = normalize_cupy(data, flats, darks)

    data_min = cp.array(-0.16163824, dtype=cp.float32)
    data_max = cp.array(2.7530956, dtype=cp.float32)

    for _ in range(10):
        assert_allclose(cp.min(data), data_min, rtol=1e-05)
        assert_allclose(cp.max(data), data_max, rtol=1e-05)

    # free up GPU memory by no longer referencing the variables
    data, flats, darks, data_min, data_max = None, None, None, None, None
    cp._default_memory_pool.free_all_blocks()
