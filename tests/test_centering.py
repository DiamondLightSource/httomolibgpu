import cupy as cp
import numpy as np
from cupy.testing import assert_allclose

from httomolib.recon.rotation import find_center_vo_cupy
from httomolib.prep.normalize import normalize_cupy

def test_find_center_of_rotation():
    in_file = 'data/tomo_standard.npz'
    datafile = np.load(in_file) #keys: data, flats, darks, angles, angles_total, detector_y, detector_x
    host_data = datafile['data']
    host_flats = datafile['flats']
    host_darks = datafile['darks']
    
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

