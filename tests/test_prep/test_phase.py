import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose

from httomolib.prep.phase import fresnel_filter, paganin_filter

in_file = 'tests/test_data/tomo_standard.npz'
datafile = np.load(in_file) #keys: data, flats, darks, angles, angles_total, detector_y, detector_x
host_data = datafile['data']
data = cp.array(host_data)
eps = 1e-6

def test_fresnel_filter():
    #--- testing the Fresnel filter on tomo_standard ---#
    pattern = 'PROJECTION'
    ratio = 100.0
    filtered_data = fresnel_filter(data, pattern, ratio)

    for _ in range(5):
        assert_allclose(cp.mean(filtered_data), 802.1125)
        assert_allclose(cp.max(filtered_data), 1039.5293)
        assert_allclose(cp.min(filtered_data), 95.74562)

    pattern = 'SINOGRAM'
    filtered_data = fresnel_filter(data, pattern, ratio)
    for _ in range(5):
        assert_allclose(cp.mean(filtered_data), 806.74347, rtol=eps)
        assert_allclose(cp.max(filtered_data), 1063.7007)
        assert_allclose(cp.min(filtered_data), 87.91508)

    # free up GPU memory by no longer referencing the variable
    filtered_data = None
    cp._default_memory_pool.free_all_blocks()


def test_paganin_filter():
    #--- testing the Paganin filter on tomo_standard ---#
    filtered_data = paganin_filter(data)

    for _ in range(5):
        assert filtered_data.ndim == 3
        assert_allclose(cp.mean(filtered_data), -770.5339, rtol=eps)
        assert_allclose(cp.max(filtered_data), -679.80945, rtol=eps)

    filtered_data = paganin_filter(data, energy=100.0)

    assert_allclose(cp.mean(filtered_data), -778.61926, rtol=1e-05)
    assert_allclose(cp.min(filtered_data), -808.9013, rtol=eps)

    filtered_data = paganin_filter(data, pad_method='mean')

    assert_allclose(cp.mean(filtered_data), -765.3401, rtol=eps)
    assert_allclose(cp.min(filtered_data), -793.68787, rtol=eps)

    _data = cp.ones(10)
    pytest.raises(ValueError, lambda: paganin_filter(_data))

    # free up GPU memory by no longer referencing the variable
    filtered_data, _data = None, None
    cp._default_memory_pool.free_all_blocks()
