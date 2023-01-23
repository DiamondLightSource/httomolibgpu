import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose

from httomolib.prep.phase import (
    fresnel_filter,
    paganin_filter,
    retrieve_phase,
)

in_file = 'tests/test_data/tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']
data = cp.array(host_data)
eps = 1e-6


def test_fresnel_filter():
    # --- testing the Fresnel filter on tomo_standard ---#
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

    _data = cp.ones(10)
    pytest.raises(ValueError, lambda: fresnel_filter(_data, pattern, ratio))

    # free up GPU memory by no longer referencing the variable
    filtered_data = _data = None
    cp._default_memory_pool.free_all_blocks()


def test_paganin_filter():
    # --- testing the Paganin filter on tomo_standard ---#
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


@cp.testing.gpu
def test_retrieve_phase():
    #: testing the phase retrieval on tomo_standard
    _data = cp.ones(10)
    pytest.raises(ValueError, lambda: retrieve_phase(_data))

    phase_data = retrieve_phase(data)
    assert phase_data.shape == (180, 128, 160)
    assert cp.sum(phase_data) == 2994544952
    assert_allclose(cp.mean(phase_data), 812.3223068576389, rtol=1e-7)

    phase_data = retrieve_phase(data, dist=34.3, energy=100.0, pad=False)
    for _ in range(3):
        assert_allclose(cp.mean(phase_data), 978.7444444444444, rtol=1e-7)
        assert_allclose(cp.std(phase_data), 8.995135859774523, rtol=1e-7)

    # free up GPU memory by no longer referencing the variable
    phase_data, _data = None, None
    cp._default_memory_pool.free_all_blocks()
