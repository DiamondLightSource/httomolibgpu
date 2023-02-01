import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from httomolib.prep.phase import (
    fresnel_filter,
    paganin_filter,
    retrieve_phase,
)

eps = 1e-6


@cp.testing.gpu
def test_fresnel_filter_projection(data):
    # --- testing the Fresnel filter on tomo_standard ---#
    pattern = "PROJECTION"
    ratio = 100.0
    filtered_data = fresnel_filter(data, pattern, ratio).get()

    assert_allclose(np.mean(filtered_data), 802.1125, rtol=eps)
    assert_allclose(np.max(filtered_data), 1039.5293)
    assert_allclose(np.min(filtered_data), 95.74562)


@cp.testing.gpu
def test_fresnel_filter_sinogram(data):
    pattern = "SINOGRAM"
    ratio = 100.0
    filtered_data = fresnel_filter(data, pattern, ratio).get()

    assert_allclose(np.mean(filtered_data), 806.74347, rtol=eps)
    assert_allclose(np.max(filtered_data), 1063.7007)
    assert_allclose(np.min(filtered_data), 87.91508)


@cp.testing.gpu
def test_fresnel_filter_1D_raises(ensure_clean_memory):
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        fresnel_filter(_data, "SINOGRAM", 100.0)


@cp.testing.gpu
def test_paganin_filter(data):
    # --- testing the Paganin filter on tomo_standard ---#
    filtered_data = paganin_filter(data).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), -770.5339, rtol=eps)
    assert_allclose(np.max(filtered_data), -679.80945, rtol=eps)


@cp.testing.gpu
def test_paganin_filter_energy100(data):
    filtered_data = paganin_filter(data, energy=100.0).get()

    assert_allclose(np.mean(filtered_data), -778.61926, rtol=1e-05)
    assert_allclose(np.min(filtered_data), -808.9013, rtol=eps)


@cp.testing.gpu
def test_paganin_filter_padmean(data):
    filtered_data = paganin_filter(data, pad_method="mean").get()

    assert_allclose(np.mean(filtered_data), -765.3401, rtol=eps)
    assert_allclose(np.min(filtered_data), -793.68787, rtol=eps)


@cp.testing.gpu
def test_paganin_filter_1D_raises(ensure_clean_memory):
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        paganin_filter(_data)


@cp.testing.gpu
def test_retrieve_phase_1D_raises(ensure_clean_memory):
    #: testing the phase retrieval on tomo_standard
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        retrieve_phase(_data)


@cp.testing.gpu
def test_retrieve_phase(data):
    phase_data = retrieve_phase(data).get()
    assert phase_data.shape == (180, 128, 160)
    assert np.sum(phase_data) == 2994544952
    assert_allclose(np.mean(phase_data), 812.3223068576389, rtol=1e-7)


@cp.testing.gpu
def test_retrieve_phase_energy100_nopad(data):
    # TODO: retrieve_phase modifies data in-place, and reference data was generated
    # from calling it without params first. The reference should be re-based
    retrieve_phase(data)
    phase_data = retrieve_phase(data, dist=34.3, energy=100.0, pad=False).get()

    assert_allclose(np.mean(phase_data), 978.7444444444444, rtol=1e-7)
    assert_allclose(np.std(phase_data), 8.995135859774523, rtol=1e-7)
