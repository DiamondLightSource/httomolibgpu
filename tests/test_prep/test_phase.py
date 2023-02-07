import time
import cupy as cp
from cupy.cuda import nvtx
import numpy as np
import pytest
from httomolib.prep.phase import fresnel_filter, paganin_filter, retrieve_phase
from numpy.testing import assert_allclose

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

    _data = None #: free up GPU memory


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
    # test a few other slices to ensure shifting etc is right
    assert_allclose(
        filtered_data[0, 50, 1:5],
        [-785.60736, -786.20215, -786.7521, -787.25494],
        rtol=eps,
    )
    assert_allclose(filtered_data[0, 50, 40:42], 
        [-776.6436, -775.1906], rtol=eps, atol=1e-5)
    assert_allclose(filtered_data[0, 60:63, 90],
        [-737.75104, -736.6097, -735.49884], rtol=eps, atol=1e-5)

@cp.testing.gpu
@pytest.mark.perf
def test_paganin_filter_performance(ensure_clean_memory):
    # Note: low/high and size values taken from sample2_medium.yaml real run
    
    # this test needs ~20GB of memory with 1801 - we'll divide depending on GPU memory
    dev = cp.cuda.Device()
    mem_80percent = 0.8 * dev.mem_info[0]
    size = 1801
    required_mem = 40 * 1024*1024*1024
    if mem_80percent < required_mem:
        size = int(np.ceil(size / required_mem * mem_80percent))
        print(f'Using smaller size of ({size}, 5, 2560) due to memory restrictions')

    data_host = np.random.random_sample(size=(size, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)

    # run code and time it
    # cold run first
    paganin_filter(
        data,
        ratio=250.0,
        energy=53.0,
        distance=1.0,
        resolution=1.28,
        pad_y=100,
        pad_x=100,
        pad_method="edge",
        increment=0.0,
    )
    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        paganin_filter(
            data,
            ratio=250.0,
            energy=53.0,
            distance=1.0,
            resolution=1.28,
            pad_y=100,
            pad_x=100,
            pad_method="edge",
            increment=0.0,
        )
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


@cp.testing.gpu
def test_paganin_filter_1D_raises(ensure_clean_memory):
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        paganin_filter(_data)

    _data = None #: free up GPU memory


@cp.testing.gpu
def test_retrieve_phase_1D_raises(ensure_clean_memory):
    #: testing the phase retrieval on tomo_standard
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        retrieve_phase(_data)

    _data = None #: free up GPU memory


@cp.testing.gpu
def test_retrieve_phase(data):
    phase_data = retrieve_phase(data).get()
    assert phase_data.shape == (180, 128, 160)
    assert np.sum(phase_data) == 2994544952
    assert_allclose(np.mean(phase_data), 812.3223068576389, rtol=1e-7)


@cp.testing.gpu
def test_retrieve_phase_energy100_nopad(data):
    phase_data = retrieve_phase(data, dist=34.3, energy=100.0, pad=False).get()

    assert_allclose(np.mean(phase_data), 979.527778, rtol=1e-7)
    assert_allclose(np.std(phase_data), 30.053735, rtol=1e-7)
