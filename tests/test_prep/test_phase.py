import time

import cupy as cp
import numpy as np
import pytest
from cupy.cuda import nvtx
from httomolibgpu.prep.phase import paganin_filter_savu, paganin_filter_tomopy
from numpy.testing import assert_allclose

eps = 1e-6

@cp.testing.gpu
def test_paganin_savu_filter(data):
    # --- testing the Paganin filter on tomo_standard ---#
    filtered_data = paganin_filter_savu(data).get()
    
    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), -770.5339, rtol=eps)
    assert_allclose(np.max(filtered_data), -679.80945, rtol=eps)

    #: make sure the output is float32
    assert filtered_data.dtype == np.float32

@cp.testing.gpu
def test_paganin_filter_savu_energy100(data):
    filtered_data = paganin_filter_savu(data, energy=100.0).get()

    assert_allclose(np.mean(filtered_data), -778.61926, rtol=1e-05)
    assert_allclose(np.min(filtered_data), -808.9013, rtol=eps)

    assert filtered_data.ndim == 3
    assert filtered_data.dtype == np.float32


@cp.testing.gpu
def test_paganin_filter_savu_padmean(data):
    filtered_data = paganin_filter_savu(data, pad_method="mean").get()

    assert_allclose(np.mean(filtered_data), -765.3401, rtol=eps)
    assert_allclose(np.min(filtered_data), -793.68787, rtol=eps)
    # test a few other slices to ensure shifting etc is right
    assert_allclose(
        filtered_data[0, 50, 1:5],
        [-785.60736, -786.20215, -786.7521, -787.25494],
        rtol=eps,
    )
    assert_allclose(
        filtered_data[0, 50, 40:42], [-776.6436, -775.1906], rtol=eps, atol=1e-5
    )
    assert_allclose(
        filtered_data[0, 60:63, 90],
        [-737.75104, -736.6097, -735.49884],
        rtol=eps,
        atol=1e-5,
    )


@cp.testing.gpu
@pytest.mark.perf
def test_paganin_filter_savu_performance(ensure_clean_memory):
    # Note: low/high and size values taken from sample2_medium.yaml real run

    # this test needs ~20GB of memory with 1801 - we'll divide depending on GPU memory
    dev = cp.cuda.Device()
    mem_80percent = 0.8 * dev.mem_info[0]
    size = 1801
    required_mem = 20 * 1024 * 1024 * 1024
    if mem_80percent < required_mem:
        size = int(np.ceil(size / required_mem * mem_80percent))
        print(f"Using smaller size of ({size}, 5, 2560) due to memory restrictions")

    data_host = np.random.random_sample(size=(size, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)

    # run code and time it
    # cold run first
    paganin_filter_savu(
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
        paganin_filter_savu(
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
def test_paganin_filter_savu_1D_raises(ensure_clean_memory):
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        paganin_filter_savu(_data)

    _data = None  #: free up GPU memory

# paganin filter tomopy
@cp.testing.gpu
def test_paganin_filter_tomopy_1D_raises(ensure_clean_memory):
    _data = cp.ones(10)
    with pytest.raises(ValueError):
        paganin_filter_tomopy(_data)

    _data = None  #: free up GPU memory

@cp.testing.gpu
def test_paganin_filter_tomopy(data):
    # --- testing the Paganin filter from TomoPy on tomo_standard ---#
    filtered_data = paganin_filter_tomopy(data).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), -6.74213, rtol=eps)
    assert_allclose(np.max(filtered_data), -6.496699, rtol=eps)

    #: make sure the output is float32
    assert filtered_data.dtype == np.float32


@cp.testing.gpu
def test_paganin_filter_tomopy_energy100(data):
    filtered_data = paganin_filter_tomopy(data, energy=100.0).get()

    assert_allclose(np.mean(filtered_data), -6.73455, rtol=1e-05)
    assert_allclose(np.min(filtered_data), -6.909582, rtol=eps)

    assert filtered_data.ndim == 3
    assert filtered_data.dtype == np.float32


@cp.testing.gpu
def test_paganin_filter_tomopy_dist75(data):
    filtered_data = paganin_filter_tomopy(data, dist=75.0, alpha=1e-6).get()

    assert_allclose(np.sum(np.mean(filtered_data, axis=(1, 2))), -1215.4985, rtol=1e-6)
    assert_allclose(np.sum(filtered_data), -24893412., rtol=1e-6)
    assert_allclose(np.mean(filtered_data[0, 60:63, 90]), -6.645878, rtol=1e-6)
    assert_allclose(np.sum(filtered_data[50:100, 40, 1]), -343.5908, rtol=1e-6)


@cp.testing.gpu
@pytest.mark.perf
def test_paganin_filter_tomopy_performance(ensure_clean_memory):
    # Note: low/high and size values taken from sample2_medium.yaml real run

    # this test needs ~20GB of memory with 1801 - we'll divide depending on GPU memory
    dev = cp.cuda.Device()
    mem_80percent = 0.8 * dev.mem_info[0]
    size = 1801
    required_mem = 20 * 1024 * 1024 * 1024
    if mem_80percent < required_mem:
        size = int(np.ceil(size / required_mem * mem_80percent))
        print(f"Using smaller size of ({size}, 5, 2560) due to memory restrictions")

    data_host = np.random.random_sample(size=(size, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)

    # run code and time it
    # cold run first
    paganin_filter_tomopy(data)
    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        paganin_filter_tomopy(data)

    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms