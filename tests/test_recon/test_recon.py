import time
import cupy as cp
import nvtx
import numpy as np
import pytest
from httomolib.prep.normalize import normalize_cupy
from httomolib.recon.rotation import find_center_360, find_center_vo_cupy, find_center_360_cupy
from numpy.testing import assert_allclose


@cp.testing.gpu
def test_find_center_vo_cupy(data, flats, darks):
    data = normalize_cupy(data, flats, darks)

    #--- testing the center of rotation on tomo_standard ---#
    cor = find_center_vo_cupy(data).get()

    data = None #: free up GPU memory
    assert_allclose(cor, 79.5)

    #: Check that we only get a float32 output
    assert cor.dtype == np.float32


@cp.testing.gpu
def test_find_center_vo_cupy_ones(ensure_clean_memory):
    mat = cp.ones(shape=(103, 450, 230))
    cor = find_center_vo_cupy(mat).get()

    assert_allclose(cor, 59.0)
    mat = None #: free up GPU memory


def test_find_center_360_ones(gpu):
    mat = np.ones(shape=(100, 100, 100))
    if gpu:
        (cor, overlap, side, overlap_position) = find_center_360_cupy(cp.asarray(mat))
    else:
        (cor, overlap, side, overlap_position) = find_center_360(mat)
    

    assert_allclose(cor, 5.0)
    assert_allclose(overlap, 12.0)
    assert side == 0
    assert_allclose(overlap_position, 7.0)



def test_find_center_360_data(host_data, gpu):
    eps = 1e-5
    if gpu:
        (cor, overlap, side, overlap_pos) = find_center_360_cupy(cp.asarray(host_data))
    else:
        (cor, overlap, side, overlap_pos) = find_center_360(host_data)
    
    assert_allclose(cor, 132.45317, rtol=eps)
    assert_allclose(overlap, 53.093666, rtol=eps)
    assert side == 1
    assert_allclose(overlap_pos, 111.906334, rtol=eps)

    #: Check that we only get a float32 output
    assert cor.dtype == np.float32
    assert overlap.dtype == np.float32


@cp.testing.gpu
def test_find_center_360_unity(ensure_clean_memory, have_gpu):
    eps = 1e-5

    host_data = np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    data = cp.asarray(host_data)

    (cor, overlap, side, overlap_pos) = find_center_360(host_data)
    (cor_gpu, overlap_gpu, side_gpu, overlap_pos_gpu) = find_center_360_cupy(data)

    np.testing.assert_allclose(cor, cor_gpu, rtol=eps)
    np.testing.assert_allclose(overlap, overlap_gpu, rtol=eps)
    np.testing.assert_allclose(side, side_gpu, rtol=eps)
    np.testing.assert_allclose(overlap_pos, overlap_pos_gpu, rtol=eps)


@pytest.mark.perf
def test_find_center_360_performance(gpu):
    host_data = np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001

    if gpu:
        data = cp.asarray(host_data)
        # cold run
        find_center_360_cupy(data)

        dev = cp.cuda.Device()
        dev.synchronize()

        start = time.perf_counter_ns()
        nvtx.RangePush("Core")
        for _ in range(10):
            # have to take copy, as data is modified in-place
            find_center_360_cupy(data)
        nvtx.RangePop()
        dev.synchronize()
        
        duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10
    else:
        # cold run
        find_center_360(host_data)

        start = time.perf_counter_ns()
        for _ in range(10):
            # have to take copy, as data is modified in-place
            find_center_360(host_data)
        
        duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10
    
    
    assert "performance in ms" == duration_ms



def test_find_center_360_1D_raises(host_data, gpu):
    #: 360-degree sinogram must be a 3d array

    with pytest.raises(ValueError):
        if gpu:
            find_center_360_cupy(cp.asarray(host_data[:, 10, :]))
        else:
            find_center_360(host_data[:, 10, :])
    
    with pytest.raises(ValueError):
        if gpu:
            find_center_360_cupy(cp.ones(10))
        else:
            find_center_360(np.ones(10))
        
