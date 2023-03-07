import time
import cupy as cp
from cupy.cuda import nvtx
import numpy as np
import pytest
from httomolib.prep.normalize import normalize_cupy
from httomolib.recon.rotation import find_center_360, find_center_vo_cupy
from numpy.testing import assert_allclose
from .rotation_cpu_reference import find_center_360_numpy


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
    mat = cp.ones(shape=(103, 450, 230), dtype=cp.float32)
    cor = find_center_vo_cupy(mat).get()

    assert_allclose(cor, 59.0)
    mat = None #: free up GPU memory


@cp.testing.gpu
@pytest.mark.perf
def test_find_center_vo_cupy_performance():
    dev = cp.cuda.Device()
    data_host = np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)
    
    # cold run first
    find_center_vo_cupy(data)

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        find_center_vo_cupy(data)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


@cp.testing.gpu
def test_find_center_360_ones():
    mat = cp.ones(shape=(100, 100, 100), dtype=cp.float32)
    
    (cor, overlap, side, overlap_position) = find_center_360(mat)

    assert_allclose(cor, 5.0)
    assert_allclose(overlap, 12.0)
    assert side == 0
    assert_allclose(overlap_position, 7.0)


@cp.testing.gpu
def test_find_center_360_data(data):
    eps = 1e-5
    (cor, overlap, side, overlap_pos) = find_center_360(data, norm=True, denoise=False)
    
    assert_allclose(cor, 132.45317, rtol=eps)
    assert_allclose(overlap, 53.093666, rtol=eps)
    assert side == 1
    assert_allclose(overlap_pos, 111.906334, rtol=eps)

    #: Check that we only get a float32 output
    assert cor.dtype == np.float32
    assert overlap.dtype == np.float32


@cp.testing.gpu
@pytest.mark.parametrize("norm", [False, True], ids=["no_normalise", "normalise"])
@pytest.mark.parametrize("overlap", [False, True], ids=["no_overlap", "overlap"])
@pytest.mark.parametrize("denoise", [False, True], ids=["no_denoise", "denoise"])
@pytest.mark.parametrize("side", [1, 0])
@cp.testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-6)
def test_find_center_360_unity(ensure_clean_memory, xp, norm, overlap, denoise, side):
    # because it's random, we explicitly seed and use numpy only, to match the data
    np.random.seed(12345)
    data = np.random.random_sample(size=(128, 1, 512)).astype(np.float32) * 2.0 + 0.001
    data = xp.asarray(data)

    if xp.__name__ == "numpy":
        (cor, overlap, side, overlap_pos) = find_center_360_numpy(data, use_overlap=overlap, 
                                                            norm=norm, denoise=denoise, 
                                                            side=side)
    else:
        (cor, overlap, side, overlap_pos) = find_center_360(data, use_overlap=overlap, 
                                                            norm=norm, denoise=denoise, 
                                                            side=side)

    return xp.asarray([cor, overlap, side, overlap_pos])
    


@pytest.mark.perf
def test_find_center_360_performance(ensure_clean_memory):
    data = cp.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001

    # cold run
    find_center_360(data, use_overlap=True, norm=True, denoise=True)

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        # have to take copy, as data is modified in-place
        find_center_360(data, use_overlap=True, norm=True, denoise=True)
    nvtx.RangePop()
    dev.synchronize()
    
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10
    
    assert "performance in ms" == duration_ms


@cp.testing.gpu
def test_find_center_360_1D_raises(data):
    
    #: 360-degree sinogram must be a 3d array
    with pytest.raises(ValueError):
        find_center_360(data[:, 10, :])
    
    with pytest.raises(ValueError):
        find_center_360(cp.ones(10))
        
