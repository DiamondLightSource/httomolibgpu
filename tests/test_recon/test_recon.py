import time
import cupy as cp
from cupy.cuda import nvtx
import numpy as np
import pytest
from httomolib.prep.normalize import normalize_cupy
from httomolib.recon.rotation import find_center_360, find_center_vo_cupy
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
    xp = cp if gpu else np
    mat = xp.ones(shape=(100, 100, 100), dtype=xp.float32)
    
    (cor, overlap, side, overlap_position) = find_center_360(mat)

    assert_allclose(cor, 5.0)
    assert_allclose(overlap, 12.0)
    assert side == 0
    assert_allclose(overlap_position, 7.0)



def test_find_center_360_data(host_data, gpu):
    eps = 1e-5
    data = cp.asarray(host_data) if gpu else host_data
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
@cp.testing.numpy_cupy_allclose(rtol=1e-5)
def test_find_center_360_unity(ensure_clean_memory, xp, norm, overlap, denoise, side):
    eps = 1e-5

    # because it's random, we explicitly seed and use numpy only, to match the data
    np.random.seed(12345)
    data = np.random.random_sample(size=(128, 1, 512)).astype(np.float32) * 2.0 + 0.001
    data = xp.asarray(data)
    
    (cor, overlap, side, overlap_pos) = find_center_360(data, use_overlap=overlap, 
                                                        norm=norm, denoise=denoise, 
                                                        side=side)

    return xp.asarray([cor, overlap, side, overlap_pos])
    


@pytest.mark.perf
def test_find_center_360_performance(gpu, ensure_clean_memory):
    xp = cp if gpu else np
    data = xp.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001

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



def test_find_center_360_1D_raises(host_data, gpu):
    data = cp.asarray(host_data) if gpu else host_data
    xp = cp if gpu else np
    
    #: 360-degree sinogram must be a 3d array
    with pytest.raises(ValueError):
        find_center_360(data[:, 10, :])
    
    with pytest.raises(ValueError):
        find_center_360(xp.ones(10))
        
