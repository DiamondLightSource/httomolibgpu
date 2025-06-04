import math
import random
import time
from unittest import mock
import cupy as cp
from cupy.cuda import nvtx
from cupyx.scipy.ndimage import shift
import numpy as np
import pytest
from httomolibgpu.prep.normalize import normalize
from httomolibgpu.recon.rotation import (
    _calculate_chunks,
    find_center_360,
    find_center_vo,
    find_center_pc,
)
from numpy.testing import assert_allclose
from .rotation_cpu_reference import find_center_360_numpy


def test_find_center_vo(data, flats, darks):
    data = normalize(data, flats, darks)

    # --- testing the center of rotation on tomo_standard ---#
    cor = find_center_vo(
        data.copy(),
        average_radius=0,
    )

    data = None  #: free up GPU memory
    assert_allclose(cor, 79.5)


def test_find_center_vo_ones(ensure_clean_memory):
    mat = cp.ones(shape=(103, 450, 230), dtype=cp.float32)
    cor = find_center_vo(mat)

    assert_allclose(cor, 58)
    mat = None  #: free up GPU memory


def test_find_center_vo_random(ensure_clean_memory):
    np.random.seed(12345)
    data_host = np.random.random_sample(size=(900, 1, 1280)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)
    cent = find_center_vo(data)
    assert_allclose(cent, 680.5)


def test_find_center_vo_calculate_chunks():
    # we need the split to fit into the available memory, and also make sure
    # that the last chunk is either the same or smaller than the previous ones
    # (so that we can re-use the same memory as for the previous chunks, incl. FFT plan)
    # Note: With shift_size = 100 bytes, we need 600 bytes per shift
    bytes_per_shift = 600
    assert _calculate_chunks(10, 100, 1000000) == [10]
    assert _calculate_chunks(10, 100, 10 * bytes_per_shift + 100) == [4, 8, 10]
    assert _calculate_chunks(10, 100, 5 * bytes_per_shift + 100) == [2, 4, 6, 8, 10]
    assert _calculate_chunks(10, 100, 7 * bytes_per_shift + 100) == [3, 6, 9, 10]
    assert _calculate_chunks(10, 100, 9 * bytes_per_shift + 100) == [4, 8, 10]
    assert _calculate_chunks(9, 100, 5 * bytes_per_shift + 100) == [2, 4, 6, 8, 9]
    assert _calculate_chunks(10, 100, 4 * bytes_per_shift + 100) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]


@pytest.mark.perf
def test_find_center_vo_performance():
    dev = cp.cuda.Device()
    data_host = np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)

    # cold run first
    find_center_vo(data)

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        find_center_vo(data)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_find_center_360_data(data):
    eps = 1e-5
    (cor, overlap, side, overlap_pos) = find_center_360(data, norm=True, denoise=False)

    assert_allclose(cor, 132.45317, rtol=eps)
    assert_allclose(overlap, 53.093666, rtol=eps)
    assert side == 1
    assert_allclose(overlap_pos, 111.906334, rtol=eps)


def test_find_center_360_1D_raises(data):
    #: 360-degree sinogram must be a 3d array
    with pytest.raises(ValueError):
        find_center_360(data[:, 10, :])

    with pytest.raises(ValueError):
        find_center_360(cp.ones(10))


def test_find_center_360_NaN_infs_raises(data, flats, darks):
    #: find_center_360 raises if the data with NaNs or Infs given
    data = data.astype(cp.float32)
    data[:] = cp.inf
    with pytest.raises(ValueError):
        find_center_360(data)
    data[:] = cp.nan
    with pytest.raises(ValueError):
        find_center_360(data)


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
        (cor, overlap, side, overlap_pos) = find_center_360_numpy(
            data, use_overlap=overlap, norm=norm, denoise=denoise, side=side
        )
    else:
        (cor, overlap, side, overlap_pos) = find_center_360(
            data, use_overlap=overlap, norm=norm, denoise=denoise, side=side
        )

    return xp.asarray([cor, overlap, side, overlap_pos])


@pytest.mark.perf
def test_find_center_360_performance(ensure_clean_memory):
    data = (
        cp.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    )

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


def test_find_center_pc(data, flats, darks, ensure_clean_memory):
    data = normalize(data, flats, darks)[:, :, 3]
    shifted_data = shift(cp.fliplr(data), (0, 18.75), mode="reflect")

    # --- testing the center of rotation on tomo_standard ---#
    cor = find_center_pc(data, shifted_data)

    assert_allclose(cor, 73.0, rtol=1e-7)


def test_find_center_pc2(data, flats, darks, ensure_clean_memory):
    data = normalize(data, flats, darks)
    proj1 = data[0, :, :]
    proj2 = data[179, :, :]

    # --- testing the center of rotation on tomo_standard ---#
    cor = find_center_pc(proj1, proj2)

    assert_allclose(cor, 79.5, rtol=1e-7)


@pytest.mark.perf
def test_find_center_pc_performance(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.random_sample(size=(1801, 2560, 5)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)[:, :, 3]
    shifted_data = shift(cp.fliplr(data), (0, 18.75), mode="reflect")

    # cold run first
    find_center_pc(data, shifted_data)

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        find_center_pc(data, shifted_data)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
