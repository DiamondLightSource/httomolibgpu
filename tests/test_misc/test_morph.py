import time
import cupy as cp
import numpy as np
from cupy.cuda import nvtx
import pytest
from numpy.testing import assert_allclose
from httomolibgpu.misc.morph import sino_360_to_180, data_resampler


@pytest.mark.parametrize(
    "overlap, rotation",
    [
        (-10, "left"),
        (110, "left"),
        (-10, "right"),
        (110, "right"),
        (0, "invalid"),
        (0, ""),
    ],
)
def test_sino_360_to_180_invalid(ensure_clean_memory, overlap, rotation):
    data = cp.ones((10, 10, 10), dtype=cp.float32)

    with pytest.raises(ValueError):
        sino_360_to_180(data, overlap, rotation)


@pytest.mark.parametrize("shape", [(10,), (10, 10)])
def test_sino_360_to_180_wrong_dims(ensure_clean_memory, shape):
    with pytest.raises(ValueError):
        sino_360_to_180(cp.ones(shape, dtype=cp.float32))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_data_resampler(data, axis, ensure_clean_memory):
    newshape = [60, 80]
    scaled_data = data_resampler(
        data, newshape=newshape, axis=axis, interpolation="linear"
    ).get()

    assert scaled_data.ndim == 3
    if axis == 0:
        assert scaled_data.shape == (180, newshape[0], newshape[1])
        assert_allclose(np.max(scaled_data), 1111.7404)
    if axis == 1:
        assert scaled_data.shape == (newshape[0], 128, newshape[1])
        assert_allclose(np.max(scaled_data), 1102.0)
    if axis == 2:
        assert scaled_data.shape == (newshape[0], newshape[1], 160)
        assert_allclose(np.max(scaled_data), 1113.2761)
    assert scaled_data.dtype == np.float32
    assert scaled_data.flags.c_contiguous


@pytest.mark.parametrize("rotation", ["left", "right"])
@pytest.mark.perf
def test_sino_360_to_180_performance(ensure_clean_memory, rotation):
    # as this is just an index shuffling operation, the data itself doesn't matter
    data = cp.ones((1801, 400, 2560), dtype=np.float32)

    # do a cold run first
    sino_360_to_180(data, overlap=32, rotation=rotation)

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        sino_360_to_180(data, overlap=32, rotation=rotation)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
