import time
import cupy as cp
import numpy as np
from cupy.cuda import nvtx
import pytest
from tomopy.misc.morph import sino_360_to_180 as tomopy_sino_360_to_180
from httomolib.misc.morph import sino_360_to_180


@cp.testing.gpu
@pytest.mark.parametrize("overlap", [0, 1, 3, 15, 32])
@pytest.mark.parametrize("rotation", ["left", "right"])
@cp.testing.numpy_cupy_allclose(rtol=1e-6)
def test_sino_360_to_180_unity(ensure_clean_memory, xp, overlap, rotation):
    # this combination has a bug in tomopy, so we'll skip it for now
    if rotation == "right" and overlap == 0:
        pytest.skip("Skipping test due to bug in tomopy")

    np.random.seed(12345)
    data_host = (
        np.random.random_sample(size=(123, 54, 128)).astype(np.float32) * 200.0 - 100.0
    )
    data = xp.asarray(data_host)

    if xp.__name__ == "numpy":
        return tomopy_sino_360_to_180(data, overlap, rotation)
    else:
        return sino_360_to_180(data, overlap, rotation)


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
@cp.testing.gpu
def test_sino_360_to_180_invalid(ensure_clean_memory, overlap, rotation):
    data = cp.ones((10, 10, 10), dtype=cp.float32)

    with pytest.raises(ValueError):
        sino_360_to_180(data, overlap, rotation)


@pytest.mark.parametrize("shape", [(10,), (10, 10)])
@cp.testing.gpu
def test_sino_360_to_180_wrong_dims(ensure_clean_memory, shape):
    with pytest.raises(ValueError):
        sino_360_to_180(cp.ones(shape, dtype=cp.float32))


def test_sino_360_to_180_meta():
    assert sino_360_to_180.meta.gpu is True
    assert sino_360_to_180.meta.pattern == 'sinogram'


@pytest.mark.parametrize("rotation", ["left", "right"])
@pytest.mark.perf
@cp.testing.gpu
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
