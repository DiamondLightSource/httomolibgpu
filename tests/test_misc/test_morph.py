import time
import cupy as cp
import numpy as np
from cupy.cuda import nvtx
import pytest
from httomolibgpu.misc.morph import sino_360_to_180


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
