"""Check GPU NLM against a small, independent CPU definition."""

import math

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from httomolibgpu import non_local_means


def _reference(image, h, search_radius, patch_radius, num_neighbours=0):
    """Scalar definition with clipped searches and overlapping valid patches."""
    if image.ndim == 3:
        return np.stack(
            [
                _reference(plane, h, search_radius, patch_radius, num_neighbours)
                for plane in image
            ]
        )
    height, width = image.shape
    result = np.empty_like(image)
    for y, x in np.ndindex(image.shape):
        matches = []
        for qx in range(max(0, x - search_radius), min(width, x + search_radius + 1)):
            for qy in range(
                max(0, y - search_radius), min(height, y + search_radius + 1)
            ):
                if (qx, qy) == (x, y):
                    continue
                distance = 0.0
                for px in range(-patch_radius, patch_radius + 1):
                    for py in range(-patch_radius, patch_radius + 1):
                        if not (
                            0 <= x + px < width
                            and 0 <= qx + px < width
                            and 0 <= y + py < height
                            and 0 <= qy + py < height
                        ):
                            continue
                        gaussian = (
                            math.exp(-(px * px + py * py) / (2 * patch_radius**2))
                            if patch_radius
                            else 1.0
                        )
                        difference = float(image[y + py, x + px]) - float(
                            image[qy + py, qx + px]
                        )
                        distance += gaussian * difference**2
                matches.append((math.exp(-distance / h**2), float(image[qy, qx])))
        if num_neighbours:
            matches = sorted(matches, key=lambda pair: pair[0], reverse=True)[
                :num_neighbours
            ]
        result[y, x] = (float(image[y, x]) + sum(w * v for w, v in matches)) / (
            1.0 + sum(w for w, _ in matches)
        )
    return result


@pytest.mark.parametrize("shape", [(5, 7), (2, 5, 7), (1, 7), (6, 1), (1, 1)])
@pytest.mark.parametrize("patch_radius", [0, 1, 3])
@pytest.mark.parametrize("num_neighbours", [0, 1, 4])
def test_matches_cpu_reference(shape, patch_radius, num_neighbours):
    host = np.random.default_rng(51).uniform(-0.5, 0.5, shape).astype(np.float32)
    device = cp.asarray(host)
    result = non_local_means(device, 0.4, 2, patch_radius, num_neighbours)
    expected = _reference(host, 0.4, 2, patch_radius, num_neighbours)
    assert_allclose(result.get(), expected, rtol=3e-5, atol=3e-7)
    assert_array_equal(device.get(), host)
    assert result.dtype == cp.float32
    assert result.shape == device.shape
    assert result.flags.c_contiguous
    assert result.data.ptr != device.data.ptr
    assert result.device.id == device.device.id


@pytest.mark.parametrize("num_neighbours", [0, 1, 8])
def test_constant_and_identical_patches(num_neighbours):
    host = np.full((2, 5, 7), -3.25, dtype=np.float32)
    result = non_local_means(cp.asarray(host), num_neighbours=num_neighbours)
    assert_array_equal(result.get(), host)


def test_non_contiguous_and_independent_slices():
    host = np.random.default_rng(17).normal(size=(2, 9, 11)).astype(np.float32)
    device = cp.asarray(host)[:, ::-2, ::2]
    result = non_local_means(device, h=2, search_radius=2, patch_radius=1)
    assert_allclose(result.get(), _reference(host[:, ::-2, ::2], 2, 2, 1), atol=5e-7)
    for index in range(2):
        single = non_local_means(device[index], h=2, search_radius=2, patch_radius=1)
        assert_allclose(result[index].get(), single.get(), atol=5e-7)


def test_zero_search_and_underflow_return_input():
    data = cp.arange(35, dtype=cp.float32).reshape(5, 7)
    assert_array_equal(non_local_means(data, search_radius=0).get(), data.get())
    assert_array_equal(non_local_means(data, h=1e-20).get(), data.get())


def test_denoising_reduces_error():
    clean = np.full((24, 24), 0.5, dtype=np.float32)
    noisy = clean + np.random.default_rng(9).normal(0, 0.1, clean.shape).astype(
        np.float32
    )
    result = non_local_means(cp.asarray(noisy), h=0.3, search_radius=3).get()
    assert np.mean((result - clean) ** 2) < np.mean((noisy - clean) ** 2) * 0.5


def test_current_stream():
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        data = cp.arange(35, dtype=cp.float32).reshape(5, 7) / 35
        result = non_local_means(data, h=0.2, search_radius=2, patch_radius=1)
        copied = result.copy()
    stream.synchronize()
    expected = _reference(np.arange(35, dtype=np.float32).reshape(5, 7) / 35, 0.2, 2, 1)
    assert_allclose(copied.get(), expected, rtol=3e-5, atol=3e-7)


def test_equal_weight_selection_is_stable():
    # Huge h rounds all float32 weights to one, so candidate order matters.
    host = np.arange(20, dtype=np.float32).reshape(4, 5)
    result = non_local_means(
        cp.asarray(host), h=1e10, search_radius=2, patch_radius=0, num_neighbours=3
    )
    assert_allclose(result.get(), _reference(host, 1e10, 2, 0, 3), atol=1e-6)


def test_coordinates_exceed_uint16():
    data = cp.arange(65539, dtype=cp.float32).reshape(1, -1)
    result = non_local_means(data, h=2, search_radius=1, patch_radius=0)
    # The linear ramp is unchanged away from the two boundaries by symmetry.
    assert_array_equal(result[0, 1:-1].get(), data[0, 1:-1].get())
    expected_edge = (65538 + math.exp(-0.25) * 65537) / (1 + math.exp(-0.25))
    assert_allclose(result[0, -1].get(), expected_edge)


@pytest.mark.skipif(cp.cuda.runtime.getDeviceCount() < 2, reason="needs two GPUs")
def test_uses_input_device_and_restores_current_device():
    with cp.cuda.Device(1):
        data = cp.ones((3, 4), dtype=cp.float32)
    with cp.cuda.Device(0):
        result = non_local_means(data)
        assert result.device.id == 1
        assert cp.cuda.runtime.getDevice() == 0
    with cp.cuda.Device(1):
        assert_array_equal(result.get(), np.ones((3, 4), dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"h": 0}, ValueError),
        ({"h": -1}, ValueError),
        ({"h": np.inf}, ValueError),
        ({"h": np.nan}, ValueError),
        ({"h": 1e-300}, ValueError),
        ({"h": 1e300}, ValueError),
        ({"h": True}, TypeError),
        ({"h": "0.1"}, TypeError),
        ({"search_radius": -1}, ValueError),
        ({"search_radius": 2**31}, ValueError),
        ({"search_radius": 1.5}, TypeError),
        ({"search_radius": True}, TypeError),
        ({"patch_radius": -1}, ValueError),
        ({"patch_radius": False}, TypeError),
        ({"num_neighbours": -1}, ValueError),
        ({"num_neighbours": 1.0}, TypeError),
        ({"search_radius": 1, "num_neighbours": 9}, ValueError),
        ({"search_radius": 0, "num_neighbours": 1}, ValueError),
    ],
)
def test_invalid_parameters(kwargs, error):
    with pytest.raises(error):
        non_local_means(cp.ones((3, 4), dtype=cp.float32), **kwargs)


@pytest.mark.parametrize("shape", [(4,), (1, 2, 3, 4), (0, 3), (2, 0, 3)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError, match="non-empty 2D image or 3D stack"):
        non_local_means(cp.empty(shape, dtype=cp.float32))


@pytest.mark.parametrize("dtype", [cp.float64, cp.uint16, cp.int32])
def test_invalid_dtype(dtype):
    with pytest.raises(ValueError, match="float32"):
        non_local_means(cp.ones((3, 4), dtype=dtype))


def test_host_array_rejected():
    with pytest.raises(TypeError, match="CuPy array"):
        non_local_means(np.ones((3, 4), dtype=np.float32))
