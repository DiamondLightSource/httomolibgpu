import cupy as cp
import numpy as np
import pytest

import math
from itertools import product

from httomolibgpu.misc.denoise import (
    total_variation_ROF,
    total_variation_PD,
    non_local_means,
)
from numpy.testing import assert_allclose, assert_array_equal

eps = 1e-6


def test_tv_rof_wrong_dtype(data):
    with pytest.raises(ValueError):
        total_variation_ROF(data)


def test_tv_ROF(data):
    filtered_data = total_variation_ROF(
        data.astype(cp.float32), regularisation_parameter=5, iterations=200
    ).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), 809.04994, rtol=eps)
    assert_allclose(np.max(filtered_data), 1130.5753)
    assert_allclose(np.min(filtered_data), 67.425545)

    assert filtered_data.dtype == np.float32
    assert filtered_data.flags.c_contiguous


def test_tv_pd_wrong_dtype(data):
    with pytest.raises(ValueError):
        total_variation_PD(data)


def test_tv_PD(data):
    filtered_data = total_variation_PD(
        data.astype(cp.float32), regularisation_parameter=5, iterations=200
    ).get()

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), 809.04987, rtol=eps)
    assert_allclose(np.max(filtered_data), 1112.6956)
    assert_allclose(np.min(filtered_data), 77.97832)

    assert filtered_data.dtype == np.float32
    assert filtered_data.flags.c_contiguous


def test_non_local_means2D_data(data):
    filtered_data = non_local_means(
        data.astype(cp.float32),
        h=100,
        num_neighbours=0,
        full_3d=False,
    ).get()
    assert filtered_data.shape == data.shape
    assert np.isfinite(filtered_data).all()
    assert_allclose(np.mean(filtered_data), 808.9238, rtol=eps)
    assert_allclose(np.max(filtered_data), 1083.4799)
    assert_allclose(np.min(filtered_data), 84.9413)


def test_non_local_means3D_data(data):
    filtered_data = non_local_means(
        data.astype(cp.float32),
        h=100,
        num_neighbours=0,
        full_3d=True,
    ).get()
    assert filtered_data.shape == data.shape
    assert np.isfinite(filtered_data).all()
    assert_allclose(np.mean(filtered_data), 808.9159, rtol=eps)
    assert_allclose(np.max(filtered_data), 1076.7588)
    assert_allclose(np.min(filtered_data), 89.918655)


def _coordinates_in_window(centre, shape, radius):
    """Visit valid candidates in CUDA's x, then y, then z loop order."""
    axis_ranges = [
        range(max(0, position - radius), min(size, position + radius + 1))
        for position, size in zip(centre, shape)
    ]
    for reversed_coordinate in product(*reversed(axis_ranges)):
        yield tuple(reversed(reversed_coordinate))


def _patch_distance(image, centre, neighbour, patch_radius):
    """Sum Gaussian-weighted differences at offsets valid in both patches."""
    valid_offsets = [
        range(
            max(-patch_radius, -position, -other_position),
            min(patch_radius, size - 1 - position, size - 1 - other_position) + 1,
        )
        for position, other_position, size in zip(centre, neighbour, image.shape)
    ]
    distance = 0.0
    for offset in product(*valid_offsets):
        centre_sample = tuple(position + step for position, step in zip(centre, offset))
        neighbour_sample = tuple(
            position + step for position, step in zip(neighbour, offset)
        )
        squared_radius = sum(step**2 for step in offset)
        gaussian_weight = (
            math.exp(-squared_radius / (2 * patch_radius**2)) if patch_radius else 1.0
        )
        difference = float(image[centre_sample]) - float(image[neighbour_sample])
        distance += gaussian_weight * difference**2
    return distance


def _reference(image, h, search_radius, patch_radius, num_neighbours=0, full_3d=True):
    """Readable scalar NLM definition, independent of the CUDA implementation."""
    if image.ndim == 3 and not full_3d:
        return np.stack(
            [
                _reference(plane, h, search_radius, patch_radius, num_neighbours)
                for plane in image
            ]
        )

    result = np.empty_like(image)
    for centre in np.ndindex(image.shape):
        matches = []
        for neighbour in _coordinates_in_window(centre, image.shape, search_radius):
            if neighbour == centre:
                continue
            distance = _patch_distance(image, centre, neighbour, patch_radius)
            weight = math.exp(-distance / h**2)
            matches.append((weight, float(image[neighbour])))

        if num_neighbours:
            # Python's stable sort preserves the search order for equal weights.
            matches.sort(key=lambda match: match[0], reverse=True)
            matches = matches[:num_neighbours]

        # The original centre sample always contributes with weight one.
        weighted_sum = float(image[centre])
        total_weight = 1.0
        for weight, value in matches:
            weighted_sum += weight * value
            total_weight += weight
        result[centre] = weighted_sum / total_weight
    return result


@pytest.mark.parametrize(
    "shape, full_3d",
    [
        pytest.param((5, 7), True, id="2d-image"),
        pytest.param((5, 7), False, id="2d-image-with-slice-flag"),
        pytest.param((2, 5, 7), False, id="independent-slices"),
        pytest.param((3, 4, 5), True, id="3d-volume"),
        pytest.param((1, 4, 5), True, id="single-slice-volume"),
        pytest.param((5, 1, 1), True, id="line-along-z"),
        pytest.param((1, 5, 1), True, id="line-along-y"),
        pytest.param((1, 1, 5), True, id="line-along-x"),
        pytest.param((1, 1, 1), True, id="single-voxel"),
    ],
)
@pytest.mark.parametrize(
    "patch_radius", [0, 1, 3], ids=["point", "radius-1", "radius-3"]
)
@pytest.mark.parametrize(
    "num_neighbours", [0, 1, 4], ids=["all-matches", "best-1", "best-4"]
)
def test_gpu_matches_cpu_definition(shape, full_3d, patch_radius, num_neighbours):
    original = np.random.default_rng(51).uniform(-0.5, 0.5, shape).astype(np.float32)
    gpu_input = cp.asarray(original)
    parameters = dict(
        h=0.4,
        search_radius=2,
        patch_radius=patch_radius,
        num_neighbours=num_neighbours,
        full_3d=full_3d,
    )

    result = non_local_means(gpu_input, **parameters)
    expected = _reference(original, **parameters)

    assert_allclose(result.get(), expected, rtol=3e-5, atol=3e-7)
    assert_array_equal(gpu_input.get(), original)
    assert result.dtype == cp.float32
    assert result.shape == gpu_input.shape
    assert result.flags.c_contiguous
    assert result.data.ptr != gpu_input.data.ptr
    assert result.device.id == gpu_input.device.id


def test_volume_mode_averages_neighbours_above_and_below():
    volume = cp.asarray([0.0, 1.0, 0.0], dtype=cp.float32).reshape(3, 1, 1)

    result = non_local_means(volume, h=1, search_radius=1, patch_radius=0)

    expected_centre = 1 / (1 + 2 * math.exp(-1))
    assert_allclose(result[1, 0, 0].get(), expected_centre, rtol=1e-6)


def test_slice_mode_does_not_mix_neighbouring_slices():
    volume = cp.asarray([0.0, 1.0, 0.0], dtype=cp.float32).reshape(3, 1, 1)

    result = non_local_means(
        volume, h=1, search_radius=1, patch_radius=0, full_3d=False
    )

    assert_array_equal(result.get(), volume.get())


def test_single_slice_volume_matches_a_2d_image():
    image = cp.asarray(np.random.default_rng(8).normal(size=(4, 5)), dtype=cp.float32)
    image_result = non_local_means(image, h=1, search_radius=1, patch_radius=1)
    volume_result = non_local_means(image[None], h=1, search_radius=1, patch_radius=1)
    assert_array_equal(volume_result[0].get(), image_result.get())


def test_volume_axes_are_treated_equally():
    volume = cp.asarray(
        np.random.default_rng(2).normal(size=(3, 4, 5)), dtype=cp.float32
    )
    original_result = non_local_means(volume, h=2, search_radius=1, patch_radius=1)
    transposed_result = non_local_means(
        volume.transpose(2, 1, 0), h=2, search_radius=1, patch_radius=1
    )
    assert_allclose(
        transposed_result.transpose(2, 1, 0).get(), original_result.get(), atol=3e-7
    )


@pytest.mark.parametrize("num_neighbours", [0, 4], ids=["all-matches", "best-4"])
def test_chunk_overlap_reproduces_whole_volume(num_neighbours):
    volume = cp.asarray(
        np.random.default_rng(4).normal(size=(9, 3, 4)), dtype=cp.float32
    )
    search_radius = 1
    patch_radius = 1
    halo = search_radius + patch_radius
    whole_result = non_local_means(
        volume,
        h=2,
        search_radius=search_radius,
        patch_radius=patch_radius,
        num_neighbours=num_neighbours,
    )

    for start in range(0, volume.shape[0], 3):
        stop = min(start + 3, volume.shape[0])
        input_start = max(0, start - halo)
        input_stop = min(volume.shape[0], stop + halo)
        chunk_result = non_local_means(
            volume[input_start:input_stop],
            h=2,
            search_radius=search_radius,
            patch_radius=patch_radius,
            num_neighbours=num_neighbours,
        )
        interior = chunk_result[start - input_start : stop - input_start]
        assert_allclose(interior.get(), whole_result[start:stop].get(), atol=3e-7)


@pytest.mark.parametrize("full_3d", [True, False], ids=["volume", "slices"])
@pytest.mark.parametrize("num_neighbours", [0, 1, 8])
def test_constant_and_identical_patches(num_neighbours, full_3d):
    host = np.full((2, 5, 7), -3.25, dtype=np.float32)
    result = non_local_means(
        cp.asarray(host), num_neighbours=num_neighbours, full_3d=full_3d
    )
    assert_array_equal(result.get(), host)


def test_non_contiguous_and_independent_slices():
    host = np.random.default_rng(17).normal(size=(2, 9, 11)).astype(np.float32)
    device = cp.asarray(host)[:, ::-2, ::2]
    result = non_local_means(
        device, h=2, search_radius=2, patch_radius=1, full_3d=False
    )
    assert_allclose(
        result.get(), _reference(host[:, ::-2, ::2], 2, 2, 1, full_3d=False), atol=5e-7
    )
    for index in range(2):
        single = non_local_means(device[index], h=2, search_radius=2, patch_radius=1)
        assert_allclose(result[index].get(), single.get(), atol=5e-7)


@pytest.mark.parametrize("shape", [(5, 7), (3, 4, 5)], ids=["image", "volume"])
def test_zero_search_and_underflow_return_input(shape):
    data = cp.arange(math.prod(shape), dtype=cp.float32).reshape(shape)
    assert_array_equal(non_local_means(data, search_radius=0).get(), data.get())
    assert_array_equal(non_local_means(data, h=1e-20).get(), data.get())


@pytest.mark.parametrize("shape", [(24, 24), (8, 9, 10)], ids=["image", "volume"])
def test_denoising_reduces_error(shape):
    clean = np.full(shape, 0.5, dtype=np.float32)
    noise_sigma = 0.1
    noisy = clean + np.random.default_rng(9).normal(0, noise_sigma, clean.shape).astype(
        np.float32
    )
    result = non_local_means(cp.asarray(noisy), h=4 * noise_sigma).get()
    assert np.mean((result - clean) ** 2) < np.mean((noisy - clean) ** 2) * 0.5


@pytest.mark.parametrize("shape", [(5, 7), (3, 4, 5)], ids=["image", "volume"])
def test_current_stream(shape):
    host = np.arange(math.prod(shape), dtype=np.float32).reshape(shape) / math.prod(
        shape
    )
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        data = cp.asarray(host)
        result = non_local_means(data, h=0.2, search_radius=2, patch_radius=1)
        copied = result.copy()
    stream.synchronize()
    expected = _reference(host, h=0.2, search_radius=2, patch_radius=1)
    assert_allclose(copied.get(), expected, rtol=3e-5, atol=3e-7)


@pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5)], ids=["image", "volume"])
def test_equal_weight_selection_is_stable(shape):
    # Huge h rounds all float32 weights to one, so candidate order matters.
    host = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    result = non_local_means(
        cp.asarray(host), h=1e10, search_radius=2, patch_radius=0, num_neighbours=3
    )
    assert_allclose(result.get(), _reference(host, 1e10, 2, 0, 3), atol=1e-6)


def test_volume_accepts_a_reversed_and_strided_view():
    original = np.random.default_rng(3).normal(size=(5, 7, 9)).astype(np.float32)
    gpu_view = cp.asarray(original)[::-2, ::2, ::-2]

    result = non_local_means(gpu_view, h=2, search_radius=1, patch_radius=1)
    expected = _reference(
        original[::-2, ::2, ::-2], h=2, search_radius=1, patch_radius=1
    )

    assert_allclose(result.get(), expected, atol=5e-7)
    assert result.flags.c_contiguous


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
        ({"full_3d": "yes"}, TypeError),
        ({"full_3d": 1}, TypeError),
        ({"search_radius": 0, "num_neighbours": 1}, ValueError),
    ],
    ids=lambda value: (
        ",".join(f"{name}={setting}" for name, setting in value.items())
        if isinstance(value, dict)
        else value.__name__
    ),
)
def test_invalid_parameters(kwargs, error):
    with pytest.raises(error):
        non_local_means(cp.ones((3, 4), dtype=cp.float32), **kwargs)


@pytest.mark.parametrize("shape", [(4,), (1, 2, 3, 4), (0, 3), (2, 0, 3)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError, match="non-empty 2D or 3D array"):
        non_local_means(cp.empty(shape, dtype=cp.float32))


@pytest.mark.parametrize("dtype", [cp.float64, cp.uint16, cp.int32])
def test_invalid_dtype(dtype):
    with pytest.raises(ValueError, match="float32"):
        non_local_means(cp.ones((3, 4), dtype=dtype))


def test_host_array_rejected():
    with pytest.raises(TypeError, match="CuPy array"):
        non_local_means(np.ones((3, 4), dtype=np.float32))


def test_neighbour_limit_uses_selected_dimensions():
    volume = cp.ones((3, 3, 3), dtype=cp.float32)
    # A 3D radius-one search allows 26 neighbours; a 2D search allows eight.
    result = non_local_means(volume, search_radius=1, num_neighbours=26)
    assert_array_equal(result.get(), volume.get())
    with pytest.raises(ValueError, match="number of non-self candidates"):
        non_local_means(volume, search_radius=1, num_neighbours=27)
    with pytest.raises(ValueError, match="number of non-self candidates"):
        non_local_means(volume, search_radius=1, num_neighbours=9, full_3d=False)


@pytest.mark.parametrize(
    "shape, full_3d",
    [
        pytest.param((6, 7), True, id="image"),
        pytest.param((4, 5, 6), True, id="volume"),
        pytest.param((4, 5, 6), False, id="independent-slices"),
    ],
)
def test_defaults_use_radius_two_search_and_radius_one_patches(shape, full_3d):
    original = np.random.default_rng(21).normal(0.5, 0.03, shape).astype(np.float32)

    result = non_local_means(cp.asarray(original), full_3d=full_3d)
    expected = _reference(
        original,
        h=0.1,
        search_radius=2,
        patch_radius=1,
        num_neighbours=0,
        full_3d=full_3d,
    )

    assert_allclose(result.get(), expected, rtol=3e-5, atol=5e-7)
