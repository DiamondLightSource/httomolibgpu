"""Small, hand-checkable examples for the NumPy-backed utilities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from httomolibgpu.misc.blend import seam_blend_stitched_data
from httomolibgpu.misc.morph import average_projection_frames, sino_360_to_180
from httomolibgpu.misc.sorting import argsort_with_reverse
from httomolibgpu.prep.normalize import minus_log


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_average_frames_keeps_the_last_incomplete_group(dtype):
    data = np.array([2, 4, 6, 8, 12], dtype=dtype).reshape(5, 1, 1)
    original = data.copy()
    result = average_projection_frames(data, 2)
    assert_array_equal(result[:, 0, 0], [3, 7, 12])
    assert result.dtype == dtype
    assert_array_equal(data, original)


def test_average_frames_can_combine_all_projections():
    data = np.array([2, 4, 6], dtype=np.float32).reshape(3, 1, 1)
    assert_array_equal(average_projection_frames(data, 5), [[[4]]])
    assert_array_equal(average_projection_frames(data, 1), data)


@pytest.mark.parametrize("factor", [0, -1])
def test_average_frames_rejects_nonpositive_factors(factor):
    with pytest.raises(ValueError, match="projection_averaging_factor"):
        average_projection_frames(np.ones((4, 2, 3), dtype=np.float32), factor)


@pytest.mark.parametrize(
    "data", [np.ones((2, 3), np.float32), np.empty((0, 2, 3), np.float32)]
)
def test_average_frames_requires_nonempty_3d_data(data):
    with pytest.raises(ValueError, match="non-empty 3D"):
        average_projection_frames(data)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_sorting_indices_can_restore_the_original_array(axis):
    data = np.array([[3, 1, 1], [0, 5, 2]], dtype=np.float32)
    original = data.copy()
    indices, reverse = argsort_with_reverse(data, axis=axis)
    sorted_data = np.take_along_axis(data, indices, axis=axis)
    assert_array_equal(sorted_data, np.sort(data, axis=axis))
    assert_array_equal(np.take_along_axis(sorted_data, reverse, axis=axis), original)
    assert_array_equal(data, original)


def test_sorting_a_vector_handles_duplicates():
    data = np.array([3, 1, 1, 2], dtype=np.float32)
    indices, reverse = argsort_with_reverse(data)
    assert_array_equal(data[indices], [1, 1, 2, 3])
    assert_array_equal(data[indices][reverse], data)


@pytest.mark.parametrize("shape,axis", [((2, 2, 2), -1), ((2, 2), 2)])
def test_sorting_rejects_unsupported_dimensions_or_axes(shape, axis):
    with pytest.raises(ValueError):
        argsort_with_reverse(np.zeros(shape), axis=axis)


def test_negative_log_preserves_input_and_float32_dtype():
    transmission = np.exp(-np.array([0, 1, 2], dtype=np.float32))
    original = transmission.copy()
    result = minus_log(transmission)
    assert_allclose(result, [0, 1, 2], atol=1e-6)
    assert result.dtype == np.float32
    assert_array_equal(transmission, original)


@pytest.mark.parametrize(
    "side,expected",
    [("left", [7, 6, 5, 4, 0, 1, 2, 3]), ("right", [0, 1, 2, 3, 7, 6, 5, 4])],
)
def test_full_turn_stitching_flips_the_opposing_view(side, expected):
    # The unmatched final projection must not enter the stitched result.
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 99, 99, 99, 99], dtype=np.float32).reshape(
        3, 1, 4
    )
    original = data.copy()
    result = sino_360_to_180(data, overlap=0, side=side)
    assert_array_equal(result, np.array(expected, dtype=np.float32).reshape(1, 1, 8))
    assert_array_equal(data, original)


@pytest.mark.parametrize("side", ["left", "right"])
def test_stitching_constant_projections_does_not_create_a_seam(side):
    data = np.full((4, 2, 8), 7, dtype=np.float32)
    result = sino_360_to_180(data, overlap=2, side=side)
    assert result.shape == (2, 2, 16)
    assert result.dtype == np.float32
    assert_allclose(result, 7)


@pytest.mark.parametrize("overlap,side", [(7, "left"), (0, "unknown")])
def test_stitching_rejects_invalid_overlap_or_side(overlap, side):
    with pytest.raises(ValueError):
        sino_360_to_180(np.ones((4, 2, 8), dtype=np.float32), overlap, side)


def test_camera_seam_blending_preserves_a_constant_image():
    data = np.full((2, 3, 12), 5, dtype=np.float32)
    original = data.copy()
    result = seam_blend_stitched_data(data, overlap=2, seam_index=6)
    assert result.shape == (2, 3, 10)
    assert result.dtype == np.float32
    assert_allclose(result, 5)
    assert_array_equal(data, original)
