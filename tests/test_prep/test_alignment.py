import os

import cupy as cp
import numpy as np
import pytest
from httomolibgpu.prep.alignment import (
    distortion_correction_proj,
    distortion_correction_proj_discorpy,
)
from httomolibgpu import method_registry
from imageio.v2 import imread
from numpy.testing import assert_allclose

from tests import MaxMemoryHook


@cp.testing.gpu
@pytest.mark.parametrize(
    "image, mean_value, mean_sum, sum",
    [
        ("dot_pattern_03.tif", 200.16733869461675, 20016.733869461674, 214366729300),
        ("peppers.tif", 95.51871109008789, 9551.871109008789, 2503965700),
        ("cameraman.tif", 122.2400016784668, 12224.00016784668, 3204448300),
    ],
    ids=["dot_pattern_03", "peppers", "cameraman"],
)
@pytest.mark.parametrize(
    "implementation",
    [distortion_correction_proj, distortion_correction_proj_discorpy],
    ids=["cupy", "tomopy"],
)
def test_correct_distortion(
    distortion_correction_path,
    ensure_clean_memory,
    image,
    implementation,
    mean_value,
    mean_sum,
    sum,
):
    distortion_coeffs_path = os.path.join(
        distortion_correction_path, "distortion-coeffs.txt"
    )

    path = os.path.join(distortion_correction_path, image)
    im_host = np.asarray(imread(path))
    im_host = np.expand_dims(im_host, axis=0)
    # replicate into a stack of images
    im_stack = cp.asarray(cp.tile(im_host, (100, 1, 1)))

    preview = {
        "starts": [0, 0],
        "stops": [im_stack.shape[0], im_stack.shape[1]],
        "steps": [1, 1],
    }
    corrected_data = implementation(im_stack, distortion_coeffs_path, preview).get()

    assert_allclose(np.mean(corrected_data), mean_value)
    assert_allclose(np.mean(corrected_data, axis=(1, 2)).sum(), mean_sum, rtol=1e-6)
    assert np.sum(corrected_data) == sum

    assert corrected_data.dtype == np.uint8
    assert corrected_data.ndim == 3


def test_correct_distortion_1d_raises(distortion_correction_path, ensure_clean_memory):
    with pytest.raises(ValueError):
        preview = {"starts": [0, 0], "stops": [1, 1], "steps": [1, 1]}
        distortion_coeffs_path = os.path.join(
            distortion_correction_path, "distortion-coeffs.txt"
        )
        distortion_correction_proj_discorpy(cp.ones(1), distortion_coeffs_path, preview)


@pytest.mark.parametrize("stack_size", [20, 50, 100])
@pytest.mark.parametrize(
    "implementation",
    [distortion_correction_proj, distortion_correction_proj_discorpy],
    ids=["cupy", "tomopy"],
)
def test_distortion_correction_meta(
    distortion_correction_path, stack_size, implementation
):
    distortion_coeffs_path = os.path.join(
        distortion_correction_path, "distortion-coeffs.txt"
    )

    path = os.path.join(distortion_correction_path, "dot_pattern_03.tif")
    im_host = np.asarray(imread(path))
    im_host = np.expand_dims(im_host, axis=0)
    # replicate into a stack of images
    im_stack = cp.asarray(np.tile(im_host, (stack_size, 1, 1)))

    hook = MaxMemoryHook(im_stack.size * im_stack.itemsize)
    preview = {
        "starts": [0, 0],
        "stops": [im_stack.shape[1], im_stack.shape[2]],
        "steps": [1, 1],
    }

    with hook:
        implementation(im_stack, distortion_coeffs_path, preview)

    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = im_stack.shape[0]
    estimated_slices, dtype_out, output_dims = implementation.meta.calc_max_slices(
        0,
        (im_stack.shape[1], im_stack.shape[2]),
        im_stack.dtype,
        max_mem,
        metadata_path=distortion_coeffs_path,
        preview=preview,
    )
    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8

    assert (
        implementation.__name__ in method_registry["httomolibgpu"]["prep"]["alignment"]
    )
