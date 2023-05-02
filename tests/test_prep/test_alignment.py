import os

import cupy as cp
import numpy as np
import pytest
from httomolib.prep.alignment import (
    distortion_correction_proj,
    distortion_correction_proj_discorpy,
)
from httomolib import method_registry
from imageio.v2 import imread
from numpy.testing import assert_allclose

from tests import MaxMemoryHook


@cp.testing.gpu
@pytest.mark.parametrize(
    "image, max_value, mean_value",
    [
        ("dot_pattern_03.tif", 255, 200.16733869461675),
        ("peppers.tif", 228, 95.51871109008789),
        ("cameraman.tif", 254, 122.2400016784668),
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
    max_value,
    mean_value,
):
    distortion_coeffs_path = os.path.join(
        distortion_correction_path, "distortion-coeffs.txt"
    )

    path = os.path.join(distortion_correction_path, image)
    im_host = imread(path)
    im = cp.asarray(im_host)

    preview = {"starts": [0, 0], "stops": [im.shape[0], im.shape[1]], "steps": [1, 1]}
    corrected_data = implementation(im, distortion_coeffs_path, preview).get()
    
    assert_allclose(np.mean(corrected_data), mean_value)
    assert np.max(corrected_data) == max_value

    assert corrected_data.dtype == np.uint8


@pytest.mark.parametrize("stack_size", [20, 50, 100])
@pytest.mark.parametrize(
    "implementation",
    [distortion_correction_proj, distortion_correction_proj_discorpy],
    ids=["cupy", "tomopy"],
)
def test_distortion_correction_meta(distortion_correction_path, stack_size, implementation):
    distortion_coeffs_path = os.path.join(
        distortion_correction_path, "distortion-coeffs.txt"
    )

    path = os.path.join(distortion_correction_path, "dot_pattern_03.tif")
    im_host = np.asarray(imread(path))
    im_host = np.expand_dims(im_host, axis=0)
    # replicate into a stack of images
    im_stack = cp.asarray(np.tile(im_host, (stack_size, 1, 1)))
    hook = MaxMemoryHook(im_stack.size * im_stack.itemsize)
    preview = {"starts": [0, 0], "stops": [im_stack.shape[1], im_stack.shape[2]], "steps": [1, 1]}
    
    with hook:
        implementation(im_stack, distortion_coeffs_path, preview)
    
    # make sure estimator function is within range (80% min, 100% max)
    max_mem = hook.max_mem
    actual_slices = im_stack.shape[0]
    estimated_slices, _ = implementation.meta.calc_max_slices(
        0, (im_stack.shape[1], im_stack.shape[2]),
        (im_stack.shape[1], im_stack.shape[2]),
        im_stack.dtype, max_mem, 
        metadata_path=distortion_coeffs_path, preview=preview)
    assert estimated_slices <= actual_slices
    assert estimated_slices / actual_slices >= 0.8 
    
    assert implementation.__name__ in method_registry['httomolib']['prep']['alignment']
