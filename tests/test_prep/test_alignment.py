import os

import cupy as cp
import numpy as np
import pytest
from httomolibgpu.prep.alignment import (
    distortion_correction_proj_discorpy,
)
from imageio.v2 import imread
from numpy.testing import assert_allclose


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
    [distortion_correction_proj_discorpy],
    ids=["tomopy"],
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
