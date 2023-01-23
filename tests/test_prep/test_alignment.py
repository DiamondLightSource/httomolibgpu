import cupy as cp
from cupy.testing import assert_allclose
from imageio.v2 import imread

from httomolib.prep.alignment import (
    distortion_correction_proj_cupy,
    distortion_correction_proj_discorpy_cupy,
)


@cp.testing.gpu
def test_correct_distortion():
    distortion_coeffs_path = \
        'tests/test_data/distortion-correction/distortion-coeffs.txt'

    path = "tests/test_data/distortion-correction/dot_pattern_03.tif"
    im_host = imread(path)
    im = cp.asarray(im_host)

    preview = {
        'starts': [0, 0],
        'stops': [im.shape[0], im.shape[1]],
        'steps': [1, 1]
    }
    corrected_data = distortion_correction_proj_cupy(
        im, distortion_coeffs_path, preview)

    for _ in range(5):
        assert_allclose(cp.mean(corrected_data), 200.16733869461675)
        assert cp.max(corrected_data) == 255

    #: test the implementation from tomopy
    corrected_data = distortion_correction_proj_discorpy_cupy(
        im, distortion_coeffs_path, preview)
    for _ in range(5):
        assert_allclose(cp.mean(corrected_data), 200.193982357142)
        assert cp.max(corrected_data) == 255

    peppers_path = "tests/test_data/distortion-correction/peppers.tif"
    im = cp.asarray(imread(peppers_path))
    corrected_data = distortion_correction_proj_cupy(
        im, distortion_coeffs_path, preview)

    for _ in range(5):
        assert_allclose(cp.mean(corrected_data), 95.51871109008789)
        assert cp.max(corrected_data) == 228

    #: test the implementation from tomopy
    corrected_data = distortion_correction_proj_discorpy_cupy(
        im, distortion_coeffs_path, preview)
    for _ in range(5):
        assert_allclose(cp.mean(corrected_data), 87.76617813110352)
        assert cp.max(corrected_data) == 228

    cameraman_path = "tests/test_data/distortion-correction/cameraman.tif"
    im = cp.asarray(imread(cameraman_path))
    corrected_data = distortion_correction_proj_cupy(
        im, distortion_coeffs_path, preview)

    for _ in range(5):
        assert_allclose(cp.mean(corrected_data), 122.2400016784668)
        assert cp.max(corrected_data) == 254

    # free up GPU memory
    im, corrected_data = None, None
    cp._default_memory_pool.free_all_blocks()
