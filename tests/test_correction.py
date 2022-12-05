import cupy as cp

from cupy.testing import assert_allclose
from imageio.v2 import imread

from httomolib.correction import correct_distortion


def test_correct_distortion():
    path = "data/distortion-correction/dot_pattern_03.tif"
    im_host = imread(path)
    im = cp.asarray(im_host)

    preview = {
        'starts': [0, 0],
        'stops': [im.shape[0], im.shape[1]],
        'steps': [1, 1]
    }

    distortion_coeffs_path = \
        'data/distortion-correction/distortion-coeffs.txt'

    corrected_data = correct_distortion(im, distortion_coeffs_path, preview)

    for _ in range(10):
        assert_allclose(cp.mean(corrected_data), 200.16733869461675)
        assert cp.max(corrected_data) == 255
