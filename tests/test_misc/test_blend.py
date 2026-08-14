import cupy as cp
import numpy as np

from httomolibgpu.misc.blend import seam_blend_stitched_data


def test_seam_blend_stitched_data(data_stitched):
    result = seam_blend_stitched_data(
        data_stitched, overlap=20, seam_index=151, shift_seam_index=None
    )

    assert result.flags.c_contiguous
    assert result.dtype == cp.uint16
    assert result.shape == (300, 4, 280)
    np.testing.assert_array_almost_equal(int(cp.mean(result)), 11520)


def test_seam_blend_stitched_data_shift(data_stitched):
    result = seam_blend_stitched_data(
        data_stitched, overlap=20, seam_index=151, shift_seam_index=10
    )

    assert result.flags.c_contiguous
    assert result.dtype == np.uint16
    assert result.shape == (300, 4, 280)
    np.testing.assert_array_almost_equal(int(cp.mean(result)), 11506)
