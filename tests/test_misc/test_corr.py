import numpy as np
from httomolib.misc.corr import inpainting_filter3d
from numpy.testing import assert_allclose

eps = 1e-6


def test_inpainting_filter3d(host_data):
    mask = np.zeros(shape=host_data.shape)
    filtered_data = inpainting_filter3d(host_data, mask)
    
    assert_allclose(np.min(filtered_data), 62.0)
    assert_allclose(np.max(filtered_data), 1136.0)
    assert_allclose(np.mean(filtered_data), 809.04987, rtol=eps)

    #: make sure the output is float32
    assert filtered_data.dtype == np.float32
