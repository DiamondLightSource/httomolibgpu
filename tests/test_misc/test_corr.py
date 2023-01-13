import numpy as np
import pytest
from numpy.testing import assert_allclose

from httomolib.misc.corr import inpainting_filter3d

# --- Tomo standard data ---#
in_file = 'tests/test_data/tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']
eps = 1e-6


def test_inpainting_filter3d():
    mask = np.zeros(shape=host_data.shape)
    filtered_data = inpainting_filter3d(host_data, mask)
    for _ in range(5):
        assert_allclose(np.min(filtered_data), 62.0)
        assert_allclose(np.max(filtered_data), 1136.0)
        assert_allclose(np.mean(filtered_data), 809.04987, rtol=eps)
