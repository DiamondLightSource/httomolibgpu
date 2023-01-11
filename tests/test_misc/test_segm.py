import numpy as np
import pytest
from numpy.testing import assert_allclose

from httomolib.misc.segm import binary_thresholding

# --- Tomo standard data ---#
in_file = 'tests/test_data/tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']


def test_binary_thresholding():
    _data = np.zeros(shape=(100, 100, 100))
    _binary_mask = binary_thresholding(_data, 0.5)
    for _ in range(5):
        assert np.all(_binary_mask == 0)

    #: testing binary_thresholding on tomo_standard data
    binary_mask = binary_thresholding(host_data, 0.1)
    for _ in range(5):
        assert np.all(binary_mask == 1)
