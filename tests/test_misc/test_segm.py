import numpy as np
from httomolib.misc.segm import binary_thresholding


def test_binary_thresholding_zeros():
    data = np.zeros(shape=(100, 100, 100))
    binary_mask = binary_thresholding(data, 0.5)

    assert np.all(binary_mask == 0)
    assert binary_mask.dtype == np.uint8


def test_binary_thresholding_data(host_data):
    #: testing binary_thresholding on tomo_standard data
    binary_mask = binary_thresholding(host_data, 0.1, otsu=True, axis=2)

    assert np.sum(binary_mask) == 2694904
    assert binary_mask.dtype == np.uint8
