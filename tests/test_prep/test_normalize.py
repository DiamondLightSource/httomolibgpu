import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from httomolib.prep.normalize import normalize_cupy


@cp.testing.gpu
def test_normalize_1D_raises(data, flats, darks):
    _data_1d = cp.ones(10)

    #: data cannot be a 1D array
    with pytest.raises(ValueError):
        normalize_cupy(_data_1d, flats, darks)

    #: flats cannot be a 1D array
    with pytest.raises(ValueError):
        normalize_cupy(data, _data_1d, darks)


@cp.testing.gpu
def test_normalize(data, flats, darks):
    # testing cupy implementation for normalization
    data_min = np.array(-0.16163824, dtype=cp.float32)
    data_max = np.array(2.7530956, dtype=cp.float32)

    #--- testing normalize_cupy  ---#
    data_normalize = cp.asnumpy(normalize_cupy(data, flats, darks, cutoff=10, minus_log=True))
    
    assert_allclose(np.min(data_normalize), data_min, rtol=1e-06)
    assert_allclose(np.max(data_normalize), data_max, rtol=1e-06)
