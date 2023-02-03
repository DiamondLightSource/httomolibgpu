import cupy as cp
import numpy as np
import pytest
from httomolib.prep.normalize import normalize_cupy
from numpy.testing import assert_allclose


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
    #--- testing normalize_cupy  ---#
    data_normalize = cp.asnumpy(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True))
    
    assert_allclose(np.min(data_normalize), -0.16163824, rtol=1e-06)
    assert_allclose(np.max(data_normalize), 2.7530956, rtol=1e-06)
