import cupy as cp

from cupy.testing import assert_allclose
from mpi4py import MPI

from httomolib.normalisation import normalize_cupy
from loaders import standard_tomo

comm = MPI.COMM_WORLD


def test_normalize_cupy():
    # testing cupy implementation for normalization

    in_file = 'data/tomo_standard.nxs'
    data_key = '/entry1/tomo_entry/data/data'
    image_key = '/entry1/tomo_entry/data/image_key'
    dimension = 1
    preview = [None, None, None]
    pad = 0

    (
        host_data, host_flats, host_darks, _, _, _, _
    ) = standard_tomo(in_file, data_key, image_key, dimension, preview, pad, comm)

    data = cp.asarray(host_data)
    flats = cp.asarray(host_flats)
    darks = cp.asarray(host_darks)
    data = normalize_cupy(data, flats, darks)

    data_min = cp.array(-0.16163824, dtype=cp.float32)
    data_max = cp.array(2.7530956, dtype=cp.float32)

    for _ in range(10):
        assert_allclose(cp.min(data), data_min, rtol=1e-05)
        assert_allclose(cp.max(data), data_max, rtol=1e-05)
