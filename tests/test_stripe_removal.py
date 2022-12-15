import cupy as cp
from cupy.testing import assert_allclose, assert_array_equal
from mpi4py import MPI

from httomolib.normalisation import normalize_cupy
from httomolib.stripe_removal import (
    remove_stripe_based_sorting_cupy,
    remove_stripes_tomocupy,
)
from loaders import standard_tomo

comm = MPI.COMM_WORLD


def test_stripe_removal():
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

    #--- testing the CuPy implementation from TomoCupy ---#
    data_after_stripe_removal = remove_stripes_tomocupy(data)
    for _ in range(10):
        assert_allclose(cp.mean(data_after_stripe_removal), 0.28924704,
                        rtol=1e-05)
        assert_allclose(cp.max(data_after_stripe_removal), 2.715983,
                        rtol=1e-05)
        assert_allclose(cp.min(data_after_stripe_removal), -0.15378489,
                        rtol=1e-05)

    #--- testing the CuPy port of TomoPy's implementation ---#
    corrected_data = remove_stripe_based_sorting_cupy(data)
    for _ in range(10):
        assert_allclose(cp.mean(corrected_data), 0.28907317, rtol=1e-05)
        assert_allclose(cp.max(corrected_data), 2.5370452, rtol=1e-05)
        assert_allclose(cp.min(corrected_data), -0.116429195, rtol=1e-05)

    # free up GPU memory by no longer referencing the variables
    data, flats, darks, data_after_stripe_removal, corrected_data = None, None, None, None, None
    cp._default_memory_pool.free_all_blocks()
