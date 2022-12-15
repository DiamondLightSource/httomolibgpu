import cupy as cp
from cupy.testing import assert_allclose
from mpi4py import MPI

from httomolib.centering import find_center_of_rotation
from httomolib.normalisation import normalize_cupy
from httomolib.stripe_removal import remove_stripes_tomocupy
from loaders import standard_tomo

comm = MPI.COMM_WORLD


def test_find_center_of_rotation():
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

    data_after_stripe_removal = remove_stripes_tomocupy(data)

    #--- testing the center of rotation on tomo_standard ---#
    cor = find_center_of_rotation(data_after_stripe_removal)
    for _ in range(10):
        assert_allclose(cor, 79.5)

    # free up GPU memory by no longer referencing the variables
    data, flats, darks, data_after_stripe_removal = None, None, None, None
    cp._default_memory_pool.free_all_blocks()
