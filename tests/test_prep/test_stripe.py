import cupy as cp
import numpy as np
from cupy.testing import assert_allclose, assert_array_equal

from httomolib.prep.normalize import normalize_cupy
from httomolib.prep.stripe import (
    remove_stripe_based_sorting_cupy,
    remove_stripes_titarenko_cupy,
)


def test_stripe_removal():
    in_file = 'tests/test_data/tomo_standard.npz'
    # keys: data, flats, darks, angles, angles_total, detector_y, detector_x
    datafile = np.load(in_file)
    host_data = datafile['data']
    host_flats = datafile['flats']
    host_darks = datafile['darks']

    data = cp.asarray(host_data)
    flats = cp.asarray(host_flats)
    darks = cp.asarray(host_darks)
    data = normalize_cupy(data, flats, darks, cutoff=10, minus_log=True)

    # --- testing the CuPy implementation from TomoCupy ---#
    data_after_stripe_removal = remove_stripes_titarenko_cupy(data)
    for _ in range(10):
        assert_allclose(cp.mean(data_after_stripe_removal), 0.28924704,
                        rtol=1e-05)
        assert_allclose(cp.max(data_after_stripe_removal), 2.715983,
                        rtol=1e-05)
        assert_allclose(cp.min(data_after_stripe_removal), -0.15378489,
                        rtol=1e-05)

    # --- testing the CuPy port of TomoPy's implementation ---#
    corrected_data = remove_stripe_based_sorting_cupy(data)
    for _ in range(10):
        assert_allclose(cp.mean(corrected_data), 0.28907317, rtol=1e-05)
        assert_allclose(cp.max(corrected_data), 2.5370452, rtol=1e-05)
        assert_allclose(cp.min(corrected_data), -0.116429195, rtol=1e-05)

    # free up GPU memory by no longer referencing the variables
    data, flats, darks, data_after_stripe_removal, corrected_data = None, None, None, None, None
    cp._default_memory_pool.free_all_blocks()
