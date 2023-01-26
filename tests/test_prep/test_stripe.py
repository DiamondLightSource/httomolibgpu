import cupy as cp
import numpy as np
from cupy.testing import assert_allclose, assert_array_equal

from httomolib.prep.normalize import normalize_cupy
from httomolib.prep.stripe import (
    detect_stripes,
    merge_stripes,
    remove_stripe_based_sorting_cupy,
    remove_stripes_titarenko_cupy,
)

# --- Tomo standard data ---#
in_file = 'tests/test_data/tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']
host_flats = datafile['flats']
host_darks = datafile['darks']


@cp.testing.gpu
def test_stripe_removal():
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
        assert_allclose(cp.mean(corrected_data), 0.2886111, rtol=1e-07)
        assert_allclose(cp.max(corrected_data), 2.4899824, rtol=1e-07)
        assert_allclose(cp.min(corrected_data), -0.1081188, rtol=1e-07)

    # free up GPU memory by no longer referencing the variables
    data = flats = darks = data_after_stripe_removal = corrected_data = None
    cp._default_memory_pool.free_all_blocks()


def test_detect_stripes():
    stripe_detected = detect_stripes(host_data)
    assert_allclose(np.min(stripe_detected), 4.3)
    assert_allclose(np.max(stripe_detected), 381.4)


def test_merge_stripes():
    stripe_merged = merge_stripes(host_data)
    assert stripe_merged.shape == (180, 128, 160)
    assert np.min(stripe_merged) == 0
    assert np.max(stripe_merged) == 1
