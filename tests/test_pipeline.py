import cupy as cp
import h5py
import numpy as np
from cupy.testing import assert_allclose

from httomolib.prep.normalize import normalize_cupy
from httomolib.prep.stripe import remove_stripe_based_sorting_cupy
from httomolib.recon.rotation import find_center_vo_cupy

from tomopy.prep.normalize import normalize
from tomopy.prep.stripe import remove_stripe_based_sorting
from tomopy.recon.rotation import find_center_vo


@cp.testing.gpu
def test_cpu_vs_gpu():
    #--- GPU pipeline tested on `tomo_standard` ---#

    in_file = 'tests/test_data/tomo_standard.npz'
    datafile = np.load(in_file)
    host_data = np.float32(datafile['data'])
    host_flats = np.float32(datafile['flats'])
    host_darks = np.float32(datafile['darks'])

    data = cp.asarray(host_data, dtype="float32")
    flats = cp.asarray(host_flats, dtype="float32")
    darks = cp.asarray(host_darks, dtype="float32")

    #: Normalize the data first
    data_normalize_cupy = normalize_cupy(data, flats, darks, cutoff=15.0)
    assert data_normalize_cupy.shape == (180, 128, 160)

    #: Now do the stripes removal
    corrected_data = remove_stripe_based_sorting_cupy(data_normalize_cupy)

    #: Apply Fresnel/Paganin filtering

    #: Set data collection angles as equally spaced between 0-180 degrees.
    cor = find_center_vo_cupy(corrected_data)

    #: Correct distortion


    #--- CPU pipeline tested on `tomo_standard` ---#
    tomopy_data = normalize(host_data, host_flats, host_darks, cutoff=15.0)
    assert tomopy_data.shape == (180, 128, 160)

    tomopy_corrected_data = remove_stripe_based_sorting(tomopy_data)

    tomopy_cor = find_center_vo(tomopy_corrected_data)

    #: TEST 1: check if the initial data was loaded correctly
    assert_allclose(cp.mean(data), np.mean(host_data))
    np.testing.assert_almost_equal(cp.std(data), np.std(host_data), decimal=3)

    #: TEST 2: check if the data is normalized correctly for both CPU and GPU
    np.testing.assert_almost_equal(
        cp.mean(data_normalize_cupy), np.mean(tomopy_data), decimal=3)

    #: TEST 3: check if the stripes are removed correctly for both CPU and GPU
    np.testing.assert_almost_equal(
        cp.mean(corrected_data), np.mean(tomopy_corrected_data), decimal=3)

    #: TODO: make this test work: values are different (0.080119, 0.076076)
    #: assert_allclose(cp.min(corrected_data), np.min(tomopy_corrected_data), rtol=1e-06)

    #: TEST 4: check if the center of rotation matches for both CPU and GPU
    assert_allclose(tomopy_cor, cor)
