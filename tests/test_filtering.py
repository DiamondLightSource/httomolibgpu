import cupy as cp
from cupy.testing import assert_allclose
from mpi4py import MPI

from httomolib.filtering import fresnel_filter, paganin_filter
from loaders import standard_tomo

comm = MPI.COMM_WORLD


def test_fresnel_filter():
    in_file = 'data/tomo_standard.nxs'
    data_key = '/entry1/tomo_entry/data/data'
    image_key = '/entry1/tomo_entry/data/image_key'
    dimension = 1
    preview = [None, None, None]
    pad = 0
    eps = 1e-6

    host_data = \
        standard_tomo(in_file, data_key, image_key, dimension, preview, pad, comm)[0]

    data = cp.array(host_data)

    #--- testing the Fresnel filter on tomo_standard ---#
    pattern = 'PROJECTION'
    ratio = 100.0
    filtered_data = fresnel_filter(data, pattern, ratio)

    for _ in range(5):
        assert_allclose(cp.mean(filtered_data), 802.1125)
        assert_allclose(cp.max(filtered_data), 1039.5293)
        assert_allclose(cp.min(filtered_data), 95.74562)

    pattern = 'SINOGRAM'
    filtered_data = fresnel_filter(data, pattern, ratio)
    for _ in range(5):
        assert_allclose(cp.mean(filtered_data), 806.74347, rtol=eps)
        assert_allclose(cp.max(filtered_data), 1063.7007)
        assert_allclose(cp.min(filtered_data), 87.91508)

    #--- testing the Paganin filter on tomo_standard ---#
    filtered_data = paganin_filter(data)

    for _ in range(5):
        assert filtered_data.ndim == 3
        assert_allclose(cp.mean(filtered_data), -770.5339, rtol=eps)
        assert_allclose(cp.max(filtered_data), -679.80945, rtol=eps)

    filtered_data = paganin_filter(data, energy=100.0)

    assert_allclose(cp.mean(filtered_data), -778.61926, rtol=1e-05)
    assert_allclose(cp.min(filtered_data), -808.9013, rtol=eps)

    filtered_data = paganin_filter(data, pad_method='mean')

    assert_allclose(cp.mean(filtered_data), -765.3401, rtol=eps)
    assert_allclose(cp.min(filtered_data), -793.68787, rtol=eps)
