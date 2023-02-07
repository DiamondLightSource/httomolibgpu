import numpy as np
from httomolib.recon.algorithm import reconstruct_tomobar, reconstruct_tomopy
from numpy.testing import assert_allclose
from tomopy.prep.normalize import normalize


def test_reconstruct_methods(
    host_data,
    host_flats,
    host_darks,
    ensure_clean_memory
):
    cor = 79.5 #: center of rotation for tomo_standard
    data = normalize(host_data, host_flats, host_darks, cutoff=15.0)
    angles = np.linspace(0. * np.pi / 180., 180. * np.pi / 180., data.shape[0])

    #--- reconstructing the data ---#
    recon_data = reconstruct_tomobar(data, angles, cor, algorithm="FBP3D_host")
    recon_data_tomopy = reconstruct_tomopy(data, angles, cor, algorithm="FBP_CUDA")

    assert recon_data.shape == (128, 160, 160)
    assert_allclose(np.mean(recon_data), -0.00047175083, rtol=1e-07, atol=1e-8)
    assert_allclose(np.std(recon_data), 0.0034436132, rtol=1e-07, atol=1e-8)

    assert_allclose(np.mean(recon_data_tomopy), 0.008697214, rtol=1e-07, atol=1e-8)
    assert_allclose(np.std(recon_data_tomopy), 0.009089365, rtol=1e-07, atol=1e-8)

    #: check that the reconstructed data is of type float32
    assert recon_data.dtype == np.float32
    assert recon_data_tomopy.dtype == np.float32
