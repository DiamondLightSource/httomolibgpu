import cupy as cp
import numpy as np
from httomolib.prep.normalize import normalize_cupy
from httomolib.recon.algorithm import (
    reconstruct_tomobar,
    reconstruct_tomopy,
)
from numpy.testing import assert_allclose
from tomopy.prep.normalize import normalize


def test_reconstruct_tomobar_fbp3d_host(
    host_data,
    host_flats,
    host_darks,
):
    cor = 79.5 #: center of rotation for tomo_standard
    data = normalize(host_data, host_flats, host_darks, cutoff=15.0)
    angles = np.linspace(0. * np.pi / 180., 180. * np.pi / 180., data.shape[0])

    #--- reconstructing the data ---#
    recon_data = reconstruct_tomobar(data, angles, cor, algorithm="FBP3D_host")

    assert recon_data.shape == (128, 160, 160)
    assert_allclose(np.mean(recon_data), -0.00047175083, rtol=1e-07, atol=1e-8)
    assert_allclose(np.std(recon_data), 0.0034436132, rtol=1e-07, atol=1e-8)

    #: check that the reconstructed data is of type float32
    assert recon_data.dtype == np.float32


@cp.testing.gpu
def test_reconstruct_tomobar_fbp3d_device(
    data,
    flats,
    darks,
    ensure_clean_memory
):
    angles = np.linspace(0. * np.pi / 180., 180. * np.pi / 180., data.shape[0])
    recon_data = reconstruct_tomobar(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True),
        angles,
        79.5,
        algorithm="FBP3D_device"
    )

    assert_allclose(np.mean(recon_data), 0.000798, rtol=1e-07, atol=1e-6)
    assert_allclose(np.std(recon_data), 0.006293, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32


def test_reconstruct_tomopy_fbp_cuda(
    host_data,
    host_flats,
    host_darks,
    ensure_clean_memory
):
    data = normalize(host_data, host_flats, host_darks, cutoff=15.0)
    angles = np.linspace(0. * np.pi / 180., 180. * np.pi / 180., data.shape[0])

    recon_data_tomopy = reconstruct_tomopy(data, angles, 79.5, algorithm="FBP_CUDA")

    assert_allclose(np.mean(recon_data_tomopy), 0.008697214, rtol=1e-07, atol=1e-8)
    assert_allclose(np.std(recon_data_tomopy), 0.009089365, rtol=1e-07, atol=1e-8)

    #: check that the reconstructed data is of type float32
    assert recon_data_tomopy.dtype == np.float32
