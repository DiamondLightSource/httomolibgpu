import cupy as cp
import numpy as np
from httomolib.prep.normalize import normalize_cupy
from httomolib.recon.algorithm import (
    reconstruct_tomobar,
    reconstruct_tomopy_astra,
)
from numpy.testing import assert_allclose
from tomopy.prep.normalize import normalize
import time
import pytest
from cupy.cuda import nvtx


# def test_reconstruct_tomobar_fbp3d_host(
#     host_data,
#     host_flats,
#     host_darks,
# ):
#     cor = 79.5 #: center of rotation for tomo_standard
#     data = normalize(host_data, host_flats, host_darks, cutoff=15.0)

#     #--- reconstructing the data ---#
#     recon_data = reconstruct_tomobar(
#         data,
#         np.linspace(0. * np.pi / 180., 180. * np.pi / 180., data.shape[0]),
#         cor,
#         algorithm="FBP3D_host"
#     )

#     assert recon_data.shape == (128, 160, 160)
#     assert_allclose(np.mean(recon_data), -0.00047175083, rtol=1e-07, atol=1e-8)
#     assert_allclose(np.mean(recon_data, axis=(1, 2)).sum(), -0.06038412, rtol=1e-07)
#     assert_allclose(np.std(recon_data), 0.0034436132, rtol=1e-07, atol=1e-8)
#     assert_allclose(np.median(recon_data), 0.000302, rtol=1e-07, atol=1e6)
#     assert recon_data.dtype == np.float32

#     recon_data = reconstruct_tomobar(
#         normalize(host_data, host_flats, host_darks, cutoff=20.5),
#         np.linspace(5. * np.pi / 360., 180. * np.pi / 360., data.shape[0]),
#         15.5,
#         algorithm="FBP3D_host"
#     )
#     assert_allclose(np.mean(recon_data), -0.00015, rtol=1e-07, atol=1e-6)
#     assert_allclose(np.mean(recon_data, axis=(1, 2)).sum(), -0.019142,
#         rtol=1e-07, atol=1e-5)
#     assert_allclose(np.std(recon_data), 0.003561, rtol=1e-07, atol=1e-6)
#     assert recon_data.dtype == np.float32


@cp.testing.gpu
def test_reconstruct_tomobar_fbp3d_device_1(
    data,
    flats,
    darks,
    ensure_clean_memory
):
    recon_data = reconstruct_tomobar(
        normalize_cupy(data, flats, darks, cutoff=10, minus_log=True),
        np.linspace(0. * np.pi / 180., 180. * np.pi / 180., data.shape[0]),
        79.5,
    )

    assert_allclose(np.mean(recon_data), 0.000798, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(1, 2)).sum(), 0.102106, rtol=1e-05)
    assert_allclose(np.std(recon_data), 0.006293, rtol=1e-07, atol=1e-6)
    assert_allclose(np.median(recon_data), -0.000555, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32


@cp.testing.gpu
def test_reconstruct_tomobar_fbp3d_device_2(
    data,
    flats,
    darks,
    ensure_clean_memory
):
    recon_data = reconstruct_tomobar(
        normalize_cupy(data, flats, darks, cutoff=20.5, minus_log=False),
        np.linspace(5. * np.pi / 360., 180. * np.pi / 360., data.shape[0]),
        15.5,
    )
    assert_allclose(np.mean(recon_data), -0.00015, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(1, 2)).sum(), -0.019142,
        rtol=1e-06, atol=1e-5)
    assert_allclose(np.std(recon_data), 0.003561, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32


def test_reconstruct_tomopy_fbp_cuda(
    host_data,
    host_flats,
    host_darks,
    ensure_clean_memory
):
    data = normalize(host_data, host_flats, host_darks, cutoff=15.0)
    angles = np.linspace(0. * np.pi / 180., 180. * np.pi / 180., data.shape[0])

    recon_data_tomopy = reconstruct_tomopy_astra(data, angles, 79.5, algorithm="FBP_CUDA")

    assert_allclose(np.mean(recon_data_tomopy), 0.008697214, rtol=1e-07, atol=1e-8)
    assert_allclose(np.mean(recon_data_tomopy, axis=(1, 2)).sum(), 1.113243, rtol=1e-06)
    assert_allclose(np.median(recon_data_tomopy), 0.007031, rtol=1e-07, atol=1e-6)
    assert_allclose(np.std(recon_data_tomopy), 0.009089365, rtol=1e-07, atol=1e-8)

    #: check that the reconstructed data is of type float32
    assert recon_data_tomopy.dtype == np.float32


@cp.testing.gpu
@pytest.mark.perf
def test_reconstruct_tomobar_performance(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)
    angles = np.linspace(0. * np.pi / 180., 180. * np.pi / 180., data.shape[0])
    cor = 79.5

    # cold run first
    reconstruct_tomobar(data, angles, cor)
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        reconstruct_tomobar(data, angles, cor)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
