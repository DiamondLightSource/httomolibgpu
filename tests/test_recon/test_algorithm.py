import cupy as cp
import numpy as np
from httomolibgpu.prep.normalize import dark_flat_field_correction, minus_log
from httomolibgpu.recon.algorithm import (
    FBP2d_astra,
    FBP3d_tomobar,
    LPRec3d_tomobar,
    SIRT3d_tomobar,
    CGLS3d_tomobar,
    FISTA3d_tomobar,
    ADMM3d_tomobar,
)
from numpy.testing import assert_allclose
import time
import pytest
from cupy.cuda import nvtx
from ..conftest import MaxMemoryHook


def test_reconstruct_FBP_2d_astra(data, flats, darks, ensure_clean_memory):
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)
    recon_size = 150

    recon_data = FBP2d_astra(
        cp.asnumpy(normalised_data),
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
        0,
        filter_type="shepp-logan",
        filter_parameter=None,
        filter_d=2.0,
        recon_size=recon_size,
        recon_mask_radius=0.9,
    )
    assert recon_data.flags.c_contiguous
    assert_allclose(np.mean(recon_data), 0.0020, atol=1e-04)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.265129, rtol=1e-05)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (recon_size, 128, recon_size)


def test_reconstruct_FBP_2d_astra_pad(data, flats, darks, ensure_clean_memory):
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)
    recon_size = 150

    recon_data = FBP2d_astra(
        cp.asnumpy(normalised_data),
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=20,
        filter_type="shepp-logan",
        filter_parameter=None,
        filter_d=2.0,
        recon_size=recon_size,
        recon_mask_radius=0.9,
    )
    assert recon_data.flags.c_contiguous
    assert_allclose(np.mean(recon_data), 0.0020, atol=1e-04)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.265565, rtol=1e-05)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (recon_size, 128, recon_size)


def test_reconstruct_FBP3d_tomobar_1(data, flats, darks, ensure_clean_memory):
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = FBP3d_tomobar(
        normalised_data,
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
        filter_freq_cutoff=1.1,
        recon_mask_radius=None,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.000798, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.102106, rtol=1e-05)
    assert_allclose(np.std(recon_data), 0.006293, rtol=1e-07, atol=1e-6)
    assert_allclose(np.median(recon_data), -0.000555, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (160, 128, 160)


def test_reconstruct_FBP3d_tomobar_2(data, flats, darks, ensure_clean_memory):
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = FBP3d_tomobar(
        normalised_data,
        np.linspace(5.0 * np.pi / 360.0, 180.0 * np.pi / 360.0, data.shape[0]),
        15.5,
        filter_freq_cutoff=1.1,
        recon_mask_radius=None,
    )

    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.000265, rtol=1e-07, atol=1e-6)
    assert_allclose(
        np.mean(recon_data, axis=(0, 2)).sum(), 0.03396, rtol=1e-06, atol=1e-5
    )
    assert_allclose(np.std(recon_data), 0.006599, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32


def test_reconstruct_FBP3d_tomobar_3_detpad_true(
    data, flats, darks, ensure_clean_memory
):
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = FBP3d_tomobar(
        normalised_data,
        np.linspace(5.0 * np.pi / 360.0, 180.0 * np.pi / 360.0, data.shape[0]),
        79,  # center
        True,  # detector pad
        1.1,  # filter_freq_cutoff
        210,  # recon_size
        2.0,  # recon_mask_radius
    )

    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.000692, atol=1e-6)
    assert_allclose(
        np.mean(recon_data, axis=(0, 2)).sum(), 0.088599, rtol=1e-06, atol=1e-5
    )
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (210, 128, 210)


def test_reconstruct_LPRec3d_tomobar_1(data, flats, darks, ensure_clean_memory):
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = LPRec3d_tomobar(
        normalised_data,
        angles=np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=0,
        recon_size=130,
        recon_mask_radius=0.95,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.007, atol=1e-3)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (130, 128, 130)


def test_reconstruct_LPRec3d_tomobar_1_pad(data, flats, darks, ensure_clean_memory):
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = LPRec3d_tomobar(
        normalised_data,
        angles=np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=10,
        recon_size=130,
        recon_mask_radius=0.95,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.007, atol=1e-3)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (130, 128, 130)


def test_reconstruct_SIRT3d_tomobar(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = SIRT3d_tomobar(
        normalised_data,
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
        recon_size=objrecon_size,
        iterations=10,
        nonnegativity=True,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.0018319691, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.23449206, rtol=1e-05)
    assert recon_data.dtype == np.float32


def test_reconstruct_CGLS3d_tomobar(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = CGLS3d_tomobar(
        normalised_data,
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        recon_size=objrecon_size,
        iterations=5,
        nonnegativity=True,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.0020498533, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.26238117, rtol=1e-03)
    assert recon_data.dtype == np.float32


def test_reconstruct_CGLS3d_tomobar_detpad_true(
    data, flats, darks, ensure_clean_memory
):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = CGLS3d_tomobar(
        normalised_data,
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=True,
        recon_size=objrecon_size,
        iterations=5,
        nonnegativity=True,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.mean(recon_data), 0.0021257945, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.27210176, rtol=1e-03)
    assert recon_data.dtype == np.float32


def test_reconstruct_FISTA3d_tomobar_pd_tv(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = FISTA3d_tomobar(
        normalised_data,
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=0,
        recon_size=objrecon_size,
        iterations=15,
        subsets_number=6,
        regularisation_type="PD_TV",
        regularisation_parameter=0.00001,
        regularisation_iterations=50,
        regularisation_half_precision=True,
        nonnegativity=True,
    )
    assert recon_data.flags.c_contiguous
    recon_data = cp.asnumpy(recon_data)
    assert_allclose(np.mean(recon_data), 0.0018348347, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.23485887, rtol=1e-04)
    assert recon_data.dtype == np.float32


def test_reconstruct_FISTA3d_tomobar_pd_tv_detpad_true(
    data, flats, darks, ensure_clean_memory
):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = FISTA3d_tomobar(
        normalised_data,
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=True,
        recon_size=objrecon_size,
        recon_mask_radius=2.0,
        iterations=15,
        subsets_number=6,
        regularisation_type="PD_TV",
        regularisation_parameter=0.00001,
        regularisation_iterations=50,
        regularisation_half_precision=True,
        nonnegativity=True,
    )
    assert recon_data.flags.c_contiguous
    recon_data = cp.asnumpy(recon_data)
    assert_allclose(np.mean(recon_data), 0.00183313, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.23464072, rtol=1e-04)
    assert recon_data.dtype == np.float32


def test_reconstruct_FISTA3d_tomobar_rof_tv(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = FISTA3d_tomobar(
        normalised_data,
        np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        79.5,
        recon_size=objrecon_size,
        iterations=15,
        subsets_number=6,
        regularisation_type="ROF_TV",
        regularisation_parameter=0.00001,
        regularisation_iterations=50,
        regularisation_half_precision=True,
        nonnegativity=True,
    )
    assert recon_data.flags.c_contiguous
    recon_data = cp.asnumpy(recon_data)
    assert_allclose(np.mean(recon_data), 0.0018366273, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.23508826, rtol=1e-04)
    assert recon_data.dtype == np.float32


def test_reconstruct_ADMM3d_tomobar_pd_tv(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = ADMM3d_tomobar(
        data=normalised_data,
        angles=np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=0,
        recon_size=objrecon_size,
        iterations=5,
        subsets_number=24,
        initialisation=None,
        regularisation_type="PD_TV",
        regularisation_parameter=0.004,
        regularisation_iterations=50,
        regularisation_half_precision=True,
        nonnegativity=True,
    )

    assert recon_data.flags.c_contiguous
    recon_data = cp.asnumpy(recon_data)
    assert_allclose(np.min(recon_data), -7.5129734e-05, rtol=1e-06, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.23559943, rtol=1e-04)
    assert recon_data.dtype == np.float32


def test_reconstruct_ADMM3d_warm_tomobar_pd_tv(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = ADMM3d_tomobar(
        data=normalised_data,
        angles=np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=0,
        recon_size=objrecon_size,
        iterations=2,
        subsets_number=24,
        initialisation="FBP",
        regularisation_type="PD_TV",
        regularisation_parameter=0.004,
        regularisation_iterations=50,
        regularisation_half_precision=True,
        nonnegativity=False,
    )

    assert recon_data.flags.c_contiguous
    recon_data = cp.asnumpy(recon_data)
    assert_allclose(np.min(recon_data), -0.024014484, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.22771806, rtol=1e-04)
    assert recon_data.dtype == np.float32


def test_reconstruct_ADMM3d_warm2_tomobar_pd_tv(
    data, flats, darks, ensure_clean_memory
):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = ADMM3d_tomobar(
        data=normalised_data,
        angles=np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=0,
        recon_size=objrecon_size,
        iterations=2,
        subsets_number=24,
        initialisation="CGLS",
        regularisation_type="PD_TV",
        regularisation_parameter=0.0008,
        regularisation_iterations=50,
        regularisation_half_precision=True,
        nonnegativity=False,
    )

    assert recon_data.flags.c_contiguous
    recon_data = cp.asnumpy(recon_data)
    assert_allclose(np.mean(recon_data), 0.001847589, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.236588, rtol=1e-04)
    assert recon_data.dtype == np.float32


def test_reconstruct_ADMM3d_tomobar_pd_tv_detpad_true(
    data, flats, darks, ensure_clean_memory
):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = ADMM3d_tomobar(
        data=normalised_data,
        angles=np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=True,
        recon_size=objrecon_size,
        recon_mask_radius=2.0,
        iterations=5,
        subsets_number=24,
        initialisation=None,
        regularisation_type="PD_TV",
        regularisation_parameter=0.004,
        regularisation_iterations=50,
        regularisation_half_precision=True,
        nonnegativity=True,
    )

    assert recon_data.flags.c_contiguous
    recon_data = cp.asnumpy(recon_data)
    assert_allclose(np.min(recon_data), -8.2713435e-05, rtol=1e-06, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.235260687, rtol=1e-04)
    assert recon_data.dtype == np.float32


def test_reconstruct_ADMM3d_warm_tomobar_pd_tv_detpad_true(
    data, flats, darks, ensure_clean_memory
):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = ADMM3d_tomobar(
        data=normalised_data,
        angles=np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        detector_pad=True,
        recon_size=objrecon_size,
        recon_mask_radius=2.0,
        initialisation="FBP",
        iterations=2,
        subsets_number=24,
        regularisation_type="PD_TV",
        regularisation_parameter=0.004,
        regularisation_iterations=50,
        regularisation_half_precision=True,
        nonnegativity=False,
    )

    assert recon_data.flags.c_contiguous
    recon_data = cp.asnumpy(recon_data)
    assert_allclose(np.min(recon_data), -0.024650197, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.08829363, rtol=1e-04)
    assert recon_data.dtype == np.float32


def test_reconstruct_ADMM3d_tomobar_rof_tv(data, flats, darks, ensure_clean_memory):
    objrecon_size = data.shape[2]
    normalised_data = dark_flat_field_correction(data, flats, darks, cutoff=10)
    normalised_data = minus_log(normalised_data)

    recon_data = ADMM3d_tomobar(
        data=normalised_data,
        angles=np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0]),
        center=79.5,
        recon_size=objrecon_size,
        iterations=3,
        subsets_number=24,
        regularisation_type="ROF_TV",
        regularisation_parameter=0.01,
        regularisation_iterations=50,
        regularisation_half_precision=True,
        nonnegativity=False,
    )

    assert recon_data.flags.c_contiguous
    recon_data = cp.asnumpy(recon_data)
    assert_allclose(np.mean(recon_data), 0.0017946999, rtol=1e-07, atol=1e-6)
    assert_allclose(np.mean(recon_data, axis=(0, 2)).sum(), 0.22972171, rtol=1e-04)
    assert recon_data.dtype == np.float32


@pytest.mark.perf
def test_FBP3d_tomobar_performance(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0
    data = cp.asarray(data_host, dtype=np.float32)
    angles = np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0])
    cor = 79.5
    det_pad = 0
    filter_freq_cutoff = 1.1

    # cold run first
    FBP3d_tomobar(data, angles, cor, det_pad, filter_freq_cutoff)
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        FBP3d_tomobar(data, angles, cor)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms
