import cupy as cp
import numpy as np
import pytest
from cupy.cuda import nvtx
import time

from httomolibgpu.prep.normalize import normalize
from httomolibgpu.recon.algorithm import (
    FBP,
    LPRec,
)
from numpy.testing import assert_allclose
import time
import pytest
from cupy.cuda import nvtx

def test_reconstruct_FBP_i12_dataset1(i12_dataset1, ensure_clean_memory):

    projdata = i12_dataset1[0]
    angles = i12_dataset1[1]
    flats = i12_dataset1[2]
    darks = i12_dataset1[3]
    del i12_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    ensure_clean_memory

    recon_data = FBP(
        data_normalised,
        np.deg2rad(angles),
        center=1253.75,
        filter_freq_cutoff=0.35,     
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()
    assert_allclose(np.sum(recon_data), 46569.395, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 50, 2560)

@pytest.mark.perf
def test_FBP_performance_i13_dataset2(i13_dataset2, ensure_clean_memory):
    dev = cp.cuda.Device()
    projdata = i13_dataset2[0]
    angles = i13_dataset2[1]
    flats = i13_dataset2[2]
    darks = i13_dataset2[3]
    del i13_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    ensure_clean_memory

    # cold run first
    FBP(
        data_normalised,
        np.deg2rad(angles),
        center=1253.75,
        filter_freq_cutoff=0.35,
        )
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        FBP(
        data_normalised,
        np.deg2rad(angles),
        center=1286.25,
        filter_freq_cutoff=0.35,
        )
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_reconstruct_LPREC_i13_dataset2(i13_dataset2, ensure_clean_memory):
    
    projdata = i13_dataset2[0]
    angles = i13_dataset2[1]
    flats = i13_dataset2[2]
    darks = i13_dataset2[3]
    del i13_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    ensure_clean_memory


    recon_data = LPRec(
        data=data_normalised,
        angles=np.deg2rad(angles),
        center=1286.25,
    )
    assert recon_data.flags.c_contiguous
    recon_data = recon_data.get()

    assert_allclose(np.sum(recon_data), 2044.9531, rtol=1e-07, atol=1e-6)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (2560, 10, 2560)


@pytest.mark.perf
def test_LPREC_performance_i13_dataset2(i13_dataset2, ensure_clean_memory):
    dev = cp.cuda.Device()
    projdata = i13_dataset2[0]
    angles = i13_dataset2[1]
    flats = i13_dataset2[2]
    darks = i13_dataset2[3]
    del i13_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    ensure_clean_memory

    # cold run first
    LPRec(
        data=data_normalised,
        angles=np.deg2rad(angles),
        center=1286.25,
    )
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        LPRec(
        data=data_normalised,
        angles=np.deg2rad(angles),
        center=1286.25,
        )
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms