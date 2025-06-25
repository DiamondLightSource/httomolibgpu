import cupy as cp
import numpy as np
import pytest
from cupy.cuda import nvtx
import time

from httomolibgpu.prep.normalize import normalize
from httomolibgpu.recon.rotation import find_center_vo, find_center_pc, find_center_360
from conftest import force_clean_gpu_memory


# ----------------------------------------------------------#
# i12_dataset1 tests
def test_center_vo_i12_dataset1(i12_dataset1):
    projdata = i12_dataset1[0]
    flats = i12_dataset1[2]
    darks = i12_dataset1[3]
    del i12_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    force_clean_gpu_memory()

    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(
        data_normalised[:, mid_slice, :], smin=-100, smax=100, step=0.25
    )

    assert cor == 1253.75
    assert cor.dtype == np.float32


@pytest.mark.perf
def test_center_vo_i12_dataset1_performance(i12_dataset1, ensure_clean_memory):
    dev = cp.cuda.Device()

    projdata = i12_dataset1[0]
    flats = i12_dataset1[2]
    darks = i12_dataset1[3]
    del i12_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    ensure_clean_memory

    mid_slice = data_normalised.shape[1] // 2
    # cold run first
    cor = find_center_vo(data_normalised[:, mid_slice, :])

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        find_center_vo(data_normalised[:, mid_slice, :], smin=-100, smax=100, step=0.25)
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


# ----------------------------------------------------------#
# i12_dataset2 tests
def test_center_vo_i12_dataset2(i12_dataset2, ensure_clean_memory):
    projdata = i12_dataset2[0]
    flats = i12_dataset2[2]
    darks = i12_dataset2[3]
    del i12_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata
    ensure_clean_memory

    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(
        data_normalised[:, mid_slice, :], smin=-100, smax=100, step=0.25
    )

    assert cor == 1197.75
    assert cor.dtype == np.float32


def test_center_vo_average_i12_dataset2(i12_dataset2, ensure_clean_memory):
    projdata = i12_dataset2[0]
    flats = i12_dataset2[2]
    darks = i12_dataset2[3]
    del i12_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata

    ensure_clean_memory
    cor = find_center_vo(
        data_normalised[:, 10:25, :], average_radius=5, smin=-100, smax=100, step=0.25
    )

    assert cor == 1199.25
    assert cor.dtype == np.float32


# ----------------------------------------------------------#
# i12_dataset3 tests
# just two projections extracted in 180 degrees apart
def test_center_pc_i12_dataset3(i12_dataset3, ensure_clean_memory):
    proj1 = i12_dataset3[0]
    proj2 = i12_dataset3[1]
    flats = i12_dataset3[2]
    darks = i12_dataset3[3]
    del i12_dataset3

    projdata = cp.empty((2, np.shape(proj1)[0], np.shape(proj1)[1]))
    projdata[0, :, :] = proj1
    projdata[1, :, :] = proj2

    # normalising data
    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    proj1 = data_normalised[0, :, :]
    proj2 = data_normalised[1, :, :]
    del flats, darks, projdata, data_normalised
    ensure_clean_memory

    cor_pc = find_center_pc(proj1[200:1750, :], proj2[200:1750, :])

    assert cor_pc == 1253.5
    assert cor_pc.dtype == np.float32


# ----------------------------------------------------------#
# i13_dataset1 tests
# 360 degrees dataset
def test_center_360_i13_dataset1(i13_dataset1, ensure_clean_memory):
    projdata = i13_dataset1[0]
    flats = i13_dataset1[2]
    darks = i13_dataset1[3]
    del i13_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata

    ensure_clean_memory
    cor, overlap, side, overlap_position = find_center_360(data_normalised)

    assert int(cor) == 2322
    assert side == "right"
    assert int(overlap) == 473  # actual 473.822265625
    assert cor.dtype == np.float64


# ----------------------------------------------------------#
# i13_dataset2 tests
# 180 degrees dataset (2500 projections)
def test_center_vo_i13_dataset2(i13_dataset2, ensure_clean_memory):
    projdata = i13_dataset2[0]
    flats = i13_dataset2[2]
    darks = i13_dataset2[3]
    del i13_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata

    ensure_clean_memory
    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(
        data_normalised[:, mid_slice, :], smin=-100, smax=100, step=0.25
    )

    assert cor == 1286.25
    assert cor.dtype == np.float32


# ----------------------------------------------------------#
# i13_dataset3 tests
# 360 degrees dataset
def test_center_360_i13_dataset3(i13_dataset3, ensure_clean_memory):
    projdata = i13_dataset3[0]
    flats = i13_dataset3[2]
    darks = i13_dataset3[3]
    del i13_dataset3

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata

    ensure_clean_memory
    cor, overlap, side, overlap_position = find_center_360(
        data_normalised,
        ind=1,
        win_width=50,
        side=None,
        denoise=True,
        norm=True,
        use_overlap=True,
    )

    assert int(cor) == 218  # actual 218.08 (not correct CoR actually, should be 2341)
    assert side == "left"
    assert int(overlap) == 438  # actual 438.173828
    assert cor.dtype == np.float64


# ----------------------------------------------------------#
# geant4_dataset1 tests
def test_center_vo_geant4_dataset1(geant4_dataset1, ensure_clean_memory):
    projdata = geant4_dataset1[0]
    flats = geant4_dataset1[2]
    darks = geant4_dataset1[3]
    del geant4_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata

    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(data_normalised[:, mid_slice, :])

    assert cor == 319.5
    assert cor.dtype == np.float32


# ----------------------------------------------------------#
# k11_dataset1 tests
# 180 degrees dataset (4000 projections)
def test_center_vo_k11_dataset1(k11_dataset1, ensure_clean_memory):
    projdata = k11_dataset1[0]
    flats = k11_dataset1[2]
    darks = k11_dataset1[3]
    del k11_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata

    ensure_clean_memory
    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(
        data_normalised[:, mid_slice, :], smin=-100, smax=100, step=0.25
    )

    assert cor == 1269.25
    assert cor.dtype == np.float32
