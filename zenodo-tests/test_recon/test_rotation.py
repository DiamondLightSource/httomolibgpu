import cupy as cp
import numpy as np
import pytest
from cupy.cuda import nvtx
import time

from httomolibgpu.prep.normalize import normalize
from httomolibgpu.recon.rotation import find_center_vo


def test_center_vo_i12_dataset1(i12_dataset1, ensure_clean_memory):
    projdata = i12_dataset1[0]
    flats = i12_dataset1[2]
    darks = i12_dataset1[3]
    del i12_dataset1

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata
    ensure_clean_memory

    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(data_normalised[:, mid_slice, :])

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
        find_center_vo(data_normalised[:, mid_slice, :])
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_center_vo_i12_dataset2(i12_dataset2, ensure_clean_memory):
    projdata = i12_dataset2[0]
    flats = i12_dataset2[2]
    darks = i12_dataset2[3]
    del i12_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata

    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(data_normalised[:, mid_slice, :])

    assert cor == 1197.75
    assert cor.dtype == np.float32


def test_center_vo_average_i12_dataset2(i12_dataset2, ensure_clean_memory):
    projdata = i12_dataset2[0]
    flats = i12_dataset2[2]
    darks = i12_dataset2[3]
    del i12_dataset2

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata

    cor = find_center_vo(data_normalised[:, 10:25, :], average_radius=5)

    assert cor == 1199.25
    assert cor.dtype == np.float32


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
