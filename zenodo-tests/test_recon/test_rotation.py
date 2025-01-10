import cupy as cp
import numpy as np
import pytest

from httomolibgpu.prep.normalize import normalize
from httomolibgpu.recon.rotation import find_center_vo


def test_center_vo_i12LFOV(i12LFOV_data, ensure_clean_memory):
    projdata = i12LFOV_data[0]
    flats = i12LFOV_data[2]
    darks = i12LFOV_data[3]
    del i12LFOV_data

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata

    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(data_normalised[:, mid_slice, :])

    assert cor == 1197.75
    assert cor.dtype == np.float32


def test_center_vo_average_i12LFOV(i12LFOV_data, ensure_clean_memory):
    projdata = i12LFOV_data[0]
    flats = i12LFOV_data[2]
    darks = i12LFOV_data[3]
    del i12LFOV_data

    data_normalised = normalize(projdata, flats, darks, minus_log=False)
    del flats, darks, projdata

    cor = find_center_vo(data_normalised[:, 10:25, :], average_radius=5)

    assert cor == 1199.25
    assert cor.dtype == np.float32


def test_center_vo_i12_sandstone(i12sandstone_data, ensure_clean_memory):
    projdata = i12sandstone_data[0]
    flats = i12sandstone_data[2]
    darks = i12sandstone_data[3]
    del i12sandstone_data

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata

    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(data_normalised[:, mid_slice, :])

    assert cor == 1253.75
    assert cor.dtype == np.float32


def test_center_vo_i12_geantsim(geantsim_data, ensure_clean_memory):
    projdata = geantsim_data[0]
    flats = geantsim_data[2]
    darks = geantsim_data[3]
    del geantsim_data

    data_normalised = normalize(projdata, flats, darks, minus_log=True)
    del flats, darks, projdata

    mid_slice = data_normalised.shape[1] // 2
    cor = find_center_vo(data_normalised[:, mid_slice, :])

    assert cor == 319.5
    assert cor.dtype == np.float32
