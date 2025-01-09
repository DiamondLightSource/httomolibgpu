import os
import cupy as cp
import numpy as np
import pytest

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "large_data_archive")


@pytest.fixture(scope="session")
def data_i12LFOV_file(test_data_path):
    in_file = os.path.join(test_data_path, "i12LFOV.npz")
    return np.load(in_file)


@pytest.fixture(scope="session")
def data_i12_sandstone_file(test_data_path):
    in_file = os.path.join(test_data_path, "i12_sandstone_50sinoslices.npz")
    return np.load(in_file)


@pytest.fixture(scope="session")
def data_geant4sim_file(test_data_path):
    in_file = os.path.join(test_data_path, "geant4_640_540_proj360.npz")
    return np.load(in_file)

@pytest.fixture
def i12LFOV_data(data_i12LFOV_file):
    return (
        cp.asarray(data_i12LFOV_file["projdata"]),
        data_i12LFOV_file["angles"],
        cp.asarray(data_i12LFOV_file["flats"]),
        cp.asarray(data_i12LFOV_file["darks"]),
    )


@pytest.fixture
def i12sandstone_data(data_i12_sandstone_file):
    return (
        cp.asarray(data_i12_sandstone_file["projdata"]),
        data_i12_sandstone_file["angles"],
        cp.asarray(data_i12_sandstone_file["flats"]),
        cp.asarray(data_i12_sandstone_file["darks"]),
    )


@pytest.fixture
def geantsim_data(data_geant4sim_file):
    return (
        cp.asarray(data_geant4sim_file["projdata"]),
        data_geant4sim_file["angles"],
        cp.asarray(data_geant4sim_file["flats"]),
        cp.asarray(data_geant4sim_file["darks"]),
    )


@pytest.fixture
def ensure_clean_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    yield None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
