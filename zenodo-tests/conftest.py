import os
import cupy as cp
import numpy as np
import pytest

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "large_data_archive")


@pytest.fixture(scope="session")
def i12_dataset1_file(test_data_path):
    in_file = os.path.join(test_data_path, "i12_dataset1.npz")
    return np.load(in_file)


@pytest.fixture
def i12_dataset1(i12_dataset1_file):
    return (
        cp.asarray(i12_dataset1_file["projdata"]),
        i12_dataset1_file["angles"],
        cp.asarray(i12_dataset1_file["flats"]),
        cp.asarray(i12_dataset1_file["darks"]),
    )


@pytest.fixture(scope="session")
def i12_dataset2_file(test_data_path):
    in_file = os.path.join(test_data_path, "i12_dataset2.npz")
    return np.load(in_file)


@pytest.fixture
def i12_dataset2(i12_dataset2_file):
    return (
        cp.asarray(i12_dataset2_file["projdata"]),
        i12_dataset2_file["angles"],
        cp.asarray(i12_dataset2_file["flats"]),
        cp.asarray(i12_dataset2_file["darks"]),
    )


@pytest.fixture(scope="session")
def geant4_dataset1_file(test_data_path):
    in_file = os.path.join(test_data_path, "geant4_dataset1.npz")
    return np.load(in_file)


@pytest.fixture
def geant4_dataset1(geant4_dataset1_file):
    return (
        cp.asarray(geant4_dataset1_file["projdata"]),
        geant4_dataset1_file["angles"],
        cp.asarray(geant4_dataset1_file["flats"]),
        cp.asarray(geant4_dataset1_file["darks"]),
    )


@pytest.fixture
def ensure_clean_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    yield None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
