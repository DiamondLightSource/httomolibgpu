# Defines common fixtures and makes them available to all tests

import os

import cupy as cp
import numpy as np
import pytest
from typing import Callable, Tuple


CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def pytest_addoption(parser):
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests only",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "perf: mark test as performance test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--performance"):
        skip_other = pytest.mark.skip(reason="not a performance test")
        for item in items:
            if "perf" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="performance test - use '--performance' to run"
        )
        for item in items:
            if "perf" in item.keywords:
                item.add_marker(skip_perf)


def __get_memory_mb(device_index: int) -> int:
    """Get GPU memory in MBs

    Args:
        device_index (int): device_index (int, optional): GPU index

    Returns:
        int: Free memory amount in MBs for the given GPU card
    """

    dev = cp.cuda.Device(device_index)
    pool = cp.get_default_memory_pool()
    available_memory_gpu_bytes = dev.mem_info[0] + pool.free_bytes()
    gpu_memory_mbs = round(available_memory_gpu_bytes / (1024**2), 2)

    return gpu_memory_mbs


def memory_leak_test(
    func: Callable,
    input_array_dims: Tuple,
    iterations: int,
    tolerance_in_mb: float,
    device_index: int,
    **kwargs,
) -> bool:
    """This function will loop over a given method to check if the GPU memory leak happens.

    Args:
        func (Callable): A function to call
        input_array_dims (Tuple): Dimensions of CuPy input array to a function
        iterations (int): the number of iterations
        tolerance_in_mb (float): tolerance parameter in MBs
        device_index (int, optional): GPU index to run the test on a specific device

    Returns:
        bool: True if the tolerance is exceeded
    """
    data_input = cp.asarray(
        np.random.random_sample(size=input_array_dims).astype(np.float32) * 2.0
    )
    gpu_memory_prior_run = __get_memory_mb(device_index)
    for _ in range(iterations):
        data_output = func(cp.copy(data_input), **kwargs)
        del data_output
        # check the difference in memory after the method's execution bellow tolerance_exceeded
        gpu_memory_run = __get_memory_mb(device_index)
        difference_mb = abs(gpu_memory_prior_run - gpu_memory_run)
        if difference_mb >= tolerance_in_mb:
            return True
    return False


@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "test_data")


@pytest.fixture(scope="session")
def distortion_correction_path(test_data_path):
    return os.path.join(test_data_path, "distortion-correction")


# only load from disk once per session, and we use np.copy for the elements,
# to ensure data in this loaded file stays as originally loaded
@pytest.fixture(scope="session")
def data_file(test_data_path):
    in_file = os.path.join(test_data_path, "tomo_standard.npz")
    # keys: data, flats, darks, angles, angles_total, detector_y, detector_x
    return np.load(in_file)


@pytest.fixture
def ensure_clean_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    yield None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@pytest.fixture(scope="session")
def sino3600_file(test_data_path):
    in_file = os.path.join(test_data_path, "3600proj_sino.npz")
    return np.load(in_file)


@pytest.fixture
def sino3600(sino3600_file):
    return cp.asarray(sino3600_file["sinogram"])


@pytest.fixture
def host_data(data_file):
    return np.copy(data_file["data"])


@pytest.fixture
def data(host_data, ensure_clean_memory):
    return cp.asarray(host_data)


@pytest.fixture
def host_flats(data_file):
    return np.copy(data_file["flats"])


@pytest.fixture
def flats(host_flats, ensure_clean_memory):
    return cp.asarray(host_flats)


@pytest.fixture
def host_darks(
    data_file,
):
    return np.copy(data_file["darks"])


@pytest.fixture
def darks(host_darks, ensure_clean_memory):
    return cp.asarray(host_darks)


@pytest.fixture
def host_angles(data_file):
    return np.copy(data_file["angles"])


@pytest.fixture
def angles(host_angles, ensure_clean_memory):
    return cp.asarray(host_angles)


@pytest.fixture
def host_angles_total(data_file):
    return np.copy(data_file["angles_total"])


@pytest.fixture
def angles_total(host_angles_total, ensure_clean_memory):
    return cp.asarray(host_angles_total)


@pytest.fixture
def host_detector_y(data_file):
    return np.copy(data_file["detector_y"])


@pytest.fixture
def detector_y(host_detector_y, ensure_clean_memory):
    return cp.asarray(host_detector_y)


@pytest.fixture
def host_detector_x(data_file):
    return np.copy(data_file["detector_x"])


@pytest.fixture
def detector_x(host_detector_x, ensure_clean_memory):
    return cp.asarray(host_detector_x)
