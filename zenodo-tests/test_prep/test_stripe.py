import cupy as cp
import numpy as np
import pytest

from numpy.testing import assert_allclose
from httomolibgpu.prep.stripe import (
    remove_stripe_based_sorting,
    remove_stripe_ti,
    remove_all_stripe,
    raven_filter,
)
from httomolibgpu.prep.normalize import normalize
from conftest import force_clean_gpu_memory
from math import isclose


@pytest.mark.parametrize(
    "dataset_fixture, size_filt, norm_res_expected",
    [
        (
            "i12_dataset4",
            11,
            78.5593,
        ),
        (
            "i12_dataset4",
            21,
            92.4986,
        ),
        (
            "i12_dataset4",
            31,
            99.3062,
        ),
    ],
    ids=["size_11", "size_21", "size_31"],
)
def test_remove_stripe_based_sorting_i12_dataset4(
    request, dataset_fixture, size_filt, norm_res_expected
):
    dataset = request.getfixturevalue(dataset_fixture)
    data_normalised = normalize(dataset[0], dataset[2], dataset[3], minus_log=True)

    del dataset
    force_clean_gpu_memory()

    output = remove_stripe_based_sorting(
        cp.copy(data_normalised), size=size_filt, dim=1
    )

    residual_calc = data_normalised - output
    norm_res = np.linalg.norm(residual_calc.get().flatten())

    assert isclose(norm_res, norm_res_expected, abs_tol=10**-4)

    assert output.dtype == np.float32
    assert output.flags.c_contiguous


@pytest.mark.parametrize(
    "dataset_fixture, beta_val, norm_res_expected",
    [
        (
            "i12_dataset4",
            0.01,
            321.5298,
        ),
        (
            "i12_dataset4",
            0.03,
            172.1631,
        ),
        (
            "i12_dataset4",
            0.06,
            128.0331,
        ),
    ],
    ids=["beta_001", "beta_003", "beta_006"],
)
def test_remove_stripe_ti_i12_dataset4(
    request, dataset_fixture, beta_val, norm_res_expected
):
    dataset = request.getfixturevalue(dataset_fixture)
    data_normalised = normalize(dataset[0], dataset[2], dataset[3], minus_log=True)

    del dataset
    force_clean_gpu_memory()

    output = remove_stripe_ti(cp.copy(data_normalised), beta=beta_val)

    residual_calc = data_normalised - output
    norm_res = np.linalg.norm(residual_calc.get().flatten())

    assert isclose(norm_res, norm_res_expected, abs_tol=10**-4)

    assert output.dtype == np.float32
    assert output.flags.c_contiguous


@pytest.mark.parametrize(
    "dataset_fixture, snr_val, la_size_val, sm_size_val, norm_res_expected",
    [
        ("i12_dataset4", 1.0, 31, 10, 104.8582),
        ("i12_dataset4", 2.0, 41, 17, 99.4554),
        ("i12_dataset4", 3.0, 61, 21, 102.8688),
        ("i12_dataset4", 4.0, 71, 31, 106.1418),
    ],
    ids=["snr_1", "snr_2", "snr_3", "snr_4"],
)
def test_remove_all_stripe_i12_dataset4(
    request, dataset_fixture, snr_val, la_size_val, sm_size_val, norm_res_expected
):
    dataset = request.getfixturevalue(dataset_fixture)
    data_normalised = normalize(dataset[0], dataset[2], dataset[3], minus_log=True)

    del dataset
    force_clean_gpu_memory()

    output = remove_all_stripe(
        cp.copy(data_normalised),
        snr=snr_val,
        la_size=la_size_val,
        sm_size=sm_size_val,
        dim=1,
    )

    residual_calc = data_normalised - output
    norm_res = np.linalg.norm(residual_calc.get().flatten())

    assert isclose(norm_res, norm_res_expected, abs_tol=10**-4)
    assert not np.isnan(output).any(), "Output contains NaN values"
    assert output.dtype == np.float32
    assert output.flags.c_contiguous


@pytest.mark.parametrize(
    "dataset_fixture, nvalue_val, vvalue_val, norm_res_expected",
    [
        ("i12_dataset4", 2, 4, 94.0424),
        ("i12_dataset4", 4, 2, 86.2983),
        ("i12_dataset4", 6, 5, 111.0662),
    ],
    ids=["case_1", "case_2", "case_3"],
)
def test_raven_filter_i12_dataset4(
    request, dataset_fixture, nvalue_val, vvalue_val, norm_res_expected
):
    dataset = request.getfixturevalue(dataset_fixture)
    data_normalised = normalize(
        dataset[0][:, 10:20, :],
        dataset[2][:, 10:20, :],
        dataset[3][:, 10:20, :],
        minus_log=True,
    )

    del dataset
    force_clean_gpu_memory()

    output = raven_filter(
        cp.copy(data_normalised),
        pad_y=20,
        pad_x=20,
        pad_method="edge",
        uvalue=20,
        nvalue=nvalue_val,
        vvalue=vvalue_val,
    )

    residual_calc = data_normalised - output
    norm_res = np.linalg.norm(residual_calc.get().flatten())

    assert isclose(norm_res, norm_res_expected, abs_tol=10**-4)

    assert output.dtype == np.float32
    assert output.flags.c_contiguous
