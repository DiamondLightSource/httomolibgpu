import cupy as cp
import numpy as np
import pytest

from httomolibgpu.prep.phase import paganin_filter
from conftest import force_clean_gpu_memory


# ----------------------------------------------------------#
# appplying paganin filter to i12_dataset3
@pytest.mark.parametrize(
    "test_case",
    [
        ("next_power_of_2", None, -6.2188172, -10.92260456),
        ("next_fast_length", None, -6.867182, -10.92272),
        ("use_pad_x_y", [80, 80], -6.32930, -10.92270),
    ],
)
def test_paganin_filter_i12_dataset3(i12_dataset3, test_case, ensure_clean_memory):
    padding_method, pad_x_y, test_max, test_min = test_case
    inputdata = cp.empty((3, 2050, 2560))
    inputdata[0, :, :] = i12_dataset3[0]
    inputdata[1, :, :] = i12_dataset3[1]
    inputdata[2, :, :] = i12_dataset3[2]
    del i12_dataset3

    ensure_clean_memory

    filtered_paganin = paganin_filter(
        inputdata, calculate_padding_value_method=padding_method, pad_x_y=pad_x_y
    )

    assert pytest.approx(np.max(cp.asnumpy(filtered_paganin)), rel=1e-3) == test_max
    assert pytest.approx(np.min(cp.asnumpy(filtered_paganin)), rel=1e-3) == test_min
