import cupy as cp
import numpy as np
import pytest 

from httomolibgpu.prep.normalize import normalize
from httomolibgpu.prep.phase import paganin_filter
from conftest import force_clean_gpu_memory


# ----------------------------------------------------------#
# appplying paganin filter to i12_dataset3
def test_paganin_filter_i12_dataset3(i12_dataset3, ensure_clean_memory):
    inputdata = cp.empty((3, 2050, 2560))
    inputdata[0, :, :] = i12_dataset3[0]
    inputdata[1, :, :] = i12_dataset3[1]
    inputdata[2, : , :] = i12_dataset3[2]
    del i12_dataset3

    ensure_clean_memory
    
    filtered_paganin = paganin_filter(inputdata)

    assert pytest.approx(np.max(filtered_paganin.get()), rel=1e-3) == -6.2188172
    assert pytest.approx(np.min(filtered_paganin.get()), rel=1e-3) == -10.92260456


