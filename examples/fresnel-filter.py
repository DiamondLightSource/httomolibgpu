from pathlib import Path

import cupy as cp
import numpy as np
from httomolibgpu.prep.phase import fresnel_filter

# Load data
data_folder = Path("tests/test_data/")
in_file = data_folder / 'tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']
data = cp.asarray(host_data)

# Apply CuPy implementation of Fresnel filter to projection data
pattern = 'PROJECTION'
ratio = 100.0
data = fresnel_filter(data, pattern, ratio)

# Apply CuPy implementation of Fresnel filter to sinogram data
pattern = 'SINOGRAM'
ratio = 100.0
data = fresnel_filter(cp.swapaxes(data, 0, 1), pattern, ratio)
