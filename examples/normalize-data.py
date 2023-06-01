from pathlib import Path

import cupy as cp
import numpy as np
from httomolibgpu.prep.normalize import normalize

# Load the projection data
data_folder = Path("tests/test_data/")
in_file = data_folder / 'tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']
host_flats = datafile['flats']
host_darks = datafile['darks']

# Apply CuPy implementation of normalisation to data
data = cp.asarray(host_data)
flats = cp.asarray(host_flats)
darks = cp.asarray(host_darks)
data = normalize(data, flats, darks, cutoff = 10, minus_log = True)
