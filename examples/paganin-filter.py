import cupy as cp
import numpy as np

from httomolib.prep.phase import paganin_filter

# Load data
in_file = 'tests/test_data/tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']
data = cp.asarray(host_data)

# Apply CuPy implementation of Paganin filter to projection data
data = paganin_filter(data)
