import cupy as cp
import numpy as np

from httomolib.normalisation import normalize_cupy

# Load the projection data
in_file = 'data/tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']
host_flats = datafile['flats']
host_darks = datafile['darks']

# Apply CuPy implementation of normalisation to data
data = cp.asarray(host_data)
flats = cp.asarray(host_flats)
darks = cp.asarray(host_darks)
data = normalize_cupy(data, flats, darks)
