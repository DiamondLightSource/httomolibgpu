import cupy as cp
from mpi4py import MPI

from httomolib.filtering import paganin_filter
from loaders import standard_tomo

comm = MPI.COMM_WORLD

# Define input file and relevant internal NeXuS paths
in_file = 'data/tomo_standard.nxs'
data_key = '/entry1/tomo_entry/data/data'
image_key = '/entry1/tomo_entry/data/image_key'
DIMENSION = 1
PREVIEW = [None, None, None]
PAD = 0

# Load the projection data
host_data = \
    standard_tomo(in_file, data_key, image_key, DIMENSION, PREVIEW, PAD, comm)[0]
data = cp.asarray(host_data)

# Apply CuPy implementation of Paganin filter to projection data
data = paganin_filter(data)
