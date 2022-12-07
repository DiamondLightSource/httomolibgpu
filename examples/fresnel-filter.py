import cupy as cp
import h5py
from mpi4py import MPI

from httomolib.filtering import fresnel_filter
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

# Apply CuPy implementation of Fresnel filter to projection data
PATTERN = 'PROJECTION'
RATIO = 100.0
data = fresnel_filter(data, PATTERN, RATIO)

## Apply CuPy implementation of Fresnel filter to sinogram data
#PATTERN = 'SINOGRAM'
#RATIO = 100.0
#data = fresnel_filter(cp.swapaxes(data, 0, 1), PATTERN, RATIO)
