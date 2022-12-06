import h5py
import cupy as cp
from mpi4py import MPI

from loaders import standard_tomo
from httomolib.filtering import fresnel_filter


comm = MPI.COMM_WORLD

# Define input file and relevant internal NeXuS paths
in_file = 'data/tomo_standard.nxs'
data_key = '/entry1/tomo_entry/data/data'
image_key = '/entry1/tomo_entry/data/image_key'
DIMENSION = 1
PREVIEW = [None, None, None]
PAD = 0

# Load the projection data
(   host_data,
    host_flats,
    host_darks,
    angles_radians,
    angles_total,
    detector_y,
    detector_x,
) = standard_tomo(in_file, data_key, image_key, DIMENSION, PREVIEW, PAD, comm)
data = cp.asarray(host_data)

# Apply CuPy implementation of Fresnel filter to projection data
PATTERN = 'PROJECTION'
RATIO = 100.0
data = fresnel_filter(data, PATTERN, RATIO)

## Apply CuPy implementation of Fresnel filter to sinogram data
#PATTERN = 'SINOGRAM'
#RATIO = 100.0
#data = fresnel_filter(cp.swapaxes(data, 0, 1), PATTERN, RATIO)
