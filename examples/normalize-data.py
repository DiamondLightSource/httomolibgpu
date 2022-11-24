import cupy
from mpi4py import MPI

from loaders import standard_tomo
from httomolib.normalisation import normalize_cupy


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

# Apply CuPy implementation of normalisation to data
data = cupy.asarray(host_data)
flats = cupy.asarray(host_flats)
darks = cupy.asarray(host_darks)
data = normalize_cupy(data, flats, darks)
