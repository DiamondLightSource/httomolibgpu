# A pre-processing pipeline using GPU that leads to data reconstruction

import timeit
from pathlib import Path

import cupy as cp
import numpy as np

from httomolib.prep.alignment import distortion_correction_proj_cupy
from httomolib.prep.normalize import normalize_cupy
from httomolib.prep.phase import paganin_filter
from httomolib.prep.stripe import remove_stripe_based_sorting_cupy
from httomolib.recon.rotation import find_center_vo_cupy

data_folder = Path("tests/test_data/")

# Load the projection data
in_file =  data_folder / 'tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']
host_flats = datafile['flats']
host_darks = datafile['darks']

# Apply CuPy implementation of normalisation to data
data = cp.asarray(host_data)
flats = cp.asarray(host_flats)
darks = cp.asarray(host_darks)

elapsed_time_total = 0.0
# ---------------------------------------------------------------#
#TODO: dezinger should be added here
# ---------------------------------------------------------------#
print ("Finding the Center of Rotation for the reconstruction")
start_time = timeit.default_timer()
cor = find_center_vo_cupy(data)
elapsed_time_01 = timeit.default_timer() - start_time
print('elapsed time', elapsed_time_01)
elapsed_time_total += elapsed_time_01
# ---------------------------------------------------------------#
print ("Applying normalisation and take a negative log")
start_time = timeit.default_timer()
data = normalize_cupy(data, flats, darks, 
                      cutoff = 10, minus_log = True)
elapsed_time_02 = timeit.default_timer() - start_time
print('elapsed time', elapsed_time_02)
elapsed_time_total += elapsed_time_02
# ---------------------------------------------------------------#
print ("Applying ring removal")
# TODO replace with remove all rings
start_time = timeit.default_timer()
destriped_data = remove_stripe_based_sorting_cupy(data)
elapsed_time_03 = timeit.default_timer() - start_time
print('elapsed time', elapsed_time_03)
elapsed_time_total += elapsed_time_03
# ---------------------------------------------------------------#
# Correcting for lens distortion
# TODO: Adopt distorion that is in TomoPy
# ---------------------------------------------------------------#
print ("Applying Paganin filter to increase phase-contrast")
start_time = timeit.default_timer()
phase_contrast_data = paganin_filter(destriped_data)
elapsed_time_04 = timeit.default_timer() - start_time
print('elapsed time', elapsed_time_04)
elapsed_time_total += elapsed_time_04
# ---------------------------------------------------------------#
print('Total wall time for all methods', elapsed_time_total)
# ---------------------------------------------------------------#
# Data is now ready to be reconstructed
