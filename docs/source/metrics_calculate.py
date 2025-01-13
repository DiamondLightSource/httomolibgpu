import os
import numpy as np
import cupy as cp
import scipy
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
from cupyx.scipy.fft import rfft2, fft2, fftshift
import httomolibgpu
from httomolibgpu.cuda_kernels import load_cuda_module
from httomolibgpu.prep.normalize import normalize
import httomolibgpu.recon.rotation

import matplotlib.pyplot as plt
import math
import time

import httomolibgpu.recon.rotation as rotation
import importlib.util
import sys

def _create_mask_numpy(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    cen_row = np.int16(np.ceil(nrow / 2.0) - 1)
    cen_col = np.int16(np.ceil(ncol / 2.0) - 1)
    drop = min(drop, np.int16(np.ceil(0.05 * nrow)))
    mask = np.zeros((nrow, ncol), dtype="float32")
    for i in range(nrow):
        pos = np.int16(np.round(((i - cen_row) * dv / radius) / du))
        (pos1, pos2) = np.clip(np.sort((-pos + cen_col, pos + cen_col)), 0, ncol - 1)
        mask[i, pos1 : pos2 + 1] = 1.0
    mask[cen_row - drop : cen_row + drop + 1, :] = 0.0
    mask[:, cen_col - 1 : cen_col + 2] = 0.0
    return mask

def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    cen_row = int(math.ceil(nrow / 2.0) - 1)
    cen_col = int(math.ceil(ncol / 2.0) - 1)
    drop = min([drop, int(math.ceil(0.05 * nrow))])

    block_x = 128
    block_y = 1
    block_dims = (block_x, block_y)
    grid_x = (ncol // 2 + 1 + block_x - 1) // block_x
    grid_y = nrow
    grid_dims = (grid_x, grid_y)
    mask = cp.empty((nrow, ncol // 2 + 1), dtype="uint16")
    params = (
        ncol,
        nrow,
        cen_col,
        cen_row,
        cp.float32(du),
        cp.float32(dv),
        cp.float32(radius),
        cp.float32(drop),
        mask,
    )
    module = load_cuda_module("generate_mask")
    kernel = module.get_function("generate_mask")
    kernel(grid_dims, block_dims, params)
    return mask

def _create_mask_new(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    cen_row = int(math.ceil(nrow / 2.0) - 1)
    cen_col = int(math.ceil(ncol / 2.0) - 1)
    drop = min([drop, int(math.ceil(0.05 * nrow))])

    block_x = 128
    block_y = 1
    block_dims = (block_x, block_y)
    grid_x = (ncol + block_x - 1) // block_x
    grid_y = nrow
    grid_dims = (grid_x, grid_y)
    mask = cp.empty((nrow, ncol), dtype="float32")
    params = (
        ncol,
        nrow,
        cen_col,
        cen_row,
        cp.float32(du),
        cp.float32(dv),
        cp.float32(radius),
        cp.float32(drop),
        mask,
    )
    module = load_cuda_module("generate_mask")
    kernel = module.get_function("generate_mask")
    kernel(grid_dims, block_dims, params)
    return mask

# Load the sinogram data
path_lib = os.path.dirname(httomolibgpu.__file__)
in_file = os.path.abspath(
    os.path.join(path_lib, "..", "tests/test_data/", "i12LFOV.npz")
)
l_infile = np.load(in_file)

projdata = cp.asarray(l_infile["projdata"])
flats = cp.asarray(l_infile["flats"])
darks = cp.asarray(l_infile["darks"])
del l_infile

data_normalised = normalize(projdata, flats, darks, minus_log=False)
del flats, darks, projdata

spec = importlib.util.spec_from_file_location("rotation_new", "C:/Work/DiamondLightSource/httomolibgpu/docs/source/rotation_new.py")
rotation_new = importlib.util.module_from_spec(spec)
sys.modules["rotation_new"] = rotation_new
spec.loader.exec_module(rotation_new)

# --- Running the centre of rotation algorithm  ---#
mid_slice = data_normalised.shape[1] // 2

rotation_value = rotation.find_center_vo(data_normalised[:, mid_slice, :]);
new_rotation_value = rotation_new.find_center_vo(data_normalised[:, mid_slice, :]);

print(rotation_value)
print(new_rotation_value)

#subplot(r,c) provide the no. of rows and columns
# f, axarr = plt.subplots(2,2) 

# mask_2 = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
# axarr[0, 0].imshow(mask.get())
# axarr[0, 0].set_title('Original mask')
# axarr[0, 1].imshow(mask_2.get())
# axarr[0, 1].set_title('GPU mask')
# axarr[1, 0].imshow(mask.get() - mask_2.get())
# axarr[1, 0].set_title('Difference of masks')
# axarr[1, 1].imshow(mask.get() - mask_2.get())
# axarr[1, 1].set_title('Difference of masks')

# plt.show()


        