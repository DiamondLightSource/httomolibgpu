import os
import numpy as np
import cupy as cp
import scipy
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
import httomolibgpu
from httomolibgpu.misc.raven_filter import raven_filter

import matplotlib.pyplot as plt

# Load the sinogram data
path_lib = os.path.dirname(httomolibgpu.__file__)
in_file = os.path.abspath(
    os.path.join(path_lib, "..", "tests/test_data/", "3600proj_sino.npz")
)
l_infile = np.load(in_file)
sinogram = l_infile["sinogram"]
angles = l_infile["angles"]
sinogram = cp.asarray(sinogram)

sino_shape = sinogram.shape

print("The shape of the sinogram is {}".format(cp.shape(sinogram)))

# Parameters
v0 = 2
n = 4
u0 = 20

# Make a numpy copy
sinogram_padded = np.pad(sinogram.get(), 20, "edge")

# GPU filter
sinogram_gpu_filter = raven_filter(sinogram, u0, n, v0)

# Size
width1 = sino_shape[1] + 2 * 20
height1 = sino_shape[0] + 2 * 20

# Generate filter function
centerx = np.ceil(width1 / 2.0) - 1.0
centery = np.int16(np.ceil(height1 / 2.0) - 1)
row1 = centery - v0
row2 = centery + v0 + 1
listx = np.arange(width1) - centerx
filtershape = 1.0 / (1.0 + np.power(listx / u0, 2 * n))
filtershapepad2d = np.zeros((row2 - row1, filtershape.size))
filtershapepad2d[:] = np.float64(filtershape)
filtercomplex = filtershapepad2d + filtershapepad2d * 1j

# Generate filter objects
a = pyfftw.empty_aligned((height1, width1), dtype='complex128', n=16)
b = pyfftw.empty_aligned((height1, width1), dtype='complex128', n=16)
c = pyfftw.empty_aligned((height1, width1), dtype='complex128', n=16)
d = pyfftw.empty_aligned((height1, width1), dtype='complex128', n=16)
fft_object  = pyfftw.FFTW(a, b, axes=(0, 1))
ifft_object = pyfftw.FFTW(c, d, axes=(0, 1), direction='FFTW_BACKWARD')

sino = fft.fftshift(fft_object(sinogram_padded))
sino[row1:row2] = sino[row1:row2] * filtercomplex
sino = ifft_object(fft.ifftshift(sino))

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,2) 

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0, 0].imshow(sinogram_padded)
axarr[0, 0].set_title('Original sinogram')
axarr[0, 1].imshow(sinogram_padded - sinogram_gpu_filter.get().real)
axarr[0, 1].set_title('Difference of original and GPU filtered')
axarr[1, 0].imshow(sinogram_padded - sino.real)
axarr[1, 0].set_title('Difference of original and CPU filtered')
axarr[1, 1].imshow(sinogram_gpu_filter.get().real - sino.real)
axarr[1, 1].set_title('Difference of GPU and CPU filtered')

plt.show()

