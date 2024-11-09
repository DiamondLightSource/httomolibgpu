import os
import numpy as np
import cupy as cp
import scipy
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
import httomolibgpu
from httomolibgpu.prep.stripe import raven_filter

import matplotlib.pyplot as plt
import time


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

sinogram_stack = cp.stack([sinogram] * 5, axis=1)

print("The shape of the sinogram stack is {}".format(cp.shape(sinogram_stack)))

# Parameters
v0 = 2
n = 4
u0 = 20

# Make a numpy copy
sinogram_padded = np.pad(sinogram_stack.get(), [(20, 20), (0, 0), (20, 20)], "edge")

start_time = time.time()
# GPU filter
sinogram_gpu_filter = raven_filter(sinogram_stack, u0, n, v0)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

# Size
height, images, width = sinogram_padded.shape

# Generate filter function
centerx = np.ceil(width / 2.0) - 1.0
centery = np.int16(np.ceil(height / 2.0) - 1)
row1 = centery - v0
row2 = centery + v0 + 1
listx = np.arange(width) - centerx
filtershape = 1.0 / (1.0 + np.power(listx / u0, 2 * n))
filtershapepad2d = np.zeros((row2 - row1, filtershape.size))
filtershapepad2d[:] = np.float64(filtershape)
filtercomplex = filtershapepad2d + filtershapepad2d * 1j

# Generate filter objects
a = pyfftw.empty_aligned((height, images, width), dtype='complex128', n=16)
b = pyfftw.empty_aligned((height, images, width), dtype='complex128', n=16)
c = pyfftw.empty_aligned((height, images, width), dtype='complex128', n=16)
d = pyfftw.empty_aligned((height, images, width), dtype='complex128', n=16)
fft_object  = pyfftw.FFTW(a, b, axes=(0, 2))
ifft_object = pyfftw.FFTW(c, d, axes=(0, 2), direction='FFTW_BACKWARD')

sino = fft.fftshift(fft_object(sinogram_padded), axes=(0, 2))
for m in range(sino.shape[1]):
    sino[row1:row2, m] = sino[row1:row2, m] * filtercomplex
sino = ifft_object(fft.ifftshift(sino, axes=(0, 2)))

# Remove padding
sino = sino[20:height-20, :, 20:width-20]

print("--- %s seconds ---" % (time.time() - start_time))

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,2) 

sino_index = 10

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0, 0].imshow(sinogram_stack.get()[:, sino_index, :])
axarr[0, 0].set_title('Original sinogram')
axarr[0, 1].imshow(sinogram_stack.get()[:, sino_index, :] - sinogram_gpu_filter.get().real[:, sino_index, :])
axarr[0, 1].set_title('Difference of original and GPU filtered')
axarr[1, 0].imshow(sinogram_stack.get()[:, sino_index, :] - sino.real[:, sino_index, :])
axarr[1, 0].set_title('Difference of original and CPU filtered')
axarr[1, 1].imshow(sinogram_gpu_filter.get().real[:, sino_index, :] - sino.real[:, sino_index, :])
axarr[1, 1].set_title('Difference of GPU and CPU filtered')

plt.show()

