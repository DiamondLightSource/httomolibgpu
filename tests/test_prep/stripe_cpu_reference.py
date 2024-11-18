import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fft

def raven_filter_numpy(
        sinogram,
        uvalue: int = 20,
        nvalue: int = 4,
        vvalue: int = 2,
        pad_y: int = 20,
        pad_x: int = 20,
        pad_method: str = "edge"):
    
    # Parameters
    v0 = vvalue
    n = nvalue
    u0 = uvalue

    # Make a padded copy
    sinogram_padded = np.pad(sinogram, ((pad_y,pad_y), (0, 0), (pad_x,pad_x)), pad_method)
    
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
    sinogram = sino[pad_y:height-pad_y, :, pad_x:width-pad_x]

    return sinogram.real