import cupy
from cupyx.scipy.ndimage import median_filter


# CuPy implementation from TomoCuPy
def remove_stripes_tomocupy(data: cupy.ndarray) -> cupy.ndarray:
    """Removes stripes with the method of V. Titarenko (TomoCuPy).
    Args:
        data: A cupy array of projections.
    Returns:
        cupy.ndarray: A cupy array of projections with stripes removed.
    """
    beta = 0.1  # lowering the value increases the filter strength
    gamma = beta * ((1 - beta) / (1 + beta)) ** cupy.abs(
        cupy.fft.fftfreq(data.shape[-1]) * data.shape[-1]
    )
    gamma[0] -= 1
    v = cupy.mean(data, axis=0)
    v = v - v[:, 0:1]
    v = cupy.fft.irfft(cupy.fft.rfft(v) * cupy.fft.rfft(gamma))
    data[:] += v

    return data


# Naive CuPy port of the NumPy implementation in TomoPy
def remove_stripe_based_sorting_cupy(tomo: cupy.ndarray, size: int = None,
                                     dim: int = 1) -> cupy.ndarray:
    """
    Remove full and partial stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (algorithm 3).
    Suitable for removing partial stripes.

    Parameters
    ----------
    tomo : cupy.ndarray
        3D tomographic data.
    size : int
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.

    Returns
    -------
    cupy.ndarray
        Corrected 3D tomographic data.
    """
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    if size is None:
        if tomo.shape[2] > 2000:
            size = 21
        else:
            size = max(5, int(0.01 * tomo.shape[2]))
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_sort(sino, size, matindex, dim)
    return tomo


def _create_matindex(nrow, ncol):
    """
    Create a 2D array of indexes used for the sorting technique.
    """
    listindex = cupy.arange(0.0, ncol, 1.0)
    matindex = cupy.tile(listindex, (nrow, 1))
    return matindex


def _rs_sort(sinogram, size, matindex, dim):
    """
    Remove stripes using the sorting technique.
    """
    sinogram = cupy.transpose(sinogram)
    matcomb = cupy.asarray(cupy.dstack((matindex, sinogram)))
    matsort = cupy.asarray(
        [row[row[:, 1].argsort()] for row in matcomb])
    if dim == 1:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, 1))
    else:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, size))
    matsortback = cupy.asarray(
        [row[row[:, 0].argsort()] for row in matsort])
    sino_corrected = matsortback[:, :, 1]
    return cupy.transpose(sino_corrected)
