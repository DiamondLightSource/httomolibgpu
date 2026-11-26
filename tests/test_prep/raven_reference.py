"""Independent CPU reference using Raven's real transfer function."""

import numpy as np
from scipy.fft import fft2, ifft2


def raven_reference(
    data, pad_y=20, pad_x=20, pad_method="edge", uvalue=20, nvalue=4, vvalue=2
):
    padded = np.pad(data, ((pad_y, pad_y), (0, 0), (pad_x, pad_x)), mode=pad_method)
    height, _, width = padded.shape
    u = np.rint(np.fft.fftfreq(width) * width)
    v = np.rint(np.fft.fftfreq(height) * height)
    weights = np.ones((height, width))
    with np.errstate(over="ignore"):
        weights[np.abs(v) <= vvalue] = 1 / (1 + (u / uvalue) ** (2 * nvalue))
    result = ifft2(fft2(padded, axes=(0, 2)) * weights[:, None, :], axes=(0, 2)).real
    return np.ascontiguousarray(
        result[pad_y : height - pad_y, :, pad_x : width - pad_x], dtype=np.float32
    )
