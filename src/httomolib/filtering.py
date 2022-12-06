import numpy as np


# TODO: NumPy implementation of Fresnel filter from Savu
def fresnel_filter(mat: np.ndarray, pattern: str, ratio: float,
                   apply_log: bool=True):
    """Apply Fresnel filter.

    Parameters
    ----------
    mat : np.ndarray
        The data to apply filtering to.

    pattern : str
        Choose 'PROJECTION' for filtering projection, otherwise, will be handled
        generically for other cases.

    ratio : float
        Control the strength of the filter. Greater is stronger.

    apply_log : optional, bool
        Apply negative log function to data being filtering.

    Returns
    -------
    np.ndarray
        The filtered data.
    """
    if apply_log is True:
        mat = -np.log(mat)

    # Define window
    (depth1, height1, width1) = mat.shape[:3]
    window = _make_window(height1, width1, ratio, pattern)
    pad_width = min(150, int(0.1 * width1))

    # Regardless of working with projections or sinograms, the rows and columns
    # in the images to filter are in the same dimensions of the data: rows in
    # dimension 1, columns in dimension 2 (ie, for projection images, `nrow` is
    # the number of rows in a projection image, and for sinogram images, `nrow`
    # is the number of rows in a sinogram image).
    (_, nrow, ncol) = mat.shape

    # Define array to hold result. Note that, due to the padding applied, the
    # shape of the filtered images are different to the shape of the
    # original/unfiltered images.
    padded_height = mat.shape[1] + pad_width*2
    res_height = \
        nrow if nrow < padded_height - pad_width else padded_height - pad_width
    padded_width = mat.shape[2] + pad_width*2
    res_width = \
        ncol if ncol < padded_width - pad_width else padded_width - pad_width
    res = np.zeros((mat.shape[0], res_height, res_width))

    # Loop over images and apply filter
    for i in range(mat.shape[0]):
        if pattern == "PROJECTION":
            top_drop = 10  # To remove the time stamp in some data
            mat_pad = np.pad(mat[i][top_drop:], (
            (pad_width + top_drop, pad_width), (pad_width, pad_width)),
                                mode="edge")
            win_pad = np.pad(window, pad_width, mode="edge")
            mat_dec = \
                np.fft.ifft2(np.fft.fft2(mat_pad) / np.fft.ifftshift(win_pad))
            mat_dec = np.real(
                mat_dec[pad_width:pad_width + nrow, pad_width:pad_width + ncol])
            res[i] = mat_dec
        else:
            mat_pad = \
                np.pad(mat[i], ((0, 0), (pad_width, pad_width)), mode='edge')
            win_pad = np.pad(window, ((0, 0), (pad_width, pad_width)),
                                mode="edge")
            mat_fft = np.fft.fftshift(np.fft.fft(mat_pad), axes=1) / win_pad
            mat_dec = np.fft.ifft(np.fft.ifftshift(mat_fft, axes=1))
            mat_dec = np.real(mat_dec[:, pad_width:pad_width + ncol])
            res[i] = mat_dec

    if apply_log is True:
        res = np.exp(-res)

    return np.float32(res)


def _make_window(height, width, ratio, pattern):
    center_hei = int(np.ceil((height - 1) * 0.5))
    center_wid = int(np.ceil((width - 1) * 0.5))
    if pattern == "PROJECTION":
        ulist = (1.0 * np.arange(0, width) - center_wid) / width
        vlist = (1.0 * np.arange(0, height) - center_hei) / height
        u, v = np.meshgrid(ulist, vlist)
        win2d = 1.0 + ratio * (u ** 2 + v ** 2)
    else:
        ulist = (1.0 * np.arange(0, width) - center_wid) / width
        win1d = 1.0 + ratio * ulist ** 2
        win2d = np.tile(win1d, (height, 1))
    return win2d
