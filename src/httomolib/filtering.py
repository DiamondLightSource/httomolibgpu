import cupy as cp


# CuPy implementation of Fresnel filter ported from Savu
def fresnel_filter(mat: cp.ndarray, pattern: str, ratio: float,
                   apply_log: bool=True):
    """Apply Fresnel filter.

    Parameters
    ----------
    mat : cp.ndarray
        The data to apply filtering to.

    pattern : str
        Choose 'PROJECTION' for filtering projection, otherwise, will be handled
        generically for other cases.

    ratio : float
        Control the strength of the filter. Greater is stronger.

    apply_log : optional, bool
        Apply negative log function to data being filtered.

    Returns
    -------
    cp.ndarray
        The filtered data.
    """
    if apply_log is True:
        mat = -cp.log(mat)

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
    res_height = min(nrow, padded_height - pad_width)
    padded_width = mat.shape[2] + pad_width*2
    res_width = min(ncol, padded_width - pad_width)
    res = cp.zeros((mat.shape[0], res_height, res_width))

    # Loop over images and apply filter
    for i in range(mat.shape[0]):
        if pattern == "PROJECTION":
            top_drop = 10  # To remove the time stamp in some data
            mat_pad = cp.pad(mat[i][top_drop:], (
            (pad_width + top_drop, pad_width), (pad_width, pad_width)),
                                mode="edge")
            win_pad = cp.pad(window, pad_width, mode="edge")
            mat_dec = \
                cp.fft.ifft2(cp.fft.fft2(mat_pad) / cp.fft.ifftshift(win_pad))
            mat_dec = cp.real(
                mat_dec[pad_width:pad_width + nrow, pad_width:pad_width + ncol])
            res[i] = mat_dec
        else:
            mat_pad = \
                cp.pad(mat[i], ((0, 0), (pad_width, pad_width)), mode='edge')
            win_pad = cp.pad(window, ((0, 0), (pad_width, pad_width)),
                                mode="edge")
            mat_fft = cp.fft.fftshift(cp.fft.fft(mat_pad), axes=1) / win_pad
            mat_dec = cp.fft.ifft(cp.fft.ifftshift(mat_fft, axes=1))
            mat_dec = cp.real(mat_dec[:, pad_width:pad_width + ncol])
            res[i] = mat_dec

    if apply_log is True:
        res = cp.exp(-res)

    return cp.asarray(res, dtype=cp.float32)


def _make_window(height, width, ratio, pattern):
    center_hei = int(cp.ceil((height - 1) * 0.5))
    center_wid = int(cp.ceil((width - 1) * 0.5))
    if pattern == "PROJECTION":
        ulist = (1.0 * cp.arange(0, width) - center_wid) / width
        vlist = (1.0 * cp.arange(0, height) - center_hei) / height
        u, v = cp.meshgrid(ulist, vlist)
        win2d = 1.0 + ratio * (u ** 2 + v ** 2)
    else:
        ulist = (1.0 * cp.arange(0, width) - center_wid) / width
        win1d = 1.0 + ratio * ulist ** 2
        win2d = cp.tile(win1d, (height, 1))
    return win2d
