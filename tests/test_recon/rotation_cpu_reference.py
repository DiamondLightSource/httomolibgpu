from typing import Literal, Optional
import numpy as np
import scipy.ndimage as ndi


def find_center_360_numpy(
    data: np.ndarray,
    ind: Optional[int] = None,
    win_width: int = 10,
    side: Optional[Literal[0, 1]] = None,
    denoise: bool = True,
    norm: bool = False,
    use_overlap: bool = False,
) -> tuple[float, float, Optional[Literal[0, 1]], float]:
    """
    Numpy implementation - for reference in testing the cupy
    production version.
    """
    if data.ndim != 3:
        raise ValueError("A 3D array must be provided")

    # this method works with a 360-degree sinogram.
    if ind is None:
        _sino = data[:, 0, :]
    else:
        _sino = data[:, ind, :]

    (nrow, ncol) = _sino.shape
    nrow_180 = nrow // 2 + 1
    sino_top = _sino[0:nrow_180, :]
    sino_bot = np.fliplr(_sino[-nrow_180:, :])
    (overlap, side, overlap_position) = _find_overlap(
        sino_top, sino_bot, win_width, side, denoise, norm, use_overlap
    )
    if side == 0:
        cor = overlap / 2.0 - 1.0
    else:
        cor = ncol - overlap / 2.0 - 1.0

    return float(cor), float(overlap), side, float(overlap_position)


def _find_overlap(
    mat1, mat2, win_width, side=None, denoise=True, norm=False, use_overlap=False
):
    ncol1 = mat1.shape[1]
    ncol2 = mat2.shape[1]
    win_width = int(np.clip(win_width, 6, min(ncol1, ncol2) // 2))

    if side == 1:
        (list_metric, offset) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=side,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )
        overlap_position = _calculate_curvature(list_metric)[1]
        overlap_position += offset
        overlap = ncol1 - overlap_position + win_width // 2
    elif side == 0:
        (list_metric, offset) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=side,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )
        overlap_position = _calculate_curvature(list_metric)[1]
        overlap_position += offset
        overlap = overlap_position + win_width // 2
    else:
        (list_metric1, offset1) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=1,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )
        (list_metric2, offset2) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=0,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )

        (curvature1, overlap_position1) = _calculate_curvature(list_metric1)
        overlap_position1 += offset1
        (curvature2, overlap_position2) = _calculate_curvature(list_metric2)
        overlap_position2 += offset2

        if curvature1 > curvature2:
            side = 1
            overlap_position = overlap_position1
            overlap = ncol1 - overlap_position + win_width // 2
        else:
            side = 0
            overlap_position = overlap_position2
            overlap = overlap_position + win_width // 2

    return overlap, side, overlap_position


def _search_overlap(
    mat1, mat2, win_width, side, denoise=True, norm=False, use_overlap=False
):
    if denoise is True:
        mat1 = ndi.gaussian_filter(mat1, (2, 2), mode="reflect")
        mat2 = ndi.gaussian_filter(mat2, (2, 2), mode="reflect")
    (nrow1, ncol1) = mat1.shape
    (nrow2, ncol2) = mat2.shape

    if nrow1 != nrow2:
        raise ValueError("Two images are not at the same height!!!")

    win_width = int(np.clip(win_width, 6, min(ncol1, ncol2) // 2 - 1))
    offset = win_width // 2
    win_width = 2 * offset  # Make it even
    ramp_down = np.linspace(1.0, 0.0, win_width, dtype=np.float32)
    ramp_down = ramp_down.reshape((1, ramp_down.size))
    ramp_up = 1.0 - ramp_down

    if side == 1:
        mat2_roi = mat2[:, 0:win_width]
        mat2_roi_wei = mat2_roi * ramp_up
    else:
        mat2_roi = mat2[:, ncol2 - win_width :]
        mat2_roi_wei = mat2_roi * ramp_down

    list_mean2 = np.mean(np.abs(mat2_roi), axis=1, keepdims=True)  # (Nx1)
    list_pos = np.arange(offset, ncol1 - offset)
    num_metric = len(list_pos)
    list_metric = np.empty(num_metric, dtype=np.float32)

    mat1_roi = np.empty((mat1.shape[0], 2 * offset), dtype=np.float32)

    for i, pos in enumerate(list_pos):
        mat1_roi[:] = mat1[:, pos - offset : pos + offset]
        if norm is True:
            list_fact = np.mean(np.abs(mat1_roi), axis=1, keepdims=True)
            np.divide(list_mean2, list_fact, out=list_fact)
            mat1_roi *= list_fact

        if use_overlap is True:
            if side == 1:
                mat_comb = mat1_roi * ramp_down
            else:
                mat_comb = mat1_roi * ramp_up
            mat_comb += mat2_roi_wei
            list_metric[i] = (
                _correlation_metric(mat1_roi, mat2_roi)
                + _correlation_metric(mat1_roi, mat_comb)
                + _correlation_metric(mat2_roi, mat_comb)
            ) / 3.0
        else:
            list_metric[i] = _correlation_metric(mat1_roi, mat2_roi)
    min_metric = np.min(list_metric)
    if min_metric != 0.0:
        list_metric = list_metric / min_metric

    return list_metric, offset


def _calculate_curvature(list_metric):
    radi = 2
    num_metric = list_metric.size
    min_metric_idx = int(np.argmin(list_metric))
    min_pos = int(np.clip(min_metric_idx, radi, num_metric - radi - 1))

    # work mostly on CPU here - we have very small arrays here
    list1 = list_metric[min_pos - radi : min_pos + radi + 1]
    afact1 = np.polyfit(np.arange(0, 2 * radi + 1), list1, 2)[0]
    list2 = list_metric[min_pos - 1 : min_pos + 2]
    (afact2, bfact2, _) = np.polyfit(np.arange(min_pos - 1, min_pos + 2), list2, 2)

    curvature = np.abs(afact1)
    if afact2 != 0.0:
        num = -bfact2 / (2 * afact2)
        if (num >= min_pos - 1) and (num <= min_pos + 1):
            min_pos = num

    return curvature, np.float32(min_pos)


def _correlation_metric(mat1, mat2):
    # pearsonr coefficient
    assert mat1.size == mat2.size, "matrices must be same size"
    X = np.vstack((mat1.reshape((1, mat1.size)), mat2.reshape((1, mat2.size))))
    r = np.corrcoef(X)
    return float(np.abs(1.0 - r[0, 1]))
