import numpy as xp
import numpy as np

try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability

    except xp.cuda.runtime.CUDARuntimeError:
        print("CuPy library is a major dependency for HTTomolibgpu, please install")
        import numpy as np
except ImportError:
    import numpy as np
import nvtx
from typing import Literal, Optional, Tuple, Union

__all__ = [
    "rescale_to_int",
]


@nvtx.annotate()
def rescale_to_int(
    data: xp.ndarray,
    perc_range_min: float = 0.0,
    perc_range_max: float = 100.0,
    bits: Literal[8, 16, 32] = 8,
    glob_stats: Optional[Tuple[float, float, float, int]] = None,
):
    """
    Rescales the data and converts it fit into the range of an unsigned integer type
    with the given number of bits.

    Parameters
    ----------
    data : cp.ndarray
        Required input data array, on GPU
    perc_range_min: float, optional
        The lower cutoff point in the input data, in percent of the data range (defaults to 0).
        The lower bound is computed as min + perc_range_min/100*(max-min)
    perc_range_max: float, optional
        The upper cutoff point in the input data, in percent of the data range (defaults to 100).
        The upper bound is computed as min + perc_range_max/100*(max-min)
    bits: Literal[8, 16, 32], optional
        The number of bits in the output integer range (defaults to 8).
        Allowed values are:
        - 8 -> uint8
        - 16 -> uint16
        - 32 -> uint32
    glob_stats: tuple, optional
        Global statistics of the full dataset (beyond the data passed into this call).
        It's a tuple with (min, max, sum, num_items). If not given, the min/max is
        computed from the given data.

    Returns
    -------
    cp.ndarray
        The original data, clipped to the range specified with the perc_range_min and
        perc_range_max, and scaled to the full range of the output integer type
    """

    if bits == 8:
        output_dtype: Union[type[np.uint8], type[np.uint16], type[np.uint32]] = np.uint8
    elif bits == 16:
        output_dtype = np.uint16
    else:
        output_dtype = np.uint32

    # get the min and max integer values of the output type
    output_min = np.iinfo(output_dtype).min
    output_max = np.iinfo(output_dtype).max

    if not isinstance(glob_stats, tuple):
        min_value = float(xp.min(data))
        max_value = float(xp.max(data))
    else:
        min_value = glob_stats[0]
        max_value = glob_stats[1]

    range_intensity = max_value - min_value
    input_min = (perc_range_min * (range_intensity) / 100) + min_value
    input_max = (perc_range_max * (range_intensity) / 100) + min_value

    factor = (output_max - output_min) / (input_max - input_min)

    res = xp.empty(data.shape, dtype=output_dtype)
    rescale_kernel = xp.ElementwiseKernel(
        "T x, raw T input_min, raw T input_max, raw T factor",
        "O out",
        """
        T x_clean = isnan(x) || isinf(x) ? T(0) : x;
        T x_clipped = x_clean < input_min ? input_min : (x_clean > input_max ? input_max : x_clean);
        T x_rebased = x_clipped - input_min;
        out = O(x_rebased * factor);
        """,
        "rescale_to_int",
    )
    rescale_kernel(data, input_min, input_max, factor, res)
    return res
