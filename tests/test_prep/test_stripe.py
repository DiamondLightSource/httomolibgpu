import time
import cupy as cp
from cupy.cuda import nvtx
import numpy as np
import pytest
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
from httomolibgpu.prep.normalize import normalize
from httomolibgpu.prep.stripe import (
    remove_stripe_based_sorting,
    remove_stripe_ti,
    remove_all_stripe,
    raven_filter,
)
from numpy.testing import assert_allclose

def raven_filter_cpu(
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
    sinogram_padded = cp.pad(sinogram, ((pad_y,pad_y), (0, 0), (pad_x,pad_x)), pad_method).get()
    
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

def test_remove_stripe_ti_on_data(data, flats, darks):
    # --- testing the CuPy implementation from TomoCupy ---#
    data = normalize(data, flats, darks, cutoff=10, minus_log=True)

    data_after_stripe_removal = remove_stripe_ti(cp.copy(data)).get()

    assert_allclose(np.mean(data_after_stripe_removal), 0.28924704, rtol=1e-05)
    assert_allclose(
        np.mean(data_after_stripe_removal, axis=(1, 2)).sum(), 52.064457, rtol=1e-06
    )
    assert_allclose(np.median(data_after_stripe_removal), 0.026177486, rtol=1e-05)
    assert_allclose(np.max(data_after_stripe_removal), 2.715983, rtol=1e-05)
    assert data_after_stripe_removal.flags.c_contiguous

    data = None  #: free up GPU memory
    # make sure the output is float32
    assert data_after_stripe_removal.dtype == np.float32


# def test_remove_stripe_ti_on_flats(host_flats):
#     #: testing that numpy uint16 arrays can be passed
#     corrected_data = remove_stripe_ti(np.copy(host_flats))
#     assert_allclose(np.mean(corrected_data), 976.558447, rtol=1e-7)
#     assert_allclose(np.mean(corrected_data, axis=(1, 2)).sum(), 19531.168945, rtol=1e-7)
#     assert_allclose(np.median(corrected_data), 976.0, rtol=1e-7)


# def test_remove_stripe_ti_numpy_vs_cupy_on_random_data():
#     host_data = np.random.random_sample(size=(181, 5, 256)).astype(np.float32) * 2.0
#     corrected_host_data = remove_stripe_ti(np.copy(host_data))
#     corrected_data = remove_stripe_ti(
#         cp.copy(cp.asarray(host_data, dtype=cp.float32))
#     ).get()

#     assert_allclose(np.sum(corrected_data), np.sum(corrected_host_data), rtol=1e-6)
#     assert_allclose(
#         np.median(corrected_data), np.median(corrected_host_data), rtol=1e-6
#     )

def test_stripe_removal_sorting_cupy(data, flats, darks):
    # --- testing the CuPy port of TomoPy's implementation ---#
    data = normalize(data, flats, darks, cutoff=10, minus_log=True)
    corrected_data = remove_stripe_based_sorting(data).get()

    data = None  #: free up GPU memory
    assert_allclose(np.mean(corrected_data), 0.288198, rtol=1e-06)
    assert_allclose(np.mean(corrected_data, axis=(1, 2)).sum(), 51.87565, rtol=1e-06)
    assert_allclose(np.sum(corrected_data), 1062413.6, rtol=1e-06)

    # make sure the output is float32
    assert corrected_data.dtype == np.float32
    assert corrected_data.flags.c_contiguous

def test_stripe_raven_cupy(data, flats, darks):
    # --- testing the CuPy port of TomoPy's implementation ---#

    data = normalize(data, flats, darks, cutoff=10, minus_log=True)

    data_after_raven_gpu = raven_filter(cp.copy(data)).get()
    data_after_raven_cpu = raven_filter_cpu(cp.copy(data))

    assert_allclose(data_after_raven_cpu, data_after_raven_gpu, 0, atol=4e-01)

    data = None  #: free up GPU memory
    # make sure the output is float32
    assert data_after_raven_gpu.dtype == np.float32
    assert data_after_raven_gpu.shape == data_after_raven_cpu.shape

@pytest.mark.perf
def test_stripe_removal_sorting_cupy_performance(ensure_clean_memory):
    data_host = (
        np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    )
    data = cp.asarray(data_host, dtype=np.float32)

    # do a cold run first
    remove_stripe_based_sorting(cp.copy(data))

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        # have to take copy, as data is modified in-place
        remove_stripe_based_sorting(cp.copy(data))
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


@pytest.mark.perf
def test_remove_stripe_ti_performance(ensure_clean_memory):
    data_host = (
        np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    )
    data = cp.asarray(data_host, dtype=np.float32)

    # do a cold run first
    remove_stripe_ti(cp.copy(data))

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        # have to take copy, as data is modified in-place
        remove_stripe_ti(cp.copy(data))
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms

@pytest.mark.perf
def test_raven_filter_performance(ensure_clean_memory):
    data_host = (
        np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    )
    data = cp.asarray(data_host, dtype=np.float32)

    # do a cold run first
    raven_filter(cp.copy(data))

    dev = cp.cuda.Device()
    dev.synchronize()

    start = time.perf_counter_ns()
    nvtx.RangePush("Core")
    for _ in range(10):
        # have to take copy, as data is modified in-place
        raven_filter(cp.copy(data))
    nvtx.RangePop()
    dev.synchronize()
    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms

@pytest.mark.perf
def test_raven_filter_cpu_performance(ensure_clean_memory):
    data_host = (
        np.random.random_sample(size=(1801, 5, 2560)).astype(np.float32) * 2.0 + 0.001
    )
    data = cp.asarray(data_host, dtype=np.float32)

    # do a cold run first
    raven_filter_cpu(cp.copy(data))

    start = time.perf_counter_ns()
    for _ in range(10):
        raven_filter_cpu(cp.copy(data))

    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    assert "performance in ms" == duration_ms


def test_remove_all_stripe_on_data(data, flats, darks):
    # --- testing the CuPy implementation from TomoCupy ---#
    data = normalize(data, flats, darks, cutoff=10, minus_log=True)

    data_after_stripe_removal = remove_all_stripe(cp.copy(data)).get()

    assert_allclose(np.mean(data_after_stripe_removal), 0.266914, rtol=1e-05)
    assert_allclose(
        np.mean(data_after_stripe_removal, axis=(1, 2)).sum(), 48.04459, rtol=1e-06
    )
    assert_allclose(np.median(data_after_stripe_removal), 0.015338, rtol=1e-04)
    assert_allclose(np.max(data_after_stripe_removal), 2.298123, rtol=1e-05)

    data = None  #: free up GPU memory
    # make sure the output is float32
    assert data_after_stripe_removal.dtype == np.float32
    assert data_after_stripe_removal.flags.c_contiguous
