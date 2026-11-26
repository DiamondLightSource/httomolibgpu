"""Regression tests for frequency centring, real weights and valid parameters."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from httomolibgpu.cupywrapper import cp, cupy_run
from httomolibgpu.prep.stripe import raven_filter
from tests.test_prep.raven_reference import raven_reference


@pytest.mark.parametrize("name", ["pad_y", "pad_x", "uvalue", "nvalue", "vvalue"])
@pytest.mark.parametrize("value", [-1, 2**31])
def test_raven_invalid_ranges(name, value):
    # These checks must reject input before any GPU allocation or launch.
    with pytest.raises(ValueError, match=name):
        raven_filter(np.zeros((3, 2, 5), dtype=np.float32), **{name: value})


@pytest.mark.parametrize("name", ["uvalue", "nvalue"])
def test_raven_rejects_zero_cutoff_and_order(name):
    with pytest.raises(ValueError, match=name):
        raven_filter(np.zeros((3, 2, 5), dtype=np.float32), **{name: 0})


@pytest.mark.skipif(not cupy_run, reason="CUDA GPU is required")
@pytest.mark.parametrize("shape", [(16, 3, 32), (17, 2, 31), (16, 2, 31), (17, 3, 32)])
@pytest.mark.parametrize("padding", [0, 3])
def test_raven_matches_cpu_reference(shape, padding):
    data = np.random.default_rng(42).random(shape, dtype=np.float32)
    kwargs = dict(pad_x=padding, pad_y=padding, uvalue=3, nvalue=4, vvalue=2)
    device_data = cp.asarray(data)
    result = raven_filter(device_data, **kwargs)
    assert_allclose(
        cp.asnumpy(result), raven_reference(data, **kwargs), atol=2e-6, rtol=2e-5
    )
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert_allclose(cp.asnumpy(device_data), data)


@pytest.mark.skipif(not cupy_run, reason="CUDA GPU is required")
@pytest.mark.parametrize("nvalue", [1, 4, 1000])
def test_raven_sinusoid_has_expected_attenuation_without_phase_shift(nvalue):
    width = 64
    signal = np.sin(2 * np.pi * 5 * np.arange(width) / width)
    data = np.broadcast_to(signal, (17, 2, width)).astype(np.float32).copy()
    device_data = cp.asarray(data)
    result = raven_filter(
        device_data, pad_y=0, pad_x=0, uvalue=5, nvalue=nvalue, vvalue=0
    )
    # At the cutoff, the real transfer function is exactly 1/2 for every order.
    assert_allclose(cp.asnumpy(result), data * 0.5, atol=2e-6, rtol=2e-5)
    assert_allclose(cp.asnumpy(device_data), data)


@pytest.mark.skipif(not cupy_run, reason="CUDA GPU is required")
def test_raven_preserves_dc_and_frequencies_outside_band():
    y = np.arange(17)[:, None, None]
    x = np.arange(32)[None, None, :]
    data = (2 + np.sin(2 * np.pi * 4 * y / 17) * np.cos(2 * np.pi * 7 * x / 32)).astype(
        np.float32
    )
    result = raven_filter(cp.asarray(data), pad_y=0, pad_x=0, uvalue=3, vvalue=2)
    assert_allclose(cp.asnumpy(result), data, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not cupy_run, reason="CUDA GPU is required")
@pytest.mark.parametrize(
    "dtype, suffix", [(np.complex64, "float"), (np.complex128, "double")]
)
def test_raven_kernel_uses_real_weights(dtype, suffix):
    from httomolibgpu.cuda_kernels import load_cuda_module

    shape = (7, 2, 9)
    height, images, width = shape
    # Non-Hermitian input checks complex weighting directly: taking .real
    # after the inverse FFT could otherwise conceal an imaginary multiplier.
    source = np.full(shape, 1 + 2j, dtype=dtype)
    output = cp.empty(shape, dtype=dtype)
    kernel = load_cuda_module("raven_filter").get_function("raven_filter_" + suffix)
    kernel(
        ((width + 127) // 128, images, height),
        (128, 1, 1),
        (cp.asarray(source), output, width, images, height, 3, 4, 1),
    )
    u = np.arange(width) - width // 2
    v = np.arange(height) - height // 2
    weights = np.ones((height, width))
    weights[np.abs(v) <= 1] = 1 / (1 + (u / 3) ** 8)
    expected = np.fft.ifftshift(source * weights[:, None, :], axes=(0, 2))
    assert_allclose(cp.asnumpy(output), expected, atol=2e-7, rtol=2e-6)
