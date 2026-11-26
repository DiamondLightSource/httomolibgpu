"""Fourier API compatibility and shape-only memory estimation."""

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from httomolibgpu.recon.algorithm import LPRec3d_tomobar
from tomobar.reconstruction.memory import DeviceMemStack


def test_lprec_scaling_conventions(ensure_clean_memory):
    """Retain the wrapper's old scale and allow quantitative scaling explicitly."""
    angles = np.linspace(0, np.pi, 32, endpoint=False)
    data = cp.random.RandomState(42).random_sample((32, 3, 32), dtype=cp.float32)
    options = dict(center=15.5, recon_size=24, recon_mask_radius=None, center_size=0)
    legacy = LPRec3d_tomobar(data, angles, **options)
    quantitative = LPRec3d_tomobar(
        data, angles, gpu_id=0, legacy_scaling=False, **options
    )
    assert legacy.shape == (24, 3, 24)
    assert legacy.flags.c_contiguous
    assert_allclose(
        legacy.get(), quantitative.get() * (8 / np.pi), rtol=2e-5, atol=1e-6
    )


def test_lprec_shape_only_memory_estimation(ensure_clean_memory):
    """Accept shape tuples in HTTomo's estimation context without reconstructing."""
    shape = (32, 3, 32)
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    with DeviceMemStack() as stack:
        estimate = LPRec3d_tomobar(shape, angles, recon_size=24, center_size=0)
        assert estimate == stack.highwater * 1.00625
        assert estimate >= 24 * shape[1] * 24 * np.dtype(np.float32).itemsize
    assert DeviceMemStack.instance() is None


@pytest.mark.parametrize("gpu_id", [0, 2])
@pytest.mark.parametrize("estimating", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_lprec_device_routing_and_restoration(monkeypatch, gpu_id, estimating, fail):
    """Simulate another GPU to check every wrapper stage and error restoration."""
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from httomolibgpu.recon import algorithm

    shape = (4, 2, 16)
    data = shape if estimating else cp.zeros(shape, dtype=cp.float32)
    native = cp.arange(2 * 16 * 16, dtype=cp.float32).reshape(2, 16, 16)
    active = [7]
    stages = []

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            self.previous = active[0]
            active[0] = self.index
            return self

        def __exit__(self, *exc):
            active[0] = self.previous

    def record(stage):
        assert active[0] == gpu_id
        stages.append(stage)

    def pack(array, **kwargs):
        record("pack")
        return cp.require(array, **kwargs)

    def reconstruct(*args, **kwargs):
        record("reconstruct")
        assert "gpu_id" not in kwargs  # Consumed by the wrapper, not FourierOptions.
        if fail:
            raise RuntimeError("reconstruction failed")
        if estimating:
            DeviceMemStack.instance().malloc(8192)
            return native.shape
        return native

    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.FOURIER_INV.side_effect = reconstruct

    def create(**kwargs):
        record("construct")
        assert kwargs["device_index"] == gpu_id
        return backend

    requested = []
    monkeypatch.setattr(
        algorithm.cupywrapper,
        "require_gpu",
        lambda operation, device_index=None: requested.append(device_index),
    )
    monkeypatch.setattr(algorithm, "DirectReconstructor", create)
    monkeypatch.setattr(
        algorithm,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(Device=Device),
            require=pack,
            swapaxes=cp.swapaxes,
            _default_memory_pool=SimpleNamespace(
                free_all_blocks=lambda: record("cleanup")
            ),
        ),
    )
    with ExitStack() as stack:
        if estimating:
            stack.enter_context(DeviceMemStack())
        if fail:
            with pytest.raises(RuntimeError, match="reconstruction failed"):
                algorithm.LPRec3d_tomobar(data, np.arange(shape[0]), gpu_id=gpu_id)
        else:
            result = algorithm.LPRec3d_tomobar(data, np.arange(shape[0]), gpu_id=gpu_id)
            if estimating:
                assert result == DeviceMemStack.instance().highwater * 1.00625
                assert stages == ["construct", "reconstruct", "cleanup"]
            else:
                assert result.shape == (16, 2, 16)
                assert result.flags.c_contiguous
                np.testing.assert_array_equal(result.get(), native.swapaxes(0, 1).get())
                assert stages == ["construct", "reconstruct", "cleanup", "pack"]
        assert active[0] == 7
        assert requested == ([] if estimating else [gpu_id])
    backend.__exit__.assert_called_once()
    assert DeviceMemStack.instance() is None


def test_lprec_second_gpu_execution_and_estimation(ensure_clean_memory):
    """Exercise actual cross-device routing when a second CUDA GPU is available."""
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Requires at least two CUDA devices")
    with cp.cuda.Device(1):
        data = cp.ones((32, 3, 32), dtype=cp.float32)
    with cp.cuda.Device(0):
        angles = np.linspace(0, np.pi, 32, endpoint=False)
        result = LPRec3d_tomobar(data, angles, gpu_id=1, center_size=0)
        assert result.device.id == 1
        assert result.flags.c_contiguous
        assert cp.cuda.runtime.getDevice() == 0
        with DeviceMemStack():
            estimate = LPRec3d_tomobar(data.shape, angles, gpu_id=1, center_size=0)
            assert estimate > 0
        assert cp.cuda.runtime.getDevice() == 0
