"""Select an array backend while allowing imports on machines without CUDA.

The NumPy fallback supports imports and CPU utilities; it does not implement
GPU-only algorithms. Backend selection lasts for the lifetime of this module.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

try:
    import nvtx
except ImportError:
    nvtx = Mock()

cupy_run = False
_gpu_error = None
try:
    import cupy as cp
except ImportError as exc:
    _gpu_error = str(exc)
else:
    try:
        # Check actual allocation on the current device, without selecting GPU 0.
        cp.empty(1, dtype=cp.uint8)
    except cp.cuda.runtime.CUDARuntimeError as exc:
        _gpu_error = str(exc)
    else:
        cupy_run = True

if cupy_run:
    from cupyx.scipy.fft import next_fast_len
else:
    import numpy as cp
    from scipy.fft import next_fast_len

    logging.getLogger(__name__).info("CUDA unavailable: %s", _gpu_error)


def require_gpu(operation: str, device_index: int | None = None) -> None:
    """Require CUDA execution, checking the requested device when supplied."""
    if not cupy_run:
        raise RuntimeError(
            f"{operation} requires CuPy and an accessible CUDA GPU: {_gpu_error}"
        )
    if device_index is not None:
        try:
            with cp.cuda.Device(device_index):
                cp.empty(1, dtype=cp.uint8)
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise RuntimeError(
                f"{operation} cannot use CUDA device {device_index}: {exc}"
            ) from exc


class _UnavailableGPU:
    """Import-time stand-in for a GPU dependency on a CPU-only machine."""

    def __init__(self, dependency: str):
        self._dependency = dependency

    def __call__(self, *args, **kwargs):
        require_gpu(self._dependency)
        raise RuntimeError(f"GPU dependency {self._dependency} is unavailable")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(self)(f"{self._dependency}.{name}")

    def __repr__(self):
        return f"<unavailable GPU dependency {self._dependency}>"
