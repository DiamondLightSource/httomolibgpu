"""Public iterative wrapper compatibility across the TomoBAR API migration."""

from unittest.mock import MagicMock

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from httomolibgpu.recon import algorithm


@pytest.mark.parametrize("method", ["SIRT", "CGLS"])
@pytest.mark.parametrize(
    "center,padding,size,expected_padding",
    [(None, False, None, 0), (7.25, True, 12, 4), (7.25, 2, 12, 2)],
)
def test_iterative_geometry_options_and_output(
    monkeypatch, method, center, padding, size, expected_padding
):
    """Preserve geometry conventions, solver controls and public output axes."""
    data = cp.zeros((3, 2, 16), dtype=cp.float32)
    angles = np.linspace(0, np.pi, 3)
    width = size or 16
    native = cp.arange(2 * width * width, dtype=cp.float32).reshape(2, width, width)
    backend = MagicMock()
    backend.__enter__.return_value = backend
    getattr(backend, method).return_value = native
    factory = MagicMock(return_value=backend)
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)

    result = getattr(algorithm, method + "3d_tomobar")(
        data,
        angles,
        center=center,
        detector_pad=padding,
        recon_size=size,
        iterations=7,
        nonnegativity=False,
        recon_mask_radius=None,
    )
    assert_array_equal(result.get(), native.swapaxes(0, 1).get())
    assert result.flags.c_contiguous
    assert result.dtype == cp.float32
    kwargs = factory.call_args.kwargs
    scan = kwargs["projection_geometry"]
    assert_array_equal(scan.angles, -angles)
    assert scan.detector_shape == (2, 16)
    assert scan.centre_of_rotation == 8 - (8 if center is None else center) - 0.5
    assert kwargs["volume_geometry"].shape == (2, width, width)
    assert kwargs["detector_padding"] == expected_padding
    assert kwargs["device_index"] == 0
    assert kwargs["backend"] == "astra"
    assert kwargs["num_subsets"] == 1
    assert getattr(backend, method).call_args.args[0] is data
    assert getattr(backend, method).call_args.kwargs == {
        "input_axes": ["angles", "detY", "detX"],
        "iterations": 7,
        "nonnegativity": False,
        "recon_mask_radius": None,
    }
    backend.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("method", ["SIRT", "CGLS"])
def test_iterative_closes_after_solver_failure(monkeypatch, method):
    """Release projector resources when iteration fails."""
    backend = MagicMock()
    backend.__enter__.return_value = backend
    getattr(backend, method).side_effect = RuntimeError("solver failed")
    monkeypatch.setattr(
        algorithm, "IterativeReconstructor", MagicMock(return_value=backend)
    )
    with pytest.raises(RuntimeError, match="solver failed"):
        getattr(algorithm, method + "3d_tomobar")(
            cp.zeros((3, 2, 16), dtype=cp.float32), np.arange(3.0)
        )
    backend.__exit__.assert_called_once()
    assert backend.__exit__.call_args.args[0] is RuntimeError
