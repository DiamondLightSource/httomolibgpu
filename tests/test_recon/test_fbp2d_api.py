"""Compatibility checks for the geometry-based FBP2D wrapper."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from httomolibgpu.recon import algorithm


@pytest.mark.parametrize("fail", [False, True])
def test_fbp2d_slice_orientation_and_cleanup(monkeypatch, fail):
    """Keep slice ordering and orientation, and close even on a backend error."""
    data = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    angles = np.linspace(0, np.pi, 3)
    backend = MagicMock()
    backend.__enter__.return_value = backend
    first = np.arange(16, dtype=np.float32).reshape(4, 4)
    second = first + 100
    backend.FBP.side_effect = [
        first,
        RuntimeError("backend failed") if fail else second,
    ]
    factory = MagicMock(return_value=backend)
    monkeypatch.setattr(algorithm, "DirectReconstructor", factory)

    if fail:
        with pytest.raises(RuntimeError, match="backend failed"):
            algorithm.FBP2d_astra(data, angles)
    else:
        output = algorithm.FBP2d_astra(data, angles)
        assert_array_equal(output[:, 0, :], np.flipud(first))
        assert_array_equal(output[:, 1, :], np.flipud(second))
        assert output.flags.c_contiguous
        assert output.dtype == np.float32

    assert factory.call_count == 1
    kwargs = factory.call_args.kwargs
    assert_array_equal(kwargs["projection_geometry"].angles, -angles)
    assert kwargs["projection_geometry"].centre_of_rotation == -0.5
    assert kwargs["volume_geometry"].shape == (4, 4)
    assert kwargs["detector_padding"] == 0
    for row, call in enumerate(backend.FBP.call_args_list):
        assert_array_equal(call.args[0], data[:, row, :])
    backend.__exit__.assert_called_once()
    assert backend.__exit__.call_args.args[0] is (RuntimeError if fail else None)
