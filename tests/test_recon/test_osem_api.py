"""Geometry-based OSEM option mapping, output layout and GPU execution."""

from unittest.mock import MagicMock

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from httomolibgpu.recon import algorithm


@pytest.mark.parametrize(
    "method,half_precision,subsets,padding",
    [("PD_TV", True, 3, True), ("ROF_TV", False, 1, 0)],
)
def test_osem_geometry_regularisation_and_output(
    monkeypatch, method, half_precision, subsets, padding
):
    data = cp.zeros((3, 2, 16), dtype=cp.float32)
    angles = np.linspace(0, np.pi, 3)
    native = cp.arange(2 * 12 * 12, dtype=cp.float32).reshape(2, 12, 12)
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.OSEM.return_value = native
    factory = MagicMock(return_value=backend)
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)

    result = algorithm.OSEM3d_tomobar(
        data,
        angles,
        center=7.25,
        detector_pad=padding,
        recon_size=12,
        recon_mask_radius=None,
        iterations=7,
        subsets_number=subsets,
        regularisation_type=method,
        regularisation_parameter=0.2,
        regularisation_iterations=11,
        regularisation_half_precision=half_precision,
        nonnegativity=False,
    )
    assert_array_equal(result.get(), native.swapaxes(0, 1).get())
    assert result.flags.c_contiguous
    assert result.dtype == cp.float32
    ctor = factory.call_args.kwargs
    assert_array_equal(ctor["projection_geometry"].angles, -angles)
    assert ctor["projection_geometry"].centre_of_rotation == 0.25
    assert ctor["volume_geometry"].shape == (2, 12, 12)
    assert ctor["detector_padding"] == (4 if padding is True else 0)
    assert ctor["num_subsets"] == subsets
    assert ctor["backend"] == "astra"
    assert ctor["device_index"] == 0
    assert backend.OSEM.call_args.args[0] is data
    options = dict(backend.OSEM.call_args.kwargs)
    regularisation = options.pop("regularisation")
    assert regularisation.method == method
    assert regularisation.strength == 0.2
    assert regularisation.iterations == 11
    assert regularisation.half_precision is half_precision
    assert options == {
        "input_axes": ["angles", "detY", "detX"],
        "iterations": 7,
        "nonnegativity": False,
        "recon_mask_radius": None,
    }
    backend.__exit__.assert_called_once_with(None, None, None)


def test_osem_closes_after_solver_failure(monkeypatch):
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.OSEM.side_effect = RuntimeError("solver failed")
    monkeypatch.setattr(
        algorithm, "IterativeReconstructor", MagicMock(return_value=backend)
    )
    with pytest.raises(RuntimeError, match="solver failed"):
        algorithm.OSEM3d_tomobar(
            cp.zeros((3, 2, 16), dtype=cp.float32), np.arange(3.0), subsets_number=1
        )
    backend.__exit__.assert_called_once()
    assert backend.__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize(
    "method,subsets,padding",
    [
        ("PD_TV", 1, False),
        ("PD_TV", 3, True),
        ("ROF_TV", 1, False),
        ("ROF_TV", 3, True),
    ],
)
def test_osem_native_mlem_and_subset_smoke(
    method, subsets, padding, ensure_clean_memory
):
    data = cp.ones((7, 2, 16), dtype=cp.float32)
    result = algorithm.OSEM3d_tomobar(
        data,
        np.linspace(0, np.pi, 7, endpoint=False),
        center=7.5,
        detector_pad=padding,
        recon_size=12,
        iterations=2,
        subsets_number=subsets,
        regularisation_type=method,
        regularisation_parameter=1e-3,
        regularisation_iterations=3,
        regularisation_half_precision=False,
    )
    assert result.shape == (12, 2, 12)
    assert result.dtype == cp.float32
    assert result.flags.c_contiguous
    assert bool(cp.isfinite(result).all())
    assert bool((result >= 0).all())
    assert bool((result > 0).any())
    assert_array_equal(data.get(), np.ones((7, 2, 16), dtype=np.float32))
