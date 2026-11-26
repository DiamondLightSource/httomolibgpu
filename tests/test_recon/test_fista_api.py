"""Compatibility of FISTA's public options with the typed TomoBAR API."""

from unittest.mock import MagicMock

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from httomolibgpu.recon import algorithm


@pytest.mark.parametrize(
    "regulariser,half_precision,subsets,padding",
    [("PD_TV", True, 3, True), ("ROF_TV", False, 1, 0)],
)
def test_fista_options_and_proximal_strength(
    monkeypatch, regulariser, half_precision, subsets, padding
):
    """Retain per-step smoothing strength, solver settings and array layout."""
    data = cp.zeros((3, 2, 16), dtype=cp.float32)
    angles = np.linspace(0, np.pi, 3)
    native = cp.arange(2 * 12 * 12, dtype=cp.float32).reshape(2, 12, 12)
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.num_subsets = subsets
    backend.powermethod.return_value = 20.0
    backend.FISTA.return_value = native
    factory = MagicMock(return_value=backend)
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)

    result = algorithm.FISTA3d_tomobar(
        data,
        angles,
        center=7.25,
        detector_pad=padding,
        recon_size=12,
        recon_mask_radius=None,
        iterations=7,
        subsets_number=subsets,
        regularisation_type=regulariser,
        regularisation_parameter=0.002,
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
    backend.powermethod.assert_called_once_with()
    assert backend.FISTA.call_args.args[0] is data
    options = backend.FISTA.call_args.kwargs
    regularisation = options.pop("regularisation")
    fidelity = options.pop("fidelity")
    assert fidelity.method == "LS"
    assert regularisation.method == regulariser
    assert regularisation.iterations == 11
    assert regularisation.half_precision is half_precision
    assert regularisation.strength / (subsets * 20.0) == pytest.approx(0.002)
    assert options == {
        "input_axes": ["angles", "detY", "detX"],
        "iterations": 7,
        "lipschitz_constant": 20.0,
        "nonnegativity": False,
        "recon_mask_radius": None,
    }
    backend.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("stage", ["powermethod", "FISTA"])
def test_fista_releases_resources_on_failure(monkeypatch, stage):
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.num_subsets = 1
    backend.powermethod.return_value = 20.0
    getattr(backend, stage).side_effect = RuntimeError("backend failed")
    monkeypatch.setattr(
        algorithm, "IterativeReconstructor", MagicMock(return_value=backend)
    )
    with pytest.raises(RuntimeError, match="backend failed"):
        algorithm.FISTA3d_tomobar(
            cp.zeros((3, 2, 16), dtype=cp.float32), np.arange(3.0), subsets_number=1
        )
    backend.__exit__.assert_called_once()
    assert backend.__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize("fidelity", ["PWLS", "SWLS", "unknown"])
def test_fista_rejects_fidelity_without_weights(monkeypatch, fidelity):
    factory = MagicMock()
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)
    with pytest.raises(ValueError, match="data_fidelity"):
        algorithm.FISTA3d_tomobar(
            cp.zeros((3, 2, 16), dtype=cp.float32),
            np.arange(3.0),
            data_fidelity=fidelity,
        )
    factory.assert_not_called()
