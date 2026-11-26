"""ADMM option mapping, native-axis warm starts and resource lifetime."""

from unittest.mock import MagicMock

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from httomolibgpu.recon import algorithm


@pytest.mark.parametrize("data_solver", ["linearised", "cg"])
@pytest.mark.parametrize("initialisation", [None, "FBP", "CGLS", "SIRT"])
@pytest.mark.parametrize("padding", [False, True])
def test_admm_options_warm_start_and_shape(
    monkeypatch, initialisation, padding, data_solver
):
    data = cp.zeros((3, 2, 16), dtype=cp.float32)
    angles = np.linspace(0, np.pi, 3)
    size = 24 if padding and initialisation is not None else 12
    native = cp.arange(2 * size * size, dtype=cp.float32).reshape(2, size, size)
    initial = cp.ones_like(native)
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.ADMM.return_value = native
    backend.CGLS.return_value = initial
    backend.SIRT.return_value = initial
    direct = MagicMock()
    direct.__enter__.return_value = direct
    direct.FBP.return_value = initial
    factory = MagicMock(return_value=backend)
    direct_factory = MagicMock(return_value=direct)
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)
    monkeypatch.setattr(algorithm, "DirectReconstructor", direct_factory)

    result = algorithm.ADMM3d_tomobar(
        data,
        angles,
        center=7.25,
        detector_pad=padding,
        recon_size=12,
        recon_mask_radius=None,
        initialisation=initialisation,
        iterations=3,
        rho=2.0,
        relaxation=1.4,
        data_solver=data_solver,
        cg_iterations=7,
        cg_tolerance=0.002,
        lipschitz_constant=321.0,
        regularisation_type="ROF_TV",
        regularisation_parameter=0.004,
        regularisation_iterations=11,
        regularisation_half_precision=False,
        nonnegativity=True,
    )
    assert_array_equal(result.get(), native.swapaxes(0, 1).get())
    assert result.flags.c_contiguous
    assert result.dtype == cp.float32
    ctor = factory.call_args_list[0].kwargs
    assert_array_equal(ctor["projection_geometry"].angles, -angles)
    assert ctor["projection_geometry"].centre_of_rotation == 0.25
    assert ctor["volume_geometry"].shape == (2, size, size)
    assert ctor["detector_padding"] == (4 if padding else 0)
    assert ctor["num_subsets"] == 1
    assert ctor["device_index"] == 0
    assert ctor["backend"] == "astra"
    options = dict(backend.ADMM.call_args.kwargs)
    assert backend.ADMM.call_args.args[0] is data
    assert options.pop("initial") is (None if initialisation is None else initial)
    regularisation = options.pop("regularisation")
    assert regularisation.method == "ROF_TV"
    assert regularisation.strength == 0.004  # Native ADMM divides this by rho.
    assert regularisation.iterations == 11
    assert regularisation.half_precision is False
    assert options.pop("fidelity").method == "LS"
    assert options.pop("diagnostics").residual_interval is None
    assert options == {
        "input_axes": ["angles", "detY", "detX"],
        "iterations": 3,
        "nonnegativity": True,
        "recon_mask_radius": None,
        "rho": 2.0,
        "relaxation": 1.4,
        "data_solver": data_solver,
        "cg_iterations": 7,
        "cg_tolerance": 0.002,
        "lipschitz_constant": 321.0,
        "weights": None,
    }
    if initialisation == "FBP":
        assert (
            direct_factory.call_args.kwargs["volume_geometry"]
            is ctor["volume_geometry"]
        )
        assert direct_factory.call_args.kwargs["device_index"] == 0
        assert direct.FBP.call_args.args[0] is data
        assert direct.FBP.call_args.kwargs["filter_type"] == "SINC"
        assert direct.FBP.call_args.kwargs["cutoff_freq"] == 0.35
        direct.__exit__.assert_called_once_with(None, None, None)
    else:
        direct_factory.assert_not_called()
    for method, iterations in [("CGLS", 15), ("SIRT", 150)]:
        call = getattr(backend, method)
        if initialisation == method:
            assert call.call_args.args[0] is data
            assert call.call_args.kwargs == {
                "input_axes": ["angles", "detY", "detX"],
                "iterations": iterations,
                "nonnegativity": True,
                "recon_mask_radius": None,
            }
        else:
            call.assert_not_called()
    factory.assert_called_once()
    backend.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("stage", ["FBP", "SIRT", "ADMM"])
def test_admm_closes_after_failure(monkeypatch, stage):
    backend = MagicMock()
    backend.__enter__.return_value = backend
    direct = MagicMock()
    direct.__enter__.return_value = direct
    direct.FBP.return_value = cp.zeros((2, 16, 16), dtype=cp.float32)
    target = direct if stage == "FBP" else backend
    getattr(target, stage).side_effect = RuntimeError("backend failed")
    monkeypatch.setattr(
        algorithm,
        "IterativeReconstructor",
        MagicMock(return_value=backend),
    )
    monkeypatch.setattr(
        algorithm, "DirectReconstructor", MagicMock(return_value=direct)
    )
    with pytest.raises(RuntimeError, match="backend failed"):
        algorithm.ADMM3d_tomobar(
            cp.zeros((3, 2, 16), dtype=cp.float32),
            np.arange(3.0),
            initialisation="SIRT" if stage == "SIRT" else "FBP",
        )
    backend.__exit__.assert_called_once()
    assert backend.__exit__.call_args.args[0] is RuntimeError
    if stage == "FBP":
        direct.__exit__.assert_called_once()
        assert direct.__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize(
    "options",
    [
        {"data_fidelity": "PWLS"},
        {"data_fidelity": "SWLS"},
        {"rho": 0.0},
        {"rho": float("nan")},
        {"relaxation": 2.0},
        {"relaxation": float("inf")},
        {"rho": True},
        {"data_fidelity": "unknown"},
        {"data_solver": "unknown"},
        {"cg_iterations": 0},
        {"cg_iterations": True},
        {"cg_iterations": 1.5},
        {"cg_tolerance": -1.0},
        {"cg_tolerance": 1.0},
        {"cg_tolerance": float("nan")},
        {"lipschitz_constant": 0.0},
        {"lipschitz_constant": float("inf")},
        {"beta_swls": -0.1},
        {"beta_swls": float("nan")},
    ],
)
def test_admm_rejects_invalid_options_before_initialisation(monkeypatch, options):
    factory = MagicMock()
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)
    with pytest.raises(ValueError):
        algorithm.ADMM3d_tomobar(
            cp.zeros((3, 2, 16), dtype=cp.float32), np.arange(3.0), **options
        )
    factory.assert_not_called()


@pytest.mark.parametrize("data_solver", ["linearised", "cg"])
@pytest.mark.parametrize("fidelity", ["LS", "PWLS", "SWLS"])
@pytest.mark.parametrize(
    "initialisation,padding,size",
    [(None, False, 12), ("FBP", True, 24), ("CGLS", False, 12), ("SIRT", False, 12)],
)
def test_admm_native_warm_start_smoke(
    initialisation, padding, size, ensure_clean_memory, fidelity, data_solver
):
    """Execute each warm-start path and enforce the new solver's output contract."""
    data = cp.ones((16, 3, 16), dtype=cp.float32)
    weights = None
    if fidelity == "PWLS":
        weights = cp.linspace(0.2, 4.0, data.size, dtype=cp.float32).reshape(data.shape)
    elif fidelity == "SWLS":
        weights = np.linspace(0.2, 4.0, data.size, dtype=np.float32).reshape(data.shape)
    source_weights = None if weights is None else weights.copy()
    result = algorithm.ADMM3d_tomobar(
        data,
        np.linspace(0, np.pi, 16, endpoint=False),
        center=7.5,
        detector_pad=padding,
        recon_size=12,
        initialisation=initialisation,
        data_fidelity=fidelity,
        weights=weights,
        beta_swls=0.7,
        data_solver=data_solver,
        cg_iterations=3,
        cg_tolerance=1e-5,
        iterations=2,
        regularisation_iterations=5,
        regularisation_parameter=1e-4,
        regularisation_half_precision=False,
        nonnegativity=True,
    )
    assert result.shape == (size, 3, size)
    assert result.dtype == cp.float32
    assert result.flags.c_contiguous
    assert bool(cp.isfinite(result).all())
    assert bool((result >= 0).all())
    assert_array_equal(data.get(), np.ones((16, 3, 16), dtype=np.float32))
    if fidelity == "PWLS":
        assert_array_equal(weights.get(), source_weights.get())
    elif fidelity == "SWLS":
        assert_array_equal(weights, source_weights)


@pytest.mark.parametrize("fidelity", ["PWLS", "SWLS"])
@pytest.mark.parametrize("host_weights", [False, True])
def test_admm_weighted_fidelity_passes_original_weights(
    monkeypatch, fidelity, host_weights
):
    data = cp.ones((3, 2, 16), dtype=cp.float32)
    xp = np if host_weights else cp
    weights = xp.full((6, 2, 16), 4.0, dtype=xp.float32)[::2]
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.ADMM.return_value = cp.zeros((2, 16, 16), dtype=cp.float32)
    monkeypatch.setattr(
        algorithm, "IterativeReconstructor", MagicMock(return_value=backend)
    )
    algorithm.ADMM3d_tomobar(
        data,
        np.arange(3.0),
        initialisation=None,
        data_fidelity=fidelity,
        weights=weights,
        beta_swls=0.7,
        data_solver="cg",
    )
    options = backend.ADMM.call_args.kwargs
    assert options["weights"] is weights
    assert options["fidelity"].method == fidelity
    assert options["fidelity"].beta_swls == 0.7
    assert options["data_solver"] == "cg"
    assert options["cg_iterations"] == 5
    assert options["cg_tolerance"] == 1e-4
    backend.powermethod.assert_not_called()
    backend.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize(
    "fidelity,kind",
    [("LS", "valid"), ("PWLS", "shape"), ("SWLS", "shape"), ("SWLS", "list")],
)
def test_admm_rejects_incompatible_weights_before_initialisation(
    monkeypatch, fidelity, kind
):
    data = cp.ones((3, 2, 16), dtype=cp.float32)
    weights = (
        [1.0]
        if kind == "list"
        else np.ones((1, 2, 16) if kind == "shape" else data.shape, dtype=np.float32)
    )
    factory = MagicMock()
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)
    with pytest.raises((TypeError, ValueError), match="weights"):
        algorithm.ADMM3d_tomobar(
            data, np.arange(3.0), data_fidelity=fidelity, weights=weights
        )
    factory.assert_not_called()
