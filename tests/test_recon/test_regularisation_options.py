"""Regulariser choices and native inner options across reconstruction wrappers."""

from unittest.mock import MagicMock

import cupy as cp
import numpy as np
import pytest

from httomolibgpu.recon import algorithm

WRAPPERS = ["FISTA3d_tomobar", "ADMM3d_tomobar", "OSEM3d_tomobar"]
REGULARISERS = [
    ("ROF_TV", {"time_step": 0.01}),
    ("PD_TV", {"tv_type": 1, "lipschitz_constant": 16.0}),
    ("SB_TV", {"tv_type": 1}),
    ("FGP_TV", {"tv_type": 1}),
    ("TGV", {"alpha1": 0.8, "alpha0": 2.5, "lipschitz_constant": 32.0}),
    ("NDF", {"sigma": 0.2, "time_step": 0.01, "penalty_type": 3}),
]


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("regulariser,options", REGULARISERS)
@pytest.mark.parametrize("half_precision", [False, True])
def test_regulariser_options_scaling_and_input_preservation(
    monkeypatch, wrapper, regulariser, options, half_precision
):
    if regulariser == "NDF" and half_precision:
        pytest.skip("NDF float32-only rejection is checked separately")
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.num_subsets = 2
    backend.powermethod.return_value = 11.0
    method = wrapper.split("3d_")[0]
    getattr(backend, method).return_value = cp.ones((3, 12, 12), dtype=cp.float32)
    factory = MagicMock(return_value=backend)
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)
    supplied = dict(options)
    kwargs = {"subsets_number": 2} if method != "ADMM" else {"rho": 2.0}
    result = getattr(algorithm, wrapper)(
        cp.ones((9, 3, 16), dtype=cp.float32),
        np.linspace(0, np.pi, 9),
        recon_size=12,
        iterations=2,
        regularisation_type=regulariser,
        regularisation_parameter=0.002,
        regularisation_iterations=7,
        regularisation_half_precision=half_precision,
        regularisation_options=supplied,
        **kwargs,
    )
    regularisation = getattr(backend, method).call_args.kwargs["regularisation"]
    assert regularisation.method == regulariser
    assert regularisation.iterations == 7
    assert regularisation.half_precision is half_precision
    assert regularisation.strength == pytest.approx(
        0.002 * (2 * 11 if method == "FISTA" else 1)
    )
    for field, value in options.items():
        assert getattr(regularisation, field) == value
    assert supplied == options
    assert result.shape == (12, 3, 12) and result.flags.c_contiguous
    backend.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"regularisation_type": "unknown"}, ValueError, "regularisation_type"),
        ({"regularisation_type": "NDF"}, ValueError, "half_precision"),
        ({"regularisation_options": []}, TypeError, "dictionary"),
        ({"regularisation_options": {"strength": 1}}, ValueError, "strength"),
        ({"regularisation_options": {"method": "TGV"}}, ValueError, "method"),
        ({"regularisation_options": {"typo": 1}}, ValueError, "typo"),
        ({"regularisation_options": {"time_step": 0}}, ValueError, "time_step"),
        (
            {"regularisation_type": "TGV", "regularisation_options": {"tv_type": 1}},
            ValueError,
            "isotropic",
        ),
        (
            {
                "regularisation_type": "NDF",
                "regularisation_half_precision": False,
                "regularisation_options": {"penalty_type": 6},
            },
            ValueError,
            "penalty_type",
        ),
    ],
)
def test_invalid_regularisation_rejected_before_projector_creation(
    monkeypatch, wrapper, kwargs, error, match
):
    factory = MagicMock()
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)
    with pytest.raises(error, match=match):
        getattr(algorithm, wrapper)(
            cp.ones((9, 3, 16), dtype=cp.float32),
            np.linspace(0, np.pi, 9),
            **kwargs,
        )
    factory.assert_not_called()


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("regulariser,options", REGULARISERS)
def test_all_regularisers_run_on_gpu(
    wrapper, regulariser, options, ensure_clean_memory
):
    data = cp.ones((9, 3, 16), dtype=cp.float32)
    original = data.copy()
    kwargs = {"subsets_number": 1} if wrapper != "ADMM3d_tomobar" else {}
    result = getattr(algorithm, wrapper)(
        data,
        np.linspace(0, np.pi, 9, endpoint=False),
        center=7.5,
        recon_size=12,
        recon_mask_radius=None,
        iterations=2,
        regularisation_type=regulariser,
        regularisation_parameter=1e-4,
        regularisation_iterations=3,
        regularisation_half_precision=False,
        regularisation_options=options,
        nonnegativity=True,
        **kwargs,
    )
    assert result.shape == (12, 3, 12)
    assert result.dtype == cp.float32 and result.flags.c_contiguous
    assert bool(cp.isfinite(result).all())
    assert bool((result >= 0).all()) and bool((result > 0).any())
    np.testing.assert_array_equal(data.get(), original.get())
