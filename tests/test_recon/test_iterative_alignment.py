"""Public alignment axes, option forwarding, lifecycle and CUDA integration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import cupy as cp
import numpy as np
import pytest

from httomolibgpu.recon import algorithm


def backend_result(image, views):
    return SimpleNamespace(
        image=image,
        shifts=np.zeros((views, 2)),
        shift_history=np.zeros((2, views, 2)),
        convergence=np.zeros(1),
        residuals=np.zeros(2),
        valid_views=np.ones((1, views), dtype=bool),
        converged=True,
        geometry=object(),
    )


@pytest.mark.parametrize("return_diagnostics", [False, True])
@pytest.mark.parametrize("host_initial", [False, True])
def test_alignment_axes_options_and_exports(
    monkeypatch, host_initial, return_diagnostics
):
    import httomolibgpu
    from httomolibgpu import iterative_alignment

    assert iterative_alignment is algorithm.iterative_alignment
    assert not hasattr(algorithm, "IterativeAlignmentResult")
    assert "IterativeAlignmentResult" not in algorithm.__all__
    assert not hasattr(httomolibgpu, "IterativeAlignmentResult")
    assert "IterativeAlignmentResult" not in httomolibgpu.__all__
    data = cp.zeros((5, 3, 16), dtype=cp.float32)
    angles = np.linspace(0, np.pi, 5)
    native = cp.arange(3 * 12 * 12, dtype=cp.float32).reshape(3, 12, 12)
    initial = native.swapaxes(0, 1).copy()
    if host_initial:
        initial = initial.get()
    saved_initial = initial.copy()
    shifts = np.arange(10).reshape(5, 2).astype(float)
    saved_shifts = shifts.copy()
    backend = MagicMock()
    backend.__enter__.return_value = backend
    expected = backend_result(native, 5)
    backend.align.return_value = expected
    factory = MagicMock(return_value=backend)
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)
    regularisation = algorithm.Regularisation(strength=0.01)
    options = {"nonnegativity": False, "data_solver": "cg"}
    controls = dict(
        iterations=4,
        reconstruction_method="ADMM",
        reconstruction_iterations=2,
        final_reconstruction_iterations=3,
        solver_options=options,
        regularisation=regularisation,
        upsample_factor=20,
        normalization="phase",
        registration_backend="skimage",
        taper=False,
        relaxation=0.5,
        max_shift=2.0,
        tolerance=0.02,
        reference_view=1,
        recon_mask_radius=None,
    )
    result = iterative_alignment(
        data,
        angles,
        center=7.25,
        detector_pad=True,
        recon_size=12,
        initial=initial,
        detector_shifts=shifts,
        return_diagnostics=return_diagnostics,
        **controls,
    )
    ctor = factory.call_args.kwargs
    np.testing.assert_array_equal(ctor["projection_geometry"].angles, -angles)
    np.testing.assert_array_equal(ctor["projection_geometry"].detector_shifts, shifts)
    assert ctor["projection_geometry"].centre_of_rotation == 0.25
    assert ctor["volume_geometry"].shape == (3, 12, 12)
    assert ctor["detector_padding"] == 4
    assert ctor["num_subsets"] == 1
    assert ctor["backend"] == "astra"
    assert backend.align.call_args.args[0] is data
    forwarded = backend.align.call_args.kwargs.copy()
    native_initial = forwarded.pop("initial")
    np.testing.assert_array_equal(cp.asnumpy(cp.asarray(native_initial)), native.get())
    assert forwarded.pop("input_axes") == ["angles", "detY", "detX"]
    assert forwarded == controls
    np.testing.assert_array_equal(
        cp.asnumpy(cp.asarray(initial)), cp.asnumpy(cp.asarray(saved_initial))
    )
    np.testing.assert_array_equal(shifts, saved_shifts)
    assert isinstance(result, tuple)
    assert len(result) == (3 if return_diagnostics else 2)
    image, shifts = result[:2]
    np.testing.assert_array_equal(image.get(), native.swapaxes(0, 1).get())
    assert image.flags.c_contiguous and image.dtype == cp.float32
    assert shifts is expected.shifts
    if return_diagnostics:
        diagnostics = result[2]
        assert isinstance(diagnostics, dict)
        assert set(diagnostics) == {
            "shift_history",
            "convergence",
            "residuals",
            "valid_views",
            "converged",
            "geometry",
        }
        for field, value in diagnostics.items():
            assert value is getattr(expected, field)
    backend.__exit__.assert_called_once_with(None, None, None)


def test_alignment_defaults_and_cleanup_on_failure(monkeypatch):
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.align.side_effect = RuntimeError("registration failed")
    monkeypatch.setattr(
        algorithm, "IterativeReconstructor", MagicMock(return_value=backend)
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        algorithm.iterative_alignment(
            cp.zeros((5, 3, 16), dtype=cp.float32), np.arange(5.0)
        )
    assert backend.align.call_args.kwargs["registration_backend"] == "httomolibgpu"
    assert backend.align.call_args.kwargs["initial"] is None
    backend.__exit__.assert_called_once()
    assert backend.__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize("initial", ["FBP", np.zeros((3, 12, 12), dtype=np.float32)])
def test_alignment_rejects_invalid_initial_order(initial):
    with pytest.raises((TypeError, ValueError), match="initial must"):
        algorithm.iterative_alignment(
            cp.zeros((5, 3, 16), dtype=cp.float32),
            np.arange(5.0),
            recon_size=12,
            initial=initial,
        )


def test_alignment_gpu_recovers_translations(ensure_clean_memory):
    """Exercise geometry updates and GPU registration on synthetic measurements."""
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import IterativeReconstructor

    z, y, x = cp.mgrid[:12, :32, :32]
    truth = (
        cp.exp(-((x - 12) ** 2 + (y - 14) ** 2 + (z - 4) ** 2) / 12)
        + 0.6 * cp.exp(-((x - 21) ** 2 + (y - 20) ** 2 + (z - 8) ** 2) / 8)
    ).astype(cp.float32)
    angles = np.linspace(0, np.pi, 31, endpoint=False)
    offsets = np.column_stack((0.7 * np.sin(3 * angles), 0.5 * np.cos(2 * angles)))
    offsets -= offsets.mean(axis=0)
    with IterativeReconstructor(
        projection_geometry=Parallel3DGeometry.tomography(
            angles=-angles,
            detector_shape=(12, 40),
            detector_shifts=offsets,
        ),
        volume_geometry=Volume3DGeometry(shape=truth.shape),
    ) as rec:
        data = cp.ascontiguousarray(rec.forward(truth).swapaxes(0, 1))
    saved = data.copy()
    image, shifts, diagnostics = algorithm.iterative_alignment(
        data,
        angles,
        center=19.5,
        recon_size=32,
        initial=truth.swapaxes(0, 1),
        iterations=5,
        reconstruction_iterations=1,
        recon_mask_radius=None,
        taper=False,
        upsample_factor=20,
        tolerance=0.02,
        max_shift=1.0,
        return_diagnostics=True,
    )
    assert image.shape == (32, 12, 32)
    assert image.flags.c_contiguous
    assert bool(cp.isfinite(image).all())
    np.testing.assert_array_equal(data.get(), saved.get())
    assert diagnostics["residuals"][-1] < diagnostics["residuals"][0]
    np.testing.assert_allclose(shifts, offsets, atol=0.25)
    np.testing.assert_array_equal(diagnostics["shift_history"][-1], shifts)
    assert diagnostics["valid_views"].all()


@pytest.mark.parametrize("value", [1, "yes", None])
def test_alignment_rejects_invalid_diagnostics_flag(monkeypatch, value):
    factory = MagicMock()
    monkeypatch.setattr(algorithm, "IterativeReconstructor", factory)
    with pytest.raises(ValueError, match="return_diagnostics"):
        algorithm.iterative_alignment(
            cp.zeros((5, 3, 16), dtype=cp.float32),
            np.arange(5.0),
            return_diagnostics=value,
        )
    factory.assert_not_called()
