"""Check realistic synthesis, image scaling and GPU block boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/source/scripts/methods_to_generate_images.py"
)
spec = importlib.util.spec_from_file_location("method_images", SCRIPT)
images = importlib.util.module_from_spec(spec)
spec.loader.exec_module(images)


def test_image_scaling_clips_without_unsigned_wraparound(tmp_path):
    path = tmp_path / "values.png"
    images.save_image(np.array([[-1, 0, 0.5, 1, 2]], np.float32), path, (0, 1))
    np.testing.assert_array_equal(np.asarray(Image.open(path)), [[0, 0, 128, 255, 255]])
    images.save_image(np.ones((2, 2)), path, images.display_limits(np.ones((2, 2))))
    assert np.asarray(Image.open(path)).max() == 0


def test_calibration_indices_do_not_depend_on_projection_count(tmp_path):
    args = images.parser().parse_args(["--output", str(tmp_path)])
    args.row = 3
    args.projection = 100
    writer = images.Images(args)
    writer.calibration(np.arange(2 * 8 * 9).reshape(2, 8, 9), "flats")
    assert Image.open(tmp_path / "flats_proj.png").size == (9, 8)
    assert Image.open(tmp_path / "flats_sino.png").size == (9, 2)


def test_realistic_generation_is_repeatable_and_has_expected_axes(tmp_path):
    pytest.importorskip("tomophantom")
    args = images.parser().parse_args(
        ["--sinogram-shape", "24", "48", "64", "--output", str(tmp_path)]
    )
    flat, art = images.read_settings(args)
    first = images.generate_data(args, flat, art)
    second = images.generate_data(args, flat, art)
    raw, clean, flats, darks, angles, metadata = first
    assert raw.shape == clean.shape == (48, 24, 64)
    assert flats.shape == (20, 24, 64)
    assert darks.shape == (10, 24, 64)
    assert raw.dtype == flats.dtype == darks.dtype == np.uint16
    assert clean.dtype == np.float32 and clean.max() == pytest.approx(1)
    assert angles.shape == (48,)
    assert np.isfinite(clean).all()
    for a, b in zip(first[:5], second[:5]):
        np.testing.assert_array_equal(a, b)
    assert not list(tmp_path.iterdir()), "Synthesis must not write data files"


def test_gpu_median_blocks_preserve_neighbours():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA device")
    from httomolibgpu.misc.corr import median_filter

    args = images.parser().parse_args(["--gpu-rows", "3"])
    rng = np.random.default_rng(3)
    data = rng.normal(size=(12, 10, 16)).astype(np.float32)
    blocked = images.gpu_apply(data, median_filter, args, halo=2, kernel_size=5)
    whole = cp.asnumpy(median_filter(cp.asarray(data), kernel_size=5))
    np.testing.assert_array_equal(blocked, whole)


def test_crop_removes_equal_margins_without_changing_source():
    data = np.arange(100 * 120).reshape(100, 120)
    original = data.copy()
    cropped = images.crop_view(data, 30)
    assert cropped.shape == (40, 60)
    np.testing.assert_array_equal(cropped, data[30:-30, 30:-30])
    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(images.crop_view(data, 0), data)


def test_crop_preserves_short_calibration_axis():
    data = np.arange(10 * 100).reshape(10, 100)
    np.testing.assert_array_equal(images.crop_view(data, 30), data[:, 30:-30])
    with pytest.raises(ValueError, match="non-negative"):
        images.crop_view(data, -1)


def test_dark_flat_figures_capture_transmission_before_log(tmp_path, monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA device")
    args = images.parser().parse_args(
        [
            "--sinogram-shape",
            "10",
            "8",
            "16",
            "--methods",
            "normalisation",
            "--gpu-rows",
            "3",
            "--crop",
            "0",
            "--output",
            str(tmp_path),
        ]
    )
    images.read_settings(args)
    writer = images.Images(args)
    captured = {}
    save_views = writer.views

    def capture(views, name, limits=None):
        captured[name] = {key: value.copy() for key, value in views.items()}
        return save_views(views, name, limits)

    monkeypatch.setattr(writer, "views", capture)
    raw = np.full((8, 10, 16), 600, dtype=np.uint16)
    flats = np.full((3, 10, 16), 1100, dtype=np.uint16)
    darks = np.full((2, 10, 16), 100, dtype=np.uint16)
    images.process_methods(
        raw,
        np.zeros_like(raw, dtype=np.float32),
        flats,
        darks,
        np.arange(8, dtype=np.float32),
        args,
        writer,
    )
    for view in ("proj", "sino"):
        np.testing.assert_allclose(captured["dark_flat_field_correction"][view], 0.5)
        np.testing.assert_allclose(captured["normalisation"][view], -np.log(0.5))
        np.testing.assert_array_equal(
            captured["dark_flat_field_correction_input"][view], 600
        )
    assert Image.open(tmp_path / "dark_flat_field_correction_proj.png").size == (16, 10)
