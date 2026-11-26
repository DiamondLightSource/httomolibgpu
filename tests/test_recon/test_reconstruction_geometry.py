"""CPU geometry checks shared by the reconstruction wrappers."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)

from httomolibgpu.recon import algorithm


@pytest.fixture(autouse=True)
def real_cpu_geometry(monkeypatch):
    # Exercise the helper even when the module imported without an active GPU.
    for cls in (
        Parallel2DGeometry,
        Parallel3DGeometry,
        Volume2DGeometry,
        Volume3DGeometry,
    ):
        monkeypatch.setattr(algorithm, cls.__name__, cls)


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize(
    "width,center,size,padding,offset,expected_size,expected_padding",
    [
        (16, None, None, False, -0.5, 16, 0),
        (17, None, None, False, 0.0, 17, 0),
        (16, 7.25, 12, True, 0.25, 12, 4),
        (17, 7.25, 13, 2, 0.75, 13, 2),
    ],
)
def test_geometry_conventions(
    dimensions, width, center, size, padding, offset, expected_size, expected_padding
):
    angles = np.array([0, 0.2, 1.4], dtype=np.float32)
    original = angles.copy()
    scan, grid, actual_padding = algorithm._build_reconstruction_geometry(
        (3, 2, width), angles, center, padding, size, dimensions=dimensions
    )
    assert_array_equal(scan.angles, -original)
    assert_array_equal(angles, original)
    assert scan.centre_of_rotation == offset
    assert actual_padding == expected_padding
    if dimensions == 3:
        assert isinstance(scan, Parallel3DGeometry)
        assert isinstance(grid, Volume3DGeometry)
        assert scan.detector_shape == (2, width)
        assert grid.shape == (2, expected_size, expected_size)
    else:
        assert isinstance(scan, Parallel2DGeometry)
        assert isinstance(grid, Volume2DGeometry)
        assert scan.detector_shape == (width,)
        assert grid.shape == (expected_size, expected_size)


def test_geometry_requires_supported_dimension():
    with pytest.raises(ValueError, match="dimensions must be 2 or 3"):
        algorithm._build_reconstruction_geometry(
            (3, 2, 16), np.arange(3.0), dimensions=4
        )
