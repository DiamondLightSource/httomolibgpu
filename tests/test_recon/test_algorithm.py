import numpy as np
import pytest
from numpy.testing import assert_allclose

from httomolib.recon.algorithm import reconstruct_tomobar

from tomopy.prep.normalize import normalize

in_file = 'tests/test_data/tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']
host_flats = datafile['flats']
host_darks = datafile['darks']


def test_reconstruct_tomobar():
    cor = 79.5 #: center of rotation for tomo_standard
    data = normalize(host_data, host_flats, host_darks, cutoff=15.0)
    angles = np.linspace(0. * np.pi / 180., 180. * np.pi / 180., data.shape[0])

    #--- reconstructing the data ---#
    recon_data = reconstruct_tomobar(data, angles, cor, algorithm="FBP3D_host")
    assert recon_data.shape == (128, 160, 160)
    for _ in range(3):
        assert_allclose(np.mean(recon_data), -0.00047186256, rtol=1e-06)
        assert_allclose(np.std(recon_data), 0.0034316075, rtol=1e-06)
