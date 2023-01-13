import os

import numpy as np
from PIL import Image

from httomolib.misc.images import save_to_images

# --- Tomo standard data ---#
in_file = 'tests/test_data/tomo_standard.npz'
datafile = np.load(in_file)
host_data = datafile['data']


def test_save_to_images():    
    #--- Test for bits=8
    save_to_images(host_data, 'save_to_images', bits=8)
    #: check that the folder is created
    assert os.path.exists('save_to_images/images/images8bit_tif/')

    #: check that the number of files is correct
    nfiles = len(os.listdir('save_to_images/images/images8bit_tif/'))
    assert nfiles == 180

    #: check that all files are tif
    for file in os.listdir('save_to_images/images/images8bit_tif/'):
        assert file.endswith('.tif')

    #: check that the image size is correct
    imarray = np.array(
        Image.open('save_to_images/images/images8bit_tif/00015.tif'))
    assert imarray.shape == (128, 160)

    #--- Test for bits=4423
    save_to_images(host_data, 'save_to_images',
        subfolder_name='test', file_format='png', bits=4423)
    assert os.path.exists('save_to_images/test/images32bit_png/')
    for file in os.listdir('save_to_images/test/images32bit_png/'):
        assert file.endswith('.png')
