import pathlib

import numpy as np
from httomolib.misc.images import save_to_images
from PIL import Image


def test_save_to_images_8bit(host_data, tmp_path: pathlib.Path):
    # --- Test for bits=8
    save_to_images(host_data, tmp_path / "save_to_images", bits=8)
    #: check that the folder is created
    folder = tmp_path / "save_to_images" / "images" / "images8bit_tif"
    assert folder.exists()

    nfiles = len(list(folder.glob("*")))
    assert nfiles == 180  #: check that the number of files is correct
    assert len(list(folder.glob("*.tif"))) == nfiles  #: check that all files are tif

    #: check that the image size is correct
    imarray = np.array(Image.open(folder / "00015.tif"))
    assert imarray.shape == (128, 160)


def test_save_to_images_4423bit(host_data, tmp_path: pathlib.Path):
    # --- Test for bits=4423
    save_to_images(
        host_data,
        tmp_path / "save_to_images",
        subfolder_name="test",
        file_format="png",
        bits=4423,
    )

    folder = tmp_path / "save_to_images" / "test" / "images32bit_png"
    assert folder.exists()
    assert len(list(folder.glob("*.png"))) == 180
