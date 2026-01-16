#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

import os
import sys
import argparse
from typing import Union
import numpy as np
import cupy as cp
from PIL import Image


# usage : python -m methods_to_generate_images -i /home/algol/Documents/DEV/httomolibgpu/tests/test_data/synthdata_nxtomo1.npz -o /home/algol/Documents/DEV/httomolibgpu/docs/source/_static/auto_images_methods


def __save_res_to_image(
    result: np.ndarray,
    output_folder: str,
    methods_name: str,
    slice_numb: int,
    max_scale: Union[float, None] = None,
    min_scale: Union[float, None] = None,
):
    """Saving the result of the method into the image

    Args:
        result (np.ndarray): A numpy array to save
        output_folder (str): Path to output folder
        methods_name (str): the name of the method
        slice_numb (int): Slice number to save
        max_scale (float, None): if a specific rescaling needed
        min_scale (float, None): if a specific rescaling needed

    Returns:
    """
    if min_scale is None and max_scale is None:
        result = (result - result.min()) / (result.max() - result.min())
    else:
        result = (result - min_scale) / (max_scale - min_scale)
    resut_to_save = Image.fromarray((result[:, slice_numb, :] * 255).astype(np.uint8))
    resut_to_save.save(output_folder + methods_name + "_sino.png")

    resut_to_save = Image.fromarray((result[slice_numb, :, :] * 255).astype(np.uint8))
    resut_to_save.save(output_folder + methods_name + "_proj.png")


def run_methods(path_to_data: str, output_folder: str) -> int:
    """function that selectively runs the methods with the test data provided and save the result as an image

    Args:
        path_to_data: A path to the test data.
        output_folder: path to output folder with the saved images.

    Returns:
        returns zero if the processing is successful
    """
    try:
        fileloaded = np.load(path_to_data)
    except OSError:
        print("Cannot find/open file {}".format(path_to_data))
        sys.exit()
    print("Unpacking data file...\n")
    proj_raw = fileloaded["proj_raw"]
    proj_ground_truth = fileloaded["proj_ground_truth"]
    phantom = fileloaded["phantom"]
    flats = fileloaded["flats"]
    darks = fileloaded["darks"]
    angles_degr = fileloaded["angles"]

    slice_numb = 40

    __save_res_to_image(
        proj_raw,
        output_folder,
        methods_name="raw_data",
        slice_numb=slice_numb,
    )

    __save_res_to_image(
        proj_ground_truth,
        output_folder,
        methods_name="proj_ground_truth",
        slice_numb=slice_numb,
    )
    __save_res_to_image(darks, output_folder, methods_name="darks", slice_numb=10)
    __save_res_to_image(flats, output_folder, methods_name="flats", slice_numb=10)

    print("Executing methods from the HTTomolibGPU library\n")

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods_name = "normalisation"
    print("___{}___".format(methods_name))
    from httomolibgpu.prep.normalize import dark_flat_field_correction, minus_log

    data_normalized = dark_flat_field_correction(
        cp.asarray(proj_raw), cp.asarray(flats), cp.asarray(darks)
    )

    data_normalized = minus_log(data_normalized)
    data_normalized_np = data_normalized.get()

    max_scale_data_normalized = np.max(data_normalized_np)
    min_scale_data_normalized = np.min(data_normalized_np)
    assert max_scale_data_normalized > 1.062
    assert min_scale_data_normalized < -0.04
    assert np.sum(data_normalized_np) > 41711
    # __save_res_to_image(data_normalized_np, output_folder, methods_name, slice_numb)

    methods_name = "rescale_to_int"
    print("___{}___".format(methods_name))
    from httomolibgpu.misc.rescale import rescale_to_int

    rescaled_data = rescale_to_int(
        data_normalized, perc_range_min=0, perc_range_max=100, bits=8
    )
    rescaled_data_np = rescaled_data.get()
    resut_to_save = Image.fromarray(
        (rescaled_data_np[slice_numb, :, :] * 255).astype(np.uint8)
    )
    resut_to_save.save(output_folder + methods_name + "_proj_0_to_100.png")

    rescaled_data = rescale_to_int(
        data_normalized, perc_range_min=10, perc_range_max=90, bits=8
    )
    rescaled_data_np = rescaled_data.get()
    resut_to_save = Image.fromarray(
        (rescaled_data_np[slice_numb, :, :] * 255).astype(np.uint8)
    )
    resut_to_save.save(output_folder + methods_name + "_proj_10_to_90.png")

    rescaled_data = rescale_to_int(
        data_normalized, perc_range_min=30, perc_range_max=70, bits=8
    )
    rescaled_data_np = rescaled_data.get()
    resut_to_save = Image.fromarray(
        (rescaled_data_np[slice_numb, :, :] * 255).astype(np.uint8)
    )
    resut_to_save.save(output_folder + methods_name + "_proj_30_to_70.png")

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods_name = "remove_outlier"
    print("___{}___".format(methods_name))
    from httomolibgpu.misc.corr import (
        remove_outlier,
    )

    proj_raw_mod = np.copy(proj_raw)
    proj_raw_mod[slice_numb, 20:22, 20:22] = 0.0
    proj_raw_mod[slice_numb, 60:65, 70:71] = 0.0
    proj_raw_mod[200:210, slice_numb, 200] = 0.0

    __save_res_to_image(
        proj_raw_mod,
        output_folder,
        methods_name=methods_name + "_input",
        slice_numb=slice_numb,
    )

    res_cp = remove_outlier(
        cp.asarray(proj_raw_mod, dtype=cp.float32),
        kernel_size=5,
        dif=2000,
    )
    res_np = res_cp.get()

    __save_res_to_image(
        res_np,
        output_folder,
        methods_name,
        slice_numb,
        max_scale=np.max(proj_raw_mod),
        min_scale=np.min(proj_raw_mod),
    )
    __save_res_to_image(
        np.abs(proj_raw_mod - res_np),
        output_folder,
        methods_name=methods_name + "_res",
        slice_numb=slice_numb,
    )
    del res_cp, res_np, proj_raw_mod

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods_name = "median_filter"
    print("___{}___".format(methods_name))
    from httomolibgpu.misc.corr import (
        median_filter,
    )

    res_cp = median_filter(
        cp.asarray(data_normalized, dtype=cp.float32),
        kernel_size=5,
    )
    res_np = res_cp.get()

    __save_res_to_image(
        res_np,
        output_folder,
        methods_name,
        slice_numb,
        max_scale=max_scale_data_normalized,
        min_scale=min_scale_data_normalized,
    )
    __save_res_to_image(
        np.abs(data_normalized_np - res_np),
        output_folder,
        methods_name=methods_name + "_res",
        slice_numb=slice_numb,
        max_scale=0.05,
        min_scale=0,
    )
    del res_cp, res_np

    methods_name = "raven_filter"
    print("___{}___".format(methods_name))
    from tomophantom.artefacts import artefacts_mix
    from httomolibgpu.prep.stripe import (
        raven_filter,
    )

    _stripes_ = {
        "stripes_percentage": 2.0,
        "stripes_maxthickness": 3,
        "stripes_intensity": 0.3,
        "stripes_type": "full",
        "stripes_variability": 0.005,
    }
    data_normalized_np_artefacts = artefacts_mix(
        np.swapaxes(data_normalized_np, 0, 1), **_stripes_
    )
    data_normalized_np_artefacts = np.swapaxes(data_normalized_np_artefacts, 0, 1)

    data_after_raven_gpu = raven_filter(
        cp.asarray(data_normalized_np_artefacts, dtype=cp.float32),
        uvalue=20,
        nvalue=2,
        vvalue=2,
    ).get()
    slice_numb = 64

    max_scale_data_normalized_s = np.max(data_normalized_np_artefacts)
    min_scale_data_normalized_s = -0.01

    __save_res_to_image(
        data_normalized_np_artefacts,
        output_folder,
        "data_stripes_added",
        slice_numb,
        max_scale=max_scale_data_normalized_s,
        min_scale=min_scale_data_normalized_s,
    )

    __save_res_to_image(
        data_after_raven_gpu,
        output_folder,
        methods_name,
        slice_numb,
        max_scale=max_scale_data_normalized_s,
        min_scale=min_scale_data_normalized_s,
    )
    __save_res_to_image(
        np.abs(data_after_raven_gpu - data_normalized_np_artefacts),
        output_folder,
        methods_name=methods_name + "_res",
        slice_numb=slice_numb,
        max_scale=0.1,
        min_scale=0,
    )
    del data_after_raven_gpu

    methods_name = "remove_stripe_based_sorting"
    print("___{}___".format(methods_name))
    from httomolibgpu.prep.stripe import (
        remove_stripe_based_sorting,
    )

    data_after_stripe_based_sorting = remove_stripe_based_sorting(
        cp.asarray(data_normalized_np_artefacts, dtype=cp.float32), size=21, dim=1
    )

    __save_res_to_image(
        data_after_stripe_based_sorting.get(),
        output_folder,
        methods_name,
        slice_numb,
        max_scale=max_scale_data_normalized_s,
        min_scale=min_scale_data_normalized_s,
    )
    __save_res_to_image(
        np.abs(data_after_stripe_based_sorting.get() - data_normalized_np_artefacts),
        output_folder,
        methods_name=methods_name + "_res",
        slice_numb=slice_numb,
        max_scale=0.1,
        min_scale=0,
    )
    del data_after_stripe_based_sorting

    methods_name = "remove_stripe_ti"
    print("___{}___".format(methods_name))
    from httomolibgpu.prep.stripe import (
        remove_stripe_ti,
    )

    data_after_remove_stripe_ti = remove_stripe_ti(
        cp.asarray(data_normalized_np_artefacts, dtype=cp.float32), beta=0.03
    )

    __save_res_to_image(
        data_after_remove_stripe_ti.get(),
        output_folder,
        methods_name,
        slice_numb,
        max_scale=max_scale_data_normalized_s,
        min_scale=min_scale_data_normalized_s,
    )
    __save_res_to_image(
        np.abs(data_after_remove_stripe_ti.get() - data_normalized_np_artefacts),
        output_folder,
        methods_name=methods_name + "_res",
        slice_numb=slice_numb,
        max_scale=0.1,
        min_scale=0,
    )
    del data_after_remove_stripe_ti

    methods_name = "remove_all_stripe"
    print("___{}___".format(methods_name))
    from httomolibgpu.prep.stripe import (
        remove_all_stripe,
    )

    data_after_remove_all_stripe = remove_all_stripe(
        cp.asarray(data_normalized_np_artefacts, dtype=cp.float32),
        snr=3.0,
        la_size=61,
        sm_size=21,
        dim=1,
    )

    __save_res_to_image(
        data_after_remove_all_stripe.get(),
        output_folder,
        methods_name,
        slice_numb,
        max_scale=max_scale_data_normalized_s,
        min_scale=min_scale_data_normalized_s,
    )
    __save_res_to_image(
        np.abs(data_after_remove_all_stripe.get() - data_normalized_np_artefacts),
        output_folder,
        methods_name=methods_name + "_res",
        slice_numb=slice_numb,
        max_scale=0.1,
        min_scale=0,
    )
    del data_after_remove_all_stripe

    del data_normalized_np_artefacts
    return 0


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that executes methods and "
        "generates images to be added to documentation."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="A path to the test data.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="Directory to save the images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    current_dir = os.path.basename(os.path.abspath(os.curdir))
    args = get_args()
    path_to_data = args.input
    output_folder = args.output
    return_val = run_methods(path_to_data, output_folder)
    if return_val == 0:
        print("The images have been successfully generated!")
