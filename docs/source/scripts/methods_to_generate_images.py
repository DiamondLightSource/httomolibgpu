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

"""Generate realistic in-memory TomoPhantom method illustrations, without NeXus I/O.

Synthesis follows tomophantom/scripts/nxs_generator.py (Apache-2.0): analytic
row chunks, synth_flats, raw-space artefacts, a common signal gain, Poisson dark
current, Gaussian read noise, clipping and uint16 quantisation. TomoPhantom
uses (detector_y, angle, detector_x); processing uses (angle, detector_y, detector_x).
"""

from __future__ import annotations

import argparse
import inspect
import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "_static/auto_images_methods"
METHODS = (
    "normalisation",
    "remove_outlier",
    "median_filter",
    "rescale_to_int",
    "raven_filter",
    "remove_stripe_based_sorting",
    "remove_stripe_ti",
    "remove_all_stripe",
    "remove_stripe_fw",
    "reconstruction",
)
# Keep the documented raw-stage convention: synth_flats already adds Poisson noise.
ARTEFACT_KEYS = {
    "zingers_percentage",
    "zingers_modulus",
    "stripes_percentage",
    "stripes_maxthickness",
    "stripes_intensity",
    "stripes_type",
    "stripes_variability",
    "verbose",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model-number", type=int, default=7)
    p.add_argument(
        "--sinogram-shape",
        type=int,
        nargs=3,
        default=(256, 512, 740),
        metavar=("HEIGHT", "ANGLES", "WIDTH"),
    )
    p.add_argument("--flats", type=int, default=20)
    p.add_argument("--darks", type=int, default=10)
    p.add_argument("--source-intensity", type=int, default=20000)
    p.add_argument("--artefacts", type=Path, default=HERE / "artefacts.json")
    p.add_argument("--flat-settings", type=Path, default=HERE / "flat_settings.json")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dark-mean", type=float, default=100.0)
    p.add_argument("--read-noise", type=float, default=2.0)
    p.add_argument("--signal-scale", type=float, default=0.9)
    p.add_argument("--chunk-rows", type=int, default=32)
    p.add_argument(
        "--gpu-rows",
        type=int,
        default=16,
        help="Detector rows processed together; median filters add halos.",
    )
    p.add_argument(
        "--crop",
        type=int,
        default=30,
        help="Pixels trimmed from each image edge; 0 disables zoom. "
        "Axes too short for this crop are left intact.",
    )
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument(
        "--row", type=int, help="Detector row for the sinogram (default: centre)."
    )
    p.add_argument(
        "--projection", type=int, help="Projection index (default: one quarter)."
    )
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument(
        "--synthesis-only",
        action="store_true",
        help="Save raw, calibration and ideal images without requiring CUDA.",
    )
    return p


def read_settings(args):
    from tomophantom.flatsgen import synth_flats

    flat = json.loads(args.flat_settings.read_text())
    art = json.loads(args.artefacts.read_text())
    if not isinstance(flat, dict) or not isinstance(art, dict):
        raise ValueError("Settings files must contain JSON objects.")
    allowed = set(inspect.signature(synth_flats).parameters) - {
        "projData3D_clean",
        "source_intensity",
        "flatsnum",
    }
    if set(flat) - allowed or set(art) - ARTEFACT_KEYS:
        raise ValueError(
            "Unsupported flat/artefact settings; use the bundled JSON examples."
        )
    if (
        min(
            *args.sinogram_shape,
            args.flats,
            args.darks,
            args.source_intensity,
            args.chunk_rows,
            args.gpu_rows,
        )
        < 1
        or min(args.sinogram_shape) < 2
    ):
        raise ValueError(
            "Shape dimensions must be >= 2; counts and block sizes must be positive."
        )
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32).")
    if not np.isfinite([args.dark_mean, args.read_noise, args.signal_scale]).all():
        raise ValueError("Detector parameters must be finite.")
    if args.dark_mean < 0 or args.read_noise < 0 or not 0 < args.signal_scale <= 1:
        raise ValueError("Invalid detector parameters.")
    if not 1 <= flat.get("variations_number", 3) <= 3:
        raise ValueError("synth_flats supports one to three response variations.")
    if args.crop < 0:
        raise ValueError("Crop must be non-negative.")
    height, count, _ = args.sinogram_shape
    if args.row is None:
        args.row = height // 2
    if args.projection is None:
        args.projection = count // 4
    if not 0 <= args.row < height or not 0 <= args.projection < count:
        raise ValueError("Selected row/projection is outside the generated data.")
    return flat, art


def checked(data, shape, label):
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"{label}: expected finite data with shape {shape}.")
    return data


def apply_artefacts(data, settings, seed):
    """Corrupt normalised detector counts reproducibly, not line integrals."""
    from tomophantom.artefacts import artefacts_mix

    random.seed(seed)
    np.random.seed(seed)
    options = dict(settings, noise_seed=False, noise_prelog=False)
    options.setdefault("verbose", False)
    result = artefacts_mix(np.array(data, dtype=np.float32, copy=True), **options)
    return checked(result, data.shape, "artefacts_mix")


def quantise(signal, rng, args):
    """Match the generator's detector gain, dark distribution and read noise."""
    values = np.array(signal, dtype=np.float32, copy=True)
    values *= args.signal_scale
    values += rng.poisson(args.dark_mean, size=values.shape).astype(np.float32)
    if args.read_noise:
        values += rng.normal(0, args.read_noise, size=values.shape).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Detector synthesis produced non-finite values.")
    clipped = int(np.count_nonzero((values < 0) | (values > 65535)))
    return np.rint(np.clip(values, 0, 65535)).astype(np.uint16), clipped


def generate_data(args, flat_settings, artefacts):
    """Return realistic counts, ideal line integrals and calibration frames.

    Ideal projections are scaled by their ONE global maximum, as synth_flats
    scales its input in place. They are ideal line integrals, not transmission.
    No h5py, NeXus file, temporary volume file or saved array is used.
    """
    import tomophantom
    from tomophantom import TomoP3D
    from tomophantom.flatsgen import synth_flats
    import psutil

    height, count, width = args.sinogram_shape
    estimate = height * count * width * 64 + height * width * (8 * args.flats + 256)
    if estimate > psutil.virtual_memory().available * 0.8:
        raise MemoryError(
            f"Estimated synthesis workspace {estimate/2**30:.2f} GiB exceeds available memory budget."
        )
    print(
        f"Generating model {args.model_number}, shape {tuple(args.sinogram_shape)}; "
        f"estimated synthesis workspace {estimate/2**30:.2f} GiB",
        flush=True,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    angles = np.linspace(0, 179.9, count, dtype=np.float32)
    library = str(Path(tomophantom.__file__).parent / "phantomlib/Phantom3DLibrary.dat")
    clean = np.empty((height, count, width), dtype=np.float32)
    for start in range(0, height, args.chunk_rows):
        stop = min(start + args.chunk_rows, height)
        block = TomoP3D.ModelSinoSub(
            args.model_number,
            int(width / np.sqrt(2)),
            width,
            height,
            (start, stop),
            angles,
            library,
        )
        checked(block, (stop - start, count, width), "ModelSinoSub")
        if block.min() < 0:
            raise ValueError("The selected model has negative attenuation.")
        clean[start:stop] = block
    peak = float(clean.max())
    if peak <= 0:
        raise ValueError("The phantom has no positive projection signal.")
    signal, flats, response = synth_flats(
        clean,
        source_intensity=args.source_intensity,
        flatsnum=args.flats,
        **flat_settings,
    )
    checked(signal, clean.shape, "synth_flats projections")
    checked(flats, (height, args.flats, width), "synth_flats flats")
    if not np.isfinite(response).all():
        raise ValueError("Non-finite detector response map.")
    del response
    # synth_flats normalises clean in-place. Make that contract explicit.
    # Do not rescale corrupted raw counts by their new maximum.
    raw = apply_artefacts(signal.astype(np.float32) / 65535, artefacts, args.seed)
    raw *= 65535
    # Match nxs_generator's RNG consumption: darks, flats, row-chunked projections.
    dark_stack = np.empty((args.darks, height, width), dtype=np.uint16)
    flat_stack = np.empty((args.flats, height, width), dtype=np.uint16)
    clipped = 0
    for i in range(args.darks):
        dark_stack[i], n = quantise(np.zeros((height, width), np.float32), rng, args)
        clipped += n
    for i in range(args.flats):
        flat_stack[i], n = quantise(flats[:, i, :], rng, args)
        clipped += n
    del flats, signal
    counts = np.empty((count, height, width), dtype=np.uint16)
    for start in range(0, height, args.chunk_rows):
        stop = min(start + args.chunk_rows, height)
        block, n = quantise(raw[start:stop], rng, args)
        counts[:, start:stop] = block.swapaxes(0, 1)
        clipped += n
    return (
        counts,
        np.ascontiguousarray(clean.swapaxes(0, 1)),
        flat_stack,
        dark_stack,
        angles,
        {
            "clean_projection_max": peak,
            "clipped_voxels": clipped,
            "axis_order": "angle,detector_y,detector_x",
            "ideal_domain": "line integrals divided by global clean maximum",
        },
    )


def display_limits(data):
    """Robust shared bounds; outliers must not wash out the whole comparison."""
    lo, hi = np.percentile(data, [0.5, 99.5])
    return float(lo), float(hi) if hi > lo else float(lo + 1)


def crop_view(data, margin):
    """Crop display views only; preserve short calibration/smoke-test axes."""
    if margin < 0:
        raise ValueError("Crop must be non-negative.")
    if np.ndim(data) != 2:
        raise ValueError("Image cropping requires a 2D array.")
    return data[
        tuple(
            slice(margin, -margin) if margin and n > 2 * margin else slice(None)
            for n in data.shape
        )
    ]


def save_image(data, path, limits, crop=0):
    lo, hi = limits
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Image limits must be finite and increasing.")
    values = crop_view(np.asarray(data), crop)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Images must be finite 2D arrays.")
    # Clip before casting: avoid wraparound and the old uint8 * 255 overflow.
    pixels = np.rint(
        255 * np.clip((values.astype(np.float32) - lo) / (hi - lo), 0, 1)
    ).astype(np.uint8)
    Image.fromarray(pixels).save(path)


class Images:
    def __init__(self, args):
        self.args = args
        self.records = {}

    def pair(self, data, name, limits=None):
        views = {
            "sino": data[:, self.args.row, :],
            "proj": data[self.args.projection, :, :],
        }
        return self.views(views, name, limits)

    def views(self, views, name, limits=None):
        if limits is None:
            limits = display_limits(np.concatenate([v.ravel() for v in views.values()]))
        for view, pixels in views.items():
            filename = f"{name}_{view}.png"
            save_image(pixels, self.args.output / filename, limits, self.args.crop)
            self.records[filename] = {"display_limits": list(limits), "view": view}
        return limits

    def calibration(self, data, name):
        # Calibration frame indices are independent of projection/row indices.
        limits = display_limits(data)
        for view, pixels in {
            "proj": data[0],
            "sino": data[:, self.args.row, :],
        }.items():
            filename = f"{name}_{view}.png"
            save_image(pixels, self.args.output / filename, limits, self.args.crop)
            self.records[filename] = {"display_limits": list(limits), "view": view}


def gpu_apply(data, method, args, halo=0, **kwargs):
    """Bound GPU memory by detector-row blocks; preserve 3D filter neighbours."""
    import cupy as cp

    output = None
    for start in range(0, data.shape[1], args.gpu_rows):
        stop = min(start + args.gpu_rows, data.shape[1])
        left, right = max(0, start - halo), min(data.shape[1], stop + halo)
        block = cp.asarray(np.ascontiguousarray(data[:, left:right]), dtype=cp.float32)
        result = method(block, **kwargs)
        if output is None:
            output = np.empty(data.shape, dtype=result.dtype)
        output[:, start:stop] = cp.asnumpy(result[:, start - left : stop - left])
        del block, result
    return output


def process_methods(raw, clean, flats, darks, angles, args, images):
    import cupy as cp
    from httomolibgpu.prep.normalize import dark_flat_field_correction, minus_log
    from httomolibgpu.misc.corr import remove_outlier, median_filter
    from httomolibgpu.misc.rescale import rescale_to_int
    from httomolibgpu.prep import stripe

    def normalise(data, illustrate=False):
        result = np.empty(data.shape, dtype=np.float32)
        if illustrate:
            transmission_projection = np.empty(data.shape[1:], dtype=np.float32)
            transmission_sinogram = np.empty(
                (data.shape[0], data.shape[2]), dtype=np.float32
            )
        for start in range(0, data.shape[1], args.gpu_rows):
            stop = min(start + args.gpu_rows, data.shape[1])
            corrected = dark_flat_field_correction(
                cp.asarray(data[:, start:stop]),
                cp.asarray(flats[:, start:stop]),
                cp.asarray(darks[:, start:stop]),
                lower_bound=1e-6,
            )
            if illustrate:
                # Capture transmission BEFORE minus_log, which may modify its input.
                transmission_projection[start:stop] = cp.asnumpy(
                    corrected[args.projection]
                )
                if start <= args.row < stop:
                    transmission_sinogram[:] = cp.asnumpy(
                        corrected[:, args.row - start, :]
                    )
            result[:, start:stop] = cp.asnumpy(minus_log(corrected))
        if illustrate:
            images.pair(data, "dark_flat_field_correction_input")
            images.views(
                {"proj": transmission_projection, "sino": transmission_sinogram},
                "dark_flat_field_correction",
                (0.0, 1.1),
            )
        return checked(result, data.shape, "normalisation")

    def reconstruction(data, name, limits=None):
        from httomolibgpu.recon.algorithm import FBP3d_tomobar

        left = max(0, args.row - 2)
        right = min(data.shape[1], args.row + 3)
        volume = FBP3d_tomobar(
            cp.asarray(np.ascontiguousarray(data[:, left:right])),
            np.deg2rad(angles).astype(np.float32),
            gpu_id=args.gpu_id,
        )
        image = cp.asnumpy(volume[:, args.row - left, :])
        if limits is None:
            limits = display_limits(image)
        filename = f"{name}_recon.png"
        save_image(image, args.output / filename, limits, args.crop)
        images.records[filename] = {
            "display_limits": list(limits),
            "view": "reconstruction",
        }
        del volume
        return limits

    data = normalise(raw, illustrate=True)
    limits = images.pair(data, "normalisation")
    reference_recon_limits = None
    if "reconstruction" in args.methods:
        reference_recon_limits = reconstruction(clean, "proj_ground_truth")
        reconstruction(data, "raw_data", reference_recon_limits)
    # Remove zingers before demonstrating stripe filters, rather than adding
    # arbitrary black rectangles at fixed indices as the original script did.
    threshold = 2000.0
    corrected_raw = None
    if any(
        m in args.methods
        for m in ("remove_outlier", "rescale_to_int", "reconstruction", *METHODS[4:9])
    ):
        print("Processing remove_outlier", flush=True)
        corrected_raw = gpu_apply(
            raw, remove_outlier, args, halo=2, kernel_size=5, dif=threshold
        )
        if "remove_outlier" in args.methods:
            raw_limits = images.pair(raw, "remove_outlier_input")
            images.pair(corrected_raw, "remove_outlier", raw_limits)
            images.pair(
                np.abs(raw.astype(np.float32) - corrected_raw),
                "remove_outlier_res",
                (0, threshold),
            )
    if "median_filter" in args.methods:
        result = gpu_apply(data, median_filter, args, halo=2, kernel_size=5)
        images.pair(result, "median_filter", limits)
        images.pair(np.abs(data - result), "median_filter_res", (0, 0.05))
        del result
    if corrected_raw is not None:
        del data
        data = normalise(corrected_raw)
        del corrected_raw
    rescale_limits = None
    if "rescale_to_int" in args.methods:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Exclude rare dead-detector / logarithm-floor spikes from the display
        # window, then use ONE global range for every processing block.
        rescale_limits = display_limits(data)
        rescale_input = np.clip(data, *rescale_limits)
        stats = (
            float(rescale_input.min()),
            float(rescale_input.max()),
            float(rescale_input.sum(dtype=np.float64)),
            rescale_input.size,
        )
        for lo, hi in [(0, 100), (10, 90), (30, 70)]:
            result = gpu_apply(
                rescale_input,
                rescale_to_int,
                args,
                perc_range_min=lo,
                perc_range_max=hi,
                bits=8,
                glob_stats=stats,
            )
            filename = f"rescale_to_int_proj_{lo}_to_{hi}.png"
            save_image(
                result[args.projection], args.output / filename, (0, 255), args.crop
            )
            images.records[filename] = {
                "display_limits": [0, 255],
                "view": "projection",
            }
            fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
            ax.hist(
                crop_view(result[args.projection], args.crop).ravel(),
                bins=np.arange(257),
                color="#7030a0",
            )
            ax.set(xlabel="8-bit intensity", ylabel="Pixel count", xlim=(0, 255))
            fig.savefig(args.output / f"rescale_to_int_histo_{lo}_to_{hi}.png", dpi=150)
            plt.close(fig)
            images.records[f"rescale_to_int_histo_{lo}_to_{hi}.png"] = {
                "intensity_range": [0, 255],
                "view": "histogram",
            }
            del result
    if rescale_limits is not None:
        del rescale_input
    stripe_limits = images.pair(data, "data_stripes_added")
    parameters = {
        "raven_filter": dict(uvalue=20, nvalue=2, vvalue=2),
        "remove_stripe_based_sorting": dict(size=21, dim=1),
        "remove_stripe_ti": dict(beta=0.03),
        "remove_all_stripe": dict(snr=3.0, la_size=61, sm_size=21, dim=1),
        "remove_stripe_fw": dict(sigma=1.0, wname="db5"),
    }
    if reference_recon_limits is not None:
        reconstruction(data, "data_stripes_added", reference_recon_limits)
    for name, options in parameters.items():
        if name not in args.methods:
            continue
        print(f"Processing {name}", flush=True)
        result = gpu_apply(data, getattr(stripe, name), args, **options)
        checked(result, data.shape, name)
        images.pair(result, name, stripe_limits)
        images.pair(np.abs(data - result), name + "_res", (0, 0.1))
        if reference_recon_limits is not None:
            reconstruction(result, name, reference_recon_limits)
        del result
    return {
        "remove_outlier": {"kernel_size": 5, "dif": threshold},
        "median_filter": {"kernel_size": 5},
        "stripe_filters": parameters,
        "rescale_input_limits": rescale_limits,
        "reconstruction": "FBP3d_tomobar, central detector-row slab",
    }


def main(argv=None):
    args = parser().parse_args(argv)
    flat, art = read_settings(args)
    if not args.synthesis_only:
        import cupy as cp
        from httomolibgpu.cupywrapper import require_gpu

        require_gpu("Documentation image generation", args.gpu_id)
        cp.cuda.Device(args.gpu_id).use()
        if "rescale_to_int" in args.methods:
            __import__("matplotlib")  # Fail before synthesising a full dataset.
        if "reconstruction" in args.methods:
            import astra

            if not hasattr(astra.projector3d, "direct_BP"):
                raise RuntimeError(
                    "Reconstruction requires ASTRA Toolbox >= 2.5 for the current "
                    "ToMoBAR API. Upgrade ASTRA or omit reconstruction from --methods."
                )
    args.output.mkdir(parents=True, exist_ok=True)
    raw, clean, flats, darks, angles, metadata = generate_data(args, flat, art)
    images = Images(args)
    images.pair(raw, "raw_data")
    images.pair(clean, "proj_ground_truth")
    images.calibration(flats, "flats")
    images.calibration(darks, "darks")
    method_settings = {}
    if not args.synthesis_only:
        method_settings = process_methods(
            raw, clean, flats, darks, angles, args, images
        )
    versions = {}
    for package in (
        "httomolibgpu",
        "tomophantom",
        "tomobar",
        "astra-toolbox",
        "cupy",
        "numpy",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source checkout / unknown"
    metadata["versions"] = versions
    metadata.update(
        arguments=vars(args),
        flat_settings=flat,
        artefacts=art,
        methods=method_settings,
        images=images.records,
        note="Synthetic TomoPhantom data; residuals are absolute input/output differences, not reconstruction error.",
    )
    (args.output / "generation_manifest.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(
        f"Saved method illustrations to {args.output}; no NeXus file was created.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
