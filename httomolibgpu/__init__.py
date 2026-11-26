from httomolibgpu.misc.utils import data_checker

from httomolibgpu.misc.blend import seam_blend_stitched_data
from httomolibgpu.misc.corr import median_filter, remove_outlier
from httomolibgpu.misc.denoise import (
    total_variation_ROF,
    total_variation_PD,
    total_variation_SB,
    total_variation_FGP,
    total_generalised_variation,
    nonlinear_diffusion,
    non_local_means,
)
from httomolibgpu.misc.morph import sino_360_to_180, data_resampler
from httomolibgpu.misc.rescale import rescale_to_int
from httomolibgpu.prep.alignment import distortion_correction_proj_discorpy
from httomolibgpu.prep.normalize import dark_flat_field_correction, minus_log
from httomolibgpu.prep.phase import paganin_filter
from httomolibgpu.prep.stripe import (
    remove_stripe_based_sorting,
    remove_stripe_fw,
    remove_stripe_ti,
    remove_all_stripe,
    raven_filter,
)

# Reconstruction and rotation helpers depend on ToMoBAR. Load those exports
# only when requested, so standalone utilities (including phase correlation)
# do not import reconstruction backends as a side effect.
_LAZY_EXPORTS = {
    "FBP2d_astra": "httomolibgpu.recon.algorithm",
    "FBP3d_tomobar": "httomolibgpu.recon.algorithm",
    "LPRec3d_tomobar": "httomolibgpu.recon.algorithm",
    "SIRT3d_tomobar": "httomolibgpu.recon.algorithm",
    "CGLS3d_tomobar": "httomolibgpu.recon.algorithm",
    "FISTA3d_tomobar": "httomolibgpu.recon.algorithm",
    "ADMM3d_tomobar": "httomolibgpu.recon.algorithm",
    "iterative_alignment": "httomolibgpu.recon.algorithm",
    "find_center_vo": "httomolibgpu.recon.rotation",
    "find_center_360": "httomolibgpu.recon.rotation",
    "find_center_pc": "httomolibgpu.recon.rotation",
    "find_center_metric_recon": "httomolibgpu.recon.rotation",
}

# Preserve star-import discovery as well as the usual explicit imports.
__all__ = [
    "data_checker",
    "seam_blend_stitched_data",
    "median_filter",
    "remove_outlier",
    "total_variation_ROF",
    "total_variation_PD",
    "total_variation_SB",
    "total_variation_FGP",
    "total_generalised_variation",
    "nonlinear_diffusion",
    "non_local_means",
    "sino_360_to_180",
    "data_resampler",
    "rescale_to_int",
    "distortion_correction_proj_discorpy",
    "dark_flat_field_correction",
    "minus_log",
    "paganin_filter",
    "remove_stripe_based_sorting",
    "remove_stripe_fw",
    "remove_stripe_ti",
    "remove_all_stripe",
    "raven_filter",
] + list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
