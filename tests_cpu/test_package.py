"""Check the installed wheel without compiling CUDA kernels."""

from importlib.resources import files

import httomolibgpu


def test_all_cuda_kernel_sources_are_packaged():
    kernel_dir = files("httomolibgpu.cuda_kernels")
    expected = {
        "median_kernel.cu",
        "calc_metrics.cu",
        "remove_stripe_fw.cu",
        "center_360_shifts.cu",
        "remove_nan_inf.cu",
        "generate_mask.cu",
        "raven_filter.cu",
        "non_local_means.cu",
        "count_threshold.cu",
    }
    # Check the source contents too: an empty placeholder is not a usable kernel.
    for name in expected:
        source = kernel_dir.joinpath(name)
        assert source.is_file(), f"Missing packaged CUDA kernel: {name}"
        assert "__global__" in source.read_text(), name


def test_public_processing_exports_are_available():
    for name in ("median_filter", "paganin_filter", "FBP3d_tomobar", "find_center_vo"):
        assert name in httomolibgpu.__all__
        assert callable(getattr(httomolibgpu, name))
