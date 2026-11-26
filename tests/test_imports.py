"""Standalone utilities must not eagerly import reconstruction backends."""

import subprocess
import sys
import textwrap
from pathlib import Path


def run_import_check(code):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_phase_correlation_import_does_not_load_tomobar_wrappers():
    run_import_check(
        """
        import importlib.abc
        import sys
        class BlockReconstructionImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in ('httomolibgpu.recon.algorithm', 'httomolibgpu.recon.rotation'):
                    raise AssertionError('Unexpected reconstruction import: ' + fullname)
        sys.meta_path.insert(0, BlockReconstructionImports())
        import httomolibgpu
        from httomolibgpu.recon._phase_cross_correlation import phase_cross_correlation
        assert callable(phase_cross_correlation)
        for name in ('FBP3d_tomobar', 'find_center_vo'):
            assert name in dir(httomolibgpu)
            assert name in httomolibgpu.__all__
            assert name not in httomolibgpu.__dict__
    """
    )


def test_top_level_reconstruction_exports_are_resolved_and_cached():
    run_import_check(
        """
        import sys
        from types import ModuleType
        import httomolibgpu
        algorithm = ModuleType('httomolibgpu.recon.algorithm')
        rotation = ModuleType('httomolibgpu.recon.rotation')
        def reconstruction():
            pass
        def find_center():
            pass
        algorithm.FBP3d_tomobar = reconstruction
        rotation.find_center_vo = find_center
        sys.modules[algorithm.__name__] = algorithm
        sys.modules[rotation.__name__] = rotation
        from httomolibgpu import FBP3d_tomobar, find_center_vo
        assert FBP3d_tomobar is reconstruction
        assert find_center_vo is find_center
        assert httomolibgpu.__dict__['FBP3d_tomobar'] is reconstruction
        try:
            httomolibgpu.unknown_function
        except AttributeError:
            pass
        else:
            raise AssertionError('Missing export did not raise AttributeError')
    """
    )


def test_all_modules_import_without_cuda_or_tomobar():
    run_import_check(
        """
        import importlib
        import importlib.abc
        import pkgutil
        import sys
        import numpy as np

        class BlockGPUImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('cupy', 'cupyx', 'tomobar', 'nvtx'):
                    raise ModuleNotFoundError('Blocked optional dependency: ' + fullname)

        sys.meta_path.insert(0, BlockGPUImports())
        import httomolibgpu
        from httomolibgpu import cupywrapper
        assert not cupywrapper.cupy_run
        assert cupywrapper.cp is np
        for module in pkgutil.walk_packages(httomolibgpu.__path__, 'httomolibgpu.'):
            importlib.import_module(module.name)
        assert not any(name.startswith(('cupyx', 'tomobar')) for name in sys.modules)

        # Real NumPy-based utilities retain their behaviour.
        from httomolibgpu.misc.morph import average_projection_frames, sino_360_to_180
        from httomolibgpu.misc.sorting import argsort_with_reverse
        from httomolibgpu.prep.normalize import minus_log
        values = np.arange(8 * 3 * 12, dtype=np.float32).reshape(8, 3, 12)
        np.testing.assert_array_equal(average_projection_frames(values, 2),
                                     values.reshape(4, 2, 3, 12).mean(axis=1))
        assert sino_360_to_180(values, overlap=0).shape == (4, 3, 24)
        indices, reverse = argsort_with_reverse(np.array([3., 1., 2.]))
        np.testing.assert_array_equal(indices[reverse], np.arange(3))
        np.testing.assert_allclose(minus_log(np.array([1., np.exp(-1)])), [0., 1.])

        # GPU algorithms must not silently return an unprocessed/fake result.
        operations = [
            ('httomolibgpu.cuda_kernels', 'load_cuda_module', ('does-not-exist',)),
            ('httomolibgpu.misc.corr', 'median_filter', (values,)),
            ('httomolibgpu.misc.rescale', 'rescale_to_int', (values,)),
            ('httomolibgpu.prep.normalize', 'dark_flat_field_correction', (values, values, values)),
            ('httomolibgpu.prep.alignment', 'distortion_correction_proj_discorpy', (values,)),
            ('httomolibgpu.prep.phase', 'paganin_filter', (values,)),
            ('httomolibgpu.misc.morph', 'data_resampler', (values, [3, 12])),
            ('httomolibgpu.recon._phase_cross_correlation', 'phase_cross_correlation', (values[0], values[0])),
        ]
        operations.append(('httomolibgpu.misc.utils', 'data_checker', (values,)))
        for name in ('non_local_means', 'total_variation_ROF', 'total_variation_PD', 'total_variation_SB', 'total_variation_FGP', 'total_generalised_variation', 'nonlinear_diffusion'):
            operations.append(('httomolibgpu.misc.denoise', name, (values,)))
        for name in ('remove_stripe_based_sorting', 'remove_stripe_fw', 'remove_stripe_ti', 'remove_all_stripe', 'raven_filter'):
            operations.append(('httomolibgpu.prep.stripe', name, (values,)))
        for name in ('find_center_vo', 'find_center_360', 'find_center_pc', 'find_center_metric_recon'):
            operations.append(('httomolibgpu.recon.rotation', name, (values[0], values[0]) if name == 'find_center_pc' else ((values, np.arange(8)) if name == 'find_center_metric_recon' else (values,))))
        for name in ('FBP2d_astra', 'FBP3d_tomobar', 'LPRec3d_tomobar', 'SIRT3d_tomobar', 'CGLS3d_tomobar', 'FISTA3d_tomobar', 'ADMM3d_tomobar', 'OSEM3d_tomobar', 'iterative_alignment'):
            operations.append(('httomolibgpu.recon.algorithm', name, (values, np.arange(8))))
        for module, name, args in operations:
            try:
                getattr(importlib.import_module(module), name)(*args)
            except RuntimeError as exc:
                assert 'CUDA GPU' in str(exc), (module, name, str(exc))
            else:
                raise AssertionError('GPU-only call unexpectedly succeeded: ' + name)
    """
    )


def test_installed_cupy_with_unavailable_device_falls_back_consistently():
    run_import_check(
        """
        import importlib
        import pkgutil
        import sys
        from types import ModuleType, SimpleNamespace
        import numpy as np

        class UnavailableDevice(RuntimeError):
            pass
        cupy = ModuleType('cupy')
        cupy.uint8 = np.uint8
        cupy.cuda = SimpleNamespace(runtime=SimpleNamespace(CUDARuntimeError=UnavailableDevice))
        def no_device(*args, **kwargs):
            raise UnavailableDevice('test: device is not accessible')
        cupy.empty = no_device
        sys.modules['cupy'] = cupy
        import httomolibgpu
        from httomolibgpu import cupywrapper
        assert not cupywrapper.cupy_run
        assert cupywrapper.cp is np
        for module in pkgutil.walk_packages(httomolibgpu.__path__, 'httomolibgpu.'):
            importlib.import_module(module.name)
        try:
            cupywrapper._UnavailableGPU('FFT plan').work_area()
        except RuntimeError as exc:
            assert 'test: device is not accessible' in str(exc)
        else:
            raise AssertionError('Unavailable dependency returned a fake result')
    """
    )
