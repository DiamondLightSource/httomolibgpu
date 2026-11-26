Building the documentation
==========================

Create or update the documentation environment from the repository root::

    conda env create -f docs/source/doc-conda-requirements.yml
    conda activate httomolibgpu-docs

For an existing environment::

    conda env update -f docs/source/doc-conda-requirements.yml --prune

The environment contains the documentation tools, including Pandoc for notebook
conversion. Installing the library or its GPU runtime is not required to render
the documentation: optional scientific imports are mocked in ``conf.py``.
The dependency ranges target Sphinx 9, Sphinx Book Theme 1.4 and nbsphinx 0.9.8.

Generate the API pages and build HTML::

    bash docs/sphinx-build.sh

Open ``docs/build/index.html`` in a browser. An optional first argument selects
a different output directory. The script stops on errors and treats Sphinx
warnings as failures. It updates generated API pages without deleting the
source or build directories. CI uses the same script.

Notebook pages render saved outputs; they are never executed by Sphinx.
To refresh notebook results, use the validation environment and execution
instructions in the notebooks before rebuilding the documentation.

Runtime import conventions
--------------------------

Method modules group standard-library, third-party and local imports before
binding ``cp = cupywrapper.cp``. Use deferred annotations so importing a CUDA
kernel loader does not evaluate GPU-only types on a CPU-only machine.

Import CuPy itself through ``httomolibgpu.cupywrapper``. Keep CuPyX and ToMoBAR
imports inside an ``if cupy_run`` block. Its ``else`` branch defines every
conditional dependency using ``cupywrapper._UnavailableGPU``; calling one of
these placeholders raises a clear runtime error. New GPU-only entry points
should use ``cupywrapper.require_gpu`` before creating kernels or GPU buffers.
Ordinary NumPy-backed array utilities may continue using the shared fallback.

Local CUDA kernel loading is safe to import unconditionally: the loader checks
for CUDA when called. Keep reconstruction imports lazy where needed to avoid
loading those backends while importing independent processing utilities.
Do not catch a broken GPU dependency's import error when CUDA is available;
report the installation problem rather than substituting an executable mock.
The documentation's third-party mocks in ``conf.py`` remain separate from this
runtime policy.

``tests/test_imports.py`` exercises missing CuPy/ToMoBAR and inaccessible CUDA
devices in isolated subprocesses, as well as NumPy utilities and GPU-only errors.
Run it from the development environment with ``python -m pytest tests/test_imports.py``.
