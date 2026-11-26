HTTomolibGPU
============

.. about-start

**HTTomolibGPU** provides GPU-accelerated preprocessing, reconstruction and
image denoising for computed tomography. It uses `CuPy <https://cupy.dev/>`_
for GPU arrays and `ToMoBAR <https://dkazanc.github.io/ToMoBAR/>`_ for
reconstruction and variational denoising.

Use it as a standalone Python library or as a processing backend for
`HTTomo <https://diamondlightsource.github.io/httomo/>`_. Methods include
normalisation, phase retrieval, stripe removal, tomographic reconstruction,
projection alignment and denoising. Several methods originate from
`TomoPy <https://tomopy.readthedocs.io/en/stable/>`_ and
`Savu <https://savu.readthedocs.io/en/latest/>`_.

See the `documentation <https://diamondlightsource.github.io/httomolibgpu/>`_
for the method guides, API reference and validation notebooks with saved results.

.. about-end

Run with HTTomo
---------------

All HTTomolibGPU processing methods can be run within
`HTTomo <https://diamondlightsource.github.io/httomo/>`_, the framework for
high-throughput tomography processing. Define a pipeline and let HTTomo manage
data loading, memory-aware blocks, data movement, parallel execution and saving
results. See `Run with HTTomo <https://diamondlightsource.github.io/httomolibgpu/introduction/httomo.html>`_
for an overview, or visit the `HTTomo main page <https://diamondlightsource.github.io/httomo/>`_
to get started.

Installation
------------

.. installation-start

Requirements
~~~~~~~~~~~~

The current source requires Python **3.12 or newer**, NumPy **2.4.x** and
CuPy **14.2.x**. GPU processing needs a CUDA-capable NVIDIA GPU and a compatible
NVIDIA driver. Check driver availability with ``nvidia-smi`` before installing.
ToMoBAR supplies the reconstruction backend, including ASTRA Toolbox.

The current reconstruction and denoising wrappers use ToMoBAR's
``geometry``, ``reconstruction`` and ``regularisers`` APIs. When using this
source checkout, install a compatible ToMoBAR checkout following its
`source installation guide <https://dkazanc.github.io/ToMoBAR/getting_started/alternative_installation.html>`_;
a published ToMoBAR release may expose an older API.

Conda environment
~~~~~~~~~~~~~~~~~

Create an environment with binary GPU dependencies, then install HTTomolibGPU::

    conda create --name httomolibgpu -c conda-forge \
        python=3.12 'numpy=2.4.*' 'cupy=14.2.*' 'astra-toolbox>=2.5' pip
    conda activate httomolibgpu
    python -m pip install httomolibgpu

Conda supplies the CUDA runtime libraries; the NVIDIA driver must be installed
on the host. See the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_
for selecting a CUDA version or resolving driver/runtime mismatches.

Install from source
~~~~~~~~~~~~~~~~~~~

In the compatible GPU environment above, clone the repository and install it
in editable mode::

    git clone https://github.com/DiamondLightSource/httomolibgpu.git
    cd httomolibgpu
    python -m pip install -e .

For development, install TomoPhantom's compiled generator and the test tools::

    conda install -c httomo -c conda-forge tomophantom
    python -m pip install -e '.[dev,validation]'

The ``validation`` extra supplies notebook execution and plotting tools.
TomoPhantom is used by phantom-based tests and the 3D denoising notebook.
Install ``jupyterlab`` separately if you want to edit notebooks interactively.

Virtual environment
~~~~~~~~~~~~~~~~~~~

For a pip-only installation, prepare a CUDA toolkit and a compatible C++
compiler first. The package declares ``cupy`` as a dependency, which pip builds
from source. Follow the `CuPy source installation instructions <https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-source>`_
for the build requirements, then run::

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install httomolibgpu

CUDA-specific wheels such as ``cupy-cuda12x`` have a different distribution name
and do not satisfy the declared ``cupy`` dependency. Avoid installing them
alongside ``cupy``; use the Conda route above to obtain a prebuilt package.

Check the installation
~~~~~~~~~~~~~~~~~~~~~~

Run these commands in the environment that will process your data::

    python -m pip check
    python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); print(float(cp.arange(8, dtype=cp.float32).sum()))"
    python -c "import astra; print(astra.use_cuda())"
    python -c "from httomolibgpu import total_variation_PD, FBP3d_tomobar; print('HTTomolibGPU imports OK')"

Expect a positive GPU count, a sum of ``28.0``, ``True`` for ASTRA GPU support
and successful HTTomolibGPU imports. For end-to-end examples, follow the
`validation notebooks <https://diamondlightsource.github.io/httomolibgpu/validation/index.html>`_.

If CuPy reports missing CUDA headers, check the toolkit installation. On Linux
x86_64, if ``cuda_runtime.h`` exists under
``$CONDA_PREFIX/targets/x86_64-linux/include``, point CuPy at that toolkit and
repeat the checks::

    export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation can be built without a GPU and renders saved notebook outputs.
From a source checkout, follow ``docs/sphinx_build_setup.rst`` to create the
separate documentation environment and run ``bash docs/sphinx-build.sh``.
