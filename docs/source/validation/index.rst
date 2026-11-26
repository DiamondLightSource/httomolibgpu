.. _real_data_validation:

Validation
==========

These executable notebooks cover reconstruction of experimental and simulated
projections, phase retrieval, stripe removal, centre-of-rotation and overlap estimation, and denoising of reference images. Numerical checks accompany
plots for visual review.
Each notebook describes its data, processing parameters, acceptance checks and
limitations, and records the software environment used for execution.

.. toctree::
    :maxdepth: 1

    reconstruction/index
    rotation/index
    preprocessing/index
    denoising/index
    zenodo_transfer

Running the validation
----------------------

From the repository root, install the notebook dependencies. For reconstruction
cases, also download the Zenodo datasets into the local ``Zenodo`` directory::

    python -m pip install -e '.[validation]'
    python .scripts/download_zenodo.py Zenodo

The 3D denoising notebook additionally requires TomoPhantom, including its
compiled phantom generator (``conda install httomo::tomophantom``).

An existing archive can be used by setting ``HTTOMOLIBGPU_DATA_DIR`` to its
directory. A compatible NVIDIA GPU and reconstruction dependencies are required
for execution. Run the notebook through pytest and save its outputs::

    python -m pytest --nbmake --nbmake-timeout=1800 --overwrite docs/source/validation

The documentation build renders the saved notebook outputs without executing
GPU computations. Re-execute the notebook before building to update its figures
and metrics. GPU test workflows also save the executed notebook as an artifact.

Interpreting the results
------------------------

Assertions check the stated numerical and structural expectations. Plots support
human review of image quality; successful execution alone does not establish
scientific accuracy. A failed regression remains a failure requiring investigation,
even when reconstruction images are available. Consult the notebook's recorded
validation result and provenance when interpreting its outputs.
