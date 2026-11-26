Performance benchmarks
======================

These notebooks measure execution time on real datasets. They record hardware,
software and timing protocols so results can be interpreted and compared.
They run separately from scientific validation, without universal speed thresholds.

.. toctree::
    :caption: Reconstruction
    :maxdepth: 1

    FBP3D TomoBAR · I13 dataset 2 <fbp3d_i13_dataset2>
    LPRec3D TomoBAR — I13 dataset 2 <lprec3d_i13_dataset2>

Running benchmarks
------------------

Prepare dependencies and data as described in :doc:`../validation/index`, then run
benchmarks explicitly and serially on an otherwise idle GPU::

    python -m pytest --nbmake --nbmake-timeout=1800 --overwrite docs/source/benchmarks/*.ipynb

The manual ``Reconstruction benchmarks`` GitHub Actions workflow runs these
notebooks on the GPU runner and uploads the executed reports. Routine validation
workflows run only notebooks in ``docs/source/validation``. Documentation builds
render saved benchmark outputs without executing the measurements.

.. toctree::
    :caption: Centre of rotation
    :maxdepth: 1

    Vo centre finding · I12 dataset 1 <center_vo_i12_dataset1>
