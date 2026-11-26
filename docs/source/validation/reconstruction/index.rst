Reconstruction
==============

Choose a method category, then a notebook for the dataset of interest. Each
notebook contains preprocessing, reconstruction settings, numerical checks and
plots. Parameters and reference values are specific to each case.

.. list-table:: Method overview
    :header-rows: 1
    :widths: 25 25 50

    * - Category
      - Method
      - Dataset coverage
    * - Direct
      - Filtered backprojection (FBP2D / FBP3D)
      - I12 datasets 1 and 5; I13 dataset 3
    * - Direct
      - Fourier inversion (LPRec3D)
      - I12 dataset 1; I13 datasets 1 and 2; K11 dataset 2
    * - Classical iterative
      - CGLS / SIRT
      - K11 dataset 2
    * - Regularised iterative
      - FISTA
      - K11 dataset 2; simulated GEANT4 dataset 1
    * - Regularised iterative
      - ADMM
      - K11 dataset 2

.. toctree::
    :caption: Direct reconstruction
    :maxdepth: 1

    FBP2D · I12 dataset 1 <direct/fbp2d_i12_dataset1>
    FBP3D · I12 dataset 1 <direct/fbp3d_i12_dataset1>
    FBP3D · I12 dataset 5 <direct/fbp3d_i12_dataset5>
    FBP3D · I13 dataset 3 <direct/fbp3d_i13_dataset3>
    LPRec3D · I12 dataset 1 <direct/lprec3d_i12_dataset1>
    LPRec3D · I13 dataset 1 <direct/lprec3d_i13_dataset1>
    LPRec3D · I13 dataset 2 <direct/lprec3d_i13_dataset2>
    LPRec3D · K11 dataset 2 <direct/lprec3d_k11_dataset2>

.. toctree::
    :caption: Classical iterative reconstruction
    :maxdepth: 1

    CGLS · K11 dataset 2 <iterative/cgls3d_k11_dataset2>
    SIRT · K11 dataset 2 <iterative/sirt3d_k11_dataset2>

.. toctree::
    :caption: Regularised iterative reconstruction
    :maxdepth: 1

    FISTA · K11 dataset 2 <regularised/fista3d_k11_dataset2>
    FISTA · GEANT4 dataset 1 <regularised/fista3d_geant4_dataset1>
    ADMM · K11 dataset 2 <regularised/admm3d_k11_dataset2>

Additional automated coverage
-----------------------------

:doc:`OSEM <../../reference/methods_list/reconstruction/OSEM3d_tomobar>` and
:doc:`iterative projection alignment <../../reference/methods_list/reconstruction/iterative_alignment>`
currently link to automated test coverage, including GPU execution checks.
They do not yet have saved reconstruction validation notebooks.
