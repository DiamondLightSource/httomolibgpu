Centre of rotation and 360-degree scans
=======================================

Estimate rotation centres and overlap using experimental and simulated datasets.
Each notebook describes preprocessing, method settings and numerical references,
and combines assertions with sinogram and registration plots.

.. list-table:: Validation coverage
   :header-rows: 1
   :widths: 30 45 25

   * - Method
     - Datasets and cases
     - Numerical checks
   * - Vo centre finding
     - I12 datasets 1 and 2 (including row averaging), I13 dataset 2,
       GEANT4 dataset 1 and K11 dataset 1
     - Six exact centres and float32 dtype checks
   * - Phase correlation
     - Opposing I12 dataset 3 projections, with detector rows 200:1750
     - Exact centre and float32 dtype
   * - 360-degree overlap
     - I12 dataset 5 and I13 datasets 1 and 3
     - Three integer centre, integer overlap and overlap-side checks

.. toctree::
   :maxdepth: 1

   Vo centre finding across datasets <center_vo>
   Phase correlation on opposing projections <center_pc>
   360-degree overlap and stitching <center_360>

The :doc:`execution instructions <../index>` cover dependencies, archive
location and updating saved outputs. The notebooks check numerical references,
dtype and stitched shape, with plots for visual review.

For execution times, see the separate :doc:`Vo centre-search benchmark
<../../benchmarks/center_vo_i12_dataset1>`.

For method descriptions, see :doc:`Vo centre finding
<../../reference/methods_list/cor_finder/find_center_vo>`,
:doc:`phase correlation <../../reference/methods_list/cor_finder/find_center_pc>`
and :doc:`360-degree overlap <../../reference/methods_list/cor_finder/find_center_360>`.
