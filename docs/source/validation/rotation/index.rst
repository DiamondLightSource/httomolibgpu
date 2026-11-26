Centre of rotation and 360-degree scans
=======================================

These notebooks transfer all ten numerical cases from the former Zenodo
``test_rotation.py`` suite. Cases are grouped by the public method, with shared
preprocessing for the central-row and averaging cases on I12 dataset 2.
Each notebook records archive checksums, software/GPU provenance, parameters,
observed values and saved figures, then enforces the original regression checks.

.. list-table:: Validation coverage
   :header-rows: 1
   :widths: 30 45 25

   * - Method
     - Datasets and cases
     - Original checks
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
location and updating saved outputs. Exact reference values and integer
conversions are retained from the tests; figures do not relax those checks.
The :download:`transfer manifest <transfer_manifest.json>` maps all eleven
original test names, including the performance case, to their destinations.

The timing case is a separate :doc:`Vo centre-search benchmark
<../../benchmarks/center_vo_i12_dataset1>`, excluded from routine validation.
It records synchronised timing samples rather than comparing milliseconds
with the original placeholder string assertion.

For method descriptions, see :doc:`Vo centre finding
<../../reference/methods_list/cor_finder/find_center_vo>`,
:doc:`phase correlation <../../reference/methods_list/cor_finder/find_center_pc>`
and :doc:`360-degree overlap <../../reference/methods_list/cor_finder/find_center_360>`.

The overlap notebook also transfers all three full-stack stitching cases from
``test_morph.py``, retaining their original fixed fractional overlaps, detector
rows and output shape/dtype/contiguity assertions. See :doc:`../zenodo_transfer`.
