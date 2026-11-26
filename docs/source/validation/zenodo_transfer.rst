Zenodo test transfer
====================

All 37 original test functions (50 cases after expanding parameters) now map to
validation or benchmark notebooks. The :download:`transfer manifest
<zenodo_transfer_manifest.json>` records original source checksums, names,
parameter counts, assertions and destinations. Existing worked examples with
different settings are supplemented by notebooks preserving the original cases.

.. list-table:: Coverage
   :header-rows: 1
   :widths: 25 15 60

   * - Original module
     - Cases
     - Destination
   * - Rotation
     - 11
     - :doc:`Rotation notebooks <rotation/index>`; separate Vo benchmark
   * - Reconstruction
     - 17
     - :doc:`Reconstruction notebooks <reconstruction/index>`; separate FBP and LPRec benchmarks
   * - Stitching
     - 3
     - :doc:`Overlap and full-stack stitching <rotation/center_360>`
   * - Stripe removal
     - 16
     - :doc:`Five-method parameter comparisons <preprocessing/stripe_removal>`
   * - Phase retrieval
     - 3
     - :doc:`Paganin padding choices <preprocessing/paganin_padding>`

The original pytest files and fixtures have been retired. Notebook loading
preserves the original archive arrays and preprocessing; no notebook imports
``zenodo-tests``. GPU workflows collect the notebooks directly and upload their
executed outputs. Benchmarks run separately through :doc:`../benchmarks/index`.
The three timing placeholders now record ten synchronised measurements and
check finite positive timings; protocol and scaling changes are documented in
each benchmark rather than treated as directly comparable old timings.

Recorded regression status
--------------------------

The new and updated notebooks were executed serially on the local RTX 4060
with 8 GB of memory. Phase retrieval, all five LPRec legacy cases, full-stack
stitching, the I13 full-turn reconstruction and the FBP benchmark completed
successfully. The existing FBP2D, CGLS and SIRT notebooks were also re-executed
successfully to verify their retained original cases.

Seven original cases fail their unchanged numerical references in this
environment: three Raven parameter cases, two FBP padding cases, FISTA and ADMM.
All seven failures were reproduced with the unchanged original pytest functions
before removing the duplicate suite. ADMM also returns shape ``(2560, 3, 2560)``
where its original assertion expects ``(4150, 3, 4150)``; the notebook records
both the sum and shape failures rather than stopping at the first assertion.

The unpadded 50-row I12 dataset 1 FBP case and full 15-row I12 dataset 5
full-turn FBP case exceed this GPU's memory. Their original full-data cases
remain in the notebooks, with allocation failures recorded as failed checks;
subsequent cases still execute. A GPU with sufficient free memory is needed
to finish those two computations. The smaller worked examples remain available.

These failures are visible in saved outputs and fail notebook execution in CI.
The transfer does not relax tolerances, replace numerical references, mark
known failures as expected, or reduce the original detector stacks.
The :download:`recorded failure details <zenodo_transfer_results.json>` include
observed values, expected values and memory-allocation messages. Consult each
notebook's provenance when comparing results between environments.
