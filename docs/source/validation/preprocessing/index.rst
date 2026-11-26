Preprocessing
=============

These notebooks retain the original experimental-data regressions for phase
retrieval and stripe removal, including all padding and filtering parameter
combinations. Each records archive checksums, the GPU environment, measured
values and before/after figures. Numerical references are enforced after all
figures have been saved, so an out-of-tolerance result remains a failed notebook.

.. toctree::
   :maxdepth: 1

   Paganin padding comparison <paganin_padding>
   Five stripe-removal methods <stripe_removal>

The Paganin notebook preserves the historical raw projection/flat stack used by
the test. It is a regression example, rather than a recommended acquisition
processing pipeline. Stripe comparisons preserve all sixteen original cases;
Raven uses its original detector rows 10:20.

See :doc:`../zenodo_transfer` for the transfer map and recorded regression status.
