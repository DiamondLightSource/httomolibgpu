.. _image_reconstruction_module:

Image reconstruction methods
****************************

Choose a reconstruction method for your measurement model and data quality.
The :doc:`data conventions guide <../../introduction/workflow>` explains array
layouts, angle units and preprocessing. All methods are listed below; speed
and memory use depend on dataset size, geometry, settings and hardware.

.. list-table:: Reconstruction choices
   :header-rows: 1
   :widths: 25 25 50

   * - Method
     - Measurement model
     - Starting point
   * - :doc:`FBP3D <reconstruction/FBP3d_tomobar>`
     - Transmission line integrals
     - Direct reconstruction using GPU filtering and backprojection.
   * - :doc:`Log-polar <reconstruction/LPRec3d_tomobar>`
     - Transmission line integrals
     - Direct reconstruction; inspect sampling requirements and GPU workspace.
   * - :doc:`FBP2D <reconstruction/FBP2d_astra>`
     - Transmission line integrals
     - Slice-wise processing with NumPy input when limiting GPU workspace matters.
   * - :doc:`CGLS <reconstruction/CGLS3d_tomobar>` / :doc:`SIRT <reconstruction/SIRT3d_tomobar>`
     - Transmission line integrals
     - Classical iterative reconstruction; iteration count controls fitting and noise amplification.
   * - :doc:`FISTA <reconstruction/FISTA3d_tomobar>` / :doc:`ADMM <reconstruction/ADMM3d_tomobar>`
     - Transmission line integrals
     - Iterative reconstruction with configurable regularisation.
   * - :doc:`OSEM <reconstruction/OSEM3d_tomobar>`
     - Nonnegative emission measurements
     - Ordered-subsets EM; not a replacement for transmission reconstruction after a negative logarithm.
   * - :doc:`Iterative alignment <reconstruction/iterative_alignment>`
     - Depends on the chosen reconstruction solver
     - Estimate per-view translations while alternating reconstruction and reprojection.

See :doc:`../../benchmarks/index` for measured performance with recorded settings,
and :doc:`../../validation/reconstruction/index` for validation examples.

.. toctree::
   :maxdepth: 2

   reconstruction/LPRec3d_tomobar
   reconstruction/FBP3d_tomobar
   reconstruction/FBP2d_astra
   reconstruction/regularisation
   reconstruction/FISTA3d_tomobar
   reconstruction/ADMM3d_tomobar
   reconstruction/CGLS3d_tomobar
   reconstruction/SIRT3d_tomobar
   reconstruction/iterative_alignment
   reconstruction/OSEM3d_tomobar
