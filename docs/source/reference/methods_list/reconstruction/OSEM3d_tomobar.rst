Ordered-subsets expectation maximisation
========================================

``OSEM3d_tomobar``

Reconstruct nonnegative emission measurements using ordered-subsets EM, with optional regularisation after subset updates. This is an emission model, rather than a transmission line-integral reconstruction.

Principle
---------

.. math::

   x^{+}=\frac{x}{A_s^{\mathsf T}\mathbf{1}}\odot\left[A_s^{\mathsf T}\left(\frac{b_s}{A_sx}\right)\right]

Usage and parameters
--------------------

This is the ideal unregularised subset update, with elementwise products and division. A_s is the subset projector and b_s its emission data. subsets_number=1 gives MLEM. iterations counts full visits to all subsets. Increasing subsets changes convergence and noise behaviour. See :ref:`iterative_regularisation` for regularisation settings; regularisation_parameter=0 disables smoothing.

Validation coverage
-------------------

The :download:`automated validation tests <../../../../../tests/test_recon/test_osem_api.py>`
cover option forwarding, geometry/output layout and native MLEM/subset GPU execution.
Run them from a compatible source checkout with a CUDA device; the downloaded
file is test-suite source, not a standalone notebook.

There is currently no saved validation notebook for this method. Browse the
:doc:`reconstruction validation collection <../../../validation/reconstruction/index>`
for the available notebook-based examples.

Papers and further reading
--------------------------

* `Accelerated image reconstruction using ordered subsets of projection data <https://doi.org/10.1109/42.363108>`_.
* `CCPi-Regularisation toolkit for computed tomographic image reconstruction with proximal splitting algorithms <https://doi.org/10.1016/j.softx.2019.04.003>`_.

API reference
-------------

:func:`httomolibgpu.recon.algorithm.OSEM3d_tomobar` — signature, input/output types and full parameter details.
