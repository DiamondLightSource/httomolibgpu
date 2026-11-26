.. _method_ADMM3d_tomobar:

ADMM 3D (ToMoBAR)
^^^^^^^^^^^^^^^^^

**Description**

The Alternating Direction Method of Multipliers (ADMM) :cite:`boyd2011distributed` in tomography is an iterative reconstruction algorithm
that optimises image quality by breaking complex, large-scale, and non-smooth optimization problems into smaller, manageable subproblems.
It is particularly effective for high-quality, 3D imaging from sparse-view or low-dose data, where traditional methods like filtered
back-projection (FBP) produce severe artifacts. It can also employ various regularisation :cite:`kazantsev2019ccpi` models to suppress the noise.
There are few types of regularisation that can be used, please see method's API :func:`httomolibgpu.recon.algorithm.ADMM3d_tomobar`.

**Where and how to use it:**

When the data is highly inaccurate,  noisy, incomplete, or limited-angle data. Due to added regularisation, the quality of ADMM reconstruction  is expected to be better than other classical methods, such as,
:ref:`method_CGLS3d_tomobar` or :ref:`method_SIRT3d_tomobar`.

ADMM always uses the full dataset. ``data_solver="linearised"`` preserves the
existing default; select ``data_solver="cg"`` with ``cg_iterations`` and
``cg_tolerance`` to control the inner data solve. The public penalty and
relaxation names are now ``rho`` and ``relaxation``.

For weighted objectives, select ``data_fidelity="PWLS"`` or ``"SWLS"`` and pass
``weights`` in the same shape and axis order as the input projections. SWLS
also accepts ``beta_swls``. ToMoBAR owns preparation and normalisation of the
weights. The defaults for iterations, initialisation and regularisation are
unchanged.

**What are the adjustable parameters:**

* :code:`iterations` controls the number of full-data ADMM updates. Choose it by checking image quality and convergence.


* :code:`initialisation` Initialise ADMM iterations using another reconstruction algorithm, like FBP, for instance. A so-called warm-start. Usually helps significantly to reduce the number of iterations. However, for very noisy, undersampled data it is recommended to use 'CGLS' as initialisation to avoid boosting up the noise and artifacts of the FBP reconstruction.

* :code:`regularisation_parameter` is probably the second most important parameter after :code:`iterations`. When one increases the value of :code:`regularisation_parameter`, one can expect the image to be smoother. The type of smoothing usually depends on the regularisation type, and for Total-Variation is the piecewise-constant smoothness.

* :code:`regularisation_iterations` controls the inner regularisation iterations in each full-data ADMM iteration. ADMM always uses all views.

* :code:`nonnegativity`. By setting this parameter to :code:`True` imposes positivity constraint on the solution.

See :ref:`iterative_regularisation` for all six regularisers, their additional
controls and strength conventions.

Papers and further reading
--------------------------

* `Distributed optimization and statistical learning via the alternating direction method of multipliers <https://doi.org/10.1561/2200000016>`_.
* `CCPi-Regularisation toolkit for computed tomographic image reconstruction with proximal splitting algorithms <https://doi.org/10.1016/j.softx.2019.04.003>`_.

Validation examples and images
------------------------------

See the saved reconstruction images, processing settings and dataset details
in the following validation notebooks:

* :doc:`ADMM · K11 dataset 2 <../../../validation/reconstruction/regularised/admm3d_k11_dataset2>`


API reference
-------------

:func:`httomolibgpu.recon.algorithm.ADMM3d_tomobar` — signature, input/output types and full parameter details.
