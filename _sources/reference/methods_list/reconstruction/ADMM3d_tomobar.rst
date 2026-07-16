.. _method_ADMM3d_tomobar:

ADMM 3D (ToMoBAR)
^^^^^^^^^^^^^^^^^^

**Description**

The Alternating Direction Method of Multipliers (ADMM) :cite:`boyd2011distributed` in tomography is an iterative reconstruction algorithm
that optimises image quality by breaking complex, large-scale, and non-smooth optimization problems into smaller, manageable subproblems.
It is particularly effective for high-quality, 3D imaging from sparse-view or low-dose data, where traditional methods like filtered
back-projection (FBP) produce severe artifacts. It can also employ various regularisation :cite:`kazantsev2019ccpi` models to suppress the noise.
There are few types of regularisation that can be used, please see method's API :mod:`httomolibgpu.recon.algorithm.ADMM3d_tomobar`.

**Where and how to use it:**

When the data is highly inaccurate,  noisy, incomplete, or limited-angle data. Due to added regularisation, the quality of ADMM reconstruction  is expected to be better than other classical methods, such as,
:ref:`method_CGLS3d_tomobar` or :ref:`method_SIRT3d_tomobar`.

**What are the adjustable parameters:**

* The number of :code:`iterations` is an important parameter as one would like to achieve a trade-off between the resolution and SNR. For ADMM-OS method (when :code:`subsets_number > 1`) the range of iterations depends on how many :code:`subsets_number` used. For 24 subsets, 3-5 iterations is usually enough.

* :code:`subsets_number` usually helps with the faster convergence of the reconstruction algorithm. Fortunately, ADMM is much more robust for higher numbers of subsets, compared to :ref:`method_FISTA3d_tomobar`. 24 or larger number of subsets can be used and :code:`iterations` can be reduced when the subsets grow.

* :code:`initialisation` Initialise ADMM iterations using another reconstruction algorithm, like FBP, for instance. A so-called warm-start. Usually helps significantly to reduce the number of iterations. However, for very noise, undersampled, data it is recommended to use 'CGLS' as initialisation to avoid boosting up the noise and artifacts of the FBP reconstruction.

* :code:`regularisation_parameter` is probably the second most important parameter after :code:`iterations`. When one increases the value of :code:`regularisation_parameter`, one can expect the image to be smoother. The type of smoothing usually depends on the regularisation type, and for Total-Variation is the piecewise-constant smoothness.

* :code:`regularisation_iterations` defines how many inner iterations for regularisation performed on every step of the outer algorithm. This can depend on :code:`regularisation_type` and  :code:`subsets_number`. The general rule is when :code:`subsets_number` is smaller then :code:`regularisation_iterations` should be increased.

* :code:`nonnegativity`. By setting this parameter to :code:`True` imposes positivity constraint on the solution.

