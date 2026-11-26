.. _iterative_regularisation:

Regularisation options
^^^^^^^^^^^^^^^^^^^^^^

:func:`~httomolibgpu.recon.algorithm.FISTA3d_tomobar`,
:func:`~httomolibgpu.recon.algorithm.ADMM3d_tomobar` and
:func:`~httomolibgpu.recon.algorithm.OSEM3d_tomobar` support all six ToMoBAR
regularisers. Select one with ``regularisation_type`` and configure its inner
solver with the keyword-only ``regularisation_options`` dictionary.

.. list-table:: Supported methods and additional controls
   :header-rows: 1
   :widths: 15 35 50

   * - Method
     - Description
     - Additional controls (defaults)
   * - :ref:`ROF_TV <method_total_variation_ROF>`
     - Smoothed total variation
     - ``time_step=0.005``
   * - :ref:`PD_TV <method_total_variation_PD>`
     - Primal-dual total variation (default)
     - ``tv_type=0``, ``lipschitz_constant=12.0``
   * - :ref:`SB_TV <method_total_variation_SB>`
     - Split Bregman total variation
     - ``tv_type=0``
   * - :ref:`FGP_TV <method_total_variation_FGP>`
     - Fast Gradient Projection total variation
     - ``tv_type=0``
   * - :ref:`TGV <method_total_generalised_variation>`
     - Second-order Total Generalised Variation
     - ``alpha1=1.0``, ``alpha0=2.0``, ``lipschitz_constant=12.0``
   * - :ref:`NDF <method_nonlinear_diffusion>`
     - Nonlinear diffusion
     - ``sigma=0.01``, ``penalty_type=1``, ``time_step=0.005``

``tv_type=0`` selects isotropic TV and ``1`` selects anisotropic TV for PD-TV,
SB-TV and FGP-TV. TGV supports only isotropic regularisation. Its alpha weights
balance first- and second-order terms. The inner ``lipschitz_constant`` is a
squared derivative-operator bound; ToMoBAR raises it to a safe minimum. It is
separate from the reconstruction operator's norm estimate.

For NDF, ``sigma/sqrt(2)`` is the effective threshold in reconstructed-image
intensity units; ``sigma=0`` selects linear diffusion. ``penalty_type`` selects
1 Huber, 2 Perona–Malik, 3 Tukey biweight, 4 truncated quadratic or 5 truncated
Huber. Penalties 2–5 are nonconvex: using them as denoising updates does not
provide the usual convex proximal-solver convergence guarantee. NDF caps its
time step internally and requires ``regularisation_half_precision=False``.

Strength, iterations and precision
----------------------------------

Use the existing ``regularisation_parameter``, ``regularisation_iterations``
and ``regularisation_half_precision`` arguments for the common controls.
Strength zero disables regularisation. The reconstruction's ``nonnegativity``
argument controls the image constraint separately.

Strength conventions are unchanged: FISTA's public parameter is per-proximal-step
strength; ADMM uses objective strength and divides it by ``rho`` for each
proximal update; OSEM applies the public strength at each subset update.
Do not assume the same numerical strength gives equal smoothing between methods.

``regularisation_options`` uses native ToMoBAR field names. Unknown keys and
attempts to override method, strength, iterations or precision through this
dictionary are rejected. Fields are validated even when unused by the selected
method. The supplied dictionary is not modified.

Examples
--------

.. code-block:: python

    from httomolibgpu import FISTA3d_tomobar, ADMM3d_tomobar

    # data: CuPy projections in (angles, detector rows, detector columns) order.
    # angles: NumPy angles in radians.
    reconstruction = FISTA3d_tomobar(
        data,
        angles,
        regularisation_type="TGV",
        regularisation_parameter=1e-6,
        regularisation_options={"alpha1": 1.0, "alpha0": 2.0},
    )

    reconstruction = ADMM3d_tomobar(
        data,
        angles,
        regularisation_type="NDF",
        regularisation_parameter=0.001,
        regularisation_half_precision=False,
        regularisation_options={
            "sigma": 0.02,
            "penalty_type": 2,
            "time_step": 0.005,
        },
    )

These settings illustrate the API and need adjustment for the data's scale,
noise level and selected solver.

Validation examples and images
------------------------------

See the saved reconstruction images, processing settings and dataset details
in the following validation notebooks:

* :doc:`FISTA · K11 dataset 2 <../../../validation/reconstruction/regularised/fista3d_k11_dataset2>`
* :doc:`ADMM · K11 dataset 2 <../../../validation/reconstruction/regularised/admm3d_k11_dataset2>`
