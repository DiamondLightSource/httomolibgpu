.. _method_FISTA3d_tomobar:

FISTA 3D (ToMoBAR)
^^^^^^^^^^^^^^^^^^

**Description**

FISTA stands for A Fast Iterative Shrinkage-Thresholding Algorithm :cite:`beck2009fast`. One of the major benefits of FISTA is the ability
to build complex optimisation functionals that can handle a variety of problematic data. For instance, the amplification of the noise
in iterations which is common for classical iterative methods, such as, :ref:`method_CGLS3d_tomobar` and :ref:`method_SIRT3d_tomobar`,
can be resolved by using regularisation. There are many different types of regularisation that can be used, see some are listed here :cite:`kazantsev2019ccpi`.


**Where and how to use it:**

When the data is highly inaccurate,  noisy, incomplete, or limited-angle data. Due to added regularisation, the quality of FISTA reconstruction  is expected to be better than other classical methods, such as,
:ref:`method_CGLS3d_tomobar` or :ref:`method_SIRT3d_tomobar`.

**What are the adjustable parameters:**

* The number of :code:`iterations` is an important parameter as one would like to achieve a trade-off between the resolution and SNR. For FISTA-OS method (when :code:`subsets_number > 5`) the range of iterations between 10 and 20 is a good choice normally.

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_pad_noisy_data.jpg
           :width: 200px

           :ref:`method_LPRec3d_tomobar` reconstruction is extremely noisy as the data is undersampled with a rapid exposure to the X-ray beam.

      - .. figure:: ../../../_static/figures/reconstructions/fistaos_recon_iter6.jpg
           :width: 200px

           FISTA-OS reconstruction with :code:`iterations=6` and using :ref:`method_total_variation_PD` as regularisation, :code:`regularisation_parameter = 7.0e-07`. See the improved resolution and the absence of noise.

      - .. figure:: ../../../_static/figures/reconstructions/fistaos_recon_iter15.jpg
           :width: 200px

           FISTA-OS reconstruction with :code:`iterations=15` provides even crispier resolution without significant noise amplification due to regularisation.

* :code:`regularisation_parameter` is probably the second most important parameter after :code:`iterations`. When one increases the value of :code:`regularisation_parameter`, one can expect the image to be smoother. The type of smoothing usually depends on the regularisation type, and for Total-Variation is the piecewise-constant smoothness.

.. list-table::

      * - .. figure:: ../../../_static/figures/reconstructions/fistaos_recon_iter10_regul_high.jpg
           :width: 300px

           FISTA-OS reconstruction with :code:`iterations=10` and increased value of :code:`regularisation_parameter = 5.0e-06`. See how the dynamic range has been reduced further by smoothing out smaller features.

        - .. figure:: ../../../_static/figures/reconstructions/fistaos_recon_iter10_regul_high_nonneg.jpg
           :width: 300px

           Same as left but :code:`nonnegativity = True`. See how reconstruction now almost look like a segmented image.

* :code:`regularisation_iterations` defines how many inner iterations for regularisation performed on every step of the outer (FISTA) algorithm. This can depend on :code:`regularisation_type` and  :code:`subsets_number`. The general rule is when :code:`subsets_number` is smaller then :code:`regularisation_iterations` should be increased.

* :code:`subsets_number` usually helps with the faster convergence of the reconstruction algorithm. However, higher values can lead to divergence. It is recommended to keep that value between 4 and 8. In the case of the incorrect reconstruction, one can set :code:`subsets_number` to 1, therefore enabling a classical FISTA algorithm. For the classical FISTA one needs to run significantly more outer :code:`iterations`, in the range between 200-500.

* :code:`nonnegativity`. By setting this parameter to :code:`True` imposes positivity constraint on the solution. In certain situations, when reconstructions require segmentation, one can try enabling it.


