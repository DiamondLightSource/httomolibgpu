.. _method_SIRT3d_tomobar:

SIRT 3D (ToMoBAR)
^^^^^^^^^^^^^^^^^

**Description**

Simultaneous Iterative Reconstruction Technique (SIRT) is a widely used iterative algorithm for image reconstruction. It updates the system of linear algebraic equations simultaneously using averages and therefore faster
than Algebraic Reconstruction Technique. The following iterations are performed: :math:`x^{k+1} = \mathbf{C}\mathbf{A}^{\intercal}\mathbf{R}(b - \mathbf{A}x^{k})`,
where :math:`\mathbf{A}` is the forward projection or geometry matrix, :math:`x` is the sought solution (reconstructed image), :math:`b` is the vectorised projection data,
and :math:`\mathbf{A}^{\intercal}` is the inverse projection operator. Matrices :math:`\mathbf{R}` and :math:`\mathbf{C}` are the preconditioning matrices
which are pre-calculated before the main iterations.


**Where and how to use it:**

When the data is noisy, incomplete, or limited-angle data. Normally direct methods do not work well with that kind of data so it is recommended to use iterative methods.
Note the SIRT method is slow in convergence and it also requires hundreds of iterations normally, we recommended to use faster iterative methods, such as :ref:`method_CGLS3d_tomobar` or even more advanced :ref:`method_FISTA3d_tomobar`.


**What are the adjustable parameters:**

* The number of :code:`iterations` is an important parameter for the method as as too few iterations leave the reconstruction blurry, while too many lead to noise amplification. SIRT might require 200-400 iterations, so one can also try a faster :ref:`method_CGLS3d_tomobar` method.


.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_pad_noisy_data.jpg
           :width: 200px

           :ref:`method_LPRec3d_tomobar` reconstruction is extremely noisy as the data is undersampled with a rapid exposure to the X-ray beam.

      - .. figure:: ../../../_static/figures/reconstructions/sirt_recon_iter100.jpg
           :width: 200px

           SIRT reconstruction with :code:`iterations=100` improves signal-to-noise drastically, but the reconstruction is slightly blurred.

      - .. figure:: ../../../_static/figures/reconstructions/sirt_recon_iter250.jpg
           :width: 200px

           SIRT reconstruction with :code:`iterations=250` improves the resolution further more, yet the noise levels are low.

