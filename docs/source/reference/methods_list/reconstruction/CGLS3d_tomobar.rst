.. _method_CGLS3d_tomobar:

CGLS 3D (ToMoBAR)
^^^^^^^^^^^^^^^^^

**Description**

Conjugate Gradient for Least Squares (CGLS) is an iterative approach to solve the reconstruction problems via least-squares minimisation.
The algorithm aims to solve the system of normal equations :math:`\mathbf{A}^{\intercal}\mathbf{A}x = \mathbf{A}^{\intercal} b`, where
:math:`\mathbf{A}` is the forward projection or geometry matrix, :math:`x` is the sought solution (reconstructed image), :math:`b` is the vectorised projection data,
and :math:`\mathbf{A}^{\intercal}` is the inverse projection operator.


**Where and how to use it:**

When the tomographic data is noisy, incomplete, or/and it is the limited-angle data. CGLS is more efficient and accurate than classical iterative methods but requires careful handling of the number of iterations to avoid amplifying noise (method's divergence).
Normally, noise amplification or overfitting can happen when too many iterations performed.

**What are the adjustable parameters:**

* The number of :code:`iterations` is an important parameter as too few iterations leave the reconstruction blurry, while too many lead to noise amplification. Keep the total number of iterations within the 15-30 range is usually a good practice.


.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_pad_noisy_data.jpg
           :width: 200px

           :ref:`method_LPRec3d_tomobar` reconstruction is extremely noisy as the data is undersampled with a rapid exposure to the X-ray beam.

      - .. figure:: ../../../_static/figures/reconstructions/cgls_recon_iter10.jpg
           :width: 200px

           CGLS reconstruction with :code:`iterations=10` improves signal-to-noise drastically, but the reconstruction is slightly blurred.

      - .. figure:: ../../../_static/figures/reconstructions/cgls_recon_iter20.jpg
           :width: 200px

           CGLS reconstruction with :code:`iterations=20` improves the resolution but the noise is also amplified.


* :code:`nonnegativity`. By setting this parameter to :code:`True` imposes positivity constraint on the solution. In some cases that can improve the contrast.


.. figure:: ../../../_static/figures/reconstructions/cgls_recon_iter15_nonneg.jpg
   :scale: 50 %
   :alt: CGLS rec

   CGLS reconstruction with :code:`iterations=15` and :code:`nonnegativity=True`.



