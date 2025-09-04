.. _method_CGLS3d_tomobar:

CGLS 3D (ToMoBAR)
^^^^^^^^^^^^^^^^^

**Description**

Conjugate Gradient for Least Squares (CGLS) is an iterative approach to solve the reconstruction problems via least-squares minimisation. 

  
**Where and how to use it:**

When the data is noisy, incomplete, or limited-angle data. CGLS is more efficient and accurate than classical iterative methods but requires careful handling to avoid amplifying noise. 
The noise amplification or overfitting can happen when too many iterations performed.

**What are the adjustable parameters:**

* The number of :code:`iterations` is the most important parameter as too few iterations leave the reconstruction blurry, while too many lead to noise amplification. Keep the total number of iterations within the 20-30 range is usually a good practice. 

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/cgls_recon_iter10.png
           :width: 320px

           Reconstruction using CGLS with :code:`iterations=10`.

      - .. figure:: ../../../_static/figures/reconstructions/cgls_recon_iter25.png
           :width: 320px

           Reconstruction using CGLS with :code:`iterations=25`.


