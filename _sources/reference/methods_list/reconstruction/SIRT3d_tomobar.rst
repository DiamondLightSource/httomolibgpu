.. _method_SIRT3d_tomobar:

SIRT 3D (ToMoBAR)
^^^^^^^^^^^^^^^^^

**Description**

Simultaneous Iterative Reconstruction Technique (SIRT) is a widely used iterative algorithm for image reconstruction. It updates the system of linear algebraic equations simultaneously using averages and therefore faster
than Algebraic Reconstruction Technique.  However, it is still slow in convergence and requires hundreds of iterations normally. 
  
**Where and how to use it:**

When the data is noisy, incomplete, or limited-angle data. Normally direct methods do not work well with that kind of data so it is recommended to use iterative methods.

**What are the adjustable parameters:**

* The number of :code:`iterations` is the most important parameter for the method as it requires at least 500 iterations. One can try a faster :ref:`method_CGLS3d_tomobar` method.


.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/sirt_recon_iter100.png
           :width: 320px

           Reconstruction using SIRT with :code:`iterations=100`.

      - .. figure:: ../../../_static/figures/reconstructions/sirt_recon_iter500.png
           :width: 320px

           Reconstruction using SIRT with :code:`iterations=500`.


