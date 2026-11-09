.. _method_FBP2d_astra:

FBP2D (ASTRA-Toolbox)
^^^^^^^^^^^^^^^^^^^^^

**Description**

This module performs Filtered-Back-projection direct reconstruction using `ASTRA-Toolbox <https://astra-toolbox.com>`_. It is an implementation of parallel-beam
`FBP_CUDA <https://astra-toolbox.com/docs/algs/FBP_CUDA.html>`_ reconstruction method by ASTRA.

.. note:: This is the slowest (slice-by-slice) direct reconstruction method in the library and it is recommended to use other faster methods, such as, :ref:`method_LPRec3d_tomobar` or :ref:`method_FBP3d_tomobar`.

**Where and how to use it:**

This reconstruction method can be used as a back-up option if other reconstruction algorithms do not work for some reason (e.g., GPU memory errors). As a 2D/slice-by-slice method, it has the lowest
GPU memory footprint and therefore a good choice on systems with very limited GPU memory available.

**What are the adjustable parameters:**

See :func:`httomolibgpu.recon.algorithm.FBP2d_astra` for the parameters
accepted by this wrapper. The filter options are described in ASTRA's
`FBP_CUDA documentation <https://astra-toolbox.com/docs/algs/FBP_CUDA.html>`_.

