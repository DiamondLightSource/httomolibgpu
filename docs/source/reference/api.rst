.. _reference_api:

==========================
API reference
==========================

Function signatures and parameters, grouped by processing stage.
See :doc:`methods` for method descriptions and validation examples.

Preprocessing
-------------

.. toctree::
   :maxdepth: 1

   Alignment <../api/httomolibgpu.prep.alignment>
   Normalisation <../api/httomolibgpu.prep.normalize>
   Phase retrieval <../api/httomolibgpu.prep.phase>
   Stripe removal <../api/httomolibgpu.prep.stripe>

Reconstruction
--------------

.. toctree::
   :maxdepth: 1

   Reconstruction algorithms <../api/httomolibgpu.recon.algorithm>
   Centre of rotation <../api/httomolibgpu.recon.rotation>

Image processing
----------------

.. toctree::
   :maxdepth: 1

   Corrections <../api/httomolibgpu.misc.corr>
   Denoising <../api/httomolibgpu.misc.denoise>
   Morphology and resampling <../api/httomolibgpu.misc.morph>
   Rescaling <../api/httomolibgpu.misc.rescale>
   Blending <../api/httomolibgpu.misc.blend>

Utilities
---------

.. toctree::
   :maxdepth: 1

   Data checks <../api/httomolibgpu.misc.utils>
   Sorting <../api/httomolibgpu.misc.sorting>
   CuPy compatibility <../api/httomolibgpu.cupywrapper>
   CUDA kernels <../api/httomolibgpu.cuda_kernels>
