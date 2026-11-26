Vo centre-of-rotation search
============================

``find_center_vo``

Estimate the rotation centre from sinogram consistency in Fourier space, using a coarse search followed by refinement. This avoids reconstructing every candidate centre.

Usage and parameters
--------------------

Select a representative detector row with ind. Use the search bounds and step controls to bracket the plausible axis, then refine. The Fourier mask suppresses regions that should contain little signal for a consistent sinogram. Truncation, strong stripes or insufficient angular coverage can bias the estimate. See :cite:`vo2014reliable`.

Centre-search animation
-----------------------

.. raw:: html

   <video controls playsinline preload="metadata" aria-label="Vo centre of rotation search using a Fourier consistency score"
          poster="../../../_static/workshop/cor-search-poster.png" style="width:100%;height:auto;">
     <source src="../../../_static/workshop/cor-search.mp4" type="video/mp4">
     Your browser does not support embedded video.
   </video>

:download:`Download the video <../../../_static/workshop/cor-search.mp4>`.

This synthetic demonstration from the HTTomo workshop varies a trial centre,
extends the measured sinogram to 360 degrees using mirrored data, evaluates
Fourier magnitude outside the expected support and selects the score minimum.
The 360-degree extension here is constructed for the search; it does not mean
that a physical 360-degree offset acquisition is required for this method.
See :cite:`vo2014reliable` for the method.

Papers and further reading
--------------------------

* `Reliable method for calculating the center of rotation in parallel-beam tomography <https://doi.org/10.1364/OE.22.019078>`_.

Validation examples and images
------------------------------

See :doc:`the saved validation notebook <../../../validation/rotation/center_vo>`.
Six original centre/dtype checks across five datasets, including the I12 row-averaging case.

For the separately executed timing case, see the :doc:`Vo benchmark
<../../../benchmarks/center_vo_i12_dataset1>`.


API reference
-------------

:func:`httomolibgpu.recon.rotation.find_center_vo` — signature, input/output types and full parameter details.
