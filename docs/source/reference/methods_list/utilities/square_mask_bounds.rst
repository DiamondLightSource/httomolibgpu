Square crop bounds
==================

``square_mask_bounds``

Compute a square region around a supplied image centre, clipped to the image boundaries. This selects a consistent region for reconstruction quality metrics.

Principle
---------

.. math::

   \ell=\operatorname{round}(\min(H,W)\,q/100)

Usage and parameters
--------------------

q is size_pct, H and W are image dimensions. The return order is x_start, x_end, y_start, y_end; crop as image[y_start:y_end, x_start:x_end]. Edge clipping and integer rounding may reduce the requested side length.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/recon/rotation.py>`_
for the precise numerical conventions used by this utility.

API reference
-------------

:func:`httomolibgpu.recon.rotation.square_mask_bounds` — signature, input/output types and full parameter details.
