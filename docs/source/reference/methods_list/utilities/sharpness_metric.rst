Laplacian sharpness metric
==========================

``sharpness_metric``

Measure the variance of the image Laplacian. Sharply focused boundaries tend to increase the score; it can be used when comparing reconstructed crops.

Principle
---------

.. math::

   S=\operatorname{Var}(\Delta u)

Usage and parameters
--------------------

A higher score indicates stronger high-frequency content, which may also be caused by noise. Keep crop size and intensity scaling fixed when comparing trial reconstructions.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/recon/rotation.py>`_
for the precise numerical conventions used by this utility.

API reference
-------------

:func:`httomolibgpu.recon.rotation.sharpness_metric` — signature, input/output types and full parameter details.
