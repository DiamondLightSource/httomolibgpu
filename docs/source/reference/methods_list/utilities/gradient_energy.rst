Image gradient metric
=====================

``gradient_energy``

Sum finite-difference gradient magnitudes over a reconstructed crop. Despite the function name, this is a gradient-magnitude sum rather than a sum of squared gradients.

Principle
---------

.. math::

   G=\sum_{i,j}\sqrt{(D_xu)_{ij}^2+(D_yu)_{ij}^2}

Usage and parameters
--------------------

Forward differences are used with zero differences at the trailing boundary. The score responds to both edges and noise and depends on image size and intensity scale. Compare identical crops when evaluating candidate centres.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/recon/rotation.py>`_
for the precise numerical conventions used by this utility.

API reference
-------------

:func:`httomolibgpu.recon.rotation.gradient_energy` — signature, input/output types and full parameter details.
