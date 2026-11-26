Convert an image to uint16
==========================

``to_uint16``

Scale an image independently between its finite range endpoints for display or export. This helper does not preserve physical attenuation units or common scaling across images.

Principle
---------

.. math::

   y=\operatorname{round}\left(65535\frac{x-x_{\min}}{x_{\max}-x_{\min}}\right)

Usage and parameters
--------------------

A constant image becomes zero. Range endpoints use nanmin and nanmax, but remaining non-finite values are not repaired before casting. For controlled dataset-wide scaling use rescale_to_int instead.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/recon/rotation.py>`_
for the precise numerical conventions used by this utility.

API reference
-------------

:func:`httomolibgpu.recon.rotation.to_uint16` — signature, input/output types and full parameter details.
