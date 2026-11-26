Round away from zero
====================

``round_up``

Round a real value to the next integer away from zero. This helper supports centre-search bounds.

Principle
---------

.. math::

   r(x)=\operatorname{sign}(x)\lceil |x|\rceil

Usage and parameters
--------------------

Positive values use ceil and negative values use floor. Exact integers are unchanged. This is different from rounding to the nearest integer.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/recon/rotation.py>`_
for the precise numerical conventions used by this utility.

API reference
-------------

:func:`httomolibgpu.recon.rotation.round_up` — signature, input/output types and full parameter details.
