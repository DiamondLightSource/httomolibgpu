Negative logarithm
==================

``minus_log``

Convert corrected transmission into line integrals for absorption reconstruction. Apply after dark/flat correction and any transmission-space operations.

Principle
---------

.. math::

   p = -\ln T

Usage and parameters
--------------------

T is transmission and p is the integrated attenuation. Input must be positive: zeros or negative values produce non-finite results. Paganin filtering already includes a logarithm, so do not apply minus_log again to its output.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/prep/normalize.py>`_
for the precise numerical conventions used by this utility.

Background reading
------------------

For the transmission model, sampling and reconstruction context, see
Kak and Slaney, `Principles of Computerized Tomographic Imaging
<https://www.slaney.org/pct/>`_ (author-hosted text). The numerical
controls specific to this function are defined in the API above.

API reference
-------------

:func:`httomolibgpu.prep.normalize.minus_log` — signature, input/output types and full parameter details.
