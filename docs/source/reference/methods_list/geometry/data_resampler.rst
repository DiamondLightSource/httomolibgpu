Resample detector slices
========================

``data_resampler``

Interpolate two-dimensional slices of a 3D array onto a new grid, leaving the selected axis unchanged. Use this for detector resizing or smaller exploratory datasets.

Usage and parameters
--------------------

Set newshape to the desired two-dimensional slice shape and axis to the dimension to preserve. Choose the interpolation method through the API. Resampling changes effective pixel spacing; update geometry and centre coordinates accordingly. Interpolation alone is not an anti-aliasing filter.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/misc/morph.py>`_
for the precise numerical conventions used by this utility.

Background reading
------------------

For the transmission model, sampling and reconstruction context, see
Kak and Slaney, `Principles of Computerized Tomographic Imaging
<https://www.slaney.org/pct/>`_ (author-hosted text). The numerical
controls specific to this function are defined in the API above.

API reference
-------------

:func:`httomolibgpu.misc.morph.data_resampler` — signature, input/output types and full parameter details.
