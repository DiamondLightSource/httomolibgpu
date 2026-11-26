Phase-correlation centre finding
================================

``find_center_pc``

Register one projection against a horizontally mirrored projection acquired 180 degrees apart. The horizontal translation encodes the displacement of the rotation axis from the detector centre.

Principle
---------

.. math::

   C=\mathcal{F}^{-1}\left[\frac{F_1\overline{F_2}}{|F_1\overline{F_2}|}\right]

Usage and parameters
--------------------

F_1 and F_2 are Fourier transforms of the opposing images; the peak of C estimates translation. This is the ideal normalised phase-correlation model; numerical handling and refinement follow the implementation. Choose genuinely opposing views with recognisable structure. Motion and non-overlapping features can bias the result.

Papers and further reading
--------------------------

* `Efficient subpixel image registration algorithms <https://doi.org/10.1364/OL.33.000156>`_.

Validation examples and images
------------------------------

See :doc:`the saved validation notebook <../../../validation/rotation/center_pc>`.
The I12 opposing-projection case, with the exact detector-row crop and centre/dtype assertions.


API reference
-------------

:func:`httomolibgpu.recon.rotation.find_center_pc` — signature, input/output types and full parameter details.
