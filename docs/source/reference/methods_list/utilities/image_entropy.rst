Image entropy metric
====================

``image_entropy``

Measure the Shannon entropy of the intensity histogram of a reconstructed crop. Used to compare trial reconstructions during metric-based centre finding.

Principle
---------

.. math::

   H=-\sum_{p_i>0}p_i\log_2 p_i

Usage and parameters
--------------------

p_i is a normalised histogram count. The implementation chooses the bin count from image height. Compare consistent crops and scaling; entropy alone does not distinguish noise from structure. Use a sufficiently large nonempty image for meaningful histogram bins.

Papers and further reading
--------------------------

* `A mathematical theory of communication <https://doi.org/10.1002/j.1538-7305.1948.tb01338.x>`_.

API reference
-------------

:func:`httomolibgpu.recon.rotation.image_entropy` — signature, input/output types and full parameter details.
