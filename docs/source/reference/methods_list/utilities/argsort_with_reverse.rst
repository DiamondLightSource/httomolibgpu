Sorting and inverse sorting indices
===================================

``argsort_with_reverse``

Return indices that sort a one- or two-dimensional array and indices that restore the original ordering. Sorting-based stripe removal uses this to smooth ordered intensities and then recover angular order.

Principle
---------

.. math::

   \pi^{-1}(\pi(i))=i

Usage and parameters
--------------------

Choose axis for a 2D array. Apply the returned indices with axis-aware indexing, such as take_along_axis, rather than treating them as sorted values. The second index array reverses the permutation even when input intensities repeat.

Papers and further reading
--------------------------

The sorting helper supports the following stripe-removal method; it is not
a separate reconstruction algorithm.

* `Superior techniques for eliminating ring artifacts in X-ray micro-tomography <https://doi.org/10.1364/OE.26.028396>`_.

API reference
-------------

:func:`httomolibgpu.misc.sorting.argsort_with_reverse` — signature, input/output types and full parameter details.
