.. _method_non_local_means:

Non-local means denoising
^^^^^^^^^^^^^^^^^^^^^^^^

:func:`httomolibgpu.misc.denoise.non_local_means` averages pixels whose
surrounding patches resemble the patch around the pixel being filtered.
It accepts a float32 CuPy image or stack. For a stack, it processes each
``data[index, :, :]`` image independently, without smoothing between slices.

.. code-block:: python

   import cupy as cp
   from httomolibgpu import non_local_means

   # The example h assumes image intensities are approximately in [0, 1].
   images = cp.asarray(reconstructed_volume, dtype=cp.float32)
   denoised = non_local_means(
       images,
       h=0.2,
       search_radius=5,
       patch_radius=2,
       num_neighbours=0,
   )

The input remains unchanged. The result is a new, C-contiguous float32 CuPy
array on the same GPU. Computation uses the current CuPy stream and does not
transfer image data to the CPU. Non-contiguous views are copied on the GPU.
Input values must be finite; the method does not scan for NaNs or infinities.

Patch distances and weights
--------------------------

For a pixel :math:`p` and candidate :math:`q`, the weight is

.. math::

   d(p,q) = \sum_{r \in V(p,q)}
       G(r) [u(p+r) - u(q+r)]^2, \qquad
   w(p,q) = \exp[-d(p,q)/h^2],

where :math:`G(r)=\exp[-\|r\|^2/(2R^2)]` for patch radius :math:`R>0`.
For radius zero the patch contains one pixel with Gaussian weight one.
The set :math:`V(p,q)` contains offsets inside both image patches. Search
centres outside the image are excluded. Truncated Gaussian weights are not
renormalised, so border pixels can receive stronger smoothing.

The output is the normalised weighted average of selected candidate pixels
and the original centre pixel, whose weight is one. Equal patches have
weight one and remain valid neighbours. Even when other weights underflow
to zero, the centre pixel keeps the result defined.

Choosing parameters
-------------------

* ``h`` controls smoothing in input intensity units. Increase it to give
  less-similar patches more influence. Intensities are not normalised by the
  method. Patch distances are sums, so changing the patch radius can require
  retuning ``h``. These values are not interchangeable with methods that use
  an average patch distance.
* ``search_radius`` is the half-width of the square search window. A radius
  of five searches an 11 by 11 window. Zero returns a copy of the input.
* ``patch_radius`` is the half-width of the comparison patch and its Gaussian
  standard deviation. A radius of two compares 5 by 5 patches.
* ``num_neighbours=0`` uses every candidate. A positive value selects that
  many strongest non-self matches, plus the centre pixel. Ties retain
  candidates in x-major search order. Borders may have fewer candidates.

The implementation evaluates patches directly. Work grows with both search
area and patch area; start with small radii for large reconstructions.
It does not allocate a full neighbour graph. Positive ``num_neighbours``
values use per-thread selection arrays, whose storage and insertion cost
grow with the requested count. Kernels are compiled and cached by CuPy for
each selected neighbour count; the first call includes compilation overhead.

Relation to the CCPi implementation
----------------------------------

The Gaussian-weighted distance and optional strongest-neighbour selection
follow `CCPi's PatchSelect_GPU_core.cu
<https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit/blob/main/src/Core/regularisers_GPU/PatchSelect_GPU_core.cu>`_.
That source computes a neighbour graph for non-local regularisation.
This method instead returns a denoised image by adding a weighted average.
It is not a replacement for the graph-selection API or a non-local TV solver.

The implementation also uses Gaussian indices tied to the original patch
offsets at boundaries, accepts zero-distance matches, and checks every
thread's image bounds. Consequently, it is not numerically identical to
CCPi patch selection. It supports arbitrary non-negative search radii and
does not use 16-bit neighbour coordinates.
