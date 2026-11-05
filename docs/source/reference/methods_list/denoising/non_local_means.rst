.. _method_non_local_means:

Non-local means denoising
^^^^^^^^^^^^^^^^^^^^^^^^

:func:`httomolibgpu.misc.denoise.non_local_means` averages pixels whose
surrounding patches resemble the patch around the pixel being filtered.
It accepts a float32 CuPy image or volume. By default, a volume uses cubic
search windows and cubic comparison patches, including neighbouring slices.
Set ``full_3d=False`` to process each ``data[z, :, :]`` image independently.
A 2D input always uses square search windows and patches.

.. code-block:: python

   import cupy as cp
   from httomolibgpu import non_local_means

   # The example h assumes image intensities are approximately in [0, 1].
   images = cp.asarray(reconstructed_volume, dtype=cp.float32)
   denoised = non_local_means(
       images,
       h=0.2,
       search_radius=2,
       patch_radius=1,
       num_neighbours=0,
       full_3d=True,
   )

To apply the original 2D method independently to each slice, use:

.. code-block:: python

   denoised_slices = non_local_means(
       images, h=0.2, search_radius=5, patch_radius=2, full_3d=False
   )

The input remains unchanged. The result is a new, C-contiguous float32 CuPy
array on the same GPU. Computation uses the current CuPy stream and does not
transfer image data to the CPU. Non-contiguous views are copied on the GPU.
Input values must be finite; the method does not scan for NaNs or infinities.

Patch distances and weights
--------------------------

For a pixel or voxel :math:`p` and candidate :math:`q`, the weight is

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
  retuning ``h``. Switching from 2D to 3D patches also requires retuning it.
  These values are not interchangeable with methods that use an average
  patch distance.
* ``search_radius`` is the half-width of the square or cubic search window.
  A radius of two searches a 5 by 5 by 5 window in 3D, or a 5 by 5 window
  in 2D. Zero returns a copy of the input.
* ``patch_radius`` is the half-width of the comparison patch and its Gaussian
  standard deviation. A radius of one compares 3 by 3 by 3 patches in 3D.
* ``num_neighbours=0`` uses every candidate. A positive value selects that
  many strongest non-self matches, plus the centre pixel. Ties retain
  candidates in x, then y, then z loop order (z only in 3D). Borders may
  have fewer candidates.
* ``full_3d=True`` enables volumetric processing of a 3D array. Set it to
  ``False`` for independent 2D slices along axis zero. Both flag values
  produce 2D processing when the input itself is 2D.

The implementation evaluates patches directly. Work grows with both search
size and patch size; start with small radii for large reconstructions.
For 3D, the approximate work per voxel is
``((2 * search_radius + 1)**3 - 1) * (2 * patch_radius + 1)**3``
patch-element comparisons away from boundaries. The existing defaults of
``search_radius=5`` and ``patch_radius=2`` are much more expensive in 3D;
the smaller radii in the example above are a starting point for benchmarking.
It does not allocate a full neighbour graph. Positive ``num_neighbours``
values use per-thread selection arrays, whose storage and insertion cost
grow with the requested count. Kernels are compiled and cached by CuPy for
each processing mode and selected neighbour count; the first call includes
compilation overhead.

Processing volumes in chunks
---------------------------

The method filters the array supplied to it; it does not fetch neighbouring
chunks. For full 3D processing, include an input overlap of
``search_radius + patch_radius`` voxels on each side of every chunked axis.
Keep only the output corresponding to the chunk interior. At the physical
volume boundaries, retain the existing truncated-patch behaviour.

For example, a search radius of two and patch radius of one require three
extra slices at each interior slab boundary. Without this overlap, slab
boundaries behave as image boundaries and results can differ from filtering
the whole volume. Slice-wise processing requires no overlap along axis zero.

Relation to the CCPi implementation
----------------------------------

The Gaussian-weighted distance and optional strongest-neighbour selection
extend the 2D model in `CCPi's PatchSelect_GPU_core.cu
<https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit/blob/main/src/Core/regularisers_GPU/PatchSelect_GPU_core.cu>`_.
That source computes a neighbour graph for non-local regularisation.
This method instead returns a denoised image by adding a weighted average.
It is not a replacement for the graph-selection API or a non-local TV solver.

The implementation also uses Gaussian indices tied to the original patch
offsets at boundaries, accepts zero-distance matches, and checks every
thread's image bounds. Consequently, it is not numerically identical to
CCPi patch selection. It supports arbitrary non-negative search radii and
does not use 16-bit neighbour coordinates.
