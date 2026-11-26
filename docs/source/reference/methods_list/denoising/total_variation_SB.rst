.. _method_total_variation_SB:

Total Variation denoising (Split Bregman)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**

:func:`httomolibgpu.misc.denoise.total_variation_SB` uses Split Bregman
iterations :cite:`goldstein2009split` to minimise an exact, nonsmoothed TV
penalty. Like :ref:`method_total_variation_PD`, it favours piecewise-constant
images; the optimisation strategy differs, rather than the underlying TV model.

For noisy data :math:`f`, output :math:`u` and strength :math:`\lambda`, the
isotropic denoising problem is

.. math::

   \min_{u\in C}\;\frac12\|u-f\|_2^2+\lambda\operatorname{TV}(u),
   \qquad \operatorname{TV}(u)=\sum_p\|(Du)_p\|_2.

Here :math:`D` is the forward spatial difference operator, with zero boundary
flux. :math:`C` is unconstrained by default, or the nonnegative images when
``nonnegativity=True``. Split Bregman introduces :math:`d=Du`, giving

.. math::

   \min_{u\in C,d}\;\frac12\|u-f\|_2^2+\lambda\sum_p\|d_p\|_2,
   \qquad d=Du.

The implementation alternates image-update sweeps, shrinkage of the split
gradient and a Bregman-variable update. With ``isotropic=False``, the TV term
is instead :math:`\sum_{p,j}|D_j u_p|`, penalising directional components
separately. Finite iteration counts give an approximate solution.

**Where and how to use it:**

Use it to suppress noise while retaining sharp boundaries in reconstructed
images or volumes. Constant regions suit the TV model; smoothly varying
regions may develop staircase-like steps. For smooth intensity ramps, compare
with :ref:`method_total_generalised_variation`.

Input is a finite float32 CuPy volume in ``(Z, Y, X)`` order on ``gpu_id``.
Represent a 2D image as ``image[None, :, :]``. The function returns a new
C-contiguous float32 array with the same shape; the input is not modified.

.. code-block:: python

   from httomolibgpu import total_variation_SB

   denoised = total_variation_SB(
       volume,
       regularisation_parameter=0.1,
       iterations=300,
       isotropic=True,
       nonnegativity=True,
   )

**What are the adjustable parameters:**

* ``regularisation_parameter`` is the nonnegative strength, default 0.001.
  Larger values increase smoothing; zero returns a copy, with nonnegativity
  applied if requested. Retune after changing the data's intensity scale.
* ``iterations`` is the fixed iteration count, default 150. Each iteration
  includes two image-update sweeps. More iterations add work and need not
  improve agreement with a clean reference image.
* ``isotropic`` selects the Euclidean gradient norm (True, default) or the
  sum of absolute directional differences (False).
* ``nonnegativity`` constrains image values to be nonnegative, default False.
  Enable it only when that constraint fits the data.
* ``half_precision`` uses float16 split/Bregman fields, default False. Image
  buffers and output remain float32. Compare with full precision when accuracy
  is important.
* ``gpu_id`` selects the input's CUDA device, default 0.

**Validation examples and images**

* :doc:`Pepper-image denoising <../../../validation/denoising/denoising_peppers>`
  compares Split Bregman TV with the other denoisers using saved images, PSNR and SSIM.
* :doc:`3D Shepp–Logan denoising <../../../validation/denoising/denoising_shepp_logan_3d>`
  shows central and orthogonal slices, with volume-wide PSNR and 3D SSIM.

The notebook parameters belong to their stated intensity scale and noise level;
they are starting points rather than universal defaults.

For regularisation during reconstruction, select ``regularisation_type="SB_TV"``
in FISTA, ADMM or OSEM. See :ref:`iterative_regularisation` for strength scaling
and the corresponding ``regularisation_options`` fields. Post-reconstruction
denoising and regularisation inside reconstruction solve different problems.

Papers and further reading
--------------------------

* `The split Bregman method for L1-regularized problems <https://doi.org/10.1137/080725891>`_.

API reference
-------------

:func:`httomolibgpu.misc.denoise.total_variation_SB` — signature, input/output types and full parameter details.
