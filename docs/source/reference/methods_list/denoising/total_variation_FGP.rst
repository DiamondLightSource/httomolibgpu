.. _method_total_variation_FGP:

Total Variation denoising (FGP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**

:func:`httomolibgpu.misc.denoise.total_variation_FGP` implements Fast Gradient
Projection :cite:`beck2009gradient`: an accelerated projected-gradient method
on the dual TV problem. It solves the same nonsmoothed TV model as
:ref:`method_total_variation_PD` and :ref:`method_total_variation_SB` using a
different sequence of updates.

For noisy data :math:`f`, output :math:`u` and strength :math:`\lambda`,

.. math::

   \min_{u\in C}\;\frac12\|u-f\|_2^2+
       \lambda\sum_p\|(Du)_p\|_2.

The forward difference operator :math:`D` uses zero boundary flux. The set
:math:`C` is unconstrained, or nonnegative when requested. The dual gradient
field is projected onto a unit ball at each voxel for isotropic TV. For
anisotropic TV, the components are bounded separately and the penalty becomes
:math:`\sum_{p,j}|D_j u_p|`. Extrapolation of the dual field accelerates the
iterations. The final output is a primal image computed from the projected
dual field, rather than an extrapolated image.

**Where and how to use it:**

Use it for edge-preserving denoising of images or volumes dominated by constant
regions. Compare quality and cost at appropriate iteration counts rather than
assuming equal counts make FGP, PD and Split Bregman equivalent. TV can suppress
small features and introduce staircasing; compare with
:ref:`method_total_generalised_variation` for smooth structures.

Input is a finite float32 CuPy volume in ``(Z, Y, X)`` order on ``gpu_id``.
Represent a 2D image as ``image[None, :, :]``. The function returns a new
C-contiguous float32 array with the same shape; the input is not modified.

.. code-block:: python

   from httomolibgpu import total_variation_FGP

   denoised = total_variation_FGP(
       volume,
       regularisation_parameter=0.1,
       iterations=300,
       isotropic=True,
       nonnegativity=True,
   )

**What are the adjustable parameters:**

* ``regularisation_parameter`` controls TV strength, default 0.001. Larger
  values favour smoother images. Zero disables denoising, retaining any
  requested nonnegativity projection.
* ``iterations`` sets the fixed number of accelerated dual updates, default
  150. The implementation chooses its internal step size; there is no public
  time-step parameter.
* ``isotropic`` selects isotropic TV (True, default) or anisotropic TV (False).
* ``nonnegativity`` projects primal evaluations onto nonnegative values,
  default False.
* ``half_precision`` stores dual work fields in float16, default False.
  The primal image and returned array remain float32.
* ``gpu_id`` selects the input's CUDA device, default 0.

**Validation examples and images**

* :doc:`Pepper-image denoising <../../../validation/denoising/denoising_peppers>`
  compares FGP-TV with the other denoisers using saved images, PSNR and SSIM.
* :doc:`3D Shepp–Logan denoising <../../../validation/denoising/denoising_shepp_logan_3d>`
  shows central and orthogonal slices, with volume-wide PSNR and 3D SSIM.

The notebook parameters belong to their stated intensity scale and noise level;
they are starting points rather than universal defaults.

For regularisation during reconstruction, select ``regularisation_type="FGP_TV"``
in FISTA, ADMM or OSEM. See :ref:`iterative_regularisation` for strength scaling
and the corresponding ``regularisation_options`` fields. Post-reconstruction
denoising and regularisation inside reconstruction solve different problems.

Papers and further reading
--------------------------

* `Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems <https://doi.org/10.1109/TIP.2009.2028250>`_.

API reference
-------------

:func:`httomolibgpu.misc.denoise.total_variation_FGP` — signature, input/output types and full parameter details.
