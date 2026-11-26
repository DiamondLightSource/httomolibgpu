.. _method_total_generalised_variation:

Total Generalised Variation denoising (TGV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**

:func:`httomolibgpu.misc.denoise.total_generalised_variation` implements
second-order isotropic Total Generalised Variation :cite:`bredies2010total`.
It combines first- and second-order spatial information, allowing smooth
intensity ramps as well as discontinuities. This can reduce the staircasing
associated with first-order TV.

For noisy data :math:`f`, output :math:`u` and auxiliary vector field :math:`v`,
the discrete model is

.. math::

   \min_{u\in C,v}\;\frac12\|u-f\|_2^2+
   \lambda\left(\alpha_1\|Du-v\|_{2,1}
                  +\alpha_0\|Ev\|_{F,1}\right),
   \qquad Ev=\frac12\left(Dv+(Dv)^\mathsf{T}\right).

The mixed norms sum voxel-wise Euclidean or Frobenius norms.
:math:`D` uses forward differences with zero boundary flux; :math:`E` is the
symmetrised gradient. The first term couples :math:`v` to the image gradient;
the second penalises variation in :math:`v`. Thus a smooth ramp need not pay
the same first-order penalty as it would under TV. :math:`C` optionally
constrains the image to nonnegative values, without constraining :math:`v`.
The GPU implementation uses fixed-count primal-dual iterations.

**Where and how to use it:**

Use it when preserving smooth gradients matters, for example in reconstructed
materials with varying attenuation. Compare with :ref:`method_total_variation_PD`
for piecewise-constant objects. TGV may oversmooth small structures if its
strength or higher-order weight is too large; inspect edges as well as metrics.

Input is a finite float32 CuPy volume in ``(Z, Y, X)`` order on ``gpu_id``.
Represent a 2D image as ``image[None, :, :]``. The function returns a new
C-contiguous float32 array with the same shape; the input is not modified.

.. code-block:: python

   from httomolibgpu import total_generalised_variation

   denoised = total_generalised_variation(
       volume,
       regularisation_parameter=0.1,
       alpha1=1.0,
       alpha0=1.65,
       iterations=500,
       nonnegativity=True,
   )

**What are the adjustable parameters:**

* ``regularisation_parameter`` is the overall strength, default 0.001.
  It scales both TGV terms; zero disables denoising, retaining any requested
  nonnegativity projection.
* ``alpha1`` weights the first-order coupling term, default 1.0.
* ``alpha0`` weights the symmetrised-gradient term, default 2.0. Adjust its
  ratio to ``alpha1`` to balance the terms. Both weights must be positive;
  multiplying both by a common factor also changes the effective strength.
* ``iterations`` sets the fixed inner iteration count, default 150.
* ``lipschitz_const`` is a positive squared operator-norm bound, default 12.0.
  ToMoBAR raises it to at least ``8 * spatial_ndim + 2`` for stability.
  This is not the tomographic projection operator's norm.
* ``nonnegativity`` constrains the image, default False.
* ``half_precision`` uses float16 normalised dual fields, default False.
  Primal and auxiliary-vector fields, and the output, remain float32.
* ``gpu_id`` selects the input's CUDA device, default 0.

**Validation examples and images**

* :doc:`Pepper-image denoising <../../../validation/denoising/denoising_peppers>`
  compares TGV with the other denoisers using saved images, PSNR and SSIM.
* :doc:`3D Shepp–Logan denoising <../../../validation/denoising/denoising_shepp_logan_3d>`
  shows central and orthogonal slices, with volume-wide PSNR and 3D SSIM.

The notebook parameters belong to their stated intensity scale and noise level;
they are starting points rather than universal defaults.

For regularisation during reconstruction, select ``regularisation_type="TGV"``
in FISTA, ADMM or OSEM. See :ref:`iterative_regularisation` for strength scaling
and the corresponding ``regularisation_options`` fields. Post-reconstruction
denoising and regularisation inside reconstruction solve different problems.

Papers and further reading
--------------------------

* `Total generalized variation <https://doi.org/10.1137/090769521>`_.

API reference
-------------

:func:`httomolibgpu.misc.denoise.total_generalised_variation` — signature, input/output types and full parameter details.
