.. _method_nonlinear_diffusion:

Nonlinear diffusion denoising (NDF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**

:func:`httomolibgpu.misc.denoise.nonlinear_diffusion` smooths local intensity
differences using edge-stopping fluxes, following the anisotropic-diffusion
approach of Perona and Malik :cite:`perona1990scale`. ToMoBAR supports five
penalties and retains a fidelity force towards the original noisy input.

For input :math:`f`, current image :math:`u^k`, strength :math:`\lambda` and
effective time step :math:`\tau`, the implemented update is

.. math::

   u_p^{k+1}=\Pi_C\!\left[
       u_p^k+\tau\lambda\sum_{q\in\mathcal{N}(p)}
          \phi_T(u_q^k-u_p^k)-\tau(u_p^k-f_p)
   \right],\qquad u^0=f,\quad T=\frac{\sigma}{\sqrt{2}}.

:math:`\mathcal{N}(p)` contains axis-adjacent neighbours: four in 2D or six
in 3D, omitting neighbours outside the image. The fidelity term uses the fixed
original input :math:`f`, not the previous iterate. :math:`\Pi_C` is the identity
by default, or clips negative values to zero when requested. Fluxes act on
individual directional differences, not on the full gradient magnitude.
For :math:`T>0`, the implementation uses:

.. list-table:: Edge-stopping fluxes, with intensity difference :math:`s`
   :header-rows: 1
   :widths: 8 25 67

   * - ``penalty_type``
     - Penalty
     - Flux :math:`\phi_T(s)`
   * - 1
     - Huber
     - :math:`H_T(s)=s/T` for :math:`|s|\leq T`, otherwise :math:`\operatorname{sign}(s)`
   * - 2
     - Perona–Malik
     - :math:`s/(1+(s/T)^2)`
   * - 3
     - Tukey biweight
     - :math:`s(1-(s/T)^2)^2` for :math:`|s|\leq T`, otherwise zero
   * - 4
     - Truncated quadratic
     - :math:`s` for :math:`|s|\leq T`, otherwise zero
   * - 5
     - Truncated Huber
     - :math:`H_T(s)` for :math:`|s|\leq 2T`, otherwise zero

``sigma=0`` selects linear flux :math:`\phi_0(s)=s` for every penalty.
Penalties 2–5 are nonconvex and may give different results with different
iteration counts or thresholds; a unique global solution is not guaranteed.

**Where and how to use it:**

Use it when control over edge-dependent smoothing is useful. A threshold that
is too low may classify noise as edges and leave isolated noisy features.
A threshold that is too high can smooth away real boundaries. The effects of
strength, threshold, step size and early stopping interact, so compare them
jointly using a clean reference when available and inspect the saved images.

Input is a finite float32 CuPy volume in ``(Z, Y, X)`` order on ``gpu_id``.
Represent a 2D image as ``image[None, :, :]``. The function returns a new
C-contiguous float32 array with the same shape; the input is not modified.

.. code-block:: python

   from httomolibgpu import nonlinear_diffusion

   # Huber settings from the pepper example, with Gaussian noise sigma=0.15.
   denoised = nonlinear_diffusion(
       volume,
       regularisation_parameter=0.24,
       sigma=0.018,
       iterations=64,
       time_marching_parameter=0.1,
       penalty_type=1,
       nonnegativity=True,
   )

**What are the adjustable parameters:**

* ``regularisation_parameter`` scales diffusion relative to data fidelity,
  default 0.001. Equal strengths do not give equal smoothing across penalties,
  because their flux functions have different normalisations.
* ``sigma`` is the nonnegative threshold parameter in input intensity units,
  default 0.01; the effective threshold is ``sigma / sqrt(2)``. This is not
  automatically an estimate of the noise standard deviation.
* ``penalty_type`` selects the flux above, default 1 (Huber).
* ``iterations`` is the fixed update count, default 150. Early stopping can
  preserve structures better than a long run. More iterations do not
  necessarily improve PSNR or SSIM.
* ``time_marching_parameter`` is the requested positive explicit time step,
  default 0.005. ToMoBAR caps it internally according to dimensionality,
  strength and threshold; the applied step can be smaller than requested.
* ``nonnegativity`` enables nonnegative image updates, default False.
* ``half_precision`` must be False: NDF has no auxiliary fields to store
  in float16. Computation and output use float32.
* ``gpu_id`` selects the input's CUDA device, default 0.

**Validation examples and images**

* :doc:`Pepper-image denoising <../../../validation/denoising/denoising_peppers>`
  compares all five diffusion penalties with the other denoisers using saved images, PSNR and SSIM.
* :doc:`3D Shepp–Logan denoising <../../../validation/denoising/denoising_shepp_logan_3d>`
  shows central and orthogonal slices, with volume-wide PSNR and 3D SSIM.

The notebook parameters belong to their stated intensity scale and noise level;
they are starting points rather than universal defaults.

For regularisation during reconstruction, select ``regularisation_type="NDF"``
in FISTA, ADMM or OSEM. See :ref:`iterative_regularisation` for strength scaling
and the corresponding ``regularisation_options`` fields. Post-reconstruction
denoising and regularisation inside reconstruction solve different problems.

Papers and further reading
--------------------------

* `Scale-space and edge detection using anisotropic diffusion <https://doi.org/10.1109/34.56205>`_.

API reference
-------------

:func:`httomolibgpu.misc.denoise.nonlinear_diffusion` — signature, input/output types and full parameter details.
