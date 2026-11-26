.. _method_total_variation_ROF:

Total Variation denoising (ROF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**

`Total variation (TV) denoising <https://en.wikipedia.org/wiki/Total_variation_denoising>`_ using Rudin-Osher-Fatemi (ROF) model :cite:`rudin1992nonlinear` is a seminal work that proposed edge-preserving image smoothing in 1992.
As opposed to the Gaussian smoothing, where the edges of features are not preserved, this has revolutionised the field of image recovery. The TV
denoising has become a widely-accepted way to denoise images/volumes, while preserving important features in them.
The resulting denoised images are piecewise-constant or cartoon-like.

.. list-table::


    * - .. figure:: ../../../_static/figures/FBP.png

           The noisy image

      - .. figure:: ../../../_static/figures/ROF.png

           Recovery using ROF-TV denoising

      - .. figure:: ../../../_static/figures/ROF_zoom.png

           Zoomed to ROF-TV denoised image

For noisy input :math:`f`, the smoothed ROF model balances fidelity and a
regularised gradient penalty:

.. math::

   \min_u\;\frac{1}{2}\lVert u-f\rVert_2^2
   +\lambda\sum_p\sqrt{\lVert(Du)_p\rVert_2^2+\varepsilon}.

Here :math:`D` is the spatial finite-difference operator, :math:`\lambda`
controls smoothing, and a small positive :math:`\varepsilon` prevents a
singularity at a zero gradient. This is a smoothed TV model rather than an
exact nonsmooth TV proximal solve. See :cite:`rudin1992nonlinear` and
:cite:`kazantsev2019ccpi` for the model and software background.

**Where and how to use it:**

Use it when the noise in the reconstructed image/volume needs to be removed. The TV-denoising should work better than :ref:`method_median_filter` and introduce less undesirable smoothing and features distortion/disappearance. Also the contrast improvement is expected.

.. note:: Applying denoising after the reconstruction is different to employ that kind of smoothing as a regularisation within an iterative method. Use the latter when the projection data is undersampled and of poor quality (many artefacts, distortions, etc.). In that case, just denoising, might be not very effective and iterative methods like :ref:`method_FISTA3d_tomobar` should be used.

**What are the adjustable parameters:**

* :code:`regularisation_parameter` Is the most important parameter as it controls the level of smoothing of the image. Larger values lead to more smoothing, which can lead to undesirable oversmoothing.

* :code:`iterations` Algorithm iterations. You will need a significant number of iterations to run in order to see the change in the image. It is recommended to keep this number between 1000 and 5000 iterations.

* :code:`time_marching_parameter` This parameter ensures the stability of iterations and needs to be small. If you notice unusual artefacts in the denoised image, reduce the value of the parameter. We suggest to keep it 0.001 or smaller.


**Validation examples and images**

See the :doc:`pepper-image comparison <../../../validation/denoising/denoising_peppers>`
and :doc:`3D Shepp–Logan comparison <../../../validation/denoising/denoising_shepp_logan_3d>`
for saved denoised images, parameter settings, PSNR and SSIM.

Papers and further reading
--------------------------

* `Nonlinear total variation based noise removal algorithms <https://doi.org/10.1016/0167-2789(92)90242-F>`_.
* `CCPi-Regularisation toolkit for computed tomographic image reconstruction with proximal splitting algorithms <https://doi.org/10.1016/j.softx.2019.04.003>`_.

API reference
-------------

:func:`httomolibgpu.misc.denoise.total_variation_ROF` — signature, input/output types and full parameter details.
