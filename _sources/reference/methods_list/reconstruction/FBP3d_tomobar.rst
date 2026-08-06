.. _method_FBP3d_tomobar:

FBP3D (ToMoBAR)
^^^^^^^^^^^^^^^

**Description**

Filtered Back-Projection (FBP) method is implemented in `ToMoBAR <https://dkazanc.github.io/ToMoBAR>`_ software :cite:`kazantsev2020tomographic` and exposed in HTTomolibGPU. The method
consists of two parts, filtration of the projection data in Fourier space using a custom built SINC filter and then back-projecting the filtered data using `ASTRA-Toolbox <https://astra-toolbox.com>`_. The
filtering part is implemented using CuPy API and back-projection is a ray tracing operation on the GPU using ASTRA :cite:`van2016fast`. Notably the filtered data is passed to the back-projection routine directly, i.e.,
bypassing the device-host transfer, which makes this method more efficient on the GPUs, compared to :ref:`method_FBP2d_astra`.

**Where and how to use it:**

Together with :ref:`method_LPRec3d_tomobar`, this can be a first choice reconstruction method in the library. It is also fast and in well-sampled data situations delivers a similar reconstruction quality as the Log-Polar
reconstruction method. It is also recommended to use the FBP method when the data is not well and/or unevenly sampled. Note that for that type of data it might be better to to use iterative algorithms instead, such as :ref:`method_CGLS3d_tomobar` or more advanced :ref:`method_FISTA3d_tomobar`.


**What are the adjustable parameters:**

Most of parameters are self-explanatory from the method's API :mod:`httomolibgpu.recon.algorithm.FBP3d_tomobar`, so we will mention only the ones that potentially need more explanation.

* :code:`detector_pad` This parameter is responsible for padding of both (left/right) sides of the horizontal detector of each radiograph. This type of padding extend the edge assuming the sample outside the field of view is approximately similar. This should be used in cases when the sample is larger than the field of view or when the halo-type artifacts should be minimised. Set it to :code:`True` to enable automatic padding to avoid artifacts (middle image bellow).

* :code:`recon_mask_radius` This applies the circular mask to the reconstructed volume by zeroing all data outside a certain radius (right image bellow).

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/fbp3d_recon_no_pad.jpg
           :width: 200px

           FBP3d reconstruction with :code:`detector_pad=0`. Note the artifacts in the corners of the image.

      - .. figure:: ../../../_static/figures/reconstructions/fbp3d_recon_pad.jpg
           :width: 200px

           Detector padding enabled :code:`detector_pad=True`. The artifacts were removed.

      - .. figure:: ../../../_static/figures/reconstructions/fbp3d_recon_pad_mask.jpg
           :width: 200px

           Applying circular masking :code:`recon_mask_radius=0.9`.


* :code:`filter_freq_cutoff` can behave differently for this method compared to :ref:`method_LPRec3d_tomobar`. Normally the lower values might give you noisier and sharper reconstruction, while the higher values can lead to a blur. Note that the change of the filter will lead to the change in the dynamic range of the reconstructed image (see colour bars in the figures bellow).

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/FBP3D_tomobar_filter03.png
           :width: 315px

           FBP reconstruction with :code:`filter_freq_cutoff = 0.3` (default value).

      - .. figure:: ../../../_static/figures/reconstructions/FBP3D_tomobar_filter20.png
           :width: 315px

           FBP reconstruction with :code:`filter_freq_cutoff = 2.0`. A slight loss of the resolution, yet reduced noise.


**Practical example:**
