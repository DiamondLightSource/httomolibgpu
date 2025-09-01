.. _method_FBP3d_tomobar:

FBP3D (ToMoBAR)
^^^^^^^^^^^^^^^

**Description**

Filtered Back-Projection (FBP) method is implemented in `ToMoBAR <https://dkazanc.github.io/ToMoBAR>`_ software :cite:`kazantsev2020tomographic` and exposed in HTTomolibGPU. The method
consists of two parts, filtration of the projection data in Fourier space using a custom built SINC filter and then back-projecting the filtered data using `ASTRA-Toolbox <https://astra-toolbox.com>`_. The 
filtering part is implemented using CuPy API and back-projection is a ray tracing operation on the GPU using ASTRA :cite:`van2016fast`. Notably the filtered data is passed to the back-projection routine directly, i.e., 
bypassing the device-host transfer, which makes this method more efficient on the GPUs, compared to :ref:`method_FBP2d_astra`. 
  
**Where and how to use it:**

Together with :ref:`method_LPRec3d_tomobar`, this is a first choice method in the HTTomolibGPU library. It is also fast and in well-sampled data situations delivers a similar reconstruction quality as the Log-Polar
reconstruction method. It is also recommended to use the FBP method when the data is not well-sampled or unevenly sampled. But in that scenario, however, it is recommended to use iterative algorithms instead.


**What are the adjustable parameters:**

* On :code:`detector_pad`  and :code:`recon_mask_radius` parameters see  :ref:`method_LPRec3d_tomobar` as they produce the same behaviour. 

* :code:`filter_freq_cutoff` can behave differently for this method compared to :ref:`method_LPRec3d_tomobar`. Normally the lower values might give you noisier and sharper reconstruction, while the higher values can lead to a blur. Note that the change of the filter will lead to the change in the dynamic range of the reconstructed image (see colour bars in the figures bellow).

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/FBP3D_tomobar_filter03.png
           :width: 315px

           FBP reconstruction with :code:`filter_freq_cutoff = 0.3` (default value).

      - .. figure:: ../../../_static/figures/reconstructions/FBP3D_tomobar_filter20.png
           :width: 315px

           FBP reconstruction with :code:`filter_freq_cutoff = 2.0`. A slight loss of the resolution, yet reduced noise.


**Practical example:**
