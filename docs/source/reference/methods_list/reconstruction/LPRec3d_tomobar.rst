.. _method_LPRec3d_tomobar:

Log-Polar 3D (ToMoBAR)
^^^^^^^^^^^^^^^^^^^^^^

**Description**

The Log-Polar method is especially useful for implementing filtered back-projection (FBP) and Fourier-based reconstruction techniques via Fourier Slice Theorem aka `Projection-Slice Throrem <https://en.wikipedia.org/wiki/Projection-slice_theorem>`_.
It is normally the fastest reconstruction method, as it allows Fast Fourier Transform (FFT) computations. Instead of working in standard Cartesian coordinates, the projection data (Radon transform) is mapped into log-polar coordinates, where scaling and rotation become 
simpler transformations, see more :cite:`andersson2016fast`. The method is implemented in `ToMoBAR <https://dkazanc.github.io/ToMoBAR>`_ software :cite:`kazantsev2020tomographic` using CuPy API and exposed in HTTomolibGPU. It also optimised to work with 
the 3D projection data, hence the name of the method.  
  
**Where and how to use it:**

It is the fastest direct method in the library compared to :ref:`method_FBP3d_tomobar` and :ref:`method_FBP2d_astra`. The Log-Polar accuracy is comparable to FBP if the data is well-sampled and evenly spaced. However, for poor/uneven-sampled
projection data it is generally to recommend to use FBP methods to avoid interpolation errors during coordinate transformation. Arguably, the reconstruction errors (artifacts) can be minor and generally not visible.
The Log-Polar can be used as a first choice method for fast reconstruction followed by :ref:`method_FBP3d_tomobar`.

**What are the adjustable parameters:**

Most of parameters are self-explanatory from the method's API :mod:`httomolibgpu.recon.algorithm.LPRec3d_tomobar`, so we will mention only the ones that need more explanation.

* :code:`detector_pad` This parameter is responsible for padding of both (left/right) sides of the horizontal detector of each radiograph. This type of padding extend the edge assuming the sample outside the field of view is approximately similar. This should be used in cases when the sample is larger than the field of view or when the halo-type artifacts should be minimised.

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_no_pad.png
           :width: 300px

           Reconstruction using Log-Polar method without detector padding :code:`detector_pad=0`.

      - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_pad.png
           :width: 300px

           Reconstruction using detector padding :code:`detector_pad=300`. See that the halo artifacts were removed.


* :code:`recon_mask_radius` This applies the circular mask to the reconstructed volume by zeroing all data outside a certain radius. 

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_no_pad.png
           :width: 300px

           Reconstruction with the mask  :code:`recon_mask_radius=2.0`.

      - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_mask.png
           :width: 300px

           Reconstruction with the mask  :code:`recon_mask_radius=0.78`.


* :code:`filter_type` and :code:`filter_freq_cutoff` control the type of filter and the cut-off frequency applied. Different results therefore can be achieved, for instance, when noise is either accentuated or suppressed.  

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_filter_hann.png
           :width: 300px

           Reconstruction using :code:`filter_type = 'hann'` and :code:`filter_freq_cutoff=1.0`.

      - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_filter_parzen.png
           :width: 300px

           Reconstruction using :code:`filter_type = 'parzen'` and :code:`filter_freq_cutoff=1.5`.