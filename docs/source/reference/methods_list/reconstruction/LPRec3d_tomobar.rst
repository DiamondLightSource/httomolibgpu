.. _method_LPRec3d_tomobar:

Log-Polar 3D (ToMoBAR)
^^^^^^^^^^^^^^^^^^^^^^

**Description**

The Log-Polar method is especially useful for implementing filtered back-projection (FBP) and Fourier-based reconstruction techniques via Fourier Slice Theorem aka `Projection-Slice Theorem <https://en.wikipedia.org/wiki/Projection-slice_theorem>`_.
It uses Fast Fourier Transform (FFT) computations to accelerate reconstruction. Instead of working in standard Cartesian coordinates, the projection data (Radon transform) is mapped into log-polar coordinates, where scaling and rotation become
simpler transformations, see more :cite:`andersson2016fast`. The method is implemented in `ToMoBAR <https://dkazanc.github.io/ToMoBAR>`_ software :cite:`kazantsev2020tomographic` using CuPy API and exposed in HTTomolibGPU. It is also optimised to work with
the 3D projection data.

**Where and how to use it:**

Log-polar reconstruction is a direct option for well-sampled, evenly spaced
parallel-beam data. Its coordinate transforms require GPU workspace; compare
its memory use and measured runtime with :ref:`method_FBP3d_tomobar` for your
dataset. There is no universal speed ranking across hardware and shapes.
For a recorded timing example, see :doc:`../../../benchmarks/index`.
Limited-angle or uneven sampling may require a different method and additional
validation of interpolation artefacts.

**What are the adjustable parameters:**

Most of parameters are self-explanatory from the method's API :func:`httomolibgpu.recon.algorithm.LPRec3d_tomobar`, so we will mention only the ones that potentially need more explanation.

* :code:`detector_pad` This parameter is responsible for padding of both (left/right) sides of the horizontal detector of each radiograph. This type of padding extends the edge assuming the sample outside the field of view is approximately similar. This should be used in cases when the sample is larger than the field of view or when the halo-type artifacts should be minimised. Set it to :code:`True` to enable automatic padding to avoid artifacts (middle image below).

* :code:`recon_mask_radius` This applies the circular mask to the reconstructed volume by zeroing all data outside a certain radius (right image below).

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_no_pad.jpg
           :width: 200px

           Log-Polar reconstruction with :code:`detector_pad=0`. Note the artifacts in the corners of the image.

      - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_pad.jpg
           :width: 200px

           Detector padding enabled :code:`detector_pad=True`. The artifacts were removed.

      - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_pad_mask.jpg
           :width: 200px

           Applying circular masking :code:`recon_mask_radius=0.9`.


* :code:`filter_type` and :code:`filter_freq_cutoff` control the type of filter and the cut-off frequency applied. Different results therefore can be achieved, for instance, when noise is either accentuated or suppressed.

.. list-table::


    * - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_filter_hann.png
           :width: 335px

           Reconstruction using :code:`filter_type = 'hann'` and :code:`filter_freq_cutoff=1.0`.

      - .. figure:: ../../../_static/figures/reconstructions/lprec_recon_filter_parzen.png
           :width: 335px

           Reconstruction using :code:`filter_type = 'parzen'` and :code:`filter_freq_cutoff=1.0`.

Papers and further reading
--------------------------

* `Fast algorithms and efficient GPU implementations for the Radon transform and the back-projection operator represented as convolution operators <https://doi.org/10.1137/15M1023762>`_.

Validation examples and images
------------------------------

See the saved reconstruction images, processing settings and dataset details
in the following validation notebooks:

* :doc:`LPRec3D · I12 dataset 1 <../../../validation/reconstruction/direct/lprec3d_i12_dataset1>`
* :doc:`LPRec3D · I13 dataset 1 <../../../validation/reconstruction/direct/lprec3d_i13_dataset1>`
* :doc:`LPRec3D · I13 dataset 2 <../../../validation/reconstruction/direct/lprec3d_i13_dataset2>`
* :doc:`LPRec3D · K11 dataset 2 <../../../validation/reconstruction/direct/lprec3d_k11_dataset2>`


API reference
-------------

:func:`httomolibgpu.recon.algorithm.LPRec3d_tomobar` — signature, input/output types and full parameter details.
