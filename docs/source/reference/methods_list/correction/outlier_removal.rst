.. _method_outlier_removal:

Outliers removal filter
^^^^^^^^^^^^^^^^^^^^^^^

**Description**

An outlier removal filter is also called a **dezinger**. It replaces pixels or
voxels whose absolute difference from the local median reaches or exceeds the
:code:`dif` threshold, using the :ref:`method_median_filter`. Both bright
and dark outliers can be replaced. Essentially, a dezinger is a spatially
variant median filter.
Mathematically, one can express the dezinger as:

.. math::

   \widehat f_i=\operatorname{median}\{f_j:j\in\mathcal{N}_i\},
   \qquad
   g_i=\begin{cases}
      \widehat f_i,& |f_i-\widehat f_i|\geq\sigma,\\
      f_i,& \text{otherwise}.
   \end{cases}


Where :math:`\sigma` is the threshold parameter and :math:`\mathrm{N}_{i}` is the symmetric neighbourhood (e.g. 3 x 3) of the pixel :math:`i`.

.. note:: Increasing the threshold value leads to the filter being more selective and decreasing the value makes it more like the normal median filter.

**Where and how to use it:**

Outliers removal filter should be used in situations when the data (projections or sinograms) contains pixels/voxels that do not belong to the surrounding area.
Usually, it looks like a `salt-and-pepper <https://en.wikipedia.org/wiki/Salt-and-pepper_noise>`_ noise or speckle-type noise, see Practical example below. In tomography, such
noise can be present because of the scattered X-rays and the quicker way to establish the need for the filter is to investigate the raw data visually. When removing
the outliers, it is preferable to avoid any modification of the data, therefore some experimentation with the :math:`\sigma` parameter might needed.

**What are the adjustable parameters:**

* :code:`kernel_size` defines the neighborhood size, see :math:`\mathrm{N}_{i}` in the equations above. For instance, for 2D image :code:`kernel_size = 3` defines the neighborhood of 3 by 3 pixels (one needs to use only odd numbers). Increasing :code:`kernel_size` results in more smoothing applied, it is recommended to use the higher values when the clusters of pixels are large. As a rule of thumb, keeping :code:`kernel_size` within 7-11 range is a good trade-off. The limit is set to 13 currently.

* :code:`dif` defines the threshold above which the value in the pixel is considered to be too high/low. Some empirical testing is usually required to establish the best trade-off.


**Practical example:**

The input is TomoPhantom model 7 with realistic flats, dark current, photon/read
noise and random zingers. The raw-count filter uses ``kernel_size=5`` and
``dif=2000``. See :doc:`../../../scripts/README` for generation settings.

.. list-table::


    * - .. figure:: ../../../_static/auto_images_methods/remove_outlier_input_sino.png

           Realistic synthetic detector data with randomly distributed zingers (sinogram view).

      - .. figure:: ../../../_static/auto_images_methods/remove_outlier_input_proj.png

           The same noisy detector data with zingers (projection view).

    * - .. figure:: ../../../_static/auto_images_methods/remove_outlier_sino.png

           After applying outliers removal (sinogram)

      - .. figure:: ../../../_static/auto_images_methods/remove_outlier_proj.png

           After applying outliers removal (projection)

    * - .. figure:: ../../../_static/auto_images_methods/remove_outlier_res_sino.png

           Sinogram view of absolute residual between input and output. Replaced pixels are visible; changes to real structures indicate an overly aggressive threshold.

      - .. figure:: ../../../_static/auto_images_methods/remove_outlier_res_proj.png

           Projection view of absolute residual between input and output. Replaced pixels are visible; changes to real structures indicate an overly aggressive threshold.


Reconstruction effect
---------------------

.. list-table::

   * - .. figure:: ../../../_static/auto_images_methods/raw_data_recon.png
          :alt: Reconstruction with streaks caused by zingers.

          FBP of normalised data before outlier removal.

     - .. figure:: ../../../_static/auto_images_methods/data_stripes_added_recon.png
          :alt: Reconstruction after raw-count outlier removal, with remaining rings and noise.

          FBP after removing raw-count outliers. Stripe artefacts and detector
          noise remain. Both images use the ideal reconstruction's display range.

API reference
-------------

:func:`httomolibgpu.misc.corr.remove_outlier` — signature, input/output types and full parameter details.
