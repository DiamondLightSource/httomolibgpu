.. _method_rescale_to_int:

Rescale to integers
^^^^^^^^^^^^^^^^^^^

**Description**

This method is used to rescale the data before it gets saved into the images. The method allows you to rescale the data into 8, 16,
or 32 bit of unsigned integer as well as to use the percentage scaling (explained below) to enhance contrast and remove outliers in the resulting images.

**Where and how to use it:**

The main purpose of this module is to help saving the data into images with rescaling. If the bit depth is reduced in the data, it can help to reduce the size of the saved images and in some situations simplify and accelerate of data analysis.

.. warning::  It is worth to note, however, that it is a lossy conversion when the bit depth of data is reduced. This can significantly alter the visual perception of the data as well as the quantitative side.

The rescaling module allows you to change the bit depth of a grayscale image, e.g., from 16-bit to 8-bit and also rescale the data in advance to avoid
clipping of the data and therefore the loss of the information.

.. note:: The method does not save the data into images automatically, but only rescale the data. The user needs to take care of saving the data using an image saving method, e.g., the `image saver  <https://diamondlightsource.github.io/httomolib/api/httomolib.misc.images.html>`_ of the HTTomolib library.

**What are the adjustable parameters:**

* :code:`bits` defines the number of bits in the resulting data. The input can be any data type and will be rescaled into unsigned integer of 8, 16 or 32 bit type.

* :code:`perc_range_min` defines the lower cutoff point in the input data, in percent of the data range (defaults to 0). The lower bound is computed as :math:`\frac{\mathrm{perc\_range\_min} * (\max-\min)}{100} + \min`. Note that :math:`\max` and :math:`\min` values will be automatically estimated from the input data, unless they are provided with the :code:`glob_stats` optional parameter.

* :code:`perc_range_max` defines the upper cutoff point in the input data, in percent of the data range (defaults to 100). The higher bound is computed as :math:`\frac{\mathrm{perc\_range\_max} * (\max-\min)}{100} + \min`.

**Practical example:**

The example uses realistic model 7 data after outlier removal and logarithmic
normalisation. A global 0.5--99.5 percentile display window removes rare extreme
values before the illustrated range cutoffs are applied. See
:doc:`../../../scripts/README` for reproducible generation settings.

.. list-table::


    * - .. figure:: ../../../_static/auto_images_methods/rescale_to_int_proj_0_to_100.png

           Projection data saved into 8-bit image with :code:`perc_range_min = 0` and :code:`perc_range_max = 100` scaling.

      - .. figure:: ../../../_static/auto_images_methods/rescale_to_int_histo_0_to_100.png

           Histogram of the 8-bit projection using the full prepared display window.

    * - .. figure:: ../../../_static/auto_images_methods/rescale_to_int_proj_10_to_90.png

           Projection data saved into 8-bit image with :code:`perc_range_min = 10` and :code:`perc_range_max = 90` scaling. Note that the contrast appears to be better with this scaling.
      - .. figure:: ../../../_static/auto_images_methods/rescale_to_int_histo_10_to_90.png

           Histogram after narrowing the input range. Values outside the selected range are clipped to the endpoints.

    * - .. figure:: ../../../_static/auto_images_methods/rescale_to_int_proj_30_to_70.png

           Projection data saved into 8-bit image with :code:`perc_range_min = 30` and :code:`perc_range_max = 70` scaling. This is an example of a poorer scaling when the loss of the information is clearly visible through poorer contrast.
      - .. figure:: ../../../_static/auto_images_methods/rescale_to_int_histo_30_to_70.png

           Histogram after a stronger cutoff. Endpoint accumulation shows where clipping has discarded intensity differences.

API reference
-------------

:func:`httomolibgpu.misc.rescale.rescale_to_int` — signature, input/output types and full parameter details.
