.. _method_outlier_removal:

Outliers removal filter
^^^^^^^^^^^^^^^^^^^^^^^
Also called as **dezinger**. Outliers removal filter is a part of the module :doc:`../../api/httomolibgpu.misc.corr` for data correction. 
This method seeks the pixels that do not fit normally distributed data and isolate them by using the :ref:`method_median_filter`. The user 
needs to set the threshold (:code:`dif` parameter of the :code:`remove_outlier` method and :math:`\sigma` in equations bellow), based on which the filter will consider outliers that are **above** 
the chosen threshold.  Mathematically, one can express it as: 

.. math::

   f \in \mathrm{R}^{N}: \textit{input}
   
   g \in \mathrm{R}^{N}: \textit{output}

   \hat{f}_{i} = \textit{median}(f_{j}), \textrm{where} \ j \in \mathrm{N}_{i}

   g_{i} = \hat{f}_{i} \  \textrm{if} \ | f_{i} - \hat{f}_{i} | \geq \sigma.

.. note:: Increasing the threshold value leads to the filter being more selective and decreasing the value makes it more like the normal median filter. 
  

Practical example:

.. list-table:: 


    * - .. figure:: ../../_static/auto_images_methods/median_filter_input_sino.png

           Input (sinogram view)

      - .. figure:: ../../_static/auto_images_methods/median_filter_input_proj.png

           Input (projection view)

    * - .. figure:: ../../_static/auto_images_methods/median_filter_sino.png

           After applying outliers removal (sinogram)

      - .. figure:: ../../_static/auto_images_methods/median_filter_proj.png

           After applying outliers removal (projection)

    * - .. figure:: ../../_static/auto_images_methods/median_filter_res_sino.png

           Sinogram view of absolute residual between input and output

      - .. figure:: ../../_static/auto_images_methods/median_filter_res_proj.png

           Projection view of absolute residual between input and output
