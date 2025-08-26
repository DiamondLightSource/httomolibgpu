.. _reference_methods:

Methods Description
-------------------

Here we present a list of methods that exist in the HTTomolibGPU library with more detailed information in places compared to :ref:`reference_api`.

.. _data_correction_module:

Data correction module
^^^^^^^^^^^^^^^^^^^^^^

Modules from `Data correction` can be used as pre-processing (e.g. apply :ref:`method_outlier_removal` to raw projection data) or in some
cases as post-processing (e.g. apply :ref:`method_median_filter` to the result of the reconstruction) tools.

.. toctree::
   :glob:

   methods_list/median_filter
   methods_list/outlier_removal


.. _data_denoising_module:

Data denoising module
^^^^^^^^^^^^^^^^^^^^^^

Modules from `Data denoising` can be used as post-processing tools. For instance, denoising procedures can be applied to the results of the reconstruction.

.. toctree::
   :glob:

   methods_list/total_variation_ROF
   methods_list/total_variation_PD

.. _stripes_removal_module:

Stripes removal module
^^^^^^^^^^^^^^^^^^^^^^

Modules from `Stripes removal` are normally used as a pre-processing tool. Stripes removal is equivalent to removing ring artefacts in the reconstructed images.


.. toctree::
   :glob:

   methods_list/remove_stripe_based_sorting
   methods_list/remove_all_stripe
   methods_list/remove_stripe_ti
   methods_list/raven_filter


.. _image_reconstruction_module:

Image reconstruction module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modules from `Image reconstruction` are needed in order to reconstruct data from the set of sinograms into a reconstructed volume. The reconstruction methods can be divided into two groups: Direct methods and Iterative methods. 
The former are faster and suitable for the majority of well-sampled and well-exposed data. The latter are more complex and slower methods when the data is erroneuos or/and undersampled. 

.. toctree::
   :glob:

   methods_list/LPRec3d_tomobar
   methods_list/FBP3d_tomobar
   methods_list/FBP2d_astra
   

.. _data_rescale_module:

Data rescale module
^^^^^^^^^^^^^^^^^^^

Modules from `Data rescale` is usually needed when the data needs to be rescaled to be saved in different bit-type as images.

.. toctree::
   :glob:

   methods_list/rescale_to_int

