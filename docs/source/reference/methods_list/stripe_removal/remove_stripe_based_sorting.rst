.. _method_remove_stripe_based_sorting:

Remove stripes by sorting
^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**

This filter has been developed by Dr. Nghia Vo and it works in the sinogram space to minimise stripes leading to `Ring Artifacts <https://radiopaedia.org/articles/ring-artifact-2?lang=gb>`_ in the reconstruction. See more about the filter in the original publication :cite:`vo2018superior`.
Also more information about the method can be found on the author's software `Sarepy <https://sarepy.readthedocs.io/>`_.

**Where and how to use it:**

Stripe removal method is used for data pre-processing, i.e., before the reconstruction to remove the stripes present in the data. If the stripes are not removed,
this will lead to ring artefacts in the reconstruction and further problems with quantification and segmentation.

**What are the adjustable parameters:**

The important parameter here is :code:`size` which is responsible for the size of the median filter where sorting is performed.

**Practical example:**

In this example we demonstrate how to apply the stripe removal algorithm to the normalised projection data.

.. warning:: As with any pre-processing tool, one needs to be careful in applying stripe removal algorithm to data. Sub-optimal parameters can potentially lead to other artifacts introduced in the reconstruction.
.. list-table::


    * - .. figure:: ../../../_static/auto_images_methods/data_stripes_added_sino.png

           Input (sinogram view) with the stripes added

      - .. figure:: ../../../_static/auto_images_methods/data_stripes_added_proj.png

           Input (projection view) with the stripes added

    * - .. figure:: ../../../_static/auto_images_methods/remove_stripe_based_sorting_sino.png

           After applying remove stripes by sorting (sinogram)

      - .. figure:: ../../../_static/auto_images_methods/remove_stripe_based_sorting_proj.png

           After applying remove stripes by sorting (projection)

    * - .. figure:: ../../../_static/auto_images_methods/remove_stripe_based_sorting_res_sino.png

           Sinogram view of absolute residual between input and output

      - .. figure:: ../../../_static/auto_images_methods/remove_stripe_based_sorting_res_proj.png

           Projection view of absolute residual between input and output


Reconstruction effect
---------------------

The realistic model 7 acquisition includes detector-response variations and
additional stripes from the bundled artefact settings. Raw zingers are removed
before normalisation and stripe filtering. See :doc:`../../../scripts/README`.

.. list-table::

   * - .. figure:: ../../../_static/auto_images_methods/data_stripes_added_recon.png
          :alt: FBP before stripe removal, showing ring artefacts.

          FBP before stripe removal.

     - .. figure:: ../../../_static/auto_images_methods/remove_stripe_based_sorting_recon.png
          :alt: FBP after applying remove_stripe_based_sorting to the sinograms.

          FBP after ``remove_stripe_based_sorting``, using the same reconstruction settings and
          display limits. This example illustrates behaviour, not a claim of
          optimal parameters for every dataset.

Papers and further reading
--------------------------

* `Superior techniques for eliminating ring artifacts in X-ray micro-tomography <https://doi.org/10.1364/OE.26.028396>`_.

Validation
----------

:doc:`I12 parameter comparisons <../../../validation/preprocessing/stripe_removal>` retains the original
Zenodo regression parameters and numerical checks.

API reference
-------------

:func:`httomolibgpu.prep.stripe.remove_stripe_based_sorting` — signature, input/output types and full parameter details.
