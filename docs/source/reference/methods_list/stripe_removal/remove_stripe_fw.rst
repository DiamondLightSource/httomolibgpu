Fourier–wavelet stripe removal
==============================

``remove_stripe_fw``

Decompose sinograms with wavelets, damp stripe-related Fourier components in the detail bands, and reconstruct the corrected sinogram. Stripes persisting across angles otherwise become rings after reconstruction.

Usage and parameters
--------------------

sigma controls Fourier damping; wname selects the wavelet and level the decomposition depth. Tune on a representative sinogram and reconstruction together. Strong filtering may remove real structures aligned with a stripe. This method combines wavelet scale separation with directional Fourier filtering; see :cite:`munch2009stripe`.


Reconstruction effect
---------------------

The realistic model 7 acquisition includes detector-response variations and
additional stripes from the bundled artefact settings. Raw zingers are removed
before normalisation and stripe filtering. See :doc:`../../../scripts/README`.

.. list-table::

   * - .. figure:: ../../../_static/auto_images_methods/data_stripes_added_recon.png
          :alt: FBP before stripe removal, showing ring artefacts.

          FBP before stripe removal.

     - .. figure:: ../../../_static/auto_images_methods/remove_stripe_fw_recon.png
          :alt: FBP after applying remove_stripe_fw to the sinograms.

          FBP after ``remove_stripe_fw``, using the same reconstruction settings and
          display limits. This example illustrates behaviour, not a claim of
          optimal parameters for every dataset.

Papers and further reading
--------------------------

* `Stripe and ring artifact removal with combined wavelet—Fourier filtering <https://doi.org/10.1364/OE.17.008567>`_.

Validation
----------

:doc:`I12 parameter comparisons <../../../validation/preprocessing/stripe_removal>` retains the original
Zenodo regression parameters and numerical checks.

API reference
-------------

:func:`httomolibgpu.prep.stripe.remove_stripe_fw` — signature, input/output types and full parameter details.
