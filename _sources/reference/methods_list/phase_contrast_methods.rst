.. _phase_contrast_module:

Phase-contrast enhancement
**************************

Methods from the :mod:`httomolibgpu.prep.phase` module needed when the data requires phase contrast enhancement.

**Description**

In conventional X-ray tomography, image contrast comes mainly from absorption, i.e., how much a material attenuates X-rays.
For some samples, especially biological, the absorption difference between structures can be insignificant and therefore the edge contrast
can be lost. X-rays also undergo phase shifts when passing through materials with different refractive indices and
these phase shifts carry valuable structural information that can be converted into contrast if properly reconstructed.

The phase contrast enhances edges and interfaces, revealing fine internal structure that pure absorption tomography would miss.
When we record propagation-based phase contrast images (e.g., at a certain distance from the sample), the detected intensity includes
a mixture of absorption and phase effects. To reconstruct a clean 3D volume, we must separate phase from intensity.
This process is called the phase retrieval. The Paganin method :cite:`Paganin02`  provides a simple, robust way to do this.

The Paganin formula is the following, see eq.10 in :cite:`paganin2020boosting`:

.. math:: T(x,y) = - \frac{1}{\mu}\ln\left (\mathcal{F}^{-1}\left
    (\frac{\mathcal{F}\left [ I(x, y, z = \Delta) / I_{0} \right ]}{1 +
        \frac{\delta\Delta}{\mu}\left ( k_x^2 + k_y^2 \right )}  \right )\right ),

where:

* :math:`I(x, y, z = \Delta)` - measured X-ray beam intensity at the propagation distance :math:`\Delta`.

* :math:`I_{0}` - measured X-ray beam intensity at the zero-distance from the X-ray source (incident beam).

* :math:`\Delta > 0` - propagation distance of the wavefront from sample to detector.

* :math:`\mu` - linear attenuation coefficient of the single-material object defined as :math:`\mu = 2k\beta`, where :math:`k=\frac{2\pi}{\lambda}` is the wave-number corresponding to the vacuum wavelength :math:`\lambda`.

* :math:`\delta` - the phase decrement, related to the phase shift of X-rays. It is the real part of the complex refractive index:  :math:`n = (1 - \delta) + i \beta`.

* :math:`\beta` - the absorption index, related to the attenuation. It the complex part of the material refractive index: :math:`n = (1 - \delta) + i \beta`.

One can re-write the formula above as:

.. math:: T(x,y) = - \frac{1}{\mu}\ln\left (\mathcal{F}^{-1}\left
    (\frac{\mathcal{F}\left [ I(x, y, z = \Delta) / I_{0} \right ]}{1 +
        \alpha \left ( k_x^2 + k_y^2 \right )}  \right )\right ),

where:

.. math:: \alpha = \frac{\lambda \Delta \delta}{4 \pi \beta}.

**Where and how to use it:**

Works well for single-material and relatively homogeneous or weakly heterogeneous samples (e.g., biological tissues, polymers, soft matter). The filter is particularly useful for weakly absorbing samples imaged with hard X-rays, where traditional absorption contrast is poor.

**What are the adjustable parameters:**

* :code:`input data` is the flat/dark normalised raw data before the negative log. Note that the :math:`-ln()` operation is the part of the Paganin filter.

* :code:`pixel_size` Detector pixel size (resolution) in MICRON units.

* :code:`distance` Propagation distance (:math:`\Delta`) of the wavefront from sample to detector in METRE units.

* :code:`energy` Incident beam energy in keV.

* :code:`ratio_delta_beta` The ratio of :math:`\frac{\delta}{\beta}` is a critical parameter as it defines how the algorithm balances phase contrast versus absorption contrast. Higher ratio values lead to stronger smoothing, more phase contrast recovered, but potential loss of edge sharpness. It is recommended to keep :math:`\frac{\delta}{\beta} > 250` for weakly absorbing materials (like biological tissue, polymers, and light materials). Lower values :math:`\frac{\delta}{\beta} << 250` lead to less smoothing, more edges preserved.

**Real data examples:**

In this section we will be applying Paganin filter to data obtained at Diamond Light Source I12 beamline (beamtime NT41036-2, PI: G. Burca).
The sample is a set of fixed bovine liver sections in a centrifuge tube. This is a biological sample and a good candidate for Paganin filter demonstration.

We used :ref:`method_remove_all_stripe` to remove ring artifacts and :ref:`method_LPRec3d_tomobar` for reconstruction of the filtered projections.
We also used the following parameters for Paganin filter, while varying only :code:`ratio_delta_beta`.

.. code-block:: yaml

    pixel_size: 32.4 # microns
    distance: 3.2    # meters
    energy: 53.0     # KeV

.. list-table::


    * - .. figure:: ../../_static/figures/paganin/lprec_no_paganin.jpg
           :width: 335px

           No Paganin filter applied. The reconstruction is too noisy and without any features.

      - .. figure:: ../../_static/figures/paganin/paganin_10.jpg
           :width: 335px

           Paganin filter with :math:`\frac{\delta}{\beta} = 10`. Some features are recongnisable, but the image is still too noisy.


.. list-table::


    * - .. figure:: ../../_static/figures/paganin/paganin_100.jpg
           :width: 200px

           Paganin filter with :math:`\frac{\delta}{\beta} = 100`. Still a bit noisy.

      - .. figure:: ../../_static/figures/paganin/paganin_500.jpg
           :width: 200px

           Paganin filter with :math:`\frac{\delta}{\beta} = 500`. Close to the optimal value.

      - .. figure:: ../../_static/figures/paganin/paganin_2000.jpg
           :width: 200px

           Paganin filter with :math:`\frac{\delta}{\beta} = 2000`. Oversmoothed.