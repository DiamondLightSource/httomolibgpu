.. _method_raven_filter:

Raven filter
^^^^^^^^^^^^

**Description**

Raven filter is based on a FFT filter applied in sinogram space to minimise stripes leading to `Ring Artifacts <https://radiopaedia.org/articles/ring-artifact-2?lang=gb>`_ in reconstruction. See more about the filter in the original publication :cite:`raven1998numerical`.

**Where and how to use it:**

Raven filter can be used in pre-processing, i.e., before the reconstruction to remove the stripes present in the data.

**What are the adjustable parameters:**

* :code:`uvalue` is the positive cut-off frequency in Fourier bins. Smaller
  values give stronger smoothing of the selected band.
* :code:`nvalue` is the positive integer filter order controlling the roll-off.
* :code:`vvalue` is the non-negative half-width of the band around zero
  angular frequency. Zero filters only the zero-frequency row.
* :code:`pad_y` and :code:`pad_x` are non-negative padding sizes along the
  angular and horizontal detector axes, respectively.

The Fourier-domain transfer function is real and symmetric about zero:

.. math::

   H(u,v) =
   \begin{cases}
      \frac{1}{1 + (u/u_0)^{2n}}, & |v| \leq v_0, \\
      1, & \text{otherwise}.
   \end{cases}

This expression is also given in equation (14) of
`Farzi et al. (2021) <https://doi.org/10.1109/TMI.2021.3097907>`_.
The filter scales Fourier coefficients without introducing a phase rotation.

.. note:: The frequency centring and real weighting correct the earlier
   implementation inherited from Savu. Results can differ from older versions.

**Practical example:**

In this example we demonstrate how to apply Raven filter to the normalised projection data.

.. warning:: As with any pre-processing tool, one needs to be careful in applying Raven filter to data. Sub-optimal parameters can potentially lead to other artifacts introduced in the reconstruction and blurring.
.. list-table::


    * - .. figure:: ../../../_static/auto_images_methods/data_stripes_added_sino.png

           Input (sinogram view) with the stripes added

      - .. figure:: ../../../_static/auto_images_methods/data_stripes_added_proj.png

           Input (projection view) with the stripes added

    * - .. figure:: ../../../_static/auto_images_methods/raven_filter_sino.png

           After applying Raven filter (sinogram)

      - .. figure:: ../../../_static/auto_images_methods/raven_filter_proj.png

           After applying Raven filter (projection)

    * - .. figure:: ../../../_static/auto_images_methods/raven_filter_res_sino.png

           Sinogram view of absolute residual between input and output

      - .. figure:: ../../../_static/auto_images_methods/raven_filter_res_proj.png

           Projection view of absolute residual between input and output
