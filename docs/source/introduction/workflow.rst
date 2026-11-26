Data conventions and a first reconstruction
===========================================

Start with :doc:`installation` for a compatible GPU environment. This page
explains how to pass your arrays through the standalone library; for managed
pipelines and data loading, see :doc:`httomo`.

Array layout and units
----------------------

.. list-table:: Common conventions
   :header-rows: 1
   :widths: 25 45 30

   * - Quantity
     - Layout or units
     - Details
   * - Projection stack
     - ``(angles, detector rows, detector columns)``
     - Most GPU methods accept CuPy arrays; check each function's dtype requirements.
   * - Flats and darks
     - ``(calibration frames, detector rows, detector columns)``
     - Their frame counts may differ from the projection count.
   * - Reconstruction angles
     - One-dimensional NumPy array in radians
     - One angle per projection; convert degree metadata with ``numpy.deg2rad``.
   * - Rotation centre
     - Horizontal detector coordinate in pixels
     - Supply the coordinate before reconstruction padding. Update it after detector cropping or resampling.
   * - Reconstructed volume
     - ``(Y, detector row, X)``
     - A reconstructed slice is ``volume[:, row, :]``. Detector rows occupy the middle axis.
   * - Paganin parameters
     - Pixel size in micrometres, distance in metres, energy in keV
     - The input is corrected transmission, before the negative logarithm.

``FBP2d_astra`` takes NumPy projections and returns a NumPy volume. The 3D
reconstruction wrappers take CuPy projections and return CuPy volumes. Their
input must reside on the CUDA device selected by ``gpu_id``. Utility and
centre-finding functions have their own return types; follow the API reference.

Choose the data domain before reconstruction
--------------------------------------------

For absorption tomography, a typical processing order is:

#. Correct raw-count zingers if needed with
   :func:`~httomolibgpu.misc.corr.remove_outlier`.
#. Convert counts to transmission with
   :func:`~httomolibgpu.prep.normalize.dark_flat_field_correction`.
#. Convert transmission to line integrals using **one** of these routes:

   * Absorption-only processing: :func:`~httomolibgpu.prep.normalize.minus_log`.
   * Single-material phase retrieval: :func:`~httomolibgpu.prep.phase.paganin_filter`,
     which already includes the negative logarithm.

#. Apply suitable stripe correction and determine the rotation centre.
#. Reconstruct with a transmission reconstruction method, then denoise or
   rescale the result when needed.

The order of geometric correction, stripe correction and phase retrieval
can depend on the acquisition. Consult the individual guides and inspect
intermediate images. Applying ``minus_log`` after ``paganin_filter`` applies
the logarithm twice. ``OSEM3d_tomobar`` is a separate emission model: supply
nonnegative emission measurements rather than transmission line integrals.

A minimal absorption reconstruction
-----------------------------------

This function accepts your already loaded raw projections, flats, darks,
angle metadata in degrees and estimated rotation centre. It uses a lower
transmission bound to keep the logarithm finite; choose that bound for your
data and inspect any values that are clipped.

.. code-block:: python

   import numpy as np
   import cupy as cp
   from httomolibgpu.prep.normalize import dark_flat_field_correction, minus_log
   from httomolibgpu.recon.algorithm import FBP3d_tomobar


   def reconstruct(raw, flats, darks, angles_degrees, center):
       transmission = dark_flat_field_correction(
           cp.asarray(raw, dtype=cp.float32),
           cp.asarray(flats, dtype=cp.float32),
           cp.asarray(darks, dtype=cp.float32),
           lower_bound=1e-6,
       )
       line_integrals = minus_log(transmission)
       angles = np.deg2rad(np.asarray(angles_degrees, dtype=np.float32))
       volume = FBP3d_tomobar(line_integrals, angles, center=center)
       return cp.asnumpy(volume)

For worked examples with datasets, processing parameters and saved outputs,
see :doc:`../validation/reconstruction/index`. For synthetic data and the
figures used in the method guides, see :doc:`../scripts/README`.
