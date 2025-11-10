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

:math:`n = (1 - \delta) + i \beta`
