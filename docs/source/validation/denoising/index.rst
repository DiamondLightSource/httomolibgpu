Denoising
=========

Compare denoisers on clean reference images and volumes with reproducible
Gaussian noise. The notebooks show filtered slices, PSNR and SSIM for all
ToMoBAR denoisers, including five nonlinear-diffusion penalties. The 3D case
also includes volumetric Non-Local Means.

The pepper image is bundled with the repository; the 160³ Shepp–Logan phantom
is generated with TomoPhantom :cite:`kazantsev2018tomophantom`. Neither case needs a Zenodo download.

.. toctree::
    :maxdepth: 1

    ToMoBAR denoisers · Pepper image <denoising_peppers>
    All denoisers · 3D Shepp–Logan <denoising_shepp_logan_3d>
