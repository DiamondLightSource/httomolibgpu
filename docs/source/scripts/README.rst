Generate realistic data and figures
===================================

``methods_to_generate_images.py`` generates realistic TomoPhantom data in memory,
runs HTTomolibGPU methods and saves PNG illustrations. It does not write a NeXus
file or load the old test-data NPZ archive.


TomoPhantom resources
---------------------

The analytic projections, realistic flat fields and artefact simulation in this
workflow are provided by TomoPhantom:

* `GitHub repository <https://github.com/dkazanc/TomoPhantom>`_: source code,
  examples and development information.
* `Installation guide <https://dkazanc.github.io/TomoPhantom/howto/installation.html>`_:
  installation options and build requirements for the compiled generator.
* Kazantsev et al. (2018), `TomoPhantom, a software package to generate 2D–4D
  analytical phantoms for CT image reconstruction algorithm benchmarks
  <https://doi.org/10.1016/j.softx.2018.05.003>`_, *SoftwareX* **7**, 150–155.
  Cite this publication when using TomoPhantom-generated data in research.


Requirements
------------

Run from the repository root in an environment containing the current
HTTomolibGPU and ToMoBAR checkouts, TomoPhantom with its compiled generator,
CuPy, ASTRA Toolbox 2.5 or newer, NumPy, SciPy, PyWavelets, Pillow, psutil and
Matplotlib. CUDA processing requires an NVIDIA GPU.

Run the generator
-----------------

.. code-block:: console

   python docs/source/scripts/methods_to_generate_images.py \
       --model-number 7 --sinogram-shape 256 512 740 \
       --flats 20 --darks 10 --source-intensity 20000 --seed 1 \
       --artefacts docs/source/scripts/artefacts.json \
       --flat-settings docs/source/scripts/flat_settings.json \
       --output docs/source/_static/auto_images_methods

These are the defaults. The two bundled JSON files copy the settings from
TomoPhantom's ``tomophantom/scripts/nxs_generator.py`` example. Paths to their
bundled copies are resolved relative to this script, not the current directory.
You can supply alternative JSON settings explicitly.

How the data is generated
-------------------------

The synthesis follows that generator's realistic mode: model 7 analytic
projections over 0--179.9 degrees, synthetic detector flats, raw-space artefacts,
0.9 signal gain, Poisson dark current (mean 100 ADU), Gaussian read noise
(standard deviation 2 ADU), clipping and uint16 quantisation. Ideal line
integrals use the single global projection maximum used by ``synth_flats``;
they are not unscaled attenuation values. No additional Poisson corruption is
added by ``artefacts_mix``, since ``synth_flats`` already models photon noise.

The default artefacts include both zingers and stripes. Outlier removal is
applied to raw counts; stripe filters receive logarithmic data after outlier
removal. This avoids confusing zinger suppression with stripe correction.
Rescaling examples use the outlier-corrected data and a global 0.5--99.5
percentile display window before applying the illustrated range cutoffs. This
prevents a few dead-detector/logarithm-floor spikes from flattening all contrast.
The stripe input filename ``data_stripes_added`` is retained for compatibility
with existing documentation; the stripes now originate in the realistic
acquisition simulation rather than a second post-normalisation corruption.

Generated figures and provenance
--------------------------------

Existing projection, sinogram, residual, rescaling and histogram filenames are
preserved. Reconstruction images compare the ideal, corrupted, outlier-corrected
and stripe-filtered data using FBP3d_tomobar on a small slab around the selected
detector row. They illustrate ring artefacts without reconstructing the entire
3D volume. Reconstruction images share the ideal reconstruction's display
range; method input/output images also share contrast limits. Residuals show
absolute input/output differences, not error relative to the ideal model.

The ``dark_flat_field_correction_input`` and ``dark_flat_field_correction``
image pairs show raw detector counts and corrected transmission before the
negative logarithm. Transmission uses a fixed 0--1.1 display range; raw counts
use robust contrast limits. The existing ``normalisation`` images show the
subsequent line integrals after ``minus_log``.

``generation_manifest.json`` records the parameters, settings, display limits
and clipping count. PNGs are overwritten when rerunning into the same folder;
use a different ``--output`` directory to retain an earlier image set.

Useful options
--------------

* ``--crop N`` trims N pixels from each edge of saved data and reconstruction
  images (default: 30), making artefacts easier to inspect. Use ``--crop 0`` to
  save full views. An axis with at most ``2 * N`` pixels is left intact, preserving
  short calibration stacks and small smoke-test views. Processing and display
  limits use the full data; histogram counts use the cropped projection.
* ``--row`` and ``--projection`` select the views (defaults: middle detector row
  and one-quarter of the projection count).
* ``--gpu-rows`` bounds GPU workspace through detector-row blocks. Median and
  outlier filters include neighbour halos at block boundaries. Realistic
  synthesis still needs the complete projection array in CPU memory.
* ``--methods remove_outlier raven_filter reconstruction`` selects examples.
  Normalisation and raw/calibration/ideal illustrations are always produced.
* ``--synthesis-only`` saves only raw, calibration and ideal PNGs without CUDA.
* ``--sinogram-shape 24 48 64 --output /tmp/method-images`` is a small smoke run.

The generation logic is adapted from TomoPhantom's Apache-2.0-licensed NeXus
generator; the NeXus-writing code is deliberately omitted.

Download the script and settings
--------------------------------

Run the script from a compatible library checkout. Download the files for
inspection or keep them together when using the script outside the checkout:

* :download:`Image-generation script <methods_to_generate_images.py>`
* :download:`Artefact settings <artefacts.json>`
* :download:`Flat-field settings <flat_settings.json>`

The two JSON configurations used for the default examples are shown below.

.. dropdown:: Artefact settings

   .. literalinclude:: artefacts.json
      :language: json

.. dropdown:: Flat-field settings

   .. literalinclude:: flat_settings.json
      :language: json

Example reconstruction comparison
---------------------------------

These images use the same reconstruction settings and display range. The
middle image retains detector stripes and noise after zinger removal; the
last applies ``remove_all_stripe`` before reconstruction.

.. list-table::

   * - .. figure:: ../_static/auto_images_methods/raw_data_recon.png
          :alt: Synthetic reconstruction with strong zinger streaks.

          Before correction.

     - .. figure:: ../_static/auto_images_methods/data_stripes_added_recon.png
          :alt: Synthetic reconstruction after zinger removal with ring artefacts remaining.

          After zinger removal.

     - .. figure:: ../_static/auto_images_methods/remove_all_stripe_recon.png
          :alt: Synthetic reconstruction after zinger and stripe removal.

          After stripe removal.

See :doc:`../reference/methods` for individual method explanations and further
before/after examples.
