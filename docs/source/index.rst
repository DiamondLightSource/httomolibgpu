HTTomolibGPU
============

GPU-accelerated preprocessing, reconstruction and denoising for computed
tomography. Use the library on its own or as a processing backend for HTTomo.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: Get started
        :link: introduction/installation
        :link-type: doc

        Install the library and check the GPU environment.

    .. grid-item-card:: Methods and API
        :link: reference/methods
        :link-type: doc

        Choose a processing method and explore its parameters.

    .. grid-item-card:: Validation
        :link: validation/index
        :link-type: doc

        Browse reconstruction and denoising notebooks with saved results.

    .. grid-item-card:: Benchmarks
        :link: benchmarks/index
        :link-type: doc

        Inspect performance measurements and the settings used to obtain them.

    .. grid-item-card:: Run with HTTomo
        :link: introduction/httomo
        :link-type: doc

        Run all processing methods in pipelines with automatic data handling,
        memory management and parallel execution.


    .. grid-item-card:: Data and figure generation
        :link: scripts/README
        :link-type: doc

        Generate realistic synthetic tomography data and reproduce method figures.


.. _intro_content:

.. toctree::
    :hidden:
    :caption: Introduction
    :maxdepth: 2
    :glob:

    introduction/about
    Run with HTTomo <introduction/httomo>
    introduction/installation
    introduction/workflow

.. _jn_examples:

.. _reference_content:

.. toctree::
    :hidden:
    :caption: Reference guides
    :maxdepth: 2

    reference/methods
    reference/api
    bibliography/biblio

.. toctree::
    :hidden:
    :caption: Validation
    :maxdepth: 1

    validation/index

.. toctree::
    :hidden:
    :caption: Performance benchmarks
    :maxdepth: 1

    benchmarks/index

.. toctree::
    :hidden:
    :caption: Data and figure generation
    :maxdepth: 2

    scripts/README
