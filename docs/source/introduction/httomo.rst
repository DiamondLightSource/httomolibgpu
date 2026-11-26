Run with HTTomo
===============

`HTTomo <https://diamondlightsource.github.io/httomo/>`_ is a framework for
high-throughput tomography processing. All processing methods in HTTomolibGPU
can be run within HTTomo, alongside methods from the companion
HTTomolib library, as steps in a configurable processing pipeline.

Let HTTomo manage your data
---------------------------

HTTomolibGPU supplies the processing methods; HTTomo manages the workflow and
the data passed between them. Instead of writing your own data-handling code,
you describe the methods and their parameters in a YAML pipeline. HTTomo handles:

* Loading tomography data, including dark and flat fields and rotation angles.
* Splitting large datasets into blocks to fit available memory.
* Moving and re-slicing data between processing steps as needed.
* Parallel execution with MPI and use of available CPU and GPU resources.
* Saving results and, when requested, intermediate datasets.

This lets you combine preprocessing, reconstruction and postprocessing in a
repeatable workflow, from a small data preview to a full dataset.

Get started
-----------

Visit the `HTTomo main page <https://diamondlightsource.github.io/httomo/>`_
for installation instructions, a quickstart, ready-to-use pipelines and guides
to configuring your own workflow. Use compatible versions of HTTomo, its
processing backends and the libraries, as described in the HTTomo documentation.
