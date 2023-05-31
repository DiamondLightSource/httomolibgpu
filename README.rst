HTTomolib-GPU is a library of GPU accelerated methods for tomography
--------------------------------------------------------------------

**HTTomolib-GPU** is a collection of image processing methods in Python for computed tomography.
The methods are GPU-accelerated with the open-source Python library `CuPy <https://cupy.dev/>`_. Most of the 
methods migrated from `TomoPy <https://tomopy.readthedocs.io/en/stable/>`_ and `Savu <https://savu.readthedocs.io/en/latest/>`_ software packages. They have been
optimised to ensure computational efficiency and high-throughput.

**HTTomolib-GPU** can be used as a stand-alone library, but it has been specifically developed to 
work together with the `HTTomo <https://diamondlightsource.github.io/httomo/>`_ package.
HTTomo is a user interface (UI) written in Python for fast big data processing using MPI protocols. 

Install HTTomolib-GPU as a pre-built conda Python package
=========================================================
.. code-block:: console

   $ conda env create --name httomolib # create a fresh conda environment
   $ conda install -c httomo httomolib-gpu

Setup the development environment:
==================================

.. code-block:: console
    
   $ git clone git@github.com:DiamondLightSource/httomolib-gpu.git # clone the repo
   $ conda env create --name httomolib --file conda/environment.yml # install dependencies
   $ conda activate httomolib # activate the environment
   $ pip install .[dev] # development mode

Build HTTomolib-GPU as a conda Python package
=============================================

.. code-block:: console
   $ export VERSION=1.0
   $ conda build conda/recipe/ -c conda-forge -c httomo -c astra-toolbox

An example of using the API
===========================
* The file :code:`examples/normalize-data.py` shows how to apply the CuPy implementation of dark-flat field correction to the :code:`tests/test_data/tomo_standard.npz` data.
* The file :code:`examples/fresnel-filter.py` shows how to apply the CuPy implementation of Fresnel filtering to the :code:`tests/test_data/tomo_standard.npz` data.

Input data for methods
======================

* We load the projection data from the file :code:`tests/test_data/tomo_standard.npz` using :code:`numpy.load`, which returns a dictionary-like object that can be indexed using the keys :code:`'data'` (to get :code:`host_data`), :code:`'flats'`, and :code:`'darks'`.
* The dataset :code:`/data` in :code:`tests/test_data/normalized-projs.h5` is the input for methods in :code:`httomolib-gpu.prep.stripe`
* The dataset :code:`/data` in :code:`tests/test_data/removed-stripes.h5` is the input for methods in :code:`httomolib-gpu.recon.rotation`

Run tests
=========
* Run all tests with :code:`$ pytest`. To increase verbosity, use :code:`$ pytest -v`.
* Run GPU tests separately with :code:`$ pytest -v -m gpu`.
* Run CPU tests separately with :code:`$ pytest -v -m "not gpu"`.
* Run performance tests (only) with :code:`$ pytest --performance`
  (note that performance tests always fail - they report the execution time in an assertion
  to see them in the summary easily)
