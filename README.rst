httomolib
---------

Setup
=====
* Clone the repository from GitHub using :code:`git clone git@github.com:DiamondLightSource/httomolib.git`
* Install dependencies from the environment file :code:`conda env create httomolib --file conda/environment.yml`
* Activate the environment with :code:`conda activate httomolib`
* Install the package from the conda channel :code:`conda install -c httomo httomolib`
* Or install the environment in development mode with :code:`pip install -e .[dev]`

Python conda build packaging
=====
* One can also install httomolib as a conda package
* Ensure that dependencies are installed as in the setup above 
* :code:`export VERSION=1.0`
* From the the main folder run: :code:`conda build conda/recipe/ -c conda-forge -c httomo -c astra-toolbox`

An example of using the API
===========================
* The file :code:`examples/normalize-data.py` shows how to apply the CuPy implementation of dark-flat field correction to the :code:`tests/test_data/tomo_standard.npz` data.
* The file :code:`examples/fresnel-filter.py` shows how to apply the CuPy implementation of Fresnel filtering to the :code:`tests/test_data/tomo_standard.npz` data.

Input data for methods
======================

* We load the projection data from the file :code:`tests/test_data/tomo_standard.npz` using :code:`numpy.load`, which returns a dictionary-like object that can be indexed using the keys :code:`'data'` (to get :code:`host_data`), :code:`'flats'`, and :code:`'darks'`.
* The dataset :code:`/data` in :code:`tests/test_data/normalized-projs.h5` is the input for methods in :code:`httomolib.prep.stripe`
* The dataset :code:`/data` in :code:`tests/test_data/removed-stripes.h5` is the input for methods in :code:`httomolib.recon.rotation`

Run tests
=========
* Run all tests with :code:`$ pytest`. To increase verbosity, use :code:`$ pytest -v`.
* Run GPU tests separately with :code:`$ pytest -v -m gpu`.
* Run CPU tests separately with :code:`$ pytest -v -m "not gpu"`.
* Run performance tests (only) with :code:`$ pytest --performance`
  (note that performance tests always fail - they report the execution time in an assertion
  to see them in the summary easily)
