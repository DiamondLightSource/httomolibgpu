httomolib
---------

Setup
=====
* Clone the repository from GitHub using :code:`git clone git@github.com:DiamondLightSource/tomo-methods.git`
* Install dependencies from the environment file :code:`conda env create httomolib --file conda/environment.yml` (SLOW)
* Alternatively you can install from the existing explicit file :code:`conda create --name httomolib --file conda/explicit.txt`
* Activate the environment with :code:`conda activate httomolib`

An example of running a method
==============================
* The file :code:`examples/normalize-data.py` applies the CuPy implementation of dark-flat field correction to the :code:`data/tomo_standard.nxs` data
* It can be run by navigating to the root directory, and then doing :code:`python examples/normalize-data.py`

Input data for methods
======================

* The file :code:`data/tomo_standard.nxs` contains the data that should be passed to the `standard_tomo()` loader function, and the loader's first three outputs should be passed to methods in `normalisation.py`
* The dataset :code:`/data` in :code:`data/normalized-projs.h5` is the input for methods in `stripe_removal.py`
