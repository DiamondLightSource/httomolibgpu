HTTomolib is a library of methods for tomography
--------------------------------------------------------------------

**HTTomolib** is a collection of image processing methods in Python for computed tomography.

**HTTomolib** can be used as a stand-alone library, however it has been specifically developed to 
work together with the `HTTomo <https://diamondlightsource.github.io/httomo/>`_ package.
HTTomo is a user interface (UI) written in Python for fast big data processing using MPI protocols. 

Install HTTomolib as a pre-built conda Python package
=========================================================
.. code-block:: console

   $ conda env create --name httomolib # create a fresh conda environment
   $ conda install -c httomo httomolib

Setup the development environment:
==================================

.. code-block:: console
    
   $ git clone git@github.com:DiamondLightSource/httomolib.git # clone the repo
   $ conda env create --name httomolib --file conda/environment.yml # install dependencies
   $ conda activate httomolib # activate the environment
   $ pip install .[dev] # development mode

Build HTTomolib as a conda Python package
=============================================

.. code-block:: console
   $ export VERSION=1.0
   $ conda build conda/recipe/ -c conda-forge -c httomo -c astra-toolbox
