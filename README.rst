HTTomolibGPU is a library of GPU accelerated methods for tomography
--------------------------------------------------------------------

**HTTomolibGPU** is a collection of image processing methods in Python for computed tomography.
The methods are GPU-accelerated with the open-source Python library `CuPy <https://cupy.dev/>`_. Most of the
methods migrated from `TomoPy <https://tomopy.readthedocs.io/en/stable/>`_ and `Savu <https://savu.readthedocs.io/en/latest/>`_ software packages.
Some of the methods also have been optimised to ensure higher computational efficiency, before ported to CuPy.

The purpose of HTTomolibGPU
===========================

**HTTomolibGPU** can be used as a stand-alone library, see Examples section in `Documentation <https://diamondlightsource.github.io/httomolibgpu/>`_.
However, it has been specifically developed to work together with the `HTTomo <https://diamondlightsource.github.io/httomo/>`_ package as
its backend for data processing. HTTomo is a user interface (UI) written in Python for fast big tomographic data processing using
MPI protocols.

Install HTTomolibGPU as a PyPi package
=========================================================
.. code-block:: console

   $ pip install httomolibgpu

Install HTTomolibGPU as a pre-built conda Python package
=========================================================
.. code-block:: console

   $ conda create --name httomolibgpu # create a fresh conda environment
   $ conda activate httomolibgpu # activate the environment
   $ conda install -c httomo httomolibgpu -c conda-forge # for linux users

Setup the development environment:
==================================

.. code-block:: console

   $ git clone git@github.com:DiamondLightSource/httomolibgpu.git # clone the repo
   $ conda env create --name httomolibgpu --file conda/environment.yml # install dependencies
   $ conda activate httomolibgpu # activate the environment
   $ pip install -e .[dev] # editable/development mode

Build HTTomolibGPU as a conda Python package
============================================

.. code-block:: console

   $ conda build conda/recipe/ -c conda-forge -c httomo

