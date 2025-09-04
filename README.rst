HTTomolibGPU is a library of GPU accelerated methods for tomography
--------------------------------------------------------------------

**HTTomolibGPU** is a collection of image processing methods in Python for computed tomography.
The methods are GPU-accelerated with the open-source Python library `CuPy <https://cupy.dev/>`_. Most of the
methods migrated from `TomoPy <https://tomopy.readthedocs.io/en/stable/>`_ and `Savu <https://savu.readthedocs.io/en/latest/>`_ software packages.
Some of the methods also have been optimised to ensure higher computational efficiency, before ported to CuPy.

The purpose of HTTomolibGPU
===========================

Although **HTTomolibGPU** can be used as a stand-alone library, it has been specifically developed to work together with the 
`HTTomo <https://diamondlightsource.github.io/httomo/>`_ package as
its backend for data processing. HTTomo is a user interface (UI) written in Python for fast big tomographic data processing using
MPI protocols or as well serially.

Installation
============

HTTomolibGPU is available on PyPI, so it can be installed into either a virtual environment or
a conda environment.

Virtual environment
~~~~~~~~~~~~~~~~~~~
.. code-block:: console

   $ python -m venv httomolibgpu
   $ source httomolibgpu/bin/activate
   $ pip install httomolibgpu

Conda environment
~~~~~~~~~~~~~~~~~
.. code-block:: console

   $ conda create --name httomolibgpu # create a fresh conda environment
   $ conda activate httomolibgpu # activate the environment
   $ conda install conda-forge::cupy==12.3.0
   $ pip install httomolibgpu

Setup the development environment:
==================================

.. code-block:: console

   $ git clone git@github.com:DiamondLightSource/httomolibgpu.git # clone the repo
   $ conda env create --name httomolibgpu -c conda-forge cupy==12.3.0 # install dependencies
   $ conda activate httomolibgpu # activate the environment
   $ pip install -e ./httomolibgpu[dev] # editable/development mode
