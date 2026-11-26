Load and compile a CUDA kernel module
=====================================

``load_cuda_module``

Read a bundled CUDA source file and compile it into a CuPy RawModule. This is a developer utility used by GPU processing kernels.

Usage and parameters
--------------------

file is the bundled source basename without .cu. name_expressions selects kernel names, including template instantiations, and options adds compiler flags. Compilation uses C++17 and requires an appropriate CUDA environment. The result is a kernel module, not an image array.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/cuda_kernels/__init__.py>`_
for the precise numerical conventions used by this utility.

API reference
-------------

:func:`httomolibgpu.cuda_kernels.load_cuda_module` — signature, input/output types and full parameter details.
