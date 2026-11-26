Require a CUDA device
=====================

``require_gpu``

Check that CUDA execution is available before attempting a GPU operation. This supports clear failures in environments where importing the package is possible without a working GPU.

Usage and parameters
--------------------

operation supplies context for the error; device_id optionally checks a particular device. A failed check raises an exception. This helper validates availability and does not process tomography data.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/cupywrapper.py>`_
for the precise numerical conventions used by this utility.

API reference
-------------

:func:`httomolibgpu.cupywrapper.require_gpu` — signature, input/output types and full parameter details.
