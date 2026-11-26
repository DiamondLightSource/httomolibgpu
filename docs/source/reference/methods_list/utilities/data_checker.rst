Check data for invalid values
=============================

``data_checker``

Inspect supported arrays for NaN and infinity values, optionally correct them, and report warnings. This is useful between processing stages when diagnosing numerical problems.

Usage and parameters
--------------------

infsnans_correct enables correction; zeros_warning checks whether more than half the array is zero. verbosity controls reporting and data_to_method_name identifies the upstream operation. Inspect the source of invalid data rather than relying only on replacement values.

Implementation reference
------------------------

See the `library implementation <https://github.com/DiamondLightSource/httomolibgpu/blob/main/httomolibgpu/misc/utils.py>`_
for the precise numerical conventions used by this utility.

API reference
-------------

:func:`httomolibgpu.misc.utils.data_checker` — signature, input/output types and full parameter details.
