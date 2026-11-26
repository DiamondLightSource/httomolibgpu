.. _method_iterative_alignment:

Iterative projection alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`httomolibgpu.recon.algorithm.iterative_alignment` alternates reconstruction,
reprojection and translation registration using ToMoBAR's ASTRA backend.
Iterative reprojection alignment is described by :cite:`gursoy2017rapid`.
This implementation updates detector geometry while retaining the original
measured projections.
The centre of rotation is fixed; alignment estimates per-view horizontal and
vertical translations, not rotations or deformations.

Pass projections in ``(angles, detector rows, detector columns)`` order and
angles in radians. At least three detector rows and columns are required.
For transmission tomography, apply dark/flat correction and the negative
logarithm first. OSEM instead expects nonnegative emission measurements.

.. code-block:: python

    import numpy as np
    import cupy as cp
    from httomolibgpu import iterative_alignment
    from tomobar.options import Regularisation

    # data: preprocessed CuPy projections; angles: NumPy radians.
    volume, shifts = iterative_alignment(
        data,
        angles,
        center=127.5,
        reconstruction_method="FISTA",
        subsets_number=1,
        iterations=20,
        reconstruction_iterations=2,
        regularisation=Regularisation(method="PD_TV", strength=1e-4),
        solver_options={"nonnegativity": True},
        max_shift=2.0,
    )
    # volume: C-contiguous float32 (Y, detector row, X).
    # shifts: NumPy (views, 2), horizontal then vertical pixels.
    np.save("detector_shifts.npy", shifts)
    cp.save("aligned_reconstruction.npy", volume)

GPU phase correlation is the default; ``registration_backend="skimage"`` selects
CPU registration with per-view transfers. Inner solvers and regularisation use
native ToMoBAR options and objective scaling. Alignment ``relaxation`` damps
shift updates; ADMM's separate relaxation belongs in ``solver_options``.

Offsets have the opposite sign to measured image motion. ``detector_shifts``
can supply an initial offset array, including offsets from a previous run.
``initial`` accepts a NumPy or CuPy volume in the same order as the returned volume.
The mean translation is preserved unless ``reference_view`` fixes a view.
By default the function returns the two arrays ``(volume, shifts)``. Request
histories and the final geometry with ``return_diagnostics=True``:

.. code-block:: python

    volume, shifts, diagnostics = iterative_alignment(
        data,
        angles,
        return_diagnostics=True,
    )
    print(diagnostics["converged"])
    print(diagnostics["residuals"])

The additional dictionary contains the following diagnostics. Here ``cycles``
is the number of completed alignment cycles; array-valued diagnostics use NumPy.

.. list-table:: Alignment diagnostics
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Shape and meaning
   * - ``shift_history``
     - ``(cycles + 1, views, 2)`` absolute offsets in horizontal/vertical pixels,
       including the starting geometry; the final entry equals ``shifts``.
   * - ``convergence``
     - ``(cycles,)`` RMS applied shift update in pixels.
   * - ``residuals``
     - ``(cycles + 1,)`` relative L2 projection errors before each registration
       and after the final reconstruction.
   * - ``valid_views``
     - ``(cycles, views)`` boolean usable-registration flags.
   * - ``converged``
     - True when all views are usable and the RMS update is at most tolerance.
   * - ``geometry``
     - Final ToMoBAR ``Parallel3DGeometry``, including CoR and detector shifts;
       angles follow the wrapper's negated-angle convention.

A small update does not guarantee absolute alignment accuracy. Resources are
closed before return; no live ToMoBAR solver state is retained.

``solver_options`` supplies native inner-solver controls, with nonnegativity
True by default. Keep ``initial``, ``input_axes``, ``iterations`` and
``recon_mask_radius`` out of this dictionary; the wrapper sets them through its
dedicated controls. Weights follow the input projection axis order. Fidelity
and diagnostics options accept native ``tomobar.options.DataFidelity`` and
``Diagnostics`` objects. See the function API for all alignment controls.

Papers and further reading
--------------------------

* `Rapid alignment of nanotomography data using joint iterative reconstruction and reprojection <https://doi.org/10.1038/s41598-017-12141-9>`_.

Validation coverage
-------------------

The :download:`automated validation tests <../../../../../tests/test_recon/test_iterative_alignment.py>`
cover alignment options, diagnostics and recovery of synthetic detector translations on the GPU.
Run them from a compatible source checkout with a CUDA device; the downloaded
file is test-suite source, not a standalone notebook.

There is currently no saved validation notebook for this method. Browse the
:doc:`reconstruction validation collection <../../../validation/reconstruction/index>`
for the available notebook-based examples.

API reference
-------------

:func:`httomolibgpu.recon.algorithm.iterative_alignment` — signature, input/output types and full parameter details.
