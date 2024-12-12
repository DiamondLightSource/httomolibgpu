cupy_run = False
try:
    import cupy as cp
    import nvtx

    try:
        cp.cuda.Device(0).compute_capability
        cupy_run = True
    except cp.cuda.runtime.CUDARuntimeError:
        print("CuPy library is installed but GPU is not accessible")
        import numpy as cp
except ImportError as e:
    print(
        f"Failed to import module in {__file__} with error: {e}; defaulting to CPU-only mode"
    )
    from unittest.mock import Mock
    import numpy as cp

    nvtx = Mock()
