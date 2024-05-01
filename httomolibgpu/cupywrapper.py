cupy_run = False
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
        cupy_run = True
    except cp.cuda.runtime.CUDARuntimeError:
        print("CuPy library is a major dependency for HTTomolibgpu, please install")
        import numpy as cp
except ImportError:
    import numpy as cp