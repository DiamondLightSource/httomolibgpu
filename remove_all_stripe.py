import cupy as cp
import numpy as np
import os
import time
from cupy.cuda import memory_hooks
from datetime import datetime
from math import isclose
from cupyx.profiler import time_range

from httomolibgpu.prep.stripe import remove_all_stripe

test_data_path = "/mnt/gpfs03/scratch/data/imaging/tomography/zenodo"
data_path = os.path.join(test_data_path, "synth_tomophantom1.npz")
data_file = np.load(data_path)
projdata = cp.asarray(cp.swapaxes(data_file["projdata"], 0, 1))
angles = cp.asarray(data_file["angles"])

with time_range("all_stripe", color_id=0):
    remove_all_stripe(
        cp.copy(projdata),
        snr=0.1,
        la_size=71,
        sm_size=31,
        dim=1
    )


# cold run
remove_all_stripe(
    cp.copy(projdata),
    snr=0.1,
    la_size=71,
    sm_size=31,
    dim=1,
)

dev = cp.cuda.Device()
dev.synchronize()
start = time.perf_counter_ns()
for _ in range(10):
    remove_all_stripe(
        cp.copy(projdata),
        snr=0.1,
        la_size=71,
        sm_size=31,
        dim=1,
    )   
    
dev.synchronize()
duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

print(duration_ms)


output = remove_all_stripe(cp.copy(projdata), snr=0.1, la_size=61, sm_size=21, dim=1)
residual_calc = projdata - output
norm_res = cp.linalg.norm(residual_calc.flatten())
assert isclose(norm_res, 67917.71, abs_tol=10**-2)

output = remove_all_stripe(cp.copy(projdata), snr=0.001, la_size=61, sm_size=21, dim=1)
residual_calc = projdata - output
norm_res = cp.linalg.norm(residual_calc.flatten())
assert isclose(norm_res, 70015.51, abs_tol=10**-2)

hook = memory_hooks.LineProfileHook()
with hook:
    remove_all_stripe(
        cp.copy(projdata),
        snr=0.1,
        la_size=71,
        sm_size=31,
        dim=1
    )
hook.print_report()