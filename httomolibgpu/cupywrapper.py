cupy_run = True
import cupy as cp
from cupyx.scipy.fft import next_fast_len
from unittest.mock import Mock
from scipy.fft import next_fast_len
nvtx = Mock()
