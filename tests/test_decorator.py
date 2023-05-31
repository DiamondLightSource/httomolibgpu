from httomolib import method_all, method_proj, method_sino, method_registry
import inspect
import numpy as np
from numpy import int32

def test_adds_metdata():
    @method_all(
        calc_max_slices=lambda slice_dim, otherdims, dtype, available_memory, **kwargs: (available_memory
        // dtype().itemsize
        // np.prod(otherdims)
        // 2, dtype)
    )
    def myfunc(a: int) -> int:
        return a**2

    assert myfunc.meta.method_name == "myfunc"
    assert myfunc.meta.module == ["tests", "test_decorator"]
    assert myfunc.meta.pattern == "all"
    assert myfunc.meta.cpu is False
    assert myfunc.meta.gpu is True
    # last parameter '2' is mapped to the kwargs
    assert myfunc.meta.calc_max_slices(0, 
                                       (10, 10), 
                                       int32, 40000, a=2) == (50, int32)
    assert myfunc.__name__ == "myfunc"
    assert inspect.getfullargspec(myfunc).args == ["a"]
    assert myfunc(2) == 4

    # also make sure it's in the method registry
    assert method_registry["tests"]["test_decorator"]["myfunc"] == myfunc.meta
    assert method_registry["tests"]["test_decorator"]["myfunc"](2) == 4


def test_metadata_sino():
    @method_sino(calc_max_slices=None)
    def otherfunc(a: int):
        pass

    assert otherfunc.meta.pattern == "sinogram"

def test_metadata_proj():
    @method_proj(calc_max_slices=None)
    def otherfunc(a: int):
        pass

    assert otherfunc.meta.pattern == "projection"


def test_metadata_cpu():
    @method_all(cpuonly=True)
    def otherfunc(a: int):
        pass

    assert otherfunc.meta.cpu is True
    assert otherfunc.meta.gpu is False


def test_metadata_cpu_and_gpu():
    @method_all(cpugpu=True)
    def otherfunc(a: int):
        pass

    assert otherfunc.meta.cpu is True
    assert otherfunc.meta.gpu is True