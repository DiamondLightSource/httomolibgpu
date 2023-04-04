import inspect

from typing import Callable, List, Literal, Tuple, Protocol
import numpy as np
from dataclasses import dataclass, field

method_registry = dict()


class MemoryFunction(Protocol):
    """
    A callable signature for a function to determine the maximum chunk size
    supported given the chunking dimension,
    the size of the other dimensions,
    the data type,
    and the available memory in bytes.
    It takes the actual method parmaters as kwargs, which it may use if needed.

    It returns the maximum size in slice_dim dimension that can be supported
    within the given memory.
    """

    def __call__(
        self,
        slice_dim: int,
        other_dims: Tuple[int, int],
        dtype: np.dtype,
        available_memory: int,
        **kwargs,
    ) -> int:
        ...


class MemorySinglePattern(Protocol):
    """Signature for a function to calculate the max chunk size for a method that
    supports only one pattern.
    It avoids the slice_dim parameter, as that's redundant in that case
    """

    def __call__(
        self,
        other_dims: Tuple[int, int],
        dtype: np.dtype,
        available_memory: int,
        **kwargs,
    ) -> int:
        ...


def calc_max_slices_default(
    slice_dim: int,
    other_dims: Tuple[int, int],
    dtype: np.dtype,
    available_memory: int,
    **kwargs,
) -> int:
    """Default function for calculating maximum slices, which simply assumes
    space for input and output only is required, both with the same datatype,
    and no temporaries."""

    return available_memory // (np.prod(other_dims) * dtype.itemsize * 2)


def calc_max_slices_single_pattern_default(
    other_dims: Tuple[int, int], dtype: np.dtype, available_memory: int, **kwargs
) -> int:
    """Default function for calculating maximum slices, which simply assumes
    space for input and output only is required, both with the same datatype,
    and no temporaries."""

    return calc_max_slices_default(0, other_dims, dtype, available_memory, **kwargs)


@dataclass(frozen=True)
class MethodMeta:
    """Class for meta properties of a function"""

    method_name: str
    signature: inspect.Signature
    module: List[str]
    calc_max_slices: MemoryFunction
    pattern: Literal["projection", "sinogram", "all"]
    cpu: bool
    gpu: bool
    function: Callable
    others: dict = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def method(
    calc_max_slices: MemoryFunction = calc_max_slices_default,
    cpuonly=False,
    cpugpu=False,
    **others,
):
    """
    Decorator for exported tomography methods, annotating the method with
    a function to calculate the memory requirements as well as other properties.

    Parameters
    ----------
    calc_max_slices: MemoryFunction, optional
        Function to calculate how many slices can fit in the given GPU memory at max.
        If not given, it assumes memory is only needed for input and output and no
        temporaries is needed.
        It is not used for CPU-only functions.
    cpuonly: bool, optional
        Marks the method as CPU only (default: False).
    cpugpu: bool, optional
        Marks the method as supporting both CPU and GPU input arrays (default: False).
    pattern: string, optional
        Sets the patterns supported by the method: 'sinogram', 'projection', 'all'. (default: 'all')
    **other: dict, optional
        Any other keyword arguments will be added to the meta info as a dictionary.
    """

    def _method(func):
        pattern = others["pattern"] if "pattern" in others else "all"
        gpu = not cpuonly
        cpu = cpugpu or cpuonly
        assert not (cpuonly and cpugpu), "cpuonly and cpugpu are mutually exclusive"
        func.meta = MethodMeta(
            method_name=func.__name__,
            signature=inspect.signature(func),
            module=inspect.getmodule(func).__name__.split("."),
            calc_max_slices=calc_max_slices,
            pattern=pattern,
            gpu=gpu,
            cpu=cpu,
            others=others,
            function=func,
        )

        # register the method
        lvl = method_registry
        for m in func.meta.module:
            if not m in lvl:
                lvl[m] = dict()
            lvl = lvl[m]
        lvl[func.meta.method_name] = func.meta

        return func

    return _method


def method_sino(
    calc_max_slices: MemorySinglePattern = calc_max_slices_single_pattern_default,
    cpuonly=False,
    cpugpu=False,
    **others,
):
    """
    Decorator for exported tomography methods, annotating the method with
    a function to calculate the memory requirements as well as other properties.

    This is a convenience version for sinogram pattern - see method for details.
    """

    def _calc_max_slices(_, otherdims, dtype, available_memory, **others):
        return calc_max_slices(otherdims, dtype, available_memory, **others)

    return method(_calc_max_slices, cpuonly, cpugpu, **others, pattern="sinogram")


def method_proj(
    calc_max_slices: MemorySinglePattern = calc_max_slices_single_pattern_default,
    cpuonly=False,
    cpugpu=False,
    **others,
):
    """
    Decorator for exported tomography methods, annotating the method with
    a function to calculate the memory requirements as well as other properties.

    This is a convenience version for projection pattern - see method for details.
    """

    def _calc_max_slices(_, otherdims, dtype, available_memory, **others):
        return calc_max_slices(otherdims, dtype, available_memory, **others)

    return method(_calc_max_slices, cpuonly, cpugpu, **others, pattern="projection")


method_all = method
