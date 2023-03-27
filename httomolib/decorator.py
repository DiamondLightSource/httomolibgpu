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
    def __call__(self, other_dims: Tuple[int, int], dtype: np.dtype, available_memory: int) -> int: ...

@dataclass(frozen=True)
class MethodMeta:
    """Class for meta properties of a function"""

    method_name: str
    signature: inspect.Signature
    module: List[str]
    calc_max_slices: MemoryFunction
    pattern: Literal["projection", "sinogram", "all"]
    function: Callable
    others: dict = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def method(calc_max_slices: MemoryFunction, **others):
    def _method(func):
        pattern = others["pattern"] if "pattern" in others else "all"
        func.meta = MethodMeta(
            method_name=func.__name__,
            signature=inspect.signature(func),
            module=inspect.getmodule(func).__name__.split("."),
            calc_max_slices=calc_max_slices,
            pattern=pattern,
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


def method_sino(calc_max_slices: MemorySinglePattern, **others):
    
    def _calc_max_slices(_, otherdims, dtype, available_memory, **others):
        return calc_max_slices(otherdims, dtype, available_memory, **others)
    
    return method(calc_max_slices=_calc_max_slices, **others, pattern="sinogram")


def method_proj(calc_max_slices: MemorySinglePattern, **others):
    def _calc_max_slices(_, otherdims, dtype, available_memory, **others):
        return calc_max_slices(otherdims, dtype, available_memory, **others)
    
    return method(calc_max_slices=_calc_max_slices, **others, pattern="projection")


method_all = method
