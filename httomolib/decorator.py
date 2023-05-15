import inspect

from typing import Callable, Dict, List, Literal, Tuple, Protocol, TypeAlias, Union
import numpy as np
from dataclasses import dataclass, field


class MemoryFunction(Protocol):
    """
    A callable signature for a function to determine the maximum chunk size supported,
    given
    - the slicing dimension,
    - the size of the remaining (after slicing) input dimensions,
    - the data type for the input,
    - and the available memory in bytes.
    It takes the actual method parameters as kwargs, which it may use if needed.

    It returns the maximum size in slice_dim dimension that can be supported
    within the given memory.
    """

    def __call__(
        self,
        slice_dim: int,
        non_slice_dims_shape: Tuple[int, int],
        dtype: np.dtype,
        available_memory: int,
        **kwargs,
    ) -> Tuple[int, np.dtype, Tuple[int, int]]:
        """
        Calculate the maximum number of slices that can fit in the given memory,
        for a method with the 'all' pattern.

        Parameters
        ----------
        slice_dim : int
            The dimension in which the slicing happens (0 for projection, 1 for sinogram)
        non_slice_dims_shape : Tuple[int, int]
            Shape of the data input in the other 2 dimensions that are not sliced
        dtype : np.dtype
            The numpy datatype for the input data
        available_memory : int
            The available memory to fit the slices, in bytes
        kwargs : dict
            Dictionary of the extra method parameters (apart from the data input)

        Returns
        -------
        Tuple[int, np.dtype]
            Tuple consisting of:
            - the maximum number of slices that it can fit into the given available memory
            - the output dtype for the given input dtype
            - the output data shape

        """
        ...


class MemorySinglePattern(Protocol):
    """
    Signature for a function to calculate the max chunk size for a method that
    supports only one pattern.
    It avoids the slice_dim parameter, as that is redundant in that case.
    """

    def __call__(
        self,
        non_slice_dims_shape: Tuple[int, int],
        dtype: np.dtype,
        available_memory: int,
        **kwargs,
    ) -> Tuple[int, np.dtype, Tuple[int, int]]:
        """
        Calculate the maximum number of slices that can fit in the given memory,
        for a method with the 'projection' or 'sinogram' pattern.

        Parameters
        ----------
        non_slice_dims_shape : Tuple[int, int]
            Shape of the data input in the other 2 dimensions that are not sliced
        dtype : np.dtype
            The numpy datatype for the input data
        available_memory : int
            The available memory to fit the slices, in bytes
        kwargs : dict
            Dictionary of the extra method paramters (apart from the data input)

        Returns
        -------
        Tuple[int, np.dtype]
            Tuple consisting of:
            - the maximum number of slices that it can fit into the given available memory
            - the output dtype for the given input dtype
            - the output data shape

        """
        ...


@dataclass(frozen=True)
class MethodMeta:
    """
    Class for meta properties of a method (to be stored as func.meta by the decorator).

    Attributes
    ----------
    method_name : str
        Name of the method as a string
    signature : Signature
        An inspect signature of the method with its arguments
    module : List[str]
        A list representing the module hierarchy where the method is defined, e.g.
        ['httomolib', 'prep', 'normlize'] for 'httomolib/prep/normalize.py'
    calc_max_slices : MemoryFunction
        Method to calculate the maximum number of slices that can fit in the given
        available memory.
    pattern : Literal["projection", "sinogram", "all"]
        The pattern supported by the method.
    cpu : bool
        Whether the method supports CPU data (numpy)
    gpu : bool
        Whether the method supports GPU data (cupy)
    function : Callable
        The actual method itself
    others : dict
        Dictionary of additional arbitrary meta information
    """
    
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


MetaDict: TypeAlias = Dict[str, MethodMeta]
method_registry: Dict[str, Union[MetaDict, MethodMeta]] = dict()

def calc_max_slices_default(
    slice_dim: int,
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    available_memory: int,
    **kwargs,
) -> Tuple[int, np.dtype, Tuple[int, int]]:
    """
    Default function for calculating maximum slices, which simply assumes
    space for input and output only is required, both with the same datatype,
    and no temporaries.
    """
    slices_max = available_memory // int((np.prod(non_slice_dims_shape) + np.prod(non_slice_dims_shape)) * dtype.itemsize)
    output_dims = non_slice_dims_shape
    return (slices_max, dtype, output_dims)

def calc_max_slices_single_pattern_default(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    available_memory: int,
    **kwargs
) -> Tuple[int, np.dtype, Tuple[int, int]]:
    """
    Default function for calculating maximum slices, which simply assumes
    space for input and output only is required, both with the same datatype,
    and no temporaries.
    """

    return calc_max_slices_default(0,
                                   non_slice_dims_shape,
                                   dtype,
                                   available_memory,
                                   **kwargs)

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

    This is a convenience version for sinogram pattern.

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
    **other: dict, optional
        Any other keyword arguments will be added to the meta info as a dictionary.
    """

    def _calc_max_slices(
        slice_dim: int,
        non_slice_dims_shape: Tuple[int, int],
        dtype: np.dtype,
        available_memory: int,
        **kwargs,
    ) -> Tuple[int, np.dtype, Tuple[int, int]]:
        return calc_max_slices(non_slice_dims_shape,
                               dtype,
                               available_memory,
                               **kwargs)

    return method(_calc_max_slices,
                  cpuonly,
                  cpugpu,
                  **others,
                  pattern="sinogram")


def method_proj(
    calc_max_slices: MemorySinglePattern = calc_max_slices_single_pattern_default,
    cpuonly=False,
    cpugpu=False,
    **others,
):
    """
    Decorator for exported tomography methods, annotating the method with
    a function to calculate the memory requirements as well as other properties.

    This is a convenience version for projection pattern.

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
    **other: dict, optional
        Any other keyword arguments will be added to the meta info as a dictionary.
    """

    def _calc_max_slices(
        slice_dim: int,
        non_slice_dims_shape: Tuple[int, int],
        dtype: np.dtype,
        available_memory: int,
        **others,
    ) -> Tuple[int, np.dtype, Tuple[int, int]]:
        return calc_max_slices(non_slice_dims_shape,
                               dtype,
                               available_memory,
                               **others)

    return method(_calc_max_slices,
                  cpuonly,
                  cpugpu,
                  **others,
                  pattern="projection")


method_all = method
