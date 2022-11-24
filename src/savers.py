from pathlib import Path
from typing import Tuple

import numpy
import h5py
from mpi4py import MPI
from mpi4py.MPI import Comm

from utils import print_once


def intermediate_dataset(data: numpy.ndarray, run_out_dir: Path,
                         filename: str, comm: Comm) -> None:
    """Save an intermediate dataset as an hdf file.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be written.
    run_out_dir : Path
        The directory to write the file to.
    filename : str
        The name of the hdf file to write.
    comm : Comm
        The MPI communicator to use.
    """
    (vert_slices, x, y) = data.shape
    chunks = (1, x, y)
    print_once(f"Saving intermediate file: {filename}", comm)
    _save_dataset(run_out_dir, filename, data, 1, chunks=chunks, comm=comm)


def _save_dataset(
    out_folder: str,
    file_name: str,
    data: numpy.ndarray,
    slice_dim: int = 1,
    chunks: Tuple=(150, 150, 10),
    path: str="/data",
    comm: MPI.Comm=MPI.COMM_WORLD,
) -> None:
    """Save dataset in parallel.
    Parameters
    ----------
    out_folder : str
        Path to output folder.
    file_name : str
        Name of file to save dataset in.
    data : numpy.ndarray
        Data to save to file.
    slice_dim : int
        Where data has been parallelized (split into blocks, each of which is
        given to an MPI process), provide the dimension along which the data was
        sliced so it can be pieced together again.
    chunks : Tuple
        Specify how the data should be chunked when saved.
    path : str
        Path to dataset within the file.
    comm : MPI.Comm
        MPI communicator object.
    """
    shape = _get_data_shape(data, slice_dim - 1, comm)
    dtype = data.dtype
    with h5py.File(f"{out_folder}/{file_name}", "a", driver="mpio", comm=comm) as f:
        dataset = f.create_dataset(path, shape, dtype, chunks=chunks)
        _save_data_parallel(dataset, data, slice_dim)


def _save_data_parallel(
    dataset: h5py.Dataset,
    data: numpy.ndarray,
    slice_dim: int,
    comm: MPI.Comm=MPI.COMM_WORLD,
) -> None:
    """Save data to dataset in parallel.
    Parameters
    ----------
    dataset : h5py.Dataset
        Dataset to save data to.
    data : numpy.ndarray
        Data to save to dataset.
    slice_dim : int
        Where data has been parallelized (split into blocks, each of which is
        given to an MPI process), provide the dimension along which the data was
        sliced so it can be pieced together again.
    comm : MPI.Comm
        MPI communicator object.
    """
    rank = comm.rank
    nproc = comm.size
    length = dataset.shape[slice_dim - 1]
    i0 = round((length / nproc) * rank)
    i1 = round((length / nproc) * (rank + 1))
    if slice_dim == 1:
        dataset[i0:i1] = data[...]
    elif slice_dim == 2:
        dataset[:, i0:i1] = data[...]
    elif slice_dim == 3:
        dataset[:, :, i0:i1] = data[...]


def _get_data_shape(data: numpy.ndarray, dim: int,
                   comm: MPI.Comm=MPI.COMM_WORLD) -> Tuple:
    """Gets the shape of a distributed dataset.
    Parameters
    ----------
    data : ndarray
        The process data.
    dim : int
        The dimension in which to get the shape.
    comm : MPI.Comm
        The MPI communicator.
    Returns
    -------
    Tuple
        The shape of the given distributed dataset.
    """
    shape = list(data.shape)
    lengths = comm.gather(shape[dim], 0)
    lengths = comm.bcast(lengths, 0)
    shape[dim] = sum(lengths)
    shape = tuple(shape)
    return shape
