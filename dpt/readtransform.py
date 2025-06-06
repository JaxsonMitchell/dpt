"""
Name: Jaxson Mitchell
Date: 7/1/2024

H5 File IO functions used within the transformations for data storage. 
As well as the functions to read and write DPT, STPT, ISTPT, and VVT objects
"""

import h5py
import numpy as np


def makeFile(fileName: str) -> h5py.File:
    """
    input:
        fileName (str) - file name

    output:
        f (h5py.File) - The opened h5file
    """
    f = h5py.File(f"{fileName}.h5", "a")
    return f


def readFile(fileName: str) -> h5py.File:
    """
    input:
        fileName (str) - file name

    output:
        f (h5py.File) - The opened h5file
    """
    f = h5py.File(f"{fileName}.h5", "r")
    return f


def closeFile(f: h5py.File) -> None:
    """
    input:
        f (h5py.File) - File to be closed

    output:
        None (None)
    """
    f.close()
    return None


def createGroup(f: h5py.File, groupName: str) -> h5py.Group:
    """
    input:
        f (h5py.File) - The HDF5 file
        groupName (str) - Name of the group to create

    output:
        g (h5py.Group) - The created group
    """
    g = f.create_group(groupName)
    return g


def writeDataWithMetadata(group: h5py.Group, dataName: str, data: np.ndarray, metadata: dict) -> None:
    """
    input:
        group (h5py.Group) - The group to write data into
        dataName (str) - Name of the dataset
        data (np.ndarray) - The data to write
        metadata (dict) - Dictionary of metadata

    output:
        None (None)
    """
    dset = group.create_dataset(dataName, data=data)
    for key, value in metadata.items():
        dset.attrs[key] = value
    return None


def readDataset(group: h5py.Group, dataName: str, slice_indices: tuple = None) -> tuple:
    """
    input:
        group (h5py.Group) - The group to read data from
        dataName (str) - Name of the dataset
        slice_indices (tuple) - Optional slice indices to read a subset of the data

    output:
        data (np.ndarray) - The read data
        metadata (dict) - Dictionary of metadata
    """
    dset = group[dataName]
    if slice_indices:
        data = dset[slice_indices]
    else:
        data = dset[()]
    metadata = {key: dset.attrs[key] for key in dset.attrs}
    return data, metadata