'''
Utility functions for optimizing netCDF dataset chunking.

Copyright 2014, 2015 Jared Oyler except chunk_shape_3D and chunk_shape 
functions and associated private functions written by Russ Rew, NCAR/UCAR. See:
http://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes
http://www.unidata.ucar.edu/staff/russ/public/chunk_shape_3D.py.

This file is part of TopoWx.

TopoWx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TopoWx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.
'''

__all__ = ['chunk_shape_3D', 'chunk_shape', 'chunk_cache_slots', 'set_chunk_cache_params']

import math
import operator
import numpy as np

def _binlist(n, width=0):
    """Return list of bits that represent a non-negative integer.

    n      -- non-negative integer
    width  -- number of bits in returned zero-filled list (default 0)
    """
    return map(int, list(bin(n)[2:].zfill(width)))

def _numVals(shape):
    """Return number of values in chunk of specified shape, given by a list of dimension lengths.

    shape -- list of variable dimension sizes"""
    if(len(shape) == 0):
        return 1
    return reduce(operator.mul, shape)

def _perturbShape(shape, onbits):
    """Return shape perturbed by adding 1 to elements corresponding to 1 bits in onbits

    shape  -- list of variable dimension sizes
    onbits -- non-negative integer less than 2**len(shape)
    """
    return map(sum, zip(shape, _binlist(onbits, len(shape))))

def chunk_shape_3D(varShape, valSize=4, chunkSize=4096):
    """
    Return a 'good shape' for a 3D variable, assuming balanced 1D, 2D access

    varShape  -- length 3 list of variable dimension sizes
    valSize   -- size of each data value, in bytes (default 4)
    chunkSize -- maximum chunksize desired, in bytes (default 4096)

    Returns integer chunk lengths of a chunk shape that provides
    balanced access of 1D subsets and 2D subsets of a netCDF or HDF5
    variable var with shape (T, X, Y), where the 1D subsets are of the
    form var[:,x,y] and the 2D slices are of the form var[t,:,:],
    typically 1D time series and 2D spatial slices.  'Good shape' for
    chunks means that the number of chunks accessed to read either
    kind of 1D or 2D subset is approximately equal, and the size of
    each chunk (uncompressed) is no more than chunkSize, which is
    often a disk block size.
    """

    rank = 3  # this is a special case of n-dimensional function chunk_shape
    chunkVals = chunkSize / float(valSize)  # ideal number of values in a chunk
    numChunks = varShape[0] * varShape[1] * varShape[2] / chunkVals  # ideal number of chunks
    axisChunks = numChunks ** 0.25  # ideal number of chunks along each 2D axis
    cFloor = []  # will be first estimate of good chunk shape
    # cFloor  = [varShape[0] // axisChunks**2, varShape[1] // axisChunks, varShape[2] // axisChunks]
    # except that each chunk shape dimension must be at least 1
    # chunkDim = max(1.0, varShape[0] // axisChunks**2)
    if varShape[0] / axisChunks ** 2 < 1.0:
        chunkDim = 1.0
        axisChunks = axisChunks / math.sqrt(varShape[0] / axisChunks ** 2)
    else:
        chunkDim = varShape[0] // axisChunks ** 2
    cFloor.append(chunkDim)
    prod = 1.0  # factor to increase other dims if some must be increased to 1.0
    for i in range(1, rank):
        if varShape[i] / axisChunks < 1.0:
            prod *= axisChunks / varShape[i]
    for i in range(1, rank):
        if varShape[i] / axisChunks < 1.0:
            chunkDim = 1.0
        else:
            chunkDim = (prod * varShape[i]) // axisChunks
        cFloor.append(chunkDim)

    # cFloor is typically too small, (_numVals(cFloor) < chunkSize)
    # Adding 1 to each shape dim results in chunks that are too large,
    # (_numVals(cCeil) > chunkSize).  Want to just add 1 to some of the
    # axes to get as close as possible to chunkSize without exceeding
    # it.  Here we use brute force, compute _numVals(cCand) for all
    # 2**rank candidates and return the one closest to chunkSize
    # without exceeding it.
    bestChunkSize = 0
    cBest = cFloor
    for i in range(8):
        # cCand = map(sum,zip(cFloor, _binlist(i, rank)))
        cCand = _perturbShape(cFloor, i)
        thisChunkSize = valSize * _numVals(cCand)
        if bestChunkSize < thisChunkSize <= chunkSize:
            bestChunkSize = thisChunkSize
            cBest = list(cCand)  # make a copy of best candidate so far
    return map(int, cBest)

def chunk_shape(varShape, valSize=4, chunkSize=4096):
    """
    Return a good chunk shape for an n-dimensional variable, assuming balanced 1D/(n-1)D access

    varShape  -- list of variable dimension sizes
    chunkSize -- maximum chunksize desired, in bytes (default 4096)
    valSize   -- size of each data value, in bytes (default 4)
    
    Returns integer chunk lengths of a chunk shape that provides
    balanced access of 1D subsets and (n-1)D subsets of a netCDF or
    HDF5 variable var with shape (T, X, Y, ...), where the 1D subsets
    are of the form var[:, x, y, ...] and the (n-1)-dimensional slices
    are of the form var[t, :, :, ...], typically a 1D time series and
    (n-1)D spatial slices.  'Good shape' for chunks means that the
    number of chunks accessed to read either kind of 1D or (n-1)D
    subset is approximately equal, and the size of each chunk
    (uncompressed) is no more than chunkSize, which is often a disk
    block size.
    """

    rank = len(varShape)
    chunkVals = chunkSize / float(valSize)  # ideal number of values in a chunk
    numChunks = _numVals(varShape) / chunkVals  # ideal number of chunks
    axisChunks = numChunks ** (1.0 / (rank + 1))  # ideal number of chunks along each (n-1)D axis
    cFloor = []  # will be first estimate of good chunk shape
    # each chunk shape dimension must be at least 1
    if varShape[0] / axisChunks ** (rank - 1) < 1.0:
        chunkDim = 1.0
        axisChunks = axisChunks / (varShape[0] / axisChunks ** (rank - 1)) ** (1.0 / (rank - 1))
    else:
        chunkDim = varShape[0] // axisChunks ** 2
    chunkDim = max(1.0, varShape[0] // axisChunks ** (rank - 1))
    cFloor.append(chunkDim)
    prod = 1.0  # factor to increase other dims if some must be increased to 1.0
    for i in range(1, rank):
        if varShape[i] / axisChunks < 1.0:
            prod *= axisChunks / varShape[i]
    for i in range(1, rank):
        if varShape[i] / axisChunks < 1.0:
            chunkDim = 1.0
        else:
            chunkDim = (prod * varShape[i]) // axisChunks
        cFloor.append(chunkDim)
    
    # cFloor is typically too small, (numVals(cFloor) < chunkSize)
    # Adding 1 to each shape dim results in chunks that are too large,
    # (numVals(cCeil) > chunkSize).  Want to just add 1 to some of the
    # axes to get as close as possible to chunkSize without exceeding
    # it.  Here we use brute force, compute numVals(cCand) for all
    # 2**rank candidates and return the one closest to chunkSize
    # without exceeding it (impractical for large ranks).
    bestChunkSize = 0
    cBest = cFloor
    for i in range(2 ** rank):
        cCand = _perturbShape(cFloor, i)
        thisChunkSize = valSize * _numVals(cCand)
        if bestChunkSize < thisChunkSize <= chunkSize:
            bestChunkSize = thisChunkSize
            cBest = list(cCand)  # make a copy of best candidate so far
    return map(int, cBest)

def set_chunk_cache_params(cache_size, nc_var, vcc_preemption=0):
    '''
    Set optimal netCDF chunk cache parameters on a netCDF4.Variable
    based on a desired cache size. If netCDF4.Variable storage is
    contiguous and not chunked, this method will do nothing.
    
    Parameters
    ----------
    cache_size : int
        Desired cache size in bytes.
    nc_var : netCDF4.Variable
        A netCDF4.Variable object on which to set cache parameters
    vcc_preemption : float from 0.0 to 1.0, default 0
        Chunk cache preemption. See discussion of "w0"
        parameter: http://www.hdfgroup.org/HDF5/doc/Advanced/Chunking/
    '''
    
    if nc_var.chunking() != 'contiguous':
    
        nslots = chunk_cache_slots(cache_size, nc_var.chunking(),
                                   nc_var.datatype.itemsize)
            
        nc_var.set_var_chunk_cache(cache_size, nslots, vcc_preemption)

def chunk_cache_slots(cache_size, chunk_shape, val_size):
    '''
    Get optimal number of netCDF cache slots. Number of slots
    should be a prime number 100x greater than the number of
    chunks that can fit in the cache. 
    See: http://www.hdfgroup.org/HDF5/doc/Advanced/Chunking/
    
    Parameters
    ----------
    cache_size : int
        Cache size in bytes.
    chunk_shape : tuple of ints
        Shape of the chunks
    val_size : int
        Size of each data value in bytes
    
    Returns
    ----------
    nslots : ndarray
        Optimal number of netCDF cache slots    
    '''
    
    nbytes = np.float(val_size)
    chunk_size = np.prod(chunk_shape) * nbytes
    
    nchunks_cache = np.floor(cache_size / chunk_size)
    nslots = _nearest_prime(nchunks_cache * 100)
    
    return nslots
    
def _is_prime(num):
    
    num = np.float(num)
    
    for j in np.arange(2, np.floor(np.sqrt(num)) + 1):
    
        if (num % j) == 0:
    
            return False
    
    return True

def _nearest_prime(num):
    
    if num == 2:
        return np.int(num)
    
    if num % 2 == 0:
        a_num = num + 1
    else:
        a_num = num
    
    while 1:
        
        if _is_prime(a_num):
            return np.int(a_num)
        else:
            a_num += 2
     
    
