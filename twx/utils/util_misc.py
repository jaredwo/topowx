'''
Copyright 2014, Jared Oyler.

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
import numpy as np
import re
import os
import errno
import xarray as xr


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

def read_params(fname):
    d = {}
    iscmt = re.compile("^(\s)*#")
    tmp = open(fname, "r").readlines()
    for l in tmp:
        if iscmt.search(l): continue
        t = l.strip().split('=')
        if len(t) == 2 and t[0] != '' and t[1] != '':
            d[t[0].strip()] = t[1].strip()
    return d

def read_csv(filename, dtype, separator=','):
    """ Read a file with an arbitrary number of columns.
        The type of data in each column is arbitrary
        It will be cast to the given dtype at runtime
    """
    cast = np.cast
    data = [[] for dummy in xrange(len(dtype))]
    
    afile = open(filename, 'r')
    #skip header
    afile.readline()
    
    for line in afile.readlines():
        fields = line.strip().split(separator)
        for i, number in enumerate(fields):
            data[i].append(number)
    for i in xrange(len(dtype)):
        data[i] = cast[dtype[i]](data[i])

    return np.rec.array(data, dtype=dtype)

def split_list(alist, wanted_parts=10):
    '''
    From http://stackoverflow.com/questions/752308/split-array-into-smaller-arrays
    '''
    
    length = len(alist)
    return [ np.array(alist[i*length // wanted_parts: (i+1)*length // wanted_parts]) for i in range(wanted_parts) ]

def get_val_classes(vals,num_classes):
    '''
    Group a list of values into groups of equal size. 
    Returns an array containing the class of each value.
    '''    
    
    indices = np.arange(vals.size)
    class_array = np.ones(vals.size)*-1
    splits = split_list(indices,num_classes)
    
    class_num = 0
    for split in splits:
        class_array[np.in1d(indices, split, assume_unique=True)] = class_num
        class_num+=1
    return class_array 

def mkdir_p(path):
    '''
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise 
        
def read_xarray_netcdfs(fpaths, dim, transform_func=None, engine=None):
    '''
    Based off example at: http://xarray.pydata.org/en/stable/io.html#combining-multiple-files
    '''
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path, engine=engine) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    datasets = [process_one_path(p) for p in fpaths]
    combined = xr.concat(datasets, dim)
    return combined