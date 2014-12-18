'''
Utility functions performing basic principal component analysis.

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

__all__ = ['pca_svd']

import numpy as np

def pca_svd(A,center=True,scale=False):
    '''
    Run a Principal Component Analysis (PCA) on the input matrix
    using singular value decomposition (SVD).
    
    Parameters
    ----------
    A : ndarray
        A 2-D input matrix with rows as the observations
        and columns as the variables. If performing an S-Mode PCA,
        each column is a separate time series.
    center : bool
        Mean center each column of A.
    scale : bool
        Scale each column of A by dividing each column by its
        standard deviation.
    
    Returns
    ----------
    pc_loads : ndarray
        A P*P array of principal component loadings
        where P is the number of columns in the input matrix.
    pc_scores : ndarray
        A N*P array of principal component scores where
        N is number of rows in the input matrix and P
        is the number of columns
    var_explain : ndarray
        A 1-D array of size P containing the fraction
        of variance explained by each principal component.  
    '''
    
    A = np.require(A,dtype=np.float64,requirements=['C','A','W','O'])
    
    nrows,ncols = A.shape
    
    if center:
        A = A - np.mean(A,axis=0)
    
    if scale:
        A = A/np.std(A,axis=0,ddof=1)
    
    full_matrices = True if ncols > nrows else False
     
    u,s,v = np.linalg.svd(A,full_matrices=full_matrices)
    
    s = np.square(s)/(nrows-1)
    var_explain = s/np.sum(s)
    pc_loads = v.T
    pc_scores = np.dot(A,pc_loads)
    
    return pc_loads,pc_scores,var_explain