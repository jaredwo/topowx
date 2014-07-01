'''
Utility functions performing basic principal component analysis
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