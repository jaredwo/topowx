'''
Created on Nov 20, 2012

@author: jared.oyler
'''
import numpy as np

class PCA(object):
    '''
    classdocs
    '''


    def __init__(self,a):
        '''
        Constructor
        '''
        n, m = a.shape
        if n<m:
            raise RuntimeError('we assume data in a is organized with numrows>numcols')

        self.numrows, self.numcols = n, m
        self.mu = a.mean(axis=0)
        self.sigma = a.std(axis=0)

        a = self.center(a)

        self.a = a

        U, s, Vh = np.linalg.svd(a, full_matrices=False)


        Y = np.dot(Vh, a.T).T

        vars = s**2/float(len(s))
        self.fracs = vars/vars.sum()


        self.Wt = Vh
        self.Y = Y