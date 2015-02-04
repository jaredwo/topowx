'''
Utility functions for assessing model performance.

Copyright 2014, 2015 Jared Oyler.

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


def calc_ioa_d1(o,p):
    '''
    Calculate the index of agreement d1 between observed and predicted values.
    d1 ranges from 0.0 - 1.0 with values closer to 1.0 indicating better
    performance. 
    
    References:
    
    Willmott, C. J., S. G. Ackleson, R. E. Davis, J. J. Feddema, K. M. Klink,
    D. R. Legates, J. ODonnell, and C. M. Rowe (1985), Statistics for the 
    evaluation and comparison of models, J. Geophys. Res., 90(C5), 8995-9005,
    doi:10.1029/JC090iC05p08995.

    Legates, D. R., and G. McCabe (1999), Evaluating the use of goodness-of-fit 
    Measures in hydrologic and hydroclimatic model validation, Water Resour. Res.,
    35(1), PP. 233-241, doi:199910.1029/1998WR900018.

    Willmott, C. J., S. M. Robeson, and K. Matsuura (2012), A refined index of model
    performance, Int. J. Climatol., 32(13), 2088-2094, doi:10.1002/joc.2419.
    
    
    Parameters
    ----------
    o : ndarray
        Array of observations
    p : ndarray
        Array of predictions
        
    Returns
    -------
    d1 : float
        The d1 index of agreement
    '''
    
    o_mean = np.mean(o)
    denom = np.sum(np.abs(p - o_mean) + np.abs(o - o_mean))
        
    d1 = 1.0 - (np.sum(np.abs(p - o)) / denom)
    
    return d1
