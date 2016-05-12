'''
Utilities for analyzing and producing station period-of-record information.

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
__all__ = ['build_por_mask']

from twx.utils import ymdL
import numpy as np
import pandas as pd


def _build_a_por_mask(obs_cnts, min_por_yrs):
    
    days_in_mth = np.array([d.days_in_month for d in
                            pd.date_range('2015-01', '2015-12', freq='MS')])
    
    nmin = days_in_mth*min_por_yrs
    nmin.shape = (nmin.size,1)
    
    mask_por = obs_cnts >= nmin
    mask_por = np.sum(mask_por, axis=0) == 12

    return mask_por

def build_por_mask(ds, elems, start_date, end_date, min_por_yrs):
    '''Build a boolean mask for stations that have long enough period-of-record
    
    Requires obs_cnt_[elem]_[ymd-start]_[ymd-end] variable to be previously set
    by twx.db.add_obs_cnt
    
    Parameters
    ----------
    ds : netCDF4.Dataset
        Dataset point to station observation netCDF file
    elems : list
        List of element names for which to build the mask. A station will be
        included if has a long enough period-of-record for one or more of the
        specified elements.
    start_date : date-like
        The start date for period-of-record time period
    end_date : date-like
        The end date for period-of-record time period
    min_por_yrs : int
        The minimum period of record in years. The function tests
        whether a station has at least min_por_yrs years of data in each month.
    '''
    
    por_masks = []
    
    start_ymd = ymdL(start_date)
    end_ymd = ymdL(end_date)

    for elem in elems:

        vname = "obs_cnt_%s_%d_%d"%(elem, start_ymd, end_ymd)
        
        obs_cnts = ds[vname][:]
        
        mask_por = _build_a_por_mask(obs_cnts, min_por_yrs)
        
        por_masks.append(mask_por)

    por_masks = np.array(por_masks)

    por_mask_fnl = np.sum(por_masks, axis=0) >= 1
    
    return por_mask_fnl
    
