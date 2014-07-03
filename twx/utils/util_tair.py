'''
Utilities for manipulating time series of temperature.
'''
__all__ = ['TairAggregate']

import numpy as np
from twx.utils import YEAR, MONTH, get_mth_metadata

class TairAggregate():
    '''
    A class for aggregating daily temperature data
    '''

    def __init__(self, days):
        '''
        Parameters
        ----------
        days : ndarray
            A structured array of date information
            for the time period of interest produced
            from twx.utils.get_days_metadata_* methods
        '''

        u_yrs = np.unique(days[YEAR])

        self.yr_mths_masks = []

        for a_yr in u_yrs:

            for a_mth in np.arange(1, 13):

                self.yr_mths_masks.append(np.nonzero(np.logical_and(days[YEAR] == a_yr, days[MONTH] == a_mth))[0])

        self.days = days
        self.yr_mths = get_mth_metadata(u_yrs[0], u_yrs[-1])

        self.yr_masks = []

        for a_yr in u_yrs:

            self.yr_masks.append(np.nonzero(self.yr_mths[YEAR] == a_yr)[0])

        self.u_yrs = u_yrs
        
        self.norm_yrs = (1981,2010)
        self.norm_yr_mth_mask = np.nonzero(np.logical_and(self.yr_mths[YEAR] >= self.norm_yrs[0],
                                                          self.yr_mths[YEAR] <= self.norm_yrs[-1]))[0]


    def daily_to_mthly(self, tair, max_miss=9):
        '''
        Aggregate daily temperature data to monthly means.
        
        Parameters
        ----------
        tair : MaskedArray
            Time series of temperature for time period
            of inserest as numpy MaskedArray. Can be of any shape, 
            but first axis must be the time dimension.
        max_miss : int, optional
            The maximum # of missing daily observations in a month
            for a monthly mean to be calculated. If # of missing 
            observations for a month is > max_miss, the month's mean
            will be marked as missing. Set max_miss to None if there
            should not be a max_miss threshold.
        
        Returns
        -------
        tair_mthly : MaskedArray
            Time series of monthly temperature for time period of 
            interest. Missing values are masked.
        n_miss : ndarray
            Number of missing daily observations in each month
        '''

        tair_mthly = np.ma.array([np.ma.mean(np.ma.take(tair, a_mask, axis=0), axis=0, dtype=np.float) for a_mask in self.yr_mths_masks])
        
        if np.ma.is_masked(tair_mthly):
        
            n_miss = np.array([np.sum(np.ma.take(tair.mask, a_mask, axis=0), axis=0) for a_mask in self.yr_mths_masks])
        
        else:
            
            n_miss = np.zeros_like(tair_mthly)

        if max_miss is not None:

            tair_mthly[n_miss > max_miss] = np.ma.masked

        return tair_mthly, n_miss
    
    def daily_to_mthly_norms(self,tair,start_norm_yr=1981,end_norm_yr=2010,max_miss=9):
        '''
        Aggregate daily temperature data to monthly normals.
        
        Parameters
        ----------
        tair : MaskedArray
            Time series of temperature for time period
            of interest as numpy MaskedArray. Can be of any shape, 
            but first axis must be the time dimension.
        start_norm_yr : int, optional
            The start year for the normals.
        end_norm_yr : int, optional
            The end year for the normals
        max_miss : int, optional
            The maximum # of missing daily observations in a month
            for a monthly mean to be calculated. If # of missing 
            observations for a month is > max_miss, the month's mean
            will be marked as missing. Set max_miss to None if there
            should not be a max_miss threshold.
        
        Returns
        -------
        tair_mth_norms : MaskedArray
            Monthly normals for time period of interest. First axis
            is of size 12.
        '''
        
        if start_norm_yr == self.norm_yrs[0] and end_norm_yr == self.norm_yrs[-1]:
            yr_mask = self.norm_yr_mth_mask
        else:
            yr_mask = np.nonzero(np.logical_and(self.yr_mths[YEAR] >= start_norm_yr,self.yr_mths[YEAR] <= end_norm_yr))[0]   
            self.norm_yrs = (start_norm_yr,end_norm_yr)
            self.norm_yr_mth_mask = yr_mask
        
        tair_mthly = self.daily_to_mthly(tair,max_miss=max_miss)[0]
        tair_mthly = np.take(tair_mthly, yr_mask, axis=0)
        
        mths = np.take(self.yr_mths[MONTH], yr_mask)
                
        tair_mth_norms = []
        
        for mth in np.arange(1,13):
            
            a_mth_mask = np.nonzero(mths==mth)[0]            
            tair_mth_norms.append(np.ma.mean(np.take(tair_mthly, a_mth_mask,axis=0),axis=0))
            
        tair_mth_norms = np.ma.masked_array(tair_mth_norms)
        
        return tair_mth_norms
