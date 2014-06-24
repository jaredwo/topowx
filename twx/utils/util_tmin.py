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
        '''

        tair_mthly = np.ma.array([np.ma.mean(np.ma.take(tair, a_mask, axis=0), axis=0, dtype=np.float) for a_mask in self.yr_mths_masks])

        if max_miss is not None:
            n_miss = np.array([np.sum(np.ma.take(tair.mask, a_mask, axis=0), axis=0) for a_mask in self.yr_mths_masks])
            tair_mthly[n_miss > max_miss] = np.ma.masked

        return tair_mthly
