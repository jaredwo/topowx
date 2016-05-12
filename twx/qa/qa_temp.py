'''
Quality Assurance procedures for Tmin/Tmax daily observations as described in:
Durre, I., M. J. Menne, B. E. Gleason, T. G. Houston, and R. S. Vose. 2010. 
Comprehensive Automated Quality Assurance of Daily Surface Observations. 
Journal of Applied Meteorology and Climatology 49:1615-1633.

Copyright 2014,2015, Jared Oyler.

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
from twx.utils.perf_metrics import calc_ioa_d1

__all__ = ['run_qa_all','run_qa_non_spatial','run_qa_spatial_only',
           'TWX_TO_GHCN_FLAGS_MAP','GHCN_TO_TWX_FLAGS_MAP']

from twx.utils import get_mth_str_end_dates, get_md_array, get_date_array, A_WEEK, \
YMD, DAY, YDAY, YEAR, DATE, MONTH

import numpy as np
from datetime import datetime
import calendar as cal
from twx.db import LON, LAT, STN_ID, TMIN, TMAX
from datetime import timedelta
from scipy import stats
from twx.utils import grt_circle_dist

# Constants for QA flags
QA_OK = 1
QA_MISSING = 2
QA_NAUGHT = 3
DUP = 25
QA_DUP_YEAR = 4
QA_DUP_MONTH = 5
QA_DUP_YEAR_MONTH = 6
QA_DUP_WITHIN_MONTH = 7
QA_IMPOSS_VALUE = 8
QA_STREAK = 9
QA_GAP = 10
QA_INTERNAL_INCONSIST = 11
QA_LAGRANGE_INCONSIST = 12
QA_SPIKE_DIP = 13
QA_MEGA_INCONSIST = 14
QA_CLIM_OUTLIER = 15
QA_SPATIAL_REGRESS = 16
QA_SPATIAL_CORROB = 17
QA_MEGA_INCONSIST = 18

# World records for daily Tmax and Tmin in degrees C
TMAX_RECORD = 57.7
TMIN_RECORD = -89.4

# Constants for spatial regression/collab checks
NGH_RADIUS = 75.0
NGH_CORR = 0.8
NGH_RESID_CUTOFF = 8.0
NGH_RESID_STD_CUTOFF = 4.0
ANOMALY_CUTOFF = 10.0
MIN_DAYS_MTH_WINDOW = 40
MIN_NGHS = 3
MAX_NGHS = 7
NGH_STNS_ID = "NGH_STNS_ID"
NGH_STNS_MASK_OVERLAP = "NGH_STNS_MASK_OVERLAP"
NGH_STNS_WGHTS = "NGH_STNS_WGHTS"
NGH_STNS_MODEL = "NGH_STNS_MODEL"
NGH_STNS_OBS = "NGH_STNS_OBS"

# Constants for building Tmin/Tmax normals
MONTHS = np.arange(1, 13)
DATES_366 = get_date_array(datetime(2004, 1, 1), datetime(2004, 12, 31))
DATES_365 = get_date_array(datetime(2003, 1, 1), datetime(2003, 12, 31))
MIN_NORM_VALUES = 100

TWX_TO_GHCN_FLAGS_MAP = { QA_OK:"",
                          QA_MISSING:"",
                          DUP:"D",
                          QA_DUP_YEAR:"D",
                          QA_DUP_MONTH:"D",
                          QA_DUP_YEAR_MONTH:"D",
                          QA_DUP_WITHIN_MONTH:"D",
                          QA_GAP:"G",
                          QA_INTERNAL_INCONSIST:"I",
                          QA_STREAK:"K",
                          QA_MEGA_INCONSIST:"M",
                          QA_NAUGHT:"N",
                          QA_CLIM_OUTLIER:"O",
                          QA_LAGRANGE_INCONSIST:"R",
                          QA_SPATIAL_REGRESS:"S",
                          QA_SPATIAL_CORROB:"S",
                          QA_SPIKE_DIP:"T",
                          QA_IMPOSS_VALUE:"X"}

GHCN_TO_TWX_FLAGS_MAP = {"":QA_OK,
                         "D":DUP,
                         "G":QA_GAP,
                         "I":QA_INTERNAL_INCONSIST,
                         "K":QA_STREAK,
                         "M":QA_MEGA_INCONSIST,
                         "N":QA_NAUGHT,
                         "O":QA_CLIM_OUTLIER,
                         "R":QA_LAGRANGE_INCONSIST,
                         "S":QA_SPATIAL_REGRESS,
                         "T":QA_SPIKE_DIP,
                         "X":QA_IMPOSS_VALUE}


def run_qa_all(stn, stn_da, tmin, tmax, days):
    '''
    Run all quality assurance checks in the order specified by:
    
        Durre, I., M. J. Menne, B. E. Gleason, T. G. Houston, and R. S. Vose. 2010. 
        Comprehensive Automated Quality Assurance of Daily Surface Observations. 
        Journal of Applied Meteorology and Climatology 49:1615-1633.
        
    Parameters
    ----------
    stn : str
        A station record from a structured ndarray containing at least the
        following fields: STN_ID,LON,LAT
    stn_da : twx.db.StationDataDb
        A StationDataDb object pointing to a netCDF station database
    tmin : ndarray
        A 1-D ndarray time series of Tmin values (Celsius).
        Use numpy.nan for missing values
    tmax : ndarray
        A 1-D ndarray  time series of Tmax values (Celsius).
        Use numpy.nan for missing values
    days : structured ndarray
        A structured ndarray from twx.utils.get_days_metadata. 
        Provides date information for each observation
        in the Tmin/Tmax time series. 
    
    Returns
    -------
    flags_tmin : ndarray
        A 1-D array of QA flags for Tmin
    flags_tmax : ndarray
        A 1-D array of QA flags for Tmax
    '''

    flags_tmin = np.ones(tmin.size)
    flags_tmax = np.ones(tmax.size)
    tmin, tmax, flags_tmin, flags_tmax = _qa_missing(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_naught(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_dup_data(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_imposs_value(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_streak(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_gap(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_clim_outlier(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_internal_inconsist(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_spike_dip(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_lagrange_inconsist(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_mega_inconsist(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_spatial_regress(stn, stn_da, tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_spatial_corrob(stn, stn_da, tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_mega_inconsist(tmin, tmax, days, flags_tmin, flags_tmax)

    return flags_tmin, flags_tmax

def run_qa_non_spatial(tmin, tmax, days):
    '''
    Run only quality assurance checks that do not require
    neighboring station data in the order specified by:
    
        Durre, I., M. J. Menne, B. E. Gleason, T. G. Houston, and R. S. Vose. 2010. 
        Comprehensive Automated Quality Assurance of Daily Surface Observations. 
        Journal of Applied Meteorology and Climatology 49:1615-1633.
        
    Parameters
    ----------
    tmin : ndarray
        A 1-D ndarray time series of Tmin values (Celsius).
        Use numpy.nan for missing values
    tmax : ndarray
        A 1-D ndarray  time series of Tmax values (Celsius).
        Use numpy.nan for missing values
    days : structured ndarray
        A structured ndarray from twx.utils.get_days_metadata. 
        Provides date information for each observation
        in the Tmin/Tmax time series. 
    
    Returns
    -------
    flags_tmin : ndarray
        A 1-D array of QA flags for Tmin
    flags_tmax : ndarray
        A 1-D array of QA flags for Tmax
    '''

    flags_tmin = np.ones(tmin.size)
    flags_tmax = np.ones(tmax.size)
    tmin, tmax, flags_tmin, flags_tmax = _qa_missing(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_naught(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_dup_data(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_imposs_value(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_streak(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_gap(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_clim_outlier(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_internal_inconsist(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_spike_dip(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_lagrange_inconsist(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_mega_inconsist(tmin, tmax, days, flags_tmin, flags_tmax)

    return flags_tmin, flags_tmax

def run_qa_spatial_only(stn, stn_da, tmin, tmax, days):
    '''
    Run only spatial quality assurance checks in the order specified by:
    
        Durre, I., M. J. Menne, B. E. Gleason, T. G. Houston, and R. S. Vose. 2010. 
        Comprehensive Automated Quality Assurance of Daily Surface Observations. 
        Journal of Applied Meteorology and Climatology 49:1615-1633.
        
    Parameters
    ----------
    stn : str
        A station record from a structured ndarray containing at least the
        following fields: STN_ID,LON,LAT
    stn_da : twx.db.StationDataDb
        A StationDataDb object pointing to a netCDF station database
    tmin : ndarray
        A 1-D ndarray time series of Tmin values (Celsius).
        Use numpy.nan for missing values
    tmax : ndarray
        A 1-D ndarray  time series of Tmax values (Celsius).
        Use numpy.nan for missing values
    days : structured ndarray
        A structured ndarray from twx.utils.get_days_metadata. 
        Provides date information for each observation
        in the Tmin/Tmax time series. 
    
    Returns
    -------
    flags_tmin : ndarray
        A 1-D array of QA flags for Tmin
    flags_tmax : ndarray
        A 1-D array of QA flags for Tmax
    '''
    flags_tmin = np.ones(tmin.size)
    flags_tmax = np.ones(tmax.size)
    tmin, tmax, flags_tmin, flags_tmax = _qa_missing(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_spatial_regress(stn, stn_da, tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_spatial_corrob(stn, stn_da, tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_mega_inconsist(tmin, tmax, days, flags_tmin, flags_tmax)

    return flags_tmin, flags_tmax


def _qa_missing(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Flag observations that are missing
    '''

    miss_tmin = np.isnan(tmin)
    miss_tmax = np.isnan(tmax)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, miss_tmin, miss_tmax, QA_MISSING)

def _qa_naught(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Flag observations with erroneous zeros
    US stations: Tmax and Tmin = -17.8C (0F)
    Non-US: Tmax and Tmin = 0 C
    '''
    mask_us = np.logical_and(np.round(tmin, 1) == -17.8, np.round(tmax, 1) == -17.8)
    mask_nonus = np.logical_and(tmin == 0.0, tmax == 0.0)
    mask_final = np.logical_or(mask_us, mask_nonus)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_final, mask_final, QA_NAUGHT)

def _qa_dup_data(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Run all duplicate observation QA checks
    '''

    tmin, tmax, flags_tmin, flags_tmax = _qa_dup_year(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_dup_year_month(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_dup_month(tmin, tmax, days, flags_tmin, flags_tmax)
    tmin, tmax, flags_tmin, flags_tmax = _qa_dup_within_month(tmin, tmax, days, flags_tmin, flags_tmax)

    return tmin, tmax, flags_tmin, flags_tmax

def _qa_dup_within_month(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Flag all days in months with 10 or more days that Tmax=Tmin
    '''

    mask_dup = np.zeros(days[DATE].size, dtype=np.bool_)

    yrs = np.unique(days[YEAR])
    mths = np.unique(days[MONTH])


    for yr in yrs:

        mask_yr = days[YEAR] == yr

        tmin_yr = tmin[mask_yr]
        tmax_yr = tmax[mask_yr]

        mths_yr = days[MONTH][mask_yr]

        for mth in mths:

            tmin_yr_mth = tmin_yr[mths_yr == mth]
            tmax_yr_mth = tmax_yr[mths_yr == mth]

            nan_mask = np.logical_and(_not_nan(tmin_yr_mth), _not_nan(tmax_yr_mth))
            num_dups = np.nonzero(np.logical_and(nan_mask, tmin_yr_mth == tmax_yr_mth))[0].size

            if num_dups >= 10:
                mask_dup = np.logical_or(mask_dup, np.logical_and(mask_yr, days[MONTH] == mth))


    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_dup, mask_dup, QA_DUP_WITHIN_MONTH)


def _qa_dup_year_month(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Flag months in a single year that have duplicate values.
    '''

    yrs = np.unique(days[YEAR])
    yr_nums = np.arange(yrs.size)
    mask_tmin = np.zeros(days[DATE].size, dtype=np.bool_)
    mask_tmax = np.zeros(days[DATE].size, dtype=np.bool_)

    mths = np.unique(days[MONTH])
    mth_nums = np.arange(mths.size)

    for x in yr_nums:

        yr_mask = days[YEAR] == yrs[x]

        tmin_yr = tmin[yr_mask]
        tmax_yr = tmax[yr_mask]

        mths_yr = days[MONTH][yr_mask]

        for i in mth_nums:

            mth1 = mths[i]
            mth1_mask = mths_yr == mth1

            tmin_yr_mth1 = tmin_yr[mth1_mask]
            tmax_yr_mth1 = tmax_yr[mth1_mask]

            if not (mth1 == np.max(mth_nums)):

                sub_mths = np.arange(i + 1, np.max(mth_nums) + 1)

                for z in sub_mths:

                    mth2 = mths[z]
                    mth2_mask = mths_yr == mth2

                    tmin_yr_mth2 = tmin_yr[mth2_mask]
                    tmax_yr_mth2 = tmax_yr[mth2_mask]

                    if _is_dup_series(tmin_yr_mth1, tmin_yr_mth2):
                        yr_month_mask = np.logical_and(yr_mask, np.logical_or(days[MONTH] == mth1, days[MONTH] == mth2))
                        mask_tmin = np.logical_or(mask_tmin, yr_month_mask)

                    if _is_dup_series(tmax_yr_mth1, tmax_yr_mth2):
                        yr_month_mask = np.logical_and(yr_mask, np.logical_or(days[MONTH] == mth1, days[MONTH] == mth2))
                        mask_tmax = np.logical_or(mask_tmax, yr_month_mask)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_DUP_YEAR_MONTH)


def _qa_dup_month(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Flag duplicate observations for same calendar month in different years
    '''

    yrs = np.unique(days[YEAR])
    yr_nums = np.arange(yrs.size)
    mask_tmin = np.zeros(days[DATE].size, dtype=np.bool_)
    mask_tmax = np.zeros(days[DATE].size, dtype=np.bool_)

    mths = np.unique(days[MONTH])

    for x in yr_nums:

        yr1_mask = days[YEAR] == yrs[x]

        tmin_yr1 = tmin[yr1_mask]
        tmax_yr1 = tmax[yr1_mask]

        mths_yr1 = days[MONTH][yr1_mask]

        if not (x == np.max(yr_nums)):

            sub_yr_nums = np.arange(x + 1, np.max(yr_nums) + 1)

            for i in sub_yr_nums:

                yr2_mask = days[YEAR] == yrs[i]

                tmin_yr2 = tmin[yr2_mask]
                tmax_yr2 = tmax[yr2_mask]

                mths_yr2 = days[MONTH][yr2_mask]

                for month in mths:

                    mask_mth_yr1 = mths_yr1 == month
                    mask_mth_yr2 = mths_yr2 == month

                    tmin_yr1_mth = tmin_yr1[mask_mth_yr1]
                    tmax_yr1_mth = tmax_yr1[mask_mth_yr1]

                    tmin_yr2_mth = tmin_yr2[mask_mth_yr2]
                    tmax_yr2_mth = tmax_yr2[mask_mth_yr2]

                    if _is_dup_series(tmin_yr1_mth, tmin_yr2_mth):
                        yr_month_mask = np.logical_and(np.logical_or(yr1_mask, yr2_mask), days[MONTH] == month)
                        mask_tmin = np.logical_or(mask_tmin, yr_month_mask)
                    if _is_dup_series(tmax_yr1_mth, tmax_yr2_mth):
                        yr_month_mask = np.logical_and(np.logical_or(yr1_mask, yr2_mask), days[MONTH] == month)
                        mask_tmax = np.logical_or(mask_tmax, yr_month_mask)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_DUP_MONTH)

def _qa_dup_year(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Flag duplicate observations between years
    '''

    yrs = np.unique(days[YEAR])
    yr_nums = np.arange(yrs.size)
    mask_tmin = np.zeros(days[DATE].size, dtype=np.bool_)
    mask_tmax = np.zeros(days[DATE].size, dtype=np.bool_)

    for x in yr_nums:

        yr1_mask = days[YEAR] == yrs[x]

        tmin_yr1 = tmin[yr1_mask]
        tmax_yr1 = tmax[yr1_mask]

        if not (x == np.max(yr_nums)):

            sub_yr_nums = np.arange(x + 1, np.max(yr_nums) + 1)

            for i in sub_yr_nums:

                yr2_mask = days[YEAR] == yrs[i]

                tmin_yr2 = tmin[yr2_mask]
                tmax_yr2 = tmax[yr2_mask]

                if _is_dup_series(tmin_yr1, tmin_yr2):
                    yr_mask = np.logical_or(yr1_mask, yr2_mask)
                    mask_tmin = np.logical_or(mask_tmin, yr_mask)

                if _is_dup_series(tmax_yr1, tmax_yr2):
                    yr_mask = np.logical_or(yr1_mask, yr2_mask)
                    mask_tmax = np.logical_or(mask_tmax, yr_mask)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_DUP_YEAR)

def _qa_imposs_value(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Flag observations that are outside the bounds of world records
    '''

    mask_tmin = np.logical_or(tmin < TMIN_RECORD, tmin > TMAX_RECORD)
    mask_tmax = np.logical_or(tmax < TMIN_RECORD, tmax > TMAX_RECORD)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_IMPOSS_VALUE)

def _qa_streak(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Flag 20 or more consecutive Tmin or Tmax observations
    '''

    mask_tmin = _identify_streaks(tmin, days)
    mask_tmax = _identify_streaks(tmax, days)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_STREAK)

def _qa_gap(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Examine frequency distributions of tmin/tmax for calendar months and flag observations in
    a distribution's tails that are unrealistically separated from the rest of the observations
    '''

    mask_tmin = np.zeros(days[DATE].size, dtype=np.bool_)
    mask_tmax = np.zeros(days[DATE].size, dtype=np.bool_)

    mths = days[MONTH]
    uniq_mths = np.unique(mths)

    for mth in uniq_mths:

        mth_mask = mths == mth

        tmin_mth = tmin[mth_mask]
        tmax_mth = tmax[mth_mask]

        tmin_min, tmin_max = _get_gap_bounds(tmin_mth)
        tmax_min, tmax_max = _get_gap_bounds(tmax_mth)

        if tmin_min is not None:
            mask_tmin[np.logical_and(mth_mask, tmin <= tmin_min)] = True
        if tmin_max is not None:
            mask_tmin[np.logical_and(mth_mask, tmin >= tmin_max)] = True
        if tmax_min is not None:
            mask_tmax[np.logical_and(mth_mask, tmax <= tmax_min)] = True
        if tmax_max is not None:
            mask_tmax[np.logical_and(mth_mask, tmax >= tmax_max)] = True

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_GAP)

def _qa_internal_inconsist(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Flag inconsistent Tmin and Tmax observations.
    Unlike the set of inconsistent checks described by Durre et al. (2010), only check for
    Tmax < Tmin on same day and do not use a 1 deg buffer. The 1 deg buffer and
    other inconsistent checks were found to have too many false positives in SNOTEL
    and RAWS data. Also do not include checks that use time of observation.
    '''

    violation_fnd = True
    while violation_fnd:

        mask_tmin = np.zeros(days[DATE].size, dtype=np.bool_)
        mask_tmax = np.zeros(days[DATE].size, dtype=np.bool_)

        x = 0
        violation_fnd = False
        violations_tmin = np.zeros(days[DATE].size)
        violations_tmax = np.zeros(days[DATE].size)

        while x < days[DATE].size - 1:

            tmin_cur = tmin[x]
            tmax_cur = tmax[x]

            tmin_nxt = tmin[x + 1]
            tmax_nxt = tmax[x + 1]

            # Check for Tmax < Tmin.
            # Removed adjacent day checks and 1 deg buffer due to too many false positives
            if not np.isnan(tmax_cur) and not np.isnan(tmin_cur) and tmax_cur < tmin_cur:
            # if not np.isnan(tmax_cur) and not np.isnan(tmin_cur) and tmax_cur < (tmin_cur - 1.0):
                violations_tmin[x] = violations_tmin[x] + 1
                violations_tmax[x] = violations_tmax[x] + 1
                violation_fnd = True

#            if not np.isnan(tmax_cur) and not np.isnan(tmin_nxt) and tmax_cur < (tmin_nxt - 1.0):
#                violations_tmax[x] = violations_tmax[x] + 1
#                violations_tmin[x+1] = violations_tmin[x+1] + 1
#                violation_fnd = True
#
#            if not np.isnan(tmin_cur) and not np.isnan(tmax_nxt) and tmin_cur > (tmax_nxt + 1.0):
#                violations_tmin[x] = violations_tmin[x] + 1
#                violations_tmax[x+1] = violations_tmax[x+1] + 1
#                violation_fnd = True
            x += 1

        if violation_fnd:

            max_violations = np.max(np.concatenate([violations_tmin, violations_tmax]))

            mask_tmin[violations_tmin == max_violations] = True
            mask_tmax[violations_tmax == max_violations] = True

            tmin, tmax, flags_tmin, flags_tmax = _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_INTERNAL_INCONSIST)


    mask_tmin = np.zeros(days[DATE].size, dtype=np.bool_)
    mask_tmax = np.zeros(days[DATE].size, dtype=np.bool_)

    nan_mask = np.logical_and(_not_nan(tmin), _not_nan(tmax))

    mask_inconsist = np.logical_and(nan_mask, tmin > tmax)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_inconsist, mask_inconsist, QA_INTERNAL_INCONSIST)

def _qa_lagrange_inconsist(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Check for differences in excess of 40C between Tmax and warmest Tmin in current/adjacent days
    and vice versa.
    '''

    LAG_THRES = 40.0

    mask_tmin = np.zeros(days[DATE].size, dtype=np.bool_)
    mask_tmax = np.zeros(days[DATE].size, dtype=np.bool_)

    x = 0
    while x < days[DATE].size:
        
        if x == 0 and x == days[DATE].size - 1:
            window_indices = np.array([x])
        elif x == 0:
            window_indices = np.array([x, x + 1])
        elif x == days[DATE].size - 1:
            window_indices = np.array([x-1, x])
        else:
            window_indices = np.array([x - 1, x, x + 1])
        
        tmin_window = tmin[list(window_indices)]
        tmax_window = tmax[list(window_indices)]
        
        tmin_cur = tmin[x]
        tmax_cur = tmax[x]
        
        nan_mask_tmin = np.logical_not(np.isnan(tmin_window))
        nan_mask_tmax = np.logical_not(np.isnan(tmax_window))

        if not (tmin_window[nan_mask_tmin].size == 0 or tmax_window[nan_mask_tmax].size == 0):
            if tmax_cur >= (np.max(tmin_window[nan_mask_tmin]) + LAG_THRES):
                mask_tmax[x] = True
                mask_tmin[window_indices] = True
            if tmin_cur <= (np.min(tmax_window[nan_mask_tmax]) - LAG_THRES):
                mask_tmin[x] = True
                mask_tmax[window_indices] = True
        x += 1

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_LAGRANGE_INCONSIST)



def _qa_spike_dip(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Check for unrealistic swings in temperature on adjacent days
    '''

    mask_tmin = np.zeros(days[DATE].size, dtype=np.bool_)
    mask_tmax = np.zeros(days[DATE].size, dtype=np.bool_)

    x = 0
    while x < days[DATE].size:

        if x == 0:
            tmin_prev = np.NAN
            tmax_prev = np.NAN
        else:
            tmin_prev = tmin[x - 1]
            tmax_prev = tmax[x - 1]

        tmin_cur = tmin[x]
        tmax_cur = tmax[x]

        if x == days[DATE].size - 1:
            tmin_nxt = np.NAN
            tmax_nxt = np.NAN
        else:
            tmin_nxt = tmin[x + 1]
            tmax_nxt = tmax[x + 1]

        if _is_spike_dip(tmin_prev, tmin_cur, tmin_nxt):
            mask_tmin[x] = True

        if _is_spike_dip(tmax_prev, tmax_cur, tmax_nxt):
            mask_tmax[x] = True

        x += 1

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_SPIKE_DIP)

def _qa_clim_outlier(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Check for Tmin/Tmax outliers based on z-score value > 6 standard deviations of 15-day climate norm
    Must have more than 100 values within 15-day period for this check to run.
    '''

    mask_tmin = _identify_outliers(tmin, days)
    mask_tmax = _identify_outliers(tmax, days)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, QA_CLIM_OUTLIER)

def _qa_spatial_regress(stn, stn_da, tmin, tmax, days, flags_tmin, flags_tmax, ngh_data=None):
    '''
    Check for Tmin/Tmax observations that are significantly different
    than surrounding neighbor stations (i.e.--not spatially consistent)
    via a spatial regression approach.
    '''

    uniq_yrs = np.unique(days[YEAR])

    if ngh_data is None:

        ngh_mask = _stns_in_radius_mask(stn, stn_da)[0]
        ngh_ids = stn_da.stns[STN_ID][ngh_mask]
        ngh_ids = ngh_ids[np.logical_not(ngh_ids == stn[STN_ID])]
        ngh_obs = None

    else:

        ngh_ids, dists, ngh_obs = ngh_data

    mask_flag_tmin = np.zeros(days[YMD].size, dtype=np.bool_)
    mask_flag_tmax = np.zeros(days[YMD].size, dtype=np.bool_)

    if ngh_ids.size >= MIN_NGHS:

        if ngh_obs is None:
            ngh_obs = stn_da.load_all_stn_obs(ngh_ids)

        finmask_tmin = np.isfinite(tmin)
        finmask_tmax = np.isfinite(tmax)

        for yr in uniq_yrs:

            for mth in MONTHS:

                mask_mth_yr_win = _get_mask_mth_yr_win(yr, mth, days)
                mask_mth_yr = np.logical_and(days[MONTH] == mth, days[YEAR] == yr)

                mask_obs_mth_yr_win_tmin = np.logical_and(finmask_tmin, mask_mth_yr_win)
                mask_obs_mth_yr_win_tmax = np.logical_and(finmask_tmax, mask_mth_yr_win)

                mask_obs_mth_yr_tmin = np.logical_and(finmask_tmin, mask_mth_yr)
                mask_obs_mth_yr_tmax = np.logical_and(finmask_tmax, mask_mth_yr)

                if np.sum(mask_obs_mth_yr_win_tmin) >= MIN_DAYS_MTH_WINDOW:
                    mask_flag_tmin[mask_obs_mth_yr_tmin] = _get_spatial_regress_flag_mask(stn, tmin, mask_obs_mth_yr_tmin, mask_obs_mth_yr_win_tmin, TMIN, stn_da, ngh_ids, ngh_obs)

                if np.sum(mask_obs_mth_yr_win_tmax) >= MIN_DAYS_MTH_WINDOW:
                    mask_flag_tmax[mask_obs_mth_yr_tmax] = _get_spatial_regress_flag_mask(stn, tmax, mask_obs_mth_yr_tmax, mask_obs_mth_yr_win_tmax, TMAX, stn_da, ngh_ids, ngh_obs)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_flag_tmin, mask_flag_tmax, QA_SPATIAL_REGRESS)

def _qa_spatial_corrob(stn, stn_da, tmin, tmax, days, flags_tmin, flags_tmax, ngh_data=None):
    '''
    Check for tmin/tmax observations that are corroborated 
    by any neighboring observations.
    '''

    if ngh_data is None:
        
        ngh_mask, dists = _stns_in_radius_mask(stn, stn_da)
        ngh_ids = stn_da.stns[STN_ID][ngh_mask]
        
        mask_rm_target = np.logical_not(ngh_ids == stn[STN_ID])
        ngh_ids = ngh_ids[mask_rm_target]
        dists = dists[mask_rm_target]
        
        ngh_obs = None
    else:
        ngh_ids, dists, ngh_obs = ngh_data

    date_objs = days[DATE]
    mask_flag_tmin = np.zeros(date_objs.size, dtype=np.bool_)
    mask_flag_tmax = np.zeros(date_objs.size, dtype=np.bool_)

    if ngh_ids.size >= MIN_NGHS:

        if ngh_obs is None:
            ngh_obs = stn_da.load_all_stn_obs(ngh_ids)

        s_indices = np.argsort(dists)

        dists = dists[s_indices]
        ngh_ids = ngh_ids[s_indices]
        ngh_obs_tmin = ngh_obs[TMIN][:, s_indices]
        ngh_obs_tmax = ngh_obs[TMAX][:, s_indices]

        norm_masks = _get_norms_md_masks(days, DATES_365)
        norm_masks_leap = _get_norms_md_masks(days, DATES_366)

        ngh_norms_tmin = np.ones((norm_masks.shape[0], ngh_ids.size)) * np.nan
        ngh_norms_tmax = np.ones((norm_masks.shape[0], ngh_ids.size)) * np.nan

        ngh_norms_leap_tmin = np.ones((norm_masks_leap.shape[0], ngh_ids.size)) * np.nan
        ngh_norms_leap_tmax = np.ones((norm_masks_leap.shape[0], ngh_ids.size)) * np.nan

        for x in np.arange(ngh_ids.size):

            ngh_norms_tmin[:, x] = _build_mean_norms(ngh_obs_tmin[:, x], norm_masks)
            ngh_norms_tmax[:, x] = _build_mean_norms(ngh_obs_tmax[:, x], norm_masks)

            ngh_norms_leap_tmin[:, x] = _build_mean_norms(ngh_obs_tmin[:, x], norm_masks_leap)
            ngh_norms_leap_tmax[:, x] = _build_mean_norms(ngh_obs_tmax[:, x], norm_masks_leap)


        stn_norms_tmin = _build_mean_norms(tmin, norm_masks)
        stn_norms_tmax = _build_mean_norms(tmax, norm_masks)
        stn_norms_leap_tmin = _build_mean_norms(tmin, norm_masks_leap)
        stn_norms_leap_tmax = _build_mean_norms(tmax, norm_masks_leap)

        yrs = days[YEAR]
        ydays = days[YDAY]
        day_nums = np.arange(date_objs.size)

        for x in day_nums:

            # Don't perform check on first/last day of time series
            if x == 0 or x == day_nums.size - 1:
                continue

            leap_yr = cal.isleap(yrs[x])

            mask_flag_tmin[x] = _get_spatial_corrob_flag(tmin, stn_norms_tmin, stn_norms_leap_tmin, ngh_ids, ngh_obs_tmin, ngh_norms_tmin, ngh_norms_leap_tmin, leap_yr, x, ydays, yrs)
            mask_flag_tmax[x] = _get_spatial_corrob_flag(tmax, stn_norms_tmax, stn_norms_leap_tmax, ngh_ids, ngh_obs_tmax, ngh_norms_tmax, ngh_norms_leap_tmax, leap_yr, x, ydays, yrs)

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_flag_tmin, mask_flag_tmax, QA_SPATIAL_CORROB)

def _qa_mega_inconsist(tmin, tmax, days, flags_tmin, flags_tmax):
    '''
    Last check that looks for Tmin values higher than highest Tmax for a calendar month
    and Tmax values lower than lowest Tmin value for a calendar month
    '''

    mths_uniq = np.unique(days[MONTH])

    mask_flag_tmin = np.zeros(days[MONTH].size, dtype=np.bool_)
    mask_flag_tmax = np.zeros(days[MONTH].size, dtype=np.bool_)

    for mth in mths_uniq:

        mth_mask_tmin = np.logical_and(days[MONTH] == mth, np.isfinite(tmin))
        mth_mask_tmax = np.logical_and(days[MONTH] == mth, np.isfinite(tmax))

        if tmin[mth_mask_tmin].size == 0 or tmax[mth_mask_tmax].size == 0:
            continue

        min_tmin = np.min(tmin[mth_mask_tmin])
        max_tmax = np.max(tmax[mth_mask_tmax])

        mask_flag_tmin[np.logical_and(tmin > max_tmax, mth_mask_tmin)] = True
        mask_flag_tmax[np.logical_and(tmax < min_tmin, mth_mask_tmax)] = True

    return _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_flag_tmin, mask_flag_tmax, QA_MEGA_INCONSIST)


def _not_nan(vals):
    return np.logical_not(np.isnan(vals))

def _is_dup_series(vals_yr1, vals_yr2):

    if vals_yr1[_not_nan(vals_yr1)].size > 0 and vals_yr2[_not_nan(vals_yr2)].size > 0:

        # Compare up to last day of the shortest series
        last_day = np.min([vals_yr1.size, vals_yr2.size])
        dup = np.nonzero(vals_yr1[0:last_day] == vals_yr2[0:last_day])[0].size == vals_yr2[0:last_day].size
    else:
        dup = False

    return dup

def _get_mask_mth_yr_win(yr, mth, days, mth_buffer=15):

    mth_date_str, mth_date_end = get_mth_str_end_dates(mth, yr)
    str_date = mth_date_str - timedelta(days=mth_buffer)
    end_date = mth_date_end + timedelta(days=mth_buffer)

    mask_date_str = np.logical_and(np.logical_and(days[YEAR] == str_date.year, days[MONTH] == str_date.month), days[DAY] >= str_date.day)
    mask_date_end = np.logical_and(np.logical_and(days[YEAR] == end_date.year, days[MONTH] == end_date.month), days[DAY] <= end_date.day)
    mask_yr_month = np.logical_and(days[MONTH] == mth, days[YEAR] == yr)

    mask_final = np.logical_or(np.logical_or(mask_date_str, mask_date_end), mask_yr_month)

    return mask_final

def _get_spatial_regress_flag_mask(stn, stn_obs, mask_obs_mth_yr, mask_obs_mth_yr_win, var, stn_da, ngh_ids, ngh_obs):

    indices = np.nonzero(mask_obs_mth_yr_win)[0]
    indices_mth_yr = np.nonzero(mask_obs_mth_yr)[0]
    max_index = mask_obs_mth_yr_win.size - 1
    vals_est_win = np.zeros(indices.size) * np.nan
    vals_est_mth_yr = np.zeros(indices_mth_yr.size) * np.nan
    mask_flags = np.zeros(indices_mth_yr.size, dtype=np.bool)

    ngh_stns = _get_valid_ngh_stns(stn, stn_obs, mask_obs_mth_yr_win, var, stn_da, ngh_ids, ngh_obs)

    if ngh_stns is None:
        return mask_flags

    for x, j in zip(indices, np.arange(indices.size)):

        wght_ests = []
        wghts = []
        n = 0

        obs_val = stn_obs[x]
        for i in np.arange(ngh_stns[NGH_STNS_ID].size):

            ngh_overlap_prev = ngh_stns[NGH_STNS_OBS][x - 1, i] if x != 0 else np.nan
            ngh_overlap_cur = ngh_stns[NGH_STNS_OBS][x, i]
            ngh_overlap_next = ngh_stns[NGH_STNS_OBS][x + 1, i] if x != max_index else np.nan
            ngh_vals = np.array([ngh_overlap_prev, ngh_overlap_cur, ngh_overlap_next])
            ngh_vals = ngh_vals[np.isfinite(ngh_vals)]

            if ngh_vals.size > 0:

                difs = np.abs(ngh_vals - obs_val)
                ngh_val = ngh_vals[np.argmin(difs)]

                lin_mod = ngh_stns[NGH_STNS_MODEL][i]
                a = lin_mod[0]  # slope
                b = lin_mod[1]  # intercept
                d = ngh_stns[NGH_STNS_WGHTS][i]  # weight

                wght_ests.append((b + (a * ngh_val)) * d)
                wghts.append(d)
                n += 1

            if n == MAX_NGHS:
                break

        if n >= MIN_NGHS:

            wght_ests = np.array(wght_ests)
            wghts = np.array(wghts)
            vals_est_win[j] = np.sum(wght_ests) / np.sum(wghts)
            if x in indices_mth_yr:
                vals_est_mth_yr[indices_mth_yr == x] = vals_est_win[j]


    mask_finite = np.isfinite(vals_est_win)
    # mask_finite_mth_yr = np.isfinite(vals_est_mth_yr)
    if np.nonzero(mask_finite)[0].size > 0:
        r = stats.pearsonr(stn_obs[mask_obs_mth_yr_win][mask_finite], vals_est_win[mask_finite])[0]
        if r >= NGH_CORR:
            resid = np.abs(stn_obs[mask_obs_mth_yr] - vals_est_mth_yr)
            resid_win = np.abs(stn_obs[mask_obs_mth_yr_win][mask_finite] - vals_est_win[mask_finite])
            # mask_finite_resid = np.isfinite(resid)
            resid_std = np.abs((resid - np.mean(resid_win)) / np.std(resid_win))
            # resid_std = np.abs((resid - np.mean(resid[mask_finite_resid]))/np.std(resid[mask_finite_resid]))
            mask_flags[np.logical_and(resid >= NGH_RESID_CUTOFF, resid_std >= NGH_RESID_STD_CUTOFF)] = True

    return mask_flags


def _get_valid_ngh_stns(stn, stn_obs, mask_obs_mth_yr, var, stn_da, ngh_ids, ngh_obs):

        ngh_stns = {}
        ngh_stns[NGH_STNS_ID] = []
        ngh_stns[NGH_STNS_MASK_OVERLAP] = []
        ngh_stns[NGH_STNS_WGHTS] = []
        ngh_stns[NGH_STNS_MODEL] = []

        for x in np.arange(ngh_ids.size):

            mask_valid_ngh = np.isfinite(ngh_obs[var][:, x])
            mask_overlap = np.logical_and(mask_obs_mth_yr, mask_valid_ngh)

            if np.sum(mask_overlap) >= MIN_DAYS_MTH_WINDOW and stn[STN_ID] != ngh_ids[x]:

                ngh_stns[NGH_STNS_ID].append(ngh_ids[x])
                ngh_stns[NGH_STNS_MASK_OVERLAP].append(mask_overlap)

                wght, mod = _get_wght_mod_temp(stn_obs[mask_overlap], ngh_obs[var][:, x][mask_overlap])
                ngh_stns[NGH_STNS_WGHTS].append(wght)
                ngh_stns[NGH_STNS_MODEL].append(mod)

        ngh_stns[NGH_STNS_ID] = np.array(ngh_stns[NGH_STNS_ID])
        ngh_stns[NGH_STNS_MASK_OVERLAP] = np.array(ngh_stns[NGH_STNS_MASK_OVERLAP])
        ngh_stns[NGH_STNS_WGHTS] = np.array(ngh_stns[NGH_STNS_WGHTS])
        ngh_stns[NGH_STNS_MODEL] = _object_array(ngh_stns[NGH_STNS_MODEL])
        ngh_stns[NGH_STNS_OBS] = ngh_obs[var][:, np.in1d(ngh_ids, ngh_stns[NGH_STNS_ID], assume_unique=True)]

        # Check to make sure there are at least the minimum # of neighbors for this mth yr
        if ngh_stns[NGH_STNS_ID].size >= MIN_NGHS:
            return _sort_ngh_stns(ngh_stns)
        else:
            return None


def _sort_ngh_stns(ngh_stns):

    s_indices = np.argsort(ngh_stns[NGH_STNS_WGHTS])[::-1]

    ngh_stns[NGH_STNS_ID] = ngh_stns[NGH_STNS_ID][s_indices]
    ngh_stns[NGH_STNS_MASK_OVERLAP] = ngh_stns[NGH_STNS_MASK_OVERLAP][s_indices, :]
    ngh_stns[NGH_STNS_WGHTS] = ngh_stns[NGH_STNS_WGHTS][s_indices]
    ngh_stns[NGH_STNS_MODEL] = ngh_stns[NGH_STNS_MODEL][s_indices]
    ngh_stns[NGH_STNS_OBS] = ngh_stns[NGH_STNS_OBS][:, s_indices]

    return ngh_stns

def _object_array(x):
    array = np.empty(len(x), dtype=np.object)
    for i in range(len(x)):
        array[i] = x[i]
    return array

def _build_lin_model(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value, p_value, std_err

def _get_wght_mod_temp(stn_obs, ngh_obs):
    
    ioa = calc_ioa_d1(stn_obs, ngh_obs)
    lin_mod = _build_lin_model(ngh_obs, stn_obs)
    return ioa, lin_mod

def _stns_in_radius_mask(stn, stn_da, radius=NGH_RADIUS):
    dists = grt_circle_dist(stn[LON], stn[LAT], stn_da.stns[LON], stn_da.stns[LAT])
    # mask = np.logical_and(dists <= radius,np.char.startswith(stn_da.stns[STN_ID],"GHCN"))
    mask = dists <= radius
    # mask = dists <= radius
    return mask, dists[mask]

def _get_spatial_corrob_flag(vals, stn_norms_noleap, stn_norms_leap, ngh_ids, ngh_obs, ngh_norms_noleap, ngh_norms_leap, is_leap, obs_num, ydays, yrs):

    if is_leap:
        stn_norms = stn_norms_leap
    else:
        stn_norms = stn_norms_noleap

    if yrs[obs_num - 1] != yrs[obs_num]:
        is_leap_prev = cal.isleap(yrs[obs_num - 1])
    else:
        is_leap_prev = is_leap

    if yrs[obs_num + 1] != yrs[obs_num]:
        is_leap_nxt = cal.isleap(yrs[obs_num + 1])
    else:
        is_leap_nxt = is_leap

    day_num_prev = ydays[obs_num - 1] - 1
    day_num_cur = ydays[obs_num] - 1
    day_num_nxt = ydays[obs_num + 1] - 1

    anom_stn = np.abs(vals[obs_num] - stn_norms[day_num_cur])

    if np.isnan(anom_stn):
        return False
    else:
        mask_finite_prev = np.isfinite(ngh_obs[obs_num - 1, :])
        mask_finite_cur = np.isfinite(ngh_obs[obs_num, :])
        mask_finite_nxt = np.isfinite(ngh_obs[obs_num + 1, :])
        flag = False
        if np.sum(mask_finite_prev) >= MIN_NGHS and np.sum(mask_finite_cur) >= MIN_NGHS and np.sum(mask_finite_nxt) >= MIN_NGHS:

            if is_leap_prev:
                ngh_norms_prev = ngh_norms_leap[day_num_prev, mask_finite_prev]
            else:
                ngh_norms_prev = ngh_norms_noleap[day_num_prev, mask_finite_prev]

            if is_leap:
                ngh_norms_cur = ngh_norms_leap[day_num_cur, mask_finite_cur]
            else:
                ngh_norms_cur = ngh_norms_noleap[day_num_cur, mask_finite_cur]

            if is_leap_nxt:
                ngh_norms_nxt = ngh_norms_leap[day_num_nxt, mask_finite_nxt]
            else:
                ngh_norms_nxt = ngh_norms_noleap[day_num_nxt, mask_finite_nxt]

            ngh_obs_prev = ngh_obs[obs_num - 1, mask_finite_prev]
            ngh_obs_cur = ngh_obs[obs_num, mask_finite_cur]
            ngh_obs_nxt = ngh_obs[obs_num + 1, mask_finite_nxt]

            anom_ngh_prev = np.abs(ngh_obs_prev - ngh_norms_prev)
            anom_ngh_cur = np.abs(ngh_obs_cur - ngh_norms_cur)
            anom_ngh_nxt = np.abs(ngh_obs_nxt - ngh_norms_nxt)

            anom_ngh_prev = anom_ngh_prev[np.isfinite(anom_ngh_prev)][0:MAX_NGHS]
            anom_ngh_cur = anom_ngh_cur[np.isfinite(anom_ngh_cur)][0:MAX_NGHS]
            anom_ngh_nxt = anom_ngh_nxt[np.isfinite(anom_ngh_nxt)][0:MAX_NGHS]

            anom_all = np.concatenate([anom_ngh_prev, anom_ngh_cur, anom_ngh_nxt])

            difs = np.abs(anom_all - anom_stn)

            if difs[difs >= ANOMALY_CUTOFF].size == difs.size:
                flag = True
        return flag

def _build_norms(vals, days, yr_days):

    mth_days = get_md_array(days[DATE])

    # Build 15-day norms for every day in days
    day_nums = np.arange(yr_days.size)
    # Column 0: mean, Column 1: std
    norms = np.ones([yr_days.size, 2]) * np.NAN

    fin_mask = np.isfinite(vals)

    for x in day_nums:
        date = yr_days[x]
        srt_date = date - A_WEEK
        end_date = date + A_WEEK
        date_range = get_date_array(srt_date, end_date)
        mth_days_range = get_md_array(date_range)

        date_mask = np.in1d(mth_days, mth_days_range)
        vals_rng = vals[np.logical_and(fin_mask, date_mask)]

        if vals_rng.size >= MIN_NORM_VALUES:

            norms[x, 0], norms[x, 1] = _biweight_mean_std(vals_rng)

    return norms

def _get_norms_md_masks(days, yr_days):

    mth_days = get_md_array(days[DATE])

    # For non-leap years
    day_nums = np.arange(yr_days.size)

    norm_masks = np.zeros((yr_days.size, mth_days.size), dtype=np.bool)

    for x in day_nums:

        mth_day = yr_days[x]
        srt_mth_day = mth_day - A_WEEK
        end_mth_day = mth_day + A_WEEK
        date_range = get_date_array(srt_mth_day, end_mth_day)
        mth_days_range = get_md_array(date_range)

        norm_masks[x, :] = np.in1d(mth_days, mth_days_range)

    return norm_masks


def _identify_outliers(vals, days):

#    days_366 = utld.get_date_array(datetime(2004,1,1),datetime(2004,12,31))
#    days_365 = utld.get_date_array(datetime(2003,1,1),datetime(2003,12,31))

    norms_366 = _build_norms(vals, days, DATES_366)
    norms_365 = _build_norms(vals, days, DATES_365)

    date_objs = days[DATE]
    yrs = days[YEAR]
    ydays = days[YDAY]
    mask = np.zeros(date_objs.size, dtype=np.bool_)
    day_nums = np.arange(date_objs.size)

    for x in day_nums:
        day_num = ydays[x] - 1

        if cal.isleap(yrs[x]):
            norms = norms_366
        else:
            norms = norms_365

        # check that a norm exists
        if not np.isnan(norms[day_num, 0]) and not np.isnan(vals[x]):

            z_score = np.abs((vals[x] - norms[day_num, 0]) / norms[day_num, 1])

            if z_score >= 6.0:
                mask[x] = True

    return mask

def _mad(a):
    '''
    Calculates median absolute deviations 
    '''
    return np.median(np.abs(a - np.median(a)))

def _biweight_mean(X):
    # mean
    c = 7.5
    M = np.median(X)
    MAD = np.median(np.abs(X - M))

    if MAD == 0:
        # return normal mean,biweight cannot be calculated
        return np.mean(X)

    u = (X - M) / (c * MAD)
    u[np.abs(u) >= 1.0] = 1.0
    Xbi = M + (np.sum((X - M) * (1.0 - u ** 2) ** 2) / np.sum((1 - u ** 2) ** 2))
    return Xbi


def _biweight_mean_std(X):
    '''
    Calculates more robust mean/std for climate data
    Used by Durre et al. 2010 referencing Lanzante 1996
    '''

    # mean
    c = 7.5
    M = np.median(X)
    MAD = _mad(X)

    if MAD == 0:
        # return normal mean, std. biweight cannot be calculated
        return np.mean(X), np.std(X, ddof=1)

    u = (X - M) / (c * MAD)
    u[np.abs(u) >= 1.0] = 1.0
    Xbi = M + (np.sum((X - M) * (1.0 - u ** 2) ** 2) / np.sum((1 - u ** 2) ** 2))

    # std
    n = X.size
    Sbi = ((n * np.sum(((X - M) ** 2) * (1 - u ** 2) ** 4)) ** 0.5) / np.abs(np.sum((1 - u ** 2) * (1 - (5 * u ** 2))))

    # return np.mean(X),np.std(X)

    return Xbi, Sbi


def _build_mean_norms(vals, norm_md_masks):

    norms = np.ones(norm_md_masks.shape[0]) * np.NAN
    nan_mask = np.isfinite(vals)

    for x in np.arange(norms.size):

        vals_mth_day = vals[np.logical_and(nan_mask, norm_md_masks[x, :])]

        if vals_mth_day.size >= MIN_NORM_VALUES:
            norms[x] = _biweight_mean(vals_mth_day)
            # norms[x] = _biweight_mean_std(vals_mth_day)[0]

    return norms

def _update_obs_flags(tmin, tmax, flags_tmin, flags_tmax, mask_tmin, mask_tmax, flag):

    tmin[mask_tmin] = np.NAN
    tmax[mask_tmax] = np.NAN
    flags_tmin[np.logical_and(flags_tmin == QA_OK, mask_tmin)] = flag
    flags_tmax[np.logical_and(flags_tmax == QA_OK, mask_tmax)] = flag

    return tmin, tmax, flags_tmin, flags_tmax

def _identify_streaks(vals, days):

    STREAK_LEN = 20

    date_objs = days[DATE]
    date_nums = np.arange(date_objs.size)
    mask = np.zeros(date_objs.size, dtype=np.bool_)

    streak_indices = []

    for x in date_nums:

        if np.isnan(vals[x]):
            continue
        elif x == 0 or len(streak_indices) == 0:
            streak_indices.append(x)
        elif vals[x] == vals[streak_indices[-1]]:
            streak_indices.append(x)
        elif len(streak_indices) >= STREAK_LEN:
            mask[streak_indices] = True
            streak_indices = []
            streak_indices.append(x)
        else:
            streak_indices = []
            streak_indices.append(x)

    return mask

def _get_gap_bounds(vals):

    GAP_THRES = 10.0  # degrees C

    vals_sorted = np.sort(vals[_not_nan(vals)])

    if vals_sorted.size == 0:
        return None, None

    val_median = np.median(vals_sorted)
    val_top = vals_sorted[vals_sorted >= val_median]
    val_bottom = vals_sorted[vals_sorted <= val_median][::-1]

    gap_mask_top = np.ediff1d(val_top, to_begin=[0]) >= GAP_THRES

    if val_top[gap_mask_top].size > 0:
        bnds_top = val_top[gap_mask_top][0]
    else:
        bnds_top = None

    gap_mask_bottom = np.abs(np.ediff1d(val_bottom, to_begin=[0])) >= GAP_THRES

    if val_bottom[gap_mask_bottom].size > 0:
        bnds_bottom = val_bottom[gap_mask_bottom][0]
    else:
        bnds_bottom = None

    return (bnds_bottom, bnds_top)

def _is_spike_dip(prev, cur, nxt):

    SPIKE_DIP_THRES = 25.0
    if np.abs(cur - prev) >= SPIKE_DIP_THRES and np.abs(cur - nxt) >= SPIKE_DIP_THRES:
        return True
    else:
        return False

def _imposs_value_mask(tair):
    '''
    Check for values that are outside the bounds of world records
    '''

    return np.logical_or(tair < TMIN_RECORD, tair > TMAX_RECORD)
