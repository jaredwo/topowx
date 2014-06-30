'''
Functions and classes for working with NCEP/NCAR Reanalysis
netCDF data downloaded from ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.
'''

__all__ = ['create_nnr_subset','create_nnr_subset_nolevel','NNRNghData']

from netCDF4 import Dataset, num2date
import netCDF4
import numpy as np
from twx.utils import A_DAY, YMD, get_days_metadata, grt_circle_dist, get_ymd_array
from twx.db import dbDataset
from datetime import datetime
import os

# NCEP/NCAR Reanalysis has observation times
# at 0Z, 6Z, 12Z, and 18Z. For a 24Z observation
# on the current day, use the 0Z observation of
# the next day
UTC_TIMES = {0:0, 6:6, 12:12, 18:18, 24:0}

# Spatial bounds for creating North American
# subsets of the NCEP/NCAR Reanalysis data
LAT_TOP, LAT_BOTTOM = 60, 10
LON_LEFT, LON_RIGHT = -135, -55


def create_nnr_subset(path_nnr, fpath_out, yrs, days, varname_in, varname_out, levels_subset, utc_time, conv_func):
    '''
    Create a North American netCDF subset of NCEP/NCAR Reanalysis data
    for a specific time period, variable, pressure level(s), and 
    UTC observation time.
    
    Parameters
    ----------
    path_nnr : str
        Local path to directory containing the 
        original NCEP/NCAR Reanalysis netCDF files.
    fpath_out : str
        File path for the output netCDF file
    yrs : ndarray
        An array of years representing the time period
        for which the subset should be created.
    days : structured ndarray
        A structured ndarray from twx.utils.get_days_metadata
        containing daily date information for the time period.
    varname_in : str
        The name of the input NCEP/NCAR Reanalysis variable
    varname_out : str
        The name of the output variable
    levels_subset : ndarray
        An array of pressure levels for which the subset should
        be created.
    utc_time : int
        The UTC observation time for the subset (0Z,6Z,12Z,or 24Z)
    conv_func : function
        A function for performing any type of conversion/processing
        on the input variable.
    '''

    print "Creating North American NCEP/NCAR Reanalysis subset from %d to %d for %s at levels %s and %dZ..." % (yrs[0],
          yrs[-1], varname_in, ",".join(['%d' % (alevel,) for alevel in levels_subset]), utc_time)

    ds_out = dbDataset(fpath_out, 'w')

    ds = Dataset(os.path.join(path_nnr, '%s.%d.nc' % (varname_in, yrs[0])))
    levels = ds.variables['level'][:]
    lons = ds.variables['lon'][:]
    lons[lons > 180] = lons[lons > 180] - 360.0
    lats = ds.variables['lat'][:]

    mask_levels = np.in1d(levels, levels_subset, True)
    mask_lons = np.logical_and(lons >= LON_LEFT, lons <= LON_RIGHT)
    mask_lats = np.logical_and(lats >= LAT_BOTTOM, lats <= LAT_TOP)

    ds_out.db_create_global_attributes("".join(["NCEP/NCAR Daily ", varname_out, " ", str(utc_time), "Z Subset"]))
    ds_out.db_create_lonlat_dimvar(lons[mask_lons], lats[mask_lats])
    ds_out.db_create_time_dimvar(days)
    ds_out.db_create_level_dimvar(levels[mask_levels])

    ds_out.createVariable(varname_out, 'f4', ('time', 'level', 'lat', 'lon'),
                          fill_value=netCDF4.default_fillvals['f4'],
                          chunksizes=(days.size, levels[mask_levels].size, 4, 4))
    for yr in yrs:

        ds = Dataset(os.path.join(path_nnr, '%s.%d.nc' % (varname_in, yr)))

        start, end = ds.variables['time'][0], ds.variables['time'][-1] + 6

        times_yr = num2date(np.arange(start, end, 6), units=ds.variables['time'].units)
        hours_yr = np.array([x.hour for x in times_yr])
        mask_day = hours_yr == UTC_TIMES[utc_time]

        var_data = conv_func(ds.variables[varname_in][mask_day, mask_levels, mask_lats, mask_lons])

        dates_yr = times_yr[mask_day]
        if utc_time == 24:
            dates_yr = np.array([x - A_DAY for x in dates_yr])

        ymd_yr = get_ymd_array(dates_yr)

        mask_ymd = np.logical_and(ymd_yr >= days[YMD][0], ymd_yr <= days[YMD][-1])
        ymd_yr = ymd_yr[mask_ymd]
        var_data = var_data[mask_ymd, :, :, :]

        fnl_day_mask = np.in1d(days[YMD], ymd_yr, True)

        ds_out.variables[varname_out][fnl_day_mask, :, :, :] = var_data
        ds_out.sync()

        print yr

def create_nnr_subset_nolevel(path_nnr, fpath_out, yrs, days, varname_in, varname_out, utc_time, conv_func, suffix=""):
    '''
    Create a North American netCDF subset of NCEP/NCAR Reanalysis data
    for a specific time period, variable, and UTC observation time. This
    function is for variables that do not have a pressure level.
    
    Parameters
    ----------
    path_nnr : str
        Local path to directory containing the 
        original NCEP/NCAR Reanalysis netCDF files.
    fpath_out : str
        File path for the output netCDF file
    yrs : ndarray
        An array of years representing the time period
        for which the subset should be created.
    days : structured ndarray
        A structured ndarray from twx.utils.get_days_metadata
        containing daily date information for the time period.
    varname_in : str
        The name of the input NCEP/NCAR Reanalysis variable
    varname_out : str
        The name of the output variable
    utc_time : int
        The UTC observation time for the subset (0,6,12,or 24)
    conv_func : function
        A function for performing any type of conversion/processing
        on the input variable.
    suffix : str, optional
        A suffix that should be added after the variable name for the
        filenames of the input NCEP/NCAR Reanalysis netCDF files (eg .sig995)
    '''

    print "Creating North American NCEP/NCAR Reanalysis subset from %d to %d for %s at %dZ..." % (yrs[0],
          yrs[-1], varname_in, utc_time)

    ds_out = dbDataset(fpath_out, 'w')

    ds = Dataset(os.path.join(path_nnr, '%s%s.%d.nc' % (varname_in, suffix, yrs[0])))
    lons = ds.variables['lon'][:]
    lons[lons > 180] = lons[lons > 180] - 360.0
    lats = ds.variables['lat'][:]

    mask_lons = np.logical_and(lons >= LON_LEFT, lons <= LON_RIGHT)
    mask_lats = np.logical_and(lats >= LAT_BOTTOM, lats <= LAT_TOP)

    ds_out.db_create_global_attributes("".join(["NCEP/NCAR Daily ", varname_out, " ", str(utc_time), "Z Subset"]))
    ds_out.db_create_lonlat_dimvar(lons[mask_lons], lats[mask_lats])
    ds_out.db_create_time_dimvar(days)

    ds_out.createVariable(varname_out, 'f4', ('time', 'lat', 'lon'),
                          fill_value=netCDF4.default_fillvals['f4'],
                          chunksizes=(days.size, 4, 4))

    for yr in yrs:

        ds = Dataset(os.path.join(path_nnr, '%s%s.%d.nc' % (varname_in, suffix, yr)))

        start, end = ds.variables['time'][0], ds.variables['time'][-1] + 6

        times_yr = num2date(np.arange(start, end, 6), units=ds.variables['time'].units)
        hours_yr = np.array([x.hour for x in times_yr])
        mask_day = hours_yr == UTC_TIMES[utc_time]

        var_data = conv_func(ds.variables[varname_in][mask_day, mask_lats, mask_lons])

        dates_yr = times_yr[mask_day]
        if utc_time == 24:
            dates_yr = np.array([x - A_DAY for x in dates_yr])

        ymd_yr = get_ymd_array(dates_yr)

        mask_ymd = np.logical_and(ymd_yr >= days[YMD][0], ymd_yr <= days[YMD][-1])
        ymd_yr = ymd_yr[mask_ymd]
        var_data = var_data[mask_ymd, :, :]

        fnl_day_mask = np.in1d(days[YMD], ymd_yr, True)

        ds_out.variables[varname_out][fnl_day_mask, :, :] = var_data
        ds_out.sync()

        print yr

def create_thickness_nnr_subset(path_nnr, fpath_out, yrs, days, level_up, level_low, utc_time):
    '''
    Create a North American netCDF subset of NCEP/NCAR Reanalysis atmospheric
    thickness (difference in height between 2 pressure levels) for a specific 
    time period, an upper and lower pressure level, and UTC observation time.
    
    Parameters
    ----------
    path_nnr : str
        Local path to directory containing the 
        original NCEP/NCAR Reanalysis netCDF hgt 
        variable files.
    fpath_out : str
        File path for the output netCDF file
    yrs : ndarray
        An array of years representing the time period
        for which the subset should be created.
    days : structured ndarray
        A structured ndarray from twx.utils.get_days_metadata
        containing daily date information for the time period.
    level_up : int
        The upper level for thickness calculation.
    level_low: int
        The lower level for thickness calculation.
    utc_time : int
        The UTC observation time for the subset (0Z,6Z,12Z,or 24Z)
    '''

    print "Creating North American NCEP/NCAR Reanalysis atmospheric thickness subset from %d to %d for levels %d and %d at %dZ..." % (yrs[0],
          yrs[-1], level_up, level_low, utc_time)

    ds_out = dbDataset(fpath_out, 'w')

    ds = Dataset(os.path.join(path_nnr, 'hgt.%d.nc' % (yrs[0],)))
    levels = ds.variables['level'][:]
    lons = ds.variables['lon'][:]
    lons[lons > 180] = lons[lons > 180] - 360.0
    lats = ds.variables['lat'][:]

    idx_levelup = np.nonzero(levels == level_up)[0][0]
    idx_levellow = np.nonzero(levels == level_low)[0][0]

    mask_lons = np.logical_and(lons >= LON_LEFT, lons <= LON_RIGHT)
    mask_lats = np.logical_and(lats >= LAT_BOTTOM, lats <= LAT_TOP)

    ds_out.db_create_global_attributes("".join(["NCEP/NCAR Daily ", str(level_up), "-", str(level_low), " thickness ", str(utc_time), "Z Subset"]))
    ds_out.db_create_lonlat_dimvar(lons[mask_lons], lats[mask_lats])
    ds_out.db_create_time_dimvar(days)

    ds_out.createVariable('thick', 'f4', ('time', 'lat', 'lon'),
                          fill_value=netCDF4.default_fillvals['f4'],
                          chunksizes=(days.size, 4, 4))
    for yr in yrs:

        ds = Dataset(os.path.join(path_nnr, 'hgt.%d.nc' % (yr,)))

        start, end = ds.variables['time'][0], ds.variables['time'][-1] + 6

        times_yr = num2date(np.arange(start, end, 6), units=ds.variables['time'].units)
        hours_yr = np.array([x.hour for x in times_yr])
        mask_day = hours_yr == UTC_TIMES[utc_time]

        data_levelup = ds.variables['hgt'][mask_day, idx_levelup, mask_lats, mask_lons]
        data_levellow = ds.variables['hgt'][mask_day, idx_levellow, mask_lats, mask_lons]
        data_thick = data_levelup - data_levellow

        dates_yr = times_yr[mask_day]
        if utc_time == 24:
            dates_yr = np.array([x - A_DAY for x in dates_yr])

        ymd_yr = get_ymd_array(dates_yr)

        mask_ymd = np.logical_and(ymd_yr >= days[YMD][0], ymd_yr <= days[YMD][-1])
        ymd_yr = ymd_yr[mask_ymd]
        data_thick = data_thick[mask_ymd, :, :]

        fnl_day_mask = np.in1d(days[YMD], ymd_yr, True)

        ds_out.variables['thick'][fnl_day_mask, :, :] = data_thick
        ds_out.sync()

        print yr


class NNRNghData():

    NNR_VARS = np.array(['tair', 'hgt', 'thick', 'rhum', 'uwnd', 'vwnd', 'slp'])
    NNR_TIMES = np.array(['24z', '18z', '12z'])
    TMIN = 'tmin'
    TMAX = 'tmax'
    UTC_OFFSET_TIMES = {TMIN:{-4:'12z', -5:'12z', -6:'12z', -7:'12z', -8:'12z'},
                        TMAX:{-4:'18z', -5:'18z', -6:'18z', -7:'24z', -8:'24z'}}

    def __init__(self, path_nnr, startend_ymd, nnr_vars=None):

        self.ds_nnr = {}

        if nnr_vars is None:
            self.nnr_vars = self.NNR_VARS
        else:
            self.nnr_vars = nnr_vars


        for nnr_var in self.nnr_vars:

            for nnr_time in self.NNR_TIMES:

                self.ds_nnr["".join([nnr_var, nnr_time])] = Dataset("".join([path_nnr, "nnr_", nnr_var, "_", nnr_time, ".nc"]))

        eg_ds = self.ds_nnr.values()[0]

        var_time = eg_ds.variables['time']

        start, end = num2date([var_time[0], var_time[-1]], var_time.units)
        self.days = get_days_metadata(start, end)

        self.day_mask = np.nonzero(np.logical_and(self.days[YMD] >= startend_ymd[0], self.days[YMD] <= startend_ymd[1]))[0]
        self.days = self.days[self.day_mask]

        self.nnr_lons = eg_ds.variables['lon'][:]
        self.nnr_lats = eg_ds.variables['lat'][:]
        llgrid = np.meshgrid(self.nnr_lons, self.nnr_lats)

        self.grid_lons = llgrid[0].ravel()
        self.grid_lats = llgrid[1].ravel()

    def get_nngh_matrix(self, lon, lat, tair_var, utc_offset=-7, nngh=4):

        dist_nnr = grt_circle_dist(lon, lat, self.grid_lons, self.grid_lats)
        sort_dist_nnr = np.argsort(dist_nnr)

        nnr_ngh_lons = self.grid_lons[sort_dist_nnr][0:nngh]
        nnr_ngh_lats = self.grid_lats[sort_dist_nnr][0:nngh]

        nnr_time = self.UTC_OFFSET_TIMES[tair_var][utc_offset]

        nnr_matrix = None

        for x in np.arange(nnr_ngh_lons.size):

            idx_lon = np.nonzero(self.nnr_lons == nnr_ngh_lons[x])[0][0]
            idx_lat = np.nonzero(self.nnr_lats == nnr_ngh_lats[x])[0][0]

            for nnr_var in self.nnr_vars:

                ds = self.ds_nnr["".join([nnr_var, nnr_time])]

                if "level" in ds.dimensions:
                    adata = ds.variables[nnr_var][self.day_mask, :, idx_lat, idx_lon]
                else:
                    adata = ds.variables[nnr_var][self.day_mask, idx_lat, idx_lon]

                if len(adata.shape) == 1:
                    adata.shape = (adata.size, 1)

#                if np.ma.isMA(adata):
#
#                    #print "|".join(['NNR NODATA',nnr_var,str(nnr_ngh_lons[x]),str(nnr_ngh_lats[x]),str(self.days[adata.mask.ravel()])])
#                    adata = np.ma.filled(adata, np.nan)
                if nnr_matrix is None:
                    nnr_matrix = adata
                else:
                    nnr_matrix = np.hstack((nnr_matrix, adata))



        return nnr_matrix

def KtoC(k):
    return k - 273.15

def no_conv(x):
    return x

if __name__ == '__main__':

    path_nnr = '/projects/daymet2/reanalysis_data/'
    yrs = np.arange(1948, 2014)
    days = get_days_metadata(datetime(1948, 1, 1), datetime(2013, 1, 31))

############################################################
# Temperature
############################################################
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_tair_24z.nc',
#                      yrs, days,'air','tair', np.array([850]), 24, KtoC)
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_tair_12z.nc',
#                      yrs, days,'air','tair', np.array([850]), 12, KtoC)
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_tair_18z.nc',
#                      yrs, days,'air','tair', np.array([850]), 18, KtoC)

############################################################
# 500-1000 Thickness
############################################################
#    create_thickness_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_thick_24z.nc',
#                                yrs, days, 500, 1000, 24)
#
#    create_thickness_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_thick_12z.nc',
#                                yrs, days, 500, 1000, 12)
#
#    create_thickness_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_thick_18z.nc',
#                                yrs, days, 500, 1000, 18)

############################################################
# Height
############################################################

#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_hgt_24z.nc',
#                      yrs, days,'hgt','hgt', np.array([500,700]), 24, no_conv)
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_hgt_12z.nc',
#                      yrs, days,'hgt','hgt', np.array([500,700]), 12, no_conv)
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_hgt_18z.nc',
#                      yrs, days,'hgt','hgt', np.array([500,700]), 18, no_conv)

############################################################
# RH
############################################################
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_rhum_24z.nc',
#                      yrs, days,'rhum','rhum', np.array([850]), 24, no_conv)
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_rhum_12z.nc',
#                      yrs, days,'rhum','rhum', np.array([850]), 12, no_conv)
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_rhum_18z.nc',
#                      yrs, days,'rhum','rhum', np.array([850]), 18, no_conv)

############################################################
# SLP
############################################################

#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_slp_24z.nc',
#                              yrs, days, 'slp', 'slp', 24, no_conv)
#
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_slp_12z.nc',
#                              yrsnc, days, 'slp', 'slp', 12, no_conv)
#
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_slp_18z.nc',
#                              yrs, days, 'slp', 'slp', 18, no_conv)

############################################################
# Inversion Strength
############################################################

#    create_inversion_strength_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_inv_str_24z.nc',
#                                         yrs, days, 700, 24)
#    create_inversion_strength_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_inv_str_12z.nc',
#                                         yrs, days, 700, 12)
#    create_inversion_strength_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_inv_str_18z.nc',
#                                         yrs, days, 700, 18)

############################################################
# V-Wind Surface
############################################################

#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_24z.nc',
#                              yrs, days, 'vwnd', 'vwnd', 24, no_conv,".sig995")
#
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_12z.nc',
#                              yrs, days, 'vwnd', 'vwnd', 12, no_conv,".sig995")
#
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_18z.nc',
#                              yrs, days, 'vwnd', 'vwnd', 18, no_conv,".sig995")

############################################################
# U-Wind Surface
############################################################

#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_24z.nc',
#                              yrs, days, 'uwnd', 'uwnd', 24, no_conv,".sig995")
#
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_12z.nc',
#                              yrs, days, 'uwnd', 'uwnd', 12, no_conv,".sig995")
#
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_18z.nc',
#                              yrs, days, 'uwnd', 'uwnd', 18, no_conv,".sig995")



############################################################
# V-Wind 850mb
############################################################

#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_24z.nc',
#                      yrs, days,'vwnd','vwnd', np.array([850]), 24, no_conv)
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_12z.nc',
#                      yrs, days,'vwnd','vwnd', np.array([850]), 12, no_conv)
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_18z.nc',
#                      yrs, days,'vwnd','vwnd', np.array([850]), 18, no_conv)

############################################################
# U-Wind 850mb
############################################################

    create_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_24z.nc',
                      yrs, days, 'uwnd', 'uwnd', np.array([850]), 24, no_conv)

#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_12z.nc',
#                      yrs, days,'uwnd','uwnd', np.array([850]), 12, no_conv)
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_18z.nc',
#                      yrs, days,'uwnd','uwnd', np.array([850]), 18, no_conv)

