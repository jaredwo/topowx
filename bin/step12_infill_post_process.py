'''
Created on Jul 2, 2014

@author: jared.oyler
'''

from twx.db import StationSerialDataDb, BAD
from twx.infill import create_serially_complete_db, set_bad_stations, find_dup_stns,add_stn_raster_values
from twx.raster import RasterDataset
import os
import numpy as np

if __name__ == '__main__':
    
    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    
    #Paths to infilled netCDF4 databases
    path_infill_db_tmin = os.path.join(FPATH_STNDATA, 'infill','infill_tmin.nc')
    path_infill_db_tmax = os.path.join(FPATH_STNDATA, 'infill','infill_tmax.nc')
    
    #Paths to final output serially-complete netCDF4 databases
    path_serial_db_tmin = os.path.join(FPATH_STNDATA, 'infill','serial_tmin.nc')
    path_serial_db_tmax = os.path.join(FPATH_STNDATA, 'infill','serial_tmax.nc')
    
    #Create final output serially-complete netCDF4 databases
    create_serially_complete_db(path_infill_db_tmin, 'tmin', path_serial_db_tmin)
    create_serially_complete_db(path_infill_db_tmax, 'tmin', path_serial_db_tmin)
    
    #Load new serially complete dbs
    stnda_tmin = StationSerialDataDb(path_serial_db_tmin, 'tmin', mode='r+')
    stnda_tmax = StationSerialDataDb(path_serial_db_tmax, 'tmax', mode='r+')
    
    #Add a 'bad' station variable
    stnda_tmin.add_stn_variable(BAD, "bad station flag", units='', dtype='i1', fill_value=0)
    stnda_tmax.add_stn_variable(BAD, "bad station flag", units='', dtype='i1', fill_value=0)
    
    #Set 'bad' stations
    bad_stnids = np.unique(np.loadtxt('/projects/daymet2/station_data/infill/infill_20130725/BadStns.csv',
                                      delimiter=",",dtype=np.str,skiprows=1,usecols=(0,)))
    set_bad_stations(stnda_tmin.ds, bad_stnids)
    set_bad_stations(stnda_tmax.ds, bad_stnids)
    
    #Find duplicate stations and set the 'bad' flag on them
    dup_stnids_tmin = find_dup_stns(stnda_tmin)
    set_bad_stations(stnda_tmin.ds, dup_stnids_tmin, reset=False)
    dup_stnids_tmax = find_dup_stns(stnda_tmax)
    set_bad_stations(stnda_tmax.ds, dup_stnids_tmax, reset=False)
    
    #Extract auxiliary predictors and other raster-based variables
    #for each station and add to station database
    
    #Monthly Land Skin Temperature Predictors
    path_lst_rasters = '/projects/daymet2/dem/interp_grids/tifs/mthly_lst'
    long_name = 'land skin temperature'
    units = "C"
    
    #Nighttime LST
    name = 'land surface temperature'
    units = "C"
    for mth in np.arange(1,13):
        
        var_name = 'lst%02d'%mth
        print var_name
        
        a_rast = RasterDataset(os.path.join(path_lst_rasters,'MOSAIC.LST_Night_1km.%02d.C.tif'%mth))
        add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast)

    