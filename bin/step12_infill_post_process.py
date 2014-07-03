'''
Script for creating the final serially-complete station
databases to be used as input to the interpolation procedures.
Determines the final serially-complete time series to be used
at each station, sets flags on bad and duplicate stations,
adds monthly normal values to each station, and extracts auxiliary 
predictors and other raster-based variables for each station.
'''

from twx.db import StationSerialDataDb, BAD, CLIMDIV
from twx.infill import create_serially_complete_db,\
set_bad_stations, find_dup_stns,add_stn_raster_values
from twx.raster import RasterDataset
import os
import numpy as np
from twx.infill.post_infill import add_monthly_normals

if __name__ == '__main__':
    
    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    PATH_RASTERS = os.path.join(PROJECT_ROOT, 'dem','interp_grids','tifs')
    
    #Paths to infilled netCDF4 databases
    path_infill_db_tmin = os.path.join(FPATH_STNDATA, 'infill','infill_tmin.nc')
    path_infill_db_tmax = os.path.join(FPATH_STNDATA, 'infill','infill_tmax.nc')
    
    #Paths to final output serially-complete netCDF4 databases
    path_serial_db_tmin = os.path.join(FPATH_STNDATA, 'infill','serial_tmin.nc')
    path_serial_db_tmax = os.path.join(FPATH_STNDATA, 'infill','serial_tmax.nc')
    
    #Create final output serially-complete netCDF4 databases
    print "Creating final serially-complete databases..."
    create_serially_complete_db(path_infill_db_tmin, 'tmin', path_serial_db_tmin)
    create_serially_complete_db(path_infill_db_tmax, 'tmin', path_serial_db_tmin)
    
    #Load new serially complete dbs
    stnda_tmin = StationSerialDataDb(path_serial_db_tmin, 'tmin', mode='r+')
    stnda_tmax = StationSerialDataDb(path_serial_db_tmax, 'tmax', mode='r+')
    
    #Add a 'bad' station variable
    stnda_tmin.add_stn_variable(BAD, "bad station flag", units='', dtype='i1', fill_value=0)
    stnda_tmax.add_stn_variable(BAD, "bad station flag", units='', dtype='i1', fill_value=0)
    
    #Set 'bad' flag on stations that the infill algorithm failed to properly infill, 
    #have strange inhomogeneities, are completely outside raster domains, etc.  
    bad_stnids = np.unique(np.loadtxt(os.path.join(FPATH_STNDATA, 'infill','BadStns.csv'),
                                      delimiter=",",dtype=np.str,skiprows=1,usecols=(0,)))
    set_bad_stations(stnda_tmin.ds, bad_stnids)
    set_bad_stations(stnda_tmax.ds, bad_stnids)
    
    #Find duplicate stations and set the 'bad' flag on them.
    #The kriging interpolation algorithm cannot have point observations
    #at the exact same location. For two or more stations with the same
    #location, the one with the longest non-infilled period-of-record is
    #used and the others are considered duplicates and removed.
    
    #Need to use infilled databases to search for dups
    stnda_infill_tmin = StationSerialDataDb(path_infill_db_tmin, 'tmin')
    stnda_infill_tmax = StationSerialDataDb(path_infill_db_tmax, 'tmax')
    
    print "Determining duplicate Tmin stations..."
    dup_stnids_tmin = find_dup_stns(stnda_infill_tmin)
    set_bad_stations(stnda_tmin.ds, dup_stnids_tmin, reset=False)
    print "Determining duplicate Tmax stations..."
    dup_stnids_tmax = find_dup_stns(stnda_infill_tmax)
    set_bad_stations(stnda_tmax.ds, dup_stnids_tmax, reset=False)
    
    #Calculate 1981-2010 monthly normals for each station and
    #add to database
    print "Adding monthly normals to Tmin database..."
    add_monthly_normals(stnda_tmin, start_norm_yr=1981, end_norm_yr=2010)
    print "Adding monthly normals to Tmax database..."
    add_monthly_normals(stnda_tmax, start_norm_yr=1981, end_norm_yr=2010)
    
    #Extract auxiliary predictors and other raster-based variables
    #for each station and add to station database
    
    #Monthly Land Skin Temperature Predictors
    path_lst_rasters = os.path.join(PATH_RASTERS,'mthly_lst')
    long_name = 'land skin temperature'
    units = "C"
    
    #Nighttime LST
    print "Adding Nighttime LST values to Tmin stations..."
    for mth in np.arange(1,13):
        var_name = 'lst%02d'%mth
        print var_name
        a_rast = RasterDataset(os.path.join(path_lst_rasters,'MOSAIC.LST_Night_1km.%02d.C.tif'%mth))
        add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast)
    
    #Daytime LST
    print "Adding Daytime LST values to Tmax stations..."
    for mth in np.arange(1,13):
        var_name = 'lst%02d'%mth
        print var_name 
        a_rast = RasterDataset(os.path.join(path_lst_rasters,'MOSAIC.LST_Day_1km.%02d.C.tif'%mth))
        add_stn_raster_values(stnda_tmax, var_name, long_name, units, a_rast)

    #Topographic Dissection Index (TDI)
    var_name = 'tdi'
    long_name = 'topographic dissection index'
    units = "[3,6,9,12,15] km radius" 
    a_rast = RasterDataset(os.path.join(PATH_RASTERS,'tdi.tif'))

    #Tmin TDI
    print "Adding TDI values to Tmin stations..."
    add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast)    
    
    #Tmax TDI
    print "Adding TDI values to Tmax stations..."
    add_stn_raster_values(stnda_tmax, var_name, long_name, units, a_rast)
    
    #Interpolation Mask
    var_name = 'mask'
    long_name = 'Interpolation Mask'
    units = "0 or 1"
    a_rast = RasterDataset(os.path.join(PATH_RASTERS,'mask_all_nd.tif'))
    a_rast.ndata = 0
    
    #Tmin interpolation mask
    print "Adding interpolation mask values to Tmin stations..."
    add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast, handle_ndata=False, nn=True)
    
    #Tmax interpolation mask
    print "Adding interpolation mask values to Tmax stations..."
    add_stn_raster_values(stnda_tmax, var_name, long_name, units, a_rast, handle_ndata=False, nn=True)
    
    #U.S. Climate Divisions
    var_name = CLIMDIV
    long_name = "U.S. Climate Division"
    units = "" 
    a_rast = RasterDataset(os.path.join(PATH_RASTERS,'climdivLccMerge.tif'))
    
    #Tmin U.S. Climate Divisions
    print "Adding U.S. climate division values to Tmin stations..."
    add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast,handle_ndata=False,nn=True)
    
    #Tmax U.S. Climate Divisions
    print "Adding U.S. climate division values to Tmax stations..."
    add_stn_raster_values(stnda_tmax, var_name, long_name, units, a_rast,handle_ndata=False,nn=True)
        