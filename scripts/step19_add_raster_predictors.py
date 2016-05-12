'''
Script to extract auxiliary predictors and other raster-based variables for each station.
'''

from twx.db import StationSerialDataDb, CLIMDIV, MASK, TDI
from twx.infill import add_stn_raster_values
from twx.raster import RasterDataset
from twx.utils import TwxConfig
import numpy as np
import os

if __name__ == '__main__':
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
        
    # Extract auxiliary predictors and other raster-based variables
    # for each station and add to final serially complete station observation
    # netCDF files
    
    # Load new serially complete dbs
    stnda_tmin = StationSerialDataDb(twx_cfg.fpath_stndata_nc_serial_tmin,
                                     'tmin', mode='r+')
    stnda_tmax = StationSerialDataDb(twx_cfg.fpath_stndata_nc_serial_tmax,
                                     'tmax', mode='r+')
    
    # Monthly Land Skin Temperature Predictors
    # Require each station to have a LST value by setting revert_nn=True, force_data_value=True
    # This accounts for coastal stations that might not fall within a grid cell that
    # has a LST value. MODIS coverage along the coasts is very jagged and incomplete.
    path_lst_rasters = os.path.join(twx_cfg.path_rasters, 'lst')
    long_name = 'land skin temperature'
    units = "C"
     
    # Nighttime LST
    print "Adding Nighttime LST values to Tmin stations..."
    for mth in np.arange(1, 13):
        var_name = 'lst%02d' % mth
        print var_name
        a_rast = RasterDataset(os.path.join(path_lst_rasters, 'LST_Night.%.2d.tif' % mth))
        add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast,
                              revert_nn=True, force_data_value=True)
      
    # Daytime LST
    print "Adding Daytime LST values to Tmax stations..."
    for mth in np.arange(1, 13):
        var_name = 'lst%02d' % mth
        print var_name 
        a_rast = RasterDataset(os.path.join(path_lst_rasters, 'LST_Day.%.2d.tif' % mth))
        add_stn_raster_values(stnda_tmax, var_name, long_name, units, a_rast,
                              revert_nn=True, force_data_value=True)

    # Topographic Dissection Index (TDI)
    # Unlike LST, TDI coverage is complete along the coasts, so do not set
    # force_data_value=True.
    var_name = TDI
    long_name = 'topographic dissection index'
    units = "[3,6,9,12,15] km radius" 
    a_rast = RasterDataset(os.path.join(twx_cfg.path_rasters, 'tdi.tif'))
 
    # Tmin TDI
    print "Adding TDI values to Tmin stations..."
    add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast, revert_nn=True)    

    # Tmax TDI
    print "Adding TDI values to Tmax stations..."
    add_stn_raster_values(stnda_tmax, var_name, long_name, units, a_rast, revert_nn=True)
     
    # Interpolation Mask
    var_name = MASK
    long_name = 'Interpolation Mask'
    units = "0 or 1"
    a_rast = RasterDataset(os.path.join(twx_cfg.path_rasters, 'mask.tif'))
    a_rast.ndata = 0
     
    # Tmin interpolation mask
    print "Adding interpolation mask values to Tmin stations..."
    add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast, extract_method=0)
    
    # Tmax interpolation mask
    print "Adding interpolation mask values to Tmax stations..."
    add_stn_raster_values(stnda_tmax, var_name, long_name, units, a_rast, extract_method=0)
     
    # U.S. Climate Divisions
    var_name = CLIMDIV
    long_name = "U.S. Climate Division"
    units = "" 
    a_rast = RasterDataset(os.path.join(twx_cfg.path_rasters, 'climdiv.tif'))
      
    # Tmin U.S. Climate Divisions
    print "Adding U.S. climate division values to Tmin stations..."
    add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast, extract_method=0)

    # Tmax U.S. Climate Divisions
    print "Adding U.S. climate division values to Tmax stations..."
    add_stn_raster_values(stnda_tmax, var_name, long_name, units, a_rast, extract_method=0)
    