'''
Script for creating the final serially-complete station
databases to be used as input to the interpolation procedures.
Determines the final serially-complete time series to be used
at each station, sets flags on bad and duplicate stations,
adds monthly normal values to each station, and extracts auxiliary 
predictors and other raster-based variables for each station.

Copyright 2014,2015 Jared Oyler.

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

from twx.db import StationSerialDataDb, BAD, CLIMDIV
from twx.infill import create_serially_complete_db, \
set_bad_stations, find_dup_stns, add_stn_raster_values, add_monthly_normals
from twx.raster import RasterDataset
import os
import numpy as np
from twx.db.station_data import MASK, TDI
from twx.interp import XvalOutlier

if __name__ == '__main__':
    
    PROJECT_ROOT = os.getenv('TOPOWX_DATA')
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    PATH_RASTERS = os.path.join(PROJECT_ROOT, 'rasters')
    
    # Paths to infilled netCDF4 databases
    path_infill_db_tmin = os.path.join(FPATH_STNDATA, 'infill', 'infill_tmin.nc')
    path_infill_db_tmax = os.path.join(FPATH_STNDATA, 'infill', 'infill_tmax.nc')
    
    # Paths to final output serially-complete netCDF4 databases
    path_serial_db_tmin = os.path.join(FPATH_STNDATA, 'infill', 'serial_tmin.nc')
    path_serial_db_tmax = os.path.join(FPATH_STNDATA, 'infill', 'serial_tmax.nc')
    
    # Create final output serially-complete netCDF4 databases
    print "Creating final serially-complete databases..."
    create_serially_complete_db(path_infill_db_tmin, 'tmin', path_serial_db_tmin)
    create_serially_complete_db(path_infill_db_tmax, 'tmax', path_serial_db_tmax)
    
    # Load new serially complete dbs
    stnda_tmin = StationSerialDataDb(path_serial_db_tmin, 'tmin', mode='r+')
    stnda_tmax = StationSerialDataDb(path_serial_db_tmax, 'tmax', mode='r+')
         
    # Calculate 1981-2010 monthly normals for each station and
    # add to database
    print "Adding monthly normals to Tmin database..."
    add_monthly_normals(stnda_tmin, start_norm_yr=1981, end_norm_yr=2010)
    print "Adding monthly normals to Tmax database..."
    add_monthly_normals(stnda_tmax, start_norm_yr=1981, end_norm_yr=2010)
    
    # Extract auxiliary predictors and other raster-based variables
    # for each station and add to station database
    
    # Monthly Land Skin Temperature Predictors
    # Require each station to have a LST value by setting revert_nn=True, force_data_value=True
    # This accounts for coastal stations that might not fall within a grid cell that
    # has a LST value. MODIS coverage along the coasts is very jagged and incomplete.
    path_lst_rasters = os.path.join(PATH_RASTERS, 'lst')
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
    a_rast = RasterDataset(os.path.join(PATH_RASTERS, 'tdi.tif'))
 
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
    a_rast = RasterDataset(os.path.join(PATH_RASTERS, 'mask.tif'))
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
    a_rast = RasterDataset(os.path.join(PATH_RASTERS, 'climdiv.tif'))
      
    # Tmin U.S. Climate Divisions
    print "Adding U.S. climate division values to Tmin stations..."
    add_stn_raster_values(stnda_tmin, var_name, long_name, units, a_rast, extract_method=0)

    # Tmax U.S. Climate Divisions
    print "Adding U.S. climate division values to Tmax stations..."
    add_stn_raster_values(stnda_tmax, var_name, long_name, units, a_rast, extract_method=0)
    
    # Determine stations that should be marked as "bad" and not be used in interpolations
    # going forward
    
    # Reload station databases, so that all extracted raster values are loaded
    del stnda_tmin 
    del stnda_tmax
    stnda_tmin = StationSerialDataDb(path_serial_db_tmin, 'tmin', mode='r+')
    stnda_tmax = StationSerialDataDb(path_serial_db_tmax, 'tmax', mode='r+')
    stnda_infill_tmin = StationSerialDataDb(path_infill_db_tmin, 'tmin')
    stnda_infill_tmax = StationSerialDataDb(path_infill_db_tmax, 'tmax')
    
    # Load station ids that were manually marked as "bad" due to failed infilling,
    # strange inhomogeneities, etc.
    bad_stnids = np.unique(np.loadtxt(os.path.join(FPATH_STNDATA, 'infill', 'bad_stns.csv'),
                                      delimiter=",", dtype=np.str, skiprows=1, usecols=(0,)))
    
    for a_stnda, a_istnda in zip([stnda_tmin, stnda_tmax],
                                [stnda_infill_tmin, stnda_infill_tmax]):
        
        # Add a 'bad' station variable
        a_stnda.add_stn_variable(BAD, "bad station flag", units='', dtype='i1', fill_value=0)

        # Find stations that do not have a TDI value. All
        # stations should have a TDI value. If a TDI value could not be obtained for
        # a station, it lies far outside the applicable DEM domain.
        stnids_no_tdi = a_stnda.stn_ids[np.isnan(a_stnda.stns[TDI])]
    
        # Find stations that are in the interpolation mask, but do not have a climate division.
        # All stations within the interpolation mask should fall within a climate division
        mask_no_climdiv = np.logical_and(np.isnan(a_stnda.stns[CLIMDIV]),
                                         np.isfinite(a_stnda.stns[MASK]))
        stnids_no_climdiv = a_stnda.stn_ids[mask_no_climdiv]
        
        # Find duplicate stations.
        # The kriging interpolation algorithm cannot have point observations
        # at the exact same location. For two or more stations with the same
        # location, the one with the longest non-infilled period-of-record is
        # used and the others are considered duplicates.
        print "Finding duplicate stations for %s..." % (a_stnda.var_name,)
        dup_stnids = find_dup_stns(a_istnda)
        
        all_rm_stnids = np.unique(np.concatenate([bad_stnids, stnids_no_tdi,
                                                  stnids_no_climdiv, dup_stnids]))
        
        set_bad_stations(a_stnda.ds, all_rm_stnids)
        
        print "%d total stations removed for %s:" % (all_rm_stnids.size, a_stnda.var_name)
        print "%d stations removed due to being manually marked as bad" % (bad_stnids.size,)
        print "%d stations removed due to no TDI values" % (stnids_no_tdi.size,)
        print "%d stations removed due no climate division value, but within domain mask" % (stnids_no_climdiv.size,)
        print "%d stations removed due to being duplicates" % (dup_stnids.size,)

    # Last step of marking "bad" stations. Run a cross validation of a simple geographically
    # weighted regression model that predicts annual temperature normals. 
    # Find stations with annual normals that are extremely different than what is predicted
    # by the model (e.g. error > 6 standard deviations from the mean error).
    
    # Reload station databases to make sure bad flags are correctly set
    del stnda_tmin 
    del stnda_tmax
    del stnda_infill_tmin 
    del stnda_infill_tmax
    stnda_tmin = StationSerialDataDb(path_serial_db_tmin, 'tmin', mode='r+')
    stnda_tmax = StationSerialDataDb(path_serial_db_tmax, 'tmax', mode='r+')

    for a_stnda in [stnda_tmin, stnda_tmax]:
        
        print "Finding outlier stations for %s..." % (a_stnda.var_name,)
        
        out_xval = XvalOutlier(a_stnda)
        stn_ids = a_stnda.stn_ids[np.isnan(a_stnda.stns[BAD])]
        
        out_stnids = out_xval.find_xval_outliers(stn_ids)[0]
        
        set_bad_stations(a_stnda.ds, out_stnids, reset=False)
        
        print "%d stations removed due to being outliers" % (out_stnids.size,)
        
