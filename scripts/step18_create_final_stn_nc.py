'''
Script for creating the final serially-complete station
databases to be used as input to the interpolation procedures.
Determines the final serially-complete time series to be used
at each station.
'''

from twx.db import StationSerialDataDb
from twx.infill import create_serially_complete_db, add_monthly_normals
from twx.utils import TwxConfig
import os

if __name__ == '__main__':
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    # Create final output serially-complete netCDF4 databases
    print "Creating final serially-complete databases..."
    create_serially_complete_db(twx_cfg.fpath_stndata_nc_infill_tmin, 'tmin',
                                twx_cfg.fpath_stndata_nc_serial_tmin)
    create_serially_complete_db(twx_cfg.fpath_stndata_nc_infill_tmax, 'tmax',
                                twx_cfg.fpath_stndata_nc_serial_tmax)
    
    # Load new serially complete dbs
    stnda_tmin = StationSerialDataDb(twx_cfg.fpath_stndata_nc_serial_tmin,
                                     'tmin', mode='r+')
    stnda_tmax = StationSerialDataDb(twx_cfg.fpath_stndata_nc_serial_tmax,
                                     'tmax', mode='r+')
         
    # Calculate 1981-2010 monthly normals for each station and
    # add to database
    print "Adding monthly normals to Tmin database..."
    add_monthly_normals(stnda_tmin, start_norm_yr=1981, end_norm_yr=2010)
    print "Adding monthly normals to Tmax database..."
    add_monthly_normals(stnda_tmax, start_norm_yr=1981, end_norm_yr=2010)