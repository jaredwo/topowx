'''
Created on Jun 27, 2014

@author: jared.oyler
'''
import os
from twx.db import StationDataDb, UTC_OFFSET, UtcOffset
import netCDF4
import numpy as np

if __name__ == '__main__':
    
    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    
    path_homog_db = os.path.join(FPATH_STNDATA, 'all', 'tair_homog_1948_2012.nc')
    
    stnda = StationDataDb(path_homog_db, mode='r+') 
    
    var_utc = stnda.add_stn_variable(UTC_OFFSET,UTC_OFFSET,"hours","i2")
    
    ndata = netCDF4.default_fillvals['i2']
    geonames_usrname = open('/home/jared.oyler/.geonames_username').readline().strip()
    utc = UtcOffset(ndata, geonames_usrname)
    
    for x in np.arange(stnda.stns.size):
        
        var_utc[x] = 