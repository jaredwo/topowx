'''
Created on Feb 18, 2013

@author: jared.oyler
'''
from twx.db.station_data import station_data_ncdb,LON,LAT
from twx.utils.timezone import TZGeonamesClient,GeonamesError
from twx.infill.obs_por import load_por_csv,build_valid_por_masks
import numpy as np
from twx.utils.status_check import status_check
import sys
import time

if __name__ == '__main__':
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc',mode='r+')
    varutc = db.add_stn_variable("utc_offset","utc offset","hours","i2")
    utc_off = np.ma.filled(varutc[:].astype(np.float),np.nan)
    
    tz = TZGeonamesClient('jaredntsg')
    
    #Load the period-of-record datafile
    por = load_por_csv('/projects/daymet2/station_data/all/all_por_1948_2012.csv')
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por)[0:2]
    
    mask_stns = np.logical_and(np.logical_or(mask_por_tmin,mask_por_tmax),np.isnan(utc_off))
    idxs = np.nonzero(mask_stns)[0]
    
    print idxs.size
    #sys.exit()
    
    statchk = status_check(idxs.size,50)
    
    for x in idxs:
        
        try:
            varutc[x] = tz.get_utc_offset(db.stns[LON][x], db.stns[LAT][x])
        except GeonamesError as e:
            if 'no timezone information' in e.status:
                print 'No timezone info for ' + str(db.stns[x])
            else:
                raise e
            
        db.ds.sync()
        statchk.increment()
        time.sleep(1.0)
    
    
    
    
    
