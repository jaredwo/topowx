'''
Script to update locations of stations that failed location quality assurance
and had their location manually corrected.
'''

from twx.db import StationDataDb, build_por_mask, LON, LAT, ELEV
from twx.qa import LocQA
from twx.utils import TwxConfig
import numpy as np
import os
import pandas as pd

if __name__ == '__main__':

    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    # Load station locations that failed QA but were manually fixed
    stns_fixed_loc = pd.read_csv(twx_cfg.fpath_locqa_fail_csv)
    stns_fixed_loc = stns_fixed_loc.set_index('station_id', drop=False)
    stns_fixed_loc = stns_fixed_loc.dropna(how='any', subset=['longitude_qa',
                                                              'latitude_qa',
                                                              'elevation_qa'])
    # Load location QA store
    locqa = LocQA(twx_cfg.fpath_locqa_hdf,
                  usrname_geonames=twx_cfg.username_geonames)
    # Update store with the new fixed station locations
    locqa.update_locqa_hdf(stns_fixed_loc)
    
    # Get stations with at least a 5 year period-of-record
    stndb = StationDataDb(twx_cfg.fpath_stndata_nc_all, mode='r+')
    por_mask = build_por_mask(stndb.ds, twx_cfg.obs_main_elems,
                              twx_cfg.interp_start_date, twx_cfg.interp_end_date,
                              min_por_yrs=5)
    stns = stndb.stns_df[por_mask]
    
    # Add location QA columns
    stns = locqa.add_locqa_cols(stns)
    
    # Subset to stations that have QA location to be updated
    stns_update_loc = stns.dropna(how='any', subset=['longitude_qa',
                                                     'latitude_qa',
                                                     'elevation_qa'])
    
    # Get netcdf station_id index for stations to be updated
    # Updates need to sorted by index
    i_stns = stndb.stns_df.loc[stns_update_loc.index, 'station_index'].values
    i_sort = np.argsort(i_stns)
    i_stns = i_stns[i_sort]
    
    update_lon = stns_update_loc.longitude_qa.values[i_sort]
    update_lat = stns_update_loc.latitude_qa.values[i_sort]
    update_elev = stns_update_loc.elevation_qa.values[i_sort]
    
    print "Updating locations of %d stations..."%i_stns.size
    
    stndb.ds[LON][i_stns] = update_lon
    stndb.ds[LAT][i_stns] = update_lat
    stndb.ds[ELEV][i_stns] = update_elev
    stndb.ds.sync()
    stndb.ds.close()
    locqa.close()
