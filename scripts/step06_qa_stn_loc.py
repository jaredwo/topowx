'''
Script to run station location quality assurance for stations in netCDF database.
'''
from twx.db import StationDataDb
from twx.db.obs_por import build_por_mask
from twx.qa.qa_location import LocQA
from twx.utils import StatusCheck
from twx.utils.config import TwxConfig
import numpy as np
import os


if __name__ == '__main__':

    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    stndb = StationDataDb(twx_cfg.fpath_stndata_nc_all)
    
    # To save processing time, only run location for stations with at least a 5
    # year period-of-record
    por_mask = build_por_mask(stndb.ds, twx_cfg.obs_main_elems,
                              twx_cfg.interp_start_date, twx_cfg.interp_end_date,
                              min_por_yrs=5)
    
    stns = stndb.stns_df[por_mask]
    
    # Load location QA HDF file
    locqa = LocQA(twx_cfg.fpath_locqa_hdf,
                  usrname_geonames=twx_cfg.username_geonames)
    # Add location QA data columns to stations
    stns = locqa.add_locqa_cols(stns)
    
    # Retrieve the DEM-based elevations for any stations that does not currently
    # have one and update the location QA HDF file
    stns_elevdem = stns[stns.elevation_dem.isnull()]
    
    print "Retrieving DEM elevation data for %d stations..." % len(stns_elevdem)
                
    write_chk = 50
    schk = StatusCheck(len(stns), check_cnt=write_chk)
    
    for i in np.arange(len(stns_elevdem), step=write_chk):
        
        stns_chk = stns_elevdem.iloc[i:(i + write_chk)].copy()
        
        for stnid in stns_chk.station_id:
            
            lon, lat = stns_chk.loc[stnid, ['longitude', 'latitude']]
            elevdem = locqa.get_elevation_dem(lon, lat)
            stns_chk.loc[stnid, 'elevation_dem'] = elevdem
            schk.increment()
            
        locqa.update_locqa_hdf(stns_chk, reload_locqa=False)
    
    locqa.reload_stns_locqa()
    stns = locqa.add_locqa_cols(stns)
    
    # Find stations that have a 200-m difference between their provided elevation
    # and the DEM-based elevation
    stns_fail = locqa.get_locqa_fail_stns(stns, elev_dif_thres=200)
    
    # Write out CSV file of failed station locations for manual investigtion
    print "%d stations failed location QA. Writing to %s" % (len(stns_fail),
                                                             twx_cfg.fpath_locqa_fail_csv)
    
    stns_fail['station_name'] = stns_fail.station_name.str.replace(',', ' ')
    stns_fail[['station_id', 'station_name', 'longitude', 'latitude',
               'elevation', 'elevation_dem', 'elevation_dif', 'longitude_qa',
               'latitude_qa', 'elevation_qa']].to_csv(twx_cfg.fpath_locqa_fail_csv,
                                                     index=False)
    locqa.close()
