'''
Script to add period-of-record daily observations counts to the station
observation netCDF file.
'''
import os
from twx.utils import TwxConfig, ymdL
from twx.db import add_obs_cnt


if __name__ == '__main__':

    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    for elem in twx_cfg.obs_main_elems:
        
        print ("Adding monthly observation counts for %s from %d to %d... " % 
               (elem, ymdL(twx_cfg.obs_start_date), ymdL(twx_cfg.obs_end_date)))
    
        add_obs_cnt(twx_cfg.fpath_stndata_nc_all, elem,
                    twx_cfg.obs_start_date, twx_cfg.obs_end_date,
                    twx_cfg.stn_agg_chunk)
        
        print ("Adding monthly observation counts for %s from %d to %d... " % 
               (elem, ymdL(twx_cfg.interp_start_date), ymdL(twx_cfg.interp_end_date)))
        
        add_obs_cnt(twx_cfg.fpath_stndata_nc_all, elem,
                    twx_cfg.interp_start_date, twx_cfg.interp_end_date,
                    twx_cfg.stn_agg_chunk)