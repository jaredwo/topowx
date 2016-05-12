'''
Script to process time-of-observation data from GHCN-Daily,
perform time-of-observation adjustments, and output a new
time-of-observation adjusted netCDF station observation database.
'''
import twx
import os
import numpy as np
from twx.utils import DATE, TwxConfig
from twx.db import build_por_mask, StationDataDb

if __name__ == '__main__':
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    stndb = StationDataDb(twx_cfg.fpath_stndata_nc_all)
    
    # Require 1 year of observations for station to be inserted into the
    # time-of-observation adjusted netcdf file
    por_mask_tmin = build_por_mask(stndb.ds, ['tmin'],
                                   twx_cfg.obs_start_date, twx_cfg.obs_end_date,
                                   min_por_yrs=1)
    por_mask_tmax = build_por_mask(stndb.ds, ['tmax'],
                                   twx_cfg.obs_start_date, twx_cfg.obs_end_date,
                                   min_por_yrs=1)
    por_mask = np.logical_or(por_mask_tmin, por_mask_tmax)
         
    # Create a new station database that has tobs adjustments
    # Only insert stations that meet the period of record requirements
    insert_tobs = twx.homog.InsertTobs(stndb, stndb.stns[por_mask_tmin],
                                       stndb.stns[por_mask_tmax])
    twx.db.create_netcdf_db(twx_cfg.fpath_stndata_nc_tair_tobs_adj,
                            stndb.days[DATE][0], stndb.days[DATE][-1],
                            [insert_tobs])
    twx.db.insert_data_netcdf_db(twx_cfg.fpath_stndata_nc_tair_tobs_adj,
                                 [insert_tobs])
     
#     # Add monthly means to the new station database
#     twx.db.add_monthly_means(fpath_db_tobs_adj_out, 'tmin')
#     twx.db.add_monthly_means(fpath_db_tobs_adj_out, 'tmax')
    
