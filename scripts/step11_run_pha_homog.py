'''
Script to homogenize station data using the Pairwise Homogenization Algorithm:

Menne, M.J., and C.N. Williams, Jr., 2009: Homogenization of temperature series 
via pairwise comparisons. J. Climate, 22, 1700-1717.
'''

from ftplib import FTP
from twx.db import add_obs_cnt
from twx.homog import HomogDaily
from twx.utils import DATE, mkdir_p, ymdL, TwxConfig
import numpy as np
import os
import twx

if __name__ == '__main__':
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    path_tmin_pha_src = os.path.join(twx_cfg.path_homog_pha, 'tmin', 'src')
    path_tmin_pha_run = os.path.join(twx_cfg.path_homog_pha, 'tmin', 'run')
    mkdir_p(path_tmin_pha_src)
    mkdir_p(path_tmin_pha_run)
    
    path_tmax_pha_src = os.path.join(twx_cfg.path_homog_pha, 'tmax', 'src')
    path_tmax_pha_run = os.path.join(twx_cfg.path_homog_pha, 'tmax', 'run')
    mkdir_p(path_tmax_pha_src)
    mkdir_p(path_tmax_pha_run)
    
    # Download PHA source code tgz file if not already on local filesystem
    if not os.path.exists(twx_cfg.fpath_pha_tgz):
          
        with open(twx_cfg.fpath_pha_tgz, 'wb') as f:
              
            print "Downloading phav52i.tar.gz from FTP..."
            ftp = FTP('ftp.ncdc.noaa.gov')
            ftp.login()
            ftp.retrbinary("RETR " +
                           'pub/data/ghcn/v3/software/52i/phav52i.tar.gz',
                           f.write)
            ftp.close()
         
    stnda = twx.db.StationDataDb(twx_cfg.fpath_stndata_nc_tair_tobs_adj)
    stns = stnda.stns
       
    # Perform PHA setup for Tmin
    mthly_tmin = np.ma.masked_invalid(stnda.xrds['tmin_mth'][:].values)
    twx.homog.setup_pha(twx_cfg.fpath_pha_tgz, path_tmin_pha_src, path_tmin_pha_run,
                        twx_cfg.obs_start_date.year, twx_cfg.obs_end_date.year,
                        stns, mthly_tmin, 'tmin')
    # Remove monthly Tmin observations from memory
    del mthly_tmin
    
    # Run PHA for Tmin
    twx.homog.run_pha(path_tmin_pha_run, 'tmin')
       
    # Perform PHA setup for Tmax
    mthly_tmax = np.ma.masked_invalid(stnda.xrds['tmax_mth'][:].values)
    twx.homog.setup_pha(twx_cfg.fpath_pha_tgz, path_tmax_pha_src, path_tmax_pha_run,
                        twx_cfg.obs_start_date.year, twx_cfg.obs_end_date.year,
                        stns, mthly_tmax, 'tmax')
    # Remove monthly Tmax observations from memory
    del mthly_tmax
              
    # Run PHA for Tmax
    twx.homog.run_pha(path_tmax_pha_run, 'tmax')
    
    # Use PHA results to homogenize daily station data and insert
    # into new homogenized database
    homog_tmin = HomogDaily(stnda, path_tmin_pha_run, 'tmin')
    homog_tmax = HomogDaily(stnda, path_tmax_pha_run, 'tmax')
           
    insert_homog = twx.homog.InsertHomog(stnda, homog_tmin, homog_tmax,
                                         path_tmin_pha_run, path_tmax_pha_run)
    twx.db.create_netcdf_db(twx_cfg.fpath_stndata_nc_tair_homog,
                            stnda.days[DATE][0], stnda.days[DATE][-1], [insert_homog])
    twx.db.insert_data_netcdf_db(twx_cfg.fpath_stndata_nc_tair_homog, [insert_homog])
       
    # Add period-of-record data to homogenized netcdf file
                 
    for elem in ['tmin', 'tmax']:
     
        print ("Updating monthly observation counts for %s from %d to %d... " % 
               (elem, ymdL(twx_cfg.obs_start_date),
                ymdL(twx_cfg.obs_end_date)))
 
        add_obs_cnt(twx_cfg.fpath_stndata_nc_tair_homog, elem,
                    twx_cfg.obs_start_date, twx_cfg.obs_end_date,
                    twx_cfg.stn_agg_chunk)
         
        print ("Updating monthly observation counts for %s from %d to %d... " % 
               (elem, ymdL(twx_cfg.interp_start_date), ymdL(twx_cfg.interp_end_date)))
         
        add_obs_cnt(twx_cfg.fpath_stndata_nc_tair_homog, elem,
                    twx_cfg.interp_start_date, twx_cfg.interp_end_date,
                    twx_cfg.stn_agg_chunk)