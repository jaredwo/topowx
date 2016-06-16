'''
Script to homogenize station data using the Pairwise Homogenization Algorithm:

Menne, M.J., and C.N. Williams, Jr., 2009: Homogenization of temperature series 
via pairwise comparisons. J. Climate, 22, 1700-1717.
'''

from ftplib import FTP
from obsio.factory import ObsIoFactory
from twx.db import add_obs_cnt, STN_ID, LON, LAT
from twx.homog import HomogDaily, load_snotel_sensor_hist
from twx.utils import DATE, mkdir_p, ymdL, TwxConfig
import numpy as np
import os
import pandas as pd
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
        
    # Include previously homogenized USHCN observations in PHA run
    # Load USHCN reference series
    obsiof = ObsIoFactory(elems=['tmin_mth_fls', 'tmax_mth_fls'],
                          start_date=twx_cfg.obs_start_date,
                          end_date=twx_cfg.obs_end_date)
    ushcnio = obsiof.create_obsio_mthly_ushcn(local_data_path=twx_cfg.path_stndata,
                                              download_updates=False)
    ref_ushcn = ushcnio.read_obs(data_structure='array')
    ref_ushcn = ref_ushcn.reindex({'time':stnda.xrds.time_mth})
    
    ref_stns = ref_ushcn[[STN_ID, LON, LAT]].to_dataframe().reset_index().to_records(index=False)
    ref_tmin = ref_ushcn.tmin_mth_fls.values
    ref_tmax = ref_ushcn.tmax_mth_fls.values
    
    # Include SNOTEL history metadata for YSI extended range sensor installs 
    # Load station history metadata for snotel
    stnhist = load_snotel_sensor_hist(stns[STN_ID])
    
    # Perform PHA setup for Tmin
    mthly_tmin = stnda.xrds['tmin_mth'][:].values
    mthly_tmin = np.hstack((mthly_tmin, ref_tmin))
    mthly_tmin = np.ma.masked_invalid(mthly_tmin)
      
    stns_all_tmin = pd.concat((pd.DataFrame(stns[[STN_ID,LON,LAT]]),
                               pd.DataFrame(ref_stns[[STN_ID,LON,LAT]])),
                               ignore_index=True).to_records(index=False)
         
    twx.homog.setup_pha(twx_cfg.fpath_pha_tgz, path_tmin_pha_src, path_tmin_pha_run,
                        twx_cfg.obs_start_date.year, twx_cfg.obs_end_date.year,
                        stns_all_tmin, mthly_tmin, 'tmin', stnhist)
    # Remove monthly Tmin observations from memory
    del mthly_tmin
    
    # Run PHA for Tmin
    twx.homog.run_pha(path_tmin_pha_run, 'tmin')
       
    # Perform PHA setup for Tmax
    mthly_tmax = stnda.xrds['tmax_mth'][:].values
    mthly_tmax = np.hstack((mthly_tmax, ref_tmax))
    mthly_tmax = np.ma.masked_invalid(mthly_tmax)
     
    stns_all_tmax = pd.concat((pd.DataFrame(stns[[STN_ID,LON,LAT]]),
                               pd.DataFrame(ref_stns[[STN_ID,LON,LAT]])),
                               ignore_index=True).to_records(index=False)
    
    twx.homog.setup_pha(twx_cfg.fpath_pha_tgz, path_tmax_pha_src, path_tmax_pha_run,
                        twx_cfg.obs_start_date.year, twx_cfg.obs_end_date.year,
                        stns_all_tmax, mthly_tmax, 'tmax', stnhist)
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
