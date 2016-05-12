'''
Script for preparing NCEP/NCAR reanalysis data from
ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis
for input to the TopoWx missing value infilling procedures.
'''

from datetime import datetime
from twx.db import create_nnr_subset, create_thickness_nnr_subset,\
    create_nnr_subset_nolevel
from twx.utils import YEAR, get_days_metadata, TwxConfig
import numpy as np
import os
import shlex
import subprocess

if __name__ == '__main__':
        
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    # Download NCEP/NCAR reanalysis data
    cmd = ("wget --mirror -A 'air*.nc','hgt*.nc','rhum*.nc','vwnd*.nc','uwnd*.nc'"
           " -nd --directory-prefix "+twx_cfg.path_reanalysis_data+
           " ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/pressure/")
    subprocess.call(shlex.split(cmd))
    
    cmd = ("wget --mirror -A 'slp*.nc'"
           " -nd --directory-prefix "+twx_cfg.path_reanalysis_data+
           " ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/")
    subprocess.call(shlex.split(cmd))
    
    # Create NCEP/NCAR Reanalysis North American subsets of variables used in the
    # infilling of missing station observations.
    # Make the start and end years for processing NNR data the first and last
    # year + 1 year of the start and end dates for interpolation 
    start_date_nnr = datetime(twx_cfg.interp_start_date.year,1,1)
    end_date_nnr = datetime(twx_cfg.interp_end_date.year+1,12,31)
    days = get_days_metadata(start_date_nnr, end_date_nnr)
    yrs = np.unique(days[YEAR])
    # Conversion functions
    k_to_c = lambda k: k - 273.15
    no_conv = lambda x: x
       
    # 850mb Temperature at 12z, 18z, and 24z
    # Requires air.[year].nc files as input
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica,
                                   'nnr_tair_12z.nc'),
                      yrs, days, 'air', 'tair', np.array([850]), 12, k_to_c)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica,
                                   'nnr_tair_18z.nc'),
                      yrs, days, 'air', 'tair', np.array([850]), 18, k_to_c)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_tair_24z.nc'),
                      yrs, days, 'air', 'tair', np.array([850]), 24, k_to_c)
       
    # 500mb-1000mb Thickness at 12z, 18z, and 24z
    # Requires hgt.[year].nc files as input
    create_thickness_nnr_subset(twx_cfg.path_reanalysis_data,
                                os.path.join(twx_cfg.path_reanalysis_namerica, 
                                             'nnr_thick_12z.nc'),
                                yrs, days, 500, 1000, 12)
    create_thickness_nnr_subset(twx_cfg.path_reanalysis_data,
                                os.path.join(twx_cfg.path_reanalysis_namerica, 
                                             'nnr_thick_18z.nc'),
                                yrs, days, 500, 1000, 18)
    create_thickness_nnr_subset(twx_cfg.path_reanalysis_data,
                                os.path.join(twx_cfg.path_reanalysis_namerica, 
                                             'nnr_thick_24z.nc'),
                                yrs, days, 500, 1000, 24)
        
    # 500mb and 700mb height at 12z, 18z, and 24z
    # Requires hgt.[year].nc files as input
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_hgt_12z.nc'),
                      yrs, days, 'hgt', 'hgt', np.array([500, 700]), 12, no_conv)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_hgt_18z.nc'),
                      yrs, days, 'hgt', 'hgt', np.array([500, 700]), 18, no_conv)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_hgt_24z.nc'),
                      yrs, days, 'hgt', 'hgt', np.array([500, 700]), 24, no_conv)
    
    # 850mb relative humidity at 12z, 18z, and 24z
    # Requires rhum.[year].nc files as input
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_rhum_12z.nc'),
                      yrs, days, 'rhum', 'rhum', np.array([850]), 12, no_conv)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_rhum_18z.nc'),
                      yrs, days, 'rhum', 'rhum', np.array([850]), 18, no_conv)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_rhum_24z.nc'),
                      yrs, days, 'rhum', 'rhum', np.array([850]), 24, no_conv)
        
    # 850mb v-wind at 12z, 18z, and 24z
    # Requires vwnd.[year].nc files as input
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_vwnd_12z.nc'),
                      yrs, days, 'vwnd', 'vwnd', np.array([850]), 12, no_conv)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_vwnd_18z.nc'),
                      yrs, days, 'vwnd', 'vwnd', np.array([850]), 18, no_conv)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_vwnd_24z.nc'),
                      yrs, days, 'vwnd', 'vwnd', np.array([850]), 24, no_conv)
        
    # 850mb u-wind at 12z, 18z, and 24z
    # Requires uwnd.[year].nc files as input
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_uwnd_12z.nc'),
                      yrs, days, 'uwnd', 'uwnd', np.array([850]), 12, no_conv)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_uwnd_18z.nc'),
                      yrs, days, 'uwnd', 'uwnd', np.array([850]), 18, no_conv)
    create_nnr_subset(twx_cfg.path_reanalysis_data,
                      os.path.join(twx_cfg.path_reanalysis_namerica, 
                                   'nnr_uwnd_24z.nc'),
                      yrs, days, 'uwnd', 'uwnd', np.array([850]), 24, no_conv)
    
    # Sea level pressure at 12z, 18z, and 24z
    # Requires slp.[year].nc files as input
    create_nnr_subset_nolevel(twx_cfg.path_reanalysis_data,
                              os.path.join(twx_cfg.path_reanalysis_namerica, 
                                           'nnr_slp_12z.nc'),
                              yrs, days, 'slp', 'slp', 12, no_conv)
    create_nnr_subset_nolevel(twx_cfg.path_reanalysis_data,
                              os.path.join(twx_cfg.path_reanalysis_namerica, 
                                           'nnr_slp_18z.nc'),
                              yrs, days, 'slp', 'slp', 18, no_conv)
    create_nnr_subset_nolevel(twx_cfg.path_reanalysis_data,
                              os.path.join(twx_cfg.path_reanalysis_namerica, 
                                           'nnr_slp_24z.nc'),
                              yrs, days, 'slp', 'slp', 24, no_conv)
