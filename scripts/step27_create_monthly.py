'''
Script to create monthly version of TopoWx. Aggregates the mosaiced daily 
netCDF files to monthly.
'''
from netCDF4 import Dataset
from twx.interp import write_ds_mthly
from twx.utils import TwxConfig, mkdir_p
import numpy as np
import os

if __name__ == '__main__':
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))    
    elems = ['tmin', 'tmax']
    yrs = np.arange(twx_cfg.interp_start_date.year,
                    twx_cfg.interp_end_date.year+1)
    
    for a_elem in elems:
    
        for a_yr in yrs:
        
            print "Processing %s for %d..." % (a_elem, a_yr)
            fpath_ds_in = os.path.join(twx_cfg.path_mosaic_daily, a_elem,
                                       '%s_%d.nc'%(a_elem,a_yr))
            path_ds_out = os.path.join(twx_cfg.path_mosaic_monthly, a_elem)
            mkdir_p(path_ds_out)
            fpath_ds_out = os.path.join(path_ds_out, '%s_%d.nc'%(a_elem, a_yr))
            
            write_ds_mthly(Dataset(fpath_ds_in), fpath_ds_out, a_elem, a_yr,
                           twx_cfg.twx_data_version)   