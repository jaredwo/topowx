'''
Script to create additional auxiliary data and readmes as part of the final
TopoWx data outputs.
'''
from datetime import datetime, date
from twx.db import create_aux_db, StationDataDb
from twx.homog import get_pha_adj_df
from twx.utils import TwxConfig, create_main_readme, create_stnobs_readme,\
    create_auxgrid_readme
import glob
import netCDF4 as nc
import os
import pandas as pd
import twx

if __name__ == '__main__':
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    # Write Station observation netcdf files for tmin and tmax
    create_aux_db(twx_cfg.fpath_stndata_nc_all,
                  twx_cfg.fpath_stndata_nc_infill_tmin,
                  twx_cfg.fpath_stndata_nc_serial_tmin,
                  twx_cfg.fpath_stndata_nc_aux_tmin, 'tmin',
                  twx_cfg.interp_start_date.year,
                  twx_cfg.interp_end_date.year,
                  twx_cfg.twx_data_version)
    
    create_aux_db(twx_cfg.fpath_stndata_nc_all,
                  twx_cfg.fpath_stndata_nc_infill_tmax,
                  twx_cfg.fpath_stndata_nc_serial_tmax,
                  twx_cfg.fpath_stndata_nc_aux_tmax, 'tmax',
                  twx_cfg.interp_start_date.year,
                  twx_cfg.interp_end_date.year,
                  twx_cfg.twx_data_version)
    
    # Write CSV file of PHA adjustments
    stns = StationDataDb(twx_cfg.fpath_stndata_nc_all).stns
    
    def build_pha_log_fpath(elem):
        
        fpath_adj_log = os.path.join(twx_cfg.path_homog_pha, elem, 'run', 'data',
                                     'benchmark', 'world1','output',
                                     'pha_adj_%s.log'%elem)
        return fpath_adj_log
    
    pha_adj_tmin = get_pha_adj_df(build_pha_log_fpath('tmin'), stns, 'tmin')
    pha_adj_tmax = get_pha_adj_df(build_pha_log_fpath('tmax'), stns, 'tmax')
    
    pha_adj = pd.concat([pha_adj_tmin.reset_index(),
                         pha_adj_tmax.reset_index()],
                        ignore_index=True)
    pha_adj = pha_adj[['YEAR_MONTH_START', 'YEAR_MONTH_END', 'ADJ(C)',
                       'VARIABLE', 'STN_ID', 'NAME', 'LON', 'LAT', 'ELEV(m)']]
    pha_adj.to_csv(twx_cfg.fpath_pha_adj_aux, index=False)
    
    # Update metadata attributes on auxiliary grids
    fpaths_grids = sorted(glob.glob(os.path.join(twx_cfg.path_aux_grids,
                                                 'topowx_grids_netcdf', '*.nc')))
    for fpath_grid in fpaths_grids:
        
        ds = nc.Dataset(fpath_grid, 'r+')
        ds.source = ("TopoWx software version %s "
                     "(https://github.com/jaredwo/topowx)")%twx.__version__
        ds.history = "".join(["Created on: ",datetime.strftime(date.today(),
                                                               "%Y-%m-%d"),
                              " , ","dataset version %s"%twx_cfg.twx_data_version])
        ds.close()
    
    # Write READMEs
    create_main_readme(os.path.join(twx_cfg.path_final_output,'README.txt'),
                       twx_cfg.interp_start_date.year,
                       twx_cfg.interp_end_date.year,
                       twx.__version__,
                       twx_cfg.twx_data_version)
    
    create_stnobs_readme(os.path.join(twx_cfg.path_aux_stndata, 'README.txt'),
                         twx_cfg.interp_start_date.year,
                         twx_cfg.interp_end_date.year,
                         twx_cfg.fpath_stndata_nc_aux_tmin,
                         twx_cfg.fpath_stndata_nc_aux_tmax)
    
    create_auxgrid_readme(os.path.join(twx_cfg.path_aux_grids, 'README.txt'),
                          fpaths_grids)
    