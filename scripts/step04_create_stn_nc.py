'''
Script to insert station observation data into a single netCDF file.
'''

from twx.utils import TwxConfig
import obsio
import os
from twx.db import StationDataDb
import numpy as np

if __name__ == '__main__':

    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    ghcnd = obsio.HdfObsIO(twx_cfg.fpath_stndata_hdf_ghcnd)
    snotel = obsio.HdfObsIO(twx_cfg.fpath_stndata_hdf_snotel)
    raws = obsio.HdfObsIO(twx_cfg.fpath_stndata_hdf_raws)
    allio = obsio.MultiObsIO([ghcnd, snotel, raws])
    
    # Sort station inserts by station id
    stnids_sorted = np.sort(allio.stns.station_id.values)
    
    # Create netcdf
    allio.to_netcdf(twx_cfg.fpath_stndata_nc_all, stnids_sorted,
                    twx_cfg.obs_start_date, twx_cfg.obs_end_date,
                    chk_rw=twx_cfg.stn_write_chunk_nc, verbose=True)
    
    # Add QA Flag observation variable for each element
    stndb = StationDataDb(twx_cfg.fpath_stndata_nc_all, mode='r+')
    
    for elem in twx_cfg.obs_main_elems:
        
        varname = "qflag_"+elem
        long_name = "quality assurance flag "+elem
        stndb.add_obs_variable(varname, long_name, '', 'S1', fill_value='',
                               zlib=True, chunksizes=(stndb.days.size, 1),
                               reset=True)
        
    stndb.ds.close()