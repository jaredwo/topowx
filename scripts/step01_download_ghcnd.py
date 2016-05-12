'''
Script to download GHCN-D station data from NCEI.
'''

import obsio
import os
from twx.utils import TwxConfig


if __name__ == '__main__':


    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    # Bounding box for stations
    bbox = obsio.BBox(*twx_cfg.stn_bbox)
        
    obsiof = obsio.ObsIoFactory(twx_cfg.obs_elems, bbox, twx_cfg.obs_start_date,
                                twx_cfg.obs_end_date)
    ghcnd = obsiof.create_obsio_dly_ghcnd(nprocs=3, bulk=True,
                                          local_data_path=twx_cfg.path_stndata,
                                          download_updates=True)
    
    # Do not process stns that are in WRCC RAWS and NRCS SNOTEL
    stns = ghcnd.stns[((ghcnd.stns.sub_provider != 'RAWS') &
                       (ghcnd.stns.sub_provider != 'SNOTEL'))]
    
    # Download observations to PyTables HDF file
    ghcnd.to_hdf(fpath=twx_cfg.fpath_stndata_hdf_ghcnd,
                 stn_ids=stns.station_id,
                 chk_rw=twx_cfg.stn_read_chunk_ghcnd, verbose=True,
                 complevel=5, complib='zlib')