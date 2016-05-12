'''
Script to download RAWS station data from WRCC.
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
    raws = obsiof.create_obsio_dly_wrcc_raws(nprocs=5)
    
    # Only download RAWS stations that are part of GHCND (i.e.-permanent stations)
    ghcnd = obsiof.create_obsio_dly_ghcnd()
    ghcndraws_ids = ghcnd.stns.loc[ghcnd.stns.sub_provider=='RAWS',
                                   'station_id'].str[-4:].values
    stns = raws.stns[raws.stns.station_id.isin(ghcndraws_ids)]
    
    # Download observations to PyTables HDF file.
    # Downloads from NRCS raws can take several days
    raws.to_hdf(fpath=twx_cfg.fpath_stndata_hdf_raws,
                stn_ids=stns.station_id, chk_rw=twx_cfg.stn_read_chunk_raws,
                verbose=True, complevel=5, complib='zlib')