'''
Script to download GHCN-D, SNOTEL, and RAWS station data
from NCDC, NRCS, and WRCC.
'''

import twx
import numpy as np
import os

if __name__ == '__main__':

    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')

    #Download GHCN-D data from NCDC
    twx.db.ghcnd_download_data(os.path.join(FPATH_STNDATA, 'ghcn'))

    #Download GHCN-D data from NCDC in "by year" format for 1948-2013
    twx.db.ghcnd_download_byyr_data(os.path.join(FPATH_STNDATA,
                                                 'ghcn', 'ghcn_yrly'),
                                   yrs=np.arange(1948, 2014))

    #Mirror SNOTEL tab-delimited data from NRCS FTP
    twx.db.snotel_mirror_tabdata(os.path.join(FPATH_STNDATA,
                                              'snotel', 'current'))

    snotel_path_hist = os.path.join(FPATH_STNDATA, 'snotel', 'historical')
    snotel_fpath_locs = os.path.join(FPATH_STNDATA, 'snotel', 'locations',
                                     '2011-03-31-WCC-high-resolution-snotel.csv')
    snotel_fpath_meta = os.path.join(FPATH_STNDATA, 'snotel', 'cleaned',
                                     'snotel_stns.csv')
    snotel_path_tab = os.path.join(FPATH_STNDATA, 'snotel', 'current',
                                   'ftp.wcc.nrcs.usda.gov', 'data', 'snow',
                                   'snotel', 'cards')

    twx.db.snotel_write_stn_metadata(snotel_path_hist, snotel_fpath_locs,
                                     snotel_fpath_meta)

    twx.db.snotel_find_no_metadata_stns(snotel_path_tab, snotel_fpath_meta)
    
    
    #    find_no_metadata_stns('/projects/daymet2/station_data/snotel/current/ftp.wcc.nrcs.usda.gov/data/snow/snotel/cards/',
#                          '/projects/daymet2/station_data/snotel/cleaned/snotel_stns.csv')

    #Download Raws from WRCC (http://www.raws.dri.edu) using web crawler

    #Generate list of RAWS station ids that are available on the WRCC website
    fpath_raws_ids = os.path.join(FPATH_STNDATA, 'raws', 'raws_stnids.txt')
    twx.db.raws_save_stnid_list(fpath_raws_ids)

    #Subset RAWS stations to only those considered permanent in GHCN-D
    fpath_ghcn_stns = os.path.join(FPATH_STNDATA, 'ghcn', 'ghcnd-stations.txt')
    fpath_raws_ids_ghcn = os.path.join(FPATH_STNDATA, 'raws', 'raws_ghcn_stnids.txt')
    twx.db.raws_to_ghcn_subset(fpath_raws_ids, fpath_ghcn_stns, fpath_raws_ids_ghcn)

    #Build a RAWS station metadata file
    fpath_raws_meta = os.path.join(FPATH_STNDATA, 'raws', 'raws_meta.txt')
    twx.db.raws_build_stn_metadata(fpath_raws_ids_ghcn, fpath_raws_meta)

    #Download the actual daily RAWS data for each station.
    #This can take several days
    fpath_raws_dly_data = os.path.join(FPATH_STNDATA, 'raws', 'data')
    twx.db.raws_save_all_dly_series(fpath_raws_meta, fpath_raws_dly_data)
