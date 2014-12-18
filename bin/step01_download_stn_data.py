'''
Script to download GHCN-D, SNOTEL, and RAWS station data
from NCDC, NRCS, and WRCC.

Copyright 2014, Jared Oyler.

This file is part of TopoWx.

TopoWx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TopoWx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.
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
    #Cleanup SNOTEL station data
    snotel_path_hist = os.path.join(FPATH_STNDATA, 'snotel', 'historical')
    snotel_fpath_locs = os.path.join(FPATH_STNDATA, 'snotel', 'locations',
                                     '2011-03-31-WCC-high-resolution-snotel.csv')
    snotel_fpath_meta = os.path.join(FPATH_STNDATA, 'snotel', 'cleaned',
                                     'snotel_stns.csv')
    snotel_path_tab = os.path.join(FPATH_STNDATA, 'snotel', 'current',
                                   'ftp.wcc.nrcs.usda.gov', 'data', 'snow',
                                   'snotel', 'cards')
    snotel_path_clean_obs = os.path.join(FPATH_STNDATA, 'snotel', 'cleaned')

    twx.db.snotel_write_stn_metadata(snotel_path_hist, snotel_fpath_locs,
                                     snotel_fpath_meta)

    twx.db.snotel_find_no_metadata_stns(snotel_path_tab, snotel_fpath_meta)

    twx.db.snotel_write_stn_obs(snotel_path_hist, snotel_path_tab, snotel_fpath_meta,
                                snotel_path_clean_obs)

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
