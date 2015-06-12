'''
Script to download GHCN-D, SNOTEL, and RAWS station data
from NCDC, NRCS, and WRCC.

Copyright 2014,2015 Jared Oyler.

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
import sys
from twx.db import SnotelDataService
from twx.utils import Unbuffered

sys.stdout = Unbuffered(sys.stdout)

if __name__ == '__main__':

    PROJECT_ROOT = os.getenv('TOPOWX_DATA')
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    START_YEAR = 1895
    END_YEAR = 2015
 
    # Download GHCN-D data from NCDC
    twx.db.ghcnd_download_data(os.path.join(FPATH_STNDATA, 'ghcn'))
  
    # Download GHCN-D data from NCDC in "by year" format
    # Need this for time-of-observation data
    twx.db.ghcnd_download_byyr_data(os.path.join(FPATH_STNDATA, 'ghcn', 'ghcn_yrly'),
                                                 yrs=np.arange(START_YEAR, END_YEAR + 1))
 
    # Download SNOTEL/SCAN data from NRCS in CSV format
    sntl = SnotelDataService()
     
    # don't download observations from Alaska,
    # Antarctica, Hawaii, Puerto Rico, Virgin Islands
    # and those without a CDBS ID (the main SNOTEL station ID)
    skip_states = ['AK', 'AY', 'HI', 'PR', 'VI']
    stnids_snotel = sntl.stns_df['station id'][(~sntl.stns_df['state'].isin(skip_states)) & (sntl.stns_df['cdbs_id'] != '')]    
     
    print "Downloading all observations for %d snotel/scan stations." % stnids_snotel.size  
    path_out_snotel_csv = os.path.join(FPATH_STNDATA, 'snotel', 'csv')
       
    for a_id in stnids_snotel:
           
        sntl.write_stn_obs(a_id, path_out_snotel_csv, userEmail='[put user email here]')
      
    # Download Raws from WRCC (http://www.raws.dri.edu) using web crawler
  
    # Generate list of RAWS station ids that are available on the WRCC website
    fpath_raws_ids = os.path.join(FPATH_STNDATA, 'raws', 'raws_stnids.txt')
    twx.db.raws_save_stnid_list(fpath_raws_ids)
    
    # Build a RAWS station metadata file
    fpath_raws_meta = os.path.join(FPATH_STNDATA, 'raws', 'raws_meta.txt')
    twx.db.raws_build_stn_metadata(fpath_raws_ids, fpath_raws_meta)
 
    # Download the actual daily RAWS data for each station.
    # This can take several days
    fpath_raws_dly_data = os.path.join(FPATH_STNDATA, 'raws', 'data')
    twx.db.raws_save_all_dly_series(fpath_raws_meta, fpath_raws_dly_data)
