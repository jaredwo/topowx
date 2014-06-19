'''
Script to insert station observation data from GHCN-D, SNOTEL, and RAWS
into a single netCDF database file 
'''
import os
from datetime import datetime
from twx.db import InsertGhcn, InsertSnotel, InsertRaws

if __name__ == '__main__':

    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')

    min_date = datetime(1948, 1, 1)
    max_date = datetime(2012, 12, 31)

    ghcn_path_stn_file = os.path.join(FPATH_STNDATA, 'ghcn', 'ghcnd-stations.txt')
    ghcn_path_ob = os.path.join(FPATH_STNDATA, 'ghcn', 'ghcnd_all')
    insert_ghcn = InsertGhcn(ghcn_path_stn_file, ghcn_path_ob, min_date, max_date)

    snotel_path_obs = os.path.join(FPATH_STNDATA, 'snotel', 'cleaned')
    snotel_path_stn_file = os.path.join(FPATH_STNDATA, 'snotel', 'cleaned', 'snotel_stns.csv')
    insert_snotel = InsertSnotel(snotel_path_stn_file, snotel_path_obs, min_date, max_date)

    raws_path_meta = os.path.join(FPATH_STNDATA, 'raws', 'raws_meta.txt')
    raws_path_stnids = os.path.join(FPATH_STNDATA, 'raws', 'raws_ghcn_stnids.txt')
    raws_path_obs = os.path.join(FPATH_STNDATA, 'raws', 'data')
    insert_raws = InsertRaws(raws_path_meta, raws_path_stnids, raws_path_obs, min_date, max_date)