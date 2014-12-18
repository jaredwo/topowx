'''
Script to insert station observation data from GHCN-D, SNOTEL, and RAWS
into a single netCDF database file.

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

import os
from datetime import datetime
from twx.db import InsertGhcn, InsertSnotel, InsertRaws, StationDataDb
import twx

if __name__ == '__main__':

    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')

    # The period-of-record for the database
    min_date = datetime(1948, 1, 1)
    max_date = datetime(2012, 12, 31)

    # Build Insert objects for inserting GHCN-D, SNOTEL, and RAWS data
    # into a netCDF database file
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

    # Create and initialize the database
    fpath_db = os.path.join(FPATH_STNDATA, 'all', 'all_1948_2012.nc')
    inserts = [insert_ghcn, insert_snotel, insert_raws]
    twx.db.create_netcdf_db(fpath_db, min_date, max_date, inserts)

    # Insert all data
    twx.db.insert_data_netcdf_db(fpath_db, inserts)
    
    #Calculate and add monthly data to database
    twx.db.add_monthly_means(fpath_db, 'tmin')
    twx.db.add_monthly_means(fpath_db, 'tmax')

    # Create a period-of-record file for the database
    fpath_por_out = os.path.join(FPATH_STNDATA, 'all', 'all_por_1948_2012.csv')
    stn_da = StationDataDb(fpath_db, startend_ymd=(19480101, 20121231))
    stns = stn_da.stns
    twx.db.output_por_csv(stn_da, stns, fpath_por_out)
