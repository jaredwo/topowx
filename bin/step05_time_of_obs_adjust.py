'''
Script to process time-of-observation data from GHCN-Daily,
perform time-of-observation adjustments, and output a new
time-of-observation adjusted netCDF station observation database.

Copyright 2014,2015, Jared Oyler.

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
import os
import numpy as np
from datetime import datetime
from twx.utils import DATE

if __name__ == '__main__':
    
    PROJECT_ROOT = os.getenv('TOPOWX_DATA')
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    START_YEAR = 1895
    END_YEAR = 2015
    
    # Create a file of Tmax time-of-observation (tobs) times for observations
    # that have a non-calendar day observation time
    path_ghcn_yrly = os.path.join(FPATH_STNDATA, 'ghcn', 'ghcn_yrly')
    fpath_out_tobs_file = os.path.join(FPATH_STNDATA, 'ghcn', 'tobs_tmax.csv')
    yrs = np.arange(START_YEAR, END_YEAR + 1)
    twx.homog.create_tobs_file(path_ghcn_yrly, fpath_out_tobs_file, yrs, element='TMAX')
    
    # Using tobs file, create a netCDF database of tobs times
    # Only save tobs times for stations that meet the period of record requirements
    fpath_db_tobs_out = os.path.join(FPATH_STNDATA, 'ghcn', 'tobs_tmax.nc')
    fpath_por = os.path.join(FPATH_STNDATA, 'all', 'all_por_%d_%d.csv' % (START_YEAR, END_YEAR))
    fpath_stn_db = os.path.join(FPATH_STNDATA, 'all', 'all_%d_%d.nc' % (START_YEAR, END_YEAR))
    por = twx.db.load_por_csv(fpath_por)
    # require 1 year of observations
    mask_por_tmin, mask_por_tmax = twx.db.build_valid_por_masks(por, min_por=1) 
    mask_por = np.logical_or(mask_por_tmin, mask_por_tmax)
    stnda = twx.db.StationDataDb(fpath_stn_db)
    stnids = stnda.stn_ids[mask_por]
    min_date = datetime(START_YEAR, 1, 1)
    max_date = datetime(END_YEAR, 12, 31)    
    twx.homog.create_tobs_db(fpath_out_tobs_file, fpath_db_tobs_out, stnids, min_date, max_date)
     
    # Create a new station database that has tobs adjustments
    # Only insert stations that meet the period of record requirements
    insert_tobs = twx.homog.InsertTobs(stnda, fpath_db_tobs_out, stnda.stns[mask_por_tmin], stnda.stns[mask_por_tmax])
    fpath_db_tobs_adj_out = os.path.join(FPATH_STNDATA, 'all', 'tair_tobs_adj_%d_%d.nc' % (START_YEAR, END_YEAR))
    twx.db.create_netcdf_db(fpath_db_tobs_adj_out, stnda.days[DATE][0], stnda.days[DATE][-1], [insert_tobs])
    twx.db.insert_data_netcdf_db(fpath_db_tobs_adj_out, [insert_tobs])
     
    # Add monthly means to the new station database
    twx.db.add_monthly_means(fpath_db_tobs_adj_out, 'tmin')
    twx.db.add_monthly_means(fpath_db_tobs_adj_out, 'tmax')
    
