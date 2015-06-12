'''
Script to homogenize station data using the Pairwise Homogenization Algorithm:

Menne, M.J., and C.N. Williams, Jr., 2009: Homogenization of temperature series 
via pairwise comparisons. J. Climate, 22, 1700-1717.

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

import os
import twx
from twx.homog import HomogDaily
from twx.utils import DATE, mkdir_p, ymdL
from twx.db import StationDataDb
from datetime import datetime

if __name__ == '__main__':
    
    PROJECT_ROOT = os.getenv('TOPOWX_DATA')
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    FPATH_PHA_RUNS = os.path.join(FPATH_STNDATA, 'pha')
    START_YEAR = 1895
    END_YEAR = 2015
    POR_START_YEAR = 1948
    POR_END_YEAR = 2014
    
    path_tmin_pha_src = os.path.join(FPATH_PHA_RUNS, 'tmin', 'src')
    path_tmin_pha_run = os.path.join(FPATH_PHA_RUNS, 'tmin', 'run')
    mkdir_p(path_tmin_pha_src)
    mkdir_p(path_tmin_pha_run)
    
    path_tmax_pha_src = os.path.join(FPATH_PHA_RUNS, 'tmax', 'src')
    path_tmax_pha_run = os.path.join(FPATH_PHA_RUNS, 'tmax', 'run')
    mkdir_p(path_tmax_pha_src)
    mkdir_p(path_tmax_pha_run)
    
    path_pha_tar = os.path.join(FPATH_PHA_RUNS, 'phav52i.tar.gz')
    
    path_stndb = os.path.join(FPATH_STNDATA, 'all', 'tair_tobs_adj_%s_%s.nc' % (START_YEAR, END_YEAR))
    
    stnda = twx.db.StationDataDb(path_stndb)
    stns = stnda.stns
      
    # Perform PHA setup for Tmin
    mthly_tmin = stnda.ds.variables['tmin_mth'][:]
    twx.homog.setup_pha(path_pha_tar, path_tmin_pha_src, path_tmin_pha_run,
                        START_YEAR, END_YEAR, stns, mthly_tmin, 'tmin')
    # Remove monthly Tmin observations from memory
    mthly_tmin = None
      
    # Perform PHA setup for Tmax
    mthly_tmax = stnda.ds.variables['tmax_mth'][:]
    twx.homog.setup_pha(path_pha_tar, path_tmax_pha_src, path_tmax_pha_run,
                        START_YEAR, END_YEAR, stns, mthly_tmax, 'tmax')
    # Remove monthly Tmax observations from memory
    mthly_tmax = None
      
    # Run PHA for Tmin
    twx.homog.run_pha(path_tmin_pha_run, 'tmin')
      
    # Run PHA for Tmax
    twx.homog.run_pha(path_tmax_pha_run, 'tmax')
    
    # Use PHA results to homogenize daily station data and insert
    # into new homogenized database
    homog_tmin = HomogDaily(stnda, path_tmin_pha_run, 'tmin')
    homog_tmax = HomogDaily(stnda, path_tmax_pha_run, 'tmax')
       
    path_out_homog_db = os.path.join(FPATH_STNDATA, 'all', 'tair_homog_%s_%s.nc' % (START_YEAR, END_YEAR))    
    insert_homog = twx.homog.InsertHomog(stnda, homog_tmin, homog_tmax, path_tmin_pha_run, path_tmax_pha_run)
    twx.db.create_netcdf_db(path_out_homog_db, stnda.days[DATE][0], stnda.days[DATE][-1], [insert_homog])
    twx.db.insert_data_netcdf_db(path_out_homog_db, [insert_homog])
      
    # Create a period-of-record file for the homogenized database
    fpath_por_out = os.path.join(FPATH_STNDATA, 'all', 'homog_por_%s_%s.csv' % (POR_START_YEAR, POR_END_YEAR))
    stn_da = StationDataDb(path_out_homog_db, startend_ymd=(ymdL(datetime(POR_START_YEAR, 1, 1)),
                                                           ymdL(datetime(POR_END_YEAR, 12, 31))))
    stns = stn_da.stns
    twx.db.output_por_csv(stn_da, stns, fpath_por_out)
    
    
