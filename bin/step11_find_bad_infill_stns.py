'''
Script to output station ids whose infilling in step10 did not
converge to a reasonable solution.  

Copyright 2015, Jared Oyler.

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

import readline
import os
from twx.infill.post_infill import get_bad_infill_stnids
import pandas as pd
import numpy as np
import xray
import twx.infill.infill_daily as idly
idly._load_R()

if __name__ == '__main__':
    
    PROJECT_ROOT = os.getenv('TOPOWX_DATA')
    FPATH_STNDATA_INFILL = os.path.join(PROJECT_ROOT, 'station_data', 'infill')
    FPATH_INFILL_LOGFILE = os.path.join(PROJECT_ROOT, 'mpi_runs',
                                        'step10_mpi_infill_stn_daily_1948_2014_run2.log')
    
    # World records for daily Tmax and Tmin in degrees C
    TMAX_RECORD = 57.7
    TMIN_RECORD = -89.4
    
    stnids = get_bad_infill_stnids(FPATH_INFILL_LOGFILE)
        
    path_infill_db_tmin = os.path.join(FPATH_STNDATA_INFILL, 'infill_tmin.nc')
    path_infill_db_tmax = os.path.join(FPATH_STNDATA_INFILL, 'infill_tmax.nc')
    
    ds_tmin = xray.open_dataset(path_infill_db_tmin)
    ds_tmax = xray.open_dataset(path_infill_db_tmax)
    
    # For each station marked as suspect in the infill log, check to see
    # if there are any major variance changepoints across the station's
    # full time series. If so, mark the station as "bad".
    
    bad_ids = []
    
    for a_id in stnids:
            
        try:
            tmin = ds_tmin.tmin.loc[:, a_id].values
            chgpts_tmin = np.array(idly.r.getVarChgPt(idly.robjects.FloatVector(tmin)))
            has_imposs_tmin = np.sum(tmin > TMAX_RECORD) > 0 or np.sum(tmin < TMIN_RECORD) > 0
        except KeyError:
            # Station does not have Tmin observations
            # Set to zero changepoints and no impossible values
            chgpts_tmin = np.array([])
            has_imposs_tmin = False
            
        try:
            tmax = ds_tmax.tmax.loc[:, a_id].values
            chgpts_tmax = np.array(idly.r.getVarChgPt(idly.robjects.FloatVector(tmax)))
            has_imposs_tmax = np.sum(tmax > TMAX_RECORD) > 0 or np.sum(tmax < TMIN_RECORD) > 0
        except KeyError:
            # Station does not have Tmax observations
            # Set to zero changepoints
            chgpts_tmax = np.array([])
            has_imposs_tmax = False
    
        # If any changepoints or impossible values in Tmin or Tmax, set station as "bad"
        if chgpts_tmin.size > 0 or chgpts_tmax.size > 0 or has_imposs_tmin or has_imposs_tmax:
            bad_ids.append(a_id)
    
    # output bad station ids to csv
    df = pd.DataFrame({'stn_id':bad_ids, 'reason':'infill issue'})
    df = df[['stn_id', 'reason']]
    df.to_csv(os.path.join(FPATH_STNDATA_INFILL, 'bad_stns.csv'), index=False)
    
