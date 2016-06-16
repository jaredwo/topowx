"""
Script to find stations with suspect infilled time series.
"""

from rpy2.rinterface import RRuntimeError
from twx.db import STN_ID
from twx.infill import get_bad_infill_stnids
from twx.infill import infill_daily as idly
from twx.utils import TwxConfig
import numpy as np
import os
import pandas as pd
import xarray as xr

if __name__ == '__main__':
    
    # Load R environment used for daily infilling
    idly._load_R()
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
        
    stnids = get_bad_infill_stnids(twx_cfg.fpath_log_daily_infill)
            
    ds_tmin = xr.open_dataset(twx_cfg.fpath_stndata_nc_infill_tmin)
    ds_tmax = xr.open_dataset(twx_cfg.fpath_stndata_nc_infill_tmax)
    
    # For each station marked as suspect in the infill log, check to see
    # if there are any major variance changepoints across the station's
    # full time series or if the station has impossible values outside
    # world records. If so, mark the station as "bad".
    
    # World records for daily Tmax and Tmin in degrees C
    record_tmax = 57.7
    record_tmin = -89.4
    
    bad_ids = []
    
    def has_bad_infill(ds, elem, stnid):
        
        is_bad = False
        
        try:
        
            tair = ds[elem].loc[:, stnid].values
            chgpts = np.array(idly.r.getVarChgPt(idly.robjects.FloatVector(tair)))
            has_imposs = (tair > record_tmax).any() or (tair < record_tmin).any()
            
            if chgpts.size > 0 or has_imposs:
                is_bad = True
        
        except KeyError:
            # Station does not have Tmin observations
            pass
        
        except RRuntimeError as e:
        
            if e.args[0].find("Missing value: NA is not allowed") != -1:
                
                # Infill completely failed on this station and its values are
                # all NA
                is_bad = True
            else:
                raise
        
        return is_bad
    
    for a_id in stnids:
        
        is_bad_tmin = has_bad_infill(ds_tmin, 'tmin', a_id)
        is_bad_tmax = has_bad_infill(ds_tmax, 'tmax', a_id)   
    
        # If any changepoints or impossible values in Tmin or Tmax, set station as "bad"
        if is_bad_tmin or is_bad_tmax:
            bad_ids.append(a_id)
    
    # output bad station ids to csv
    df = pd.DataFrame({STN_ID:bad_ids, 'reason':'infill issue'})
    df = df[[STN_ID, 'reason']]
    df.to_csv(twx_cfg.fpath_flagged_bad_stns, index=False)
    
