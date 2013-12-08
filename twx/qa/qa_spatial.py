'''
Created on Sep 13, 2012

@author: jared.oyler
'''
import numpy as np
import qa_temp
import qa_prcp
from qa_temp import stns_in_radius_mask,MIN_NGHS
from db.station_data import STN_ID

def run_qa_spatial(stn, stn_da, tmin, tmax, prcp, days):
    
    flags_tmin = np.ones(tmin.size)
    flags_tmax = np.ones(tmax.size)
    flags_prcp = np.ones(prcp.size)
    
    tmin, tmax, flags_tmin, flags_tmax = qa_temp.qa_missing(tmin, tmax, days, flags_tmin, flags_tmax)
    prcp,flags_prcp = qa_prcp.qa_missing(prcp,days,flags_prcp)
    
    ngh_mask, dists = stns_in_radius_mask(stn, stn_da)
    ngh_ids = stn_da.stns[STN_ID][ngh_mask]
    ngh_ids = ngh_ids[np.logical_not(ngh_ids==stn[STN_ID])]
    dists = dists[np.logical_not(ngh_ids == stn[STN_ID])]
    
    if ngh_ids.size >= MIN_NGHS:
        
        ngh_obs = stn_da.load_all_stn_obs(ngh_ids)
        ngh_data = (ngh_ids,dists,ngh_obs)
        
        tmin, tmax, flags_tmin, flags_tmax = qa_temp.qa_spatial_regress(stn, stn_da, tmin, tmax, days, flags_tmin, flags_tmax,ngh_data)
        tmin, tmax, flags_tmin, flags_tmax = qa_temp.qa_spatial_corrob(stn, stn_da, tmin, tmax, days, flags_tmin, flags_tmax,ngh_data)
        
        prcp,flags_prcp = qa_prcp.qa_spatial_corrob(stn, stn_da, prcp, days, flags_prcp,ngh_data)
    
    tmin, tmax, flags_tmin, flags_tmax = qa_temp.qa_mega_inconsist(tmin, tmax, days, flags_tmin, flags_tmax)
    
    return  flags_tmin,flags_tmax,flags_prcp

if __name__ == '__main__':
    pass