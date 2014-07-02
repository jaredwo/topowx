'''
Created on Jul 2, 2014

@author: jared.oyler
'''
import numpy as np
from netCDF4 import Dataset, num2date
import netCDF4
from twx.db import create_quick_db,STN_ID,BAD,LON,LAT
from twx.db.station_data import _build_stn_struct
from twx.utils import get_days_metadata, StatusCheck, grt_circle_dist

SERIAL_DB_VARIABLES = {'tmin':[('tmin', 'f4', netCDF4.default_fillvals['f4'], 'minimum air temperature', 'C'),
                               ('flag_infilled', 'i1', netCDF4.default_fillvals['i1'], 'infilled flag', '')],
                       'tmax':[('tmax', 'f4', netCDF4.default_fillvals['f4'], 'maximum air temperature', 'C'),
                               ('flag_infilled', 'i1', netCDF4.default_fillvals['i1'], 'infilled flag', '')]}

#5 years of data
USE_ALL_INFILL_THRESHOLD = np.round(365.25 * 5.0)

def update_daily_infill(stnid,tair_var,fpath_db,fnl_tair, mask_infill, infill_tair):
    '''
    Update the daily values for a station in an infilled netCDF station database
    
    Parameters
    ----------
    stnid : str
        The station id of the station to update
    tair_var : str
        The temperature variable ('tmin' or 'tmax') to update
    fpath_db : str
        The file path to the infilled netCDF station database
    fnl_tair : ndarray
        Time series of station observations with missing values infilled
    mask_infill : ndarray
        Boolean array specifying which values were infilled in fnl_tair
    infill_tair : ndarray
        Time series of station observations with all observations replaced with
        values from the infill model regardless of whether or not an
        observation was originally missing. 
    '''
    
    obs_mask = np.logical_not(mask_infill)
    difs = infill_tair[obs_mask] - fnl_tair[obs_mask]
    mae = np.mean(np.abs(difs))
    bias = np.mean(difs)

    ds_out = Dataset(fpath_db,'r+')
    stnids = ds_out.variables['stn_id'][:].astype(np.str)
    i = np.nonzero(stnids==stnid)[0][0]

    ds_out.variables[tair_var][:, i] = fnl_tair
    ds_out.variables["".join([tair_var, "_infilled"])][:, i] = infill_tair
    ds_out.variables['flag_infilled'][:, i] = mask_infill
    ds_out.variables['bias'][i] = bias
    ds_out.variables['mae'][i] = mae
    ds_out.close()
    
def create_serially_complete_db(fpath_infill_db,tair_var,fpath_out_serial_db):
    
    ds_infill = Dataset(fpath_infill_db)
    var_time = ds_infill.variables['time']
    stns = _build_stn_struct(ds_infill)
    start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
    days = get_days_metadata(start, end)
    
    create_quick_db(fpath_out_serial_db, stns, days, SERIAL_DB_VARIABLES[tair_var])
    ds_out = Dataset(fpath_out_serial_db,'r+')
    
    all_infill_flags = np.ones(days.size,dtype=np.bool)
    all_infill_stns = np.zeros(stns.size,dtype=np.bool)
    
    stat_chk = StatusCheck(stns.size,100)
    
    for x in np.arange(stns.size):
        
        infill_mask = ds_infill.variables['flag_infilled'][:,x].astype(np.bool)
        infill_runs = _runs_of_ones_array(infill_mask)
        
        if infill_runs.size > 0:
            max_infill = np.max(infill_runs)
        else:
            max_infill = 0
        
        if max_infill >= USE_ALL_INFILL_THRESHOLD:
            
            #This station has greater than USE_ALL_INFILL_THRESHOLD continuous
            #years of missing data. Use all infilled values for this station to avoid 
            #discontinuities between infilled and observed portions of time series
            tair_stn = ds_infill.variables["".join([tair_var,"_infilled"])][:,x]
            flag_stn = all_infill_flags
            
            all_infill_stns[x] = True
        
        else:
            
            tair_stn = ds_infill.variables[tair_var][:,x]
            flag_stn = infill_mask
            
        ds_out.variables[tair_var][:,x] = tair_stn
        ds_out.variables['flag_infilled'][:,x] = flag_stn
        ds_out.sync()
        
        stat_chk.increment()
    
    ds_out.close()
    
    print "% of stns with all infilled values: "+str((np.sum(all_infill_stns)/np.float(all_infill_stns.size))*100.)

def set_bad_stations(ds_serial,bad_ids):
    
    db_stnids = ds_serial.variables[STN_ID][:].astype(np.str)
    
    ds_serial.variables[BAD] = 0
    
    for aid in bad_ids:
        
        try:
            x = np.nonzero(db_stnids==aid)[0][0]
        except IndexError:
            pass
        
        ds_serial.variables[BAD][x] = 1
    
    ds_serial.sync()

def find_dup_stns(stnda):
    
    dup_stnids = []
    rm_stnids = []
    
    stat_chk = StatusCheck(stnda.stns.size,100)
    
    for stn in stnda.stns:
        
        if stn[STN_ID] not in dup_stnids:
        
            ngh_stns = stnda.stns[stnda.stn_ids != stn[STN_ID]]
            dists = grt_circle_dist(stn[LON],stn[LAT],ngh_stns[LON],ngh_stns[LAT])
            
            dup_nghs = ngh_stns[dists==0]
            
            if dup_nghs.size > 0:

                dup_stnids.extend(dup_nghs[STN_ID])
    
                stn_ids_load = np.sort(np.concatenate([np.array([stn[STN_ID]]).ravel(),np.array([dup_nghs[STN_ID]]).ravel()]))
                print stn_ids_load
                stn_idxs = np.nonzero(np.in1d(stnda.stn_ids, stn_ids_load, True))[0]
                imp_flgs = stnda.ds.variables['flag_infilled'][:,stn_idxs]
                imp_flg_sum = np.sum(imp_flgs, axis=0)
                
                stn_ids_rm = stn_ids_load[imp_flg_sum != np.min(imp_flg_sum)]
                
                rm_stnids.extend(stn_ids_rm)
                
        stat_chk.increment()
    
    rm_stnids = np.array(rm_stnids)
    
    return rm_stnids

def _runs_of_ones_array(bits):
    #http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts