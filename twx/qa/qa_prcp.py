'''
Quality Assurance procedures for prcp daily observations as described in:
Durre, I., M. J. Menne, B. E. Gleason, T. G. Houston, and R. S. Vose. 2010. 
Comprehensive Automated Quality Assurance of Daily Surface Observations. 
Journal of Applied Meteorology and Climatology 49:1615-1633.
'''

import utils.util_dates as utld
import numpy as np
from datetime import datetime
import utils.util_geo as utlg
from db.station_data import LON,LAT,STN_ID,PRCP,YEAR,DATE,MONTH,YDAY
from scipy import stats
import math
from qa.qa_temp import stns_in_radius_mask

QA_OK=1
QA_MISSING=2
QA_NAUGHT=3
DUP=25
QA_DUP_YEAR=4
QA_DUP_MONTH=5
QA_DUP_YEAR_MONTH=6
QA_DUP_WITHIN_MONTH=7
QA_IMPOSS_VALUE=8
QA_STREAK=9
QA_GAP=10
QA_INTERNAL_INCONSIST=11
QA_LAGRANGE_INCONSIST=12
QA_SPIKE_DIP=13
QA_MEGA_INCONSIST=14
QA_CLIM_OUTLIER=15
QA_SPATIAL_REGRESS=16
QA_SPATIAL_CORROB=17
QA_MEGA_INCONSIST=18
QA_FREQUENT=19
QA_YR_CLIM_OUTLIER=20

PRCP_MIN_VALUE = 0.0
# World Record Daily Prcp (cm)
PRCP_MAX_VALUE = 182.88

#Constants for spatial collaboration checks
MAX_SPATIAL_COLLAB_THRES = 26.924
MAX_SPATIAL_COLLAB_THRES_MM = MAX_SPATIAL_COLLAB_THRES*10.0
NGH_RADIUS = 75.0
ANOMALY_CUTOFF = 10.0
MIN_DAYS_MTH_WINDOW = 40
MIN_NGHS = 3
MAX_NGHS = 7

STREAK_LEN = 20
GAP_THRES = 30.0 #cm

MONTHS = np.arange(1,13)
DATES_366 = utld.get_date_array(datetime(2004,1,1),datetime(2004,12,31))
DATES_365 = utld.get_date_array(datetime(2003,1,1),datetime(2003,12,31))
MIN_PERCENTILE_VALUES = 20

#constants for qa_yr_clim_outlier
MIN_YR_DAYS_PRCP_OBS = 245
MIN_NUM_YRS_PRCP = 7
NGH_RESID_CUTOFF = 1000
NGH_RESID_STD_CUTOFF = 0.0


def run_qa(stn,stn_da,prcp,tavg,days):
    '''
    Runs all QA checks in specific order
    '''
    
    flags_prcp = np.ones(prcp.size)
    
    prcp,flags_prcp = qa_missing(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_dup_data(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_imposs_value(prcp,days,flags_prcp)
    #prcp,flags_prcp = qa_streak(prcp,days,flags_prcp)
    #prcp,flags_prcp = qa_frequent(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_gap(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_clim_outlier(prcp,tavg,days,flags_prcp)
    prcp,flags_prcp = qa_spatial_corrob(stn, stn_da, prcp, days, flags_prcp)
    
    return flags_prcp

def run_qa_non_spatial(prcp,tavg,days):
    '''
    Runs all QA checks in specific order.
    Only includes checks that do not require neighboring station data
    '''
    
    flags_prcp = np.ones(prcp.size)
    
    prcp,flags_prcp = qa_missing(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_dup_data(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_imposs_value(prcp,days,flags_prcp)
    #prcp,flags_prcp = qa_streak(prcp,days,flags_prcp)
    #prcp,flags_prcp = qa_frequent(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_gap(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_clim_outlier(prcp,tavg,days,flags_prcp)
    
    return flags_prcp
    
def run_qa_spatial_only(stn,stn_da,prcp,tavg,days):
    '''
    Runs all spatial QA checks.
    Only includes checks that do not require neighboring station data
    '''
    
    flags_prcp = np.ones(prcp.size)
    prcp,flags_prcp = qa_missing(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_spatial_corrob(stn, stn_da, prcp, days, flags_prcp)
    #prcp,flags_prcp = qa_yr_clim_outlier(stn,stn_da,prcp,days,flags_prcp)
    
    return flags_prcp
    

def qa_missing(prcp,days,flags_prcp):
    '''
    Flags all observations that are missing in the specified period of record 
    '''

    mask_miss=np.isnan(prcp)
    
    return update_obs_flags(prcp,flags_prcp,mask_miss,QA_MISSING)

def qa_dup_data(prcp,days,flags_prcp):
    '''
    Runs all duplicate data QA checks
    '''
    
    prcp,flags_prcp = qa_dup_year(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_dup_year_month(prcp,days,flags_prcp)
    prcp,flags_prcp = qa_dup_month(prcp,days,flags_prcp)
    
    return prcp,flags_prcp

def qa_dup_year_month(prcp,days,flags_prcp):
    '''
    Check for duplicate values between different months in the same year
    '''
    
    yrs = np.unique(days[YEAR])
    yr_nums = np.arange(yrs.size)
    mask_prcp = np.zeros(days[DATE].size,dtype=np.bool_)
    
    mths = np.unique(days[MONTH])
    mth_nums = np.arange(mths.size)
    
    for x in yr_nums:
        
        yr_mask = days[YEAR]==yrs[x]
        
        prcp_yr = prcp[yr_mask]
        
        mths_yr = days[MONTH][yr_mask]
        
        for i in mth_nums:
            
            mth1=mths[i]
            mth1_mask = mths_yr==mth1
            
            prcp_yr_mth1 = prcp_yr[mth1_mask]
            
            
            if not (mth1==np.max(mth_nums)) and prcp_yr_mth1[np.logical_and(np.isfinite(prcp_yr_mth1),prcp_yr_mth1>0)].size >= 3:
                
                sub_mths = np.arange(i+1,np.max(mth_nums)+1)
                
                for z in sub_mths:
                    
                    mth2=mths[z]
                    mth2_mask = mths_yr==mth2
                    
                    prcp_yr_mth2 = prcp_yr[mth2_mask]
                    
                    if prcp_yr_mth2[np.logical_and(np.isfinite(prcp_yr_mth2),prcp_yr_mth2>0)].size >= 3 and is_dup_series(prcp_yr_mth1, prcp_yr_mth2):
                        yr_month_mask = np.logical_and(yr_mask,np.logical_or(days[MONTH]==mth1,days[MONTH]==mth2))
                        mask_prcp = np.logical_or(mask_prcp,yr_month_mask)
    
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_DUP_YEAR_MONTH)

def qa_dup_year(prcp,days,flags_prcp):
    '''
    Check for duplicate values between years
    '''
    
    yrs = np.unique(days[YEAR])
    yr_nums = np.arange(yrs.size)
    mask_prcp = np.zeros(days[DATE].size,dtype=np.bool_) 
    
    for x in yr_nums:
        
        yr1_mask = days[YEAR]==yrs[x]
        
        prcp_yr1=prcp[yr1_mask]
        
        #need at least 3 nonzero values in the year
        if prcp_yr1[np.logical_and(np.isfinite(prcp_yr1),prcp_yr1>0)].size < 3:
            continue
    
        if not (x==np.max(yr_nums)):
            
            sub_yr_nums = np.arange(x+1,np.max(yr_nums)+1)
            
            for i in sub_yr_nums:
                
                yr2_mask = days[YEAR]==yrs[i]
                
                prcp_yr2=prcp[yr2_mask]

                #need at least 3 nonzero values in the year
                if prcp_yr2[np.logical_and(np.isfinite(prcp_yr2),prcp_yr2>0)].size < 3:
                    continue
                
                if is_dup_series(prcp_yr1, prcp_yr2):
                    yr_mask = np.logical_or(yr1_mask,yr2_mask)
                    mask_prcp = np.logical_or(mask_prcp,yr_mask)
    
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_DUP_YEAR)


def qa_dup_month(prcp,days,flags_prcp):
    '''
    Check for duplicate values for same calendar month in different years
    '''
    
    yrs = np.unique(days[YEAR])
    yr_nums = np.arange(yrs.size)
    mask_prcp = np.zeros(days[DATE].size,dtype=np.bool_)
    
    mths = np.unique(days[MONTH])
    
    for x in yr_nums:
        
        yr1_mask = days[YEAR]==yrs[x]
        
        prcp_yr1=prcp[yr1_mask]
        
        mths_yr1=days[MONTH][yr1_mask]
        
        if not (x==np.max(yr_nums)):
            
            sub_yr_nums = np.arange(x+1,np.max(yr_nums)+1)
            
            for i in sub_yr_nums:
                
                yr2_mask = days[YEAR]==yrs[i]
                
                prcp_yr2=prcp[yr2_mask]
                
                mths_yr2=days[MONTH][yr2_mask]
                
                for month in mths:
                    
                    mask_mth_yr1 = mths_yr1==month
                    mask_mth_yr2 = mths_yr2==month
                    
                    prcp_yr1_mth = prcp_yr1[mask_mth_yr1]
                    prcp_yr2_mth = prcp_yr2[mask_mth_yr2]
                    
                    if prcp_yr1_mth[np.logical_and(np.isfinite(prcp_yr1_mth),prcp_yr1_mth>0)].size >= 3 and prcp_yr2_mth[np.logical_and(np.isfinite(prcp_yr2_mth),prcp_yr2_mth>0)].size >= 3 and is_dup_series(prcp_yr1_mth, prcp_yr2_mth):
                        yr_month_mask = np.logical_and(np.logical_or(yr1_mask,yr2_mask),days[MONTH]==month)
                        mask_prcp = np.logical_or(mask_prcp,yr_month_mask)
    
    
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_DUP_MONTH)


def qa_imposs_value(prcp,days,flags_prcp):
    '''
    Check for values that are outside the bounds of world records or invalid (i.e.--<0)
    '''
    
    mask_prcp = np.logical_or(prcp<PRCP_MIN_VALUE,prcp>PRCP_MAX_VALUE)
    
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_IMPOSS_VALUE)

def qa_streak(prcp,days,flags_prcp):
    '''
    Identify 20 or more consecutive prcp values
    '''    
    
    date_objs = days[DATE]
    date_nums = np.arange(date_objs.size)
    mask_prcp = np.zeros(date_objs.size,dtype=np.bool_)
    
    streak_indices = []
    
    for x in date_nums:
        
        if prcp[x] == 0 or np.isnan(prcp[x]):
            continue
        elif x == 0 or len(streak_indices) == 0:
            streak_indices.append(x)
        elif prcp[x] == prcp[streak_indices[-1]]:
            streak_indices.append(x)
        elif len(streak_indices) >= 20:
            mask_prcp[streak_indices] = True
            streak_indices = []
            streak_indices.append(x)
        else:
            streak_indices = []
            streak_indices.append(x)
            
    
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_STREAK)

def qa_frequent(prcp,days,flags_prcp):
    '''
    Checks for frequent occurrences of the same value that might not necessarily
    be consecutive
    '''

    percentiles_leap = build_percentiles(prcp, days,DATES_366)

    date_objs = days[DATE]
    mask_prcp = np.zeros(date_objs.size,dtype=np.bool_)
    
    indices_consec = np.nonzero(np.logical_and(not_nan(prcp),prcp>0))[0]
    prcp_consec = prcp[indices_consec]
    
    for x in np.arange(prcp_consec.size):
        
        
        
        vals_rng = prcp_consec[x:x+10]
        indices_rng = indices_consec[x:x+10]
        yr_days_rng = days[YDAY][indices_rng] - 1
        pctiles_rng = percentiles_leap[yr_days_rng,:]
        
        uniq_vals = np.unique(vals_rng)
        for val in uniq_vals:
            
            mask_val = vals_rng==val
            if vals_rng[mask_val].size >= 9 and pctiles_rng[mask_val,0][np.isfinite(pctiles_rng[mask_val,0])].size == pctiles_rng[mask_val,0].size and pctiles_rng[mask_val,0][val >= pctiles_rng[mask_val,0]].size == pctiles_rng[mask_val,0].size:
                mask_prcp[indices_rng[mask_val]] = True
            elif vals_rng[mask_val].size >= 8 and pctiles_rng[mask_val,1][np.isfinite(pctiles_rng[mask_val,1])].size == pctiles_rng[mask_val,1].size and pctiles_rng[mask_val,1][val >= pctiles_rng[mask_val,1]].size == pctiles_rng[mask_val,1].size:
                mask_prcp[indices_rng[mask_val]] = True
            elif vals_rng[mask_val].size >= 7 and pctiles_rng[mask_val,2][np.isfinite(pctiles_rng[mask_val,2])].size == pctiles_rng[mask_val,2].size and pctiles_rng[mask_val,2][val >= pctiles_rng[mask_val,2]].size == pctiles_rng[mask_val,2].size:
                mask_prcp[indices_rng[mask_val]] = True
            elif vals_rng[mask_val].size >= 5 and pctiles_rng[mask_val,3][np.isfinite(pctiles_rng[mask_val,3])].size == pctiles_rng[mask_val,3].size and pctiles_rng[mask_val,3][val >= pctiles_rng[mask_val,3]].size == pctiles_rng[mask_val,3].size:
                mask_prcp[indices_rng[mask_val]] = True        
    
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_FREQUENT)

def qa_gap(prcp,days,flags_prcp):
    '''
    Examines frequency distributions of prcp for calendar months and flags values in
    a distribution's tail that are unrealistically separated from the rest of the values
    '''

    mask_prcp = np.zeros(days[DATE].size,dtype=np.bool_)

    mths = days[MONTH]
    uniq_mths = np.unique(mths)
    
    for mth in uniq_mths:
        
        mth_mask = np.logical_and(np.logical_and(mths==mth,not_nan(prcp)),prcp>0)
        
        prcp_mth = prcp[mth_mask]
        if prcp_mth.size == 0:
            continue
        prcp_mth = np.sort(prcp_mth)
        
        gap_mask = np.ediff1d(prcp_mth,to_begin=[0]) >= GAP_THRES
    
        if prcp_mth[gap_mask].size > 0:
            gap_val = prcp_mth[gap_mask][0]
            mask_prcp[np.logical_and(mth_mask,prcp>=gap_val)] = True
    
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_GAP)   

def qa_yr_clim_outlier(stn,stn_da,prcp,days,flags_prcp):
    
    mask_prcp = np.zeros(days[DATE].size,dtype=np.bool_)
    
    yrs = np.unique(days[YEAR])
    prcp_yrs = []
    chk_yrs = []
    for yr in yrs:
        
        mask_yr = np.logical_and(days[YEAR]==yr,np.isfinite(prcp))
    
        if prcp[mask_yr].size >= MIN_YR_DAYS_PRCP_OBS:
            prcp_yrs.append(np.sum(prcp[mask_yr]))
            chk_yrs.append(yr)
    
    prcp_yrs = np.array(prcp_yrs)
    chk_yrs = np.array(chk_yrs)
    if prcp_yrs.size >= MIN_NUM_YRS_PRCP:
        
        ngh_mask,dists = stns_in_radius_mask(stn,stn_da)
        ngh_ids = stn_da.stns[STN_ID][ngh_mask]
        ngh_ids = ngh_ids[np.logical_not(ngh_ids==stn[STN_ID])]
        dists = dists[np.logical_not(ngh_ids==stn[STN_ID])]
        ngh_obs = stn_da.load_all_stn_obs(ngh_ids)
        s_indices = np.argsort(dists)
        
        dists = dists[s_indices]
        ngh_ids = ngh_ids[s_indices]
        ngh_obs_prcp = ngh_obs[PRCP][:,s_indices]

        obs_ngh_ann_prcp = []
        for x in np.arange(ngh_ids.size):
            
            ngh_prcp = ngh_obs_prcp[:,x]
            ngh_ann_prcp = np.ones(len(prcp_yrs))*np.nan
            
            for i in np.arange(len(chk_yrs)):
                
                mask_stn_yr = np.logical_and(days[YEAR]==chk_yrs[i],np.isfinite(ngh_prcp))
                
                if ngh_prcp[mask_stn_yr].size >= MIN_YR_DAYS_PRCP_OBS:
                    ngh_ann_prcp[i] = np.sum(ngh_prcp[mask_stn_yr])
            
            if ngh_ann_prcp[np.isfinite(ngh_ann_prcp)].size >= MIN_NUM_YRS_PRCP*(2.0/3.0):
                obs_ngh_ann_prcp.append(ngh_ann_prcp)
        
        if len(obs_ngh_ann_prcp) >= MIN_NGHS:
            
            ioas = []
            lin_mods = []
            for ngh_ann_prcp in obs_ngh_ann_prcp:
                
                mask_overlap = np.isfinite(ngh_ann_prcp)
                ioas.append(calc_ioa(prcp_yrs[mask_overlap],ngh_ann_prcp[mask_overlap]))
                lin_mods.append(build_lin_model(ngh_ann_prcp[mask_overlap],prcp_yrs[mask_overlap]))
        
            ann_prcp_predicts = np.ones(prcp_yrs.size)*np.nan
            
            ioas = np.array(ioas)
            obs_ngh_ann_prcp = object_array(obs_ngh_ann_prcp)
            lin_mods = object_array(lin_mods)
            
            s_indices = np.argsort(ioas)
            s_indices = s_indices[::-1]
            obs_ngh_ann_prcp = obs_ngh_ann_prcp[s_indices]
            ioas = ioas[s_indices]
            lin_mods = lin_mods[s_indices]
            
            for x in np.arange(prcp_yrs.size):
                
                wght_ests = []
                wghts = []
                n = 0
                for i in np.arange(len(obs_ngh_ann_prcp)):
                    
                    ngh_ann_prcp = obs_ngh_ann_prcp[i]
                    
                    if np.isfinite(ngh_ann_prcp[x]):
                        
                        lin_mod = lin_mods[i]
                        a = lin_mod[0] #slope
                        b = lin_mod[1] #intercept
                        d = ioas[i] #weight

                        wght_ests.append((b+(a*ngh_ann_prcp[x]))*d)
                        wghts.append(d)
                        n+=1
                    if n == MAX_NGHS:
                        break
                
                if n >= MIN_NGHS:
                    
                    wght_ests = np.array(wght_ests)
                    wghts = np.array(wghts)
                    ann_prcp_predicts[x] = np.sum(wght_ests)/np.sum(wghts)
            
            prcp_yrs[prcp_yrs==0] = 1.0
            resid = np.abs(prcp_yrs-ann_prcp_predicts)/prcp_yrs*100
            mask_finite_resid = np.isfinite(resid)
            resid_std = np.abs((resid - np.mean(resid[mask_finite_resid]))/np.std(resid[mask_finite_resid]))
            yrs_flagged = chk_yrs[np.logical_and(resid>=NGH_RESID_CUTOFF,resid_std>=NGH_RESID_STD_CUTOFF)]
            if yrs_flagged.size > 0:
                mask_prcp[np.in1d(days[YEAR], yrs_flagged)] = True
                
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_YR_CLIM_OUTLIER)       
            

def qa_clim_outlier(prcp,tavg,days,flags_prcp):
    
    '''
    Checks for prcp outliers based on relation to 29-day climate norm 95th percentile
    Must have 20 nonzero values in the 29-day period of record for the check to run
    '''
    
    pctiles = build_percentiles(prcp, days, DATES_366)
    
    date_objs = days[DATE]
    ydays = days[YDAY]
    mask_prcp = np.zeros(date_objs.size,dtype=np.bool_)
    
    indices = np.nonzero(np.logical_and(not_nan(prcp),prcp>0))[0]
    prcp_nonzero = prcp[indices]
    
    for x in np.arange(prcp_nonzero.size):
        
        day_num = ydays[indices[x]] - 1
        
        #check that a val and norm exists
        if not_nan(pctiles[day_num,4]):
            
            if not_nan(tavg[indices[x]]) and tavg[indices[x]] < 0.0:
                m = 5.0
            else:
                m = 9.0
            
            mask_prcp[indices[x]] =  prcp_nonzero[x] >= (m*pctiles[day_num,4])
    
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_CLIM_OUTLIER)

def qa_spatial_corrob(stn, stn_da, prcp, days, flags_prcp,ngh_data=None):
    '''
    Check for prcp observations that are not corroborated by any neighboring observations
    '''
    
    date_objs = days[DATE]
    mask_prcp = np.zeros(date_objs.size,dtype=np.bool_)
    
    if ngh_data is None:
    
        ngh_mask,dists = stns_in_radius_mask(stn,stn_da)
        ngh_ids = stn_da.stns[STN_ID][ngh_mask]
        ngh_ids = ngh_ids[np.logical_not(ngh_ids==stn[STN_ID])]
        dists = dists[np.logical_not(ngh_ids==stn[STN_ID])]
        ngh_obs = None
    else:
        ngh_ids,dists,ngh_obs = ngh_data
        
    if ngh_ids.size >= MIN_NGHS:
    
        if ngh_obs is None:
            ngh_obs = stn_da.load_all_stn_obs(ngh_ids)
        
        s_indices = np.argsort(dists)
        
        dists = dists[s_indices]
        ngh_ids = ngh_ids[s_indices]
        ngh_obs_prcp = ngh_obs[PRCP][:,s_indices]
        
        pctiles_masks = get_pctiles_md_masks(days, DATES_366)
        
        ngh_pors = []
        for x in np.arange(ngh_obs_prcp.shape[1]):
            ngh_pors.append(build_pors(ngh_obs_prcp[:,x], pctiles_masks))
        ngh_pors = object_array(ngh_pors)
        stn_pors = build_pors(prcp, pctiles_masks)
        
        for x in  np.arange(date_objs.size):
            
            #Don't perform check on first/last day of time series
            if x == 0 or x == date_objs.size - 1 or np.isnan(prcp[x]):
                continue
            
            mask_prcp[x]=get_spatial_corrob_flag(prcp,ngh_obs_prcp,stn_pors,ngh_pors,x,days)
    
#    else:
#        #Station had no nghs in range
#        print "".join([stn[STN_ID],": could not perform prcp qa_spatial_corrob--no nghs in range."])
    
    
    return update_obs_flags(prcp,flags_prcp,mask_prcp,QA_SPATIAL_CORROB)

def calc_ioa(x,y):
    y_mean = np.mean(y)
    ioa = 1.0 - (np.sum(np.abs(y-x))/np.sum(np.abs(x-y_mean)+np.abs(y-y_mean)))
    return ioa

def build_lin_model(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return slope, intercept, r_value, p_value, std_err 

def build_pors(prcp,pctiles_masks):
    
    pors = []
    for x in np.arange(pctiles_masks.shape[0]):
        
        pors.append(prcp[np.logical_and(np.logical_and(not_nan(prcp),prcp>0.0),pctiles_masks[x,:])])
    return pors

def get_spatial_corrob_flag(prcp,ngh_obs_prcp,stn_pors,ngh_pors,obs_num,days):
    
    day_num_prev = days[YDAY][obs_num-1] - 1
    day_num_cur = days[YDAY][obs_num] - 1 
    day_num_nxt = days[YDAY][obs_num+1] - 1
    
    mask_finite_prev = np.isfinite(ngh_obs_prcp[obs_num-1,:])
    mask_finite_cur = np.isfinite(ngh_obs_prcp[obs_num,:])
    mask_finite_nxt = np.isfinite(ngh_obs_prcp[obs_num+1,:])
    
    if np.nonzero(mask_finite_prev)[0].size >= MIN_NGHS and np.nonzero(mask_finite_cur)[0].size >= MIN_NGHS and np.nonzero(mask_finite_nxt)[0].size >= MIN_NGHS:
                        
        ngh_obs_prev = ngh_obs_prcp[obs_num-1,mask_finite_prev]
        ngh_obs_cur = ngh_obs_prcp[obs_num,mask_finite_cur]
        ngh_obs_nxt = ngh_obs_prcp[obs_num+1,mask_finite_nxt]
        ngh_obs_all = np.concatenate([ngh_obs_prev[0:MAX_NGHS],ngh_obs_cur[0:MAX_NGHS],ngh_obs_nxt[0:MAX_NGHS]])
        
        if prcp[obs_num] > np.max(ngh_obs_all):
            abs_dif = abs(prcp[obs_num]-np.max(ngh_obs_all))
        elif prcp[obs_num] < np.min(ngh_obs_all):
            abs_dif = abs(prcp[obs_num]-np.min(ngh_obs_all))
        else: 
            abs_dif = 0.0
            return False
        
        if stn_pors[day_num_cur].size > MIN_PERCENTILE_VALUES:
            
            prcp_pctile = stats.percentileofscore(stn_pors[day_num_cur],prcp[obs_num])
            
            ngh_pors_prev = ngh_pors[mask_finite_prev]
            ngh_pors_cur = ngh_pors[mask_finite_cur]
            ngh_pors_nxt = ngh_pors[mask_finite_nxt]
            
            ngh_pctiles_prev = []
            for ngh in np.arange(ngh_obs_prev.size):
                
                ngh_prcp = ngh_obs_prev[ngh]
                
                if ngh_pors_prev[ngh][day_num_prev].size > MIN_PERCENTILE_VALUES:
                    
                    ngh_pctiles_prev.append(stats.percentileofscore(ngh_pors_prev[ngh][day_num_prev],ngh_prcp))
                if len(ngh_pctiles_prev) == MAX_NGHS:
                    break
                
            ngh_pctiles_prev = np.array(ngh_pctiles_prev)
            ##############################
            ngh_pctiles_cur = []
            for ngh in np.arange(ngh_obs_cur.size):
                
                ngh_prcp = ngh_obs_cur[ngh]
                
                
                if ngh_pors_cur[ngh][day_num_cur].size > MIN_PERCENTILE_VALUES:
                    
                    ngh_pctiles_cur.append(stats.percentileofscore(ngh_pors_cur[ngh][day_num_cur],ngh_prcp))  
                
                if len(ngh_pctiles_cur) == MAX_NGHS:
                    break
                
            ngh_pctiles_cur = np.array(ngh_pctiles_cur)
            ##############################
            ngh_pctiles_nxt = []
            for ngh in np.arange(ngh_obs_nxt.size):
                
                ngh_prcp = ngh_obs_nxt[ngh]
                
                if ngh_pors_nxt[ngh][day_num_nxt].size > MIN_PERCENTILE_VALUES:
                    
                    ngh_pctiles_nxt.append(stats.percentileofscore(ngh_pors_nxt[ngh][day_num_nxt],ngh_prcp))
                    
                if len(ngh_pctiles_nxt) == MAX_NGHS:
                    break
                     
            ngh_pctiles_nxt = np.array(ngh_pctiles_nxt)
            
            if ngh_pctiles_prev.size >= MIN_NGHS and ngh_pctiles_cur.size >= MIN_NGHS and ngh_pctiles_nxt.size >= MIN_NGHS:
                
                ngh_pctiles_all = np.concatenate([ngh_obs_prev,ngh_obs_cur,ngh_obs_nxt])
                
                if prcp_pctile > np.max(ngh_pctiles_all):
                    pctile_dif = abs(prcp_pctile-np.max(ngh_pctiles_all))
                elif prcp_pctile < np.min(ngh_pctiles_all):
                    pctile_dif = abs(prcp_pctile-np.min(ngh_pctiles_all))
                else: 
                    pctile_dif = 0.0
                
                if pctile_dif == 0:
                    thres = MAX_SPATIAL_COLLAB_THRES
                else:
                    thres = ((-45.72*math.log(pctile_dif)) + MAX_SPATIAL_COLLAB_THRES_MM)/10.0
            else:
                thres = MAX_SPATIAL_COLLAB_THRES
        else:
            thres = MAX_SPATIAL_COLLAB_THRES
        
#        if abs_dif > thres:
#            print "|".join([str(abs_dif),str(pctile_dif),str(thres)])
        return abs_dif > thres
    else:
        return False

def stns_in_radius_mask(stn,stn_da,radius=NGH_RADIUS):
    dists = utlg.grt_circle_dist(stn[LON],stn[LAT],stn_da.stns[LON],stn_da.stns[LAT])
    #mask = np.logical_and(dists <= radius,np.char.startswith(stn_da.stns[STN_ID],"GHCN"))
    mask = dists <= radius
    return mask,dists[mask]

def build_percentiles(vals,days,yr_days):
    
    mth_days = utld.get_md_array(days[DATE])
    
    #Build 29-day percentiles for every day in yr_days
    day_nums = np.arange(yr_days.size)
    #Columns: 30th,50th,70th,90th,95th
    percentiles = np.ones([yr_days.size,5])*np.NAN
    
    for x in day_nums:
        date = yr_days[x]
        srt_date = date - utld.TWO_WEEKS
        end_date = date + utld.TWO_WEEKS
        date_range = utld.get_date_array(srt_date, end_date)
        mth_days_range = utld.get_md_array(date_range)
        
        date_mask = np.in1d(mth_days,mth_days_range)
        vals_rng = vals[np.logical_and(np.logical_and(not_nan(vals),vals>0),date_mask)]
        
        if vals_rng.size >= MIN_PERCENTILE_VALUES:
            percentiles[x,:] = [np.percentile(vals_rng,30),np.percentile(vals_rng,50),np.percentile(vals_rng,70),np.percentile(vals_rng,90),np.percentile(vals_rng,95)]
    
    return percentiles

def get_pctiles_md_masks(days,yr_days):
    
    mth_days = utld.get_md_array(days[DATE])
    
    day_nums = np.arange(yr_days.size)
    
    pctile_masks = np.zeros((yr_days.size,mth_days.size),dtype=np.bool)
    
    for x in day_nums:
        
        mth_day = yr_days[x]
        srt_mth_day = mth_day - utld.TWO_WEEKS
        end_mth_day = mth_day + utld.TWO_WEEKS
        date_range = utld.get_date_array(srt_mth_day, end_mth_day)
        mth_days_range = utld.get_md_array(date_range)
        
        pctile_masks[x,:] = np.in1d(mth_days,mth_days_range)
    
    return pctile_masks

def is_dup_series(vals_yr1,vals_yr2):
    
    if vals_yr1[not_nan(vals_yr1)].size > 0 and vals_yr2[not_nan(vals_yr2)].size > 0:
        
        #Compare up to last day of the shortest series
        last_day = np.min([vals_yr1.size,vals_yr2.size])  
        dup=np.nonzero(vals_yr1[0:last_day] == vals_yr2[0:last_day])[0].size == vals_yr2[0:last_day].size
    else:
        dup=False
    
    return dup

def not_nan(vals):
    return np.logical_not(np.isnan(vals))

def apply_2d_mask(a,row_mask,col_mask):
    b = a[row_mask,:]
    b = b[:,col_mask]
    return b

def object_array(x):
    array = np.empty(len(x), dtype=np.object)
    for i in range(len(x)):
        array[i] = x[i]
    return array

def update_obs_flags(prcp,flags_prcp,mask_prcp,flag):
    
    prcp[mask_prcp] = np.NAN
    flags_prcp[np.logical_and(flags_prcp==QA_OK,mask_prcp)] = flag

    return prcp,flags_prcp

def mad(a):
    '''
    Calculates median absolute deviations 
    '''
    return np.median(np.abs(a-np.median(a)))

def biweight_mean_std(X):
    '''
    Calculates more robust mean/std for climate data
    Used by Durre et al. 2010 referencing Lanzante 1996
    '''
    
    #mean
    c = 7.5
    M = np.median(X)
    MAD = mad(X)
    
    if MAD == 0:
        #return normal mean, std. biweight cannot be calculated
        return np.mean(X),np.std(X,ddof=1)
    
    u = (X - M)/(c*MAD)
    u[np.abs(u)>=1.0]=1.0
    Xbi = M + (np.sum((X-M)*(1.0-u**2)**2)/np.sum((1-u**2)**2))
    
    #std
    n = X.size
    Sbi = ((n*np.sum(((X-M)**2)*(1-u**2)**4))**0.5)/np.abs(np.sum((1-u**2)*(1-(5*u**2))))
    
    #return np.mean(X),np.std(X)
    
    return Xbi,Sbi 

if __name__ == '__main__':
    pass



