'''
Classes and functions for performing regression-based and weighted average based infilling of missing station observations similar to
the methods in Durre et al. (2010). Comprehensive Automated Quality Assurance of Daily Surface Observations. Appendix B.
This is mainly used to get a better mean value estimate (i.e.--"normal") over a period-of-record that has missing values. 

@author: jared.oyler
'''
import numpy as np
import utils.util_geo as utlg
from db.station_data import station_data_ncdb, STN_ID, LON, LAT,UTC_OFFSET
from utils.util_dates import MONTH, MTH_SRT_END_DATES, DAY
from datetime import timedelta
from scipy import stats
import interp.clibs as clibs

MAX_DISTANCE = 75 #in km
MAX_NGHS_LOAD = 2000
MIN_POR_OVERLAP = 2.0 / 3.0
MIN_DAILY_NGHBRS = 3
MAX_DAILY_NGHBRS = 7
NNGH_NNR = 4
MTH_BUFFER = 15
MAX_EXTEND = 2000 #km
MAX_COLS_NORM_IMPUTE = 31
MIN_NNR_COR = 0.5

#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri
r = robjects.r

class mth_ngh_matrix(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station for a particular month.
    '''

    def __init__(self, mthbuf_mask, mth_mask, nghs):
        '''
        
        @param mthbuf_mask: a mask of months (from build_mth_masks) over the time series of interest with a user-defined buffer at the end/beginning of each month
        @param mth_mask: a mask of months (from build_mth_masks) over the time series of interest
        @param nghs: a ngh_matrix object
        '''
        
        valid_mask = nghs.valid_mask_matrix
        nghs_matrix = nghs.angh_matrix
        nghs_dist = nghs.ngh_dists
        
        mth_mask = np.logical_and(nghs.day_mask,mth_mask)
        
        #########################################################################
        
        tair_target = nghs_matrix[:, 0]
        tair_target = tair_target[mthbuf_mask]
        valid_mask_target = valid_mask[:, 0]
        valid_mask_target = valid_mask_target[mthbuf_mask]
        
        tair_nghs = nghs_matrix[:, 1:]
        tair_nghs = tair_nghs[mthbuf_mask, :]
        valid_mask_nghs = valid_mask[:, 1:]
        valid_mask_nghs = valid_mask_nghs[mthbuf_mask, :]
        
        #########################################################################
        
        mth_tair_target = nghs_matrix[:, 0]
        mth_tair_target = mth_tair_target[mth_mask]
        mth_valid_mask_target = valid_mask[:, 0]
        mth_valid_mask_target = mth_valid_mask_target[mth_mask]
        
        mth_tair_nghs = nghs_matrix[:, 1:]
        mth_tair_nghs = mth_tair_nghs[mth_mask, :]
        mth_valid_mask_nghs = valid_mask[:, 1:]
        mth_valid_mask_nghs = mth_valid_mask_nghs[mth_mask, :]
        
        #########################################################################
        
        nnghs = tair_nghs.shape[1]
        ntarget_valid = np.nonzero(valid_mask_target)[0].size
        nthres_overlap = np.round(MIN_POR_OVERLAP * ntarget_valid)
        
        #Number of observations threshold for entire period that is being infilled
        nthres_all = np.round(MIN_POR_OVERLAP * tair_target.size) 
        
        overlap_mask_tair = np.zeros(nnghs, dtype=np.bool)
        ioa = np.zeros(nnghs)
        
        for x in np.arange(nnghs):
            
            valid_mask_ngh = valid_mask_nghs[:, x]
            
            overlap_mask = np.logical_and(valid_mask_target, valid_mask_ngh)
                        
            nlap = np.nonzero(valid_mask_ngh)[0].size
            
            nlap_stn = np.nonzero(overlap_mask)[0].size
            
            if nlap >= nthres_all and nlap_stn >= nthres_overlap:
                
                ioa[x] = calc_ioa(tair_target[overlap_mask], tair_nghs[:, x][overlap_mask])
                overlap_mask_tair[x] = True
        
        ioa = ioa[overlap_mask_tair]
        nghs_dist = nghs_dist[overlap_mask_tair]
        mth_tair_nghs = mth_tair_nghs[:, overlap_mask_tair]
        mth_valid_mask_nghs = mth_valid_mask_nghs[:, overlap_mask_tair]
        
        if ioa.size > 0:
        
            ioa_sort = np.argsort(ioa)[::-1]
            ioa = ioa[ioa_sort]
            nghs_dist = nghs_dist[ioa_sort]
            mth_tair_nghs = mth_tair_nghs[:, ioa_sort]
            mth_valid_mask_nghs = mth_valid_mask_nghs[:, ioa_sort]
            nnghs_per_day = np.sum(mth_valid_mask_nghs, axis=1)
            
            mth_tair_target.shape = (mth_tair_target.size, 1)    
            mth_matrix = np.hstack((mth_tair_target, mth_tair_nghs))
            
            mth_valid_mask_target.shape = (mth_valid_mask_target.size, 1) 
            valid_mask_matrix = np.hstack((mth_valid_mask_target, mth_valid_mask_nghs))
        
        else:
            
            mth_tair_target.shape = (mth_tair_target.size, 1)  
            mth_matrix = mth_tair_target
            
            mth_valid_mask_target.shape = (mth_valid_mask_target.size, 1) 
            valid_mask_matrix = mth_valid_mask_target
            
            nnghs_per_day = np.zeros(mth_tair_target.shape[0])
            
        
        self.mth_matrix = mth_matrix
        self.valid_mask_matrix = valid_mask_matrix
        self.nghs_dist = nghs_dist
        self.ioa = ioa
        self.nnghs_per_day = nnghs_per_day
    
    def has_min_daily_nghs(self):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        return np.min(self.nnghs_per_day) >= MIN_DAILY_NGHBRS
    
    def merge(self, matrix2):
        '''
        Merges this mth_ngh_matrix with another mth_ngh_matrix
        @param matrix2: a mth_ngh_matrix
        '''
        
        self.mth_matrix = np.hstack((self.mth_matrix, matrix2.mth_matrix[:, 1:]))
        self.valid_mask_matrix = np.hstack((self.valid_mask_matrix, matrix2.valid_mask_matrix[:, 1:]))
        self.nghs_dist = np.concatenate((self.nghs_dist, matrix2.nghs_dist))
        self.ioa = np.concatenate((self.ioa, matrix2.ioa))
        
        if self.ioa.size > 0:
            
            ioa_sort = np.argsort(self.ioa)[::-1]
            self.ioa = self.ioa[ioa_sort]
            self.nghs_dist = self.nghs_dist[ioa_sort]
            
            ioa_sort = np.concatenate([np.zeros(1, dtype=np.int), ioa_sort + 1])
            self.mth_matrix = self.mth_matrix[:, ioa_sort]
            self.valid_mask_matrix = self.valid_mask_matrix[:, ioa_sort]
            
            self.nnghs_per_day = np.sum(self.valid_mask_matrix[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.mth_matrix.shape[1])
            
    
    def infill(self):
        '''
        Infills the target stations values for each day in a particular month using the surrounding
        neighbors
        
        @return: an array of the infilled values for each day in the time series
        '''
        trim_imp_tair_mat = self.mth_matrix
        max_stns = MAX_COLS_NORM_IMPUTE
        
        #Impute norm can only have MAX_COLS_NORM_IMPUTE stations total
        if self.mth_matrix.shape[1] > max_stns:
            trim_imp_tair_mat = trim_imp_tair_mat[:,0:max_stns]
            
        mean_tair = np.array(r.impute_norm(robjects.Matrix(trim_imp_tair_mat)))[0]
                
        return mean_tair


class mth_ngh_matrix_prcp(object):
    '''
    A class for building a data matrix of surrounding prcp amount neighbor station observations for a 
    target station for a particular month.
    '''

    def __init__(self, mthbuf_mask, mth_mask, nghs, use_prcp_only=False, wgt_by_dist=False):
        '''
        
        @param mthbuf_mask: a mask of months (from build_mth_masks) over the time series of interest with a user-defined buffer at the end/beginning of each month
        @param mth_mask: a mask of months (from build_mth_masks) over the time series of interest
        @param nghs: a ngh_matrix object
        @param use_prcp_only: only use days when there was actual prcp at the target to build a model with ngh stations (default: False)
        '''
        
        valid_mask = nghs.valid_mask_matrix
        nghs_matrix = nghs.angh_matrix
        nghs_dist = nghs.ngh_dists
        
        #########################################################################
        #mask by month buffer mask
        prcp_target = nghs_matrix[:, 0].astype(np.float64)
        prcp_target = prcp_target[mthbuf_mask]
        valid_mask_target = valid_mask[:, 0]
        valid_mask_target = valid_mask_target[mthbuf_mask]
        valid_mask_target_prcp_only = np.logical_and(valid_mask_target,prcp_target > 0)
        
        prcp_nghs = nghs_matrix[:, 1:].astype(np.float64)
        prcp_nghs = prcp_nghs[mthbuf_mask, :]
        valid_mask_nghs = valid_mask[:, 1:]
        valid_mask_nghs = valid_mask_nghs[mthbuf_mask, :]
        valid_mask_nghs_prcp_only = np.logical_and(valid_mask_nghs,prcp_nghs > 0)
        
        #########################################################################
        #mask by month mask
        mth_prcp_target = nghs_matrix[:, 0].astype(np.float64)
        mth_prcp_target = mth_prcp_target[mth_mask]
        mth_valid_mask_target = valid_mask[:, 0]
        mth_valid_mask_target = mth_valid_mask_target[mth_mask]
        mth_valid_mask_target_prcp_only = np.logical_and(mth_valid_mask_target,mth_prcp_target > 0)
        
        mth_prcp_nghs = nghs_matrix[:, 1:].astype(np.float64)
        mth_prcp_nghs = mth_prcp_nghs[mth_mask, :]
        mth_valid_mask_nghs = valid_mask[:, 1:]
        mth_valid_mask_nghs = mth_valid_mask_nghs[mth_mask, :]
        mth_valid_mask_nghs_prcp_only = np.logical_and(mth_valid_mask_nghs,mth_prcp_nghs > 0)
        
        #########################################################################
        
        nnghs = prcp_nghs.shape[1]
        
        if use_prcp_only:
        
            ntarget_valid = np.nonzero(valid_mask_target_prcp_only)[0].size
            
            if ntarget_valid < 10:
                #not enough days with prcp for target in month+buffer to build reliable model
                #switch over to use_prcp_only = False for this month
                ntarget_valid = np.nonzero(valid_mask_target)[0].size
                use_prcp_only = False
                print "Switched from prcp only!"
        
        else:
     
            ntarget_valid = np.nonzero(valid_mask_target)[0].size
        
        nthres_overlap = np.round(MIN_POR_OVERLAP * ntarget_valid) 
        
        overlap_mask_prcp = np.zeros(nnghs, dtype=np.bool)
        wgt = np.zeros(nnghs)
        mdls = np.zeros((1, nnghs))
        
        for x in np.arange(nnghs):
            
            if use_prcp_only:
                valid_mask_ngh = valid_mask_nghs_prcp_only[:, x]
                overlap_mask = np.logical_and(valid_mask_target_prcp_only, valid_mask_ngh)
                ntarget_prcp = np.sum(prcp_target[valid_mask_target_prcp_only] > 0)
            else:
                valid_mask_ngh = valid_mask_nghs[:, x]
                overlap_mask = np.logical_and(valid_mask_target, valid_mask_ngh)
                ntarget_prcp = np.sum(prcp_target[valid_mask_target] > 0)
                
            nlap = np.nonzero(overlap_mask)[0].size
            
            if nlap >= nthres_overlap:
                
                if ntarget_prcp > 0:
                
                    a_wgt = calc_hss(prcp_target[overlap_mask] > 0, prcp_nghs[:, x][overlap_mask] > 0)
                    
                    if a_wgt > 0:
                
                        wgt[x] = a_wgt
                            
                        mean_prcp_target = np.mean(prcp_target[overlap_mask])
                        mean_prcp_ngh = np.mean(prcp_nghs[:, x][overlap_mask])
                        
                        if mean_prcp_ngh == 0:
                            # no observed prcp at ngh station
                            mdls[0,x] = 1.0
                        else:
                            mdls[0,x] = mean_prcp_target/mean_prcp_ngh

                        overlap_mask_prcp[x] = True
                        
                else:
                    #no prcp observed at target during overlap
                    wgt[x] = 0.1
                    mdls[0,x] = 1.0
                    overlap_mask_prcp[x] = True
        
        
        wgt = wgt[overlap_mask_prcp]
        nghs_dist = nghs_dist[overlap_mask_prcp]
        
        if np.sum(wgt==0.1) == wgt.size or wgt_by_dist:
            #wgts are the same for every ngh. Wgt by inv. dist instead of ioa/hss metric
            wgt = 1.0/(nghs_dist**2)
            wgt_by_dist = True
            
        mth_prcp_nghs = mth_prcp_nghs[:, overlap_mask_prcp]
        mth_valid_mask_nghs = mth_valid_mask_nghs[:, overlap_mask_prcp]
        mdls = mdls[:, overlap_mask_prcp]
        
        if wgt.size > 0:
        
            wgt_sort = np.argsort(wgt)[::-1]
            wgt = wgt[wgt_sort]
            nghs_dist = nghs_dist[wgt_sort]
            mth_prcp_nghs = mth_prcp_nghs[:, wgt_sort]
            mth_valid_mask_nghs = mth_valid_mask_nghs[:, wgt_sort]
            mdls = mdls[:, wgt_sort]
            nnghs_per_day = np.sum(mth_valid_mask_nghs, axis=1)
            
            mth_prcp_target.shape = (mth_prcp_target.size, 1)    
            mth_matrix = np.hstack((mth_prcp_target, mth_prcp_nghs))
            
            mth_valid_mask_target.shape = (mth_valid_mask_target.size, 1) 
            valid_mask_matrix = np.hstack((mth_valid_mask_target, mth_valid_mask_nghs))
        
        else:
            
            mth_prcp_target.shape = (mth_prcp_target.size, 1)  
            mth_matrix = mth_prcp_target
            
            mth_valid_mask_target.shape = (mth_valid_mask_target.size, 1) 
            valid_mask_matrix = mth_valid_mask_target
            
            nnghs_per_day = np.zeros(mth_prcp_target.shape[0])
            
        
        self.mth_matrix = mth_matrix
        self.valid_mask_matrix = valid_mask_matrix
        self.nghs_dist = nghs_dist
        self.wgt = wgt
        self.mdls = mdls
        self.nnghs_per_day = nnghs_per_day
        self.wgt_by_dist = wgt_by_dist
    
    def has_min_daily_nghs(self):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        return np.min(self.nnghs_per_day) >= MIN_DAILY_NGHBRS
    
    def merge(self, matrix2):
        '''
        Merges this mth_ngh_matrix with another mth_ngh_matrix
        @param matrix2: a mth_ngh_matrix
        '''
        
        self.mth_matrix = np.hstack((self.mth_matrix, matrix2.mth_matrix[:, 1:]))
        self.valid_mask_matrix = np.hstack((self.valid_mask_matrix, matrix2.valid_mask_matrix[:, 1:]))
        self.nghs_dist = np.concatenate((self.nghs_dist, matrix2.nghs_dist))
        self.wgt = np.concatenate((self.wgt, matrix2.wgt))
        self.mdls = np.hstack((self.mdls, matrix2.mdls))
        
        if self.wgt.size > 0:
            
            wgt_sort = np.argsort(self.wgt)[::-1]
            self.wgt = self.wgt[wgt_sort]
            self.nghs_dist = self.nghs_dist[wgt_sort]
            self.mdls = self.mdls[:, wgt_sort]
            
            wgt_sort = np.concatenate([np.zeros(1, dtype=np.int), wgt_sort + 1])
            self.mth_matrix = self.mth_matrix[:, wgt_sort]
            self.valid_mask_matrix = self.valid_mask_matrix[:, wgt_sort]
            
            self.nnghs_per_day = np.sum(self.valid_mask_matrix[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.mth_matrix.shape[1])
            
    
    def infill(self):
        '''
        Infills the target stations values for each day in a particular month using the surrounding
        neighbors
        
        @return: an array of the infilled values for each day in the time series
        '''
        
        ndays = self.mth_matrix.shape[0]
        
        infill_vals = np.empty(ndays)
        
        for x in np.arange(ndays):
            
            obs_day = self.mth_matrix[x, 1:]
            valid_mask_day = self.valid_mask_matrix[x, 1:]
            
            obs_day = obs_day[valid_mask_day][0:MAX_DAILY_NGHBRS]
            wgt_day = self.wgt[valid_mask_day][0:MAX_DAILY_NGHBRS]
            mdl_day = self.mdls[:, valid_mask_day][:, 0:MAX_DAILY_NGHBRS]
            
            #((obs - mean ngh)*fstd) + mean target
            #mdl_day_vals = ((obs_day - mdl_day[0, :])*mdl_day[1, :])+mdl_day[2, :]
            
            mdl_day_vals = obs_day * mdl_day[0, :]
            
            wgts = wgt_day
            
            infill_vals[x] = np.average(mdl_day_vals, weights=wgts)
        
        return infill_vals
         
def build_mth_masks(days, buffer=0):
    '''
    Builds a mask for each month over a time series
    @param days: a days object from utils.util_dates.get_days_metadata 
    @param buffer: a buffer, in days, to add on to the beginning and end of a month
    @return: a list of 12 month masks
    '''
    
    mth_masks = []
    
    if buffer == 0:
    
        for mth in np.arange(1, 13):
            mth_masks.append(days[MONTH] == mth)
    
    else:
        
        delta_buff = timedelta(days=buffer)
        for mth in np.arange(1, 13):
            
            mth_date_str, mth_date_end = MTH_SRT_END_DATES[mth]
            str_date_buf = mth_date_str - delta_buff
            end_date_buf = mth_date_end + delta_buff
        
            mask_mth = np.logical_and(days[MONTH] == str_date_buf.month, days[DAY] >= str_date_buf.day) 
            mask_mth = np.logical_or(mask_mth, days[MONTH] == mth)
            mask_mth = np.logical_or(mask_mth, np.logical_and(days[MONTH] == end_date_buf.month, days[DAY] <= end_date_buf.day))
            
            mth_masks.append(mask_mth)
            
    return mth_masks

class ImputeMatrix(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station and performing imputation to determine the statistical distribution of a station's
    observations over a set time period (e.g.mean and variance).
    '''

    def __init__(self, stn_id, stn_da, stns_mask, tair_var, nnr_ds, aclib, min_dist= -1, max_dist=MAX_DISTANCE, tair_mask=None,trim_nan=True,add_bestngh=True):
        '''
        
        @param stn_id: the stn_id of the target
        @param stn_da: a station_data_ncdb object
        @param tair_var: the tair variable (tmin, tmax)
        @param min_dist: the min distance for which to search for neighbors (exclusive)
        @param max_dist: the max distance for which to search for neighbors (inclusive)
        @param nnr_ds: a NNRds for loading reanalysis data from nearest grid cells
        @param tair_mask: a mask for which observations at the target should be set to nan (default: None)
        '''
    
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
        
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0]
        target_tair = target_tair.astype(np.float64)
        
        if tair_mask is not None and trim_nan:
            day_mask = np.isfinite(target_tair)    
        else:
            day_mask = np.ones(target_tair.size,dtype=np.bool)
        
        if tair_mask is not None:
            target_tair[tair_mask] = np.nan
        
        #Number of observations threshold for entire period that is being infilled
        nthres_all = np.round(MIN_POR_OVERLAP * target_tair.size)
        
        #Number of observations threshold just for the target's period of record
        valid_tair_mask = np.isfinite(target_tair)
        ntair_valid = np.nonzero(valid_tair_mask)[0].size
        nthres_target_por = np.round(MIN_POR_OVERLAP * ntair_valid)    
        
        #Make sure to not include the target station itself as a neighbor station
        stns_mask = np.logical_and(stn_da.stns[STN_ID] != stn_id,stns_mask)
        all_stns = stn_da.stns[stns_mask]
        
        dists = utlg.grt_circle_dist(stn[LON], stn[LAT], all_stns[LON], all_stns[LAT])
        mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        while np.nonzero(mask_dists)[0].size == 0:
            max_dist += MAX_DISTANCE / 2.0
            mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
                
        ngh_stns = all_stns[mask_dists]
        ngh_dists = dists[mask_dists]
        
        ngh_ids = ngh_stns[STN_ID]
        ngh_tair = stn_da.load_all_stn_obs_var(ngh_ids, tair_var, set_flagged_nan=True)[0]
        ngh_tair = ngh_tair.astype(np.float64)
        
        if len(ngh_tair.shape) == 1:
            ngh_tair.shape = (ngh_tair.size, 1) 
        
        dist_sort = np.argsort(ngh_dists)
        ngh_stns = ngh_stns[dist_sort]
        ngh_dists = ngh_dists[dist_sort]
        ngh_tair = ngh_tair[:, dist_sort]
        
        overlap_mask_tair = np.zeros(ngh_stns.size, dtype=np.bool)
        ioa = np.zeros(ngh_stns.size)
        
        best_ioa = 0
        i = None
        
        for x in np.arange(ngh_stns.size):
            
            valid_ngh_mask = np.isfinite(ngh_tair[:, x])
            
            nlap = np.nonzero(valid_ngh_mask)[0].size
            
            overlap_mask = np.logical_and(valid_tair_mask, valid_ngh_mask)
            
            nlap_stn = np.nonzero(overlap_mask)[0].size
            
            if nlap >= nthres_all and nlap_stn >= nthres_target_por:
            #if nlap_stn >= nthres_target_por:
                ioa[x] = calc_ioa(target_tair[overlap_mask], ngh_tair[:, x][overlap_mask])

                overlap_mask_tair[x] = True
            
            elif nlap_stn >= nthres_target_por and add_bestngh:
                
                aioa = calc_ioa(target_tair[overlap_mask], ngh_tair[:, x][overlap_mask])
                
                if aioa > best_ioa:
                    
                    ioa[x] = aioa
                    overlap_mask_tair[x] = True
                    
                    if i != None:
                        overlap_mask_tair[i] = False
                    
                    i = x
                    best_ioa = aioa
        
        if add_bestngh and i is not None:
            
            if ioa[i] != np.max(ioa) or ioa[i] < 0.7:
                
                overlap_mask_tair[i] = False
                
#            else:
#                
#                valid_ngh_mask = np.isfinite(ngh_tair[:, i])
#                overlap_mask = np.logical_and(valid_tair_mask, valid_ngh_mask)
#                ngh_tair[np.logical_not(overlap_mask),i] = np.nan
                
        
        ioa = ioa[overlap_mask_tair]
        ngh_dists = ngh_dists[overlap_mask_tair]
        ngh_tair = ngh_tair[:, overlap_mask_tair]
        
        if ioa.size > 0:
            
            ioa_sort = np.argsort(ioa)[::-1]
            ioa = ioa[ioa_sort]
            ngh_dists = ngh_dists[ioa_sort]
            ngh_tair = ngh_tair[:, ioa_sort]
            
            target_tair.shape = (target_tair.size, 1)
            
            imp_tair_mat = np.hstack((target_tair, ngh_tair))
            ngh_dists = np.concatenate((np.zeros(1), ngh_dists))
            ioa = np.concatenate((np.ones(1), ioa))
            
            valid_imp_mask = np.isfinite(imp_tair_mat)
            
            nnghs_per_day = np.sum(valid_imp_mask , axis=1)
        
        else:
            
            target_tair.shape = (target_tair.size, 1)  
            imp_tair_mat = target_tair
            
            valid_tair_mask.shape = (valid_tair_mask.size, 1) 
            valid_imp_mask = valid_tair_mask
            
            ioa = np.ones(1)
            ngh_dists = np.zeros(1)
            
            nnghs_per_day = np.zeros(target_tair.shape[0])        
        
        #############################################################
        self.imp_tair_mat = np.array(imp_tair_mat, dtype=np.float64)
        self.valid_imp_mask = valid_imp_mask
        self.ngh_ioa = ioa
        self.ngh_dists = ngh_dists
        self.max_dist = max_dist
        self.stn_id = stn_id
        self.stn_da = stn_da
        self.tair_var = tair_var
        self.tair_mask = tair_mask
        self.nnghs_per_day = nnghs_per_day
        self.stns_mask = stns_mask
        self.nnr_ds = nnr_ds
        self.stn = stn
        self.day_mask = day_mask
        self.clib = aclib
    
    def extend_ngh_radius(self, extend_by):
        '''
        Extends the search radius for neighbor stations. The minimum of the search radius
        is the previous max distance.
        @param extend_by: The amount (km) by which to extend the radius by
        '''
        
        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by

        imp_matrix2 = ImputeMatrix(self.stn_id, self.stn_da,self.stns_mask, self.tair_var,self.nnr_ds,self.clib, min_dist, max_dist, self.tair_mask,add_bestngh=False)

        self.merge(imp_matrix2)
        self.max_dist = imp_matrix2.max_dist

    def merge(self, matrix2):
        '''
        Merges this ImputeMatrix with another ImputeMatrix
        @param matrix2: a ImputeMatrix
        '''    
    
        self.imp_tair_mat = np.hstack((self.imp_tair_mat, matrix2.imp_tair_mat[:, 1:]))
        self.valid_imp_mask = np.hstack((self.valid_imp_mask, matrix2.valid_imp_mask[:, 1:]))
        self.ngh_ioa = np.concatenate((self.ngh_ioa, matrix2.ngh_ioa[1:]))
        self.ngh_dists = np.concatenate((self.ngh_dists, matrix2.ngh_dists[1:]))
        
        if self.ngh_ioa.size > 0:
            
            ioa_sort = np.argsort(self.ngh_ioa[1:])[::-1]
            ioa_sort = np.concatenate([np.zeros(1, dtype=np.int), ioa_sort + 1])
            
            self.imp_tair_mat = self.imp_tair_mat[:, ioa_sort]
            self.valid_imp_mask = self.valid_imp_mask[:, ioa_sort]
            self.ngh_ioa = self.ngh_ioa[ioa_sort]
            self.ngh_dists = self.ngh_dists[ioa_sort]
            
            self.nnghs_per_day = np.sum(self.valid_imp_mask[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.imp_tair_mat.shape[1])

    def has_min_daily_nghs(self, nnghs,min_daily_nnghs):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        trim_valid_mask = self.valid_imp_mask[:, 0:1 + nnghs]
        nnghs_per_day = np.sum(trim_valid_mask[:, 1:], axis=1)

        return np.min(nnghs_per_day) >= min_daily_nnghs
    

    def impute(self, min_daily_nnghs=MIN_DAILY_NGHBRS,nnghs_nnr=NNGH_NNR,max_nnr_var=0.99):
        '''
        Imputes the target station's values for each day in the time series using the surrounding
        N nghs and N ncep/ncar reanalysis grid cell.
        
        @return imp_tair: an array of the imputed values for each day in the time series
        @return obs_tair: an array of the observed daily values of the target
        '''
        
        nnghs = min_daily_nnghs
        
        trim_imp_tair_mat = self.imp_tair_mat[:, 0:1 + nnghs]
        
        engh_dly_nghs = self.has_min_daily_nghs(nnghs,min_daily_nnghs)
        actual_nnghs = trim_imp_tair_mat.shape[1] - 1
        
        while actual_nnghs < nnghs or not engh_dly_nghs:
        
            if actual_nnghs == nnghs and not engh_dly_nghs:
                
                nnghs += 1
            
            else:
                
                self.extend_ngh_radius(MAX_DISTANCE / 2.0)

            trim_imp_tair_mat = self.imp_tair_mat[:, 0:1 + nnghs]
            engh_dly_nghs = self.has_min_daily_nghs(nnghs,min_daily_nnghs)
            actual_nnghs = trim_imp_tair_mat.shape[1] - 1

        #############################################################

        nnr_tair = self.nnr_ds.get_nngh_matrix(self.stn[LON],self.stn[LAT],self.tair_var,utc_offset=self.stn[UTC_OFFSET],nngh=nnghs_nnr)
        
        trim_imp_tair_mat = trim_imp_tair_mat[self.day_mask,:]
        nnr_tair = nnr_tair[self.day_mask,:]
#        
        pc_loads, pc_scores, var_explain, error = self.clib.pca_basic(nnr_tair, True, True)
        cusum_var = np.cumsum(var_explain)
    
        i = np.nonzero(cusum_var >= max_nnr_var)[0][0]
        
        #print var_explain
        nnr_tair = pc_scores[:,0:i+1]
        
#        obs_ma = np.ma.masked_array(trim_imp_tair_mat[:,0],mask=np.isnan(trim_imp_tair_mat[:,0]))
#        nnr_ma = np.ma.masked_array(nnr_tair)

#        ccoefs = np.zeros(nnr_ma.shape[1])
#        for x in np.arange(nnr_ma.shape[1]):
#            ccoefs[x] =  np.abs(np.ma.corrcoef(obs_ma, nnr_ma[:,x],allow_masked=True)[0,1])
#        
#        ccoefs_mask = ccoefs >= min_nnr_cor
#        
#        nnr_tair = nnr_tair[:,ccoefs_mask]
        
        
        #ccoefs = ccoefs[ccoefs_mask]
        
        #max out the number of nnr variables to use at the number of neighbor stations
        #nnr_tair = nnr_tair[:,np.argsort(ccoefs)[::-1][0:trim_imp_tair_mat.shape[1]]]
        #nnr_tair = nnr_tair[:,np.argsort(ccoefs)[::-1][0:30]]
        
        #preNCols = trim_imp_tair_mat.shape[1]
        trim_imp_tair_mat = shrinkMatrix(trim_imp_tair_mat, min_daily_nnghs)
        #print "".join([self.stn_id,": removed ",str(preNCols- trim_imp_tair_mat.shape[1])," cols."])
        #print "".join([self.stn_id,": has min daily nghs: ",str(self.has_min_daily_nghs(nnghs, min_daily_nnghs))])
        
        #Impute norm can only have MAX_COLS_NORM_IMPUTE columns total
        if trim_imp_tair_mat.shape[1] > MAX_COLS_NORM_IMPUTE:
            
            #Too many ngh stations, don't use NNR and trim to MAX_COLS_NORM_IMPUTE
            trim_imp_tair_mat = trim_imp_tair_mat[:,0:MAX_COLS_NORM_IMPUTE]
            
            validMask = np.isfinite(trim_imp_tair_mat)
            obsPerday = np.sum(validMask,axis=1)
            
            if np.min(obsPerday) == 0:
                
                trim_imp_tair_mat[:,-1] = nnr_tair[:,0]
                
                nNoObs = np.sum(obsPerday==0)
                
                print "".join(["WARNING: ",self.stn_id," matrix trimming caused no observations on ",str(nNoObs)," days."])
                
        else:
            
            trim_imp_tair_mat = np.hstack((trim_imp_tair_mat,nnr_tair))
        
            #Impute norm can only have MAX_COLS_NORM_IMPUTE columns total
            if trim_imp_tair_mat.shape[1] > MAX_COLS_NORM_IMPUTE:
                trim_imp_tair_mat = trim_imp_tair_mat[:,0:MAX_COLS_NORM_IMPUTE]
        
        
        #max_stns = MAX_COLS_NORM_IMPUTE
        ######################################################################
        
        #Impute norm can only have MAX_COLS_NORM_IMPUTE stations total
#        if trim_imp_tair_mat.shape[1] > max_stns:
#            
#            trim_imp_tair_mat = trim_imp_tair_mat[:,0:max_stns]
            
        #trim_imp_tair_mat = trim_imp_tair_mat[:,0]
        #trim_imp_tair_mat.shape = (trim_imp_tair_mat.size,1)
        
        #trim_imp_tair_mat = np.hstack((trim_imp_tair_mat,nnr_tair))
            
        trim_imp_tair_mat = np.require(trim_imp_tair_mat,dtype=np.float64,requirements=['C','A','W','O'])
        
        #validMask = np.isfinite(trim_imp_tair_mat)
        #obsPerday = np.sum(validMask,axis=1)
        
        fit_tair = np.array(r.impute_norm(robjects.Matrix(trim_imp_tair_mat)))
         
        obs_tair = trim_imp_tair_mat[:, 0]
        
        return fit_tair, obs_tair, nnghs, self.max_dist

def shrinkMatrix(aMatrix,minNghs):
    
    validMask = np.isfinite(aMatrix[:,1:minNghs+1])
    nObs = np.sum(validMask,axis=1)
    maskBelowMin = nObs < minNghs
    
    keepCol = np.ones(aMatrix.shape[1],dtype=np.bool)
    
    for x in np.arange(minNghs+1,aMatrix.shape[1]):
        
        aCol = aMatrix[:,x]
        aColValidMask = np.isfinite(aCol)
        
        if np.sum(np.logical_and(maskBelowMin,aColValidMask)) > 0:
            
            aColValidMask.shape = (aColValidMask.size,1)
            validMask = np.hstack((validMask,aColValidMask))
            nObs = np.sum(validMask,axis=1)
            maskBelowMin = nObs < minNghs
        
        else:
            
            keepCol[x] = False
    
    return aMatrix[:,keepCol]
            
class ngh_matrix(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station.
    '''

    def __init__(self, stn_id, stn_da, tair_var,stns_mask, min_dist=-1, max_dist=MAX_DISTANCE, tair_mask=None,trim_nan=True):
        '''
        
        @param stn_id: the stn_id of the target
        @param stn_da: a station_data_ncdb object
        @param tair_var: the tair variable (tmin, tmax)
        @param min_dist: the min distance for which to search for neighbors (exclusive)
        @param max_dist: the max distance for which to search for neighbors (inclusive)
        @param tair_mask: a mask for which observations at the target should be set to nan (default: None)
        '''
    
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
        
        #Load observations of the target
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0]
        
        
        if tair_mask is not None and trim_nan:
            day_mask = np.isfinite(target_tair)    
        else:
            day_mask = np.ones(target_tair.size,dtype=np.bool)
        
        if tair_mask is not None:
            target_tair[tair_mask] = np.nan 
        
        #Make sure to not include the target station itself as a neighbor station
        stns_mask = np.logical_and(stn_da.stns[STN_ID] != stn_id,stns_mask)
        all_stns = stn_da.stns[stns_mask]
        
        #Find stations within the min/max range
        dists = utlg.grt_circle_dist(stn[LON], stn[LAT], all_stns[LON], all_stns[LAT])
        mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        #Make sure at least one station is found. If not, increase by 1/2 the MAX_DISTANCE
        #until one or more is found
        while np.nonzero(mask_dists)[0].size == 0:
            max_dist += MAX_DISTANCE / 2.0
            mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        ngh_stns = all_stns[mask_dists]
        ngh_dists = dists[mask_dists]
        
        #Load neighbor observations    
        ngh_tair = stn_da.load_all_stn_obs_var(ngh_stns[STN_ID], tair_var, set_flagged_nan=True)[0]
        
        if ngh_stns.size == 1:
            ngh_tair.shape = (ngh_tair.size, 1)
        
        target_tair.shape = (target_tair.size, 1)
        
        matrix_tair = np.hstack((target_tair, ngh_tair))
        #ngh_dists = np.concatenate((np.zeros(1),ngh_dists))
        valid_mask_matrix = np.isfinite(matrix_tair)
        
        self.angh_matrix = np.array(matrix_tair, dtype=np.float64)
        self.valid_mask_matrix = valid_mask_matrix
        self.ngh_dists = ngh_dists
        self.max_dist = max_dist
        
        self.stn_id = stn_id
        self.stn_da = stn_da
        self.tair_var = tair_var
        self.tair_mask = tair_mask
        self.day_mask = day_mask
    
    def extend_ngh_radius(self, extend_by):
        '''
        Extends the search radius of a current ngh_matrix object. The minimum of the search radius
        is the max distance of the current neighbors.
        @param extend_by: The amount (km) by which to extend the radius by
        @return: a ngh_matrix object representing neighbors in the extended search radius  
        '''
        
#        min_dist = np.max(self.ngh_dists) + 0.00001
#        max_dist = self.max_dist + extend_by
        
        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by
        
        if max_dist > MAX_EXTEND:
            raise Exception("Max distance extended beyond maximum allowed.")
 
        return ngh_matrix(self.stn_id, self.stn_da, self.tair_var, min_dist, max_dist, tair_mask=self.tair_mask)

class ngh_matrix_prcp(ngh_matrix):
    '''
    A class for building a data matrix of surrounding neighbor station prcp amount observations for a 
    target station.
    '''

    def __init__(self, stn_id, stn_da, min_dist=0, max_dist=MAX_DISTANCE, prcp_mask=None):
        ngh_matrix.__init__(self, stn_id, stn_da, 'prcp', min_dist, max_dist, prcp_mask)
      
def calc_ioa(x, y):
    '''
    Calculate the index of agreement (Durre et al. 2010; Legates and McCabe 1999) between x and y
    '''
    
    y_mean = np.mean(y)
    d = np.sum(np.abs(x - y_mean) + np.abs(y - y_mean))
    
    if d == 0.0:
        #print "|".join(["WARNING: calc_ioa: x, y identical"])
        #The x and y series are exactly the same
        #Return a perfect ioa
        return 1.0
    
    ioa = 1.0 - (np.sum(np.abs(y - x)) / d)
    
#    if ioa == 0:
#        print "|".join(["WARNING: calc_ioa: ioa == 0"])
#        #Means all ys are the same or only one observation.
#        #This could possibly happen with prcp in arid regions
#        #Add on an extra observation to the time series that has same difference as x[0] and y[0]
#        x_new = np.concatenate([x, np.array([x[0] + (x[0] * .1)])])
#        y_new = np.concatenate([y, np.array([y[0] + (x[0] * .1)])])
#        
#        y_mean = np.mean(y_new)
#        ioa = 1.0 - (np.sum(np.abs(y_new - x_new)) / np.sum(np.abs(x_new - y_mean) + np.abs(y_new - y_mean)))  
    
    return ioa


def calc_hss(obs_po,mod_po):
    '''
    Calculates heidke skill score of modeled prcp occurrence
    See http://www.wxonline.info/topics/verif2.html
    @param obs: array of observed occurrences (1s and 0s)
    @param mod: array of modeled occurrences (1s and 0s)
    @return hss: heidke skill score
    '''
    
    #model_obs
    true_true = mod_po[np.logical_and(obs_po==1,mod_po==1)].size
    false_false = mod_po[np.logical_and(obs_po==0,mod_po==0)].size
    true_false = mod_po[np.logical_and(obs_po==0,mod_po==1)].size
    false_true = mod_po[np.logical_and(obs_po==1,mod_po==0)].size
    
    a = float(true_true)
    b = float(true_false)
    c = float(false_true)
    d = float(false_false)
    
    #special case handling
    if a == 0.0 and c == 0.0: #and b != 0:
        #This means that were no observed days of rain so can't calc
        #appropriate hss. Set a = 1 to get a more appropriate hss
        a = 1.0
        
        if b == 0:
            b = 1.0
            
        c = 1.0 
    
    if b == 0.0 and d == 0.0: #and c != 0.0:
        #This means that there was observed rain every day so can't calc
        #appropriate hss. Set d = 1 to get a more appropriate hss
        d = 1.0
        
        if c == 0:
            c = 1.0    

        b = 1.0

    den = ((a+c)*(c+d))+((a+b)*(b+d))
    
    if den == 0.0:
        #This is a perfect forecast with all true_true or false_false
        return 1.0
    
    return (2.0*((a*d)-(b*c)))/den

def build_lin_model(x, y):
    '''
    Builds a linear model of the form y~x
    @return: (slope,intercept)
    '''

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept 

def impute_tair_norm(stn_id,stn_da,stn_mask,tair_var,nnr_ds,aclib,tair_mask=None,trim_nan=True,nnghs=MIN_DAILY_NGHBRS,nnghs_nnr=NNGH_NNR):
    '''
    The main method for performing imputation to determine the statistical distribution of a station's
    observations over a set time period (e.g.mean and variance).
    '''

    a_matrix = ImputeMatrix(stn_id, stn_da, stn_mask, tair_var,nnr_ds,aclib,tair_mask=tair_mask,trim_nan=trim_nan)
    imp_tair, obs_tair, fnl_nnghs, max_dist = a_matrix.impute(nnghs,nnghs_nnr)
    
    return imp_tair, obs_tair
#self, stn_id, stn_da, stns_mask, tair_var, nnr_ds,min_dist= -1, max_dist=MAX_DISTANCE, tair_mask=None,trim_nan=True
def infill_tair(stn_id, stn_da, days, tair_var,stns_mask, mth_masks, mthbuf_masks, tair_mask=None,trim_nan=True):
    '''
    The main method for infilling missing target station values using a regression-based approach 
    similar to Durre et al. 2010 Appendix B.
    @param stn_id: the stn_id of the target
    @param stn_da: a station_data_ncdb object
    @param days: a days object from utils.util_dates.get_days_metadata representing the time series to infill/expand
    @param tair_var: the tair variable (tmin, tmax)
    @param mthbuf_mask: a mask of months (from build_mth_masks) over the time series of interest with a user-defined buffer at the end/beginning of each month
    @param mth_mask: a mask of months (from build_mth_masks) over the time series of interest
    @param tair_mask: a mask for which observations at the target should be set to nan (default: None)
    '''
    
    nghs = ngh_matrix(stn_id, stn_da, tair_var,stns_mask, tair_mask=tair_mask,trim_nan=trim_nan)
    nghs_extend_ls = []
    
    mth_nghs_ls = []
    
    for x in np.arange(len(mth_masks)):
        
        mth_nghs = mth_ngh_matrix(mthbuf_masks[x], mth_masks[x], nghs)
        
        i = 0
        while not mth_nghs.has_min_daily_nghs():
            
            try:
                
                nghs_extend = nghs_extend_ls[i]
            
            except IndexError:
                
                if i == 0:
                    
                    nghs_extend_ls.append(nghs.extend_ngh_radius(MAX_DISTANCE / 2.0))
                
                else:
                    
                    nghs_extend_ls.append(nghs_extend_ls[i - 1].extend_ngh_radius(MAX_DISTANCE / 2.0))
            
                nghs_extend = nghs_extend_ls[i]
            
            mth_nghs_extend = mth_ngh_matrix(mthbuf_masks[x], mth_masks[x], nghs_extend)
            mth_nghs.merge(mth_nghs_extend)
            i += 1
        
        mth_nghs_ls.append(mth_nghs)
    
    fit_tair = np.empty(12)
    
    for x in np.arange(len(mth_masks)):
        
        mth_nghs = mth_nghs_ls[x]
        
        fit_tair[x] = mth_nghs.infill()
    
    obs_tair = nghs.angh_matrix[:, 0]
    
    return np.mean(fit_tair)

def infill_prcp_norm(stn_id, stn_da, days,mth_masks, mthbuf_masks, use_prcp_only=False, prcp_mask=None):
    '''
    The main method for infilling missing target station prcp amount values using a weighted average-based approach 
    similar to Durre et al. 2010 Appendix B.
    @param stn_id: the stn_id of the target
    @param stn_da: a station_data_ncdb object
    @param days: a days object from utils.util_dates.get_days_metadata representing the time series to infill/expand
    @param mthbuf_mask: a mask of months (from build_mth_masks) over the time series of interest with a user-defined buffer at the end/beginning of each month
    @param mth_mask: a mask of months (from build_mth_masks) over the time series of interest
    @param use_prcp_only: only use days when there was actual prcp at the target to build a model with ngh stations (default: False)
    @param prcp_mask: a mask for which observations at the target should be set to nan (default: None)
    '''
    
    nghs = ngh_matrix_prcp(stn_id, stn_da, prcp_mask=prcp_mask)
    nghs_extend_ls = []
    
    mth_nghs_ls = []
    
    for x in np.arange(len(mth_masks)):
        
        mth_nghs = mth_ngh_matrix_prcp(mthbuf_masks[x], mth_masks[x], nghs,use_prcp_only)
        
        i = 0
        while not mth_nghs.has_min_daily_nghs():
            
            try:
                
                nghs_extend = nghs_extend_ls[i]
            
            except IndexError:
                
                if i == 0:
                    
                    nghs_extend_ls.append(nghs.extend_ngh_radius(MAX_DISTANCE / 2.0))
                
                else:
                    
                    nghs_extend_ls.append(nghs_extend_ls[i - 1].extend_ngh_radius(MAX_DISTANCE / 2.0))
            
                nghs_extend = nghs_extend_ls[i]
            
            mth_nghs_extend = mth_ngh_matrix_prcp(mthbuf_masks[x], mth_masks[x], nghs_extend,use_prcp_only,mth_nghs.wgt_by_dist)
            mth_nghs.merge(mth_nghs_extend)
            i += 1
        
        mth_nghs_ls.append(mth_nghs)
    
    fit_prcp = np.zeros(days.size)
    
    for x in np.arange(len(mth_masks)):
        
        mth_nghs = mth_nghs_ls[x]
        
        fit_prcp[mth_masks[x]] = mth_nghs.infill()
    
    obs_prcp = nghs.angh_matrix[:, 0]
    
    obs_mask = np.isfinite(obs_prcp)
    fill_mask = np.logical_not(obs_mask)
    
    fnl_prcp = np.copy(obs_prcp)
    fnl_prcp[fill_mask] = fit_prcp[fill_mask]
    
    rslts = prcp_norm_results(stn_id)
    rslts.prcp = fnl_prcp
    rslts.prcp_fit = fit_prcp
    rslts.fill_mask = fill_mask
    rslts.obs_mask = obs_mask
    
    return rslts

class prcp_norm_results(object):
    '''
    A class for holding results from infill_prcp_norm
    '''

    def __init__(self, stn_id):
        
        self.stn_id = stn_id
        
        self.prcp = None
        self.prcp_fit = None
        self.fill_mask = None
        self.obs_mask = None
        
        self.perr_ttlamt = None
    
    def calc_obs_vs_fit_stats(self):
        
        #Total amount cm
        self.perr_ttlamt = ((np.sum(self.prcp_fit[self.obs_mask]) - np.sum(self.prcp[self.obs_mask])) / np.sum(self.prcp[self.obs_mask])) * 100.
           
if __name__ == '__main__':
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_da.set_day_mask(19480101, 20111231)
    days = stn_da.days[stn_da.day_mask]
    stn_id = 'GHCN_CA007090960'
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days, MTH_BUFFER)
    tair_var = 'tmax'
    #n_yrs_mod = 5
    #nmask = int(np.round(n_yrs_mod*365.25))
    #idxs = np.arange(days.size)
    
    #obs_tair = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0],dtype=np.float64)
    
    #fin_tair = np.isfinite(obs_tair)
    #last_idxs = np.nonzero(fin_tair)[0][-nmask:]
    #xval_mask_tair = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_tair)
    
    infill_tmin, obs_tmin = infill_tair(stn_id, stn_da, days, tair_var, mth_masks, mthbuf_masks)#,xval_mask_tair)
    
    fin_mask = np.isfinite(obs_tmin)
    #xval_mask_tair = fin_mask
    
    difs = infill_tmin[fin_mask] - obs_tmin[fin_mask]
    
    #difs = infill_tmin[xval_mask_tair] - obs_tair[xval_mask_tair]
    
    print np.mean(np.abs(difs))
    print np.mean(difs)
#    
#    plt.plot(difs)
#    plt.show()
    
            
            
