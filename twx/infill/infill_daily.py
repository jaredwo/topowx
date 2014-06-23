'''
Classes and functions for performing principal component analysis based infilling of 
daily missing station observations. Uses the R package pcaMethods from  the BioConductor
repository (http://www.bioconductor.org/packages/release/bioc/html/pcaMethods.html). 

@author: jared.oyler
'''
import matplotlib.pyplot as plt
import numpy as np
from twx.db.station_data import STN_ID, LON, LAT,MONTH,UTC_OFFSET
import twx.utils.util_geo as utlg
import sys
import twx.utils.util_misc as utlm
sys.stdout = utlm.Unbuffered(sys.stdout)
from twx.utils.util_dates import YEAR
from twx.infill.infill_normals import infill_prcp_norm
import netCDF4
import scipy.stats as ss
import twx.interp.clibs as clibs
#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri
r = robjects.r
from scipy import stats

MIN_POR_OVERLAP = 2.0 / 3.0#.95#.95#3.0/4.0
MAX_DISTANCE = 100
MAX_NGHS_LOAD = 2000
MAX_NNR_VAR = 0.99
MIN_NNR_VAR = 0.90
MIN_DAILY_NGHBRS = 3
NNGH_NNR = 4
PO_THRESHS = np.arange(0.0001, 0.9999, .0001)
#PO_THRESHS = np.arange(0.01,1.0,0.01)

MIN_POR_OVERLAP_STNPOR = 2.0 / 3.0
MIN_POR_OVERLAP_ALL = 2.0 / 3.0
MIN_NNR_COR = 0.5

FRAC_OBS_PO = np.array([0.85])
FRAC_OBS_PRCP = np.array([0.85])
MAX_R2CUM_PRCP = np.array([0.95])

NODATA_NORMS = netCDF4.default_fillvals['f4']

P_PATH_RPCA = '/home/jared.oyler/ecl_helios_workspace/wxTopo_R/pca_infill.R'

def source_r(path_rpca=P_PATH_RPCA):
    r.source(path_rpca)

class ioapca_matrix(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station to used for pca-based infilling.
    '''

    def __init__(self, stn_id, stn_da, tair_var, norms, min_dist= -1, max_dist=MAX_DISTANCE, tair_mask=None):
        '''
        
        @param stn_id: the stn_id of the target
        @param stn_da: a station_data_ncdb object
        @param tair_var: the tair variable (tmin, tmax)
        @param norms: an array of the estimated mean daily values at every station for the time series
        @param min_dist: the min distance for which to search for neighbors (exclusive)
        @param max_dist: the max distance for which to search for neighbors (inclusive)
        @param tair_mask: a mask for which observations at the target should be set to nan (default: None)
        '''
    
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
        
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0]
        target_norm = norms[stn_da.stn_ids == stn_id][0]
        
        if tair_mask is not None:
            target_tair[tair_mask] = np.nan
        
        #Number of observations threshold for entire period that is being infilled
        nthres_all = np.round(MIN_POR_OVERLAP * target_tair.size)
        
        #Number f observations threshold just for the target's period of record
        valid_tair_mask = np.isfinite(target_tair)
        ntair_valid = np.nonzero(valid_tair_mask)[0].size
        nthres_target_por = np.round(MIN_POR_OVERLAP * ntair_valid)    
        
        #Make sure to not include the target station itself as a neighbor station
        stns_mask = np.logical_and(stn_da.stns[STN_ID] != stn_id,norms != NODATA_NORMS)
        all_stns = stn_da.stns[stns_mask]
        
        dists = utlg.grt_circle_dist(stn[LON], stn[LAT], all_stns[LON], all_stns[LAT])
        mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        while np.nonzero(mask_dists)[0].size == 0:
            max_dist += MAX_DISTANCE / 2.0
            mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        ngh_stns = all_stns[mask_dists]
        ngh_dists = dists[mask_dists]
        
        ngh_ids = ngh_stns[STN_ID]
        ngh_norms = norms[np.in1d(stn_da.stn_ids, ngh_ids, assume_unique=True)]
        ngh_tair = stn_da.load_all_stn_obs_var(ngh_ids, tair_var, set_flagged_nan=True)[0]
        
        if len(ngh_tair.shape) == 1:
            ngh_tair.shape = (ngh_tair.size, 1) 
        
        dist_sort = np.argsort(ngh_dists)
        ngh_stns = ngh_stns[dist_sort]
        ngh_dists = ngh_dists[dist_sort]
        ngh_norms = ngh_norms[dist_sort]
        ngh_tair = ngh_tair[:, dist_sort]
        
        overlap_mask_tair = np.zeros(ngh_stns.size, dtype=np.bool)
        
        for x in np.arange(ngh_stns.size):
            
            valid_ngh_mask = np.isfinite(ngh_tair[:, x])
            
            nlap = np.nonzero(valid_ngh_mask)[0].size
            
            overlap_mask = np.logical_and(valid_tair_mask, valid_ngh_mask)
            
            nlap_stn = np.nonzero(overlap_mask)[0].size
            
            if nlap >= nthres_all and nlap_stn >= nthres_target_por:
                
                overlap_mask_tair[x] = True
        
        ngh_dists = ngh_dists[overlap_mask_tair]
        ngh_tair = ngh_tair[:, overlap_mask_tair]
        ngh_norms = ngh_norms[overlap_mask_tair]
        
        if ngh_dists.size > 0:
            
            wgts = self.__get_wgts(target_tair, ngh_tair)
            wgt_sort = np.argsort(wgts)[::-1]
            
            wgts = wgts[wgt_sort]
            ngh_dists = ngh_dists[wgt_sort]
            ngh_tair = ngh_tair[:, wgt_sort]
            ngh_norms = ngh_norms[wgt_sort]
            
            target_tair.shape = (target_tair.size, 1)
            
            pca_tair = np.hstack((target_tair, ngh_tair))
            ngh_dists = np.concatenate((np.zeros(1), ngh_dists))
            wgts = np.concatenate((np.ones(1), wgts))
            ngh_norms = np.concatenate((np.array([target_norm]), ngh_norms))
            
            valid_pca_mask = np.isfinite(pca_tair)
            
            nnghs_per_day = np.sum(valid_pca_mask , axis=1)
        
        else:
            
            target_tair.shape = (target_tair.size, 1)  
            pca_tair = target_tair
            
            valid_tair_mask.shape = (valid_tair_mask.size, 1) 
            valid_pca_mask = valid_tair_mask
            
            wgts = np.ones(1)
            ngh_dists = np.zeros(1)
            ngh_norms = np.array([target_norm])
            
            nnghs_per_day = np.zeros(target_tair.shape[0])        
        
        #############################################################
        self.pca_tair = np.array(pca_tair, dtype=np.float64)
        self.valid_pca_mask = valid_pca_mask
        self.ngh_wgts = wgts
        self.ngh_dists = ngh_dists
        self.ngh_norms = ngh_norms
        self.max_dist = max_dist
        self.stn_id = stn_id
        self.stn_da = stn_da
        self.tair_var = tair_var
        self.tair_mask = tair_mask
        self.norms = norms
        self.nnghs_per_day = nnghs_per_day
    
    def __get_wgts(self,tair_target,tair_ngh):
        
        ioa = np.zeros(tair_ngh.shape[1])
        valid_tair_mask = np.isfinite(tair_target)
        
        for x in np.arange(tair_ngh.shape[1]):
            
            valid_ngh_mask = np.isfinite(tair_ngh[:, x])
            
            overlap_mask = np.logical_and(valid_tair_mask, valid_ngh_mask)
                
            ioa[x] = _calc_ioa(tair_target[overlap_mask], tair_ngh[:, x][overlap_mask])
                  
        return ioa
    
    def extend_ngh_radius(self, extend_by):
        '''
        Extends the search radius for neighbor stations. The minimum of the search radius
        is the previous max distance.
        @param extend_by: The amount (km) by which to extend the radius by
        '''
        
        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by

        pca_matrix2 = ioapca_matrix(self.stn_id, self.stn_da, self.tair_var, self.norms, min_dist, max_dist, self.tair_mask)

        self.merge(pca_matrix2)
        self.max_dist = max_dist

    def merge(self, matrix2):
        '''
        Merges this pca_matrix with another pca_matrix
        @param matrix2: a mth_ngh_matrix
        '''    
    
        self.pca_tair = np.hstack((self.pca_tair, matrix2.pca_tair[:, 1:]))
        self.valid_pca_mask = np.hstack((self.valid_pca_mask, matrix2.valid_pca_mask[:, 1:]))
        self.ngh_dists = np.concatenate((self.ngh_dists, matrix2.ngh_dists[1:]))
        self.ngh_norms = np.concatenate((self.ngh_norms, matrix2.ngh_norms[1:]))
        
        if self.ngh_dists.size > 0:
            
            max_dist = np.max(np.array((self.max_dist,matrix2.max_dist)))
                
            wgts = self.__get_wgts(self.pca_tair[:,0],self.pca_tair)
            wgt_sort = np.argsort(wgts)[::-1]
            wgts = wgts[wgt_sort]
            
            self.pca_tair = self.pca_tair[:, wgt_sort]
            self.valid_pca_mask = self.valid_pca_mask[:, wgt_sort]
            self.ngh_wgts = wgts
            self.ngh_dists = self.ngh_dists[wgt_sort]
            self.ngh_norms = self.ngh_norms[wgt_sort]
            
            self.nnghs_per_day = np.sum(self.valid_pca_mask[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.pca_tair.shape[1])

    def has_min_daily_nghs(self, nnghs):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        trim_valid_mask = self.valid_pca_mask[:, 0:1 + nnghs]
        nnghs_per_day = np.sum(trim_valid_mask[:, 1:], axis=1)
        
        return np.min(nnghs_per_day) >= MIN_DAILY_NGHBRS

    def infill(self, nnghs):
        '''
        Infills the target station's values for each day in the time series using the surrounding
        neighbors and a pca-based approach.
        
        @return fit_tair: an array of the infilled values for each day in the time series
        @return obs_tair: an array of the observed daily values of the target
        '''
        
        trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
        trim_ngh_norms = self.ngh_norms[0:1 + nnghs]
        trim_ngh_wgts = self.ngh_wgts[0:1 + nnghs]
        
        engh_dly_nghs = self.has_min_daily_nghs(nnghs)
        actual_nnghs = trim_pca_tair.shape[1] - 1
        
        while actual_nnghs < nnghs or not engh_dly_nghs:
        
            if actual_nnghs == nnghs and not engh_dly_nghs:
                
                nnghs += 1
            
            else:
                
                self.extend_ngh_radius(MAX_DISTANCE / 2.0)

            trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
            trim_ngh_norms = self.ngh_norms[0:1 + nnghs]
            trim_ngh_wgts = self.ngh_wgts[0:1 + nnghs]
            engh_dly_nghs = self.has_min_daily_nghs(nnghs)
            actual_nnghs = trim_pca_tair.shape[1] - 1
        
        trim_pca_tair = trim_pca_tair * trim_ngh_wgts
        trim_ngh_norms = trim_ngh_norms * trim_ngh_wgts
        #ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(trim_pca_tair))
        ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(trim_pca_tair), robjects.FloatVector(trim_ngh_norms))
        fit_tair = np.array(ppca_rslt.rx('ppca_fit'))
        fit_tair.shape = (fit_tair.shape[1],)
        npcs = ppca_rslt.rx('npcs')[0][0]
        
        obs_tair = trim_pca_tair[:, 0]
        
        return fit_tair, obs_tair, npcs, nnghs, self.max_dist


class gwrpca_matrix(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station to used for pca-based infilling.
    '''

    def __init__(self, stn_id, stn_da, tair_var, norms, min_dist= -1, max_dist=MAX_DISTANCE, tair_mask=None):
        '''
        
        @param stn_id: the stn_id of the target
        @param stn_da: a station_data_ncdb object
        @param tair_var: the tair variable (tmin, tmax)
        @param norms: an array of the estimated mean daily values at every station for the time series
        @param min_dist: the min distance for which to search for neighbors (exclusive)
        @param max_dist: the max distance for which to search for neighbors (inclusive)
        @param tair_mask: a mask for which observations at the target should be set to nan (default: None)
        '''
    
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
        
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0]
        target_norm = norms[stn_da.stn_ids == stn_id][0]
        
        if tair_mask is not None:
            target_tair[tair_mask] = np.nan
        
        #Number of observations threshold for entire period that is being infilled
        nthres_all = np.round(MIN_POR_OVERLAP * target_tair.size)
        
        #Number f observations threshold just for the target's period of record
        valid_tair_mask = np.isfinite(target_tair)
        ntair_valid = np.nonzero(valid_tair_mask)[0].size
        nthres_target_por = np.round(MIN_POR_OVERLAP * ntair_valid)    
        
        #Make sure to not include the target station itself as a neighbor station
        stns_mask = np.logical_and(stn_da.stns[STN_ID] != stn_id,norms != NODATA_NORMS)
        all_stns = stn_da.stns[stns_mask]
        
        dists = utlg.grt_circle_dist(stn[LON], stn[LAT], all_stns[LON], all_stns[LAT])
        mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        while np.nonzero(mask_dists)[0].size == 0:
            max_dist += MAX_DISTANCE / 2.0
            mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        ngh_stns = all_stns[mask_dists]
        ngh_dists = dists[mask_dists]
        
        ngh_ids = ngh_stns[STN_ID]
        ngh_norms = norms[np.in1d(stn_da.stn_ids, ngh_ids, assume_unique=True)]
        ngh_tair = stn_da.load_all_stn_obs_var(ngh_ids, tair_var, set_flagged_nan=True)[0]
        
        if len(ngh_tair.shape) == 1:
            ngh_tair.shape = (ngh_tair.size, 1) 
        
        dist_sort = np.argsort(ngh_dists)
        ngh_stns = ngh_stns[dist_sort]
        ngh_dists = ngh_dists[dist_sort]
        ngh_norms = ngh_norms[dist_sort]
        ngh_tair = ngh_tair[:, dist_sort]
        
        overlap_mask_tair = np.zeros(ngh_stns.size, dtype=np.bool)
        
        for x in np.arange(ngh_stns.size):
            
            valid_ngh_mask = np.isfinite(ngh_tair[:, x])
            
            nlap = np.nonzero(valid_ngh_mask)[0].size
            
            overlap_mask = np.logical_and(valid_tair_mask, valid_ngh_mask)
            
            nlap_stn = np.nonzero(overlap_mask)[0].size
            
            if nlap >= nthres_all and nlap_stn >= nthres_target_por:
                
                overlap_mask_tair[x] = True
        
        ngh_dists = ngh_dists[overlap_mask_tair]
        ngh_tair = ngh_tair[:, overlap_mask_tair]
        ngh_norms = ngh_norms[overlap_mask_tair]
        
        if ngh_dists.size > 0:
            
            wgts = self.__get_wgts(ngh_dists, max_dist)
            wgt_sort = np.argsort(wgts)[::-1]
            
            wgts = wgts[wgt_sort]
            ngh_dists = ngh_dists[wgt_sort]
            ngh_tair = ngh_tair[:, wgt_sort]
            ngh_norms = ngh_norms[wgt_sort]
            
            target_tair.shape = (target_tair.size, 1)
            
            pca_tair = np.hstack((target_tair, ngh_tair))
            ngh_dists = np.concatenate((np.zeros(1), ngh_dists))
            wgts = np.concatenate((np.ones(1), wgts))
            ngh_norms = np.concatenate((np.array([target_norm]), ngh_norms))
            
            valid_pca_mask = np.isfinite(pca_tair)
            
            nnghs_per_day = np.sum(valid_pca_mask , axis=1)
        
        else:
            
            target_tair.shape = (target_tair.size, 1)  
            pca_tair = target_tair
            
            valid_tair_mask.shape = (valid_tair_mask.size, 1) 
            valid_pca_mask = valid_tair_mask
            
            wgts = np.ones(1)
            ngh_dists = np.zeros(1)
            ngh_norms = np.array([target_norm])
            
            nnghs_per_day = np.zeros(target_tair.shape[0])        
        
        #############################################################
        self.pca_tair = np.array(pca_tair, dtype=np.float64)
        self.valid_pca_mask = valid_pca_mask
        self.ngh_wgts = wgts
        self.ngh_dists = ngh_dists
        self.ngh_norms = ngh_norms
        self.max_dist = max_dist
        self.stn_id = stn_id
        self.stn_da = stn_da
        self.tair_var = tair_var
        self.tair_mask = tair_mask
        self.norms = norms
        self.nnghs_per_day = nnghs_per_day
    
    def __get_wgts(self,ngh_dists,max_dist):
      
        wgts = (1.0+np.cos(np.pi*(ngh_dists/max_dist)))/2.0
        #wgts = wgts/np.sum(wgts)
            
        return wgts
    
    def extend_ngh_radius(self, extend_by):
        '''
        Extends the search radius for neighbor stations. The minimum of the search radius
        is the previous max distance.
        @param extend_by: The amount (km) by which to extend the radius by
        '''
        
        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by

        pca_matrix2 = gwrpca_matrix(self.stn_id, self.stn_da, self.tair_var, self.norms, min_dist, max_dist, self.tair_mask)

        self.merge(pca_matrix2)
        self.max_dist = max_dist

    def merge(self, matrix2):
        '''
        Merges this pca_matrix with another pca_matrix
        @param matrix2: a mth_ngh_matrix
        '''    
    
        self.pca_tair = np.hstack((self.pca_tair, matrix2.pca_tair[:, 1:]))
        self.valid_pca_mask = np.hstack((self.valid_pca_mask, matrix2.valid_pca_mask[:, 1:]))
        self.ngh_dists = np.concatenate((self.ngh_dists, matrix2.ngh_dists[1:]))
        self.ngh_norms = np.concatenate((self.ngh_norms, matrix2.ngh_norms[1:]))
        
        if self.ngh_dists.size > 0:
            
            max_dist = np.max(np.array((self.max_dist,matrix2.max_dist)))
                              
            wgts = self.__get_wgts(self.ngh_dists,max_dist)
            wgt_sort = np.argsort(wgts)[::-1]
            wgts = wgts[wgt_sort]
            
            self.pca_tair = self.pca_tair[:, wgt_sort]
            self.valid_pca_mask = self.valid_pca_mask[:, wgt_sort]
            self.ngh_wgts = wgts
            self.ngh_dists = self.ngh_dists[wgt_sort]
            self.ngh_norms = self.ngh_norms[wgt_sort]
            
            self.nnghs_per_day = np.sum(self.valid_pca_mask[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.pca_tair.shape[1])

    def has_min_daily_nghs(self, nnghs):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        trim_valid_mask = self.valid_pca_mask[:, 0:1 + nnghs]
        nnghs_per_day = np.sum(trim_valid_mask[:, 1:], axis=1)
        
        return np.min(nnghs_per_day) >= MIN_DAILY_NGHBRS

    def infill(self, nnghs):
        '''
        Infills the target station's values for each day in the time series using the surrounding
        neighbors and a pca-based approach.
        
        @return fit_tair: an array of the infilled values for each day in the time series
        @return obs_tair: an array of the observed daily values of the target
        '''
        
        trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
        trim_ngh_norms = self.ngh_norms[0:1 + nnghs]
        trim_ngh_wgts = self.ngh_wgts[0:1 + nnghs]
        
        engh_dly_nghs = self.has_min_daily_nghs(nnghs)
        actual_nnghs = trim_pca_tair.shape[1] - 1
        
        while actual_nnghs < nnghs or not engh_dly_nghs:
        
            if actual_nnghs == nnghs and not engh_dly_nghs:
                
                nnghs += 1
            
            else:
                
                self.extend_ngh_radius(MAX_DISTANCE / 2.0)

            trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
            trim_ngh_norms = self.ngh_norms[0:1 + nnghs]
            trim_ngh_wgts = self.ngh_wgts[0:1 + nnghs]
            engh_dly_nghs = self.has_min_daily_nghs(nnghs)
            actual_nnghs = trim_pca_tair.shape[1] - 1
        
        ################################################
        #PPCA
        trim_pca_tair = trim_pca_tair * trim_ngh_wgts
        trim_ngh_norms = trim_ngh_norms * trim_ngh_wgts
        #ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(trim_pca_tair))
        ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(trim_pca_tair), robjects.FloatVector(trim_ngh_norms))
        fit_tair = np.array(ppca_rslt.rx('ppca_fit'))
        fit_tair.shape = (fit_tair.shape[1],)
        npcs = ppca_rslt.rx('npcs')[0][0]
        ################################################
        
        ################################################
        #Amelia
#        fit_tair = np.array(r.impute_amelia(robjects.Matrix(trim_pca_tair)))
#        npcs = 0
        
        ################################################
        
        obs_tair = trim_pca_tair[:, 0]
        
        return fit_tair, obs_tair, npcs, nnghs, self.max_dist

class nnrpca_matrix(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station to used for pca-based infilling.
    '''

    def __init__(self, stn_id, stn_da, tair_var, ncep_ds, n_ngh, tair_mask=None):
    
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
        
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0]
        
        if tair_mask is not None:
            target_tair[tair_mask] = np.nan
        
        self.target_stn = stn
        self.target_tair = target_tair
        self.target_tair.shape = (self.target_tair.size, 1)
        self.ncep_ds = ncep_ds

    def infill(self):
            
        pca_scores = self.ncep_ds.pca_scores_ngh_data(self.target_stn[LON], self.target_stn[LAT])
        impute_mat = np.hstack((self.target_tair, pca_scores))
        
        fit_tair = np.array(r.impute_amelia(robjects.Matrix(impute_mat)))
                
        return fit_tair, self.target_tair.ravel()

class impute_matrix(object):
    '''
    '''

    def __init__(self, stn_id, stn_da, stns_mask, tair_var, min_dist= -1, max_dist=MAX_DISTANCE, tair_mask=None):
        '''
        
        @param stn_id: the stn_id of the target
        @param stn_da: a station_data_ncdb object
        @param tair_var: the tair variable (tmin, tmax)
        @param min_dist: the min distance for which to search for neighbors (exclusive)
        @param max_dist: the max distance for which to search for neighbors (inclusive)
        @param tair_mask: a mask for which observations at the target should be set to nan (default: None)
        '''
    
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
        
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0]
        
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
                
        ngh_stns = all_stns[mask_dists]
        ngh_dists = dists[mask_dists]
        
        ngh_ids = ngh_stns[STN_ID]
        ngh_tair = stn_da.load_all_stn_obs_var(ngh_ids, tair_var, set_flagged_nan=True)[0]
        
        if len(ngh_tair.shape) == 1:
            ngh_tair.shape = (ngh_tair.size, 1) 
        
        dist_sort = np.argsort(ngh_dists)
        ngh_stns = ngh_stns[dist_sort]
        ngh_dists = ngh_dists[dist_sort]
        ngh_tair = ngh_tair[:, dist_sort]
        
        overlap_mask_tair = np.zeros(ngh_stns.size, dtype=np.bool)
        ioa = np.zeros(ngh_stns.size)
        
        for x in np.arange(ngh_stns.size):
            
            valid_ngh_mask = np.isfinite(ngh_tair[:, x])
            
            nlap = np.nonzero(valid_ngh_mask)[0].size
            
            overlap_mask = np.logical_and(valid_tair_mask, valid_ngh_mask)
            
            nlap_stn = np.nonzero(overlap_mask)[0].size
            
            if nlap >= nthres_all and nlap_stn >= nthres_target_por:
            #if nlap_stn >= nthres_target_por:
                
                ioa[x] = _calc_ioa(target_tair[overlap_mask], ngh_tair[:, x][overlap_mask])
                overlap_mask_tair[x] = True
        
        ioa = ioa[overlap_mask_tair]
        ngh_dists = ngh_dists[overlap_mask_tair]
        ngh_tair = ngh_tair[:, overlap_mask_tair]
        
        if ioa.size > 0:
            
            ioa_sort = np.argsort(ioa)[::-1]
            ioa = ioa[ioa_sort]
            ngh_dists = ngh_dists[ioa_sort]
            ngh_tair = ngh_tair[:, ioa_sort]
            
            target_tair.shape = (target_tair.size, 1)
            
            pca_tair = np.hstack((target_tair, ngh_tair))
            ngh_dists = np.concatenate((np.zeros(1), ngh_dists))
            ioa = np.concatenate((np.ones(1), ioa))
            
            valid_pca_mask = np.isfinite(pca_tair)
            
            nnghs_per_day = np.sum(valid_pca_mask , axis=1)
        
        else:
            
            target_tair.shape = (target_tair.size, 1)  
            pca_tair = target_tair
            
            valid_tair_mask.shape = (valid_tair_mask.size, 1) 
            valid_pca_mask = valid_tair_mask
            
            ioa = np.ones(1)
            ngh_dists = np.zeros(1)
            
            nnghs_per_day = np.zeros(target_tair.shape[0])        
        
        #############################################################
        self.pca_tair = np.array(pca_tair, dtype=np.float64)
        self.valid_pca_mask = valid_pca_mask
        self.ngh_ioa = ioa
        self.ngh_dists = ngh_dists
        self.max_dist = max_dist
        self.stn_id = stn_id
        self.stn_da = stn_da
        self.tair_var = tair_var
        self.tair_mask = tair_mask
        self.nnghs_per_day = nnghs_per_day
        self.stns_mask = stns_mask
    
    def extend_ngh_radius(self, extend_by):
        '''
        Extends the search radius for neighbor stations. The minimum of the search radius
        is the previous max distance.
        @param extend_by: The amount (km) by which to extend the radius by
        '''
        
        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by

        pca_matrix2 = impute_matrix(self.stn_id, self.stn_da,self.stns_mask, self.tair_var, min_dist, max_dist, self.tair_mask)

        self.merge(pca_matrix2)
        self.max_dist = max_dist

    def merge(self, matrix2):
        '''
        Merges this pca_matrix with another pca_matrix
        @param matrix2: a mth_ngh_matrix
        '''    
    
        self.pca_tair = np.hstack((self.pca_tair, matrix2.pca_tair[:, 1:]))
        self.valid_pca_mask = np.hstack((self.valid_pca_mask, matrix2.valid_pca_mask[:, 1:]))
        self.ngh_ioa = np.concatenate((self.ngh_ioa, matrix2.ngh_ioa[1:]))
        self.ngh_dists = np.concatenate((self.ngh_dists, matrix2.ngh_dists[1:]))
        
        if self.ngh_ioa.size > 0:
            
            ioa_sort = np.argsort(self.ngh_ioa[1:])[::-1]
            ioa_sort = np.concatenate([np.zeros(1, dtype=np.int), ioa_sort + 1])
            
            self.pca_tair = self.pca_tair[:, ioa_sort]
            self.valid_pca_mask = self.valid_pca_mask[:, ioa_sort]
            self.ngh_ioa = self.ngh_ioa[ioa_sort]
            self.ngh_dists = self.ngh_dists[ioa_sort]
            
            self.nnghs_per_day = np.sum(self.valid_pca_mask[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.pca_tair.shape[1])

    def has_min_daily_nghs(self, nnghs):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        trim_valid_mask = self.valid_pca_mask[:, 0:1 + nnghs]
        nnghs_per_day = np.sum(trim_valid_mask[:, 1:], axis=1)
        
        return np.min(nnghs_per_day) >= MIN_DAILY_NGHBRS

    def infill(self, nnghs):
        '''
        Infills the target station's values for each day in the time series using the surrounding
        neighbors and a pca-based approach.
        
        @return fit_tair: an array of the infilled values for each day in the time series
        @return obs_tair: an array of the observed daily values of the target
        '''
        
        trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
        
        engh_dly_nghs = self.has_min_daily_nghs(nnghs)
        actual_nnghs = trim_pca_tair.shape[1] - 1
        
        while actual_nnghs < nnghs or not engh_dly_nghs:
        
            if actual_nnghs == nnghs and not engh_dly_nghs:
                
                nnghs += 1
            
            else:
                
                self.extend_ngh_radius(MAX_DISTANCE / 2.0)

            trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
            engh_dly_nghs = self.has_min_daily_nghs(nnghs)
            actual_nnghs = trim_pca_tair.shape[1] - 1

        #fit_tair = np.array(r.impute_amelia(robjects.Matrix(trim_pca_tair)))      
        fit_tair = np.array(r.impute_norm(robjects.Matrix(trim_pca_tair)))
        #fit_tair = np.array(r.impute_mvnmle(robjects.Matrix(trim_pca_tair)))
        #fit_tair = np.array(r.impute_hmisc(robjects.Matrix(trim_pca_tair)))
         
        obs_tair = trim_pca_tair[:, 0]
        
        return fit_tair, obs_tair, nnghs, self.max_dist

class pca_matrix(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station to used for pca-based infilling.
    '''

    def __init__(self, stn_id, stn_da, tair_var, norms, min_dist= -1, max_dist=MAX_DISTANCE, tair_mask=None):
        '''
        
        @param stn_id: the stn_id of the target
        @param stn_da: a station_data_ncdb object
        @param tair_var: the tair variable (tmin, tmax)
        @param norms: an array of the estimated mean daily values at every station for the time series
        @param min_dist: the min distance for which to search for neighbors (exclusive)
        @param max_dist: the max distance for which to search for neighbors (inclusive)
        @param tair_mask: a mask for which observations at the target should be set to nan (default: None)
        '''
    
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
        
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0]
        target_norm = norms[stn_da.stn_ids == stn_id][0]
        
        if tair_mask is not None:
            target_tair[tair_mask] = np.nan
        
        #Number of observations threshold for entire period that is being infilled
        nthres_all = np.round(MIN_POR_OVERLAP * target_tair.size)
        
        #Number f observations threshold just for the target's period of record
        valid_tair_mask = np.isfinite(target_tair)
        ntair_valid = np.nonzero(valid_tair_mask)[0].size
        nthres_target_por = np.round(MIN_POR_OVERLAP * ntair_valid)    
        
        #Make sure to not include the target station itself as a neighbor station
        stns_mask = np.logical_and(stn_da.stns[STN_ID] != stn_id,norms != NODATA_NORMS)
        all_stns = stn_da.stns[stns_mask]
        
        dists = utlg.grt_circle_dist(stn[LON], stn[LAT], all_stns[LON], all_stns[LAT])
        mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        while np.nonzero(mask_dists)[0].size == 0:
            max_dist += MAX_DISTANCE / 2.0
            mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        ngh_stns = all_stns[mask_dists]
        ngh_dists = dists[mask_dists]
        
        ngh_ids = ngh_stns[STN_ID]
        ngh_norms = norms[np.in1d(stn_da.stn_ids, ngh_ids, assume_unique=True)]
        ngh_tair = stn_da.load_all_stn_obs_var(ngh_ids, tair_var, set_flagged_nan=True)[0]
        
        if len(ngh_tair.shape) == 1:
            ngh_tair.shape = (ngh_tair.size, 1) 
        
        dist_sort = np.argsort(ngh_dists)
        ngh_stns = ngh_stns[dist_sort]
        ngh_dists = ngh_dists[dist_sort]
        ngh_norms = ngh_norms[dist_sort]
        ngh_tair = ngh_tair[:, dist_sort]
        
        overlap_mask_tair = np.zeros(ngh_stns.size, dtype=np.bool)
        ioa = np.zeros(ngh_stns.size)
        
        for x in np.arange(ngh_stns.size):
            
            valid_ngh_mask = np.isfinite(ngh_tair[:, x])
            
            nlap = np.nonzero(valid_ngh_mask)[0].size
            
            overlap_mask = np.logical_and(valid_tair_mask, valid_ngh_mask)
            
            nlap_stn = np.nonzero(overlap_mask)[0].size
            
            if nlap >= nthres_all and nlap_stn >= nthres_target_por:
            #if nlap_stn >= nthres_target_por:
                
                ioa[x] = _calc_ioa(target_tair[overlap_mask], ngh_tair[:, x][overlap_mask])
                overlap_mask_tair[x] = True
        
        ioa = ioa[overlap_mask_tair]
        ngh_dists = ngh_dists[overlap_mask_tair]
        ngh_tair = ngh_tair[:, overlap_mask_tair]
        ngh_norms = ngh_norms[overlap_mask_tair]
        
        if ioa.size > 0:
            
            ioa_sort = np.argsort(ioa)[::-1]
            ioa = ioa[ioa_sort]
            ngh_dists = ngh_dists[ioa_sort]
            ngh_tair = ngh_tair[:, ioa_sort]
            ngh_norms = ngh_norms[ioa_sort]
            
            target_tair.shape = (target_tair.size, 1)
            
            pca_tair = np.hstack((target_tair, ngh_tair))
            ngh_dists = np.concatenate((np.zeros(1), ngh_dists))
            ioa = np.concatenate((np.ones(1), ioa))
            ngh_norms = np.concatenate((np.array([target_norm]), ngh_norms))
            
            valid_pca_mask = np.isfinite(pca_tair)
            
            nnghs_per_day = np.sum(valid_pca_mask , axis=1)
        
        else:
            
            target_tair.shape = (target_tair.size, 1)  
            pca_tair = target_tair
            
            valid_tair_mask.shape = (valid_tair_mask.size, 1) 
            valid_pca_mask = valid_tair_mask
            
            ioa = np.ones(1)
            ngh_dists = np.zeros(1)
            ngh_norms = np.array([target_norm])
            
            nnghs_per_day = np.zeros(target_tair.shape[0])        
        
        #############################################################
        self.pca_tair = np.array(pca_tair, dtype=np.float64)
        self.valid_pca_mask = valid_pca_mask
        self.ngh_ioa = ioa
        self.ngh_dists = ngh_dists
        self.ngh_norms = ngh_norms
        self.max_dist = max_dist
        self.stn_id = stn_id
        self.stn_da = stn_da
        self.tair_var = tair_var
        self.tair_mask = tair_mask
        self.norms = norms
        self.nnghs_per_day = nnghs_per_day
    
    def extend_ngh_radius(self, extend_by):
        '''
        Extends the search radius for neighbor stations. The minimum of the search radius
        is the previous max distance.
        @param extend_by: The amount (km) by which to extend the radius by
        '''
        
        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by

        pca_matrix2 = pca_matrix(self.stn_id, self.stn_da, self.tair_var, self.norms, min_dist, max_dist, self.tair_mask)

        self.merge(pca_matrix2)
        self.max_dist = max_dist

    def merge(self, matrix2):
        '''
        Merges this pca_matrix with another pca_matrix
        @param matrix2: a mth_ngh_matrix
        '''    
    
        self.pca_tair = np.hstack((self.pca_tair, matrix2.pca_tair[:, 1:]))
        self.valid_pca_mask = np.hstack((self.valid_pca_mask, matrix2.valid_pca_mask[:, 1:]))
        self.ngh_ioa = np.concatenate((self.ngh_ioa, matrix2.ngh_ioa[1:]))
        self.ngh_dists = np.concatenate((self.ngh_dists, matrix2.ngh_dists[1:]))
        self.ngh_norms = np.concatenate((self.ngh_norms, matrix2.ngh_norms[1:]))
        
        if self.ngh_ioa.size > 0:
            
            ioa_sort = np.argsort(self.ngh_ioa[1:])[::-1]
            ioa_sort = np.concatenate([np.zeros(1, dtype=np.int), ioa_sort + 1])
            
            self.pca_tair = self.pca_tair[:, ioa_sort]
            self.valid_pca_mask = self.valid_pca_mask[:, ioa_sort]
            self.ngh_ioa = self.ngh_ioa[ioa_sort]
            self.ngh_dists = self.ngh_dists[ioa_sort]
            self.ngh_norms = self.ngh_norms[ioa_sort]
            
            self.nnghs_per_day = np.sum(self.valid_pca_mask[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.pca_tair.shape[1])

    def has_min_daily_nghs(self, nnghs):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        trim_valid_mask = self.valid_pca_mask[:, 0:1 + nnghs]
        nnghs_per_day = np.sum(trim_valid_mask[:, 1:], axis=1)
        
        return np.min(nnghs_per_day) >= MIN_DAILY_NGHBRS

    def infill(self, nnghs):
        '''
        Infills the target station's values for each day in the time series using the surrounding
        neighbors and a pca-based approach.
        
        @return fit_tair: an array of the infilled values for each day in the time series
        @return obs_tair: an array of the observed daily values of the target
        '''
        
        trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
        trim_ngh_norms = self.ngh_norms[0:1 + nnghs]
        
        engh_dly_nghs = self.has_min_daily_nghs(nnghs)
        actual_nnghs = trim_pca_tair.shape[1] - 1
        
        while actual_nnghs < nnghs or not engh_dly_nghs:
        
            if actual_nnghs == nnghs and not engh_dly_nghs:
                
                nnghs += 1
            
            else:
                
                self.extend_ngh_radius(MAX_DISTANCE / 2.0)

            trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
            trim_ngh_norms = self.ngh_norms[0:1 + nnghs]
            engh_dly_nghs = self.has_min_daily_nghs(nnghs)
            actual_nnghs = trim_pca_tair.shape[1] - 1
            
        #ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(trim_pca_tair))
        ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(trim_pca_tair), robjects.FloatVector(trim_ngh_norms))
        fit_tair = np.array(ppca_rslt.rx('ppca_fit'))
        fit_tair.shape = (fit_tair.shape[1],)
        npcs = ppca_rslt.rx('npcs')[0][0]
        
        obs_tair = trim_pca_tair[:, 0]
        
        return fit_tair, obs_tair, npcs, nnghs, self.max_dist

class ImputeMatrixPCA(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station to used for pca-based imputation.
    '''

    def __init__(self, stn_id, stn_da, tair_var, nnr_ds,aclib, min_dist= -1, max_dist=MAX_DISTANCE, tair_mask=None,add_bestngh=True):
        '''
        
        @param stn_id: the stn_id of the target
        @param stn_da: a station_data_ncdb object
        @param tair_var: the tair variable (tmin, tmax)
        @param norms: an array of the estimated mean daily values at every station for the time series
        @param min_dist: the min distance for which to search for neighbors (exclusive)
        @param max_dist: the max distance for which to search for neighbors (inclusive)
        @param nnr_ds: a NNRds for loading reanalysis data from nearest grid cells
        @param tair_mask: a mask for which observations at the target should be set to nan (default: None)
        '''
        
        idx_target = np.nonzero(stn_da.stn_ids == stn_id)[0][0]
        
        stn = stn_da.stns[idx_target]
        
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0]
        target_norm = stn_da.get_stn_mean(tair_var,idx_target)
        target_std = stn_da.get_stn_std(tair_var,idx_target)
        
        if tair_mask is not None:
            target_tair[tair_mask] = np.nan
                    
        #Number of observations threshold for entire period that is being infilled
        nthres_all = np.round(MIN_POR_OVERLAP * target_tair.size)
        
        #Number f observations threshold just for the target's period of record
        valid_tair_mask = np.isfinite(target_tair)
        ntair_valid = np.nonzero(valid_tair_mask)[0].size
        nthres_target_por = np.round(MIN_POR_OVERLAP * ntair_valid)    
        
        #Make sure to not include the target station itself as a neighbor station
        stns_mask = np.logical_and(np.logical_and(stn_da.stns[STN_ID] != stn_id,
                                                  np.isfinite(stn_da.get_stn_mean(tair_var))),
                                   stn_da.stns[STN_ID] != 'GHCN_CA002503650')
        all_stns = stn_da.stns[stns_mask]
        
        dists = utlg.grt_circle_dist(stn[LON], stn[LAT], all_stns[LON], all_stns[LAT])
        mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        while np.nonzero(mask_dists)[0].size == 0:
            max_dist += MAX_DISTANCE / 2.0
            mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        ngh_stns = all_stns[mask_dists]
        ngh_dists = dists[mask_dists]
        
        ngh_ids = ngh_stns[STN_ID]
        nghid_mask = np.in1d(stn_da.stn_ids, ngh_ids, assume_unique=True)
        ngh_norms = stn_da.get_stn_mean(tair_var,nghid_mask)
        ngh_std = stn_da.get_stn_std(tair_var,nghid_mask)
        ngh_tair = stn_da.load_all_stn_obs_var(ngh_ids, tair_var, set_flagged_nan=True)[0]
        
        if len(ngh_tair.shape) == 1:
            ngh_tair.shape = (ngh_tair.size, 1) 
        
        dist_sort = np.argsort(ngh_dists)
        ngh_stns = ngh_stns[dist_sort]
        ngh_dists = ngh_dists[dist_sort]
        ngh_norms = ngh_norms[dist_sort]
        ngh_std = ngh_std[dist_sort]
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
                
                ioa[x] = _calc_ioa(target_tair[overlap_mask], ngh_tair[:, x][overlap_mask])
                overlap_mask_tair[x] = True
            
            elif nlap_stn >= nthres_target_por and add_bestngh:
                
                aioa = _calc_ioa(target_tair[overlap_mask], ngh_tair[:, x][overlap_mask])
                
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
        ngh_norms = ngh_norms[overlap_mask_tair]
        ngh_std = ngh_std[overlap_mask_tair]
        
        if ioa.size > 0:
            
            ioa_sort = np.argsort(ioa)[::-1]
            ioa = ioa[ioa_sort]
            ngh_dists = ngh_dists[ioa_sort]
            ngh_tair = ngh_tair[:, ioa_sort]
            ngh_norms = ngh_norms[ioa_sort]
            ngh_std = ngh_std[ioa_sort]
            
            target_tair.shape = (target_tair.size, 1)
            
            pca_tair = np.hstack((target_tair, ngh_tair))
            ngh_dists = np.concatenate((np.zeros(1), ngh_dists))
            ioa = np.concatenate((np.ones(1), ioa))
            ngh_norms = np.concatenate((np.array([target_norm]), ngh_norms))
            ngh_std = np.concatenate((np.array([target_std]), ngh_std))
            
            valid_pca_mask = np.isfinite(pca_tair)
            
            nnghs_per_day = np.sum(valid_pca_mask , axis=1)
        
        else:
            
            target_tair.shape = (target_tair.size, 1)  
            pca_tair = target_tair
            
            valid_tair_mask.shape = (valid_tair_mask.size, 1) 
            valid_pca_mask = valid_tair_mask
            
            ioa = np.ones(1)
            ngh_dists = np.zeros(1)
            ngh_norms = np.array([target_norm])
            ngh_std = np.array([target_std])
            
            nnghs_per_day = np.zeros(target_tair.shape[0])        
        
        #############################################################
        self.pca_tair = np.array(pca_tair, dtype=np.float64)
        self.valid_pca_mask = valid_pca_mask
        self.ngh_ioa = ioa
        self.ngh_dists = ngh_dists
        self.ngh_norms = ngh_norms
        self.ngh_std = ngh_std
        self.max_dist = max_dist
        self.stn_id = stn_id
        self.stn_da = stn_da
        self.tair_var = tair_var
        self.tair_mask = tair_mask
        self.nnghs_per_day = nnghs_per_day
        self.nnr_ds = nnr_ds
        self.stn = stn
        self.clib = aclib
    
    def extend_ngh_radius(self, extend_by):
        '''
        Extends the search radius for neighbor stations. The minimum of the search radius
        is the previous max distance.
        @param extend_by: The amount (km) by which to extend the radius by
        '''
        
        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by

        pca_matrix2 = ImputeMatrixPCA(self.stn_id, self.stn_da, self.tair_var,self.nnr_ds, self.clib, min_dist, max_dist, self.tair_mask,add_bestngh=False)

        self.merge(pca_matrix2)
        self.max_dist = max_dist

    def merge(self, matrix2):
        '''
        Merges this pca_matrix with another pca_matrix
        @param matrix2: a mth_ngh_matrix
        '''    
    
        self.pca_tair = np.hstack((self.pca_tair, matrix2.pca_tair[:, 1:]))
        self.valid_pca_mask = np.hstack((self.valid_pca_mask, matrix2.valid_pca_mask[:, 1:]))
        self.ngh_ioa = np.concatenate((self.ngh_ioa, matrix2.ngh_ioa[1:]))
        self.ngh_dists = np.concatenate((self.ngh_dists, matrix2.ngh_dists[1:]))
        self.ngh_norms = np.concatenate((self.ngh_norms, matrix2.ngh_norms[1:]))
        self.ngh_std = np.concatenate((self.ngh_std, matrix2.ngh_std[1:]))
        
        if self.ngh_ioa.size > 0:
            
            ioa_sort = np.argsort(self.ngh_ioa[1:])[::-1]
            ioa_sort = np.concatenate([np.zeros(1, dtype=np.int), ioa_sort + 1])
            
            self.pca_tair = self.pca_tair[:, ioa_sort]
            self.valid_pca_mask = self.valid_pca_mask[:, ioa_sort]
            self.ngh_ioa = self.ngh_ioa[ioa_sort]
            self.ngh_dists = self.ngh_dists[ioa_sort]
            self.ngh_norms = self.ngh_norms[ioa_sort]
            self.ngh_std = self.ngh_std[ioa_sort]
            
            self.nnghs_per_day = np.sum(self.valid_pca_mask[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.pca_tair.shape[1])

    def has_min_daily_nghs(self, nnghs, min_daily_nghs):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        trim_valid_mask = self.valid_pca_mask[:, 0:1 + nnghs]
        nnghs_per_day = np.sum(trim_valid_mask[:, 1:], axis=1)
        
        return np.min(nnghs_per_day) >= min_daily_nghs

    def impute(self, min_daily_nnghs=MIN_DAILY_NGHBRS,nnghs_nnr=NNGH_NNR,max_nnr_var=MAX_NNR_VAR,chk_perf=True,npcs=0,frac_obs_initnpcs=0.5,ppca_varyexplain=0.99,ppcaConThres=1e-5):
        '''
        Infills the target station's values for each day in the time series using the surrounding
        neighbors and a pca-based approach.
        
        @return fit_tair: an array of the infilled values for each day in the time series
        @return obs_tair: an array of the observed daily values of the target
        '''
        nnghs = min_daily_nnghs
        
        trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
        trim_ngh_norms = self.ngh_norms[0:1 + nnghs]
        trim_ngh_std = self.ngh_std[0:1 + nnghs]
        
        engh_dly_nghs = self.has_min_daily_nghs(nnghs,min_daily_nnghs)
        actual_nnghs = trim_pca_tair.shape[1] - 1
        
        while actual_nnghs < nnghs or not engh_dly_nghs:
        
            if actual_nnghs == nnghs and not engh_dly_nghs:
                
                nnghs += 1
            
            else:
                
                self.extend_ngh_radius(MAX_DISTANCE / 2.0)

            trim_pca_tair = self.pca_tair[:, 0:1 + nnghs]
            trim_ngh_norms = self.ngh_norms[0:1 + nnghs]
            trim_ngh_std = self.ngh_std[0:1 + nnghs]
            engh_dly_nghs = self.has_min_daily_nghs(nnghs,min_daily_nnghs)
            actual_nnghs = trim_pca_tair.shape[1] - 1
        
        #############################################################
        nnr_tair = self.nnr_ds.get_nngh_matrix(self.stn[LON],self.stn[LAT],self.tair_var,utc_offset=self.stn[UTC_OFFSET],nngh=nnghs_nnr)
        pc_loads, pc_scores, var_explain, error = self.clib.pca_basic(nnr_tair, True, True)
        cusum_var = np.cumsum(var_explain)
    
        i = np.nonzero(cusum_var >= max_nnr_var)[0][0]
        
        #print var_explain
        nnr_tair = pc_scores[:,0:i+1]
        #nnr_tair = pc_scores[:,0:min_daily_nnghs]
        
        
#        obs_ma = np.ma.masked_array(trim_pca_tair[:,0],mask=np.isnan(trim_pca_tair[:,0]))
#        nnr_ma = np.ma.masked_array(nnr_tair)
#
#        ccoefs = np.zeros(nnr_ma.shape[1])
#        for x in np.arange(nnr_ma.shape[1]):
#            ccoefs[x] =  np.abs(np.ma.corrcoef(obs_ma, nnr_ma[:,x],allow_masked=True)[0,1])
#        
#        ccoefs_mask = ccoefs >= .5
#        
#        nnr_tair = nnr_tair[:,ccoefs_mask]
#        print ccoefs[ccoefs_mask]
#        print trim_pca_tair.shape
        
        #preNCols = trim_pca_tair.shape[1]
        trim_pca_tair, trim_ngh_norms, trim_ngh_std = shrinkMatrix(trim_pca_tair, trim_ngh_norms, trim_ngh_std, min_daily_nnghs)
        #print "".join([self.stn_id,": removed ",str(preNCols- trim_pca_tair.shape[1])," cols."])
        
        if nnr_tair.size > 0:
        
            nnr_norms = np.mean(nnr_tair,dtype=np.float,axis=0)
            #nnr_norms = np.zeros(nnr_tair.shape[1])
            nnr_std = np.std(nnr_tair,dtype=np.float,axis=0,ddof=1)
            
            trim_pca_tair = np.hstack((trim_pca_tair,nnr_tair))
            trim_ngh_norms = np.concatenate((trim_ngh_norms,nnr_norms))
            trim_ngh_std = np.concatenate((trim_ngh_std,nnr_std))
        ############################################################

        ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(trim_pca_tair), 
                                        robjects.FloatVector(trim_ngh_norms),
                                        robjects.FloatVector(trim_ngh_std),
                                        frac_obs=frac_obs_initnpcs,
                                        max_r2cum=ppca_varyexplain,
                                        npcs=npcs,
                                        convThres=ppcaConThres)
        
        impute_tair = np.array(ppca_rslt.rx('ppca_fit'))
        impute_tair.shape = (impute_tair.shape[1],)
        npcsr = ppca_rslt.rx('npcs')[0][0]
        
        #############################################################
    
        obs_tair = trim_pca_tair[:, 0]
        
        if chk_perf:
            
            badImp,reasons = isBadImp(impute_tair, self)
            
            if badImp:
                
                print "".join(["WARNING|",self.stn_id," had bad impute for ",
                               self.tair_var,". Reasons: ",reasons,
                               ". Retrying..."])
                
                if MIN_NNR_VAR < max_nnr_var:
                    impute_tair2, obs_tair2, npcsr2, nnghs2 = self.impute(min_daily_nnghs, nnghs_nnr, MIN_NNR_VAR, False, npcs, frac_obs_initnpcs, ppca_varyexplain,ppcaConThres)[0:4]    
                    badImp,reasons = isBadImp(impute_tair2, self)
                
                if badImp:
                
                    newThres = [1e-6,1e-7]
                    
                    for aThres in newThres:
                    
                        impute_tair2, obs_tair2, npcsr2, nnghs2 = self.impute(min_daily_nnghs, nnghs_nnr, max_nnr_var, False, npcs, frac_obs_initnpcs, ppca_varyexplain,aThres)[0:4]
                                    
                        badImp,reasons = isBadImp(impute_tair2, self)
                        
                        if not badImp:
                            break
                
                if badImp:
                    
                    print "".join(["ERROR|",self.stn_id," had bad impute for ",
                                   self.tair_var," even after retries. Reasons: ",reasons])
                
                else:
                    
                    print "".join(["SUCCESS IMPUTE RETRY|",self.stn_id," fixed bad impute for ",self.tair_var])
                    
                    impute_tair, obs_tair, npcsr, nnghs = impute_tair2, obs_tair2, npcsr2, nnghs2
                       
        return impute_tair, obs_tair, npcsr, nnghs, self.max_dist

def isBadImp(impTair,impMatrix):
    
    rerun_imp = False
    reasons = ''
    
    obs_tair = impMatrix.pca_tair[:,0]
    chk_obs = obs_tair[impMatrix.valid_pca_mask[:,0]]
    chk_fit = impTair[impMatrix.valid_pca_mask[:,0]]
    mae = np.mean(np.abs(chk_fit-chk_obs))
    r_value = stats.linregress(chk_obs, chk_fit)[2]
    var_pct = r_value**2 #r-squared value; variance explained
    
    hasVarChgPt = r.hasVarChgPt(robjects.FloatVector(impTair))[0]
    
    #print "MAE|VAR "+str(mae)+" "+str(var_pct)
    
    #check for low impute performance
    if mae > 2.0 or var_pct < 0.7:
        
        rerun_imp = True
        reasons = "|".join([reasons,"low impute performance (%.2f,%.2f)"%(mae,var_pct)])
    
    #Check for extreme values
    if np.sum(impTair > 57.7) > 0 or np.sum(impTair < -89.4) > 0:
        
        rerun_imp = True
        reasons = "|".join([reasons,"impossible impute values"])
    
    #Check for variance change point
    if hasVarChgPt:
        
        rerun_imp = True
        reasons = "|".join([reasons,"variance change point"])
    
    return rerun_imp,reasons
    
    

def shrinkMatrix(aMatrix, nghNorms, nghStd, minNghs):
    
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
    
#    print np.min(nObs),np.max(nObs)
#    plt.boxplot(nObs)
#    plt.show()
    
    return aMatrix[:,keepCol],nghNorms[keepCol],nghStd[keepCol] 

def transform_bin(x,params=None):
    
    x = np.copy(x)
    x[x > 0] = 1
    return x,params

def btransform_square(x,params=None):
    
    x = np.copy(x)
    x[x < 0] = 0
    return np.square(x)

def transform_sqrt(x,params=None):
    
    return np.sqrt(x),None

def transform_blank(x,params=None):
    return x,params

def btransform_blank(x,params=None):
    return x

class pca_matrix_prcp(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station to be used for pca-based infilling of prcp occurrence and amount.
    '''

    def __init__(self, stn_id, stn_da, ds_norms, min_dist= -1, max_dist=MAX_DISTANCE,prcp_norm=None, prcp_mask=None):
        '''
        
        @param stn_id: the stn_id of the target
        @param stn_da: a station_data_ncdb object
        @param min_dist: the min distance for which to search for neighbors (exclusive)
        @param max_dist: the max distance for which to search for neighbors (inclusive)
        @param prcp_mask: a mask for which observations at the target should be set to nan (default: None)
        '''
        
        #Prcp normals dataset only has subset of station ids from original database
        norms_stnids = np.array(ds_norms.variables['stn_id'][:], dtype="<S16")
        prcp_stn_mask = np.in1d(stn_da.stn_ids, norms_stnids, assume_unique=True)
    
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
        
        target_prcp = stn_da.load_all_stn_obs_var(np.array([stn_id]), 'prcp')[0]
        
        if prcp_mask is not None:
            target_prcp[prcp_mask] = np.nan
            
        if prcp_norm is None:
            target_prcp_norm_ts = ds_norms.variables['prcp'][:, np.nonzero(norms_stnids == stn_id)[0][0]]
        
        else:
            target_prcp_norm_ts = prcp_norm
        
        #Number of observations threshold for entire period that is being infilled
        nthres_all = np.round(MIN_POR_OVERLAP * target_prcp.size)
        
        #Number of observations threshold just for the target's period of record
        valid_prcp_mask = np.isfinite(target_prcp)
        nprcp_valid = np.nonzero(valid_prcp_mask)[0].size
        nthres_target_por = np.round(MIN_POR_OVERLAP * nprcp_valid)    
        
        #Make sure to not include the target station itself as a neighbor station nor excluded prcp stations
        all_stns = stn_da.stns[np.logical_and(stn_da.stns[STN_ID] != stn_id,prcp_stn_mask)]
        
        dists = utlg.grt_circle_dist(stn[LON], stn[LAT], all_stns[LON], all_stns[LAT])
        mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        while np.nonzero(mask_dists)[0].size == 0:
            max_dist += MAX_DISTANCE / 2.0
            mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        ngh_stns = all_stns[mask_dists]
        ngh_dists = dists[mask_dists]
        
        ngh_ids = ngh_stns[STN_ID]
        ngh_prcp = stn_da.load_all_stn_obs_var(ngh_ids, 'prcp', set_flagged_nan=True)[0]
        
        ngh_prcp_norm_ts = ds_norms.variables['prcp'][:, np.nonzero(np.in1d(norms_stnids, ngh_ids, assume_unique=True))[0]]
        
        if len(ngh_prcp.shape) == 1:
            ngh_prcp.shape = (ngh_prcp.size, 1)
            
        if len(ngh_prcp_norm_ts.shape) == 1:
            ngh_prcp_norm_ts.shape = (ngh_prcp_norm_ts.size, 1)  
        
        dist_sort = np.argsort(ngh_dists)
        ngh_stns = ngh_stns[dist_sort]
        ngh_dists = ngh_dists[dist_sort]
        ngh_prcp = ngh_prcp[:, dist_sort]
        ngh_prcp_norm_ts = ngh_prcp_norm_ts[:, dist_sort]
        
        overlap_mask_prcp = np.zeros(ngh_stns.size, dtype=np.bool)
        hss = np.zeros(ngh_stns.size)
        
        for x in np.arange(ngh_stns.size):
            
            valid_ngh_mask = np.isfinite(ngh_prcp[:, x])
            
            nlap = np.nonzero(valid_ngh_mask)[0].size
            
            overlap_mask = np.logical_and(valid_prcp_mask, valid_ngh_mask)
            
            nlap_stn = np.nonzero(overlap_mask)[0].size
            
            if nlap >= nthres_all and nlap_stn >= nthres_target_por:
                
                hss[x] = calc_hss(target_prcp[overlap_mask].astype(np.bool), ngh_prcp[:, x][overlap_mask].astype(np.bool))
                overlap_mask_prcp[x] = True
        
        hss = hss[overlap_mask_prcp]
        ngh_dists = ngh_dists[overlap_mask_prcp]
        ngh_prcp = ngh_prcp[:, overlap_mask_prcp]
        ngh_prcp_norm_ts = ngh_prcp_norm_ts[:, overlap_mask_prcp]
        
        if hss.size > 0:
            
            hss_sort = np.argsort(hss)[::-1]
            hss = hss[hss_sort]
            ngh_dists = ngh_dists[hss_sort]
            ngh_prcp = ngh_prcp[:, hss_sort]
            ngh_prcp_norm_ts = ngh_prcp_norm_ts[:, hss_sort]
            
            target_prcp.shape = (target_prcp.size, 1)
            target_prcp_norm_ts.shape = (target_prcp_norm_ts.size,1)
            
            pca_prcp = np.hstack((target_prcp, ngh_prcp))
            ngh_dists = np.concatenate((np.zeros(1), ngh_dists))
            ngh_prcp_norm_ts = np.hstack((target_prcp_norm_ts,ngh_prcp_norm_ts))
            hss = np.concatenate((np.ones(1), hss))
            
            valid_pca_mask = np.isfinite(pca_prcp)
            
            nnghs_per_day = np.sum(valid_pca_mask , axis=1)
        
        else:
            
            target_prcp.shape = (target_prcp.size, 1)  
            pca_prcp = target_prcp
            
            target_prcp_norm_ts.shape = (target_prcp_norm_ts.size,1)
            ngh_prcp_norm_ts = target_prcp_norm_ts
            
            valid_prcp_mask.shape = (valid_prcp_mask.size, 1) 
            valid_pca_mask = valid_prcp_mask
            
            hss = np.ones(1)
            ngh_dists = np.zeros(1)
            
            nnghs_per_day = np.zeros(target_prcp.shape[0])        
        
        #############################################################
        self.pca_prcp = np.array(pca_prcp, dtype=np.float64)
        self.valid_pca_mask = valid_pca_mask
        self.ngh_hss = hss
        self.ngh_dists = ngh_dists
        self.max_dist = max_dist
        self.stn_id = stn_id
        self.stn_da = stn_da
        self.prcp_mask = prcp_mask
        self.nnghs_per_day = nnghs_per_day
        self.ds_norms = ds_norms
        self.prcp_norm = prcp_norm
        self.ngh_prcp_norm_ts = ngh_prcp_norm_ts
    
    def extend_ngh_radius(self, extend_by):
        '''
        Extends the search radius for neighbor stations. The minimum of the search radius
        is the previous max distance.
        @param extend_by: The amount (km) by which to extend the radius by
        '''
        
        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by
        
        pca_matrix2 = pca_matrix_prcp(self.stn_id, self.stn_da, self.ds_norms, min_dist, max_dist, 
                                            self.prcp_norm, self.prcp_mask)

        self.merge(pca_matrix2)
        self.max_dist = max_dist

    def merge(self, matrix2):
        '''
        Merges this pca_matrix with another pca_matrix
        @param matrix2: a mth_ngh_matrix
        '''    
    
        self.pca_prcp = np.hstack((self.pca_prcp, matrix2.pca_prcp[:, 1:]))
        self.valid_pca_mask = np.hstack((self.valid_pca_mask, matrix2.valid_pca_mask[:, 1:]))
        self.ngh_hss = np.concatenate((self.ngh_hss, matrix2.ngh_hss[1:]))
        self.ngh_dists = np.concatenate((self.ngh_dists, matrix2.ngh_dists[1:]))
        self.ngh_prcp_norm_ts = np.hstack((self.ngh_prcp_norm_ts, matrix2.ngh_prcp_norm_ts[:, 1:]))
        
        if self.ngh_hss.size > 0:
            
            hss_sort = np.argsort(self.ngh_hss[1:])[::-1]
            hss_sort = np.concatenate([np.zeros(1, dtype=np.int), hss_sort + 1])
            
            self.pca_prcp = self.pca_prcp[:, hss_sort]
            self.ngh_prcp_norm_ts = self.ngh_prcp_norm_ts[:, hss_sort]
            self.valid_pca_mask = self.valid_pca_mask[:, hss_sort]
            self.ngh_hss = self.ngh_hss[hss_sort]
            self.ngh_dists = self.ngh_dists[hss_sort]
            
            self.nnghs_per_day = np.sum(self.valid_pca_mask[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.pca_prcp.shape[1])

    def has_min_daily_nghs(self, nnghs, po_mask=None):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        trim_valid_mask = self.valid_pca_mask[:, 0:1 + nnghs]
        
        if po_mask is not None:
            trim_valid_mask = trim_valid_mask[po_mask,:]
        
        nnghs_per_day = np.sum(trim_valid_mask[:, 1:], axis=1)
        
        return np.min(nnghs_per_day) >= MIN_DAILY_NGHBRS

    def infill(self, nnghs,po_mask=None,mean_center=False,scale=False,trans_funcs=(transform_bin,btransform_blank)):
        '''
        Infills the target station's values for each day in the time series using the surrounding
        neighbors and a pca-based approach.
        
        @return fit_prcp: an array of the infilled values for each day in the time series
        @return obs_prcp: an array of the observed daily values of the target
        '''
        
        if po_mask is None:
            po_mask = np.ones(self.pca_prcp.shape[0], dtype=np.bool)
        
        
        trim_pca_prcp = self.pca_prcp[:, 0:1 + nnghs]
        trim_pca_prcp = trim_pca_prcp[po_mask,:]
            
        trim_pca_norms = self.ngh_prcp_norm_ts[:,0:1 + nnghs]
        trim_pca_norms = trim_pca_norms[po_mask,:]
        
        engh_dly_nghs = self.has_min_daily_nghs(nnghs)
        actual_nnghs = trim_pca_prcp.shape[1] - 1
        
        while actual_nnghs < nnghs or not engh_dly_nghs:
        
            if actual_nnghs == nnghs and not engh_dly_nghs:
                
                nnghs += 1
            
            else:
                
                self.extend_ngh_radius(MAX_DISTANCE / 2.0)

            trim_pca_prcp = self.pca_prcp[:, 0:1 + nnghs]
            trim_pca_prcp = trim_pca_prcp[po_mask,:]
            
            trim_pca_norms = self.ngh_prcp_norm_ts[:,0:1 + nnghs]
            trim_pca_norms = trim_pca_norms[po_mask,:]
            
            engh_dly_nghs = self.has_min_daily_nghs(nnghs)
            actual_nnghs = trim_pca_prcp.shape[1] - 1
                
        ttrim_pca_prcp,trans_params = trans_funcs[0](trim_pca_prcp.astype(np.float64))
        ttrim_pca_norms = trans_funcs[0](trim_pca_norms.astype(np.float64),trans_params)[0]
        
        norms_mean = np.mean(ttrim_pca_norms,axis=0,dtype=np.float64)
        
        if mean_center:
            pca_mc = robjects.FloatVector(norms_mean)
        else:
            pca_mc = robjects.BoolVector(np.array([False]))
        
        ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(ttrim_pca_prcp),pca_mc,robjects.BoolVector(np.array([scale])),
                                        robjects.FloatVector(FRAC_OBS_PRCP),robjects.FloatVector(MAX_R2CUM_PRCP))
        fit_prcp = np.array(ppca_rslt.rx('ppca_fit'))
        fit_prcp.shape = (fit_prcp.shape[1],)
        npcs = ppca_rslt.rx('npcs')[0][0]        
        fit_prcp = trans_funcs[1](fit_prcp,trans_params)
        
        obs_prcp = trim_pca_prcp[:, 0]
        
        return fit_prcp, obs_prcp, npcs, nnghs, self.max_dist
 
def build_yr_mth_masks(days):
    
    masks = []
    yrs = np.unique(days[YEAR])
    
    for yr in yrs:
        
        mths_yr = np.unique(days[MONTH][days[YEAR]==yr])
        
        for mth in mths_yr:
            
            masks.append(np.logical_and(days[YEAR]==yr,days[MONTH]==mth))
    
    return masks

def infill_prcp(stn_id, stn_da, ds_norms, days, mth_masks, mthbuf_masks, nnghs,prcp_trans_funcs=(transform_bin,btransform_blank), mean_c=False, scale=False, prcp_mask=None, prcp_norm=None):
    '''
    The main method for infilling missing daily prcp values    
    @param stn_id: the stn_id of the target
    @param stn_da: a station_data_ncdb object
    @param ds_norms: a netCDF4 dataset object for the daily prcp normals dataset produced by obs_infill_normal
    @param days: a days object from twx.utils.util_dates.get_days_metadata representing the time series to infill/expand
    @param mth_masks: a mask of months (from build_mth_masks) over the time series of interest
    @param mthbuf_masks: a mask of months with a buffer (from build_mth_masks) over the time series of interest
    @param nnghs: the number of neighbor stations to use for infilling
    @param prcp_trans_funcs: a tuple of a transform and backtransform function for prcp values
    @param mean_c: mean center pca matrix
    @param scale: scale pca matrix
    @param prcp_mask: a mask for which observations at the target should be set to nan (default: None)
    @param prcp_norm: a prcp norm time series that should be used for the target instead of the one from ds_norms (default: None)
    @return fnl_prcp: time series of infilled prcp values
    '''
    
    norms_stnids = np.array(ds_norms.variables['stn_id'][:], dtype="<S16")
    prcp_norm_validate = ds_norms.variables['prcp_mod'][:, np.nonzero(norms_stnids == stn_id)[0][0]]
    
    if prcp_norm is None:
        
        prcp_norm = ds_norms.variables['prcp'][:, np.nonzero(norms_stnids == stn_id)[0][0]]
                 
    a_pca_matrix = pca_matrix_prcp(stn_id, stn_da, ds_norms,prcp_norm=prcp_norm, prcp_mask=prcp_mask)

    fit_po, obs_prcp, npcs, actual_nnghs, max_dist = a_pca_matrix.infill(nnghs,mean_center=mean_c,scale=scale,trans_funcs=prcp_trans_funcs)
    
    fin_mask = np.isfinite(obs_prcp)
                
    fit_po_fnl = np.zeros(fit_po.size,dtype=np.bool)
    obs_po_w_mask = obs_prcp > 0
    for mth_mask in mth_masks:
        
        fin_mth_mask = np.logical_and(mth_mask, fin_mask)
        obs_po_mth = obs_po_w_mask[fin_mth_mask]
        fit_po_mth = fit_po[fin_mth_mask]
        
        max_hss = 0
        max_thres = 0
        for thres in PO_THRESHS:
            
            thres_mask = fit_po_mth >= thres
            po_mth = np.array(thres_mask, dtype=np.int)
            
            hss = calc_hss(obs_po_mth, po_mth)
            if hss > max_hss:
                max_hss = hss
                max_thres = thres
        #print max_hss,max_thres
        thres_mask = np.logical_and(mth_mask, fit_po >= max_thres)
        fit_po_fnl[thres_mask] = True 
    
    fit_po_validate = np.copy(fit_po_fnl)
    fit_po = fit_po_fnl
    fit_po[fin_mask] = obs_po_w_mask[fin_mask]
    
    fit_prcp = np.zeros(fit_po.size)
    fit_prcp_validate = np.zeros(fit_po.size)
    
    #Get prcp amounts method 1: use amounts from prcp norm
    #######################################################
    norm_on_po = prcp_norm[fit_po]
    norm_on_po_validate = prcp_norm_validate[fit_po_validate]
    
    norm_on_po[norm_on_po < 0.01] = 0.01
    norm_on_po_validate[norm_on_po_validate < 0.01] = 0.01
     
    fit_prcp[fit_po] = norm_on_po
    fit_prcp_validate[fit_po_validate] = norm_on_po_validate
    #######################################################
    
    nan_mask = np.logical_not(fin_mask)
    ##########################################################################
    #Scale in one shot
    ###########################################################################
    #Only scale the daily amounts that were predicted
    prcp_norm_mean = np.mean(prcp_norm,dtype=np.float64)
    scaler = ((prcp_norm_mean*prcp_norm.size) - np.sum(fit_prcp[fin_mask]))/np.sum(fit_prcp[nan_mask])
    fit_prcp[nan_mask] = scaler * fit_prcp[nan_mask]
    #for validation prcp, scale all values
    fit_prcp_validate = (np.sum(prcp_norm_validate) / np.sum(fit_prcp_validate)) * fit_prcp_validate
    ############################################################################
    ############################################################################
    
    #Scale by month
#        for mth in np.unique(days[MONTH]):
#            
#            mth_mask = days[MONTH] == mth
#            
#            if np.sum(fit_prcp[mth_mask]) > 0:
#            
#                fin_mth_mask = np.logical_and(mth_mask,fin_mask)
#                nan_mth_mask = np.logical_and(mth_mask,nan_mask)
#                
#                mth_mean = np.mean(prcp_norm[mth_mask],dtype=np.float64)
#                
#                scaler = ((mth_mean*prcp_norm[mth_mask].size) - np.sum(fit_prcp[fin_mth_mask]))/np.sum(fit_prcp[nan_mth_mask])
#                fit_prcp[nan_mth_mask] = scaler * fit_prcp[nan_mth_mask]
#                
#                #for validation prcp, scale all values
#                fit_prcp_validate[mth_mask] = (np.sum(prcp_norm_validate[mth_mask]) / np.sum(fit_prcp_validate[mth_mask])) * fit_prcp_validate[mth_mask]
#    ############################################################################
        
    rslts = prcp_infill_results(stn_id,days)
    rslts.prcp = fit_prcp
    rslts.prcp_fit = fit_prcp_validate
    rslts.fill_mask = nan_mask
    rslts.obs_mask = fin_mask
    rslts.npcs_po = npcs
    rslts.nnghs_po = actual_nnghs
    rslts.maxdist_po = max_dist
        
    return rslts

class prcp_infill_results(object):
    '''
    A class for holding results from infill_prcp
    '''

    def __init__(self, stn_id,days):
        
        self.stn_id = stn_id
        self.days = days
        
        self.prcp = None
        self.prcp_fit = None
        self.fill_mask = None
        self.obs_mask = None
        
        self.npcs_po = None
        self.nnghs_po = None
        self.maxdist_po = None
        
        self.hss = None
        self.perr_ttlamt = None
        self.perr_freq = None
        self.perr_intsy = None
        self.var_obs = None
        self.var_fit = None
        self.mth_cor = None
    
    def calc_obs_vs_fit_stats(self,yr_mth_masks=None):
        
        #Total amount cm
        self.perr_ttlamt = ((np.sum(self.prcp_fit[self.obs_mask]) - np.sum(self.prcp[self.obs_mask])) / np.sum(self.prcp[self.obs_mask])) * 100.
    
        #Intensity cm wd-1
        intsy_obs = np.sum(self.prcp[self.obs_mask]) / np.sum(np.logical_and(self.obs_mask,self.prcp > 0))
        intsy_fit = np.sum(self.prcp_fit[self.obs_mask]) / np.sum(np.logical_and(self.obs_mask,self.prcp_fit > 0))
        self.perr_intsy = ((intsy_fit - intsy_obs) / intsy_obs) * 100.
        
        #Frequency wd d-1
        freq_obs = np.sum(np.logical_and(self.obs_mask,self.prcp > 0)) / np.float(np.sum(self.obs_mask))
        freq_fit = np.sum(np.logical_and(self.obs_mask,self.prcp_fit > 0)) / np.float(np.sum(self.obs_mask))
        self.perr_freq = ((freq_fit - freq_obs) / freq_obs) * 100.
        
        #HSS
        self.hss = calc_hss(self.prcp[self.obs_mask] > 0, self.prcp_fit[self.obs_mask] > 0)
        
        #Variance
        mask_both_prcp = np.logical_and(np.logical_and(self.prcp_fit > 0,self.prcp > 0),self.obs_mask)        
        self.var_obs = np.var(self.prcp[mask_both_prcp], ddof=1)
        self.var_fit = np.var(self.prcp_fit[mask_both_prcp], ddof=1)
        
        #Monthly correlation
        if yr_mth_masks is None:
            yr_mth_masks = build_yr_mth_masks(self.days)
        
        mth_fit_prcp = []
        mth_obs_prcp = []
        
        for yrmth_mask in yr_mth_masks:
            
            yrmth_obs_mask = np.logical_and(yrmth_mask,self.obs_mask)
            
            if np.sum(yrmth_obs_mask) > 0:
                
                mth_fit_prcp.append(np.sum(self.prcp_fit[yrmth_obs_mask]))
                mth_obs_prcp.append(np.sum(self.prcp[yrmth_obs_mask]))
        
        mth_fit_prcp = np.array(mth_fit_prcp)
        mth_obs_prcp = np.array(mth_obs_prcp)
        
        self.mth_cor = ss.pearsonr(mth_fit_prcp,mth_obs_prcp)[0] 
        
    
    def print_obs_vs_fit(self):
        
        print " ".join([self.stn_id,"HSS: %.4f"%(self.hss,)])
        print " ".join([self.stn_id,"PCT ERROR AMT: %.4f"%(self.perr_ttlamt,)])
        print " ".join([self.stn_id,"PCT ERROR FREQ: %.4f"%(self.perr_freq,)])
        print " ".join([self.stn_id,"PCT ERROR INTSY: %.4f"%(self.perr_intsy,)])
        print " ".join([self.stn_id,"VARIANCE OBS|MODELED: %.4f|%.4f"%(self.var_obs,self.var_fit)])
        print " ".join([self.stn_id,"MONTHLY COR: %.4f"%(self.mth_cor,)])
            
def _calc_ioa(x, y):
    '''
    Calculate the index of agreement (Durre et al. 2010; Legates and McCabe 1999) between x and y
    '''
    
    y_mean = np.mean(y)
    d = np.sum(np.abs(x - y_mean) + np.abs(y - y_mean))
    
    if d == 0.0:
        print "|".join(["WARNING: _calc_ioa: x, y identical"])
        #The x and y series are exactly the same
        #Return a perfect ioa
        return 1.0
    
    ioa = 1.0 - (np.sum(np.abs(y - x)) / d)
    
#    if ioa == 0:
#        print "|".join(["WARNING: _calc_ioa: ioa == 0"])
#        #Means all ys are the same or only one observation.
#        #This could possibly happen with prcp in arid regions
#        #Add on an extra observation to the time series that has same difference as x[0] and y[0]
#        x_new = np.concatenate([x, np.array([x[0] + (x[0] * .1)])])
#        y_new = np.concatenate([y, np.array([y[0] + (x[0] * .1)])])
#        
#        y_mean = np.mean(y_new)
#        ioa = 1.0 - (np.sum(np.abs(y_new - x_new)) / np.sum(np.abs(x_new - y_mean) + np.abs(y_new - y_mean)))  
    
    return ioa

def calc_forecast_scores(obs, mod):
    '''
    Calculates various forecast scores of modeled prcp occurrence
    See http://www.wxonline.info/topics/verif2.html
    @param obs: array of observed occurrences (1s and 0s)
    @param mod: array of modeled occurrences (1s and 0s)
    @return: % correct, hit rate, false alarm ratio, threat score, bias, heidke skill score
    '''
    
    #model_obs
    true_true = mod[np.logical_and(obs == 1, mod == 1)].size
    false_false = mod[np.logical_and(obs == 0, mod == 0)].size
    true_false = mod[np.logical_and(obs == 0, mod == 1)].size
    false_true = mod[np.logical_and(obs == 1, mod == 0)].size
    
    a = float(true_true)
    b = float(true_false)
    c = float(false_true)
    d = float(false_false)
    n = float(obs.size)
     
    #http://www.wxonline.info/topics/verif2.html
    try:
        pc = (a + d) / n #% correct
    except ZeroDivisionError:
        pc = np.nan
    
    try:
        h = a / (a + c) #hit rate
    except ZeroDivisionError:
        h = np.nan
    
    try:
        far = b / (a + b) #false alarm ratio
    except ZeroDivisionError:
        far = np.nan
    
    try:
        ts = a / (a + b + c) #threat score
    except ZeroDivisionError:
        ts = np.nan
        
    try:    
        b = (a + b) / (a + c) #bias 
    except ZeroDivisionError:
        b = np.nan
        
    hss = calc_hss(obs, mod) #Heidke Skill Score
    
    return pc, h, far, ts, b, hss

def calc_hss(obs_po, mod_po):
    '''
    Calculates heidke skill score of modeled prcp occurrence
    See http://www.wxonline.info/topics/verif2.html
    @param obs: array of observed occurrences (1s and 0s)
    @param mod: array of modeled occurrences (1s and 0s)
    @return hss: heidke skill score
    '''
    
    #model_obs
    true_true = mod_po[np.logical_and(obs_po == 1, mod_po == 1)].size
    false_false = mod_po[np.logical_and(obs_po == 0, mod_po == 0)].size
    true_false = mod_po[np.logical_and(obs_po == 0, mod_po == 1)].size
    false_true = mod_po[np.logical_and(obs_po == 1, mod_po == 0)].size
    
    a = float(true_true)
    b = float(true_false)
    c = float(false_true)
    d = float(false_false)
    
    #special case handling
    if a == 0.0 and c == 0.0 and b != 0:
        #This means that were no observed days of rain so can't calc
        #appropriate hss. Set a = 1 to get a more appropriate hss
        a = 1.0
    
    if b == 0.0 and d == 0.0 and c != 0.0:
        #This means that there was observed rain every day so can't calc
        #appropriate hss. Set d = 1 to get a more appropriate hss
        d = 1.0    

    den = ((a + c) * (c + d)) + ((a + b) * (b + d))
    
    if den == 0.0:
        #This is a perfect forecast with all true_true or false_false
        return 1.0
    
    return (2.0 * ((a * d) - (b * c))) / den

def tmin_tmax_fixer(tmin, tmax, tail=15):

    invalid_days = np.nonzero(tmin >= tmax)[0]
    
    if invalid_days.size > 0:
        
        tmin = np.copy(tmin)
        tmax = np.copy(tmax)
        for x in invalid_days:
            
            tavg = (tmin[x] + tmax[x]) / 2.0
            start = x - tail
            end = x + tail + 1
            if start < 0: start = 0
            if end > tmin.size: end = tmin.size
            tmin_win = tmin[start:end]
            tmax_win = tmax[start:end]
            mask = tmin_win < tmax_win
            tmin_win = tmin_win[mask]
            tmax_win = tmax_win[mask]
            if tmin_win.size == 0:
                raise Exception('No valid tmin/tmax in window')
            tdir_half = np.mean(tmax_win - tmin_win, dtype=np.float64) / 2.0
            tmin[x] = tavg - tdir_half
            tmax[x] = tavg + tdir_half
        
    return tmin, tmax, invalid_days.size
 
if __name__ == '__main__':
    pass

 
    
