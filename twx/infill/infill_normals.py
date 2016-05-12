'''
Functions and classes for infilling the mean and variance of 
an incomplete station time series.

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
from twx.utils.perf_metrics import calc_ioa_d1

__all__ = ['infill_mean_variance']

import numpy as np
from twx.db import STN_ID, LON, LAT, UTC_OFFSET
import os
from twx.utils import pca_svd, grt_circle_dist

MAX_DISTANCE = 75  # in km
MIN_POR_OVERLAP = 2.0 / 3.0
MIN_DAILY_NGHBRS = 3
NNGH_NNR = 4
MAX_COLS_NORM_IMPUTE = 31

# rpy2
robjects = None
numpy2ri = None
r = None
ri = None
R_LOADED = False


class _InfillMatrix(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station to determine the mean and variance of the station's observations over 
    a set time period that the station has an incomplete record.
    '''

    def __init__(self, stn_id, stn_da, stns_mask, tair_var, nnr_ds, min_dist=-1, max_dist=MAX_DISTANCE, tair_mask=None, day_mask=None, add_bestngh=True):
        '''
        Parameters
        ----------
        stn_id : str
            The station id of the target station
        stn_da : twx.db.StationDataDb
            The station database from which all target and neighboring
            station observations should be loaded
        stns_mask : ndarray
            A boolean array mask specifying which stations in the database
            can be used as neighbors. Mask size must equal the number of
            stations in the database
        tair_var : str
            The temperature variable ('tmin' or 'tmax') of focus.
        nnr_ds : twx.db.NNRNghData
            A NNRNghData object for loading reanalysis data to help supplement
            the neighboring station data.
        min_dist : int, optional
            The minimum distance (exclusive) for which to search for neighboring stations.
            Pass -1 if there should be no minimum distance
        max_dist : int, optional
            The maximum distance (inclusive) for which to search for neighboring stations.
            Defaults to MAX_DISTANCE
        tair_mask : ndarray, optional
            A boolean mask specifying which observations at the target should
            artificially be set to nan. This can be used for cross-validation.
            Mask size must equal the time series length specified by the passed
            StationDataDb.
        day_mask : boolean, optional
            If true and tair_mask is not None, days with actual missing observations will
            be removed before station mean and variance estimation. Ignored if
            tair_mask is None.
        add_bestngh : boolean optional
            Add the best correlated neighbor to the data matrix even if the time
            series period-of-record of the neighbor is less than the
            MIN_POR_OVERLAP threshold for the entire period over which 
            the target station's mean and variance is being estimated
        '''

        # Get target station metadata
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]

        # Load target station observations
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0]
        target_tair = target_tair.astype(np.float64)

        if tair_mask is not None:
            target_tair[tair_mask] = np.nan
            
        if day_mask is None:
            day_mask = np.ones(target_tair.size,dtype=np.bool)

        day_idx = np.nonzero(day_mask)[0]
        
        target_tair = np.take(target_tair, day_idx)

        # Number of observations threshold for entire period that is being infilled
        nthres_all = np.round(MIN_POR_OVERLAP * target_tair.size)

        # Number of observations threshold just for the target's period of record
        valid_tair_mask = np.isfinite(target_tair)
        ntair_valid = np.nonzero(valid_tair_mask)[0].size
        nthres_target_por = np.round(MIN_POR_OVERLAP * ntair_valid)

        # Make sure to not include the target station itself as a neighbor station
        stns_mask = np.logical_and(stn_da.stns[STN_ID] != stn_id, stns_mask)
        all_stns = stn_da.stns[stns_mask]

        dists = grt_circle_dist(stn[LON], stn[LAT], all_stns[LON], all_stns[LAT])
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

        ngh_tair = np.take(ngh_tair, day_idx, axis=0)

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
                                
                ioa[x] = calc_ioa_d1(target_tair[overlap_mask], ngh_tair[:, x][overlap_mask])

                overlap_mask_tair[x] = True

            elif nlap_stn >= nthres_target_por and add_bestngh:

                aioa = calc_ioa_d1(target_tair[overlap_mask], ngh_tair[:, x][overlap_mask])

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
        self.day_idx = day_idx
        self.day_mask = day_mask

    def __extend_ngh_radius(self, extend_by):
        '''
        Extend the search radius for neighboring stations.
        The minimum of the search radius is the previous max distance.
        
        Parameters
        ----------
        extend_by: int 
            The amount (km) by which to extend the radius.
        '''

        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by

        imp_matrix2 = _InfillMatrix(self.stn_id, self.stn_da, self.stns_mask, self.tair_var,
                                    self.nnr_ds, min_dist, max_dist, self.tair_mask, day_mask=self.day_mask, add_bestngh=False)

        self.__merge(imp_matrix2)
        self.max_dist = imp_matrix2.max_dist

    def __merge(self, matrix2):
        '''
        Merge this _InfillMatrix with another _InfillMatrix
        
        Parameters
        ----------
        matrix2: _InfillMatrix
            The other _InfillMatrix with which to merge
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

    def __has_min_daily_nghs(self, nnghs, min_daily_nnghs):
        '''
        Check to see if there is a minimum number of 
        neighbor observations each day
        '''

        trim_valid_mask = self.valid_imp_mask[:, 0:1 + nnghs]
        nnghs_per_day = np.sum(trim_valid_mask[:, 1:], axis=1)

        return np.min(nnghs_per_day) >= min_daily_nnghs


    def infill(self, min_daily_nnghs=MIN_DAILY_NGHBRS, nnghs_nnr=NNGH_NNR, max_nnr_var=0.99):
        '''
        Infill/estimate the target station's mean and variance over the set time period.
        
        Parameters
        ----------
        min_daily_nnghs : int, optional
            The minimum neighbors required for each day.
        nnghs_nnr : int, optional
            The number of neighboring NCEP/NCAR Reanalysis grid cells
        max_nnr_var : float, optional
            The required variance explained by principal components of
            a S-Mode PCA of the reanalysis data.
        
        Returns
        ----------
        stn_mean : float
            The infilled/estimated mean of the target station's observations
        stn_variance : float
            The infilled/estimated variance of the target station's observations
        '''

        nnghs = min_daily_nnghs

        trim_imp_tair_mat = self.imp_tair_mat[:, 0:1 + nnghs]

        engh_dly_nghs = self.__has_min_daily_nghs(nnghs, min_daily_nnghs)
        actual_nnghs = trim_imp_tair_mat.shape[1] - 1

        while actual_nnghs < nnghs or not engh_dly_nghs:

            if actual_nnghs == nnghs and not engh_dly_nghs:

                nnghs += 1

            else:

                self.__extend_ngh_radius(MAX_DISTANCE / 2.0)

            trim_imp_tair_mat = self.imp_tair_mat[:, 0:1 + nnghs]
            engh_dly_nghs = self.__has_min_daily_nghs(nnghs, min_daily_nnghs)
            actual_nnghs = trim_imp_tair_mat.shape[1] - 1

        #############################################################

        nnr_tair = self.nnr_ds.get_nngh_matrix(self.stn[LON], self.stn[LAT], self.tair_var,
                                               utc_offset=self.stn[UTC_OFFSET], nngh=nnghs_nnr)
        nnr_tair = np.take(nnr_tair, self.day_idx, axis=0)
        
        pc_loads, pc_scores, var_explain = pca_svd(nnr_tair, True, True)
        cusum_var = np.cumsum(var_explain)

        i = np.nonzero(cusum_var >= max_nnr_var)[0][0]

        nnr_tair = pc_scores[:, 0:i + 1]

        trim_imp_tair_mat = _shrink_matrix(trim_imp_tair_mat, min_daily_nnghs)

        # Impute norm can only have MAX_COLS_NORM_IMPUTE columns total
        if trim_imp_tair_mat.shape[1] > MAX_COLS_NORM_IMPUTE:

            # Too many ngh stations, don't use NNR and trim to MAX_COLS_NORM_IMPUTE
            trim_imp_tair_mat = trim_imp_tair_mat[:, 0:MAX_COLS_NORM_IMPUTE]

            validMask = np.isfinite(trim_imp_tair_mat)
            obsPerday = np.sum(validMask, axis=1)

            if np.min(obsPerday) == 0:

                trim_imp_tair_mat[:, -1] = nnr_tair[:, 0]

                nNoObs = np.sum(obsPerday == 0)

                print "".join(["WARNING: ", self.stn_id, " matrix trimming caused no observations on ", str(nNoObs), " days."])

        else:

            trim_imp_tair_mat = np.hstack((trim_imp_tair_mat, nnr_tair))

            # Impute norm can only have MAX_COLS_NORM_IMPUTE columns total
            if trim_imp_tair_mat.shape[1] > MAX_COLS_NORM_IMPUTE:
                trim_imp_tair_mat = trim_imp_tair_mat[:, 0:MAX_COLS_NORM_IMPUTE]

        trim_imp_tair_mat = np.require(trim_imp_tair_mat, dtype=np.float64, requirements=['C', 'A', 'W', 'O'])
        
        stn_mean,stn_variance = np.array(r.infill_mu_sigma(robjects.Matrix(trim_imp_tair_mat)))

        return stn_mean, stn_variance

def _shrink_matrix(aMatrix, minNghs):
    '''
    After top minNghs stations, if a neighboring station time series
    does not add observations on days with < minNghs, remove
    it from the matrix.  
    '''

    validMask = np.isfinite(aMatrix[:, 1:minNghs + 1])
    nObs = np.sum(validMask, axis=1)
    maskBelowMin = nObs < minNghs

    keepCol = np.ones(aMatrix.shape[1], dtype=np.bool)

    for x in np.arange(minNghs + 1, aMatrix.shape[1]):

        aCol = aMatrix[:, x]
        aColValidMask = np.isfinite(aCol)

        if np.sum(np.logical_and(maskBelowMin, aColValidMask)) > 0:

            aColValidMask.shape = (aColValidMask.size, 1)
            validMask = np.hstack((validMask, aColValidMask))
            nObs = np.sum(validMask, axis=1)
            maskBelowMin = nObs < minNghs

        else:

            keepCol[x] = False

    return aMatrix[:, keepCol]


def _load_R():

    global R_LOADED

    if not R_LOADED:
        
        global robjects
        global numpy2ri
        global r
        global ri
        
        # https://github.com/ContinuumIO/anaconda-issues/issues/152
        import readline
        
        import rpy2
        import rpy2.robjects
        robjects = rpy2.robjects
        r = robjects.r
        import rpy2.rinterface
        ri = rpy2.rinterface
        
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()

        path_root = os.path.dirname(__file__)
        fpath_rscript = os.path.join(path_root, 'rpy', 'norm_infill.R')
        r.source(fpath_rscript)
        R_LOADED = True

def infill_mean_variance(stn_id, stn_da, stn_mask, tair_var, nnr_ds, tair_mask=None, day_masks=None, nnghs=MIN_DAILY_NGHBRS, nnghs_nnr=NNGH_NNR):
    '''
    Infill/estimate a target station's temperature mean and variance over a set time period that the station
    has an incomplete record. Uses neighboring station and reanalysis data to performing the
    infilling/estimation. 
    
    Parameters
    ----------
    stn_id : str
        The station id of the target station
    stn_da : twx.db.StationDataDb
        The station database from which all target and neighboring
        station observations should be loaded
    stn_mask : ndarray
        A boolean array mask specifying which stations in the database
        can be used as neighbors. Mask size must equal the number of
        stations in the database
    tair_var : str
        The temperature variable ('tmin' or 'tmax') of focus.
    nnr_ds : twx.db.NNRNghData
        A NNRNghData object for loading reanalysis data to help supplement
        the neighboring station data.
    tair_mask : ndarray, optional
        A boolean mask specifying which observations at the target should
        artificially be set to nan. This can be used for cross-validation.
        Mask size must equal the time series length specified by the passed
        StationDataDb.
    day_mask : boolean, optional
        If true and tair_mask is not None, days with actual missing observations will
        be removed before station mean and variance estimation. Ignored if
        tair_mask is None.
    nnghs : int, optional
        The minimum neighboring observations required for each day.
    nnghs_nnr : int, optional
        The number of neighboring NCEP/NCAR Reanalysis grid cells
        
    Returns
    ----------
    stn_mean : float
        The infilled/estimated mean of the target station's observations
    stn_variance : float
        The infilled/estimated variance of the target station's observations
    '''

    _load_R()
    
    if day_masks == None:
        
        a_matrix = _InfillMatrix(stn_id, stn_da, stn_mask, tair_var, nnr_ds, tair_mask=tair_mask, day_mask=None)
        stn_mean, stn_variance = a_matrix.infill(nnghs, nnghs_nnr)
        
    else:
        
        n_masks = len(day_masks)
        
        stn_mean = np.empty(n_masks)
        stn_variance = np.empty(n_masks)
        
        for x in np.arange(n_masks):
            
            a_matrix = _InfillMatrix(stn_id, stn_da, stn_mask, tair_var, nnr_ds, tair_mask=tair_mask, day_mask=day_masks[x])
            a_mean, a_variance = a_matrix.infill(nnghs, nnghs_nnr)
            stn_mean[x] = a_mean
            stn_variance[x] = a_variance
            
    return stn_mean, stn_variance
