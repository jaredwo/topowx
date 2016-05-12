'''
Classes and functions for performing cross validation
of the infilling procedures in infill_daily.py.

Copyright 2014, Jared Oyler.

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
from twx.utils.util_dates import MONTH
from twx.db.station_data import get_mean_varname, get_variance_varname
from twx.infill.infill_daily import infill_daily_obs

__all__ = ['XvalInfill', 'XvalInfillParams', 'load_default_xval_stnids']

import os
import numpy as np
from twx.infill import infill_mean_variance, InfillMatrixPPCA

class XvalInfill(object):
    '''
    A class for performing cross validations of
    daily Tmin/Tmax infill models.
    '''

    def __init__(self, stnda, var_tair, infill_params, xval_stnids=None, ntrain_yrs=5):
        '''
        Parameters
        ----------
        stnda : twx.db.StationDataDb
            The station database from which all target and neighboring
            station observations should be loaded.
        var_tair : str
            The temperature variable ('tmin' or 'tmax') of focus.
        infill_params : XvalInfillParams
            Parameters for the infill model.
        xval_stnids : ndarray, optional
            A list of station ids to be used for cross validation.
            If None, default set of longer term stations is used
        ntrain_yrs : int, optional
            The number of years of data to be used for training
            the infill model. Currently, the last ntrain_yrs in a
            station's period-of-record will be used for training and
            all previous observations will be artificially set to
            missing and used for validation
        '''

        if xval_stnids is None:
            xval_stnids = load_default_xval_stnids(stnda.stn_ids)

        # Load observations for each station
        obs = stnda.load_all_stn_obs_var(xval_stnids, var_tair)[0]

        if len(obs.shape) == 1:
            obs.shape = (obs.shape[0], 1)

        days = stnda.days

        # The number of observations that should not be set to nan
        # and are used to build the infill model
        nmask = int(np.round(ntrain_yrs * 365.25))

        # Build masks of the data values that should be set to nan for each station
        xval_masks = []

        idxs = np.arange(days.size)

        for x in np.arange(xval_stnids.size):

            fin_obs = np.isfinite(obs[:, x])

            last_idxs = np.nonzero(fin_obs)[0][-nmask:]
            xval_mask_obs = np.logical_and(np.logical_not(np.in1d(idxs, last_idxs, assume_unique=True)), fin_obs)
            xval_masks.append(xval_mask_obs)
        
        self.mths = np.arange(1,13)
        self.mth_masks = [stnda.days[MONTH]==mth for mth in self.mths]
        self.vnames_mean = [get_mean_varname(var_tair,mth) for mth in self.mths]
        self.vnames_vari = [get_variance_varname(var_tair,mth) for mth in self.mths]
        
        # Neighbor station mask
        self.ngh_stn_mask = np.isfinite(stnda.stns[get_mean_varname(var_tair,1)])
        self.stn_ids = xval_stnids
        self.stn_obs = obs
        self.stn_xval_masks = xval_masks
        self.stnda = stnda
        self.var_tair = var_tair
        self.infill_params = infill_params

    def run_xval(self, stn_id):
        '''
        Run a cross validation for a specific station.
        
        Parameters
        ----------
        stn_id : str
            The station id for which to run a 
            cross validation.
        
        Returns
        ----------
        obs_tair : ndarray
            The original observations for the station.
            Days with original missing observations and 
            observations used for training are set to nan.
        infill_tair : ndarray
            The infilled observations for the station
            Days with original missing observations and 
            observations used for training are set to nan.      
        '''

        i = np.nonzero(self.stn_ids == stn_id)[0][0]
        x = self.stnda.stn_idxs[stn_id]
        
        mean_orig = np.array([self.stnda.stns[x][a_vname] for a_vname in self.vnames_mean])
        vari_orig = np.array([self.stnda.stns[x][a_vname] for a_vname in self.vnames_vari])
        
        try:

            tair_mask = self.stn_xval_masks[i]
            obs_tair = self.stn_obs[:, i].astype(np.float)

            stn_means, stn_varis = infill_mean_variance(stn_id, self.stnda, self.ngh_stn_mask, self.var_tair,
                                                      self.infill_params.nnr_ds, tair_mask, self.mth_masks,
                                                      self.infill_params.min_daily_nnghs, self.infill_params.nnghs_nnr)
            
            for vname_mean,vname_vari,stn_mean,stn_vari in zip(self.vnames_mean,self.vnames_vari,stn_means,stn_varis):
            
                self.stnda.stns[vname_mean][x] = stn_mean
                self.stnda.stns[vname_vari][x] = stn_vari
            
            
            infill_tair = infill_daily_obs( stn_id, self.stnda, self.var_tair, self.infill_params.nnr_ds, self.vnames_mean, self.vnames_vari, tair_mask,
                                            self.mth_masks, True, self.infill_params.min_daily_nnghs, self.infill_params.nnghs_nnr,
                                            self.infill_params.max_nnr_var, self.infill_params.chk_perf, self.infill_params.npcs,
                                            self.infill_params.frac_obs_initnpcs, self.infill_params.ppca_varyexplain,
                                            verbose=self.infill_params.verbose)[-1]
            
            # Set days with original observations that were missing or
            # where used for training to nan.
            infill_tair[~tair_mask] = np.nan
            obs_tair[~tair_mask] = np.nan

        finally:

            # Reset mean and var            
            for vname_mean,vname_vari,stn_mean,stn_vari in zip(self.vnames_mean,self.vnames_vari,mean_orig,vari_orig):
            
                self.stnda.stns[vname_mean][x] = stn_mean
                self.stnda.stns[vname_vari][x] = stn_vari

        return obs_tair, infill_tair

class XvalInfillParams(object):
    '''
    A class for holding parameters used by twx.infill.infill_mean_variance
    and twx.infill.InfillMatrixPPCA.
    '''

    def __init__(self, nnr_ds, min_daily_nnghs, nnghs_nnr,
                 max_nnr_var, chk_perf, npcs, frac_obs_initnpcs, ppca_varyexplain, verbose):
        '''
        Parameters
        ----------
        nnr_ds : twx.db.NNRNghData
            A NNRNghData object for loading reanalysis data to help supplement
            the neighboring station data.
        min_daily_nnghs : int, optional
            The minimum neighbors required for each day.
        nnghs_nnr : int, optional
            The number of neighboring NCEP/NCAR Reanalysis grid cells
        max_nnr_var : float, optional
            The required variance explained by principal components of
            a S-Mode PCA of the reanalysis data.
        chk_perf : boolean, optional
            If true, check performance of infilled output and if there
            are any bad infilled values. If there are bad infilled values,
            PPCA will be rerun with different configurations to try to eliminate
            the bad values.
        npcs : int, optional
            Use a specific set number of PCs. If npcs = 0, the number of PCs is determined
            dynamically based on ppca_varyexplain.
        frac_obs_initnpcs : float, optional
            The fraction of the total number of columns that should be used as the
            initial number of PCs. Example: if frac_obs is 0.5 and the number of columns
            is 10, the initial number of PCs will be 5.
        ppca_varyexplain : float, optional
            The required variance to be explained by the PCs. Example: if 0.99, the first
            n PCs that account for 99% of the variance in pca_matrix will be used
        verbose : boolean, optional
            If true, output PPCA algorithm progress.
        '''

        self.nnr_ds = nnr_ds
        self.min_daily_nnghs = min_daily_nnghs
        self.nnghs_nnr = nnghs_nnr
        self.max_nnr_var = max_nnr_var
        self.chk_perf = chk_perf
        self.npcs = npcs
        self.frac_obs_initnpcs = frac_obs_initnpcs
        self.ppca_varyexplain = ppca_varyexplain
        self.verbose = verbose

def load_default_xval_stnids(stnids_in_db=None):
    '''
    Load a default set of longer term stations for
    cross validation. The set includes GHCN-D stations
    that are part of USHCN and at least 95% complete
    from 1948-2012 and RAWS and SNOTEL stations that
    have at least 20 years of data.
    
    Parameters
    ----------
    stnids_in_db : ndarray, optional
        An array of station ids in the current station
        database. Any station ids that are part of the default 
        cross validation and not in the current station database
        will be removed.
    
    Returns
    ----------
    stnids : ndarray
        An array of station ids for cross validation.
    '''

    path_root = os.path.dirname(__file__)

    stnids_hcn = np.loadtxt(os.path.join(path_root, 'data', 'xval_stnids_hcn.txt'), dtype=np.str)
    stnids_raws_sntl = np.loadtxt(os.path.join(path_root, 'data', 'xval_stnids_raws_snotel.txt'), dtype=np.str)
    stnids = np.unique(np.concatenate((stnids_hcn, stnids_raws_sntl)))

    if stnids_in_db is not None:
        mask_not_in_db = ~np.in1d(stnids, stnids_in_db, True)
        if np.sum(mask_not_in_db) > 0:
            print "Default xval station ids not in current database: " + str(stnids[mask_not_in_db])
            stnids = stnids[~mask_not_in_db]


    return stnids
