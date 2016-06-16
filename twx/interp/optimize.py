'''
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

__all__ = ['create_climdiv_optim_nstns_db', 'XvalTairNorm',
           'set_optim_nstns_tair_norm', 'set_optim_nstns_tair_anom',
           'build_nstn_bandwidths', 'StationKrigParams',
           'XvalTairAnom', 'XvalTairOverall','XvalOutlier']

import numpy as np
from twx.db import StationSerialDataDb, STN_ID, get_norm_varname, \
get_optim_varname, get_optim_anom_varname, BAD, dbDataset, CLIMDIV, \
LAT, LON, get_lst_varname
from twx.interp import StationSelect, KrigTairAll, BuildKrigParams, \
GwrTairAnom, KrigTair, InterpTair
from netCDF4 import Dataset
import netCDF4
from twx.utils import StatusCheck
import scipy.stats as stats
import os
import pandas as pd
import statsmodels.formula.api as sm

def create_climdiv_optim_nstns_db(path_out, tair_var, stn_ids, nstns_rng, climdiv):
    '''
    Create a netCDF file for storing cross validation mean absolute error (MAE)
    for monthly normals interpolation for a set of stations and neighbor station 
    bandwidths within a single climate division. The MAE netCDF variable will be 
    of shape (12,N,P) where 12 is the number of months, N is number of bandwidths
    specified by nstns_rng and P is the number stations.  
    
    Parameters
    ----------
    path_out : str
        The directory path for file output. The output file will be named:
        optim_nstns_[tair_var]_climdiv[climdiv_id].nc
    tair_var : str
        The temperature variable ('tmin' or 'tmax')
    stn_ids : ndarray
        The station ids used for cross validation
    climdiv : int
        The U.S. climate division id
    
    '''

    fpath = os.path.join(path_out, "optim_nstns_%s_climdiv%d.nc" % (tair_var, climdiv))

    ds = dbDataset(fpath, 'w')
    ds.db_create_global_attributes("Cross Validation MAE for Different N Neighboring Stations: " + tair_var)

    ds.createDimension('min_nghs', nstns_rng.size)
    ds.db_create_stnid_dimvar(stn_ids)

    nghs = ds.createVariable('min_nghs', 'i4', ('min_nghs',), fill_value=False)
    nghs.long_name = "min_nghs"
    nghs.standard_name = "min_nghs"
    nghs[:] = nstns_rng

    ds.createDimension('mth', 12)
    mthVar = ds.createVariable('mth', 'i4', ('mth',), fill_value=False)
    mthVar[:] = np.arange(1, 13)

    ds.db_create_mae_var(('mth', 'min_nghs', 'stn_id'))

    ds.sync()

    return ds

class XvalOutlier(object):
    '''
    Class for running a leave-one-out cross validation of simple
    geographically weighted regression models of station monthly and annual normals 
    to determine if a station is an outlier and has possible erroneous values 
    based on unrealistic model error.
    '''
    
    def __init__(self, stn_da):
        '''        
        Parameters
        ----------
        stnda : twx.db.StationSerialDataDb
            A StationSerialDataDb object pointing to the
            database from which observations will be loaded.
        '''
        
        self.stn_da = stn_da
        mask_stns = np.isnan(self.stn_da.stns[BAD])
        self.stn_slct = StationSelect(self.stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)

        self.vnames_norm = [get_norm_varname(mth) for mth in np.arange(1, 13)]
        self.vnames_lst = [get_lst_varname(mth) for mth in np.arange(1, 13)]
        
        self.df_stns = pd.DataFrame(self.stn_da.stns)
        self.df_stns.index = self.df_stns[STN_ID]
        
        # Calculate annual means for monthly LST and Tair normals
        self.df_stns['lst'] = self.df_stns[self.vnames_lst].mean(axis=1)
        self.df_stns['norm'] = self.df_stns[self.vnames_norm].mean(axis=1)
        
        
    def run_xval_stn(self, stn_id, bw_nngh=100):
        '''
        Run a single leave-one-out cross validation of a geographically
        weighted regression model of a station's monthly and annual normals
        (norm~lst+elev+lon+lat).
        
        Parameters
        ----------
        stn_id : str
            The stn_id for which to run the cross validation
        bw_nngh : int, optional
            The number of neighbors to use for the
            geographically weighted regression. Default: 100.
        
        Returns
        ----------
        err : float
            The difference between predicted and observed
            (predicted minus observed)
        '''
        
        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]
        df_xval_stn = self.df_stns.loc[stn_id, :]
        self.stn_slct.set_ngh_stns(xval_stn[LAT], xval_stn[LON], bw_nngh,
                                   load_obs=False, stns_rm=stn_id)
        df_nghs = self.df_stns.loc[self.stn_slct.ngh_stns[STN_ID], :]
        
        errs = np.empty(13)
        
        # Errors for monthly normals
        for mth in np.arange(1,13):
            
            ls_form = 'norm%.2d~lst%.2d+elevation+longitude+latitude'%(mth, mth)
            ls_fit = sm.wls(ls_form, data=df_nghs, weights=self.stn_slct.ngh_wgt).fit()
            err = ls_fit.predict(df_xval_stn)[0] - df_xval_stn['norm%.2d'%mth]
            errs[mth-1] = err
        
        # Error for annual normal
        ls_form = 'norm~lst+elevation+longitude+latitude'
        ls_fit = sm.wls(ls_form, data=df_nghs, weights=self.stn_slct.ngh_wgt).fit()
        err = ls_fit.predict(df_xval_stn)[0] - df_xval_stn['norm']
        errs[-1] = err
        
        return errs
    
    def find_xval_outliers(self, stn_ids=None, bw_nngh=100, zscore_threshold=6):
        '''
        Runs a leave-one-out cross validation of a geographically
        weighted regression model of station monthly and annual normals
        (norm~lst+elev+lon+lat) and returns those stations whose error is
        a specified # of standard deviations above/below the mean
        
        Parameters
        ----------
        stn_ids : list_like, optional
            The station ids for which to run the cross validation.
            If None, the cross validation will be run for all stations
            in the database
        bw_nngh : int, optional
            The number of neighbors to use for the
            geographically weighted regression. Default: 100.
        zscore_threshold : float, optional
            The zcore threshold by which a station's error should be
            considered an outlier.
        
        Returns
        ----------
        out_stnids : ndarray
            The outlier stations
        out_errs : ndarray
            The model error associated with each outlier
        '''

        if stn_ids is None:
            stn_ids = self.stn_da.stn_ids
        
        schk = StatusCheck(stn_ids.size, check_cnt=250)
        
        xval_errs = np.zeros((13,stn_ids.size))
        
        for i, a_id in enumerate(stn_ids):
            
            xval_errs[:,i] = self.run_xval_stn(a_id, bw_nngh)
            schk.increment()
            
        xval_errs = pd.DataFrame(xval_errs)
        xval_errs.columns = stn_ids
        zscores = (xval_errs.subtract(xval_errs.mean(axis=1), axis=0).
                   divide(xval_errs.std(axis=1), axis=0).abs())
        out_stnids = zscores.columns[(zscores > zscore_threshold).any(axis=0)].values
                
        return out_stnids
        
        
class XvalTairNorm(object):
    '''
    Class for running a cross validation to optimize the local number
    of neighboring stations to use for moving window regression kriging
    of monthly temperature normals
    '''

    def __init__(self, path_db, tair_var):
        '''
        Parameters
        ----------
        path_db : str
            File path to a serially complete netCDF
            station database containing the stations and
            temperature variable for interpolation.
        tair_var : str
            The temperature variable for interpolation ('tmin' or 'tmax')
        '''

        stn_da = StationSerialDataDb(path_db, tair_var)
        mask_stns = np.isnan(stn_da.stns[BAD])
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)

        self.krig = KrigTairAll(stn_slct)
        self.stn_da = stn_da

    def run_xval(self, stn_id, abw_nngh):
        '''
        Run leave-one-out cross validations for a specific station id
        and a set of different neighbor bandwidths.
        
        Parameters
        ----------
        stn_id : str
            Station id for cross validation
        abw_nngh : ndarray
            A 1-D array of different neighbor station bandwidths
            for which to run cross validation.
        
        Returns
        ----------
        err : ndarray
            A 12 * N array of cross validations error (modeled - observed)
            where N is the number of bandwidths specified by abw_nngh
            
        '''
        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]

        err = np.zeros((12, abw_nngh.size))
        xvalNorms = np.array([xval_stn[get_norm_varname(mth)] for mth in np.arange(1, 13)])

        for bw_nngh, x in zip(abw_nngh, np.arange(abw_nngh.size)):

            interp_norms = self.krig.krigall(xval_stn, bw_nngh, stns_rm=xval_stn[STN_ID])
            err[:, x] = interp_norms - xvalNorms

        return err

def set_optim_nstns_tair_norm(stnda, path_xval_ds):
    '''
    Set the local optimal number of stations to be used for monthly
    normal interpolation for each U.S. climate division based on 
    cross-validation mean absolute error.
    
    Parameters
    ----------
    stnda : twx.db.StationSerialDataDb
        A StationSerialDataDb object pointing to the
        database for which the local optimal number of
        neighbors should be set. 
    path_xval_ds : str
        Path where netCDF cross-validation MAE files from
        create_climdiv_optim_nstns_db are located
    '''

    climdiv_stns = stnda.stns[CLIMDIV]

    vars_optim = {}
    for mth in np.arange(1, 13):

        varname_optim = get_optim_varname(mth)
        long_name = "Optimal number of neighbors to use for monthly normal interpolation for month %d" % mth
        var_optim = stnda.add_stn_variable(varname_optim, long_name, "", 'f8',
                                          fill_value=netCDF4.default_fillvals['f8'])
        vars_optim[mth] = var_optim

    divs = np.unique(climdiv_stns[np.isfinite(climdiv_stns)])

    stchk = StatusCheck(divs.size, 10)

    for clim_div in divs:

        fpath = os.path.join(path_xval_ds, "optim_nstns_%s_climdiv%d.nc" % (stnda.var_name, clim_div))

        ds_climdiv = Dataset(fpath)

        mae_climdiv = ds_climdiv.variables['mae'][:]
        nnghs_climdiv = ds_climdiv.variables['min_nghs'][:]

        climdiv_mask = np.nonzero(climdiv_stns == clim_div)[0]

        for mth in np.arange(1, 13):

            mae_climdiv_mth = mae_climdiv[mth - 1, :, :]
            mmae = np.mean(mae_climdiv_mth, axis=1)
            min_idx = np.argmin(mmae)
            vars_optim[mth][climdiv_mask] = nnghs_climdiv[min_idx]

        stchk.increment()

    stnda.ds.sync()

def set_optim_nstns_tair_anom(stnda, path_xval_ds):
    '''
    Set the local optimal number of stations to be used for anomaly
    interpolation each U.S. climate division based on cross-validation
    mean absolute error.
    
    Parameters
    ----------
    stnda : twx.db.StationSerialDataDb
        A StationSerialDataDb object pointing to the
        database for which the local optimal number of
        neighbors should be set. 
    path_xval_ds : str
        Path where netCDF cross-validation MAE files from
        create_climdiv_optim_nstns_db are located
    '''
    
    climdiv_stns = stnda.stns[CLIMDIV]

    vars_optim = {}
    for mth in np.arange(1, 13):

        var_name_optim = get_optim_anom_varname(mth)
        long_name = "Optimal number of neighbors to use for daily anomaly interpolation for month %d" % mth
        var_optim = stnda.add_stn_variable(var_name_optim, long_name, "", 'f8',
                                          fill_value=netCDF4.default_fillvals['f8'])
        vars_optim[mth] = var_optim
        
    divs = np.unique(climdiv_stns[np.isfinite(climdiv_stns)])

    stchk = StatusCheck(divs.size, 10)

    for clim_div in divs:

        fpath = os.path.join(path_xval_ds, "optim_nstns_%s_climdiv%d.nc" % (stnda.var_name, clim_div))

        ds_climdiv = Dataset(fpath)

        mae_climdiv = ds_climdiv.variables['mae'][:]
        nnghs_climdiv = ds_climdiv.variables['min_nghs'][:]

        climdiv_mask = np.nonzero(climdiv_stns == clim_div)[0]

        for mth in np.arange(1, 13):

            mae_climdiv_mth = mae_climdiv[mth - 1, :, :]
            mmae = np.mean(mae_climdiv_mth, axis=1)
            min_idx = np.argmin(mmae)
            vars_optim[mth][climdiv_mask] = nnghs_climdiv[min_idx]

        stchk.increment()

    stnda.ds.sync()

def build_nstn_bandwidths(rng_min, rng_max, pct_step):
    '''
    Build a range of bandwidths within a given interval 
    for the number of stations to use in interpolation.
    
    Parameters
    ----------
    rng_min : int
        The minimum bandwidth for the interval
    rng_max : int
        The maximum bandwidth for the interval
    pct_step : float
        A fractional step increase from 0.0-1.0 for
        specifying the spacing between bandwidths within
        the interval 
    
    Returns
    ----------
    ndarray
        An array of station bandwidths.
    '''

    min_nghs = []
    n = rng_min

    while n <= rng_max:
        min_nghs.append(n)
        n = n + np.round(pct_step * n)

    return np.array(min_nghs)


class StationKrigParams(object):
    '''
    Class to build moving window regression kriging
    variogram parameters for station locations. This is mainly
    used to determine the monthly normal regression kriging
    parameters at each station location once the optimal station
    bandwidths have been set.
    '''
    
    def __init__(self, path_db, tair_var):
        '''
        Parameters
        ----------
        path_db : str
            File path to a serially complete netCDF
            station database containing the stations and
            temperature variable for interpolation.
        tair_var : str
            The temperature variable for interpolation ('tmin' or 'tmax')
        '''

        stn_da = StationSerialDataDb(path_db, tair_var)
        mask_stns = np.isnan(stn_da.stns[BAD])

        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=False)
        krigparams = BuildKrigParams(stn_slct)

        self.stn_da = stn_da
        self.krigparams = krigparams

    def get_krig_params(self, stn_id):
        '''
        Get the monthly-varying moving window regression kriging variogram
        parameters for a specific station point. Currently
        assumes an exponential variogram.
        
        Parameters
        ----------
        stn_id : str
            Station id for which to retrieve variogram parameters.
            
        Returns
        ----------
        nugs : ndarray
            Array of exponential variogram nuggets for each month.
        psills : ndarray
            Array exponential variogram partial sills for each month.
        rngs : ndarray
            Array of exponential variogram ranges for each month.
        
        '''
        
        stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]

        nugs = np.zeros(12)
        psills = np.zeros(12)
        rngs = np.zeros(12)

        for mth in np.arange(1, 13):

            nug, psill, rng = self.krigparams.get_krig_params(stn, mth)

            nugs[mth - 1] = nug
            psills[mth - 1] = psill
            rngs[mth - 1] = rng

        return nugs, psills, rngs


class XvalTairAnom(object):
    '''
    Class for running a cross validation to optimize the local number
    of neighboring stations to use for geographically weighted regression
    of daily temperature anomalies.
    '''

    def __init__(self, path_db, tair_var):
        '''
        Parameters
        ----------
        path_db : str
            File path to a serially complete netCDF
            station database containing the stations and
            temperature variable for interpolation.
        tair_var : str
            The temperature variable for interpolation ('tmin' or 'tmax')
        '''

        stn_da = StationSerialDataDb(path_db, tair_var, vcc_size=470560000 * 2)
        mask_stns = np.isnan(stn_da.stns[BAD])

        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
        gwr = GwrTairAnom(stn_slct)

        self.stn_da = stn_da
        self.gwr = gwr

    def run_xval(self, stn_id, a_nnghs):

        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]
        xval_obs = self.stn_da.load_obs(xval_stn[STN_ID])

        biasAll = np.zeros((a_nnghs.size, 12))
        maeAll = np.zeros((a_nnghs.size, 12))
        r2All = np.zeros((a_nnghs.size, 12))

        for x in np.arange(a_nnghs.size):

            nnghs = a_nnghs[x]

            biasMths = np.zeros(12)
            maeMths = np.zeros(12)
            r2Mths = np.zeros(12)

            for mth in np.arange(1, 13):
                
                xval_anom = xval_obs[self.stn_da.mth_idx[mth]] - xval_stn[get_norm_varname(mth)]

                interp_tair = self.gwr.gwr_mth(xval_stn, mth, nnghs, stns_rm=xval_stn[STN_ID])
                interp_anom = interp_tair - xval_stn[get_norm_varname(mth)]

                difs = interp_anom - xval_anom

                bias = np.mean(difs)
                mae = np.mean(np.abs(difs))

                r_value = stats.linregress(interp_anom, xval_anom)[2]
                r2 = r_value ** 2  # r-squared value; variance explained

                biasMths[mth - 1] = bias
                maeMths[mth - 1] = mae
                r2Mths[mth - 1] = r2

            biasAll[x, :] = biasMths
            maeAll[x, :] = maeMths
            r2All[x, :] = r2Mths

        return biasAll, maeAll, r2All

class XvalTairOverall():
    '''
    Class for running a cross validation of interpolated
    monthly temperature normals and daily temperatures using
    previously optimized variogram and number of stations bandwidth
    parameters.
    '''

    def __init__(self, path_db, tair_var):
        '''
        Parameters
        ----------
        path_db : str
            File path to a serially complete netCDF
            station database containing the stations and
            temperature variable for interpolation.
        tair_var : str
            The temperature variable for interpolation ('tmin' or 'tmax')
        '''

        stn_da = StationSerialDataDb(path_db, tair_var, vcc_size=470560000 * 2)
        mask_stns = np.isnan(stn_da.stns[BAD])
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)

        krig_tair = KrigTair(stn_slct)
        gwr_tair = GwrTairAnom(stn_slct)
        interp_tair = InterpTair(krig_tair, gwr_tair)

        self.stn_da = stn_da
        self.interp_tair = interp_tair
        self.mth_masks = stn_da.mth_idx

    def run_interp(self, stn_id):
        '''
        Run leave-one-out cross validations 
        for a specific station id.
        
        Parameters
        ----------
        stn_id : str
            Station id for cross validation
        
        Returns
        ----------
        tair_daily : ndarray
            A 1-D array of interpolated daily temperatures.
        tair_norms : ndarray
            A 1-D array of size 12 with the interpolated 
            monthly temperature normals.
        tair_se : ndarray
            A 1-D array of size 12 with the kriging standard
            errors for the interpolated monthly temperature
            normals.
        '''

        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]
        tair_daily, tair_norms, tair_se = self.interp_tair.interp(xval_stn, xval_stn[STN_ID])
        return tair_daily, tair_norms, tair_se
