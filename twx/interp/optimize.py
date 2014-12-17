'''
Created on Sep 25, 2013

@author: jared.oyler
'''

import numpy as np
from twx.db import StationSerialDataDb, STN_ID, get_norm_varname, \
get_optim_varname, get_optim_anom_varname, BAD, dbDataset, CLIMDIV
from twx.interp import StationSelect, KrigTairAll, BuildKrigParams, \
GwrTairAnom
from netCDF4 import Dataset
import netCDF4
from twx.utils import StatusCheck
import scipy.stats as stats
import os
from twx.interp.interp_tair import KrigTair, InterpTair

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

class OptimGwrNormBwNstns(object):
    '''
    classdocs
    '''

    def __init__(self, pathDb, tairVar):

        stn_da = StationSerialDataDb(pathDb, tairVar)
        mask_stns = np.isnan(stn_da.stns[BAD])
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)

        self.gwr_norm = it.GwrTairNorm(stn_slct)
        self.stn_da = stn_da

    def run_xval(self, stnId, abw_nngh):

        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stnId]]

        err = np.zeros((12, abw_nngh.size))
        xvalNorms = np.array([xval_stn[get_norm_varname(mth)] for mth in np.arange(1, 13)])

        for bw_nngh, x in zip(abw_nngh, np.arange(abw_nngh.size)):

            interp_norms = np.zeros(12)

            for i in np.arange(interp_norms.size):

                mth = i + 1
                interp_mth = self.gwr_norm.gwr_predict(xval_stn, mth, nnghs=bw_nngh, stns_rm=xval_stn[STN_ID])[0]
                interp_norms[i] = interp_mth

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

class XvalGwrNormOverall(object):
    '''
    classdocs
    '''

    def __init__(self, pathDb, tairVar):

        stn_da = StationSerialDataDb(pathDb, tairVar)
        mask_stns = np.isnan(stn_da.stns[BAD])
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)

        self.gwr_norm = it.GwrTairNorm(stn_slct)
        self.stn_da = stn_da
        self.blank_err = np.zeros(13)

    def run_xval(self, stnId):

        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stnId]]

        xval_norms = np.array([xval_stn[get_norm_varname(mth)] for mth in np.arange(1, 13)])
        xval_norms = np.concatenate((xval_norms, np.array([np.mean(xval_norms)])))

        interp_norms = np.zeros(13)
        interp_se = np.zeros(12)

        for i in np.arange(12):

            mth = i + 1
            interp_mth, vary_mth = self.gwr_norm.gwr_predict(xval_stn, mth, stns_rm=xval_stn[STN_ID])
            interp_norms[i] = interp_mth
            interp_se[i] = np.sqrt(vary_mth) if vary_mth >= 0 else 0

        interp_norms[-1] = np.mean(interp_norms[0:12])

        err = interp_norms - xval_norms

        biasNorm = err
        maeNorm = np.abs(err)
        maeDly = self.blank_err
        biasDly = self.blank_err
        r2Dly = self.blank_err
        tair_se = interp_se

        return biasNorm, maeNorm, maeDly, biasDly, r2Dly, tair_se

def perfXvalTairOverall():

    optim = XvalTairOverall("/Users/jaredwo/Documents/data/serial_tmax.nc", 'tmax')

    biasNorm, maeNorm, maeDly, biasDly, r2Dly, seNorm = optim.run_xval('GHCN_USC00244558')

    print "MAE Norm"
    print maeNorm
    print "SE Norm"
    print seNorm
    print "MAE Daily"
    print maeDly

#    def runPerf():
#        biasNorm,maeNorm,maeDly,biasDly,r2Dly = optim.run_xval('SNOTEL_13C01S')
#
#    global runAPerf
#    runAPerf = runPerf
#
#    cProfile.run('runAPerf()')

def perfOptimTairAnom():

    optim = XvalTairAnom("/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc", 'tmin')

    min_ngh_wins = build_nstn_bandwidths(10, 150, 0.10)

    # biasAll, maeAll, r2All = optim.run_xval('SNOTEL_13C01S', min_ngh_wins)
    # biasAll, maeAll, r2All = optim.run_xval('GHCN_USC00244558', min_ngh_wins) #Kalispell
    biasAll, maeAll, r2All = optim.run_xval('GHCN_USC00247448', min_ngh_wins)  # Seeley Lake
    # biasAll, maeAll, r2All = optim.run_xval('GHCN_USW00014755', min_ngh_wins)

    mae_argmin = np.argmin(maeAll, 0)
    cols = np.arange(maeAll.shape[1])

    print "NNGHS"
    print min_ngh_wins[mae_argmin]
    print "MAE"
    print maeAll[mae_argmin, cols]
    print "BIAS"
    print biasAll[mae_argmin, cols]
    print "R2"
    print r2All[mae_argmin, cols]

#    print min_ngh_wins
#    maeMth = maeAll[:,7]
#    print maeMth
#    print min_ngh_wins[np.argmin(maeMth)]
#    print np.min(maeMth)
#    def runPerf():
#        optim.run_xval('SNOTEL_13C01S',min_ngh_wins,0.20)
#
#    global runAPerf
#    runAPerf = runPerf
#
#    cProfile.run('runAPerf()')

def perfOptimKrigParams():

    optim = StationKrigParams("/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc", 'tmax')

    nugs, psills, rngs = optim.get_krig_params('RAWS_CCRN')  # ('SNOTEL_19L43S')GHCN_USC00049043
    print nugs
#    def runPerf():
#        print optim.get_krig_params('SNOTEL_13C01S')
#
#    global runAPerf
#    runAPerf = runPerf
#
#    cProfile.run('runAPerf()')


def analyze_xval_tairmean():
    ds = Dataset('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
    se = ds.variables['xval_stderr_mthly'][:]
    climDiv = ds.variables['neon'][:].data
    maskClimDiv = climDiv == 2401

    maskStns = np.logical_and(~se[0, :].mask, maskClimDiv)
    e = ds.variables['xval_err_mthly'][:, maskStns]
    se = se[:, maskStns]

    eOld = np.mean(np.abs(ds.variables['xval_err'][maskStns]))
    print "eOld", eOld

    norms = []
    for mth in np.arange(1, 13):
        norm = ds.variables[get_norm_varname(mth)][maskStns]
        norm.shape = (1, norm.size)
        norms.append(norm)

    norms = np.vstack(norms)
    norms = np.vstack((norms, np.mean(norms, axis=0)))

    interps = norms + e

    crit = stats.norm.ppf(0.025)
    cir = np.abs(se * crit)

    cil, ciu = interps - cir, interps + cir

    inCi = np.logical_and(norms >= cil, norms <= ciu)

    print "In CI", np.sum(inCi, axis=1) / np.float(inCi.shape[1])
    print "MAE", np.mean(np.abs(e), axis=1)
    print "Bias", np.mean(e, axis=1)
    print "SE", np.mean(se, axis=1)

def analyze_xval_overall():
    ds = Dataset('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
    se = ds.variables['xval_stderr_mthly'][:]
    climDiv = ds.variables['neon'][:].data
    maskClimDiv = climDiv == 2401

    maskStns = np.logical_and(~se[0, :].mask, maskClimDiv)
    e = ds.variables['xval_err_mthly'][:, maskStns]
    se = se[:, maskStns]

    eOld = np.mean(np.abs(ds.variables['xval_err'][maskStns]))
    print "eOld", eOld

    norms = []
    for mth in np.arange(1, 13):
        norm = ds.variables[get_norm_varname(mth)][maskStns]
        norm.shape = (1, norm.size)
        norms.append(norm)

    norms = np.vstack(norms)
    norms = np.vstack((norms, np.mean(norms, axis=0)))

    interps = norms + e

    crit = stats.norm.ppf(0.025)
    cir = np.abs(se * crit)

    cil, ciu = interps - cir, interps + cir

    inCi = np.logical_and(norms >= cil, norms <= ciu)

    print "In CI", np.sum(inCi, axis=1) / np.float(inCi.shape[1])
    print "MAE", np.mean(np.abs(e), axis=1)
    print "Bias", np.mean(e, axis=1)
    print "SE", np.mean(se, axis=1)

def perfPtInterpTair():

    stndaTmin = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    stndaTmax = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')


    # stndaTmin = it.StationDataWrkChk('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    # stndaTmax = it.StationDataWrkChk('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')

    path = '/projects/daymet2/dem/interp_grids/conus/ncdf/'

    auxFpaths = ["".join([path, 'fnl_elev.nc']),
                 "".join([path, 'fnl_tdi.nc']),
                 "".join([path, 'fnl_climdiv.nc'])]

    for mth in np.arange(1, 13):
        auxFpaths.append("".join([path, 'fnl_lst_tmin%02d.nc' % mth]))
        auxFpaths.append("".join([path, 'fnl_lst_tmax%02d.nc' % mth]))

    ptInterp = it.PtInterpTair(stndaTmin, stndaTmax,
                               '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                               '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', auxFpaths)

    lon, lat = -114.02853698, 47.87532492
    tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterp.interp_to_lonlat(lon, lat, fixInvalid=False)
    print tmin_norms

#    print ninvalid
#    tmin_dly_norm = np.take(tmin_dly, ptInterp.daysNormMask)
#    tmax_dly_norm = np.take(tmax_dly, ptInterp.daysNormMask)
#    tmin_mthly = np.array([np.mean(np.take(tmin_dly_norm,amask)) for amask in ptInterp.yrMthsMasks])
#    tmax_mthly = np.array([np.mean(np.take(tmax_dly_norm,amask)) for amask in ptInterp.yrMthsMasks])
#    tmin_norms2 = np.array([np.mean(np.take(tmin_mthly,amask)) for amask in ptInterp.mth_masks])
#    tmax_norms2 = np.array([np.mean(np.take(tmax_mthly,amask)) for amask in ptInterp.mth_masks])
#    print tmin_norms2-tmin_norms,tmax_norms2-tmax_norms
#    plt.plot(tmin_norms2-tmin_norms,'o-')
#    plt.plot(tmax_norms2-tmax_norms,'o-')
#    plt.show()

def perftOptimKrigBwStns():

    optim = XvalTairNorm('/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc', 'tmin')

    abw_nngh = build_nstn_bandwidths(35, 150, 0.10)

    err = optim.run_xval('GHCN_USC00244558', abw_nngh)
    # err = optim.run_xval('SNOTEL_13C01S', abw_nngh)

    mae = np.abs(err)

    print abw_nngh[np.argmin(mae, 1)]
    print mae[np.arange(12), np.argmin(mae, 1)]

#     optim = OptimTairMean("/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc",
#                      '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R', 'tmax')
#
#    #aStn = optim.stn_da.stns[optim.stn_da.stns[STN_ID]=='SNOTEL_13C01S'][0]
#
#    rngMinNghs = build_nstn_bandwidths(100,150, 0.10)

if __name__ == '__main__':

    # perftOptimKrigBwStns()
    # perfPtInterpTair()
    # analyze_ci()
    perfXvalTairOverall()
    # perfOptimTairAnom()
    # perfOptimKrigParams()
    # perfOptimTairMean()
    # perfXvalTairMean()
