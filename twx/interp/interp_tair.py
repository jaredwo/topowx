'''
Classes and functions for performing Tair interpolation.

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

__all__ = ["KrigTairAll", "BuildKrigParams", "GwrTairAnom",
           'KrigTair', 'InterpTair', 'StationDataWrkChk',
           'PtInterpTair']

from twx.interp import StationSelect
import numpy as np
from twx.db import LON, LAT, ELEV, TDI, LST, \
    VARIO_NUG, VARIO_PSILL, VARIO_RNG, BAD, MASK, \
    StationSerialDataDb, get_norm_varname, \
    get_optim_varname, get_krigparam_varname, get_lst_varname, \
    get_optim_anom_varname, CLIMDIV
import scipy.stats as stats
from twx.utils.util_ncdf import GeoNc
from netCDF4 import Dataset
from twx.utils import MONTH, YEAR, get_mth_metadata
import mpl_toolkits.basemap as bm
import os
# rpy2
robjects = None
numpy2ri = None
r = None
ri = None
R_LOADED = False

KRIG_TREND_VARS = (LON, LAT, ELEV, LST)
GWR_TREND_VARS = (LON, LAT, ELEV, TDI, LST)
LST_TMAX = 'lst_tmax'
LST_TMIN = 'lst_tmin'

DFLT_INIT_NNGHS = 100


def _init_interp_R_env():
    
    global R_LOADED

    if not R_LOADED:
        
        global robjects
        global numpy2ri
        global r
        global ri
        
        import rpy2
        import rpy2.robjects
        robjects = rpy2.robjects
        r = robjects.r
        import rpy2.rinterface
        ri = rpy2.rinterface
        
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        
        # get system path to twx
        twx_path = os.path.split(os.path.split(__file__)[0])[0]
        # get system path to interp.R
        rsrc_path = os.path.join(twx_path, 'interp', 'rpy', 'interp.R')
        
        r.source(rsrc_path)
        
        # Set trend formula
        ri.globalenv['FORMULA'] = ri.globalenv.get("build_formula")(ri.StrSexpVector(["tair"]), ri.StrSexpVector(KRIG_TREND_VARS))
        
        R_LOADED = True
            
class PredictorGrids():

    def __init__(self, ncFpaths, interpOrders=None):
        
        self.ncDs = {}
        self.ncData = {}
        self.xGrid = None
        self.yGrid = None
        
        interpOrders = np.zeros(len(ncFpaths)) if interpOrders is None else interpOrders 
        
        for fpath, interpOrder in zip(ncFpaths, interpOrders):
            
            geoNc = GeoNc(Dataset(fpath))
            varKeys = np.array(geoNc.ds.variables.keys())
            ncVarname = varKeys[np.logical_and(varKeys != 'lon', varKeys != 'lat')][0]
            self.ncDs[ncVarname] = geoNc
            
            if interpOrder == 1:
                a = geoNc.ds.variables[ncVarname][:]
                aflip = np.flipud(a)
                # aflip = aflip.astype(np.float)
                self.ncData[ncVarname] = aflip
                
                if self.xGrid == None:
                    self.xGrid = geoNc.ds.variables['lon'][:]
                    self.yGrid = np.sort(geoNc.ds.variables['lat'][:])
                        
    def setPtValues(self, aPt, chgLatLon=True):
                
        chged = False
        for varname, geoNc in self.ncDs.items():
            
            if chgLatLon or not self.ncData.has_key(varname):
            
                row, col, gridlon, gridlat = geoNc.get_row_col(aPt[LON], aPt[LAT])
                aPt[varname] = geoNc.ds.variables[varname][row, col]
                
                if chgLatLon and not chged:
                    aPt[LON] = gridlon
                    aPt[LAT] = gridlat
                    chged = True
            
            else:
                
                rval = bm.interp(self.ncData[varname].astype(np.float), self.xGrid, self.yGrid, np.array(aPt[LON]), np.array(aPt[LAT]), checkbounds=False, masked=True, order=1)
        
                if np.ma.is_masked(rval):
                    
                    rval = bm.interp(self.ncData[varname], self.xGrid, self.yGrid, np.array(aPt[LON]), np.array(aPt[LAT]), checkbounds=False, masked=True, order=0)
                        
                    if np.ma.is_masked(rval):
                        rval = geoNc.ds.variables[varname].missing_value
                    
                aPt[varname] = rval
            
def tmin_tmax_fixer(tmin, tmax, tail=15):
    '''
    Checks for days where Tmin >= Tmax. If found, applies
    a fix. The average of the invalid Tmin, Tmax pair is
    calculated and then the average diurnal temperature 
    range for specified temporal window (x-tail to x+tail)
    is added (subtracted) to the average to get the
    fixed Tmax (Tmin).
    
    Parameters
    ----------
    tmin : ndarray
        Daily Tmin
    tmax : ndarray
        Daily Tmax
    tail : int, optional
        Number of days  before/after an invalid Tmin >= Tmax over which
        an average diurnal temperature range should be calculated.
        Default: 15.
    
    Returns
    ----------
    tmin : ndarray
        Daily Tmin with invalid values fixed
    tmax : ndarray
        Daily Tmax with invalid values fixed
    invalid_days : int
        The number of days that were fixed
    '''

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


def build_empty_pt():
    
    ptDtype = [(LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
               (TDI, np.float64), (CLIMDIV, np.float64), (MASK, np.float64)]
    ptDtype.extend([("tmin%02d" % mth, np.float64) for mth in np.arange(1, 13)]) 
    ptDtype.extend([("tmax%02d" % mth, np.float64) for mth in np.arange(1, 13)])
    ptDtype.extend([(get_norm_varname(mth),np.float64) for mth in np.arange(1,13)])
    ptDtype.extend([(get_optim_varname(mth),np.float64) for mth in np.arange(1,13)])
    ptDtype.extend([(get_lst_varname(mth),np.float64) for mth in np.arange(1,13)])
    ptDtype.extend([(get_optim_anom_varname(mth),np.float64) for mth in np.arange(1,13)])
    
    a_pt = np.empty(1, dtype=ptDtype)
    
    return a_pt[0]

class GwrTairAnom(object):
    '''
    Class to perform geographically weighted regression interpolation
    of daily temperature anomalies for a specified month.
    '''
    
    def __init__(self, stn_slct):
        '''
        Parameters
        ----------
        stn_slct : StationSelect
            A StationSelect object for finding and
            selecting neighboring stations to a point.
        '''
        
        self.stn_slct = stn_slct
        
        mthly_predictors = {}
        predictors = np.array(GWR_TREND_VARS, dtype="<S16")
        mthly_predictors[None] = predictors
        
        for mth in np.arange(1, 13):
            
            mthP = np.copy(predictors)
            mthP[mthP == LST] = get_lst_varname(mth)
            mthly_predictors[mth] = mthP
        
        self.mthly_predictors = mthly_predictors

    
    def __get_nnghs(self, pt, mth, stns_rm=None):
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], DFLT_INIT_NNGHS, load_obs=False, stns_rm=stns_rm)
        
        fin_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_anom_varname(mth)])
        
        if np.sum(fin_mask) == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
    
        p_stns = self.stn_slct.ngh_stns[fin_mask]
        p_wgt = self.stn_slct.ngh_wgt[fin_mask]
        
        nnghs = np.int(np.round(np.average(p_stns[get_optim_anom_varname(mth)], weights=p_wgt)))

        return nnghs
    
    def gwr_mth(self, pt, mth, nnghs=None, stns_rm=None):
        '''
        Run geographically weighted regression of daily temperature anomalies 
        for a specific month and point location. The function interpolates
        daily anomalies using GWR and then adds the anomalies to the point's
        monthly normal and returns the actual daily values.
        
        Parameters
        ----------
        pt : structured array
            A structured array containing the point's latitude, longitude,
            elevation, topographic dissection index, and average land skin 
            temperatures for each month. An empty point can be initialized
            with build_empty_pt()
        nnghs : int, optional
            The number of closest neighboring stations to use for the GWR.
            If None, nnghs will be set to the optimized number of GWR neighbors
            for the point's region.
        stns_rm : ndarray or str, optional
            An array of station ids or a single station id for stations that
            should not be considered neighbors for the specific point.
        
        Returns
        ----------
        interp_vals : ndarray
            A 1-D array containing the GWR interpolated actual daily values
            for the specified month. 
        '''
        
        if nnghs == None:
            # Get the nnghs to use from the optimal values 
            # at surrounding stations
            nnghs = self.__get_nnghs(pt, mth, stns_rm)
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], nnghs, load_obs=True, stns_rm=stns_rm, obs_mth=mth)
        
        ngh_obs = self.stn_slct.ngh_obs
        ngh_stns = self.stn_slct.ngh_stns
        ngh_wgt = self.stn_slct.ngh_wgt
        ngh_obs_cntr = ngh_obs - ngh_stns[get_norm_varname(mth)]
        
        # Perform a GWR for each day
        X = [ngh_stns[avar] for avar in self.mthly_predictors[mth]]
        X = np.column_stack(X)
    
        x = [pt[avar] for avar in self.mthly_predictors[mth]]
        x = np.array(x)
        
        interp_anom = _gwr_series(X, x, ngh_obs_cntr, ngh_wgt)        
        
        # Add interpolated anomalies to monthly norm to get actual values
        interp_vals = interp_anom + pt[get_norm_varname(mth)]
                
        return interp_vals 

class GwrTairAnomR(GwrTairAnom):
    
    def __init__(self, stn_slct):
        
        _init_interp_R_env()
        GwrTairAnom.__init__(self, stn_slct)
        
    
    def gwr_mth(self, pt, mth, nnghs=None, stns_rm=None):
        
        if nnghs == None:
            # Get the nnghs to use from the optimal values at surrounding stations
            nnghs = self._GwrTairAnom__get_nnghs(pt, mth, stns_rm)
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], nnghs, load_obs=True, stns_rm=stns_rm, obs_mth=mth)
        
        ngh_obs = self.stn_slct.ngh_obs
        ngh_stns = self.stn_slct.ngh_stns
        ngh_wgt = self.stn_slct.ngh_wgt
        ngh_obs_cntr = ngh_obs - ngh_stns[get_norm_varname(mth)]
        
        a_pt = np.array([pt[LON], pt[LAT], pt[ELEV], pt[TDI], pt[get_lst_varname(mth)]])
                   
        rslt = r.gwr_anomaly(robjects.FloatVector(ngh_stns[LON]),
                          robjects.FloatVector(ngh_stns[LAT]),
                          robjects.FloatVector(ngh_stns[ELEV]),
                          robjects.FloatVector(ngh_stns[TDI]),
                          robjects.FloatVector(ngh_stns[get_lst_varname(mth)]),
                          robjects.FloatVector(ngh_wgt),
                          robjects.Matrix(ngh_obs_cntr),
                          robjects.FloatVector(a_pt))
                
        fit_anom = np.array(rslt.rx('fit_anom'))
        nrow = np.array(rslt.rx('fit_nrow'))[0]
        ncol = np.array(rslt.rx('fit_ncol'))[0]
        fit_anom = np.reshape(fit_anom, (nrow, ncol), order='F')
        
        interp_anom = np.array(rslt.rx('pt_anom')).ravel()
                
        interp_vals = interp_anom + pt[get_norm_varname(mth)]
                
        return interp_vals 

class GwrTairAnomBlank(GwrTairAnom):
    
    def __init__(self, stn_slct):
        
        GwrTairAnom.__init__(self, stn_slct)
        self.blank_vals = {}
        for mth in np.arange(1, 13):
            self.blank_vals[mth] = np.zeros(self.mth_idx[mth].size)
        
    def gwr_mth(self, pt, mth, nnghs=None, stns_rm=None):  
        return self.blank_vals[mth] 
      
class InterpTair(object):
    '''
    Class to interpolate monthly temperature normals
    with moving window regression and daily temperatures
    with geographically weighted regression for single
    temperature variable (Tmin or Tmax). 
    '''
    
    def __init__(self, krig_tair, gwr_tair):
        '''
        Parameters
        ----------
        krig_tair : KrigTair
            A KrigTair object for running
            moving window regression kriging.
        gwr_tair : GwrTairAnom
            A GwrTairAnom object for running
            geographically weighted regression.
        '''
        
        self.krig_tair = krig_tair
        self.gwr_tair = gwr_tair
        self.mth_masks = self.gwr_tair.stn_slct.stn_da.mth_idx
        self.ndays = self.gwr_tair.stn_slct.stn_da.days.size
        
    def interp(self, pt, stns_rm=None):
        '''
        Interpolate monthly temperature normals and daily 
        temperatures to a single point location.
        
        Parameters
        ----------
        pt : structured array
            A structured array containing the point's latitude, longitude,
            elevation, topographic dissection index, and average land skin 
            temperatures for each month. An empty point can be initialized
            with build_empty_pt()
        stns_rm : ndarray or str, optional
            An array of station ids or a single station id for stations that
            should not be considered neighbors for the specific point.
        
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
        
        tair_daily = np.zeros(self.ndays)
        tair_norms = np.zeros(12)
        tair_se = np.zeros(12)
                
        for mth in np.arange(1, 13):
            
            tair_mean, tair_var = self.krig_tair.krig(pt, mth, stns_rm=stns_rm)
            std_err, ci = self.krig_tair.std_err_ci(tair_mean, tair_var)
            pt[get_norm_varname(mth)] = tair_mean
            tair_norms[mth - 1] = tair_mean
            tair_se[mth - 1] = std_err
    
            tair_daily[self.mth_masks[mth]] = self.gwr_tair.gwr_mth(pt, mth, stns_rm=stns_rm)
        
        return tair_daily, tair_norms, tair_se

class PtInterpTair(object):
    '''
    Class to interpolate monthly temperature normals
    with moving window regression and daily temperatures
    with geographically weighted regression for both
    Tmin and Tmax. 
    '''
    
    def __init__(self, stn_da_tmin, stn_da_tmax, aux_fpaths=None, interp_orders=None, norms_only=False):
        '''
        Parameters
        ----------
        stn_da_tmin : twx.db.StationSerialDataDb
            A StationSerialDataDb object pointing to the
            database from which Tmin observations should
            be loaded.
        stn_da_tmax : twx.db.StationSerialDataDb
            A StationSerialDataDb object pointing to the
            database from which Tmax observations should
            be loaded.      
        '''
        self.days = stn_da_tmin.days
        self.stn_da_tmin = stn_da_tmin
        self.stn_da_tmax = stn_da_tmax
        
        # Masks for calculating monthly norms after daily Tmin/Tmax values 
        # had to be adjusted due to Tmin >= Tmax
        self.daysNormMask = np.nonzero(np.logical_and(self.days[YEAR] >= 1981, self.days[YEAR] <= 2010))[0] 
        daysNorm = self.days[self.daysNormMask]
        
        uYrs = np.unique(daysNorm[YEAR])
        self.yr_mths = get_mth_metadata(uYrs[0], uYrs[-1])
        
        self.yrMthsMasks = []
        for aYr in uYrs:
            for aMth in np.arange(1, 13):
                self.yrMthsMasks.append(np.nonzero(np.logical_and(daysNorm[YEAR] == aYr, daysNorm[MONTH] == aMth))[0])
        
        self.mth_masks = []
        for mth in np.arange(1, 13):
            self.mth_masks.append(np.nonzero(self.yr_mths[MONTH] == mth)[0])
        
        mask_stns_tmin = np.isnan(stn_da_tmin.stns[BAD]) 
        mask_stns_tmax = np.isnan(stn_da_tmax.stns[BAD])
        
        stn_slct_tmin = StationSelect(stn_da_tmin, mask_stns_tmin)
        stn_slct_tmax = StationSelect(stn_da_tmax, mask_stns_tmax)
        
        domain_stns_tmin = stn_da_tmin.stns[np.logical_and(mask_stns_tmin, np.isfinite(stn_da_tmin.stns[MASK]))]
        domain_stns_tmax = stn_da_tmax.stns[np.logical_and(mask_stns_tmax, np.isfinite(stn_da_tmax.stns[MASK]))]
        self.nnghparams_tmin = _get_rgn_nnghs_dict(domain_stns_tmin)
        self.nnghparams_tmax = _get_rgn_nnghs_dict(domain_stns_tmax)
        
        krig_tmin = KrigTair(stn_slct_tmin)
        krig_tmax = KrigTair(stn_slct_tmax)
        
        if norms_only:
            gwr_tmin = GwrTairAnomBlank(stn_slct_tmin)
            gwr_tmax = GwrTairAnomBlank(stn_slct_tmax)
        else:
            gwr_tmin = GwrTairAnom(stn_slct_tmin)
            gwr_tmax = GwrTairAnom(stn_slct_tmax)
                
        self.interp_tmin = InterpTair(krig_tmin, gwr_tmin)
        self.interp_tmax = InterpTair(krig_tmax, gwr_tmax)
        
        if aux_fpaths is not None:
            self.pGrids = PredictorGrids(aux_fpaths, interp_orders)
        
        self.a_pt = build_empty_pt()
        
    
    def interp_to_lonlat(self, lon, lat, fixInvalid=True, chgLatLon=True, stns_rm=None, elev=None):
        
        self.a_pt[LON] = lon
        self.a_pt[LAT] = lat
        self.pGrids.setPtValues(self.a_pt, chgLatLon)
        if elev is not None:
            self.a_pt[ELEV] = elev
        
        if self.a_pt[MASK] == 0:
            raise Exception('Point is outside interpolation region')
        
        return self.interp_pt(fixInvalid, stns_rm)
    
    def interp_pt(self, fix_invalid=True, stns_rm=None):
        '''
        Interpolate daily and monthly normal Tmin and Tmax values
        for the current PtInterpTair.a_pt
        
        Parameters
        ----------
        fix_invalid : boolean, optional
            If True, apply a fix on days where interpolated
            Tmax > Tmin. Default: True.
        stns_rm : ndarray or str, optional
            An array of station ids or a single station id for stations that
            should not be considered neighbors for the specific point.
            
        Returns
        ----------
        tmin_dly : ndarray
            Daily interpolated Tmin
        tmax_dly : ndarray
            Daily interpolated Tmax
        tmin_norms : ndarray
            Interpolated monthly Tmin normals
        tmax_norms : ndarray
            Interpolated monthly Tmax normals
        tmin_se : ndarray
            Kriging standard error for monthly Tmin normals
        tmax_se : ndarray
            Kriging standard error for monthly Tmax normals
        ninvalid : int
            The number of days where Tmax > Tmin was fixed.
            If fix_invalid is False, will be set to 0.        
        '''
        
        # Set the monthly lst values and optim nnghs on the point
        for mth in np.arange(1, 13):
            
            self.a_pt[get_lst_varname(mth)] = self.a_pt["tmin%02d" % mth]
            self.a_pt[get_optim_varname(mth)], self.a_pt[get_optim_anom_varname(mth)] = self.nnghparams_tmin[self.a_pt[CLIMDIV]][mth]

        # Perform Tmin interpolation
        tmin_dly, tmin_norms, tmin_se = self.interp_tmin.interp(self.a_pt, stns_rm=stns_rm)
        
        # Set the monthly lst values and optim nnghs on the point
        for mth in np.arange(1, 13):
            
            self.a_pt[get_lst_varname(mth)] = self.a_pt["tmax%02d" % mth]
            self.a_pt[get_optim_varname(mth)], self.a_pt[get_optim_anom_varname(mth)] = self.nnghparams_tmax[self.a_pt[CLIMDIV]][mth]
        
        # Perform Tmax interpolation
        tmax_dly, tmax_norms, tmax_se = self.interp_tmax.interp(self.a_pt, stns_rm=stns_rm)
        
        ninvalid = 0
        
        if fix_invalid:
            
            tmin_dly, tmax_dly, ninvalid = tmin_tmax_fixer(tmin_dly, tmax_dly)
    
            if ninvalid > 0:
                
                tmin_dly_norm = np.take(tmin_dly, self.daysNormMask)
                tmax_dly_norm = np.take(tmax_dly, self.daysNormMask)
                tmin_mthly = np.array([np.mean(np.take(tmin_dly_norm, amask)) for amask in self.yrMthsMasks])
                tmax_mthly = np.array([np.mean(np.take(tmax_dly_norm, amask)) for amask in self.yrMthsMasks])
                tmin_norms = np.array([np.mean(np.take(tmin_mthly, amask)) for amask in self.mth_masks])
                tmax_norms = np.array([np.mean(np.take(tmax_mthly, amask)) for amask in self.mth_masks])
        
        return tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid

def _get_rgn_nnghs_dict(stns):
    
    rgns = np.unique(stns[CLIMDIV][np.isfinite(stns[CLIMDIV])])
    
    nnghsAll = {}
    
    for rgn in rgns:
        
        rgn_mask = stns[CLIMDIV] == rgn
        nnghsRgn = {}
        
        for mth in np.arange(1, 13):
            nnghsRgn[mth] = (stns[get_optim_varname(mth)][rgn_mask][0], stns[get_optim_anom_varname(mth)][rgn_mask][0])
        
        nnghsAll[rgn] = nnghsRgn
        
    return nnghsAll
    
class BuildKrigParams(object):
    '''
    Class to build moving window regression kriging
    variogram parameters for a specific point. This is mainly
    used to determine the monthly normal regression kriging
    parameters at each station location once the optimal station
    bandwidths have been set.
    '''
    
    def __init__(self, stn_slct):
        '''
        Parameters
        ----------
        stn_slct : StationSelect
            A StationSelect object for finding and
            selecting neighboring stations to a point.
        '''
        
        _init_interp_R_env()
        self.stn_slct = stn_slct
        self.r_func = ri.globalenv.get('get_vario_params')
        
        
    def get_krig_params(self, pt, mth, rm_stnid=None):
        '''
        Get the moving window regression kriging variogram
        parameters for a specific point and month. Currently
        assumes an exponential variogram
                
        Parameters
        ----------
        pt : structured array
            A structured array containing the point's latitude, longitude,
            elevation, topographic dissection index, and average land skin 
            temperatures for each month. An empty point can be initialized
            with build_empty_pt()
        mth : int
            The specific month as an integer (1-12)
        rm_stnid : ndarray or str, optional
            An array of station ids or a single station id for stations that
            should not be considered neighbors for the specific point.
        
        Returns
        ----------
        nug : float
            Exponential variogram nugget.
        psill : float
            Exponential variogram partial sill.
        rng : float
            Exponential variogram range.
        
        '''
        
        # First determine the nnghs to use based on smoothed weighted average of 
        # the optimal nnghs bandwidth at each station point.
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], DFLT_INIT_NNGHS, load_obs=False)
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_varname(mth)])
        
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        
        if domain_stns.size == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
        
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]
                    
        nnghs = np.int(np.round(np.average(domain_stns[get_optim_varname(mth)], weights=n_wgt)))
    
        # Now use the optimal nnghs to get the krig params for this mth
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], nnghs, load_obs=False)
     
        nghs = self.stn_slct.ngh_stns
        ngh_lon = ri.FloatSexpVector(nghs[LON])
        ngh_lat = ri.FloatSexpVector(nghs[LAT])
        ngh_elev = ri.FloatSexpVector(nghs[ELEV])
        ngh_tdi = ri.FloatSexpVector(nghs[TDI])
        ngh_lst = ri.FloatSexpVector(nghs[get_lst_varname(mth)])
        ngh_tair = ri.FloatSexpVector(nghs[get_norm_varname(mth)])
        ngh_wgt = ri.FloatSexpVector(self.stn_slct.ngh_wgt)
        ngh_dists = ri.FloatSexpVector(self.stn_slct.ngh_dists)
        
        rslt = self.r_func(ngh_lon, ngh_lat, ngh_elev, ngh_tdi, ngh_lst, ngh_tair, ngh_wgt, ngh_dists)
        nug = rslt[0]
        psill = rslt[1]
        rng = rslt[2]
                
        return nug, psill, rng

class KrigTairAll(object):
    '''
    Class to perform moving window variogram fitting and regression kriging
    of monthly normals all in one step. Uses the R gstat package. This is
    mainly used in the optimization of the local number of neighboring
    stations bandwidth to be used for moving window regression kriging in
    each month.
    '''
    
    def __init__(self, stn_slct):
        '''
        Parameters
        ----------
        stn_slct : StationSelect
            A StationSelect object for finding and
            selecting neighboring stations to a point.
        '''
        
        _init_interp_R_env()
        self.stn_slct = stn_slct
        self.r_func = ri.globalenv.get('krig_all')
        
    def krigall(self, pt, nnghs, stns_rm=None):
        '''
        Run moving window variogram fitting and regression kriging
        to interpolate monthly temperature normals to a single point location.
        
        Parameters
        ----------
        pt : structured array
            A structured array containing the point's latitude, longitude,
            elevation, topographic dissection index, and average land skin 
            temperatures for each month. An empty point can be initialized
            with build_empty_pt()
        nnghs : int
            The number of neighboring stations to use.
        stns_rm : ndarray or str, optional
            An array of station ids or a single station id for stations that
            should not be considered neighbors for the specific point.
        
        Returns
        ----------
        interp_norms : ndarray
            A 1-D array of size 12 containing 
            the interpolated monthly normals
        
        '''
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], nnghs, load_obs=False, stns_rm=stns_rm)
                
        nghs = self.stn_slct.ngh_stns
        ngh_lon = ri.FloatSexpVector(nghs[LON])
        ngh_lat = ri.FloatSexpVector(nghs[LAT])
        ngh_elev = ri.FloatSexpVector(nghs[ELEV])
        ngh_tdi = ri.FloatSexpVector(nghs[TDI])
        ngh_wgt = ri.FloatSexpVector(self.stn_slct.ngh_wgt)
        ngh_dists = ri.FloatSexpVector(self.stn_slct.ngh_dists)
        
        interp_norms = np.zeros(12)
        
        for mth in np.arange(1, 13):
            
            ngh_lst = ri.FloatSexpVector(nghs[get_lst_varname(mth)])
            ngh_tair = ri.FloatSexpVector(nghs[get_norm_varname(mth)])
            pt_svp = ri.FloatSexpVector((pt[LON], pt[LAT], pt[ELEV], pt[TDI], pt[get_lst_varname(mth)]))
            rslt = self.r_func(pt_svp, ngh_lon, ngh_lat, ngh_elev, ngh_tdi, ngh_lst, ngh_tair, ngh_wgt, ngh_dists)
            
            interp_norms[mth - 1] = rslt[0]
                            
        return interp_norms
                    
class KrigTair(object):
    '''
    Class to perform moving window regression kriging
    of monthly normals. Unlike KrigTairAll, the variogram
    is not fit. Variogram parameters (nugget, range, sill)
    must be passed or they are smoothed from previously fit 
    variogram parameters at neighboring stations.
    '''
    
    def __init__(self, stn_slct):
        '''
        Parameters
        ----------
        stn_slct : StationSelect
            A StationSelect object for finding and
            selecting neighboring stations to a point.
        '''
        
        _init_interp_R_env()
        self.stn_slct = stn_slct
        self.r_krig_func = ri.globalenv.get('krig_meantair')
        self.ci_critval = stats.norm.ppf(0.025)
            
    def std_err_ci(self, tair_mean, tair_var):
        '''
        Calculate the kriging standard error and 95%
        confidence interval for an interpolated temperature
        normal.
        
        Parameters
        ----------
        tair_mean : float
            A interpolated temperature normal.
        tair_var : float
            The kriging prediction variance for the 
            temperature normal.
            
        Returns
        ----------
        std_err : float
            Kriging standard error
        ci : tuple
            The 95% C.I. (tuple of size 2)
        '''
        
        std_err = np.sqrt(tair_var) if tair_var >= 0 else 0
        ci_r = np.abs(std_err * self.ci_critval) 
        
        return std_err, (tair_mean - ci_r, tair_mean + ci_r)
    
    def __get_nnghs(self, pt, mth, stns_rm=None):
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], DFLT_INIT_NNGHS, load_obs=False, stns_rm=stns_rm)
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_varname(mth)])
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        
        if domain_stns.size == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
        
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]
                    
        nnghs = np.int(np.round(np.average(domain_stns[get_optim_varname(mth)], weights=n_wgt)))
        
        return nnghs
    
    def __get_vario_params(self, pt, mth):
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_krigparam_varname(mth, VARIO_NUG)])
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        
        if domain_stns.size == 0:
            raise Exception("Cannot determine variogram params!")
        
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]
        
        nug = np.average(domain_stns[get_krigparam_varname(mth, VARIO_NUG)], weights=n_wgt)
        psill = np.average(domain_stns[get_krigparam_varname(mth, VARIO_PSILL)], weights=n_wgt) 
        vrange = np.average(domain_stns[get_krigparam_varname(mth, VARIO_RNG)], weights=n_wgt)  
                
        return nug, psill, vrange
    
    def krig(self, pt, mth, nnghs=None, vario_params=None, stns_rm=None):
        '''
        Run moving window regression kriging to interpolate a 
        temperature normal for a specific month to a single point location.
        
        Parameters
        ----------
        pt : structured array
            A structured array containing the point's latitude, longitude,
            elevation, topographic dissection index, and average land skin 
            temperatures for each month. An empty point can be initialized
            with build_empty_pt()
        mth : int, optional
            The month for which to interpolate a temperature normal
        nnghs : int, optional
            The number of neighboring stations to use. If not provided, nnghs
            will be determined from the previously determined optimal number
            at surrounding neighboring stations.
        vario_params : tuple, optional
            A tuple of size 3 (nugget, sill, range). If not provided, the
            variogram parameters will be determined from previously fit 
            variogram parameters at neighboring stations
        stns_rm : ndarray or str, optional
            An array of station ids or a single station id for stations that
            should not be considered neighbors for the specific point.
        
        Returns
        ----------
        tair_mean : float
            The interpolated temperature normal.
        tair_var : float
            The kriging prediction variance for the 
            temperature normal.
        '''
        
        if nnghs is None:
            # Get the nnghs to use from the optimal values at surrounding stations
            nnghs = self.__get_nnghs(pt, mth, stns_rm)
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], nnghs, load_obs=False, stns_rm=stns_rm)
        
        if vario_params is None:
            nug, psill, vrange = self.__get_vario_params(pt, mth)
        else:
            nug, psill, vrange = vario_params
        
        nghs = self.stn_slct.ngh_stns

        ngh_lon = ri.FloatSexpVector(nghs[LON])
        ngh_lat = ri.FloatSexpVector(nghs[LAT])
        ngh_elev = ri.FloatSexpVector(nghs[ELEV])
        ngh_tdi = ri.FloatSexpVector(nghs[TDI])
        ngh_lst = ri.FloatSexpVector(nghs[get_lst_varname(mth)])
        ngh_tair = ri.FloatSexpVector(nghs[get_norm_varname(mth)])
                
        pt_svp = ri.FloatSexpVector((pt[LON], pt[LAT], pt[ELEV], pt[TDI], pt[get_lst_varname(mth)]))
        
        nug = ri.FloatSexpVector([nug])
        psill = ri.FloatSexpVector([psill])
        vrange = ri.FloatSexpVector([vrange])
        
        ngh_wgt = ri.FloatSexpVector(self.stn_slct.ngh_wgt)
        
        rslt = self.r_krig_func(ngh_lon, ngh_lat, ngh_elev, ngh_tdi, ngh_lst,
                                ngh_tair, ngh_wgt, pt_svp, nug, psill, vrange)
      
        tair_mean = rslt[0]
        tair_var = rslt[1]
        bad_interp = rslt[2]
        
        if bad_interp != 0:
            print "".join(["ERROR: ", str(bad_interp), " bad interp: ", str(pt)])
        
        return tair_mean, tair_var
    

class GwrTairNorm(object):
    '''
    A class for performing monthly normal interpolation using GWR 
    to be used as a comparison to regression kriging.
    '''
    
    
    def __init__(self, stn_slct):
        
        self.stn_slct = stn_slct
        self.r_gwr_func = ri.globalenv.get('gwr_meantair')
        self.ci_critval = stats.norm.ppf(0.025)
            
    def std_err_ci(self, tair_mean, tair_var):
        
        std_err = np.sqrt(tair_var) if tair_var >= 0 else 0
        ci_r = np.abs(std_err * self.ci_critval) 
        
        return std_err, (tair_mean - ci_r, tair_mean + ci_r)
    
    def __get_nnghs(self, pt, mth, stns_rm=None):
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], DFLT_INIT_NNGHS, load_obs=False, stns_rm=stns_rm)
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_varname(mth)])
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        
        if domain_stns.size == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
        
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]
                    
        nnghs = np.int(np.round(np.average(domain_stns[get_optim_varname(mth)], weights=n_wgt)))
        
        return nnghs
        
    def gwr_predict(self, pt, mth, nnghs=None, stns_rm=None):
        
        if nnghs is None:
            # Get the nnghs to use from the optimal values at surrounding stations
            nnghs = self.__get_nnghs(pt, mth, stns_rm)
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], nnghs, load_obs=False, stns_rm=stns_rm)
        
        nghs = self.stn_slct.ngh_stns

        ngh_lon = ri.FloatSexpVector(nghs[LON])
        ngh_lat = ri.FloatSexpVector(nghs[LAT])
        ngh_elev = ri.FloatSexpVector(nghs[ELEV])
        ngh_tdi = ri.FloatSexpVector(nghs[TDI])
        ngh_lst = ri.FloatSexpVector(nghs[get_lst_varname(mth)])
        ngh_tair = ri.FloatSexpVector(nghs[get_norm_varname(mth)])
                
        pt_svp = ri.FloatSexpVector((pt[LON], pt[LAT], pt[ELEV], pt[TDI], pt[get_lst_varname(mth)]))
        
        ngh_wgt = ri.FloatSexpVector(self.stn_slct.ngh_wgt)
        
        rslt = self.r_gwr_func(ngh_lon, ngh_lat, ngh_elev, ngh_tdi, ngh_lst, ngh_tair, ngh_wgt, pt_svp)
      
        tair_mean = rslt[0]
        tair_var = rslt[1]
        bad_interp = rslt[2]
        
        if bad_interp != 0:
            print "".join(["ERROR: ", str(bad_interp), " bad interp: ", str(pt)])
        
        return tair_mean, tair_var

class StationDataWrkChk(StationSerialDataDb):
    '''
    A wrapper class around StationSerialDataDb that can be used to
    preload and cache all station observations within a lon/lat bounding box.
    Normally used to cache observations for a tile work chunk. 
    '''
    
    def __init__(self, nc_path, var_name, vcc_size=None, vcc_nelems=None, vcc_preemption=None):
        '''
        Parameters
        ----------
        nc_path : str
            File path to the netCDF4 dataset
        var_name : tuple of 2 ints, optional
            The name of main variable to be loaded.
        vcc_size : int, optional
            The netCDF4 variable chunk cache size in bytes
        vcc_nelems : int, optional
            The netCDF4 number of chunk slots in the 
            raw data chunk cache hash table.
        vcc_preemption : int, optional
            The netCDF4 var chunk cache preemption value.
        '''
        
        StationSerialDataDb.__init__(self, nc_path, var_name, vcc_size, vcc_nelems, vcc_preemption)
        self.chk_stnids = None
        self.chk_obs = None
        self.chk_deg_buf = None
        self.chk_bnds = None
    
    def set_obs(self, bnds, deg_buf=3):
        '''
        Load and cache all station observations within
        a specified lon/lat bounding box.
        
        Parameters
        ----------
        bnds : tuple
            The bounding box specified as:
            (min lat,max lat,min lon,max lon)
        deg_buf : int, optional
            A buffer (in degrees) to put around the bounding
            box. Defaults to 3. 
        '''
    
        minLat = bnds[0] - deg_buf
        maxLat = bnds[1] + deg_buf
        minLon = bnds[2] - deg_buf
        maxLon = bnds[3] + deg_buf
        
        maskStns = np.logical_and(np.logical_and(self.stns[LAT] >= minLat, self.stns[LAT] <= maxLat),
                                  np.logical_and(self.stns[LON] >= minLon, self.stns[LON] <= maxLon))
        maskStns = np.nonzero(maskStns)[0]
        
        self.chk_stnids = np.take(self.stn_ids, maskStns)
        achkObs = self.var[:, maskStns]
        self.chk_obs = {}
        for mth in np.arange(1, 13):
            self.chk_obs[mth] = np.take(achkObs, self.mth_idx[mth], axis=0)
            
        self.chk_deg_buf = deg_buf
        self.chk_bnds = bnds
        
        print "Chunk obs set. %d total stations" % self.chk_stnids.size
               
    def load_obs(self, stn_ids, mth=None):
        '''
        Load station observations.
        
        Parameters
        ----------
        stn_ids : ndarray of str or str
            A numpy array of N station ids or a single station id
        mth : int, optional
            Only load observations for a specific month
            
        Returns
        -------
        obs : ndarray
            The station observations of shape P*N where P is the
            number of days and N is the number of stations. If only
            1 station, returns a 1-D array.
        '''
        
        if mth == None:
            return StationSerialDataDb.load_obs(self, stn_ids, mth)
        else:
            mask = np.nonzero(np.in1d(self.chk_stnids, stn_ids, assume_unique=True))[0]
            
            allNgh = False
            while not allNgh:
                if mask.size != stn_ids.size:
                    print "WARNING: Increasing obs chunk..."
                    self.set_obs(self.chk_bnds, self.chk_deg_buf + 1)
                    mask = np.nonzero(np.in1d(self.chk_stnids, stn_ids, assume_unique=True))[0]
                else:
                    allNgh = True
            
            obs = np.take(self.chk_obs[mth], mask, axis=1)
                   
            return obs

def _gwr_series(model_x, predict_x, y, wgt):
    '''
    A performance-optimized method for repeatedly running a geographically weighted
    regression over a time series where the independent predictors of the regression
    remain constant but the dependent variable varies at each point in time. 
    
    Parameters
    ----------
    model_x : ndarray
        A N*M array where N is the # of observation points and M is the # of predictors
    predict_x : ndarray
        A 1-D array of M predictor values for the prediction point
    y : ndarray
        A K*N array where K is the number of observations in the time series
        and N is the number of observation points.
    wgt : ndarray
        A 1-D array of size N containing a weight for each observation point
    
    Returns
    ----------
    predict_y : ndarray
        A 1-D array of size K containing the predicted values for the prediciton
        point specified by predict_x.
    '''
    model_x = np.require(model_x, dtype=np.float64, requirements=['C', 'A', 'W', 'O'])
    predict_x = np.require(predict_x, dtype=np.float64, requirements=['C', 'A', 'W', 'O'])
    y = np.require(y, dtype=np.float64, requirements=['C', 'A', 'W', 'O'])
    wgt = np.require(wgt, dtype=np.float64, requirements=['C', 'A', 'W', 'O'])
    
    X = np.column_stack((np.ones(model_x.shape[0]), model_x))
    x = np.insert(predict_x, 0, 1)
    x.shape = (x.shape[0], 1)
    
    W = np.diag(wgt)
    
    X_t = np.transpose(X)
    
    m1 = np.linalg.inv(np.dot(np.dot(X_t, W), X))
    m2 = np.dot(X_t, W)
    m3 = np.dot(m1, m2)
    
    z = np.dot(np.transpose(x), m3)
    # Z = np.dot(X,m3)
        
    predict_y = np.inner(z, y).ravel()    
    # fit_y = np.inner(Z,y).T
    
    return predict_y  # ,fit_y
