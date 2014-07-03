'''
Classes and functions for performing Tair interpolation
'''
import numpy as np
from twx.db.station_data import LON,LAT,ELEV,TDI,LST,MEAN_OBS,NEON,OPTIM_NNGH,\
    VARIO_NUG, VARIO_PSILL, VARIO_RNG, OPTIM_NNGH_ANOM, BAD, MASK, StationSerialDataDb,STN_ID,MONTH,get_norm_varname,\
    get_optim_varname, get_krigparam_varname, get_lst_varname,\
    get_optim_anom_varname, DTYPE_NORMS, DTYPE_OPTIM, DTYPE_LST,DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM,DTYPE_ANOM_OPTIM,YEAR,\
    DTYPE_INTERP_OPTIM_ALL
from twx.interp.clibs import clib_wxTopo
import scipy.stats as stats
from twx.utils.util_ncdf import GeoNc
from netCDF4 import Dataset
from twx.interp.station_select import StationSelect
from matplotlib.mlab import griddata 
import matplotlib.pyplot as plt
import twx.utils.util_dates as utld
import mpl_toolkits.basemap as bm
#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
from twx.db.ushcn import TairAggregate
robjects.conversion.py2ri = numpy2ri
r = robjects.r
import rpy2.rinterface as ri
import os

KRIG_TREND_VARS = (LON,LAT,ELEV,LST)
GWR_TREND_VARS = (LON,LAT,ELEV,TDI,LST)
LST_TMAX = 'lst_tmax'
LST_TMIN = 'lst_tmin'

MIN_RADIUS_INFLUENCE = 10
DFLT_INIT_NNGHS = 100

R_LOADED = False

def _init_interp_R_env():
    
    global R_LOADED

    if not R_LOADED:
    
        print "Loading R environment for interp_tair..."
        
        #get system path to twx
        twx_path = os.path.split(os.path.split(__file__)[0])[0]
        #get system path to interp.R
        rsrc_path = os.path.join(twx_path,'lib','rpy','interp.R')
        
        r.source(rsrc_path)
        #Set trend formula
        ri.globalenv['FORMULA'] =  ri.globalenv.get("build_formula")(ri.StrSexpVector(["tair"]),ri.StrSexpVector(KRIG_TREND_VARS))
        
        R_LOADED = True
        

class SVD_Exception(Exception):
    pass

class PredictorGrids():

    def __init__(self,ncFpaths,interpOrders=None):
        
        self.ncDs = {}
        self.ncData = {}
        self.xGrid = None
        self.yGrid = None
        
        interpOrders = np.zeros(len(ncFpaths)) if interpOrders is None else interpOrders 
        
        for fpath,interpOrder in zip(ncFpaths,interpOrders):
            
            geoNc = GeoNc(Dataset(fpath))
            varKeys =  np.array(geoNc.ds.variables.keys())
            ncVarname = varKeys[np.logical_and(varKeys != 'lon',varKeys != 'lat')][0]
            self.ncDs[ncVarname] = geoNc
            
            if interpOrder == 1:
                a = geoNc.ds.variables[ncVarname][:]
                aflip = np.flipud(a)
                #aflip = aflip.astype(np.float)
                self.ncData[ncVarname] = aflip
                
                if self.xGrid == None:
                    self.xGrid = geoNc.ds.variables['lon'][:]
                    self.yGrid = np.sort(geoNc.ds.variables['lat'][:])
                        
    def setPtValues(self,aPt,chgLatLon=True):
                
        chged = False
        for varname,geoNc in self.ncDs.items():
            
            if chgLatLon or not self.ncData.has_key(varname):
            
                row,col,gridlon,gridlat =  geoNc.get_row_col(aPt[LON],aPt[LAT])
                aPt[varname] = geoNc.ds.variables[varname][row,col]
                
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
            
def check_tmin_tmax_valid(tmin,tmax,mean_tmin,mean_tmax,ci_tmin,ci_tmax):
    
    tmin,tmax,ninvalid = tmin_tmax_fixer(tmin, tmax)
    
    if ninvalid > 0:
        
        new_mean_tmin = np.mean(tmin)
        new_mean_tmax = np.mean(tmax)
        
        dif_mean_tmin = new_mean_tmin - mean_tmin
        dif_mean_tmax = new_mean_tmax - mean_tmax
        
        ci_tmin = (ci_tmin[0]+dif_mean_tmin,ci_tmin[1]+dif_mean_tmin)
        ci_tmax = (ci_tmax[0]+dif_mean_tmax,ci_tmax[1]+dif_mean_tmax)
        mean_tmin = new_mean_tmin
        mean_tmax = new_mean_tmax
        
    return tmin,tmax,mean_tmin,mean_tmax,ci_tmin,ci_tmax,ninvalid
    

def tmin_tmax_fixer(tmin,tmax,tail=15):

    invalid_days = np.nonzero(tmin>=tmax)[0]
    
    if invalid_days.size > 0:
        
        tmin = np.copy(tmin)
        tmax = np.copy(tmax)
        for x in invalid_days:
            
            tavg = (tmin[x]+tmax[x])/2.0
            start = x-tail
            end = x+tail+1
            if start < 0: start = 0
            if end > tmin.size: end = tmin.size
            tmin_win = tmin[start:end]
            tmax_win = tmax[start:end]
            mask = tmin_win<tmax_win
            tmin_win = tmin_win[mask]
            tmax_win = tmax_win[mask]
            if tmin_win.size == 0:
                raise Exception('No valid tmin/tmax in window')
            tdir_half = np.mean(tmax_win-tmin_win,dtype=np.float64)/2.0
            tmin[x] = tavg - tdir_half
            tmax[x] = tavg + tdir_half
        
    return tmin,tmax,invalid_days.size


def build_empty_pt():
    
    ptDtype = [(LON, np.float64), (LAT, np.float64), (ELEV, np.float64),(TDI, np.float64),(NEON, np.float64),(MASK,np.float64)]
    ptDtype.extend(DTYPE_NORMS)
    ptDtype.extend(DTYPE_OPTIM)
    ptDtype.extend(DTYPE_LST)
    ptDtype.extend(DTYPE_ANOM_OPTIM)
    ptDtype.extend([("tmin%02d"%mth,np.float64) for mth in np.arange(1,13)]) 
    ptDtype.extend([("tmax%02d"%mth,np.float64) for mth in np.arange(1,13)]) 
    a_pt = np.empty(1, dtype=ptDtype)
    
    return a_pt[0]

class GwrTairAnom(object):
    
    def __init__(self,stn_slct,days):
        
        self.stn_slct = stn_slct
        self.days = days
        self.aclib = clib_wxTopo()
        
        mthlyPredictors = {}
        predictors = np.array(GWR_TREND_VARS,dtype="<S16")
        mthlyPredictors[None] = predictors
        
        for mth in np.arange(1,13):
            
            mthP = np.copy(predictors)
            mthP[mthP==LST] = get_lst_varname(mth)
            mthlyPredictors[mth] = mthP
        
        self.mthlyPredictors = mthlyPredictors
        
        mthMasks = {}
        mthIdx = {}
        for mth in np.arange(1,13):
            mthMasks[mth] = days[MONTH] == mth
            mthIdx[mth] = np.nonzero(mthMasks[mth])[0]
        mthMasks[None] = np.ones(days.size,dtype=np.bool)
        mthIdx[None] = np.nonzero(mthMasks[None])[0]
        self.mthMasks = mthMasks
        self.mthIdx = mthIdx
    
    def __get_nnghs(self,pt,mth,stns_rm=None):
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], DFLT_INIT_NNGHS, load_obs=False, stns_rm=stns_rm)
        
        fin_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_anom_varname(mth)])
        
        if np.sum(fin_mask) == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
    
        p_stns = self.stn_slct.ngh_stns[fin_mask]
        p_wgt = self.stn_slct.ngh_wgt[fin_mask]
        
        nnghs = np.int(np.round(np.average(p_stns[get_optim_anom_varname(mth)],weights=p_wgt)))

        return nnghs
    
    def gwr_mth(self,pt,mth,nnghs=None,stns_rm=None):
        
        if nnghs == None:
            #Get the nnghs to use from the optimal values at surrounding stations
            nnghs = self.__get_nnghs(pt,mth,stns_rm)
        
        self.stn_slct.set_ngh_stns(pt[LAT],pt[LON],nnghs,load_obs=True,stns_rm=stns_rm,obs_mth=mth)
        
        ngh_obs = self.stn_slct.ngh_obs
        #ngh_obs = np.take(self.stn_slct.ngh_obs, self.mthIdx[mth], 0)
        ngh_stns = self.stn_slct.ngh_stns
        ngh_wgt = self.stn_slct.ngh_wgt
           
        ngh_obs_cntr = ngh_obs - ngh_stns[get_norm_varname(mth)]
        
        #Perform a GWR for each day
        X = [ngh_stns[avar] for avar in self.mthlyPredictors[mth]]
        X.insert(0,np.ones(ngh_stns.size))
        X = np.column_stack(X)
        
        x = [pt[avar] for avar in self.mthlyPredictors[mth]]
        x.insert(0,1)
        x = np.array(x)
                
        interp_anom,fit_anom = self.aclib.repRegress(X, ngh_obs_cntr, x, ngh_wgt)
    
        #Perform IDW of GWR residuals
#        resids = ngh_obs_cntr-fit_anom
#        dists = np.copy(self.stn_slct.ngh_dists)        
#        dists = np.round(dists)
#        idw_wgts = np.ones(dists.size)
#        dists_idw = dists - MIN_RADIUS_INFLUENCE
#        dists_idw_mask = np.nonzero(dists_idw > 0)[0]
#        idw_wgts[dists_idw_mask] = 1.0/(np.take(dists_idw,dists_idw_mask)**2)
#        
#        #interp_resids = np.average(resids, axis=1, weights=1.0/(dists**2))
#        interp_resids = np.average(resids, axis=1, weights=idw_wgts)   
#        
#        
##        plt.plot(runningMean(interp_resids, 30))
##        plt.show()
#        
#        interp_anom = interp_anom+interp_resids 
        
        interp_vals = interp_anom + pt[get_norm_varname(mth)]
                
        return interp_vals 

class GwrTairAnomR(GwrTairAnom):
    
    def __init__(self,stn_slct,days):
        
        GwrTairAnom.__init__(self, stn_slct, days)
        
    
    def gwr_mth(self,pt,mth,nnghs=None,stns_rm=None):
        
        if nnghs == None:
            #Get the nnghs to use from the optimal values at surrounding stations
            nnghs = self._GwrTairAnom__get_nnghs(pt,mth,stns_rm)
        
        self.stn_slct.set_ngh_stns(pt[LAT],pt[LON],nnghs,load_obs=True,stns_rm=stns_rm)
        
        ngh_obs = np.take(self.stn_slct.ngh_obs, self.mthIdx[mth], 0)
        ngh_stns = self.stn_slct.ngh_stns
        ngh_wgt = self.stn_slct.ngh_wgt
        a_pt = np.array([pt[LON],pt[LAT],pt[ELEV],pt[TDI],pt[get_lst_varname(mth)]])
           
        ngh_obs_cntr = ngh_obs - ngh_stns[get_norm_varname(mth)]
        
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
        fit_anom = np.reshape(fit_anom, (nrow,ncol), order='F')
        
        interp_anom = np.array(rslt.rx('pt_anom'))
        
        #Perform IDW of GWR residuals
        resids = ngh_obs_cntr-fit_anom
        dists = np.copy(self.stn_slct.ngh_dists)
        dists[dists==0] = 0.01
        interp_resids = np.average(resids, axis=1, weights=1.0/(dists**2))    
        interp_anom = interp_anom+interp_resids
        
        interp_vals = interp_anom + pt[get_norm_varname(mth)]
                
        return interp_vals 

class GwrTairAnomBlank(GwrTairAnom):
    
    def __init__(self,stn_slct,days):
        
        GwrTairAnom.__init__(self, stn_slct, days)
        self.blank_vals = {}
        for mth in np.arange(1,13):
            self.blank_vals[mth] = np.zeros(self.mthIdx[mth].size)
        
    
    def gwr_mth(self,pt,mth,nnghs=None,stns_rm=None):  
        return self.blank_vals[mth] 
      
class InterpTair(object):
    
    def __init__(self,krig_tair,gwr_tair):
        
        self.krig_tair = krig_tair
        self.gwr_tair = gwr_tair
        self.mthMasks = gwr_tair.mthMasks
        self.ndays = gwr_tair.days.size
        
    def interp(self,pt,stns_rm=None):
        
        tair_daily = np.zeros(self.ndays)
        tair_norms = np.zeros(12)
        tair_se = np.zeros(12)
                
        for mth in np.arange(1,13):
            
            tair_mean,tair_var = self.krig_tair.krig(pt,mth,stns_rm=stns_rm)
            std_err,ci = self.krig_tair.std_err_ci(tair_mean,tair_var)
            pt[get_norm_varname(mth)] = tair_mean
            tair_norms[mth-1] = tair_mean
            tair_se[mth-1] = std_err
    
            tair_daily[self.mthMasks[mth]] = self.gwr_tair.gwr_mth(pt,mth,stns_rm=stns_rm)
        
        return tair_daily,tair_norms,tair_se

class PtInterpTair(object):
    
    def __init__(self,stn_da_tmin,stn_da_tmax,auxFpaths=None,interpOrders=None,norms_only=False):    

        self.days = stn_da_tmin.days
        self.stn_da_tmin = stn_da_tmin
        self.stn_da_tmax = stn_da_tmax
        
        #Masks for calculating monthly norms after daily Tmin/Tmax values had to be adjusted due to Tmin >= Tmax
        
        self.daysNormMask = np.nonzero(np.logical_and(self.days[YEAR] >= 1981,self.days[YEAR] <= 2010))[0] 
        daysNorm = self.days[self.daysNormMask]
        
        uYrs = np.unique(daysNorm[YEAR])
        self.yr_mths = utld.get_mth_metadata(uYrs[0],uYrs[-1])
        
        self.yrMthsMasks = []
        for aYr in uYrs:
            for aMth in np.arange(1,13):
                self.yrMthsMasks.append(np.nonzero(np.logical_and(daysNorm[YEAR]==aYr,daysNorm[MONTH]==aMth))[0])
        
        self.mthMasks = []
        for mth in np.arange(1,13):
            self.mthMasks.append(np.nonzero(self.yr_mths[MONTH]==mth)[0])
        
        mask_stns_tmin = np.isnan(stn_da_tmin.stns[BAD]) 
        mask_stns_tmax = np.isnan(stn_da_tmax.stns[BAD])
        
        stn_slct_tmin = StationSelect(stn_da_tmin, mask_stns_tmin)
        stn_slct_tmax = StationSelect(stn_da_tmax, mask_stns_tmax)
        
        domain_stns_tmin = stn_da_tmin.stns[np.logical_and(mask_stns_tmin,np.isfinite(stn_da_tmin.stns[MASK]))]
        domain_stns_tmax = stn_da_tmax.stns[np.logical_and(mask_stns_tmax,np.isfinite(stn_da_tmax.stns[MASK]))]
        self.nnghparams_tmin = get_rgn_nnghs_dict(domain_stns_tmin)
        self.nnghparams_tmax = get_rgn_nnghs_dict(domain_stns_tmax)
        
        krig_tmin = KrigTair(stn_slct_tmin)
        krig_tmax = KrigTair(stn_slct_tmax)
        
        if norms_only:
            gwr_tmin = GwrTairAnomBlank(stn_slct_tmin, self.days)
            gwr_tmax = GwrTairAnomBlank(stn_slct_tmax, self.days)
        else:
            gwr_tmin = GwrTairAnom(stn_slct_tmin, self.days)
            gwr_tmax = GwrTairAnom(stn_slct_tmax, self.days)
                
        self.interp_tmin = InterpTair(krig_tmin, gwr_tmin)
        self.interp_tmax = InterpTair(krig_tmax, gwr_tmax)
        
        if auxFpaths is not None:
            self.pGrids = PredictorGrids(auxFpaths,interpOrders)
        
        self.a_pt = build_empty_pt()
        
    
    def interpLonLatPt(self,lon,lat,fixInvalid=True,chgLatLon=True,stns_rm=None,elev=None):
        
        self.a_pt[LON] = lon
        self.a_pt[LAT] = lat
        self.pGrids.setPtValues(self.a_pt,chgLatLon)
        if elev is not None:
            self.a_pt[ELEV] = elev
        
        if self.a_pt[MASK] == 0:
            raise Exception('Point is outside interpolation region')
        
        return self.interpPt(fixInvalid,stns_rm)
    
    def interpPt(self,fixInvalid=True,stns_rm=None):
        
        #Set the monthly lst values and optim nnghs on the point
        for mth in np.arange(1,13):
            
            self.a_pt[get_lst_varname(mth)] = self.a_pt["tmin%02d"%mth]
            self.a_pt[get_optim_varname(mth)],self.a_pt[get_optim_anom_varname(mth)] = self.nnghparams_tmin[self.a_pt[NEON]][mth]

        #Perform Tmin interpolation
        tmin_dly, tmin_norms, tmin_se = self.interp_tmin.interp(self.a_pt,stns_rm=stns_rm)
        
        #Set the monthly lst values and optim nnghs on the point
        for mth in np.arange(1,13):
            
            self.a_pt[get_lst_varname(mth)] = self.a_pt["tmax%02d"%mth]
            self.a_pt[get_optim_varname(mth)],self.a_pt[get_optim_anom_varname(mth)] = self.nnghparams_tmax[self.a_pt[NEON]][mth]
        
        #Perform Tmax interpolation
        tmax_dly, tmax_norms, tmax_se = self.interp_tmax.interp(self.a_pt,stns_rm=stns_rm)
        
        ninvalid = 0
        
        if fixInvalid:
            
            tmin_dly,tmax_dly,ninvalid = tmin_tmax_fixer(tmin_dly, tmax_dly)
    
            if ninvalid > 0:
                
                tmin_dly_norm = np.take(tmin_dly, self.daysNormMask)
                tmax_dly_norm = np.take(tmax_dly, self.daysNormMask)
                tmin_mthly = np.array([np.mean(np.take(tmin_dly_norm,amask)) for amask in self.yrMthsMasks])
                tmax_mthly = np.array([np.mean(np.take(tmax_dly_norm,amask)) for amask in self.yrMthsMasks])
                tmin_norms = np.array([np.mean(np.take(tmin_mthly,amask)) for amask in self.mthMasks])
                tmax_norms = np.array([np.mean(np.take(tmax_mthly,amask)) for amask in self.mthMasks])
        
        return tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid

def get_rgn_nnghs_dict(stns):
    
    rgns = np.unique(stns[NEON][np.isfinite(stns[NEON])])
    
    nnghsAll = {}
    
    for rgn in rgns:
        
        rgn_mask = stns[NEON]==rgn
        nnghsRgn = {}
        
        for mth in np.arange(1,13):
            nnghsRgn[mth] = (stns[get_optim_varname(mth)][rgn_mask][0],stns[get_optim_anom_varname(mth)][rgn_mask][0])
        
        nnghsAll[rgn] = nnghsRgn
        
    return nnghsAll
    
class BuildKrigParams(object):
    
    def __init__(self,stn_slct):
        
        self.stn_slct = stn_slct
        self.r_kparam_func = ri.globalenv.get('get_vario_params')
        
    def get_krig_params(self,pt,mth,rm_stnid=None):
        
        #First determine the nnghs to use based on smoothed weighted average of 
        #the optimal nnghs at each station point.
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], DFLT_INIT_NNGHS, load_obs=False)
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_varname(mth)])
        
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        
        if domain_stns.size == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
        
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]
                    
        nnghs = np.int(np.round(np.average(domain_stns[get_optim_varname(mth)],weights=n_wgt)))
    
        #Now use the optimal nnghs to get the krig params for this mth
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
        
        rslt = self.r_kparam_func(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,ngh_dists)
        nug = rslt[0]
        psill = rslt[1]
        rng = rslt[2]
                
        return nug,psill,rng

class KrigTairAll(object):
    '''
    Class to perform moving window variogram fitting and regression kriging
    of monthly normals all in one step. Uses the R gstat package. This is
    mainly used in the optimization of the local number of neighboring
    stations bandwidth to be used for moving window regression kriging in
    each month.
    '''
    
    def __init__(self,stn_slct):
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
        
    def krigall(self,pt,nnghs,stns_rm=None):
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
        
        self.stn_slct.set_ngh_stns(pt[LAT],pt[LON],nnghs,load_obs=False,stns_rm=stns_rm)
                
        nghs = self.stn_slct.ngh_stns
        ngh_lon = ri.FloatSexpVector(nghs[LON])
        ngh_lat = ri.FloatSexpVector(nghs[LAT])
        ngh_elev = ri.FloatSexpVector(nghs[ELEV])
        ngh_tdi = ri.FloatSexpVector(nghs[TDI])
        ngh_wgt = ri.FloatSexpVector(self.stn_slct.ngh_wgt)
        ngh_dists = ri.FloatSexpVector(self.stn_slct.ngh_dists)
        
        interp_norms = np.zeros(12)
        
        for mth in np.arange(1,13):
            
            ngh_lst = ri.FloatSexpVector(nghs[get_lst_varname(mth)])
            ngh_tair = ri.FloatSexpVector(nghs[get_norm_varname(mth)])
            pt_svp = ri.FloatSexpVector((pt[LON],pt[LAT],pt[ELEV],pt[TDI],pt[get_lst_varname(mth)]))
            rslt = self.r_func(pt_svp,ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,ngh_dists)
            
            interp_norms[mth-1] = rslt[0]
                            
        return interp_norms
                    
class KrigTair(object):
    
    def __init__(self,stn_slct):
        
        self.stn_slct = stn_slct
        self.r_krig_func = ri.globalenv.get('krig_meantair')
        self.ci_critval = stats.norm.ppf(0.025)
            
    def std_err_ci(self,tair_mean,tair_var):
        
        std_err = np.sqrt(tair_var) if tair_var >= 0 else 0
        ci_r = np.abs(std_err*self.ci_critval) 
        
        return std_err,(tair_mean-ci_r,tair_mean+ci_r)
    
    def __get_nnghs(self,pt,mth,stns_rm=None):
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], DFLT_INIT_NNGHS, load_obs=False, stns_rm=stns_rm)
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_varname(mth)])
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        
        if domain_stns.size == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
        
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]
                    
        nnghs = np.int(np.round(np.average(domain_stns[get_optim_varname(mth)],weights=n_wgt)))
        
        return nnghs
    
    def __get_vario_params(self,pt,mth):
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_krigparam_varname(mth, VARIO_NUG)])
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        
        if domain_stns.size == 0:
            raise Exception("Cannot determine variogram params!")
        
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]
        
        nug = np.average(domain_stns[get_krigparam_varname(mth, VARIO_NUG)],weights=n_wgt)
        psill = np.average(domain_stns[get_krigparam_varname(mth, VARIO_PSILL)],weights=n_wgt) 
        vrange = np.average(domain_stns[get_krigparam_varname(mth, VARIO_RNG)],weights=n_wgt)  
                
        return nug,psill,vrange
    
    def krig(self,pt,mth,nnghs=None,vario_params=None,stns_rm=None):
        
        if nnghs is None:
            #Get the nnghs to use from the optimal values at surrounding stations
            nnghs = self.__get_nnghs(pt,mth,stns_rm)
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], nnghs, load_obs=False, stns_rm=stns_rm)
        
        if vario_params is None:
            #should variogram params be smoothed with DFLT_INIT_NNGHS or with the optimized nnghs?
            nug,psill,vrange = self.__get_vario_params(pt, mth)
        else:
            nug,psill,vrange = vario_params
        
        nghs = self.stn_slct.ngh_stns

        ngh_lon = ri.FloatSexpVector(nghs[LON])
        ngh_lat = ri.FloatSexpVector(nghs[LAT])
        ngh_elev = ri.FloatSexpVector(nghs[ELEV])
        ngh_tdi = ri.FloatSexpVector(nghs[TDI])
        ngh_lst = ri.FloatSexpVector(nghs[get_lst_varname(mth)])
        ngh_tair = ri.FloatSexpVector(nghs[get_norm_varname(mth)])
                
        pt_svp = ri.FloatSexpVector((pt[LON],pt[LAT],pt[ELEV],pt[TDI],pt[get_lst_varname(mth)]))
        
        nug = ri.FloatSexpVector([nug])
        psill = ri.FloatSexpVector([psill])
        vrange = ri.FloatSexpVector([vrange])
        
        ngh_wgt = ri.FloatSexpVector(self.stn_slct.ngh_wgt)
        
        rslt = self.r_krig_func(ngh_lon, ngh_lat, ngh_elev, ngh_tdi, ngh_lst, ngh_tair, ngh_wgt, pt_svp, nug, psill, vrange)
      
        tair_mean = rslt[0]
        tair_var = rslt[1]
        bad_interp = rslt[2]
        
        if bad_interp != 0:
            print "".join(["ERROR: ",str(bad_interp)," bad interp: ",str(pt)])
        
        return tair_mean,tair_var
    

class GwrTairNorm(object):
    '''
    A class for performing monthly normal interpolation using GWR 
    to be used as a comparison to regression kriging.
    '''
    
    
    def __init__(self,stn_slct):
        
        self.stn_slct = stn_slct
        self.r_gwr_func = ri.globalenv.get('gwr_meantair')
        self.ci_critval = stats.norm.ppf(0.025)
            
    def std_err_ci(self,tair_mean,tair_var):
        
        std_err = np.sqrt(tair_var) if tair_var >= 0 else 0
        ci_r = np.abs(std_err*self.ci_critval) 
        
        return std_err,(tair_mean-ci_r,tair_mean+ci_r)
    
    def __get_nnghs(self,pt,mth,stns_rm=None):
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], DFLT_INIT_NNGHS, load_obs=False, stns_rm=stns_rm)
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_varname(mth)])
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        
        if domain_stns.size == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
        
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]
                    
        nnghs = np.int(np.round(np.average(domain_stns[get_optim_varname(mth)],weights=n_wgt)))
        
        return nnghs
        
    def gwr_predict(self,pt,mth,nnghs=None,stns_rm=None):
        
        if nnghs is None:
            #Get the nnghs to use from the optimal values at surrounding stations
            nnghs = self.__get_nnghs(pt,mth,stns_rm)
        
        self.stn_slct.set_ngh_stns(pt[LAT], pt[LON], nnghs, load_obs=False, stns_rm=stns_rm)
        
        nghs = self.stn_slct.ngh_stns

        ngh_lon = ri.FloatSexpVector(nghs[LON])
        ngh_lat = ri.FloatSexpVector(nghs[LAT])
        ngh_elev = ri.FloatSexpVector(nghs[ELEV])
        ngh_tdi = ri.FloatSexpVector(nghs[TDI])
        ngh_lst = ri.FloatSexpVector(nghs[get_lst_varname(mth)])
        ngh_tair = ri.FloatSexpVector(nghs[get_norm_varname(mth)])
                
        pt_svp = ri.FloatSexpVector((pt[LON],pt[LAT],pt[ELEV],pt[TDI],pt[get_lst_varname(mth)]))
        
        ngh_wgt = ri.FloatSexpVector(self.stn_slct.ngh_wgt)
        
        rslt = self.r_gwr_func(ngh_lon, ngh_lat, ngh_elev, ngh_tdi, ngh_lst, ngh_tair, ngh_wgt, pt_svp)
      
        tair_mean = rslt[0]
        tair_var = rslt[1]
        bad_interp = rslt[2]
        
        if bad_interp != 0:
            print "".join(["ERROR: ",str(bad_interp)," bad interp: ",str(pt)])
        
        return tair_mean,tair_var

class StationDataWrkChk(StationSerialDataDb):
    '''
    A station_data class for accessing stations and observations from a single variable infilled netcdf weather station database.
    '''
    
    def __init__(self, nc_path, var_name,vcc_size=None,vcc_nelems=None,vcc_preemption=None,stn_dtype=DTYPE_INTERP_OPTIM_ALL):
        
        StationSerialDataDb.__init__(self, nc_path, var_name, vcc_size, vcc_nelems, vcc_preemption, stn_dtype)
        self.chkStnIds = None
        self.chkObs = None
        self.chkDegBuf = None
        self.chkBnds = None
    
    def set_obs(self,bnds,degBuf=3):
    
        minLat = bnds[0] - degBuf
        maxLat = bnds[1] + degBuf
        minLon = bnds[2] - degBuf
        maxLon = bnds[3] + degBuf
        
        maskStns = np.logical_and(np.logical_and(self.stns[LAT] >= minLat,self.stns[LAT] <= maxLat),
                                  np.logical_and(self.stns[LON] >= minLon,self.stns[LON] <= maxLon))
        maskStns = np.nonzero(maskStns)[0]
        
        self.chkStnIds = np.take(self.stn_ids,maskStns)
        achkObs = self.var[:,maskStns]
        self.chkObs = {}
        for mth in np.arange(1,13):
            self.chkObs[mth] = np.take(achkObs, self.mthIdx[mth], axis=0)
            
        self.chkDegBuf = degBuf
        self.chkBnds = bnds
        
        print "Chunk obs set. %d total stations"%self.chkStnIds.size
               
    def load_obs(self, stn_ids, mth = None):
        
        if mth == None:
            return StationSerialDataDb.load_obs(self, stn_ids, mth)
        else:
            mask = np.nonzero(np.in1d(self.chkStnIds, stn_ids, assume_unique=True))[0]
            
            allNgh = False
            while not allNgh:
                if mask.size != stn_ids.size:
                    print "WARNING: Increasing obs chunk..."
                    self.set_obs(self.chkBnds,self.chkDegBuf + 1)
                    mask = np.nonzero(np.in1d(self.chkStnIds, stn_ids, assume_unique=True))[0]
                else:
                    allNgh = True
            
            obs = np.take(self.chkObs[mth], mask, axis=1)
                   
            return obs

def buildDefaultPtInterp(norms_only=False):
    #Normal
    stndaTmin = StationSerialDataDb('/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc', 'tmin')
    stndaTmax = StationSerialDataDb('/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc', 'tmax')
    #No LST
#    stndaTmin = StationSerialDataDb('/projects/daymet2/station_data/infill/serial_nolst/serial_tmin.nc', 'tmin')
#    stndaTmax = StationSerialDataDb('/projects/daymet2/station_data/infill/serial_nolst/serial_tmax.nc', 'tmax')
    #No homogenization
#    stndaTmin = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmin.nc', 'tmin')
#    stndaTmax = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmax.nc', 'tmax')
    
    gridPath = '/projects/daymet2/dem/interp_grids/conus/ncdf/'
    auxFpaths = ["".join([gridPath,'fnl_elev.nc']),
                 "".join([gridPath,'fnl_tdi.nc']),
                 "".join([gridPath,'fnl_climdiv.nc']),
                 "".join([gridPath,'fnl_mask.nc'])]
    auxFpaths.extend(["".join([gridPath,'fnl_lst_tmin%02d.nc'%mth]) for mth in np.arange(1,13)])
    auxFpaths.extend(["".join([gridPath,'fnl_lst_tmax%02d.nc'%mth]) for mth in np.arange(1,13)])
    
    interpOrders = np.zeros(len(auxFpaths))
    #interpOrders = np.ones(len(auxFpaths))
    #interpOrders[2] = 0 #climdiv
    #interpOrders[3] = 0 #mask
    #interpOrders = None
    ptInterp = PtInterpTair(stndaTmin,stndaTmax,auxFpaths,interpOrders,norms_only) 
    return ptInterp