'''
Classes and functions for performing Tair interpolation
'''
import numpy as np
from twx.db.station_data import LON,LAT,ELEV,TDI,LST,MEAN_OBS,NEON,OPTIM_NNGH,\
    VARIO_NUG, VARIO_PSILL, VARIO_RNG, OPTIM_NNGH_ANOM, BAD, MASK, station_data_infill,STN_ID,MONTH,get_norm_varname,\
    get_optim_varname, get_krigparam_varname, get_lst_varname,\
    get_optim_anom_varname, DTYPE_NORMS, DTYPE_OPTIM, DTYPE_LST,DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM,DTYPE_ANOM_OPTIM,YEAR
from twx.interp.clibs import clib_wxTopo
import scipy.stats as stats
from twx.utils.util_ncdf import GeoNc
from netCDF4 import Dataset
from twx.interp.station_select import station_select
from matplotlib.mlab import griddata 
import matplotlib.pyplot as plt
import twx.utils.util_dates as utld
#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
from twx.db.ushcn import TairAggregate
robjects.conversion.py2ri = numpy2ri
r = robjects.r
import rpy2.rinterface as ri

KRIG_TREND_VARS = (LON,LAT,ELEV,LST)
GWR_TREND_VARS = (LON,LAT,ELEV,TDI,LST)
LST_TMAX = 'lst_tmax'
LST_TMIN = 'lst_tmin'

DFLT_INIT_NNGHS = 100

class SVD_Exception(Exception):
    pass

class PredictorGrids():

    def __init__(self,ncFpaths):
        
        self.ncDs = {}
        
        for fpath in ncFpaths:
            
            geoNc = GeoNc(Dataset(fpath))
            varKeys =  np.array(geoNc.ds.variables.keys())
            ncVarname = varKeys[np.logical_and(varKeys != 'lon',varKeys != 'lat')][0]
            self.ncDs[ncVarname] = geoNc
        
    def setPtValues(self,aPt,chgLatLon=True):
                
        chged = False
        for varname,geoNc in self.ncDs.items():
            
            row,col,gridlon,gridlat =  geoNc.getRowCol(aPt[LON],aPt[LAT])
            aPt[varname] = geoNc.ds.variables[varname][row,col]
            
            if chgLatLon and not chged:
                aPt[LON] = gridlon
                aPt[LAT] = gridlat
                chged = True
            
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

class GwrPcaTair(object):

    def __init__(self,stn_slct,path_clib,days,set_pt=True):
        
        self.stn_slct = stn_slct
        self.aclib = clib_wxTopo(path_clib)        
        self.set_pt = set_pt
        self.pt = None
        
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
        
        
    def setup_for_pt(self,pt,rm_stnid=None):
        pass
        
    def gwr_pca(self,mth=None):
        
        #ngh_obs = self.stn_slct.ngh_obs[self.mthMasks[mth],:]
        #ngh_obs = np.take(self.stn_slct.ngh_obs, self.mthIdx[mth], 0)
        ngh_obs = self.stn_slct.ngh_obs
        ngh_stns = self.stn_slct.ngh_stns
        ngh_wgt = self.stn_slct.ngh_wgt
           
        ngh_obs_cntr = ngh_obs - ngh_stns[get_norm_varname(mth)]
        
        X = [ngh_stns[avar] for avar in self.mthlyPredictors[mth]]
        X.insert(0,np.ones(ngh_stns.size))
        X = np.column_stack(X)
        
        x = [self.pt[avar] for avar in self.mthlyPredictors[mth]]
        x.insert(0,1)
        x = np.array(x)
        
#        X = np.column_stack((np.ones(ngh_stns.size),ngh_stns[LON],ngh_stns[LAT],ngh_stns[ELEV],ngh_stns[TDI],ngh_stns[LST]))
#        x = np.array([1.,self.pt[LON],self.pt[LAT],self.pt[ELEV],self.pt[TDI],self.pt[LST]])

        
        interp_anom = self.aclib.repRegress(X, ngh_obs_cntr, x, ngh_wgt)
        #interp_anom = np.average(ngh_obs_cntr,axis=1,weights=ngh_wgt)
        interp_vals = interp_anom + self.pt[get_norm_varname(mth)]
                
        return interp_vals
    
#    def gwr_pca(self):
#        
#        ngh_obs = self.stn_slct.ngh_obs
#        ngh_stns = self.stn_slct.ngh_stns
#        ngh_wgt = self.stn_slct.ngh_wgt
#           
#        ngh_obs_cntr = ngh_obs - ngh_stns[MEAN_OBS]
#    
#        pt = np.array([self.pt[LON],self.pt[LAT],self.pt[ELEV],self.pt[TDI],self.pt[LST]])
#
#        rslt = r.gwr_anomaly(robjects.FloatVector(ngh_stns[LON]),
#                                  robjects.FloatVector(ngh_stns[LAT]),
#                                  robjects.FloatVector(ngh_stns[ELEV]),
#                                  robjects.FloatVector(ngh_stns[TDI]),
#                                  robjects.FloatVector(ngh_stns[LST]),
#                                  robjects.FloatVector(ngh_wgt),
#                                  robjects.Matrix(ngh_obs_cntr),
#                                  robjects.FloatVector(pt))
#        
#        interp_vals = np.array(rslt).ravel() + self.pt[MEAN_OBS] 
#                
#        return interp_vals
    
class GwrPcaTairDynamic(GwrPcaTair):
    
    def __init__(self,stn_slct,path_clib,days,set_pt=True):
        
        GwrPcaTair.__init__(self, stn_slct, path_clib, days, set_pt)
        self.mth = None        
            
    def setup_for_pt(self,pt,rm_stnid=None):
                
        if self.set_pt:
            self.stn_slct.set_pt(pt[LAT],pt[LON],stns_rm=rm_stnid)
        self.pt = pt
        
        min_nngh = pt[get_optim_anom_varname(self.mth)]
        max_nngh = min_nngh+np.round(min_nngh*0.20) 
            
        #self.stn_slct.set_params(DFLT_INIT_NNGHS,DFLT_INIT_NNGHS,1.0)
        self.stn_slct.set_nngh_stns(DFLT_INIT_NNGHS,load_obs=False)
        #self.stn_slct.set_ngh_stns(load_obs=False)
        
        fin_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_anom_varname(self.mth)])
        
        if np.sum(fin_mask) == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
    
        p_stns = self.stn_slct.ngh_stns[fin_mask]
        p_wgt = self.stn_slct.ngh_wgt[fin_mask]/np.sum(self.stn_slct.ngh_wgt[fin_mask])
        
        min_nngh = np.int(np.round(np.average(p_stns[get_optim_anom_varname(self.mth)],weights=p_wgt)))
        max_nngh = np.int(min_nngh+np.round(min_nngh*0.20))
        
        self.stn_slct.set_params(min_nngh, max_nngh,1.0)
        self.stn_slct.set_ngh_stns(load_obs=True,obs_mth=self.mth)        
        
class GwrPcaTairStatic(GwrPcaTair):
    
    def __init__(self,stn_slct,path_clib,days,min_nngh,max_nngh,gw_p,set_pt=True):
        
        GwrPcaTair.__init__(self, stn_slct, path_clib,days, set_pt)
        self.min_nngh = min_nngh
        self.max_nngh = max_nngh
        self.gw_p = gw_p        
    
    def reset_params(self,min_nngh,max_nngh,gw_p):
        
        self.min_nngh = min_nngh
        self.max_nngh = max_nngh
        self.gw_p = gw_p
        
    def setup_for_pt(self,pt,rm_stnid=None):
                
        if self.set_pt:
            self.stn_slct.set_pt(pt[LAT],pt[LON],stns_rm=rm_stnid)
        self.pt = pt
        
        self.stn_slct.set_params(self.min_nngh, self.max_nngh,self.gw_p)
        self.stn_slct.set_ngh_stns(load_obs=True)
        
    def gwr_pca(self,mth=None):
        
        ngh_obs = np.take(self.stn_slct.ngh_obs, self.mthIdx[mth], 0)
        ngh_stns = self.stn_slct.ngh_stns
        ngh_wgt = self.stn_slct.ngh_wgt
           
        ngh_obs_cntr = ngh_obs - ngh_stns[get_norm_varname(mth)]
        
        X = [ngh_stns[avar] for avar in self.mthlyPredictors[mth]]
        X.insert(0,np.ones(ngh_stns.size))
        X = np.column_stack(X)
        
        x = [self.pt[avar] for avar in self.mthlyPredictors[mth]]
        x.insert(0,1)
        x = np.array(x)
        
#        X = np.column_stack((np.ones(ngh_stns.size),ngh_stns[LON],ngh_stns[LAT],ngh_stns[ELEV],ngh_stns[TDI],ngh_stns[LST]))
#        x = np.array([1.,self.pt[LON],self.pt[LAT],self.pt[ELEV],self.pt[TDI],self.pt[LST]])

        
        interp_anom = self.aclib.repRegress(X, ngh_obs_cntr, x, ngh_wgt)
        #interp_anom = np.average(ngh_obs_cntr,axis=1,weights=ngh_wgt)
        interp_vals = interp_anom + self.pt[get_norm_varname(mth)]
                
        return interp_vals 
    
class InterpTair(object):
    
    def __init__(self,krig_tair,gwrpca_tair):
        
        self.krig_tair = krig_tair
        self.gwrpca_tair = gwrpca_tair
        self.mthMasks = gwrpca_tair.mthMasks
        self.ndays = self.mthMasks[None].size
        
    def interp(self,pt,rm_stnid=None):
        
        tair_daily = np.zeros(self.ndays)
        tair_norms = np.zeros(12)
        tair_se = np.zeros(12)
        
        set_pt = True
        
        for mth in np.arange(1,13):
            
            set_pt = True if mth == 1 else False
            tair_mean,tair_var = self.krig_tair.krig(pt,rm_stnid,mth=mth,set_pt=set_pt)
            std_err,ci = self.krig_tair.std_err_ci(tair_mean,tair_var)
            pt[get_norm_varname(mth)] = tair_mean
            tair_norms[mth-1] = tair_mean
            tair_se[mth-1] = std_err
            
            self.gwrpca_tair.mth = mth
            self.gwrpca_tair.setup_for_pt(pt,rm_stnid)
            tair_daily[self.mthMasks[mth]] = self.gwrpca_tair.gwr_pca(mth)
        
        return tair_daily,tair_norms,tair_se

class PtInterpTair(object):
    
    #def __init__(self,dspathTmin,dspathTmax,pathRlib,pathClib,auxFpaths=None):
    def __init__(self,stn_da_tmin,stn_da_tmax,pathRlib,pathClib,auxFpaths=None):    
        #stn_da_tmin = station_data_infill(dspathTmin, 'tmin',vcc_size=470560000*2)
        #stn_da_tmax = station_data_infill(dspathTmax, 'tmax',vcc_size=470560000*2)
        self.days = stn_da_tmin.days
        self.stn_da_tmin = stn_da_tmin
        self.stn_da_tmax = stn_da_tmax
        
        #Masks for calculating monthly norms after daily Tmin/Tmax values had to be adjusted due to Tmin >= Tmax
        
        self.daysNormMask = np.nonzero(np.logical_and(self.days[YEAR] >= 1981,self.days[YEAR] <= 2010))[0] 
        daysNorm = self.days[self.daysNormMask]
        
        uYrs = np.unique(daysNorm[YEAR])
        self.yrMths = utld.get_mth_metadata(uYrs[0],uYrs[-1])
        
        self.yrMthsMasks = []
        for aYr in uYrs:
            for aMth in np.arange(1,13):
                self.yrMthsMasks.append(np.nonzero(np.logical_and(daysNorm[YEAR]==aYr,daysNorm[MONTH]==aMth))[0])
        
        self.mthMasks = []
        for mth in np.arange(1,13):
            self.mthMasks.append(np.nonzero(self.yrMths[MONTH]==mth)[0])
        
        mask_stns_tmin = np.isnan(stn_da_tmin.stns[BAD]) 
        mask_stns_tmax = np.isnan(stn_da_tmax.stns[BAD])
        
        stn_slct_tmin = station_select(stn_da_tmin, mask_stns_tmin)
        stn_slct_tmax = station_select(stn_da_tmax, mask_stns_tmax)
        
        domain_stns_tmin = stn_da_tmin.stns[np.logical_and(mask_stns_tmin,np.isfinite(stn_da_tmin.stns[MASK]))]
        domain_stns_tmax = stn_da_tmax.stns[np.logical_and(mask_stns_tmax,np.isfinite(stn_da_tmax.stns[MASK]))]
        self.nnghparams_tmin = get_rgn_nnghs_dict(domain_stns_tmin)
        self.nnghparams_tmax = get_rgn_nnghs_dict(domain_stns_tmax)
        
        init_interp_R_env(pathRlib)
        krig_tmin = KrigTair(stn_slct_tmin)
        krig_tmax = KrigTair(stn_slct_tmax)
        
        gwr_tmin = GwrPcaTairDynamic(stn_slct_tmin, pathClib, self.days, set_pt=False)
        gwr_tmax = GwrPcaTairDynamic(stn_slct_tmax, pathClib, self.days, set_pt=False)
        
        self.interp_tmin = InterpTair(krig_tmin, gwr_tmin)
        self.interp_tmax = InterpTair(krig_tmax, gwr_tmax)
        
        if auxFpaths is not None:
            self.pGrids = PredictorGrids(auxFpaths)
        
        self.a_pt = build_empty_pt()
        
    
    def interpLonLatPt(self,lon,lat,fixInvalid=True,chgLatLon=True,rm_stnid=None):
        
        self.a_pt[LON] = lon
        self.a_pt[LAT] = lat
        self.pGrids.setPtValues(self.a_pt,chgLatLon)
        
        if self.a_pt[MASK] == 0:
            raise Exception('Point is outside interpolation region')
        
        return self.interpPt(fixInvalid,rm_stnid)
    
    def interpPt(self,fixInvalid=True,rm_stnid=None):
        
        for mth in np.arange(1,13):
            
            self.a_pt[get_lst_varname(mth)] = self.a_pt["tmin%02d"%mth]
            self.a_pt[get_optim_varname(mth)],self.a_pt[get_optim_anom_varname(mth)] = self.nnghparams_tmin[self.a_pt[NEON]][mth]

        tmin_dly, tmin_norms, tmin_se = self.interp_tmin.interp(self.a_pt,rm_stnid=rm_stnid)

        for mth in np.arange(1,13):
            
            self.a_pt[get_lst_varname(mth)] = self.a_pt["tmax%02d"%mth]
            self.a_pt[get_optim_varname(mth)],self.a_pt[get_optim_anom_varname(mth)] = self.nnghparams_tmax[self.a_pt[NEON]][mth]

        tmax_dly, tmax_norms, tmax_se = self.interp_tmax.interp(self.a_pt,rm_stnid=rm_stnid)
        
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
    
def init_interp_R_env(path_r_src):
    r.source(path_r_src)
    #Set trend formula
    ri.globalenv['FORMULA'] =  ri.globalenv.get("build_formula")(ri.StrSexpVector(["tair"]),ri.StrSexpVector(KRIG_TREND_VARS))

class BuildKrigParams(object):
    
    def __init__(self,stn_slct):
        
        self.stn_slct = stn_slct
        self.r_kparam_func = ri.globalenv.get('get_vario_params')
        
    def get_krig_params(self,pt,min_nngh,max_nngh,rm_stnid=None,mth=None,set_pt=True):
        
        if set_pt:
            self.stn_slct.set_pt(pt[LAT],pt[LON],stns_rm=rm_stnid)   
        self.stn_slct.set_params(DFLT_INIT_NNGHS,DFLT_INIT_NNGHS)
        self.stn_slct.set_ngh_stns(load_obs=False)
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_varname(mth)])
        
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        
        if domain_stns.size == 0:
            raise Exception("Cannot determine the optimal # of neighbors to use!")
        
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]/np.sum(self.stn_slct.ngh_wgt[indomain_mask])
                    
        min_nngh = np.int(np.round(np.average(domain_stns[get_optim_varname(mth)],weights=n_wgt)))
        max_nngh = np.int(min_nngh+np.round(min_nngh*0.20))
    
        self.stn_slct.set_params(min_nngh,max_nngh)
        self.stn_slct.set_ngh_stns(load_obs=False) 
        
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

class GwrTairMean(object):
    
    def __init__(self,stn_slct):
        
        self.stn_slct = stn_slct
        self.r_gwr_func = ri.globalenv.get('gwr_meantair')
        
    def gwr(self,pt,min_nngh,max_nngh,rm_stnid=None,reset_pt=True,mth=None):
        
        if reset_pt:
            self.stn_slct.set_pt(pt[LAT],pt[LON],stns_rm=rm_stnid)   
        
        self.stn_slct.set_params(min_nngh,max_nngh)
        self.stn_slct.set_ngh_stns(load_obs=False)
        
        nghs = self.stn_slct.ngh_stns
        ngh_lon = ri.FloatSexpVector(nghs[LON])
        ngh_lat = ri.FloatSexpVector(nghs[LAT])
        ngh_elev = ri.FloatSexpVector(nghs[ELEV])
        ngh_tdi = ri.FloatSexpVector(nghs[TDI])
        ngh_lst = ri.FloatSexpVector(nghs[get_lst_varname(mth)])
        ngh_tair = ri.FloatSexpVector(nghs[get_norm_varname(mth)])
        ngh_wgt = ri.FloatSexpVector(self.stn_slct.ngh_wgt)
        
        pt_svp = ri.FloatSexpVector((pt[LON],pt[LAT],pt[ELEV],pt[TDI],pt[get_lst_varname(mth)],pt[NEON]))
        
        rslt = self.r_gwr_func(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,pt_svp)
        tair_mean = rslt[0]
        bad_interp = rslt[1]
        
        if bad_interp != 0:
            print "".join([str(bad_interp)," bad interp: ",str(pt)])
        
        return tair_mean
                    
class KrigTair(object):
    
    def __init__(self,stn_slct):
        
        self.stn_slct = stn_slct
        self.r_krig_func = ri.globalenv.get('krig_meantair')
        self.ci_critval = stats.norm.ppf(0.025)
            
    def std_err_ci(self,tair_mean,tair_var):
        
        std_err = np.sqrt(tair_var)
        ci_r = np.abs(std_err*self.ci_critval) 
        
        return std_err,(tair_mean-ci_r,tair_mean+ci_r)
     
    def krig(self,pt,rm_stnid=None,smth_nnghs=True, gw_p=1.0,mth=None,set_pt=True):
        
        if set_pt:
            self.stn_slct.set_pt(pt[LAT],pt[LON],stns_rm=rm_stnid)   
        
        min_nngh = pt[get_optim_varname(mth)]
        max_nngh = min_nngh+np.round(min_nngh*0.20) 
        
        if smth_nnghs:
        
            #self.stn_slct.set_params(DFLT_INIT_NNGHS,DFLT_INIT_NNGHS)
            self.stn_slct.set_nngh_stns(DFLT_INIT_NNGHS,load_obs=False)
            
            indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_optim_varname(mth)])
            domain_stns = self.stn_slct.ngh_stns[indomain_mask]
            
            if domain_stns.size == 0:
                raise Exception("Cannot determine the optimal # of neighbors to use!")
            
            n_wgt = self.stn_slct.ngh_wgt[indomain_mask]/np.sum(self.stn_slct.ngh_wgt[indomain_mask])
                        
            min_nngh = np.int(np.round(np.average(domain_stns[get_optim_varname(mth)],weights=n_wgt)))
            max_nngh = np.int(min_nngh+np.round(min_nngh*0.20))
        
        self.stn_slct.set_params(min_nngh,max_nngh,gw_p=gw_p)
        self.stn_slct.set_ngh_stns(load_obs=False)
        
        indomain_mask = np.isfinite(self.stn_slct.ngh_stns[get_krigparam_varname(mth, VARIO_NUG)])
        domain_stns = self.stn_slct.ngh_stns[indomain_mask]
        n_wgt = self.stn_slct.ngh_wgt[indomain_mask]/np.sum(self.stn_slct.ngh_wgt[indomain_mask])
        
        nug = np.average(domain_stns[get_krigparam_varname(mth, VARIO_NUG)],weights=n_wgt)
        psill = np.average(domain_stns[get_krigparam_varname(mth, VARIO_PSILL)],weights=n_wgt) 
        vrange = np.average(domain_stns[get_krigparam_varname(mth, VARIO_RNG)],weights=n_wgt)  
                
        nghs = self.stn_slct.ngh_stns

        ngh_lon = ri.FloatSexpVector(nghs[LON])
        ngh_lat = ri.FloatSexpVector(nghs[LAT])
        ngh_elev = ri.FloatSexpVector(nghs[ELEV])
        ngh_tdi = ri.FloatSexpVector(nghs[TDI])
        ngh_lst = ri.FloatSexpVector(nghs[get_lst_varname(mth)])
        ngh_tair = ri.FloatSexpVector(nghs[get_norm_varname(mth)])
        
#        plt.plot(nghs[get_lst_varname(mth)],nghs[get_norm_varname(mth)],'.')
#        plt.show()
        
        pt_svp = ri.FloatSexpVector((pt[LON],pt[LAT],pt[ELEV],pt[TDI],pt[get_lst_varname(mth)]))

        nug = ri.FloatSexpVector([nug])
        psill = ri.FloatSexpVector([psill])
        vrange = ri.FloatSexpVector([vrange])
        
        ngh_wgt = ri.FloatSexpVector(self.stn_slct.ngh_wgt)

#        rslt = self.r_krig_func(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,
#                                pt_svp,nug,psill,vrange)
      
        rslt = self.rkrig(ngh_lon, ngh_lat, ngh_elev, ngh_tdi, ngh_lst, ngh_tair, ngh_wgt, pt_svp, nug, psill, vrange)
      
        tair_mean = rslt[0]
        tair_var = rslt[1]
        bad_interp = rslt[2]
        
        if bad_interp != 0:
            print "".join([str(bad_interp)," bad interp: ",str(pt)])
        
        return tair_mean,tair_var
    
    def rkrig(self,ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,pt_svp,nug,psill,vrange):
        rslt = self.r_krig_func(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,
                        pt_svp,nug,psill,vrange)
        return rslt


class StationDataWrkChk(station_data_infill):
    '''
    A station_data class for accessing stations and observations from a single variable infilled netcdf weather station database.
    '''
    
    def __init__(self, nc_path, var_name,vcc_size=None,vcc_nelems=None,vcc_preemption=None,stn_dtype=DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM):
        
        station_data_infill.__init__(self, nc_path, var_name, vcc_size, vcc_nelems, vcc_preemption, stn_dtype)
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
            return station_data_infill.load_obs(self, stn_ids, mth)
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

def buildDefaultPtInterp():
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
    
    gridPath = '/projects/daymet2/dem/interp_grids/conus/ncdf/'
    auxFpaths = ["".join([gridPath,'fnl_elev.nc']),
                 "".join([gridPath,'fnl_tdi.nc']),
                 "".join([gridPath,'fnl_climdiv.nc']),
                 "".join([gridPath,'fnl_mask.nc'])]
    auxFpaths.extend(["".join([gridPath,'fnl_lst_tmin%02d.nc'%mth]) for mth in np.arange(1,13)])
    auxFpaths.extend(["".join([gridPath,'fnl_lst_tmax%02d.nc'%mth]) for mth in np.arange(1,13)])
    
    ptInterp = PtInterpTair(stndaTmin,stndaTmax,
                               '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R', 
                               '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C',auxFpaths) 
    return ptInterp