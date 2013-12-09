'''
Created on Sep 25, 2013

@author: jared.oyler
'''

import numpy as np
from twx.db.station_data import station_data_infill,STN_ID,BAD,get_norm_varname,LAT,LON,\
    get_optim_varname, get_optim_anom_varname, get_lst_varname
from twx.interp.station_select import station_select
import twx.interp.interp_tair as it
import cProfile
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF4
from twx.utils.status_check import status_check
import scipy.stats as stats

class OptimTairMean(object):
    '''
    classdocs
    '''

    def __init__(self,pathDb,pathRlib,tairVar):
       
        stn_da = station_data_infill(pathDb, tairVar)
        mask_stns = np.isnan(stn_da.stns[BAD])
        stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
              
        it.init_interp_R_env(pathRlib)
        
        self.gwrtair = it.GwrTairMean(stn_slct)
        self.stn_da = stn_da
        
    def runXval(self,stnId,minNnghs,maxNnghs):
        
        aStn = self.stn_da.stns[self.stn_da.stns[STN_ID]==stnId][0]
        
        bias = np.zeros(12)
        mae = np.zeros(12)
        
        self.gwrtair.stn_slct.set_pt(aStn[LAT],aStn[LON],stns_rm=np.array([aStn[STN_ID]])) 
        
        for mth in np.arange(1,13):
            
            tairNormMth = self.gwrtair.gwr(aStn, minNnghs, maxNnghs, rm_stnid=np.array([aStn[STN_ID]]),mth=mth,reset_pt=False)
            xvalTairNormMth = aStn[get_norm_varname(mth)]
            biasMth = tairNormMth - xvalTairNormMth
            maeMth = np.abs(biasMth)
            bias[mth-1] = biasMth
            mae[mth-1] = maeMth
                
        return mae,bias

def setOptimTairParams(pathDb,pathXvalDs):
#    '/projects/daymet2/station_data/infill/infill_20130725/xval/optimTairMean/tmax/xval_tmax_mean' 
    dsStns = Dataset(pathDb,'r+')
    climDivStns = dsStns.variables['neon'][:].data
    
    varsOptim = {}
    for mth in np.arange(1,13):
        
        varNameOptim = get_optim_varname(mth)
        if varNameOptim not in dsStns.variables.keys():
            varOptim = dsStns.createVariable(varNameOptim,'f8',('stn_id',),fill_value=netCDF4.default_fillvals['f8'])
        else:
            varOptim = dsStns.variables[varNameOptim]
        varOptim[:] = netCDF4.default_fillvals['f8']
        varsOptim[mth] = varOptim
    
    divs = dsStns.variables['neon'][:]
    divs = np.unique(divs.data[np.logical_not(divs.mask)])
    
    stchk = status_check(divs.size, 10)
    
    for climDiv in divs:
        
        fpath = "".join([pathXvalDs,"_",str(climDiv),".nc"])
        dsClimDiv = Dataset(fpath)
        
        maeClimDiv = dsClimDiv.variables['mae'][:]
        nnghsClimDiv = dsClimDiv.variables['min_nghs'][:]
        
        climDivMask = np.nonzero(climDivStns==climDiv)[0]
        
        for mth in np.arange(1,13):
            maeClimDivMth = maeClimDiv[mth-1,:,:]
            mmae = np.mean(maeClimDivMth,axis=1)
            minIdx = np.argmin(mmae)
            varsOptim[mth][climDivMask] = nnghsClimDiv[minIdx]
        
        stchk.increment()
    dsStns.sync()

def setOptimTairAnomParams(pathDb,pathXvalDs):
#    '/projects/daymet2/station_data/infill/infill_20130725/xval/optimTairMean/tmax/xval_tmax_mean' 
    dsStns = Dataset(pathDb,'r+')
    climDivStns = dsStns.variables['neon'][:].data
    
    varsOptim = {}
    for mth in np.arange(1,13):
        
        varNameOptim = get_optim_anom_varname(mth)
        if varNameOptim not in dsStns.variables.keys():
            varOptim = dsStns.createVariable(varNameOptim,'f8',('stn_id',),fill_value=netCDF4.default_fillvals['f8'])
        else:
            varOptim = dsStns.variables[varNameOptim]
        varOptim[:] = netCDF4.default_fillvals['f8']
        varsOptim[mth] = varOptim
    
    divs = dsStns.variables['neon'][:]
    divs = np.unique(divs.data[np.logical_not(divs.mask)])
    
    stchk = status_check(divs.size, 10)
    
    for climDiv in divs:
        
        fpath = "".join([pathXvalDs,"_",str(climDiv),".nc"])
        dsClimDiv = Dataset(fpath)
        
        maeClimDiv = dsClimDiv.variables['mae'][:]
        nnghsClimDiv = dsClimDiv.variables['min_nghs'][:]
        
        climDivMask = np.nonzero(climDivStns==climDiv)[0]
        
        for mth in np.arange(1,13):
            maeClimDivMth = maeClimDiv[:,mth-1,:]
            mmae = np.mean(maeClimDivMth,axis=1)
            minIdx = np.argmin(mmae)
            varsOptim[mth][climDivMask] = nnghsClimDiv[minIdx]
        
        stchk.increment()
    dsStns.sync()

def build_min_ngh_windows(rng_min,rng_max,pct_step):
    
    min_nghs = []
    n = rng_min
    
    while n <= rng_max:
        min_nghs.append(n)
        n = n + np.round(pct_step*n)
    
    return np.array(min_nghs)


class OptimKrigParams(object):
    '''
    classdocs
    '''

    def __init__(self,pathDb,pathRlib,tairVar):
        
        stn_da = station_data_infill(pathDb, tairVar)
        mask_stns = np.isnan(stn_da.stns[BAD])
        stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=False) 
        it.init_interp_R_env(pathRlib)
        krigparams = it.BuildKrigParams(stn_slct)

        self.stn_da = stn_da
        self.krigparams = krigparams
        
    def getKrigParams(self,stnId):
        stn = self.stn_da.stns[self.stn_da.stn_idxs[stnId]]
        
        nugs = np.zeros(12)
        psills = np.zeros(12)
        rngs = np.zeros(12)
        
        set_pt = True
        
        for mth in np.arange(1,13):
            
            min_ngh = stn[get_optim_varname(mth)]
            max_ngh = min_ngh+np.round(min_ngh*0.20) 
            set_pt = True if mth == 1 else False
            nug,psill,rng = self.krigparams.get_krig_params(stn, min_ngh, max_ngh,mth=mth,set_pt=set_pt)
            nugs[mth-1] = nug
            psills[mth-1] = psill
            rngs[mth-1] = rng
        
        return nugs,psills,rngs


class XvalTairMean(object):
    '''
    classdocs
    '''

    def __init__(self,pathDb,pathRlib,tairVar):
        
        stn_da = station_data_infill(pathDb, tairVar)
        mask_stns = np.isnan(stn_da.stns[BAD])         
        stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
              
        it.init_interp_R_env(pathRlib)
       
        self.krig = it.KrigTair(stn_slct)
        self.stn_da = stn_da
        
    def runXval(self,stnId):
        
        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stnId]]
        
        self.krig.stn_slct.set_pt(xval_stn[LAT],xval_stn[LON],stns_rm=np.array([xval_stn[STN_ID]])) 
        
        stdErrs = np.zeros(13)
        errs = np.zeros(13)
        mths = np.arange(1,13)
        tairMthly = np.zeros(12)
        
        xvalAnnMean = np.mean([xval_stn[get_norm_varname(mth)] for mth in mths])
        
        for mth in mths:
            
            tair_mean,tair_var =self.krig.krig(xval_stn, np.array([xval_stn[STN_ID]]), mth=mth, set_pt=False)
            std_err,ci = self.krig.std_err_ci(tair_mean, tair_var)
            xvalTairNormMth = xval_stn[get_norm_varname(mth)]
            err = tair_mean - xvalTairNormMth
            
            stdErrs[mth-1] = std_err
            errs[mth-1] = err
            tairMthly[mth-1] = tair_mean
        
        ann,annStderr = ann_mean_stderr(tairMthly, stdErrs[0:12])
        
        errs[-1] = ann - xvalAnnMean
        stdErrs[-1] = annStderr
        
        return errs,stdErrs

def ann_mean_stderr(tairMthly,stdErrMthly):
    
    ann = np.mean(tairMthly)
    annStderr = np.sqrt(np.sum(stdErrMthly**2))/np.float(stdErrMthly.size)
    
#    varMthly = stdErrMthly**2
#    print np.sqrt(np.sum(varMthly*((1.0/12.0)**2)))
#    
#    
#    sumVariance = np.sum(np.square(stdErrMthly))
#    sse = np.square(tairMthly-ann)
#    
#    print np.sqrt(sumVariance/np.float(tairMthly.size))
#    
#    annStderr = np.sqrt((sumVariance + np.sum(sse))/np.float(tairMthly.size))

    return ann,annStderr


class OptimTairAnom(object):

    def __init__(self,pathDb,pathClib,tairVar):
        
        stn_da = station_data_infill(pathDb, tairVar,vcc_size=470560000*2)
        mask_stns = np.isnan(stn_da.stns[BAD]) 
            
        stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
        gwr_pca = it.GwrPcaTairStatic(stn_slct,pathClib,stn_da.days,None,None,None)
        
        self.stn_da = stn_da
        self.gwr_pca = gwr_pca
        
    def runXval(self,stn_id,min_ngh_wins,max_nngh_delta):
        
        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]
        xval_obs = self.stn_da.load_obs(xval_stn[STN_ID])
        
        biasAll = np.zeros((min_ngh_wins.size,12))
        maeAll = np.zeros((min_ngh_wins.size,12))
        r2All = np.zeros((min_ngh_wins.size,12))
        
        set_pt = True
        for x in np.arange(min_ngh_wins.size):
            
            min_ngh = min_ngh_wins[x]
            max_ngh = min_ngh+np.round(min_ngh*max_nngh_delta)
            
            self.gwr_pca.reset_params(min_ngh, max_ngh, 1.0)
            set_pt = True if x == 0 else False
            self.gwr_pca.set_pt = set_pt
            self.gwr_pca.setup_for_pt(xval_stn, np.array([xval_stn[STN_ID]]))
        
            biasMths = np.zeros(12)
            maeMths = np.zeros(12)
            r2Mths = np.zeros(12)
            
            for mth in np.arange(1,13):
            
                xval_anom = xval_obs[self.gwr_pca.mthMasks[mth]] - xval_stn[get_norm_varname(mth)]
                
                interp_tair = self.gwr_pca.gwr_pca(mth=mth)
                interp_anom = interp_tair - xval_stn[get_norm_varname(mth)]
                
                difs = interp_anom - xval_anom
                                
                bias = np.mean(difs)
                mae = np.mean(np.abs(difs))
                
                r_value = stats.linregress(interp_anom, xval_anom)[2]
                r2 = r_value**2 #r-squared value; variance explained
                
                biasMths[mth-1] = bias
                maeMths[mth-1] = mae
                r2Mths[mth-1] = r2
            
            biasAll[x,:] = biasMths
            maeAll[x,:] = maeMths
            r2All[x,:] = r2Mths
            
        return biasAll,maeAll,r2All

class XvalTairOverall():
    
    def __init__(self,pathDb,pathRlib,pathClib,tairVar):
        
        stn_da = station_data_infill(pathDb, tairVar)
        mask_stns = np.isnan(stn_da.stns[BAD])         
        stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
          
        it.init_interp_R_env(pathRlib)
            
        krig_tair = it.KrigTair(stn_slct)
        gwr_tair = it.GwrPcaTairDynamic(stn_slct, pathClib, stn_da.days, set_pt=False)
        interp_tair = it.InterpTair(krig_tair, gwr_tair)
        
        self.stn_da = stn_da
        self.interp_tair = interp_tair
        self.mthMasks = gwr_tair.mthMasks
        
    def runXval(self,stn_id):

        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]
        xval_obs = self.stn_da.load_obs(xval_stn[STN_ID])
        xval_norms = np.array([xval_stn[get_norm_varname(mth)] for mth in np.arange(1,13)])

        tair_daily,tair_norms,tair_se = self.interp_tair.interp(xval_stn, np.array([xval_stn[STN_ID]]))
        
        #Monthly + Annual error stats
        maeDly = np.zeros(13)
        biasDly = np.zeros(13)
        r2Dly = np.zeros(13)
        maeNorm = np.zeros(13)
        biasNorm = np.zeros(13)
        
        difsNorm = tair_norms - xval_norms
        biasNorm[0:12] = difsNorm
        maeNorm[0:12] = np.abs(difsNorm)
        biasNorm[-1] = np.mean(tair_norms) - np.mean(xval_norms)
        maeNorm[-1] = np.abs(biasNorm[-1])
        
        difsDly = tair_daily - xval_obs
        
        #Monthly Error Stats
        for mth in np.arange(1,13):
            
            biasDly[mth-1] = np.mean(difsDly[self.mthMasks[mth]])
            maeDly[mth-1] = np.mean(np.abs(difsDly[self.mthMasks[mth]]))
            
            r_value = stats.linregress(tair_daily[self.mthMasks[mth]], xval_obs[self.mthMasks[mth]])[2]
            r2Dly[mth-1] = r_value**2 #r-squared value; variance explained
        
        #Annual Error Stats
        biasDly[-1] = np.mean(difsDly)
        maeDly[-1] = np.mean(np.abs(difsDly))
        r_value = stats.linregress(tair_daily, xval_obs)[2]
        r2Dly[-1] = r_value**2
        
        return biasNorm,maeNorm,maeDly,biasDly,r2Dly

def perfOptimTairMean():
    
    optim = OptimTairMean("/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc", 
                          '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R', 'tmax')
    
    #aStn = optim.stn_da.stns[optim.stn_da.stns[STN_ID]=='SNOTEL_13C01S'][0]
    
    rngMinNghs = build_min_ngh_windows(100,150, 0.10)
    
    for minNgh in rngMinNghs:
        
        mae,bias = optim.runXval('RAWS_CCRN', minNgh,minNgh + np.round(0.20*minNgh))
        print minNgh,mae


def perfXvalTairOverall():
    
    optim = XvalTairOverall("/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc",
                            '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                            '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 'tmin')
    
    biasNorm,maeNorm,maeDly,biasDly,r2Dly = optim.runXval('RAWS_CLAH')
    
    def runPerf():
        biasNorm,maeNorm,maeDly,biasDly,r2Dly = optim.runXval('RAWS_CLAH')
    
    global runAPerf
    runAPerf = runPerf
    
    cProfile.run('runAPerf()')
    
def perfOptimTairAnom():
    
    optim = OptimTairAnom("/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc", 
                          '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 'tmin')
    
    min_ngh_wins = build_min_ngh_windows(100, 150, 0.10)
    
    def runPerf():
        optim.runXval('SNOTEL_13C01S',min_ngh_wins,0.20)
    
    global runAPerf
    runAPerf = runPerf
    
    cProfile.run('runAPerf()')

def perfOptimKrigParams():
    
    optim = OptimKrigParams("/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc",
                            '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R', 'tmax')
    
    def runPerf():
        print optim.getKrigParams('SNOTEL_13C01S')
    
    global runAPerf
    runAPerf = runPerf
    
    cProfile.run('runAPerf()')
    
def perfXvalTairMean():
    
    xval = XvalTairMean("/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc",
                        '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R', 'tmin')
        
    def runPerf():
        errs,stdErrs = xval.runXval('RAWS_CLAH')
        print errs
        print stdErrs
    
    global runAPerf
    runAPerf = runPerf
    
    cProfile.run('runAPerf()')

def analyze_xval_tairmean():
    ds = Dataset('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
    se = ds.variables['xval_stderr_mthly'][:]
    climDiv = ds.variables['neon'][:].data
    maskClimDiv = climDiv==2401
    
    maskStns = np.logical_and(~se[0,:].mask,maskClimDiv)
    e = ds.variables['xval_err_mthly'][:,maskStns]
    se = se[:,maskStns]
    
    eOld = np.mean(np.abs(ds.variables['xval_err'][maskStns]))
    print "eOld",eOld
    
    norms = []
    for mth in np.arange(1,13):
        norm = ds.variables[get_norm_varname(mth)][maskStns]
        norm.shape = (1,norm.size)
        norms.append(norm)
    
    norms = np.vstack(norms)
    norms = np.vstack((norms,np.mean(norms,axis=0)))
    
    interps = norms + e
    
    crit = stats.norm.ppf(0.025)
    cir = np.abs(se*crit)
    
    cil,ciu = interps-cir,interps+cir
    
    inCi = np.logical_and(norms>=cil,norms<=ciu)

    print "In CI",np.sum(inCi,axis=1)/np.float(inCi.shape[1])
    print "MAE",np.mean(np.abs(e),axis=1)
    print "Bias",np.mean(e,axis=1)
    print "SE",np.mean(se,axis=1)

def analyze_xval_overall():
    ds = Dataset('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
    se = ds.variables['xval_stderr_mthly'][:]
    climDiv = ds.variables['neon'][:].data
    maskClimDiv = climDiv==2401
    
    maskStns = np.logical_and(~se[0,:].mask,maskClimDiv)
    e = ds.variables['xval_err_mthly'][:,maskStns]
    se = se[:,maskStns]
    
    eOld = np.mean(np.abs(ds.variables['xval_err'][maskStns]))
    print "eOld",eOld
    
    norms = []
    for mth in np.arange(1,13):
        norm = ds.variables[get_norm_varname(mth)][maskStns]
        norm.shape = (1,norm.size)
        norms.append(norm)
    
    norms = np.vstack(norms)
    norms = np.vstack((norms,np.mean(norms,axis=0)))
    
    interps = norms + e
    
    crit = stats.norm.ppf(0.025)
    cir = np.abs(se*crit)
    
    cil,ciu = interps-cir,interps+cir
    
    inCi = np.logical_and(norms>=cil,norms<=ciu)

    print "In CI",np.sum(inCi,axis=1)/np.float(inCi.shape[1])
    print "MAE",np.mean(np.abs(e),axis=1)
    print "Bias",np.mean(e,axis=1)
    print "SE",np.mean(se,axis=1)

def perfPtInterpTair():
    
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
   
    
    #stndaTmin = it.StationDataWrkChk('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    #stndaTmax = it.StationDataWrkChk('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
    
    path = '/projects/daymet2/dem/interp_grids/conus/ncdf/'
    
    auxFpaths = ["".join([path,'fnl_elev.nc']),
                 "".join([path,'fnl_tdi.nc']),
                 "".join([path,'fnl_climdiv.nc'])]
    
    for mth in np.arange(1,13):
        auxFpaths.append("".join([path,'fnl_lst_tmin%02d.nc'%mth]))
        auxFpaths.append("".join([path,'fnl_lst_tmax%02d.nc'%mth]))
        
    ptInterp = it.PtInterpTair(stndaTmin,stndaTmax, 
                               '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R', 
                               '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C',auxFpaths)
    
    lon,lat = -114.02853698,47.87532492
    tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterp.interpLonLatPt(lon, lat,fixInvalid=False)
    print tmin_norms
    
#    print ninvalid
#    tmin_dly_norm = np.take(tmin_dly, ptInterp.daysNormMask)
#    tmax_dly_norm = np.take(tmax_dly, ptInterp.daysNormMask)
#    tmin_mthly = np.array([np.mean(np.take(tmin_dly_norm,amask)) for amask in ptInterp.yrMthsMasks])
#    tmax_mthly = np.array([np.mean(np.take(tmax_dly_norm,amask)) for amask in ptInterp.yrMthsMasks])
#    tmin_norms2 = np.array([np.mean(np.take(tmin_mthly,amask)) for amask in ptInterp.mthMasks])
#    tmax_norms2 = np.array([np.mean(np.take(tmax_mthly,amask)) for amask in ptInterp.mthMasks])
#    print tmin_norms2-tmin_norms,tmax_norms2-tmax_norms
#    plt.plot(tmin_norms2-tmin_norms,'o-')
#    plt.plot(tmax_norms2-tmax_norms,'o-')
#    plt.show()
 
if __name__ == '__main__':

    perfPtInterpTair()
    #analyze_ci()
    #perfXvalTairOverall()
    #perfOptimTairAnom()
    #perfOptimKrigParams()
    #perfOptimTairMean()
    #perfXvalTairMean()