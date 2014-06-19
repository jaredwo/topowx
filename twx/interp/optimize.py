'''
Created on Sep 25, 2013

@author: jared.oyler
'''

import numpy as np
from twx.db.station_data import StationSerialDataDb,STN_ID,BAD,get_norm_varname,LAT,LON,\
    get_optim_varname, get_optim_anom_varname, get_lst_varname,DTYPE_STN_BASIC,MASK,TDI,BAD,NEON,DTYPE_NORMS,DTYPE_LST,\
    DTYPE_INTERP_OPTIM, DTYPE_INTERP_OPTIM_ALL, DTYPE_INTERP
from twx.interp.station_select import StationSelect
import twx.interp.interp_tair as it
import cProfile
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF4
from twx.utils.status_check import status_check
import scipy.stats as stats

class OptimKrigBwNstns(object):
    '''
    classdocs
    '''

    def __init__(self,pathDb,tairVar):
                
        stn_da = StationSerialDataDb(pathDb, tairVar,stn_dtype=DTYPE_INTERP)
        mask_stns = np.isnan(stn_da.stns[BAD])         
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
                     
        self.krig = it.KrigTairAll(stn_slct)
        self.stn_da = stn_da
        
    def runXval(self,stnId,abw_nngh):
        
        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stnId]]
        
        err = np.zeros((12,abw_nngh.size))
        xvalNorms = np.array([xval_stn[get_norm_varname(mth)] for mth in np.arange(1,13)])
        
        for bw_nngh,x in zip(abw_nngh,np.arange(abw_nngh.size)):
            
            interp_norms = self.krig.krigall(xval_stn, bw_nngh,stns_rm=xval_stn[STN_ID])
            err[:,x] = interp_norms-xvalNorms
                
        return err

class OptimGwrNormBwNstns(object):
    '''
    classdocs
    '''

    def __init__(self,pathDb,tairVar):
                
        stn_da = StationSerialDataDb(pathDb, tairVar)
        mask_stns = np.isnan(stn_da.stns[BAD])         
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
                     
        self.gwr_norm = it.GwrTairNorm(stn_slct)
        self.stn_da = stn_da
        
    def runXval(self,stnId,abw_nngh):
        
        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stnId]]
        
        err = np.zeros((12,abw_nngh.size))
        xvalNorms = np.array([xval_stn[get_norm_varname(mth)] for mth in np.arange(1,13)])
        
        for bw_nngh,x in zip(abw_nngh,np.arange(abw_nngh.size)):
            
            interp_norms = np.zeros(12)
            
            for i in np.arange(interp_norms.size):
            
                mth = i+1
                interp_mth = self.gwr_norm.gwr_predict(xval_stn, mth, nnghs=bw_nngh, stns_rm=xval_stn[STN_ID])[0]
                interp_norms[i] = interp_mth
            
            err[:,x] = interp_norms-xvalNorms
                
        return err

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

    def __init__(self,pathDb,tairVar):
        
        stn_da = StationSerialDataDb(pathDb, tairVar,stn_dtype=DTYPE_INTERP_OPTIM)
        mask_stns = np.isnan(stn_da.stns[BAD])
                
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=False) 
        krigparams = it.BuildKrigParams(stn_slct) 

        self.stn_da = stn_da
        self.krigparams = krigparams
        
    def getKrigParams(self,stnId):
        stn = self.stn_da.stns[self.stn_da.stn_idxs[stnId]]
        
        nugs = np.zeros(12)
        psills = np.zeros(12)
        rngs = np.zeros(12)
                
        for mth in np.arange(1,13):
            
            nug,psill,rng = self.krigparams.get_krig_params(stn,mth)

            nugs[mth-1] = nug
            psills[mth-1] = psill
            rngs[mth-1] = rng
        
        return nugs,psills,rngs


class OptimTairAnom(object):

    def __init__(self,pathDb,tairVar):
        
        stn_da = StationSerialDataDb(pathDb, tairVar,vcc_size=470560000*2)
        mask_stns = np.isnan(stn_da.stns[BAD]) 
            
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns,rm_zero_dist_stns=True)
        gwr = it.GwrTairAnom(stn_slct, stn_da.days)
        
        self.stn_da = stn_da
        self.gwr = gwr
        
    def runXval(self,stn_id,a_nnghs):
        
        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]
        xval_obs = self.stn_da.load_obs(xval_stn[STN_ID])
        
        biasAll = np.zeros((a_nnghs.size,12))
        maeAll = np.zeros((a_nnghs.size,12))
        r2All = np.zeros((a_nnghs.size,12))
        
        for x in np.arange(a_nnghs.size):
            
            nnghs = a_nnghs[x]
                    
            biasMths = np.zeros(12)
            maeMths = np.zeros(12)
            r2Mths = np.zeros(12)
            
            for mth in np.arange(1,13):
            
                xval_anom = xval_obs[self.gwr.mthMasks[mth]] - xval_stn[get_norm_varname(mth)]
                
                interp_tair = self.gwr.gwr_mth(xval_stn,mth,nnghs,stns_rm=xval_stn[STN_ID])
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
    
    def __init__(self,pathDb,tairVar):
        
        stn_da = StationSerialDataDb(pathDb, tairVar,stn_dtype=DTYPE_INTERP_OPTIM_ALL,vcc_size=470560000*2)
        mask_stns = np.isnan(stn_da.stns[BAD])       
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
                      
        krig_tair = it.KrigTair(stn_slct)
        gwr_tair = it.GwrTairAnom(stn_slct, stn_da.days)
        interp_tair = it.InterpTair(krig_tair, gwr_tair)
        
        self.stn_da = stn_da
        self.interp_tair = interp_tair
        self.mthMasks = gwr_tair.mthMasks
    
    def run_interp(self,stn_id):
        
        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]
        tair_daily,tair_norms,tair_se = self.interp_tair.interp(xval_stn, xval_stn[STN_ID])
        return tair_daily,tair_norms,tair_se
      
    def run_xval(self,stn_id):

        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stn_id]]
        xval_obs = self.stn_da.load_obs(xval_stn[STN_ID])
        xval_norms = np.array([xval_stn[get_norm_varname(mth)] for mth in np.arange(1,13)])
        
        tair_daily,tair_norms,tair_se = self.run_interp(stn_id)
        
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
        
        return biasNorm,maeNorm,maeDly,biasDly,r2Dly,tair_se

class XvalGwrNormOverall(object):
    '''
    classdocs
    '''

    def __init__(self,pathDb,tairVar):
                
        stn_da = StationSerialDataDb(pathDb, tairVar)
        mask_stns = np.isnan(stn_da.stns[BAD])         
        stn_slct = StationSelect(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
                     
        self.gwr_norm = it.GwrTairNorm(stn_slct)
        self.stn_da = stn_da
        self.blank_err = np.zeros(13)
        
    def run_xval(self,stnId):
        
        xval_stn = self.stn_da.stns[self.stn_da.stn_idxs[stnId]]
        
        xval_norms = np.array([xval_stn[get_norm_varname(mth)] for mth in np.arange(1,13)])
        xval_norms = np.concatenate((xval_norms,np.array([np.mean(xval_norms)])))
        
        interp_norms = np.zeros(13)
        interp_se = np.zeros(12)
           
        for i in np.arange(12):
        
            mth = i+1
            interp_mth,vary_mth = self.gwr_norm.gwr_predict(xval_stn, mth, stns_rm=xval_stn[STN_ID])
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
         
        return biasNorm,maeNorm,maeDly,biasDly,r2Dly,tair_se

def perfXvalTairOverall():
    
    optim = XvalTairOverall("/Users/jaredwo/Documents/data/serial_tmax.nc", 'tmax')
    
    biasNorm,maeNorm,maeDly,biasDly,r2Dly,seNorm = optim.run_xval('GHCN_USC00244558')
    
    print "MAE Norm"
    print maeNorm
    print "SE Norm"
    print seNorm
    print "MAE Daily"
    print maeDly
    
#    def runPerf():
#        biasNorm,maeNorm,maeDly,biasDly,r2Dly = optim.runXval('SNOTEL_13C01S')
#    
#    global runAPerf
#    runAPerf = runPerf
#    
#    cProfile.run('runAPerf()')
    
def perfOptimTairAnom():
    
    optim = OptimTairAnom("/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc", 'tmin')
    
    min_ngh_wins = build_min_ngh_windows(10, 150, 0.10)
    
    #biasAll, maeAll, r2All = optim.runXval('SNOTEL_13C01S', min_ngh_wins)
    #biasAll, maeAll, r2All = optim.runXval('GHCN_USC00244558', min_ngh_wins) #Kalispell
    biasAll, maeAll, r2All = optim.runXval('GHCN_USC00247448', min_ngh_wins) #Seeley Lake
    #biasAll, maeAll, r2All = optim.runXval('GHCN_USW00014755', min_ngh_wins)
    
    mae_argmin =  np.argmin(maeAll, 0)
    cols = np.arange(maeAll.shape[1])
    
    print "NNGHS"
    print min_ngh_wins[mae_argmin]
    print "MAE"
    print maeAll[mae_argmin,cols]
    print "BIAS"
    print biasAll[mae_argmin,cols]
    print "R2"
    print r2All[mae_argmin,cols]
    
#    print min_ngh_wins
#    maeMth = maeAll[:,7]
#    print maeMth
#    print min_ngh_wins[np.argmin(maeMth)]
#    print np.min(maeMth)
#    def runPerf():
#        optim.runXval('SNOTEL_13C01S',min_ngh_wins,0.20)
#    
#    global runAPerf
#    runAPerf = runPerf
#    
#    cProfile.run('runAPerf()')

def perfOptimKrigParams():
    
    optim = OptimKrigParams("/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc", 'tmax')
    
    nugs, psills, rngs = optim.getKrigParams('RAWS_CCRN')#('SNOTEL_19L43S')GHCN_USC00049043
    print nugs
#    def runPerf():
#        print optim.getKrigParams('SNOTEL_13C01S')
#    
#    global runAPerf
#    runAPerf = runPerf
#    
#    cProfile.run('runAPerf()')
    

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
    
    stndaTmin = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stndaTmax = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
   
    
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
 
def perftOptimKrigBwStns():
     
    optim = OptimKrigBwNstns('/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc', 'tmin')
    
    abw_nngh = build_min_ngh_windows(35,150, 0.10)
    
    err = optim.runXval('GHCN_USC00244558', abw_nngh)
    #err = optim.runXval('SNOTEL_13C01S', abw_nngh)
    
    mae = np.abs(err)
    
    print abw_nngh[np.argmin(mae,1)]
    print mae[np.arange(12),np.argmin(mae,1)]
    
#     optim = OptimTairMean("/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc", 
#                      '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R', 'tmax')
#    
#    #aStn = optim.stn_da.stns[optim.stn_da.stns[STN_ID]=='SNOTEL_13C01S'][0]
#    
#    rngMinNghs = build_min_ngh_windows(100,150, 0.10)
 
if __name__ == '__main__':

    #perftOptimKrigBwStns()
    #perfPtInterpTair()
    #analyze_ci()
    perfXvalTairOverall()
    #perfOptimTairAnom()
    #perfOptimKrigParams()
    #perfOptimTairMean()
    #perfXvalTairMean()