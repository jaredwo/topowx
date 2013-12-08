'''
Created on Aug 2, 2013

@author: jared.oyler
'''
import numpy as np
from netCDF4 import Dataset
import netCDF4
import matplotlib.pyplot as plt
from db.station_data import station_data_infill,OPTIM_NNGH,OPTIM_NNGH_ANOM,NEON,MASK,BAD,LON,LAT,ELEV,TDI,NEON,MEAN_OBS,LST
import interp.interp_tair as it
from interp.interp_tair import LST_TMIN,LST_TMAX
import db.ushcn as ushcn


def set_optim_nnghs(fpathSerialDb,varTair,optimStats,optimVar=OPTIM_NNGH,longname=None):
        
    stn_da = station_data_infill(fpathSerialDb,varTair)
    stn_rgn = stn_da.stns[NEON]
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD]))
    stn_da.ds.close()
    stn_da = None
    
    ds = Dataset(fpathSerialDb,'r+')
    
    if optimVar not in ds.variables.keys():
            
        avar = ds.createVariable(optimVar,'f8',('stn_id',),fill_value=netCDF4.default_fillvals['f8'])
        if longname is not None:
            avar.long_name = longname
        avar.units = 'NA'
            
    else:
            
        avar = ds.variables[optimVar]
    
    avar[:] = avar._FillValue
    ds.sync()
    
    for optimStatsDiv in optimStats:
        divMask = np.logical_and(stn_rgn ==optimStatsDiv[0],stn_mask)
        avar[divMask] = optimStatsDiv[3]
    ds.sync()

def optimTairMeanStats(fpathSerialDb,pathXvalDs):
    
    ds = Dataset(fpathSerialDb)
    divs = ds.variables['neon'][:]
    divs = np.unique(divs.data[np.logical_not(divs.mask)])
    ds.close()
    ds = None
    
    optimStats = []

    for aDiv in divs:
        
        dsXval = Dataset("".join([pathXvalDs,"_",str(aDiv),".nc"]))
        minNghs = dsXval.variables['min_nghs'][:]
        mae = dsXval.variables['mae'][:]
        bias = dsXval.variables['bias'][:]
        #stnid = dsXval.variables['stn_id'][:].astype("<S16")
        
        mmae = np.mean(mae,axis=1)
        mbias = np.mean(bias,axis=1)
        x = np.argmin(mmae)
        
        fnlMmae = mmae[x]
        fnlNghs = minNghs[x]
        fnlMbias = mbias[x]
        
        optimStatsDiv = [aDiv,fnlMmae,fnlMbias,fnlNghs]
        optimStats.append(optimStatsDiv)       
        print "|".join([str(x) for x in optimStatsDiv])
    
    return optimStats
        
def optimTairAnomStats(fpathSerialDb,pathXvalDs):
    
    ds = Dataset(fpathSerialDb)
    divs = ds.variables['neon'][:]
    divs = np.unique(divs.data[np.logical_not(divs.mask)])
    ds.close()
    ds = None
    
    optimStats = []
    
    for aDiv in divs:
        
        dsXval = Dataset("".join([pathXvalDs,"_",str(aDiv),".nc"]))
        mae = dsXval.variables['mae'][:]
        bias = dsXval.variables['bias'][:]
        nngh = dsXval.variables['min_nghs'][:]
        r2 = dsXval.variables['r2'][:]
        
        mmae = np.mean(mae,axis=1)
        mbias = np.mean(bias,axis=1)
        mr2 = np.mean(r2,axis=1)
        
        x = np.argmin(mmae)
        
        optimStatsDiv = [aDiv,mmae[x],mbias[x],nngh[x],mr2[x]]
        optimStats.append(optimStatsDiv)    
        
        print "%d|%f|%f|%f|%d"%(aDiv,mmae[x],mbias[x],mr2[x],nngh[x])
    
    return optimStats

def xvalStns(stnIds,tairVar):
        
    auxFpaths = ['/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_climdiv.nc']
            
    ptInterper = it.PtInterpTair('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc',
                    '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc',
                    '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                    '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                    auxFpaths)
    
    bias = np.zeros(stnIds.size)
    mae = np.zeros(stnIds.size)
    
    for xvalId,x in zip(stnIds,np.arange(stnIds.size)):
        print xvalId
        xvalStnTmax = ptInterper.stn_da_tmax.stns[ptInterper.stn_da_tmax.stn_ids==xvalId][0]
        xvalStnTmin = ptInterper.stn_da_tmin.stns[ptInterper.stn_da_tmin.stn_ids==xvalId][0]
        xvalStn = xvalStnTmax
        
        rm_stnid = np.array([xvalId])
        obstmax = ptInterper.stn_da_tmax.load_obs(xvalId)
        obstmin = ptInterper.stn_da_tmin.load_obs(xvalId)
            
        aPt = it.build_empty_pt()
        aPt[LON] = xvalStn[LON]
        aPt[LAT] = xvalStn[LAT]
        aPt[ELEV] = xvalStn[ELEV]
        aPt[TDI] = xvalStn[TDI]
        aPt[LST_TMAX] = xvalStnTmax[LST]
        aPt[LST_TMIN] = xvalStnTmin[LST]
        aPt[NEON] = xvalStn[NEON]
        aPt[MEAN_OBS] = xvalStn[MEAN_OBS]
        aPt[OPTIM_NNGH] = xvalStn[OPTIM_NNGH]
        aPt[OPTIM_NNGH_ANOM] = xvalStn[OPTIM_NNGH_ANOM]        
        ptInterper.a_pt = aPt
                
        tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpPt(rm_stnid=rm_stnid)
        
        if tairVar == 'tmin':
            bias[x] = np.mean(tmin_dly-obstmin)
            mae[x] = np.mean(np.abs(tmin_dly-obstmin))
        elif tairVar == 'tmax':
            bias[x] = np.mean(tmax_dly-obstmax)
            mae[x] = np.mean(np.abs(tmax_dly-obstmax))
    
    mbias = np.mean(bias)
    mmab = np.mean(np.abs(bias))
    mmae = np.mean(mae)
    
    return mbias,mmab,mmae
    
                    
if __name__ == '__main__':
    #Mean Tmax Optim Min Nghs
#    optimStats = optimTairMeanStats('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 
#                                    '/projects/daymet2/station_data/infill/infill_20130725/xval/optimTairMean/tmax/xval_tmax_mean')
#    set_optim_nnghs('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax', 
#                    optimStats, OPTIM_NNGH, "optimal number of neighbors for mean Tair interpolation")
    
    #Mean Tmin Optim Min Nghs
#    optimStats = optimTairMeanStats('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 
#                                    '/projects/daymet2/station_data/infill/infill_20130725/xval/optimTairMean/tmin/xval_tmin_mean')
#    set_optim_nnghs('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin', 
#                    optimStats, OPTIM_NNGH, "optimal number of neighbors for mean Tair interpolation")
    
    #Anomaly Tmax Optim Min Nghs
#    optimStats = optimTairAnomStats('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc',
#                                    '/projects/daymet2/station_data/infill/infill_20130725/xval/optimTairAnom/tmax/xval_tmax_anom')
#    set_optim_nnghs('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax', 
#                    optimStats, OPTIM_NNGH_ANOM, "optimal number of neighbors for Tair anomaly interpolation")
    
    #Anomaly Tmin Optim Min Nghs
    optimStats = optimTairAnomStats('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc',
                                    '/projects/daymet2/station_data/infill/infill_20130725/xval/optimTairAnom/tmin/xval_tmin_anom')
    set_optim_nnghs('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin', 
                    optimStats, OPTIM_NNGH_ANOM, "optimal number of neighbors for Tair anomaly interpolation")

#    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
#    print xvalStns(stnda.stn_ids[np.logical_and(stnda.stns[NEON]==2502,np.isnan(stnda.stns[BAD]))],'tmax')
    
