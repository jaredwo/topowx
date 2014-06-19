'''
Created on Dec 3, 2013

@author: jared.oyler
'''
from twx.db.station_data import StationDataDb, StationSerialDataDb, STN_ID, STN_NAME, LON, LAT, MONTH, NEON, DATE, YMD, YEAR, TMIN, TMAX, ELEV, TDI
import numpy as np
from multiprocessing import Pool
import twx.interp.interp_tair as it
from twx.utils.status_check import status_check
from DatasetCompare import PrismTileRaster
from scipy import stats
import twx.utils.util_dates as utld
from datetime import datetime
import sys
import shapefile
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import brewer2mpl
from matplotlib.colors import Normalize,BoundaryNorm
import pickle
from matplotlib import cm

def stnMetaToCsv():
    stns = np.load('/projects/daymet2/ds_compare/daily/stns.npy')
    fout = open('/projects/daymet2/ds_compare/daily/stn_pts.csv','w')
    fout.write("STN_ID,LAT,LON\n")
    
    for astn in stns:
        fout.write("%s,%.5f,%.5f\n"%(astn[STN_ID],astn[LAT],astn[LON])) 
    
    fout.close()
    
def stnMetaToDaymetExtract():
    stns = np.load('/projects/daymet2/ds_compare/daily/topowx_stns.npy')
    fout = open('/projects/daymet2/ds_compare/daily/twxstn_pts_dinput.csv','w')
    #fout.write("STN_ID,LAT,LON\n")
    
    for astn,x in zip(stns,np.arange(stns.size)):
        fout.write("%s.csv,%.5f,%.5f, ingore stuff%d\n"%(astn[STN_ID],astn[LAT],astn[LON],x+1)) 
    
    fout.close()
    
def normStnDataToNpy():
    
    dbHomog = StationDataDb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc',startend_ymd=(19810101,20101231))
    dbRaw = StationDataDb('/projects/daymet2/station_data/all/all_1948_2012.nc',startend_ymd=(19810101,20101231))
    
    stnsNorms = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    stnsNorms = stnsNorms[np.in1d(stnsNorms[STN_ID], dbHomog.stn_ids)]
    np.save('/projects/daymet2/ds_compare/daily/stns.npy',stnsNorms)
    
    tmin = dbHomog.load_all_stn_obs_var(stnsNorms[STN_ID], 'tmin')[0]
    np.save('/projects/daymet2/ds_compare/daily/obs_tmin_homog.npy',tmin)
    tmin = None
    
    tmax = dbHomog.load_all_stn_obs_var(stnsNorms[STN_ID], 'tmax')[0]
    np.save('/projects/daymet2/ds_compare/daily/obs_tmax_homog.npy',tmax)
    tmax = None
    
    tmin = dbRaw.load_all_stn_obs_var(stnsNorms[STN_ID], 'tmin')[0]
    np.save('/projects/daymet2/ds_compare/daily/obs_tmin_raw.npy',tmin)
    tmin = None
    
    tmax = dbRaw.load_all_stn_obs_var(stnsNorms[STN_ID], 'tmax')[0]
    np.save('/projects/daymet2/ds_compare/daily/obs_tmax_raw.npy',tmax)
    tmax = None
    
def stnDataToNpy():
    
    dbHomog = StationDataDb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc',startend_ymd=(19810101,20101231))
    dbRaw = StationDataDb('/projects/daymet2/station_data/all/all_1948_2012.nc',startend_ymd=(19810101,20101231))
    
    stnsNorms = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    stnsNorms = stnsNorms[np.in1d(stnsNorms[STN_ID], dbHomog.stn_ids)]
    np.save('/projects/daymet2/ds_compare/daily/stns.npy',stnsNorms)
    
    tmin = dbHomog.load_all_stn_obs_var(stnsNorms[STN_ID], 'tmin')[0]
    np.save('/projects/daymet2/ds_compare/daily/obs_tmin_homog.npy',tmin)
    tmin = None
    
    tmax = dbHomog.load_all_stn_obs_var(stnsNorms[STN_ID], 'tmax')[0]
    np.save('/projects/daymet2/ds_compare/daily/obs_tmax_homog.npy',tmax)
    tmax = None
    
    tmin = dbRaw.load_all_stn_obs_var(stnsNorms[STN_ID], 'tmin')[0]
    np.save('/projects/daymet2/ds_compare/daily/obs_tmin_raw.npy',tmin)
    tmin = None
    
    tmax = dbRaw.load_all_stn_obs_var(stnsNorms[STN_ID], 'tmax')[0]
    np.save('/projects/daymet2/ds_compare/daily/obs_tmax_raw.npy',tmax)
    tmax = None

def runInterp(x):

    global ptInterp
    global stns
    global yrMask
    
    if ptInterp is None:
        ptInterp = it.buildDefaultPtInterp()
        stns = np.load('/projects/daymet2/ds_compare/daily/topowx_stns.npy')
        yrMask = np.nonzero(np.logical_and(ptInterp.days[YEAR]>=1980,ptInterp.days[YEAR]<=2012))[0]
        print "Interpolation initialized"

    error = False
    try:
        tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterp.interpLonLatPt(stns[LON][x], stns[LAT][x], fixInvalid=False, chgLatLon=True)#,rm_stnid=np.array([stns[STN_ID][x]]))
    except Exception:
        print "ERROR: Station not in interpolation domain. %s|%s|%0.4f|%0.4f"%(stns[STN_ID][x],stns[STN_NAME][x],stns[LON][x],stns[LAT][x])
        error = True
    
    if not error:
        if np.sum(np.logical_or(np.abs(tmin_norms) > 100,np.abs(tmax_norms) > 100)) > 0:
            error = True
            print "ERROR: Bad interpolated normals. %s|%s|%0.4f|%0.4f"%(stns[STN_ID][x],stns[STN_NAME][x],stns[LON][x],stns[LAT][x])
        
    if error:
        tmin_dly = np.ones(yrMask.size)*np.nan
        tmax_dly = np.ones(yrMask.size)*np.nan
    else:
        tmin_dly = np.take(tmin_dly, yrMask)
        tmax_dly = np.take(tmax_dly, yrMask)
    
    return x,tmin_dly,tmax_dly

def buildConusStations():

    stndaTmax = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
    stndaTmin = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    climDivs = np.concatenate((stndaTmax.stns[NEON],stndaTmin.stns[NEON]))
    climDivs = np.unique(climDivs[np.isfinite(climDivs)])[2:]
    
    stnsTmin = stndaTmin.stns[np.in1d(stndaTmin.stns[NEON] ,climDivs, False)]
    stnsTmax = stndaTmax.stns[np.in1d(stndaTmax.stns[NEON] ,climDivs, False)]
    
    stns = stnsTmax
    stnsTmin = stnsTmin[~np.in1d(stnsTmin[STN_ID], stns[STN_ID], True)]
    print stns.size,stnsTmin.size
    stns = np.concatenate((stns,stnsTmin))
    
    np.save('/projects/daymet2/ds_compare/daily/topowx_stns.npy', stns)
    

def runInterps():
    
    global ptInterp
    global stns
    global yrMask
    ptInterp = None
    stns = None
    yrMask = None
    pool = Pool(10)
    
    aPtInterp = it.buildDefaultPtInterp()
    astns = np.load('/projects/daymet2/ds_compare/daily/topowx_stns.npy')
    yrMask = np.nonzero(np.logical_and(aPtInterp.days[YEAR]>=1980,aPtInterp.days[YEAR]<=2012))[0]
    ndays = yrMask.size
    
    tminDly = np.zeros((ndays,astns.size),dtype=np.float32)*np.float32(np.nan)
    tmaxDly = np.zeros((ndays,astns.size),dtype=np.float32)*np.float32(np.nan)
    
    sck = status_check(astns.size, 100)
    chksize = 100
    
    for x in np.arange(astns.size,step=chksize):
    
        endx = x+chksize
        if endx > astns.size:
            endx = astns.size
        idx = np.arange(x,endx)
        results = pool.map(runInterp,idx,chunksize=1)
        
        for aresult in results:
            j,tminDlyStn,tmaxDlyStn = aresult
            tminDly[:,j] = tminDlyStn
            tmaxDly[:,j] = tmaxDlyStn
        
        sck.increment(chksize)
    
    pool.close()
    pool.join()
    
    np.save('/projects/daymet2/ds_compare/daily/interptwxstns_tmin_twx.npy', tminDly)
    np.save('/projects/daymet2/ds_compare/daily/interptwxstns_tmax_twx.npy', tmaxDly)

def buildPrismDaily():
    
    #stns = np.load('/projects/daymet2/ds_compare/daily/stns.npy')
    stns = np.load('/projects/daymet2/ds_compare/daily/topowx_stns.npy')
    
    
    pRastTmin = PrismTileRaster('/projects/daymet2/prism/4km_daily/netcdf/prism4km_conus_tmin.nc', 'tmin')
    pRastTmax = PrismTileRaster('/projects/daymet2/prism/4km_daily/netcdf/prism4km_conus_tmax.nc', 'tmax')
    
    #aTmin = pRastTmin.aVar[:]
    #aTmax = pRastTmax.aVar[:]
    
    tminDly = np.zeros((pRastTmin.days.size,stns.size),dtype=np.float32)*np.float32(np.nan)
    tmaxDly = np.zeros((pRastTmax.days.size,stns.size),dtype=np.float32)*np.float32(np.nan)
    
    blank = np.ones(pRastTmax.days.size,dtype=np.float32)*np.nan
    
    schk = status_check(stns.size,50)
    
    for stn,x in zip(stns,np.arange(stns.size)):
        
        try:
            #col,row = pRastTmin.getGridCellOffset(stn[LON], stn[LAT])
            #tminDly[:,x] = aTmin[:,row,col]
            tminDly[:,x] = pRastTmin.getTimeSeries(stn[LON], stn[LAT])
            
            #col,row = pRastTmax.getGridCellOffset(stn[LON], stn[LAT])
            #tmaxDly[:,x] = aTmax[:,row,col]
            tmaxDly[:,x] = pRastTmax.getTimeSeries(stn[LON], stn[LAT])
        
        except Exception:
            print "ERROR: %s|%s|%0.4f|%0.4f"%(stns[STN_ID][x],stns[STN_NAME][x],stns[LON][x],stns[LAT][x])
            tminDly[:,x] = blank
            tmaxDly[:,x] = blank
        
        schk.increment()
            
    np.save('/projects/daymet2/ds_compare/daily/interptwxstns_tmin_prism.npy', tminDly)
    np.save('/projects/daymet2/ds_compare/daily/interptwxstns_tmax_prism.npy', tmaxDly)

def buildDaymetDaily():
    
    stns = np.load('/projects/daymet2/ds_compare/daily/stns.npy')
    
    dpt = DaymetPt('/projects/daymet2/ds_compare/daily/daymet_pts/')
    #initialize days data
    dpt.getTair('GHCN_USC00349017')
    
    ndays = np.sum(dpt.normMask)
    
    tminDly = np.zeros((ndays,stns.size),dtype=np.float32)*np.nan
    tmaxDly = np.zeros((ndays,stns.size),dtype=np.float32)*np.nan
    
    blank = np.ones(ndays,dtype=np.float32)*np.nan
    
    schk = status_check(stns.size,50)
    
    for stn,x in zip(stns,np.arange(stns.size)):
        
        try:
            
            tminStn,tmaxStn = dpt.getTair(stn[STN_ID])[0:2]
            tminDly[:,x] = tminStn
            tmaxDly[:,x] = tmaxStn
        
        except Exception:
            print "ERROR: %s|%s|%0.4f|%0.4f"%(stns[STN_ID][x],stns[STN_NAME][x],stns[LON][x],stns[LAT][x])
            tminDly[:,x] = blank
            tmaxDly[:,x] = blank
        
        schk.increment()
            
    np.save('/projects/daymet2/ds_compare/daily/interp_tmin_daymet.npy', tminDly)
    np.save('/projects/daymet2/ds_compare/daily/interp_tmax_daymet.npy', tmaxDly)

class DaymetPt():
    
    def __init__(self,pathData):
        
        self.pathData = pathData
        self.daysDaymet = None 
        self.days = None
        self.dayMask = None
        self.normMask = None
        
    def getTair(self,stnid):
        
        a = np.loadtxt("".join([self.pathData,stnid,'.csv']), delimiter=",",skiprows=7,usecols=[0,1,2,3])
        
        if self.daysDaymet is None:
            self.daysDaymet = utld.get_days_metadata_daymet(a[:,0].astype(np.int), a[:,1].astype(np.int))
            
            startDate = datetime(self.daysDaymet[YEAR][0],1,1)
            endDate = datetime(self.daysDaymet[YEAR][-1],12,31)
            
            self.days = utld.get_days_metadata(startDate,endDate)
            self.dayMask = np.in1d(self.days[YMD],self.daysDaymet[YMD],True)
            self.normMask = np.logical_and(self.days[YEAR] >= 1981,self.days[YEAR] <= 2010)
            
        tminDaymet = a[:,3]
        tmaxDaymet = a[:,2]
        
        tmin = np.ones(self.days.size)*np.nan
        tmax = np.ones(self.days.size)*np.nan
        
        tmin[self.dayMask] = tminDaymet
        tmax[self.dayMask] = tmaxDaymet
        
        return tmin[self.normMask],tmax[self.normMask],self.days[self.normMask]

def shiftTair(tair,shift=-1):
    
    if shift != -1 and shift != 1:
        raise Exception('Unsupported shift')
    
    tairShift = np.roll(tair,shift)
    
    if shift == -1:
        tairShift[-1] = np.nan
    elif shift == 1:
        tairShift[0] = np.nan
    
    return tairShift

def ptErrorStats(interpTair,obsTair,days,stn):
    
    interpTair= np.require(interpTair,dtype=np.float64)
    obsTair= np.require(obsTair,dtype=np.float64)
    
    def getMaeBiasR2(aInterpTair,aObsTair):
        
        aMaskOverlap = np.nonzero(np.logical_and(np.isfinite(aInterpTair),np.isfinite(aObsTair)))[0]
        
        if aMaskOverlap.size <= 1:
            raise Exception("No obs overlap for "+stn[STN_ID])
        
        aObsTair = np.take(aObsTair, aMaskOverlap)
        aInterpTair = np.take(aInterpTair, aMaskOverlap)
        aDays = np.take(days, aMaskOverlap) 
        
        #mths,ann,warm season,cold season
        r2 = np.zeros(15)
        mae = np.zeros(15)
        bias = np.zeros(15)
        
        r2[-1] = stats.linregress(aInterpTair, aObsTair)[2]**2
        
        difs = aInterpTair-aObsTair
        mae[-1] = np.mean(np.abs(difs))
        bias[-1] = np.mean(difs)
        
        for mth in np.arange(1,13):
            
            mthMask = np.nonzero(aDays[MONTH]==mth)[0] 
            
            if mthMask.size <= 1:
                raise Exception("No obs overlap for "+str(mth)+" "+stn[STN_ID])
            
            aInterpTairMth = np.take(aInterpTair, mthMask)
            aObsTairMth = np.take(aObsTair, mthMask)
            
            r2[mth-1] = stats.linregress(aInterpTairMth, aObsTairMth)[2]**2
            
            difs = aInterpTairMth-aObsTairMth
            mae[mth-1] = np.mean(np.abs(difs))
            bias[mth-1] = np.mean(difs)
    
        return mae,bias,r2
    
    interpTairBack = shiftTair(interpTair, -1)
    interpTairForward = shiftTair(interpTair, 1)
    
    maeOrig,biasOrig,r2Orig = getMaeBiasR2(interpTair, obsTair)
    maeBack,biasBack,r2Back = getMaeBiasR2(interpTairBack, obsTair)
    maeForward,biasForward,r2Forward = getMaeBiasR2(interpTairForward, obsTair)
    
    maeAll = np.vstack((maeOrig,maeBack,maeForward))
    biasAll = np.vstack((biasOrig,biasBack,biasForward))
    r2All = np.vstack((r2Orig,r2Back,r2Forward))
    
    idxMae = np.argmin(maeAll,0)
    idxCol = np.arange(maeAll.shape[1])
    
    maeFnl = maeAll[idxMae,idxCol]
    biasFnl = biasAll[idxMae,idxCol]
    r2Fnl = np.max(r2All,0)
    
    return maeFnl,biasFnl,r2Fnl

def calcErrStats():
    stns = np.load('/projects/daymet2/ds_compare/daily/topowx_stns.npy')
    
    dbHomog = StationDataDb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc',startend_ymd=(19810101,20121231))
    #dbRaw = StationDataDb('/projects/daymet2/station_data/all/all_1948_2012.nc',startend_ymd=(19810101,20121231))
    
    days = utld.get_days_metadata(datetime(1981,1,1), datetime(2012,12,31))
    
    #homogTmin = np.load('/projects/daymet2/ds_compare/daily/obs_tmin_homog.npy')
    #homogTmax = np.load('/projects/daymet2/ds_compare/daily/obs_tmax_homog.npy')
#    rawTmin = np.load('/projects/daymet2/ds_compare/daily/obs_tmin_raw.npy')
#    rawTmax = np.load('/projects/daymet2/ds_compare/daily/obs_tmax_raw.npy')

    daysTwx = utld.get_days_metadata(datetime(1980,1,1), datetime(2012,12,31))
    yrMaskTwx = np.nonzero(daysTwx[YEAR]>=1981)[0]
    interpTminTwx = np.load('/projects/daymet2/ds_compare/daily/interptwxstns_tmin_twx.npy')[yrMaskTwx,:]
    #interpTminPrism = np.load('/projects/daymet2/ds_compare/daily/interptwxstns_tmin_prism.npy')
    #interpTminDaymet = np.load('/projects/daymet2/ds_compare/daily/interp_tmin_daymet.npy')
    interpTmaxTwx = np.load('/projects/daymet2/ds_compare/daily/interptwxstns_tmax_twx.npy')[yrMaskTwx,:]
    #interpTmaxPrism = np.load('/projects/daymet2/ds_compare/daily/interptwxstns_tmax_prism.npy')
    #interpTmaxDaymet = np.load('/projects/daymet2/ds_compare/daily/interp_tmax_daymet.npy')
    
    #mths,ann,warm season,cold season
    twxStatsTmin = np.zeros((3,13,stns.size))
    #prismStatsTmin = np.zeros((3,13,stns.size))
    #prismR2Tmin = np.zeros((13,stns.size))
    #daymetR2Tmin = np.zeros((13,stns.size))
    twxStatsTmax = np.zeros((3,13,stns.size))
    #prismStatsTmax = np.zeros((3,13,stns.size))
    #prismR2Tmax = np.zeros((13,stns.size))
    #daymetR2Tmax = np.zeros((13,stns.size))
    
    stchk = status_check(stns.size, 100)
    for stn,x in zip(stns,np.arange(stns.size)):
        
        try:
        
            obsStnTmin = dbHomog.load_all_stn_obs_var(stn[STN_ID], 'tmin')[0]
            stnMae,stnBias,stnR2 = ptErrorStats(interpTminTwx[:,x], obsStnTmin, days, stn)
            twxStatsTmin[0,:,x] = stnMae
            twxStatsTmin[1,:,x] = stnBias
            twxStatsTmin[2,:,x] = stnR2
            
#            obsStnTmin = dbRaw.load_all_stn_obs_var(stn[STN_ID], 'tmin')[0]
#            stnMae,stnBias,stnR2 = ptErrorStats(interpTminPrism[:,x], obsStnTmin, days, stn)
#            prismStatsTmin[0,:,x] = stnMae
#            prismStatsTmin[1,:,x] = stnBias
#            prismStatsTmin[2,:,x] = stnR2
            
            
            
            
            #interpTminTwx[:,x] = ptErrorStats(interpTminTwx[:,x], obsStnTmin, days, stn)
            
            #prismR2Tmin[:,x] = ptErrorStats(interpTminPrism[:,x], rawTmin[:,x], days, stn)
            #daymetR2Tmin[:,x] = ptErrorStats(interpTminDaymet[:,x], rawTmin[:,x], days, stn)
        
        except Exception:
            twxStatsTmin[:,:,x] = np.nan
            #prismStatsTmin[:,:,x] = np.nan
            #prismR2Tmin[:,x] = np.nan
            #daymetR2Tmin[:,x] = np.nan
            print "ERROR: Could not get TMIN error stats for %s|%s"%(stn[STN_ID],stn[STN_NAME])
            
        try:
        
            obsStnTmax = dbHomog.load_all_stn_obs_var(stn[STN_ID], 'tmax')[0]
            stnMae,stnBias,stnR2 = ptErrorStats(interpTmaxTwx[:,x], obsStnTmax, days, stn)
            twxStatsTmax[0,:,x] = stnMae
            twxStatsTmax[1,:,x] = stnBias
            twxStatsTmax[2,:,x] = stnR2
            
#            obsStnTmax = dbRaw.load_all_stn_obs_var(stn[STN_ID], 'tmax')[0]
#            stnMae,stnBias,stnR2 = ptErrorStats(interpTmaxPrism[:,x], obsStnTmax, days, stn)
#            prismStatsTmax[0,:,x] = stnMae
#            prismStatsTmax[1,:,x] = stnBias
#            prismStatsTmax[2,:,x] = stnR2
        
            #twxR2Tmax[:,x] = ptErrorStats(interpTmaxTwx[:,x], homogTmax[:,x], days, stn)
            #prismR2Tmax[:,x] = ptErrorStats(interpTmaxPrism[:,x], rawTmax[:,x], days, stn)
            #daymetR2Tmax[:,x] = ptErrorStats(interpTmaxDaymet[:,x], rawTmax[:,x], days, stn)
        
        except Exception:
            twxStatsTmax[:,:,x] = np.nan
            #prismStatsTmax[:,:,x] = np.nan
            #prismR2Tmax[:,x] = np.nan
            #daymetR2Tmax[:,x] = np.nan
            print "ERROR: Could not get TMAX error stats for %s|%s"%(stn[STN_ID],stn[STN_NAME])
        
        stchk.increment()

    np.save('/projects/daymet2/ds_compare/daily/errstats_tmin_twx.npy', twxStatsTmin)
    np.save('/projects/daymet2/ds_compare/daily/errstats_tmax_twx.npy', twxStatsTmax)
    #np.save('/projects/daymet2/ds_compare/daily/errstats_tmin_prism.npy', prismStatsTmin)
    #np.save('/projects/daymet2/ds_compare/daily/errstats_tmax_prism.npy', prismStatsTmax)
    
#    np.save('/projects/daymet2/ds_compare/daily/r2_tmin_prism.npy', prismR2Tmin)
#    np.save('/projects/daymet2/ds_compare/daily/r2_tmax_prism.npy', prismR2Tmax)
    #np.save('/projects/daymet2/ds_compare/daily/r2_tmin_daymet.npy', daymetR2Tmin)
    #np.save('/projects/daymet2/ds_compare/daily/r2_tmax_daymet.npy', daymetR2Tmax)

def plotR2Maps():
    stns = np.load('/projects/daymet2/ds_compare/daily/stns.npy')
    climDivs = np.unique(stns['neon'][np.isfinite(stns['neon'])])
    
    mth = -1
    
    r2TminTwx = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_twx.npy')[mth,:]
    r2TmaxTwx = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_twx.npy')[mth,:]
    
    r2TminPrism = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_prism.npy')[mth,:]
    r2TmaxPrism = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_prism.npy')[mth,:]
    
    r2TminDaymet = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_daymet.npy')[mth,:]
    r2TmaxDaymet = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_daymet.npy')[mth,:]
    
#    
#    r2TminTwx = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_twx.npy')[0:12,:]
#    r2TmaxTwx = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_twx.npy')[0:12,:]
#    
#    r2TminPrism = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_prism.npy')[0:12,:]
#    r2TmaxPrism = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_prism.npy')[0:12,:]
#    
#    r2TminDaymet = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_daymet.npy')[0:12,:]
#    r2TmaxDaymet = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_daymet.npy')[0:12,:] 
        
    
    r2TminTwx = np.ma.masked_array(r2TminTwx,np.isnan(r2TminTwx))
    r2TmaxTwx = np.ma.masked_array(r2TmaxTwx,np.isnan(r2TmaxTwx))
    r2TminPrism = np.ma.masked_array(r2TminPrism,np.isnan(r2TminPrism))
    r2TmaxPrism = np.ma.masked_array(r2TmaxPrism,np.isnan(r2TmaxPrism))
    r2TminDaymet = np.ma.masked_array(r2TminDaymet,np.isnan(r2TminDaymet))
    r2TmaxDaymet = np.ma.masked_array(r2TmaxDaymet,np.isnan(r2TmaxDaymet))
    
    maskAllTmin = np.logical_or(np.logical_or(r2TminTwx.mask,r2TminPrism.mask),r2TminDaymet.mask)
    r2TminTwx[maskAllTmin] = np.ma.masked
    r2TminPrism[maskAllTmin] = np.ma.masked
    r2TminDaymet[maskAllTmin] = np.ma.masked
    
    maskAllTmax = np.logical_or(np.logical_or(r2TmaxTwx.mask,r2TmaxPrism.mask),r2TmaxDaymet.mask)
    r2TmaxTwx[maskAllTmax] = np.ma.masked
    r2TmaxPrism[maskAllTmax] = np.ma.masked
    r2TmaxDaymet[maskAllTmax] = np.ma.masked
    
    
#    r2TminTwx = np.ma.mean(r2TminTwx,axis=0)
#    r2TmaxTwx = np.ma.mean(r2TmaxTwx,axis=0)
#    r2TminPrism = np.ma.mean(r2TminPrism,axis=0)
#    r2TmaxPrism = np.ma.mean(r2TmaxPrism,axis=0)
#    r2TminDaymet = np.ma.mean(r2TminDaymet,axis=0)
#    r2TmaxDaymet = np.ma.mean(r2TmaxDaymet,axis=0)
#    print r2TminPrism.shape
                
    fontsize=12
    
    r2DivsTminTwx = []
    r2DivsTmaxTwx = [] 
    r2DivsTminPrism = []
    r2DivsTmaxPrism = []
    r2DivsTminDaymet = []
    r2DivsTmaxDaymet = []
    
    for div in climDivs:
        
        maskDivStns = stns[NEON]==div
        
        r2DivsTminTwx.append(np.ma.mean(r2TminTwx[maskDivStns]))
        r2DivsTmaxTwx.append(np.ma.mean(r2TmaxTwx[maskDivStns]))
        r2DivsTminPrism.append(np.ma.mean(r2TminPrism[maskDivStns]))
        r2DivsTmaxPrism.append(np.ma.mean(r2TmaxPrism[maskDivStns]))
        r2DivsTminDaymet.append(np.ma.mean(r2TminDaymet[maskDivStns]))
        r2DivsTmaxDaymet.append(np.ma.mean(r2TmaxDaymet[maskDivStns]))

    r2DivsTminTwx = np.array(r2DivsTminTwx)
    r2DivsTmaxTwx = np.array(r2DivsTmaxTwx)
    r2DivsTminPrism = np.array(r2DivsTminPrism)
    r2DivsTmaxPrism = np.array(r2DivsTmaxPrism)
    r2DivsTminDaymet = np.array(r2DivsTminDaymet)
    r2DivsTmaxDaymet = np.array(r2DivsTmaxDaymet)
    
    #Get idxs to sort climate divisions by #
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
    
    print "Starting to plot...."
    cf = plt.gcf()
    
    grid = ImageGrid(cf,111,nrows_ncols=(3,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")

    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    
    
    
    #cmap = cm.hot_r
    r2All = np.concatenate([r2DivsTminTwx,r2DivsTmaxTwx,r2DivsTminPrism,r2DivsTmaxPrism,r2DivsTminDaymet,r2DivsTmaxDaymet])
    #norm = Normalize(np.min(errAll[np.isfinite(errAll)]), np.max(errAll[np.isfinite(errAll)]))
    print np.min(r2All[np.isfinite(r2All)]),np.max(r2All[np.isfinite(r2All)])
    #sys.exit()
    ####################################################################
#    cmap4 = brewer2mpl.get_map('YlOrRd', 'Sequential', 8, reverse=False)
#    cmap4 = cmap4.get_mpl_colormap()
#    cmap5 = brewer2mpl.get_map('YlOrRd', 'Sequential', 9, reverse=False)
#    cmap5colors = cmap5.mpl_colors
#    
#    cmapFnl = cm.hot_r.from_list('Custom cmap', cmap5colors[0:8], cmap4.N)
#    cmapFnl.set_over(cmap5colors[-1])
#    norm = BoundaryNorm([0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0], cmapFnl.N)
    ####################################################################
    cmapB = brewer2mpl.get_map('YlOrRd', 'Sequential', 6, reverse=False)
    cmap = cmapB.get_mpl_colormap()
    cmapcolors = cmapB.mpl_colors
    
    cmapFnl = cm.hot_r.from_list('Custom cmap', cmapcolors[1:], cmap.N)
    cmapFnl.set_under(cmapcolors[0])
    #norm = BoundaryNorm(np.linspace(np.min(r2All[np.isfinite(r2All)]),np.max(r2All[np.isfinite(r2All)]),9), cmapFnl.N)
    norm = BoundaryNorm(np.linspace(.9,1,6), cmapFnl.N)
    #norm = BoundaryNorm(np.linspace(0.84,1.0,9), cmapFnl.N)
    
    sm = cm.ScalarMappable(norm, cmapFnl)
    sm.set_array(r2All[np.isfinite(r2All)])
    
    def drawErrorMap(i,errs):
        
        colors = sm.to_rgba(errs)
        
        gridCell = grid[i]   
        m.ax = gridCell
        m.drawcountries(linewidth=0.5,color='white')
            
        lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
        lineCollect = lineCollect[sidx]
            
        #Put Climate Divisions on map
        for x in np.arange(lineCollect.size):
            
            lines = lineCollect[x]
            
            if np.isnan(errs[x]):
                lines.set_facecolors('grey')
            else:
                lines.set_facecolors(colors[x])
            lines.set_edgecolors('#8C8C8C')
            lines.set_linewidth(0.5)
            gridCell.add_collection(lines)
    
    drawErrorMap(0,r2DivsTminTwx)
    m.ax = grid[0]
    m.drawmeridians([-103])
    #cbar = grid[0].cax.colorbar(sm,extend='max')
    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0],extend='min')
    cbar.set_label(r"$\bar{R}^2$",fontsize=fontsize)
    grid[0].set_title(r"$TN_a$")
    grid[1].set_title(r"$TX_a$")
    grid[0].set_ylabel('TopoWx')
    grid[0].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TminTwx),),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TmaxTwx),),transform=grid[1].transAxes,fontsize=10)
    drawErrorMap(1,r2DivsTmaxTwx)
    
    drawErrorMap(2,r2DivsTminPrism)
    drawErrorMap(3,r2DivsTmaxPrism)
    grid[2].set_ylabel('PRISM')
    grid[2].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TminPrism),),transform=grid[2].transAxes,fontsize=10)
    grid[3].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TmaxPrism),),transform=grid[3].transAxes,fontsize=10)
    
    drawErrorMap(4,r2DivsTminDaymet)
    drawErrorMap(5,r2DivsTmaxDaymet)
    grid[4].set_ylabel('Daymet')
    grid[4].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TminDaymet),),transform=grid[4].transAxes,fontsize=10)
    grid[5].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TmaxDaymet),),transform=grid[5].transAxes,fontsize=10)
    
    #drawErrorMap(4,maeNormDivsTminDaymet)
    #drawErrorMap(5,maeNormDivsTmaxDaymet)
    #grid[4].set_ylabel('Daymet')
    #grid[4].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminDaymet),),transform=grid[4].transAxes,fontsize=10)
    #grid[5].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxDaymet),),transform=grid[5].transAxes,fontsize=10)
    
    


    fig =plt.gcf()
#    #fig.set_size_inches(8*2,6*3)
    fig.set_size_inches(8,6)
#    fig.subplots_adjust(hspace=0.05)
    #plt.tight_layout()
    plt.savefig('/projects/daymet2/docs/final_writeup/dsCompareR2Map.png',dpi=300)
    plt.show()

def plotErrMaps():
    stns = np.load('/projects/daymet2/ds_compare/daily/topowx_stns.npy')
    stnsNorm = np.load('/projects/daymet2/ds_compare/daily/stns.npy')
    climDivs = np.unique(stns['neon'][np.isfinite(stns['neon'])])
    
    #stnMask = np.in1d(stns[STN_ID], stnsNorm[STN_ID], True)
    stnMask = ~np.char.startswith(stns[STN_ID], 'SNOTEL')
    #stnMask = np.logical_and(stns[TDI] <= .5,stns[LON] <= -103)
    
    tminTwxXval = np.load('/projects/daymet2/ds_compare/daily/errstats_tmin_twxxval.npy')
    tmaxTwxXval = np.load('/projects/daymet2/ds_compare/daily/errstats_tmax_twxxval.npy')
    tminTwx = np.load('/projects/daymet2/ds_compare/daily/errstats_tmin_twxxval.npy')
    tmaxTwx = np.load('/projects/daymet2/ds_compare/daily/errstats_tmax_twxxval.npy')
    
    tminPrism = np.load('/projects/daymet2/ds_compare/daily/errstats_tmin_prism.npy')
    tmaxPrism = np.load('/projects/daymet2/ds_compare/daily/errstats_tmax_prism.npy')
    
    
    tminTwxXval[:,:,stnMask] = np.nan
    tmaxTwxXval[:,:,stnMask] = np.nan
    tminTwx[:,:,stnMask] = np.nan
    tmaxTwx[:,:,stnMask] = np.nan
    tminPrism[:,:,stnMask] = np.nan
    tmaxPrism[:,:,stnMask] = np.nan

    tminTwxXval = np.ma.masked_array(tminTwxXval,np.isnan(tminTwxXval))
    tmaxTwxXval = np.ma.masked_array(tmaxTwxXval,np.isnan(tmaxTwxXval))
    tminTwx = np.ma.masked_array(tminTwx,np.isnan(tminTwx))
    tmaxTwx = np.ma.masked_array(tmaxTwx,np.isnan(tmaxTwx))
    tminPrism = np.ma.masked_array(tminPrism,np.isnan(tminPrism))
    tmaxPrism = np.ma.masked_array(tmaxPrism,np.isnan(tmaxPrism))

    maskAllTmin = np.logical_or(np.logical_or(tminTwx.mask,tminPrism.mask),tminTwxXval.mask)
    tminTwxXval[maskAllTmin] = np.ma.masked
    tminTwx[maskAllTmin] = np.ma.masked
    tminPrism[maskAllTmin] = np.ma.masked
    
    maskAllTmax = np.logical_or(np.logical_or(tmaxTwx.mask,tmaxPrism.mask),tmaxTwxXval.mask)
    tmaxTwxXval[maskAllTmax] = np.ma.masked
    tmaxTwx[maskAllTmax] = np.ma.masked
    tmaxPrism[maskAllTmax] = np.ma.masked
    
    print "Twx Tmin MAE"
    print np.ma.mean(tminTwx[0,:,:],axis=1)
    print "PRISM Tmin MAE"
    print np.ma.mean(tminPrism[0,:,:],axis=1)
    print "Twx Tmax MAE"
    print np.ma.mean(tmaxTwx[0,:,:],axis=1)
    print "PRISM Tmax MAE"
    print np.ma.mean(tmaxPrism[0,:,:],axis=1)
    
    print "Twx Tmin aBias"
    print np.ma.mean(np.abs(tminTwx[1,:,:]),axis=1)
    print "PRISM Tmin aBias"
    print np.ma.mean(np.abs(tminPrism[1,:,:]),axis=1)
    print "Twx Tmax aBias"
    print np.ma.mean(np.abs(tmaxTwx[1,:,:]),axis=1)
    print "PRISM Tmax aBias"
    print np.ma.mean(np.abs(tmaxPrism[1,:,:]),axis=1)
    
    print "Twx Tmin Bias"
    print np.ma.mean(tminTwx[1,:,:],axis=1)
    print "PRISM Tmin Bias"
    print np.ma.mean(tminPrism[1,:,:],axis=1)
    print "Twx Tmax Bias"
    print np.ma.mean(tmaxTwx[1,:,:],axis=1)
    print "PRISM Tmax Bias"
    print np.ma.mean(tmaxPrism[1,:,:],axis=1)
    
    tminAnnBiasPrism = tmaxPrism[1,-1,:]
    stnsErr = stns[~tminAnnBiasPrism.mask]
    
    print stns[np.nonzero(tminAnnBiasPrism<-100)[0]]
    
    plt.plot(stnsErr[TDI],tminAnnBiasPrism[~tminAnnBiasPrism.mask],'.')
    plt.show()
    
#    r2TminTwx = np.ma.mean(r2TminTwx,axis=0)
#    r2TmaxTwx = np.ma.mean(r2TmaxTwx,axis=0)
#    r2TminPrism = np.ma.mean(r2TminPrism,axis=0)
#    r2TmaxPrism = np.ma.mean(r2TmaxPrism,axis=0)
#    r2TminDaymet = np.ma.mean(r2TminDaymet,axis=0)
#    r2TmaxDaymet = np.ma.mean(r2TmaxDaymet,axis=0)
#    print r2TminPrism.shape
                
#    fontsize=12
#    
#    r2DivsTminTwx = []
#    r2DivsTmaxTwx = [] 
#    r2DivsTminPrism = []
#    r2DivsTmaxPrism = []
#    r2DivsTminDaymet = []
#    r2DivsTmaxDaymet = []
#    
#    for div in climDivs:
#        
#        maskDivStns = stns[NEON]==div
#        
#        r2DivsTminTwx.append(np.ma.mean(r2TminTwx[maskDivStns]))
#        r2DivsTmaxTwx.append(np.ma.mean(r2TmaxTwx[maskDivStns]))
#        r2DivsTminPrism.append(np.ma.mean(r2TminPrism[maskDivStns]))
#        r2DivsTmaxPrism.append(np.ma.mean(r2TmaxPrism[maskDivStns]))
#        r2DivsTminDaymet.append(np.ma.mean(r2TminDaymet[maskDivStns]))
#        r2DivsTmaxDaymet.append(np.ma.mean(r2TmaxDaymet[maskDivStns]))
#
#    r2DivsTminTwx = np.array(r2DivsTminTwx)
#    r2DivsTmaxTwx = np.array(r2DivsTmaxTwx)
#    r2DivsTminPrism = np.array(r2DivsTminPrism)
#    r2DivsTmaxPrism = np.array(r2DivsTmaxPrism)
#    r2DivsTminDaymet = np.array(r2DivsTminDaymet)
#    r2DivsTmaxDaymet = np.array(r2DivsTmaxDaymet)
#    
#    #Get idxs to sort climate divisions by #
#    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
#    recs = shpClimDivArea.records()
#    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
#    sidx = np.argsort(climDivsShp)
#    climDivsShp = climDivsShp[sidx]
#    
#    print "Starting to plot...."
#    cf = plt.gcf()
#    
#    grid = ImageGrid(cf,111,nrows_ncols=(3,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")
#
#    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
#                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
#    
#    
#    
#    #cmap = cm.hot_r
#    r2All = np.concatenate([r2DivsTminTwx,r2DivsTmaxTwx,r2DivsTminPrism,r2DivsTmaxPrism,r2DivsTminDaymet,r2DivsTmaxDaymet])
#    #norm = Normalize(np.min(errAll[np.isfinite(errAll)]), np.max(errAll[np.isfinite(errAll)]))
#    print np.min(r2All[np.isfinite(r2All)]),np.max(r2All[np.isfinite(r2All)])
#    #sys.exit()
#    ####################################################################
##    cmap4 = brewer2mpl.get_map('YlOrRd', 'Sequential', 8, reverse=False)
##    cmap4 = cmap4.get_mpl_colormap()
##    cmap5 = brewer2mpl.get_map('YlOrRd', 'Sequential', 9, reverse=False)
##    cmap5colors = cmap5.mpl_colors
##    
##    cmapFnl = cm.hot_r.from_list('Custom cmap', cmap5colors[0:8], cmap4.N)
##    cmapFnl.set_over(cmap5colors[-1])
##    norm = BoundaryNorm([0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0], cmapFnl.N)
#    ####################################################################
#    cmapB = brewer2mpl.get_map('YlOrRd', 'Sequential', 6, reverse=False)
#    cmap = cmapB.get_mpl_colormap()
#    cmapcolors = cmapB.mpl_colors
#    
#    cmapFnl = cm.hot_r.from_list('Custom cmap', cmapcolors[1:], cmap.N)
#    cmapFnl.set_under(cmapcolors[0])
#    #norm = BoundaryNorm(np.linspace(np.min(r2All[np.isfinite(r2All)]),np.max(r2All[np.isfinite(r2All)]),9), cmapFnl.N)
#    norm = BoundaryNorm(np.linspace(.9,1,6), cmapFnl.N)
#    #norm = BoundaryNorm(np.linspace(0.84,1.0,9), cmapFnl.N)
#    
#    sm = cm.ScalarMappable(norm, cmapFnl)
#    sm.set_array(r2All[np.isfinite(r2All)])
#    
#    def drawErrorMap(i,errs):
#        
#        colors = sm.to_rgba(errs)
#        
#        gridCell = grid[i]   
#        m.ax = gridCell
#        m.drawcountries(linewidth=0.5,color='white')
#            
#        lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
#        lineCollect = lineCollect[sidx]
#            
#        #Put Climate Divisions on map
#        for x in np.arange(lineCollect.size):
#            
#            lines = lineCollect[x]
#            
#            if np.isnan(errs[x]):
#                lines.set_facecolors('grey')
#            else:
#                lines.set_facecolors(colors[x])
#            lines.set_edgecolors('#8C8C8C')
#            lines.set_linewidth(0.5)
#            gridCell.add_collection(lines)
#    
#    drawErrorMap(0,r2DivsTminTwx)
#    m.ax = grid[0]
#    m.drawmeridians([-103])
#    #cbar = grid[0].cax.colorbar(sm,extend='max')
#    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0],extend='min')
#    cbar.set_label(r"$\bar{R}^2$",fontsize=fontsize)
#    grid[0].set_title(r"$TN_a$")
#    grid[1].set_title(r"$TX_a$")
#    grid[0].set_ylabel('TopoWx')
#    grid[0].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TminTwx),),transform=grid[0].transAxes,fontsize=10)
#    grid[1].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TmaxTwx),),transform=grid[1].transAxes,fontsize=10)
#    drawErrorMap(1,r2DivsTmaxTwx)
#    
#    drawErrorMap(2,r2DivsTminPrism)
#    drawErrorMap(3,r2DivsTmaxPrism)
#    grid[2].set_ylabel('PRISM')
#    grid[2].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TminPrism),),transform=grid[2].transAxes,fontsize=10)
#    grid[3].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TmaxPrism),),transform=grid[3].transAxes,fontsize=10)
#    
#    drawErrorMap(4,r2DivsTminDaymet)
#    drawErrorMap(5,r2DivsTmaxDaymet)
#    grid[4].set_ylabel('Daymet')
#    grid[4].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TminDaymet),),transform=grid[4].transAxes,fontsize=10)
#    grid[5].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(np.ma.mean(r2TmaxDaymet),),transform=grid[5].transAxes,fontsize=10)
#    
#    #drawErrorMap(4,maeNormDivsTminDaymet)
#    #drawErrorMap(5,maeNormDivsTmaxDaymet)
#    #grid[4].set_ylabel('Daymet')
#    #grid[4].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminDaymet),),transform=grid[4].transAxes,fontsize=10)
#    #grid[5].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxDaymet),),transform=grid[5].transAxes,fontsize=10)
#    
#    
#
#
#    fig =plt.gcf()
##    #fig.set_size_inches(8*2,6*3)
#    fig.set_size_inches(8,6)
##    fig.subplots_adjust(hspace=0.05)
#    #plt.tight_layout()
#    plt.savefig('/projects/daymet2/docs/final_writeup/dsCompareR2Map.png',dpi=300)
#    plt.show()

def plotR2ErrorBars():
    
    stns = np.load('/projects/daymet2/ds_compare/daily/stns.npy')
        
    r2TminTwx = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_twx.npy')[0:12,:]
    r2TmaxTwx = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_twx.npy')[0:12,:]
    
    r2TminPrism = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_prism.npy')[0:12,:]
    r2TmaxPrism = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_prism.npy')[0:12,:]
    
    r2TminDaymet = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_daymet.npy')[0:12,:]
    r2TmaxDaymet = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_daymet.npy')[0:12,:] 
        
    r2TminTwx = np.ma.masked_array(r2TminTwx,np.isnan(r2TminTwx))
    r2TmaxTwx = np.ma.masked_array(r2TmaxTwx,np.isnan(r2TmaxTwx))
    r2TminPrism = np.ma.masked_array(r2TminPrism,np.isnan(r2TminPrism))
    r2TmaxPrism = np.ma.masked_array(r2TmaxPrism,np.isnan(r2TmaxPrism))
    r2TminDaymet = np.ma.masked_array(r2TminDaymet,np.isnan(r2TminDaymet))
    r2TmaxDaymet = np.ma.masked_array(r2TmaxDaymet,np.isnan(r2TmaxDaymet))
    
    maskAllTmin = np.logical_or(np.logical_or(r2TminTwx.mask,r2TminPrism.mask),r2TminDaymet.mask)
    r2TminTwx[maskAllTmin] = np.ma.masked
    r2TminPrism[maskAllTmin] = np.ma.masked
    r2TminDaymet[maskAllTmin] = np.ma.masked
    
    maskAllTmax = np.logical_or(np.logical_or(r2TmaxTwx.mask,r2TmaxPrism.mask),r2TmaxDaymet.mask)
    r2TmaxTwx[maskAllTmax] = np.ma.masked
    r2TmaxPrism[maskAllTmax] = np.ma.masked
    r2TmaxDaymet[maskAllTmax] = np.ma.masked
    
    maskStns = stns[LON] <= -103.0#stnsNorm[LON] > -99999#stnsNorm[LON] <= -104.0
    print np.sum(maskStns)
    stns = stns[maskStns]
    
    r2TminTwx = np.ma.mean(r2TminTwx[:,maskStns],axis=1)
    r2TmaxTwx = np.ma.mean(r2TmaxTwx[:,maskStns],axis=1)
    r2TminPrism = np.ma.mean(r2TminPrism[:,maskStns],axis=1)
    r2TmaxPrism = np.ma.mean(r2TmaxPrism[:,maskStns],axis=1)
    r2TminDaymet = np.ma.mean(r2TminDaymet[:,maskStns],axis=1)
    r2TmaxDaymet = np.ma.mean(r2TmaxDaymet[:,maskStns],axis=1)
    
##############################################################################

    print "Starting to plot...."

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    bwidth = .25
    xlim = (-.25,12.90)
        
    plt.sca(ax1)
    plt.bar(np.arange(r2TminTwx.size),r2TminTwx,width=bwidth,color="k")
    plt.bar(np.arange(r2TminPrism.size)+bwidth,r2TminPrism,width=bwidth,color="grey")
    plt.bar(np.arange(r2TminDaymet.size)+(bwidth*2),r2TminDaymet,width=bwidth,color="w",hatch="//")
    xlim = plt.xlim(-.2,12)
    plt.xticks(np.arange(r2TminTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),fontsize=10)
    #plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    #plt.ylim((0,1.3))
    plt.ylim(0,.9)
    plt.yticks(np.linspace(0,.9,19),fontsize=10)
    plt.ylabel(r"$\bar{R}^2$",fontsize=10)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("a.",loc="left")
    
    plt.sca(ax2)
    plt.bar(np.arange(r2TmaxTwx.size),r2TmaxTwx,width=bwidth,color="k")
    plt.bar(np.arange(r2TmaxPrism.size)+bwidth,r2TmaxPrism,width=bwidth,color="grey")
    plt.bar(np.arange(r2TmaxDaymet.size)+(bwidth*2),r2TmaxDaymet,width=bwidth,color="w",hatch="//")
    plt.legend(("TopoWx","PRISM","Daymet"),fontsize=10,loc=4)
    #plt.ylim((0,1.3))
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(r2TmaxDaymet.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),fontsize=10)
    #plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("b.",loc="left")
    
    plt.tight_layout()
    
    f.set_size_inches(8,3.2)
    plt.savefig('/projects/daymet2/docs/final_writeup/dsCompareR2Bar.png',dpi=250)
    plt.show()

def plotR2MapsDifs():
    stns = np.load('/projects/daymet2/ds_compare/daily/stns.npy')
    climDivs = np.unique(stns['neon'][np.isfinite(stns['neon'])])
    
    mth = -1
    
    r2TminTwx = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_twx.npy')[0:12,:]
    r2TmaxTwx = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_twx.npy')[0:12,:]
    
    r2TminPrism = np.load('/projects/daymet2/ds_compare/daily/r2_tmin_prism.npy')[0:12,:]
    r2TmaxPrism = np.load('/projects/daymet2/ds_compare/daily/r2_tmax_prism.npy')[0:12,:]   
    
    r2TminTwx = np.ma.masked_array(r2TminTwx,np.isnan(r2TminTwx))
    r2TmaxTwx = np.ma.masked_array(r2TmaxTwx,np.isnan(r2TmaxTwx))
    r2TminPrism = np.ma.masked_array(r2TminPrism,np.isnan(r2TminPrism))
    r2TmaxPrism = np.ma.masked_array(r2TmaxPrism,np.isnan(r2TmaxPrism))
    
    r2TminTwx = np.ma.mean(r2TminTwx,axis=0)
    r2TmaxTwx = np.ma.mean(r2TmaxTwx,axis=0)
    r2TminPrism = np.ma.mean(r2TminPrism,axis=0)
    r2TmaxPrism = np.ma.mean(r2TmaxPrism,axis=0)
    print r2TminPrism.shape
       
    fontsize=12
    
    r2DivsTminTwx = []
    r2DivsTmaxTwx = [] 
    r2DivsTminPrism = []
    r2DivsTmaxPrism = []
    
    for div in climDivs:
        
        maskDivStns = stns[NEON]==div
        
        r2DivsTminTwx.append(np.ma.mean(r2TminTwx[maskDivStns]))
        r2DivsTmaxTwx.append(np.ma.mean(r2TmaxTwx[maskDivStns]))
        r2DivsTminPrism.append(np.ma.mean(r2TminPrism[maskDivStns]))
        r2DivsTmaxPrism.append(np.ma.mean(r2TmaxPrism[maskDivStns]))

    r2DivsTminTwx = np.array(r2DivsTminTwx)
    r2DivsTmaxTwx = np.array(r2DivsTmaxTwx)
    r2DivsTminPrism = np.array(r2DivsTminPrism)
    r2DivsTmaxPrism = np.array(r2DivsTmaxPrism)
    
    #Get idxs to sort climate divisions by #
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
    
    print "Starting to plot...."
    cf = plt.gcf()
    
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")

    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    
    
    r2DivsTminPrism = r2DivsTminPrism - r2DivsTminTwx
    r2DivsTmaxPrism = r2DivsTmaxPrism - r2DivsTmaxTwx
    #cmap = cm.hot_r
    #r2All = np.concatenate([r2DivsTminTwx,r2DivsTmaxTwx,r2DivsTminPrism,r2DivsTmaxPrism])
    r2All = np.concatenate([r2DivsTminPrism,r2DivsTmaxPrism])
    #norm = Normalize(np.min(errAll[np.isfinite(errAll)]), np.max(errAll[np.isfinite(errAll)]))
    print np.min(r2All[np.isfinite(r2All)]),np.max(r2All[np.isfinite(r2All)])
    #sys.exit()
    
    clrsRed = brewer2mpl.get_map('Reds', 'Sequential', 4, reverse=False).mpl_colors
    clrsBlue = brewer2mpl.get_map('Blues', 'Sequential', 4, reverse=True).mpl_colors
    N = brewer2mpl.get_map('Reds', 'Sequential', 6, reverse=False).get_mpl_colormap().N 
    clrsBlue.append("grey")
    clrsBlue.extend(clrsRed)
    clrs = clrsBlue
    
    cmapFnl = cm.hot_r.from_list('Custom cmap', clrs[1:-1], N)
    cmapFnl.set_over(clrs[-1])
    cmapFnl.set_under(clrs[0])
    levels = [-0.2 , -0.15, -0.1 , -0.05,  0.05,  0.1 ,  0.15,  0.2 ]
    norm = BoundaryNorm(levels, cmapFnl.N)
    
    sm = cm.ScalarMappable(norm, cmapFnl)
    sm.set_array(r2All[np.isfinite(r2All)])
    
    def drawErrorMap(i,errs):
        
        colors = sm.to_rgba(errs)
        
        gridCell = grid[i]   
        m.ax = gridCell
        m.drawcountries(linewidth=0.5,color='white')
            
        lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
        lineCollect = lineCollect[sidx]
            
        #Put Climate Divisions on map
        for x in np.arange(lineCollect.size):
            
            lines = lineCollect[x]
            
            if np.isnan(errs[x]):
                lines.set_facecolors('grey')
            else:
                lines.set_facecolors(colors[x])
            lines.set_edgecolors('#8C8C8C')
            lines.set_linewidth(0.5)
            gridCell.add_collection(lines)
    
    drawErrorMap(0,r2DivsTminPrism)
    #cbar = grid[0].cax.colorbar(sm,extend='max')
    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0])#,extend='max')
    cbar.set_label(r"$\bar{R}^2$",fontsize=fontsize)
    grid[0].set_title(r"$TN_a$")
    grid[1].set_title(r"$TX_a$")
    grid[0].set_ylabel('TopoWx')
    grid[0].text(0.025,0.075,"Overall\n$\bar{R}^2$: %.2f$^\circ$C"%(np.ma.mean(difNormsTminTwx),),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxTwx),),transform=grid[1].transAxes,fontsize=10)
    drawErrorMap(1,r2DivsTmaxPrism)
    
#    drawErrorMap(2,r2DivsTminPrism)
#    drawErrorMap(3,r2DivsTmaxPrism)
#    grid[2].set_ylabel('PRISM')
    #grid[2].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminPrism),),transform=grid[2].transAxes,fontsize=10)
    #grid[3].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxPrism),),transform=grid[3].transAxes,fontsize=10)
    
    #drawErrorMap(4,maeNormDivsTminDaymet)
    #drawErrorMap(5,maeNormDivsTmaxDaymet)
    #grid[4].set_ylabel('Daymet')
    #grid[4].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminDaymet),),transform=grid[4].transAxes,fontsize=10)
    #grid[5].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxDaymet),),transform=grid[5].transAxes,fontsize=10)
    
    


#    fig =plt.gcf()
#    #fig.set_size_inches(8*2,6*3)
#    fig.set_size_inches(8,6*1.25)
#    fig.subplots_adjust(hspace=0.05)
    #plt.tight_layout()
    #plt.savefig('/projects/daymet2/docs/final_writeup/climDivErrMaps.png',dpi=150)
    plt.show()
  
  
if __name__ == '__main__':
    
    #stnDataToNpy()
    #buildConusStations()
    #runInterps()
    #buildPrismDaily()
    #buildDaymetDaily()
    #calcErrStats()
    #stnMetaToCsv()
    plotErrMaps()
    #plotR2Maps()
    #plotR2ErrorBars()
    #plotR2MapsDifs()
    
    #stnMetaToDaymetExtract()