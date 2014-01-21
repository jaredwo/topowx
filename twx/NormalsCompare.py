'''
Created on Nov 19, 2013

@author: jared.oyler
'''
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from twx.db.station_data import DTYPE_STN_BASIC,STN_NAME,STN_ID,LON,LAT,ELEV,STATE,station_data_infill,NEON,MASK,BAD
from twx.utils.input_raster import RasterDataset,OutsideExtent
from copy import copy
import twx.interp.interp_tair as it
from twx.db.ushcn import TairAggregate
from twx.utils.status_check import status_check
from multiprocessing import Pool
import twx.utils.util_dates as utld
from datetime import datetime
from twx.utils.util_dates import YEAR

PATH_STNDB_SERIAL_TMIN = '/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc'
PATH_STNDB_SERIAL_TMAX = '/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc'

def saveNpyNorms(inPath,outPath):
    obsFile = open(inPath)
    lines = obsFile.readlines()
    n = len(lines)
    
    obsNorms = np.zeros((12,n))
    
    for x in np.arange(n):
        vals = lines[x].split()
        normsStn = [((np.float(anorm[0:-1])/10.0)-32.0)/1.8 for anorm in vals[1:]]
        obsNorms[:,x] = normsStn
    
    np.save(outPath, obsNorms)
    

def testStnObsMatch():
    stns = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    
    obsFile = open('/projects/daymet2/station_data/ncdc_normals/mly-tmin-normal.txt')
    
    stnIds = []    
    for aline in obsFile.readlines():
        vals = aline.split()
        stnIds.append("".join(['GHCN_',vals[0]]))
    
    stnIds = np.array(stnIds,dtype="<S16")
    print stnIds.size,stns.size
    print np.sum(stnIds != stns[STN_ID])

def saveNpyNormStns():
    
    stnfile = open('/projects/daymet2/station_data/ncdc_normals/temp-inventory.txt')
    stnda_tmin = station_data_infill('/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc','tmin')
    stnda_tmax = station_data_infill('/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc','tmax')
    stnIds = []
    lon = []
    lat = []
    elev = []
    state = []
    name = []
    method = []
    
    for aline in stnfile.readlines():
        vals = aline.split()
        stnIds.append("".join(['GHCN_',vals[0]]))
        lat.append(np.float(vals[1]))
        lon.append(np.float(vals[2]))
        elev.append(np.float(vals[3]))
        state.append(vals[4])
        name.append(" ".join(vals[5:-1]))
        method.append(vals[-1])
    
    stnDtype = copy(DTYPE_STN_BASIC)
    stnDtype.append((NEON, np.float64))
    stnDtype.append(('method', '<S16'))
    
    stns = np.empty(len(stnIds), dtype=stnDtype)
    stns[STN_ID] = stnIds
    stns[LON] = lon
    stns[LAT] = lat
    stns[ELEV] = elev
    stns[STATE] = state
    stns[STN_NAME] = name
    stns['method'] = method
    
    ds = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/climdivLccMerge.tif')
    ndata = ds.gdalDs.GetRasterBand(1).GetNoDataValue()
    
    climDivs = np.ones(stns.size)*np.nan
    
    for stn,x in zip(stns,np.arange(stns.size)):
        
        try:
            climDiv = ds.getDataValue(stn[LON],stn[LAT])
            
            if climDiv == ndata:
                continue
            else:
                climDivs[x] = climDiv
        
        except OutsideExtent:
            continue
    
    stns[NEON] = climDivs
    np.save('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy', stns)

def runInterp(x):

    global ptInterp
    global tagg
    global stns
    
    if ptInterp is None:
        ptInterp = it.buildDefaultPtInterp(norms_only=True)
        tagg = TairAggregate(ptInterp.stn_da_tmax.days)
        #stns = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
        #maskStns = np.isfinite(stns['neon'])
        #stns = stns[maskStns]
        stns = get_twx_stns()
        print "Interpolation initialized"

    error = False
    try:
        tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterp.interpLonLatPt(stns[LON][x], stns[LAT][x], fixInvalid=False, chgLatLon=True)#,stns_rm=stns[STN_ID][x],elev=stns[ELEV][x])
        #tmin_mthly = tagg.dailyToMthly(tmin_dly,-1)[0].data
        #tmax_mthly = tagg.dailyToMthly(tmax_dly,-1)[0].data
#        if x%10 == 0:
#            print "Station count: %d / %d "%(x,stns.size)
    except Exception:
        print "ERROR: Station not in interpolation domain. %s|%s|%0.4f|%0.4f"%(stns[STN_ID][x],stns[STN_NAME][x],stns[LON][x],stns[LAT][x])
        error = True
    
    if not error:
        if np.sum(np.logical_or(np.abs(tmin_norms) > 100,np.abs(tmax_norms) > 100)) > 0:
            error = True
            print "ERROR: Bad interpolated normals. %s|%s|%0.4f|%0.4f"%(stns[STN_ID][x],stns[STN_NAME][x],stns[LON][x],stns[LAT][x])
        
    if error:
        tmin_norms = np.ones(12)*np.nan
        tmax_norms = np.ones(12)*np.nan
        #tmin_mthly = np.ones(tagg.yrMths.size)*np.nan
        #tmax_mthly = np.ones(tagg.yrMths.size)*np.nan
        
    
    return x,tmin_norms,tmax_norms#,tmin_mthly,tmax_mthly

def runTwxInterps():
    
    global ptInterp
    global tagg
    global stns
    ptInterp = None
    tagg = None
    stns = None
    pool = Pool(10)#, initializer=initPtInterp)
    
    #aPtInterp = it.buildDefaultPtInterp(norms_only=True) 
    #aTagg = TairAggregate(aPtInterp.stn_da_tmax.days)
    astns = get_twx_stns()
    #astns = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    #maskStns = np.isfinite(astns['neon'])
    #astns = astns[maskStns]
    
    tminNorms = np.zeros((12,astns.size))*np.nan
    tmaxNorms = np.zeros((12,astns.size))*np.nan
    #tminMthly = np.zeros((aTagg.yrMths.size,astns.size))*np.nan
    #tmaxMthly = np.zeros((aTagg.yrMths.size,astns.size))*np.nan
    
    sck = status_check(astns.size, 100)
    chksize = 100
    
    for x in np.arange(astns.size,step=chksize):
    
        endx = x+chksize
        if endx > astns.size:
            endx = astns.size
        idx = np.arange(x,endx)
        results = pool.map(runInterp,idx,chunksize=1)
        
        for aresult in results:
            #j,tmin_norms,tmax_norms,tmin_mthly,tmax_mthly = aresult
            j,tmin_norms,tmax_norms = aresult
            tminNorms[:,j] = tmin_norms
            tmaxNorms[:,j] = tmax_norms
            #tminMthly[:,j] = tmin_mthly
            #tmaxMthly[:,j] = tmax_mthly
        
        sck.increment(chksize)
    
    pool.close()
    pool.join()
    
    np.save('/projects/daymet2/ds_compare/normals/all_stns_twx_stnids.npy', astns[STN_ID])
    np.save('/projects/daymet2/ds_compare/normals/all_stns_twx_norms_tmin.npy', tminNorms)
    np.save('/projects/daymet2/ds_compare/normals/all_stns_twx_norms_tmax.npy', tmaxNorms)
#    np.save('/projects/daymet2/ds_compare/normals/twxxval_mthly_tmin.npy', tminMthly)
#    np.save('/projects/daymet2/ds_compare/normals/twxxval_mthly_tmax.npy', tmaxMthly)

def buildDaymetNorms():
    
    days = utld.get_days_metadata(datetime(1980,1,1), datetime(2012,12,31))
    tagg = TairAggregate(days)
    mthNames = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    dataPath = '/projects/daymet2/ds_compare/normals/daymet_data/tmax_allyrs/'
#    stns = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
#    maskStns = np.isfinite(stns['neon'])
#    stns = stns[maskStns]
    stns = get_twx_stns()
        
    #tmaxNorms = np.zeros((12,stns.size))
    tmaxMthly = np.zeros((tagg.yrMths.size,stns.size))
    #tmaxMthly = np.zeros((tagg.yrMths.size,stns.size))
    
    sck = status_check(stns.size*np.arange(1980,2013).size, 10)
    
    egDs = RasterDataset("".join([dataPath,"tmax_",str(1980),"_jan.tif"]))
    
    stnsRowCol = []
    for astn in stns:
        stnsRowCol.append(egDs.getRowCol(astn[LON], astn[LAT]))
    
    yrArrays = np.zeros((12,egDs.gdalDs.RasterYSize,egDs.gdalDs.RasterXSize),dtype=np.float32)
    
    for yr in np.arange(1980,2013):
        
        yrMask = tagg.yrMths[YEAR] == yr
        
        print "Loading data for year "+str(yr)
        for j in np.arange(len(mthNames)):
            ads = RasterDataset("".join([dataPath,"tmax_",str(yr),"_",mthNames[j],".tif"]))
            yrArrays[j,:,:] = ads.gdalDs.GetRasterBand(1).ReadAsArray()
    
        tmaxMthlyYr = np.zeros((12,stns.size))
        for x in np.arange(stns.size):

            row,col = stnsRowCol[x]
            tmaxMthlyYr[:,x] = yrArrays[:,row,col]
            sck.increment()
            
        tmaxMthly[yrMask,:] = tmaxMthlyYr
        
    tmaxNorms = tagg.mthlyToMthlyNorms(tmaxMthly).data
    
    np.save('/projects/daymet2/ds_compare/normals/all_stns_daymet_stnids.npy', stns[STN_ID])
    np.save('/projects/daymet2/ds_compare/normals/all_stns_daymet_norms_tmax.npy', tmaxNorms)
    np.save('/projects/daymet2/ds_compare/normals/all_stns_daymet_mthly_tmax.npy', tmaxMthly)
    
def testInterp():
    ptInterp = it.buildDefaultPtInterp()
    
    stns = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    stn = stns[stns[STN_ID]=='GHCN_USC00403420'][0]
    tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterp.interpLonLatPt(stn[LON],stn[LAT], fixInvalid=False, chgLatLon=True)
    
    plt.plot(tmin_norms)
    plt.plot(tmax_norms)
    plt.show()

def buildPrismNorms():
#    stns = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
#    maskStns = np.isfinite(stns['neon'])
#    stns = stns[maskStns]
    stns = get_twx_stns()
    
    prismFpath = '/projects/daymet2/prism/new_norms/'
    
    tmaxNorms = np.zeros((12,stns.size))
    tminNorms = np.zeros((12,stns.size))

    egDs = RasterDataset('/projects/daymet2/prism/new_norms/PRISM_tmax_30yr_normal_800mM2_01_bil.bil')

    tmaxNormsPrism = np.zeros((12,egDs.gdalDs.RasterYSize,egDs.gdalDs.RasterXSize),dtype=np.float32)
    tminNormsPrism = np.zeros((12,egDs.gdalDs.RasterYSize,egDs.gdalDs.RasterXSize),dtype=np.float32)
    
    print "Loading data...."
    for mth in np.arange(1,13):
        ds = RasterDataset("".join([prismFpath,'PRISM_tmax_30yr_normal_800mM2_%02d_bil.bil'%mth]))
        tmaxNormsPrism[mth-1,:,:] = ds.readAsArray()
        
        ds = RasterDataset("".join([prismFpath,'PRISM_tmin_30yr_normal_800mM2_%02d_bil.bil'%mth]))
        tminNormsPrism[mth-1,:,:] = ds.readAsArray()
    
    print "Getting station norms...."
    for x in np.arange(stns.size):
        try:
            row,col = egDs.getRowCol(stns[LON][x],stns[LAT][x])
            tmaxNorms[:,x] = tmaxNormsPrism[:,row,col]
            tminNorms[:,x] = tminNormsPrism[:,row,col]
        except:
            tmaxNorms[:,x] = np.nan
            tminNorms[:,x] = np.nan
    np.save('/projects/daymet2/ds_compare/normals/all_stns_prism_stnids.npy', stns[STN_ID])
    np.save('/projects/daymet2/ds_compare/normals/all_stns_prism_norms_tmax.npy', tmaxNorms)
    np.save('/projects/daymet2/ds_compare/normals/all_stns_prism_norms_tmin.npy', tminNorms)

def get_twx_stns():
    stndaTmax = station_data_infill(PATH_STNDB_SERIAL_TMAX, 'tmax')
    stndaTmin = station_data_infill(PATH_STNDB_SERIAL_TMIN, 'tmin')
    
    stnsTmin = stndaTmin.stns[np.logical_and(np.isfinite(stndaTmin.stns[MASK]),np.isnan(stndaTmin.stns[BAD])) ]
    stnsTmax = stndaTmax.stns[np.logical_and(np.isfinite(stndaTmax.stns[MASK]),np.isnan(stndaTmax.stns[BAD])) ]
    
    stns = stnsTmax
    stnsTmin = stnsTmin[~np.in1d(stnsTmin[STN_ID], stns[STN_ID], True)]
    
    stnsAll = np.hstack((stns,stnsTmin))
    
    sIdx = np.argsort(stnsAll[STN_ID])
    stnsAll = stnsAll[sIdx]

    return stnsAll
  
if __name__ == '__main__':
        
    #saveNpyNormStns()
    #run for both tmin and tmax
#    saveNpyNorms('/projects/daymet2/station_data/ncdc_normals/mly-tmax-normal.txt',
#                 '/projects/daymet2/station_data/ncdc_normals/norm_tmax.npy')
    runTwxInterps()
    #buildDaymetNorms() #run for both tmin and tmax
    #buildPrismNorms()
    #plotNcdcNormsErrorBars()
    #plotNcdcNormsErrorMaps()
    #buildPrismNorms()
    #testInterp()
    #buildDaymetNorms()
    
    #testStnObsMatch()
    
    #saveNpyNormStns()
    

    
#    stns = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
#    
#    dbTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
#    dbTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
#    
#    maskTmin = np.in1d(stns[STN_ID], dbTmin.stns[STN_ID], True)
#    maskTmax = np.in1d(stns[STN_ID], dbTmax.stns[STN_ID], True)
#    maskFnl = np.logical_or(maskTmin,maskTmax)
#    
#    stns = stns[~maskFnl]
#    
#    for stn in stns:
#        print stn[STN_ID]
#    
#    m = Basemap(resolution='i',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
#            lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
#    m.drawcoastlines()
#    m.drawcountries()
#    m.drawstates()
#    m.scatter(stns[LON],stns[LAT],latlon=True)
#    plt.show()