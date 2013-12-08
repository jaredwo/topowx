'''
Created on Nov 19, 2013

@author: jared.oyler
'''
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from db.station_data import DTYPE_STN_BASIC,STN_NAME,STN_ID,LON,LAT,ELEV,STATE,station_data_infill,NEON
from utils.input_raster import RasterDataset,OutsideExtent
from copy import copy
import interp.interp_tair as it
from db.ushcn import TairAggregate
from utils.status_check import status_check
from multiprocessing import Pool
import utils.util_dates as utld
from datetime import datetime
from utils.util_dates import YEAR,DATE
import db.ushcn as ushcn
from qa.qa_temp import imposs_value_mask
from utils.util_ncdf import GeoNc
from netCDF4 import Dataset
from matplotlib.mlab import griddata 
from scipy import stats
from mpl_toolkits.axes_grid1 import ImageGrid
import brewer2mpl
from matplotlib import cm

MISSING_VAL = -9999

def buildUshcnAnoms():
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    strMth = stndaUS.mths[DATE][0]
    endMth = stndaUS.mths[DATE][-1]
    
    uYrs = np.unique(stndaUS.mths[YEAR])
    baseMask = np.logical_and(uYrs >= 1981,uYrs <= 2010)
    
    days = utld.get_days_metadata(datetime(strMth.year,1,1), datetime(endMth.year,12,31))
    tairAgg = ushcn.TairAggregate(days)
    stns = stndaUS.stns
    
    anomAllTmin = np.ma.masked_array(np.zeros((uYrs.size,stns.size)))
    anomAllTmax = np.ma.masked_array(np.zeros((uYrs.size,stns.size)))
    blankAnoms = np.ma.masked_equal(np.zeros(uYrs.size)*MISSING_VAL,MISSING_VAL)
    
    schk = status_check(stns.size, 50)
    for astn,x in zip(stns,np.arange(stns.size)):
        
        obsTmin = stndaUS.loadObs(astn[STN_ID], 'FLs.52i_tmin')
        obsTmax = stndaUS.loadObs(astn[STN_ID], 'FLs.52i_tmax')
        
        annTmin = tairAgg.mthlyToAnn(obsTmin,0)
        annTmax = tairAgg.mthlyToAnn(obsTmax,0)
        
        blanksSet = False
        
        if np.sum(annTmin[baseMask].mask) > 9:
            anomTmin = blankAnoms
            blanksSet = True
        else:
            anomTmin = annTmin - np.ma.mean(annTmin[baseMask],dtype=np.float)
            
        if np.sum(annTmax[baseMask].mask) > 9:
            anomTmin = blankAnoms
            blanksSet = True
        else:
            anomTmax = annTmax - np.ma.mean(annTmax[baseMask],dtype=np.float)
        
        anomAllTmin[:,x] = anomTmin
        anomAllTmax[:,x] = anomTmax
        
        if blanksSet:
            print "%s: removed from a record due to not enough base period obs"%astn[STN_ID]
                   
        schk.increment()
    
    np.save('/projects/daymet2/ds_compare/trends/ushcn_anoms_tmax.npy', np.ma.filled(anomAllTmax,MISSING_VAL)) 
    np.save('/projects/daymet2/ds_compare/trends/ushcn_anoms_tmin.npy', np.ma.filled(anomAllTmin,MISSING_VAL)) 


def buildDaymetAnoms(tairVar):
    days = utld.get_days_metadata(datetime(1980,1,1), datetime(2012,1,1))
    tagg = TairAggregate(days)
    mthNames = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    dataPath = '/projects/daymet2/ds_compare/normals/daymet_data/%s_allyrs/'%tairVar
    
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    stns = stndaUS.stns
    uYrs = np.unique(days[YEAR])
    baseMask = np.logical_and(uYrs >= 1981,uYrs <= 2010)
    
    tairMthlyAll = np.zeros((tagg.yrMths.size,stns.size))
    
    sck = status_check(stns.size*np.arange(1980,2013).size, 10)
    
    egDs = RasterDataset("".join([dataPath,"%s_"%tairVar,str(1980),"_jan.tif"]))
    
    stnsRowCol = []
    for astn in stns:
        stnsRowCol.append(egDs.getRowCol(astn[LON], astn[LAT]))
    
    yrArrays = np.zeros((12,egDs.gdalDs.RasterYSize,egDs.gdalDs.RasterXSize),dtype=np.float32)
    
    for yr in np.arange(1980,2013):
        
        yrMask = tagg.yrMths[YEAR] == yr
        
        print "Loading data for year "+str(yr)
        for j in np.arange(len(mthNames)):
            ads = RasterDataset("".join([dataPath,"%s_"%tairVar,str(yr),"_",mthNames[j],".tif"]))
            yrArrays[j,:,:] = ads.gdalDs.GetRasterBand(1).ReadAsArray()
    
        tairMthlyYr = np.zeros((12,stns.size))
        for x in np.arange(stns.size):

            row,col = stnsRowCol[x]
            tairMthlyYr[:,x] = yrArrays[:,row,col]
            sck.increment()
            
        tairMthlyAll[yrMask,:] = tairMthlyYr
    
    print "Calculating ann anomalies..."
    
    annTair = tagg.mthlyToAnn(tairMthlyAll).data
    annTair[imposs_value_mask(annTair)] = np.nan
    
    annNorms = np.mean(annTair[baseMask,:],axis=0)
    anomAll = annTair - annNorms
    np.save('/projects/daymet2/ds_compare/trends/daymet_anom_%s.npy'%tairVar, anomAll)

def buildPrismAnoms():

    uYrs = np.arange(1948,2013)
    baseMask = np.logical_and(uYrs >= 1981,uYrs <= 2010)
    
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    stns = stndaUS.stns
    
    print "Loading PRISM data...."
    dsTmin = GeoNc(Dataset('/projects/daymet2/prism/4km_annual/tmin/prism4km_tmin_ann1948-2012.nc'))
    aTmin = dsTmin.ds.variables['tmin'][:]
    dsTmax = GeoNc(Dataset('/projects/daymet2/prism/4km_annual/tmax/prism4km_tmax_ann1948-2012.nc'))
    aTmax = dsTmax.ds.variables['tmax'][:]
    
    anomAllTmin = np.zeros((uYrs.size,stns.size))
    anomAllTmax = np.zeros((uYrs.size,stns.size))
    
    print "Calculating annual anoms...."
    schk = status_check(stns.size, 50)
    for astn,x in zip(stns,np.arange(stns.size)):
        
        row,col = dsTmin.getRowCol(astn[LON],astn[LAT])[0:2]
        annTmin = aTmin[:,row,col]
        annTmin[imposs_value_mask(annTmin)] = np.nan
        anomTmin = annTmin - np.mean(annTmin[baseMask])
        
        row,col = dsTmax.getRowCol(astn[LON],astn[LAT])[0:2] 
        annTmax = aTmax[:,row,col]
        annTmax[imposs_value_mask(annTmax)] = np.nan
        anomTmax = annTmax - np.mean(annTmax[baseMask])
                
        anomAllTmin[:,x] = anomTmin
        anomAllTmax[:,x] = anomTmax
        
        schk.increment()

    np.save('/projects/daymet2/ds_compare/trends/prism_anoms_tmin.npy', anomAllTmin)
    np.save('/projects/daymet2/ds_compare/trends/prism_anoms_tmax.npy', anomAllTmax)

def runInterp(x):

    global ptInterp
    global tagg
    global stns
    global uYrs
    global baseMask
    
    if ptInterp is None:
        ptInterp = it.buildDefaultPtInterp()
        tagg = TairAggregate(ptInterp.stn_da_tmax.days)
        stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
        stns = stndaUS.stns
        uYrs = np.unique(ptInterp.stn_da_tmax.days[YEAR])
        baseMask = np.logical_and(uYrs >= 1981,uYrs <= 2010)
        print "Interpolation initialized"

    error = False
    try:
        tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterp.interpLonLatPt(stns[LON][x], stns[LAT][x], fixInvalid=False, chgLatLon=True)
        
        annTmin = tagg.dailyToAnn(tmin_dly).data
        annTmax = tagg.dailyToAnn(tmax_dly).data
        
        anomTmin = annTmin - np.mean(annTmin[baseMask])
        anomTmax = annTmax - np.mean(annTmax[baseMask])

    except Exception:
        print "ERROR: Station not in interpolation domain. %s|%s|%0.4f|%0.4f"%(stns[STN_ID][x],stns[STN_NAME][x],stns[LON][x],stns[LAT][x])
        error = True
    
    if not error:
        if np.sum(np.logical_or(np.abs(tmin_norms) > 100,np.abs(tmax_norms) > 100)) > 0:
            error = True
            print "ERROR: Bad interpolated normals. %s|%s|%0.4f|%0.4f"%(stns[STN_ID][x],stns[STN_NAME][x],stns[LON][x],stns[LAT][x])
        
    if error:
        anomTmin = np.ones(uYrs.size)*np.nan
        anomTmax = np.ones(uYrs.size)*np.nan
        
    return x,anomTmin,anomTmax

def runInterps():
    
    global ptInterp
    global tagg
    global stns
    ptInterp = None
    tagg = None
    stns = None
    pool = Pool(5)#, initializer=initPtInterp)
    
    aPtInterp = it.buildDefaultPtInterp()
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    astns = stndaUS.stns
    uYrs = np.unique(aPtInterp.stn_da_tmax.days[YEAR])
        
    anomsAllTmin = np.zeros((uYrs.size,astns.size))*np.nan
    anomsAllTmax = np.zeros((uYrs.size,astns.size))*np.nan
    
    sck = status_check(astns.size, 50)
    chksize = 50
    
    for x in np.arange(astns.size,step=chksize):
    
        endx = x+chksize
        if endx > astns.size:
            endx = astns.size
        idx = np.arange(x,endx)
        results = pool.map(runInterp,idx,chunksize=1)
        
        for aresult in results:
            j,anomsTmin,anomsTmax = aresult
            anomsAllTmin[:,j] = anomsTmin
            anomsAllTmax[:,j] = anomsTmax
        
        sck.increment(chksize)
    
    pool.close()
    pool.join()
    
    np.save('/projects/daymet2/ds_compare/trends/twx_anoms_tmin.npy', anomsAllTmin)
    np.save('/projects/daymet2/ds_compare/trends/twx_anoms_tmax.npy', anomsAllTmax)

def plotAvgAnoms():
    tairVar = 'tmin'
    
    yrs = np.arange(1948,2013)
    yrsDaymet = np.arange(1980,2013)
    
    anomUshcn = np.load('/projects/daymet2/ds_compare/trends/ushcn_anoms_%s.npy'%tairVar)
    anomTwx = np.load('/projects/daymet2/ds_compare/trends/twx_anoms_%s.npy'%tairVar)
    anomPrism = np.load('/projects/daymet2/ds_compare/trends/prism_anoms_%s.npy'%tairVar)
    anomDaymet = np.load('/projects/daymet2/ds_compare/trends/daymet_anom_%s.npy'%tairVar)

    maskBad = ~np.logical_or(np.logical_or(np.isnan(anomUshcn[0,:]),np.isnan(anomTwx[0,:])),
                            np.logical_or(np.isnan(anomDaymet[0,:]),np.isnan(anomPrism[0,:])))
    print "Removing %d stations."%np.sum(maskBad)
    
    anomUshcn = anomUshcn[:,maskBad]
    anomTwx = anomTwx[:,maskBad]
    anomPrism = anomPrism[:,maskBad]
    anomDaymet = anomDaymet[:,maskBad]

    avgAnomUschn = runningMean(np.mean(anomUshcn,axis=1))
    avgAnomTwx = runningMean(np.mean(anomTwx,axis=1))
    avgAnomDaymet = runningMean(np.mean(anomDaymet,axis=1))
    avgAnomPrism = runningMean(np.mean(anomPrism,axis=1))
    
#    avgAnomUschn = np.mean(anomUshcn,axis=1)
#    avgAnomTwx = np.mean(anomTwx,axis=1)
#    avgAnomDaymet = np.mean(anomDaymet,axis=1)
#    avgAnomPrism = np.mean(anomPrism,axis=1)
      
    avgAnomDaymetFnl = np.zeros(yrs.size)*np.nan
    avgAnomDaymetFnl[yrs>=yrsDaymet[0]] = avgAnomDaymet
    
    #plt.plot(yrs,avgAnomUschn)
    plt.plot(yrs,avgAnomTwx-avgAnomUschn)
    plt.plot(yrs,avgAnomPrism-avgAnomUschn)
    plt.plot(yrs,avgAnomDaymetFnl-avgAnomUschn)
    plt.legend(['TWX','PRISM','Daymet'],loc=2)
    plt.show()

def runningMean(a,n=10):
    sideWin = (n-1)/2
    
    rm = np.ma.masked_array(np.zeros(a.size))
    for x in np.arange(a.size):
        rm[x]=np.ma.mean(a[x-sideWin:x+sideWin+1])
    rm[0:sideWin] = np.ma.masked
    rm[-sideWin:] = np.ma.masked
    
    return rm

def saveStnMask(tairVar):
        
    anomUshcn = np.load('/projects/daymet2/ds_compare/trends/ushcn_anoms_%s.npy'%tairVar)
    anomTwx = np.load('/projects/daymet2/ds_compare/trends/twx_anoms_%s.npy'%tairVar)
    anomPrism = np.load('/projects/daymet2/ds_compare/trends/prism_anoms_%s.npy'%tairVar)
    anomDaymet = np.load('/projects/daymet2/ds_compare/trends/daymet_anom_%s.npy'%tairVar)

    maskBad = ~np.logical_or(np.logical_or(np.isnan(anomUshcn[0,:]),np.isnan(anomTwx[0,:])),
                            np.logical_or(np.isnan(anomDaymet[0,:]),np.isnan(anomPrism[0,:])))
    
    np.save('/projects/daymet2/ds_compare/trends/ushcn_stnmask_%s.npy'%tairVar, maskBad)

def calcConusTrends(anomPath,tairVar,name,anomStartYr=1948,anomEndYr=2012,startYr=1948,endYr=2012):
    
    uYrs = np.arange(startYr,endYr+1)
    anomYrs = np.arange(anomStartYr,anomEndYr+1)
    maskYrs = np.logical_and(anomYrs>=startYr,anomYrs<=endYr)
    
    annAnom = np.load(anomPath)
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    
    stnMask = np.load('/projects/daymet2/ds_compare/trends/ushcn_stnmask_%s.npy'%tairVar)
    annAnom = annAnom[:,stnMask]
    annAnom = annAnom[maskYrs,:]
    stns = stndaUS.stns[stnMask]
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    gridMask = dsGrid.gdalDs.ReadAsArray() != 19    
    yGrid,xGrid = dsGrid.getCoordGrid1d()
    yGrid = np.sort(yGrid)
    
    allAnomGrid = np.zeros((uYrs.size,yGrid.size,xGrid.size))
    allAnomGrid = np.ma.masked_array(allAnomGrid,np.isnan(allAnomGrid))
    
    for i in np.arange(uYrs.size):
    
        print uYrs[i]
        anomGrid = griddata(stns[LON],stns[LAT],annAnom[i,:],xGrid,yGrid)
        anomGrid = np.flipud(anomGrid)
        anomGrid.mask = np.logical_or(gridMask,anomGrid.mask)
        allAnomGrid[i,:,:] = anomGrid
    
    #np.save('/projects/daymet2/docs/final_writeup/conus_trends/annAnomHomogUshcnTmin.npy', np.ma.filled(allAnomGridHomog, -9999))
    #np.save('/projects/daymet2/docs/final_writeup/conus_trends/annAnomRawUshcnTmin.npy', np.ma.filled(allAnomGridRaw, -9999))
    
    trendsGrid = np.zeros((allAnomGrid.shape[1],allAnomGrid.shape[2]))
    
    schk = status_check(trendsGrid.size,1000)
    for r in np.arange(allAnomGrid.shape[1]):
    
        for c in np.arange(allAnomGrid.shape[2]):
            
            if np.ma.is_masked(allAnomGrid[0,r,c]):
                trendsGrid[r,c] = -9999
            else:
                trendsGrid[r,c] = stats.linregress(uYrs,allAnomGrid[:,r,c])[0]
            
            schk.increment()
            
    np.save('/projects/daymet2/ds_compare/trends/grid_anoms_%s_%s.npy'%(name,tairVar), np.ma.filled(allAnomGrid, -9999))
    np.save('/projects/daymet2/ds_compare/trends/grid_trends_%s_%s.npy'%(name,tairVar), trendsGrid)

def plotConusTrends():
    m = Basemap(resolution='i',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5,area_thresh= 10000)
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    lat,lon = dsGrid.getCoordGrid1d()
    latG,lonG = dsGrid.getCoordMeshGrid()
    x, y = m(*np.meshgrid(lon,lat))

    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 9, reverse=True)
    clrs = clrs.mpl_colors
    clrs[4] = "grey"
    #levels = np.linspace(-2.0,2.0,17)#[-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,0.5,0]
    #levels = np.linspace(-2.5,2.5,11)#[-3.0,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,3.0]
    levels = np.linspace(-1.5,1.5,13)
    cmap = cm.RdBu_r
    cmap.set_under('blue')
    cmap.set_over('red')
    
    tminUshcn = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_ushcn_tmin.npy'),-9999)*65
    tmaxUshcn = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_ushcn_tmax.npy'),-9999)*65
    
    tminTwx = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_twx_tmin.npy'),-9999)*65
    tmaxTwx = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_twx_tmax.npy'),-9999)*65
    
    tminPrism = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_prism_tmin.npy'),-9999)*65
    tmaxPrism = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_prism_tmax.npy'),-9999)*65
    
    def getTrendDifs(ushcn,interp):
        
        difs = np.abs(interp.data-ushcn.data)
        
        difClass = np.zeros_like(ushcn.data, dtype=np.int16)
        
        maskBothWarm = np.logical_and(interp.data > 0,ushcn.data > 0)
        
        difClass[np.logical_and(maskBothWarm,ushcn.data > interp.data)] = 0
        difClass[np.logical_and(maskBothWarm,interp.data > ushcn.data)] = 1
        difClass[np.logical_and(interp.data < 0,ushcn.data > 0)] = 2
        difClass[np.logical_and(interp.data > 0,ushcn.data < 0)] = 3
        difClass[np.logical_and(interp.data < 0,ushcn.data < 0)] = 4
        #difClass[difs < 0.05] = 
        
        difClass = np.ma.masked_array(difClass,interp.mask)
        
        maskDifs = np.nonzero(difs > 0.25)
        
        return difClass,latG[maskDifs],lonG[maskDifs]
        
#    tminTwx,tminTwxLat,tminTwxLon = getTrendDifs(tminUshcn, tminTwx)
#    tmaxTwx = getTrendDifs(tmaxUshcn, tmaxTwx)[0]
#    tminPrism,tminPrismLat,tminPrismLon  = getTrendDifs(tminUshcn, tminPrism)
#    tmaxPrism = getTrendDifs(tmaxUshcn, tmaxPrism)[0]
    
    tminTwx = tminTwx -  tminUshcn
    tmaxTwx = tmaxTwx -  tmaxUshcn
    tminPrism = tminPrism -  tminUshcn
    tmaxPrism = tmaxPrism -  tmaxUshcn
    
    print np.min(tminPrism),np.max(tminPrism)
    
#    plt.imshow(tminTwx)
#    plt.colorbar()
#    plt.show()
    
#    levels = [0,1,2,3,4,5]
#    clrs = ['#FF7F00','#E41A1C','#D9D9D9','#737373','#377EB8']#,'white']
           
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05,cbar_size="2%")#,cbar_pad=0.02)#axes_pad=.625
    
    m.ax = grid[0]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tminTwx,cmap=cmap,levels=levels,extend='both')#$,antialiased=False)
    #cf = m.contourf(x, y, tminTwx,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0],ticks=levels)#,label='$^\circ$C / decade')
    grid[0].set_ylabel("TopoWx")
    grid[0].set_title("Tmin")
    #cbar = grid.cbar_axes.colorbar(cf)
    cbar.set_ticks(levels)
    cbar.set_label(r'$^\circ$C decade$^{-1}$')
    #m.scatter(tminTwxLon,tminTwxLat,latlon=True,s=2)
    
    
    m.ax = grid[1]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tmaxTwx,cmap=cmap,levels=levels,extend='both')#,antialiased=False)
    grid[1].set_title("Tmax")
    
    m.ax = grid[2]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tminPrism,cmap=cmap,levels=levels,extend='both')
    grid[2].set_ylabel("PRISM")
    #m.scatter(tminPrismLon,tminPrismLat,latlon=True,s=2)
    
    m.ax = grid[3]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tmaxPrism,cmap=cmap,levels=levels,extend='both')#,antialiased=False)
    
    
    
    ##############################################################################
    
#    m.ax = grid[1]
#    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
#    cf = m.contourf(x, y, tmaxUshcn,colors=clrs,levels=levels,extend='both')
#    grid[1].set_title("Tmax")
#    
#    m.ax = grid[2]
#    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
#    cf = m.contourf(x, y, tminTwx,colors=clrs,levels=levels,extend='both')
#    grid[2].set_ylabel("TopoWx")
#    
#    m.ax = grid[3]
#    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
#    cf = m.contourf(x, y, tmaxTwx,colors=clrs,levels=levels,extend='both')
#    
#    m.ax = grid[4]
#    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
#    cf = m.contourf(x, y, tminPrism,colors=clrs,levels=levels,extend='both')
#    grid[4].set_ylabel("PRISM")
#    
#    m.ax = grid[5]
#    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
#    cf = m.contourf(x, y, tmaxPrism,colors=clrs,levels=levels,extend='both')
    
    #cf = plt.gcf()
    #cf.set_size_inches(8*1.5,6*1.5)
    
    #plt.savefig('/projects/daymet2/docs/final_writeup/conus_trends/trendMaps.png',dpi=150)
    plt.show()

def calcAnnGridAnoms(latGrid,anomGrid):
    
    rad = 4.0*np.arctan(1.0)/180.0
    
    lat1d = np.ravel(latGrid)
    anomYr = anomGrid[0,:,:]
    anomYr = np.ma.ravel(anomYr)
    mask = ~anomYr.mask
    lat1d = lat1d[mask]
    
    wgts = np.cos(lat1d*rad)
    
    anoms = np.zeros(anomGrid.shape[0])
    
    for x in np.arange(anoms.size):
        
        anomYr = anomGrid[x,:,:]
        anomYr = np.ma.ravel(anomYr)
        anomYr = anomYr.compressed()
        
        anoms[x] = np.average(anomYr,weights=wgts)

    return anoms

def conusAnoms():
    
    yrs = np.arange(1948,2013)
    tminUshcn = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_anoms_ushcn_tmin.npy'),-9999)
    tmaxUshcn = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_anoms_ushcn_tmax.npy'),-9999)
    
    tminTwx = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_anoms_twx_tmin.npy'),-9999)
    tmaxTwx = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_anoms_twx_tmax.npy'),-9999)
    
    tminPrism = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_anoms_prism_tmin.npy'),-9999)
    tmaxPrism = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_anoms_prism_tmax.npy'),-9999)
    
    tminDaymet = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_anoms_daymet1980_2012_tmin.npy'),-9999)
    tmaxDaymet = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_anoms_daymet1980_2012_tmax.npy'),-9999)
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    lat,lon = dsGrid.getCoordGrid1d()
    latG,lonG = dsGrid.getCoordMeshGrid()
    
    lonMask = lon <= -104#lon >= -9999#lon <= -104
    
    latG = latG[:,lonMask]
    
    tminUshcn = calcAnnGridAnoms(latG, tminUshcn[:,:,lonMask])
    tmaxUshcn = calcAnnGridAnoms(latG, tmaxUshcn[:,:,lonMask])
    tminTwx = calcAnnGridAnoms(latG, tminTwx[:,:,lonMask])
    tmaxTwx = calcAnnGridAnoms(latG, tmaxTwx[:,:,lonMask])
    tminPrism = calcAnnGridAnoms(latG, tminPrism[:,:,lonMask])
    tmaxPrism = calcAnnGridAnoms(latG, tmaxPrism[:,:,lonMask])
    tminDaymet = calcAnnGridAnoms(latG, tminDaymet[:,:,lonMask])
    tmaxDaymet = calcAnnGridAnoms(latG, tmaxDaymet[:,:,lonMask])
    
    tminDaymetFnl = np.zeros(tminUshcn.size)*np.nan
    tminDaymetFnl[yrs>=1980] = tminDaymet
    tmaxDaymetFnl = np.zeros(tmaxUshcn.size)*np.nan
    tmaxDaymetFnl[yrs>=1980] = tmaxDaymet
    
    print "##########################################"
    print '1948-2012 Trends'
    print "##########################################"
    print "TMIN"
    print "USHCN deg C per decade: %.3f"%(stats.linregress(yrs,tminUshcn)[0]*10,)
    print "TWX deg C per decade: %.3f"%(stats.linregress(yrs,tminTwx)[0]*10,)
    print "PRISM deg C per decade: %.3f"%(stats.linregress(yrs,tminPrism)[0]*10,)
    print "TMAX"
    print "USHCN deg C per decade: %.3f"%(stats.linregress(yrs,tmaxUshcn)[0]*10,)
    print "TWX deg C per decade: %.3f"%(stats.linregress(yrs,tmaxTwx)[0]*10,)
    print "PRISM deg C per decade: %.3f"%(stats.linregress(yrs,tmaxPrism)[0]*10,)
    
    maskNormPeriod = np.logical_and(yrs>=1981,yrs<=2010)
    print "##########################################"
    print '1981-2010 Trends'
    print "##########################################"
    print "TMIN"
    print "USHCN deg C per decade: %.3f"%(stats.linregress(yrs[maskNormPeriod],tminUshcn[maskNormPeriod])[0]*10,)
    print "TWX deg C per decade: %.3f"%(stats.linregress(yrs[maskNormPeriod],tminTwx[maskNormPeriod])[0]*10,)
    print "PRISM deg C per decade: %.3f"%(stats.linregress(yrs[maskNormPeriod],tminPrism[maskNormPeriod])[0]*10,)
    print "Daymet deg C per decade: %.3f"%(stats.linregress(yrs[maskNormPeriod],tminDaymetFnl[maskNormPeriod])[0]*10,)
    print "TMAX"
    print "USHCN deg C per decade: %.3f"%(stats.linregress(yrs[maskNormPeriod],tmaxUshcn[maskNormPeriod])[0]*10,)
    print "TWX deg C per decade: %.3f"%(stats.linregress(yrs[maskNormPeriod],tmaxTwx[maskNormPeriod])[0]*10,)
    print "PRISM deg C per decade: %.3f"%(stats.linregress(yrs[maskNormPeriod],tmaxPrism[maskNormPeriod])[0]*10,)
    print "Daymet deg C per decade: %.3f"%(stats.linregress(yrs[maskNormPeriod],tmaxDaymetFnl[maskNormPeriod])[0]*10,)
    
    plt.subplot(121)
    plt.plot(yrs,tminTwx-tminUshcn)
    plt.plot(yrs,tminPrism-tminUshcn)
    plt.plot(yrs,tminDaymetFnl-tminUshcn)
    plt.subplot(122)
    plt.plot(yrs,tmaxTwx-tmaxUshcn)
    plt.plot(yrs,tmaxPrism-tmaxUshcn)
    plt.plot(yrs,tmaxDaymetFnl-tmaxUshcn)
    plt.show()
    
#    plt.subplot(121)
#    plt.plot(yrs,runningMean(tminUshcn,n=10))
#    plt.plot(yrs,runningMean(tminPrism,n=10))
#    plt.plot(yrs,runningMean(tminTwx,n=10))
#    plt.subplot(122)
#    plt.plot(yrs,runningMean(tmaxUshcn,n=10))
#    plt.plot(yrs,runningMean(tmaxPrism,n=10))
#    plt.plot(yrs,runningMean(tmaxTwx,n=10))
#    plt.show()

def plotConusTrendsOrig():
    m = Basemap(resolution='i',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5,area_thresh= 10000)
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    lat,lon = dsGrid.getCoordGrid1d()
    latG,lonG = dsGrid.getCoordMeshGrid()
    x, y = m(*np.meshgrid(lon,lat))
    
    clrsRed = brewer2mpl.get_map('Reds', 'Sequential', 6, reverse=False).mpl_colors
    clrsBlue = brewer2mpl.get_map('Blues', 'Sequential', 6, reverse=True).mpl_colors
    
    clrsBlue.append("grey")
    clrsBlue.extend(clrsRed)
    clrs = clrsBlue
    
    #clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
    #clrs = clrs.mpl_colors
    #clrs[5] = "grey"
    levels = [-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]
    
    tminUshcn = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_ushcn1981_2010_tmin.npy'),-9999)*10
    tmaxUshcn = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_ushcn1981_2010_tmax.npy'),-9999)*10
    
    tminTwx = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_twx1981_2010_tmin.npy'),-9999)*10
    tmaxTwx = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_twx1981_2010_tmax.npy'),-9999)*10
    
    tminPrism = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_prism1981_2010_tmin.npy'),-9999)*10
    tmaxPrism = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_prism1981_2010_tmax.npy'),-9999)*10
    
    tminDaymet = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_daymet1981_2010_tmin.npy'),-9999)*10
    tmaxDaymet = np.ma.masked_equal(np.load('/projects/daymet2/ds_compare/trends/grid_trends_daymet1981_2010_tmax.npy'),-9999)*10
    
    all = np.ma.vstack((tminUshcn,tmaxUshcn,tminTwx,tmaxTwx,tminPrism,tmaxPrism,tminDaymet,tmaxDaymet))
    print np.min(all),np.max(all)
    
    def getTrendDifs(ushcn,interp):
        
        difs = np.abs(interp.data-ushcn.data)
        
        difClass = np.zeros_like(ushcn.data, dtype=np.int16)
        
        maskBothWarm = np.logical_and(interp.data > 0,ushcn.data > 0)
        
        difClass[np.logical_and(maskBothWarm,ushcn.data > interp.data)] = 0
        difClass[np.logical_and(maskBothWarm,interp.data > ushcn.data)] = 1
        difClass[np.logical_and(interp.data < 0,ushcn.data > 0)] = 2
        difClass[np.logical_and(interp.data > 0,ushcn.data < 0)] = 3
        difClass[np.logical_and(interp.data < 0,ushcn.data < 0)] = 4
        #difClass[difs < 0.05] = 
        
        difClass = np.ma.masked_array(difClass,interp.mask)
        
        maskDifs = np.nonzero(difs > 0.25)
        
        return difClass,latG[maskDifs],lonG[maskDifs]
        
#    tminTwx,tminTwxLat,tminTwxLon = getTrendDifs(tminUshcn, tminTwx)
#    tmaxTwx = getTrendDifs(tmaxUshcn, tmaxTwx)[0]
#    tminPrism,tminPrismLat,tminPrismLon  = getTrendDifs(tminUshcn, tminPrism)
#    tmaxPrism = getTrendDifs(tmaxUshcn, tmaxPrism)[0]
    
    #print np.min(tminPrism),np.max(tminPrism)
    
#    plt.imshow(tminTwx)
#    plt.colorbar()
#    plt.show()
    
#    levels = [0,1,2,3,4,5]
#    clrs = ['#FF7F00','#E41A1C','#D9D9D9','#737373','#377EB8']#,'white']
           
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(4,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05,cbar_size="2%")#,cbar_pad=0.02)#axes_pad=.625
    
    m.ax = grid[0]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tminUshcn,colors=clrs,levels=levels,extend='both')#$,antialiased=False)
    #cf = m.contourf(x, y, tminTwx,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0],ticks=levels)#,label='$^\circ$C / decade')
    grid[0].set_ylabel("USHCN")
    grid[0].set_title("Tmin")
    #cbar = grid.cbar_axes.colorbar(cf)
    cbar.set_ticks(levels)
    cbar.set_label(r'$^\circ$C decade$^{-1}$')
    #m.scatter(tminTwxLon,tminTwxLat,latlon=True,s=2)
        
    ##############################################################################
    
    m.ax = grid[1]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tmaxUshcn,colors=clrs,levels=levels,extend='both')
    grid[1].set_title("Tmax")
    
    m.ax = grid[2]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tminTwx,colors=clrs,levels=levels,extend='both')
    grid[2].set_ylabel("TopoWx")
    
    m.ax = grid[3]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tmaxTwx,colors=clrs,levels=levels,extend='both')
    
    m.ax = grid[4]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tminPrism,colors=clrs,levels=levels,extend='both')
    grid[4].set_ylabel("PRISM")
    
    m.ax = grid[5]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tmaxPrism,colors=clrs,levels=levels,extend='both')
    
    m.ax = grid[6]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tminDaymet,colors=clrs,levels=levels,extend='both')
    grid[6].set_ylabel("Daymet")
    
    m.ax = grid[7]
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tmaxDaymet,colors=clrs,levels=levels,extend='both')
    
    #cf = plt.gcf()
    #cf.set_size_inches(8*1.5,6*1.5)
    
    #plt.savefig('/projects/daymet2/docs/final_writeup/conus_trends/trendMaps.png',dpi=150)
    plt.show()

if __name__ == '__main__':
    
    #Get the anomalies at each stn location
    #buildUshcnAnoms()
    #buildPrismAnoms()
    #buildDaymetAnoms('tmin')
    #runInterps()
    
    #Build masks of USHCN stations that should not be included
    #saveStnMask('tmin')
    #saveStnMask('tmax')
    
    #Build gridded anomalies and trends
#    calcConusTrends('/projects/daymet2/ds_compare/trends/ushcn_anoms_tmin.npy', 'tmin', 'ushcn', 1948, 2012, 1948, 2012)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/ushcn_anoms_tmax.npy', 'tmax', 'ushcn', 1948, 2012, 1948, 2012)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/ushcn_anoms_tmin.npy', 'tmin', 'ushcn1981_2010', 1948, 2012, 1981, 2010)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/ushcn_anoms_tmax.npy', 'tmax', 'ushcn1981_2010', 1948, 2012, 1981, 2010)

#    calcConusTrends('/projects/daymet2/ds_compare/trends/twx_anoms_tmin.npy', 'tmin', 'twx', 1948, 2012,1948, 2012)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/twx_anoms_tmax.npy', 'tmax', 'twx', 1948, 2012,1948, 2012)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/twx_anoms_tmin.npy', 'tmin', 'twx1981_2010', 1948, 2012,1981, 2010)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/twx_anoms_tmax.npy', 'tmax', 'twx1981_2010', 1948, 2012,1981, 2010)
    
#    calcConusTrends('/projects/daymet2/ds_compare/trends/prism_anoms_tmin.npy','tmin', 'prism', 1948, 2012,1948, 2012)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/prism_anoms_tmax.npy','tmax', 'prism', 1948, 2012,1948, 2012)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/prism_anoms_tmin.npy','tmin', 'prism1981_2010', 1948, 2012,1981, 2010)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/prism_anoms_tmax.npy','tmax', 'prism1981_2010', 1948, 2012,1981, 2010)
    
#    calcConusTrends('/projects/daymet2/ds_compare/trends/daymet_anom_tmin.npy','tmin', 'daymet1980_2012', 1980, 2012,1980, 2012)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/daymet_anom_tmax.npy','tmax', 'daymet1980_2012', 1980, 2012,1980, 2012)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/daymet_anom_tmin.npy','tmin', 'daymet1981_2010', 1980, 2012,1981, 2010)
#    calcConusTrends('/projects/daymet2/ds_compare/trends/daymet_anom_tmax.npy','tmax', 'daymet1981_2010', 1980, 2012,1981, 2010)
    
    #conusAnoms()
    plotConusTrendsOrig()
    #plotAvgAnoms()

