'''
Created on Sep 19, 2013

@author: jared.oyler
'''
import numpy as np
from twx.db.station_data import LON, LAT,YEAR,STN_ID, DATE,MONTH
import matplotlib.pyplot as plt
from twx.utils.input_raster import RasterDataset
from mpl_toolkits.basemap import Basemap
from twx.utils.status_check import status_check
import twx.db.ushcn as ushcn
import twx.utils.util_dates as utld
from datetime import datetime
from matplotlib.mlab import griddata 
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import stats
import os
import brewer2mpl
from PaperResultsMetrics import runningMean
import matplotlib as mpl

MISSING_VAL = -9999

def mask_to_shp(fpath_shp,shp_layer,fpath_in_ds,fpath_out_ds,nodata=-9999):
    
    #cmd = "".join(['gdalwarp -cutline "',fpath_shp,'" -cl ',shp_layer,' -crop_to_cutline -dstnodata -9999 ',fpath_in_ds,' ',fpath_out_ds])
    cmd = "".join(['gdalwarp -cutline "',fpath_shp,'" -cl ',shp_layer,' -dstnodata ',str(nodata),' ',fpath_in_ds,' ',fpath_out_ds])
    print cmd
    os.system(cmd)

def calcConusTrendsSeason(mths,seasonName,tairVar):
    
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn1895_2012.nc')

    maskSeason = np.in1d(stndaUS.mths[MONTH], mths, False)
    yrsSeason = stndaUS.mths[YEAR][maskSeason]
    uYrs = np.unique(yrsSeason)
    maskYears = []
    for yr in uYrs:
        maskYears.append(yrsSeason==yr)
    baseMask = np.logical_and(uYrs >= 1961,uYrs <= 1990)
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    gridMask = dsGrid.gdalDs.ReadAsArray() != 19
    
    #stns = stndaUS.stns[np.sum(stndaUS.data['raw_tmin'].mask,axis=0)==0]
    stns = stndaUS.stns
    
    anomFLs = np.zeros((uYrs.size,stns.size))
    anomFLs = np.ma.masked_array(anomFLs,np.isnan(anomFLs))
    
    for astn,x in zip(stns,np.arange(stns.size)):
        
        obsFLs = stndaUS.loadObs(astn[STN_ID], "".join(['FLs.52i_',tairVar]))
        obsFLs = obsFLs[maskSeason]
        obsFLs = np.ma.masked_array([np.ma.mean(obsFLs[aMask],dtype=np.float) for aMask in maskYears])
        obsFLs = obsFLs - np.ma.mean(obsFLs[baseMask])
        anomFLs[:,x] = obsFLs
    
    yGrid,xGrid = dsGrid.getCoordGrid1d()
    yGrid = np.sort(yGrid)
    
    allAnomGridHomog = np.zeros((uYrs.size,yGrid.size,xGrid.size))
    allAnomGridHomog = np.ma.masked_array(allAnomGridHomog,np.isnan(allAnomGridHomog))
    
    allAnomGridRaw = np.zeros((uYrs.size,yGrid.size,xGrid.size))
    allAnomGridRaw = np.ma.masked_array(allAnomGridRaw,np.isnan(allAnomGridRaw))
    
    for i in np.arange(uYrs.size):
    
        print uYrs[i]
        anomGridHomog = griddata(stns[LON],stns[LAT],anomFLs[i,:],xGrid,yGrid)
        anomGridHomog = np.flipud(anomGridHomog)
        anomGridHomog.mask = np.logical_or(gridMask,anomGridHomog.mask)
        allAnomGridHomog[i,:,:] = anomGridHomog
        
    np.save("".join(['/projects/daymet2/docs/ccs_lecture/AnomUshcn',seasonName,"_",tairVar,".npy"]), np.ma.filled(allAnomGridHomog, -9999))
    
    trendsHomog = np.zeros((allAnomGridHomog.shape[1],allAnomGridHomog.shape[2]))
    
    schk = status_check(trendsHomog.size,10000)
    for r in np.arange(allAnomGridHomog.shape[1]):
    
        for c in np.arange(allAnomGridHomog.shape[2]):
            
            if np.ma.is_masked(allAnomGridHomog[0,r,c]):
                trendsHomog[r,c] = -9999
            else:
                trendsHomog[r,c] = stats.linregress(uYrs,allAnomGridHomog[:,r,c])[0]
            
            schk.increment()
    np.save("".join(['/projects/daymet2/docs/ccs_lecture/TrendUshcn',seasonName,"_",tairVar,".npy"]), trendsHomog)

def calcConusTrends():
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn1895_2012.nc')
    strMth = stndaUS.mths[DATE][0]
    endMth = stndaUS.mths[DATE][-1]
    
    uYrs = np.unique(stndaUS.mths[YEAR])
    baseMask = np.logical_and(uYrs >= 1961,uYrs <= 1990)
    
    days = utld.get_days_metadata(datetime(strMth.year,1,1), datetime(endMth.year,12,31))
    tairAgg = ushcn.TairAggregate(days)
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    gridMask = dsGrid.gdalDs.ReadAsArray() != 19
    
    #stns = stndaUS.stns[np.sum(stndaUS.data['raw_tmin'].mask,axis=0)==0]
    stns = stndaUS.stns
    
    anomFLs = np.zeros((uYrs.size,stns.size))
    anomFLs = np.ma.masked_array(anomFLs,np.isnan(anomFLs))
    
    for astn,x in zip(stns,np.arange(stns.size)):
        
        obsFLs = stndaUS.loadObs(astn[STN_ID], 'FLs.52i_tmax')
        obsFLs = tairAgg.mthlyToAnn(obsFLs)
        obsFLs = obsFLs - np.ma.mean(obsFLs[baseMask])
        anomFLs[:,x] = obsFLs
    
    yGrid,xGrid = dsGrid.getCoordGrid1d()
    yGrid = np.sort(yGrid)
    
    allAnomGridHomog = np.zeros((uYrs.size,yGrid.size,xGrid.size))
    allAnomGridHomog = np.ma.masked_array(allAnomGridHomog,np.isnan(allAnomGridHomog))
    
    allAnomGridRaw = np.zeros((uYrs.size,yGrid.size,xGrid.size))
    allAnomGridRaw = np.ma.masked_array(allAnomGridRaw,np.isnan(allAnomGridRaw))
    
    for i in np.arange(uYrs.size):
    
        print uYrs[i]
        anomGridHomog = griddata(stns[LON],stns[LAT],anomFLs[i,:],xGrid,yGrid)
        anomGridHomog = np.flipud(anomGridHomog)
        anomGridHomog.mask = np.logical_or(gridMask,anomGridHomog.mask)
        allAnomGridHomog[i,:,:] = anomGridHomog
        
    np.save('/projects/daymet2/docs/ccs_lecture/annAnomHomogUshcnTmax.npy', np.ma.filled(allAnomGridHomog, -9999))
    
    trendsHomog = np.zeros((allAnomGridHomog.shape[1],allAnomGridHomog.shape[2]))
    
    schk = status_check(trendsHomog.size,10000)
    for r in np.arange(allAnomGridHomog.shape[1]):
    
        for c in np.arange(allAnomGridHomog.shape[2]):
            
            if np.ma.is_masked(allAnomGridHomog[0,r,c]):
                trendsHomog[r,c] = -9999
            else:
                trendsHomog[r,c] = stats.linregress(uYrs,allAnomGridHomog[:,r,c])[0]
            
            schk.increment()
    np.save('/projects/daymet2/docs/ccs_lecture/trendsHomogUshcnTmax.npy', trendsHomog)


def plotConusTrendsMap():
    m = Basemap(resolution='i',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5,area_thresh= 10000)
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    lat,lon = dsGrid.getCoordGrid1d()
    x, y = m(*np.meshgrid(lon,lat))

    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
    clrs = clrs.mpl_colors
    clrs[5] = "grey"
    levels = [-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]

    #Trends 1895-2012
#    tmaxUshcn = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/trendsHomogUshcnTmax.npy'),-9999)*10
#    tminUshcn = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/trendsHomogUshcnTmin.npy'),-9999)*10
    
    #Trends 1948-2012
    tmaxUshcn = np.ma.masked_equal(np.load('/projects/daymet2/docs/final_writeup/conus_trends/trendsHomogUshcnTmax.npy'),-9999)*10
    tminUshcn = np.ma.masked_equal(np.load('/projects/daymet2/docs/final_writeup/conus_trends/trendsHomogUshcnTmin.npy'),-9999)*10      
    
    #Winter trends
    tmaxUshcn = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/TrendUshcnWinter_tmax.npy'),-9999)*10
    tminUshcn = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/TrendUshcnWinter_tmin.npy'),-9999)*10     
    
                     
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05,cbar_size="2%")#,cbar_pad=0.02)#axes_pad=.625
    m.ax = grid[0]
#    m.drawcountries()
#    m.drawstates()
#    m.drawcoastlines()
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')

    cf = m.contourf(x, y, tminUshcn,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0],ticks=levels)#,label='$^\circ$C / decade')
    #grid[0].set_ylabel("Tmin")
    grid[0].set_title("Tmin",fontsize=17)
    #cbar = grid.cbar_axes.colorbar(cf)
    cbar.ax.tick_params(labelsize=17)
    cbar.set_ticks(levels)
    cbar.set_label(r'$^\circ$C decade$^{-1}$',fontsize=17)
    #cbar.ax.xaxis.tick_top()
    
    m.ax = grid[1]
#    m.drawcountries()
#    m.drawstates()
#    m.drawcoastlines()
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    grid[1].set_title("Tmax",fontsize=17)
    cf = m.contourf(x, y, tmaxUshcn,colors=clrs,levels=levels,extend='both')
    
#    m.ax = grid[2]
##    m.drawcountries()
##    m.drawstates()
##    m.drawcoastlines()
#    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
#    grid[2].set_ylabel("Tmax")
#    cf = m.contourf(x, y, tmaxUshcn,colors=clrs,levels=levels,extend='both')
#    
#    m.ax = grid[3]
##    m.drawcountries()
##    m.drawstates()
##    m.drawcoastlines()
#    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
#    cf = m.contourf(x, y, tmaxTopoWx,colors=clrs,levels=levels,extend='both')
    
    #cf = plt.gcf()
    #cf.set_size_inches(8*1.5,6*1.5)
    
    #plt.savefig('/projects/daymet2/docs/final_writeup/conus_trends/trendMaps.png',dpi=150)
    fig = plt.gcf()
    fig.set_size_inches(8*2,6)
    #plt.savefig("/projects/daymet2/docs/ccs_lecture/conus_map_tair_trends48-12.png",dpi=150)
    plt.show()
    
    
    
#    plt.subplot(211)
#    m.contourf(x,y,tmaxUshcn*10,levels=levels,colors=clrs,extend='both')
#    cbar = m.colorbar()
#    cbar.set_ticks(levels)
#    m.drawstates()
#    m.drawcountries()
#    m.drawcoastlines()
#    
#    plt.subplot(212)
#    m.contourf(x,y,tmaxTopoWx*10,levels=levels,colors=clrs,extend='both')
#    cbar = m.colorbar()
#    cbar.set_ticks(levels)
#    m.drawstates()
#    m.drawcountries()
#    m.drawcoastlines()
#    
#
#    
#    
#    
#    #m.readshapefile('/projects/daymet2/dem/climate_divisions/ClimDivWGS84','ClimDivWGS84')
#
#    
#    plt.show()

def anomBarPlot(anom):
    blue = '#2166AC'
    red = '#B2182B'
    
    colors = np.array([red]*anom.size)
    colors[anom < 0] = blue
    
    plt.bar(np.arange(anom.size), anom,color=colors,alpha=.5)
    xlim = plt.xlim((0,118))
    plt.hlines(0, xlim[0], xlim[1], colors='k')
    plt.xlim((0,118))
    yrs = np.arange(1895,2013)
    loc = np.arange(5,118,10)
    plt.xticks(loc,[str(yr) for yr in yrs[loc]],fontsize=17)
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.plot(runningMean(anom),color='k',lw=2)
    #plt.ylim((-2.5,2.5))
    plt.xlabel("Year",fontsize=17)
    plt.ylabel("Anomaly ($^\circ$C)",fontsize=17)
    plt.setp(ax.get_yticklabels(), fontsize=17)

def plotConusTrends():

    rad = 4.0*np.arctan(1.0)/180.0
    re = 6371220.0
    rr  = re*rad
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    latGrid,lonGrid = dsGrid.getCoordMeshGrid()
    latGrid = np.ravel(latGrid,order='C')
    
    dsGridMt = RasterDataset('/projects/daymet2/docs/ccs_lecture/maskQtrDegMT.tif')
    mtMask = dsGridMt.readAsArray()
    mtMask = ~mtMask.mask
    mtMask = np.ravel(mtMask,order='C')
    
    #tmax = np.load('/projects/daymet2/docs/ccs_lecture/annAnomHomogUshcnTmax.npy')
    tmax = np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnWinter_tmax.npy')
    tmax = np.ma.masked_equal(tmax,-9999)
    tmax = np.reshape(tmax,(tmax.shape[0],latGrid.size))
    
    #tmin = np.load('/projects/daymet2/docs/ccs_lecture/annAnomHomogUshcnTmin.npy')
    tmin = np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnWinter_tmin.npy')
    tmin = np.ma.masked_equal(tmin,-9999)
    tmin = np.reshape(tmin,(tmin.shape[0],latGrid.size),order='C')
    
    wgts = np.cos(latGrid*rad)
    wgtsMT = np.cos(latGrid*rad)[mtMask]
    
    #tmaxAnom = np.ma.average(tmax[:,mtMask], axis=1, weights=wgts)
    #tminAnom = np.ma.average(tmin[:,mtMask], axis=1, weights=wgts)
    tmaxAnom = np.ma.average(tmax, axis=1, weights=wgts)
    tminAnom = np.ma.average(tmin, axis=1, weights=wgts)
    
    tmaxAnomMT = np.ma.average(tmax[:,mtMask], axis=1, weights=wgtsMT)
    tminAnomMT = np.ma.average(tmin[:,mtMask], axis=1, weights=wgtsMT)
    #tmaxAnom = np.ma.average(tmax[:,mtMask], axis=1, weights=wgts)
#    tminAnom = np.ma.average(tmin[:,mtMask], axis=1, weights=wgts)
    yrs = np.arange(1895,2013)
    print "CONUS Trends"
    print "TMAX",stats.linregress(yrs,tmaxAnom)[0]*yrs.size
    print "TMIN",stats.linregress(yrs,tminAnom)[0]*yrs.size
    print "Montana Trends"
    print "TMAX",stats.linregress(yrs,tmaxAnomMT)[0]*yrs.size
    print "TMIN",stats.linregress(yrs,tminAnomMT)[0]*yrs.size
    
    plt.subplot(211)
    anomBarPlot(tminAnomMT)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.xlabel("")
    plt.subplot(212)
    anomBarPlot(tmaxAnomMT)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8*1.5,6*1.5)
    plt.savefig("/projects/daymet2/docs/ccs_lecture/montana_ann_tair_trends.png",dpi=150)
    plt.show()
    

def barPlotMtSeasonalTrends():

    rad = 4.0*np.arctan(1.0)/180.0
    re = 6371220.0
    rr  = re*rad
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    latGrid,lonGrid = dsGrid.getCoordMeshGrid()
    latGrid = np.ravel(latGrid,order='C')
    
    dsGridMt = RasterDataset('/projects/daymet2/docs/ccs_lecture/maskQtrDegMT.tif')
    mtMask = dsGridMt.readAsArray()
    mtMask = ~mtMask.mask
    mtMask = np.ravel(mtMask,order='C')
    
    tmaxWinter = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnWinter_tmax.npy'),-9999)
    tmaxWinter = np.reshape(tmaxWinter,(tmaxWinter.shape[0],latGrid.size))[:,mtMask]
    tmaxSpring = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnSpring_tmax.npy'),-9999)
    tmaxSpring = np.reshape(tmaxSpring,(tmaxSpring.shape[0],latGrid.size))[:,mtMask]
    tmaxSummer = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnSummer_tmax.npy'),-9999)
    tmaxSummer = np.reshape(tmaxSummer,(tmaxSummer.shape[0],latGrid.size))[:,mtMask]
    tmaxFall = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnFall_tmax.npy'),-9999)
    tmaxFall = np.reshape(tmaxFall,(tmaxFall.shape[0],latGrid.size))[:,mtMask]
    tmax = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/annAnomHomogUshcnTmax.npy'),-9999)
    tmax = np.reshape(tmax,(tmax.shape[0],latGrid.size))[:,mtMask]
    
    tminWinter = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnWinter_tmin.npy'),-9999)
    tminWinter = np.reshape(tminWinter,(tminWinter.shape[0],latGrid.size))[:,mtMask]
    tminSpring = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnSpring_tmin.npy'),-9999)
    tminSpring = np.reshape(tminSpring,(tminSpring.shape[0],latGrid.size))[:,mtMask]
    tminSummer = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnSummer_tmin.npy'),-9999)
    tminSummer = np.reshape(tminSummer,(tminSummer.shape[0],latGrid.size))[:,mtMask]
    tminFall = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/AnomUshcnFall_tmin.npy'),-9999)
    tminFall = np.reshape(tminFall,(tminFall.shape[0],latGrid.size))[:,mtMask]
    tmin = np.ma.masked_equal(np.load('/projects/daymet2/docs/ccs_lecture/annAnomHomogUshcnTmin.npy'),-9999)
    tmin = np.reshape(tmin,(tmin.shape[0],latGrid.size))[:,mtMask]
    
    wgtsMT = np.cos(latGrid*rad)[mtMask]
    yrs = np.arange(1895,2013)
    
    def getTrend(wgts,anom,yrs):
        meanAnom = np.ma.average(anom, axis=1, weights=wgts)
        return stats.linregress(yrs,meanAnom)[0]*yrs.size
    
    #Winter
    tWinterTmax = getTrend(wgtsMT,tmaxWinter, yrs)
    tWinterTmin = getTrend(wgtsMT,tminWinter, yrs)
    #Spring
    tSpringTmax = getTrend(wgtsMT,tmaxSpring, yrs)
    tSpringTmin = getTrend(wgtsMT,tminSpring, yrs)
    #Summer
    tSummerTmax = getTrend(wgtsMT,tmaxSummer, yrs)
    tSummerTmin = getTrend(wgtsMT,tminSummer, yrs)
    #Fall
    tFallTmax = getTrend(wgtsMT,tmaxFall, yrs)
    tFallTmin = getTrend(wgtsMT,tminFall, yrs)
    #Annual
    tAnnTmax = getTrend(wgtsMT,tmax, yrs)
    tAnnTmin = getTrend(wgtsMT,tmin, yrs)
    
    
    n_groups = 5
    
    index = np.arange(n_groups)
    bar_width = 0.25
    colors = ['#984EA3','#377EB8','#4DAF4A','#E41A1C','#FF7F00']
    
    rects1 = plt.bar(index, [tAnnTmin,tWinterTmin,tSpringTmin,tSummerTmin,tFallTmin], bar_width,color=colors,label='Tmin')
        
    rects2 = plt.bar(index + bar_width, [tAnnTmax,tWinterTmax,tSpringTmax,tSummerTmax,tFallTmax], bar_width,
                     color=colors,
                     label='Tmax',
                     hatch="/")
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
#    
#    plt.xlabel('Climate Division',fontsize=17)
    plt.ylabel('$^\circ$C',fontsize=17)
    plt.setp(plt.gca().get_yticklabels(), fontsize=17)
    plt.xticks(index + bar_width, ('Annual', 'Winter', 'Spring', 'Summer', 'Fall'),fontsize=17)
    p = mpl.patches.Rectangle((0,0),1,1,fc="white")
    p2 = mpl.patches.Rectangle((0,0),1,1,fc="white",hatch="//")
    plt.legend([p,p2], ["Tmin","Tmax"],fontsize=17,loc=1)
    fig = plt.gcf()
    fig.set_size_inches(8*1.5,6*1.5)
    plt.savefig("/projects/daymet2/docs/ccs_lecture/montana_seasonal_tair_trends.png",dpi=150)
    plt.show()


PRCP_BOZEMAN = [0.48,0.43,0.94,1.60,2.45,2.40,1.12,1.08,1.11,1.10,0.76,0.52]
TMIN_BOZEMAN = [11.1,14.9,23.4,31.0,39.0,45.9,51.1,49.2,40.5,30.6,19.5,9.3]
TMAX_BOZEMAN = [35.9,39.5,49.6,59.2,68.3,77.2,87.8,87.2,75.1,61.2,44.9,33.3]

PRCP_MISSOULA = [0.85,0.70,1.00,1.22,2.01,2.07,0.99,1.19,1.17,0.88,1.01,1.04]
TMIN_MISSOULA = [18.3,21.2,27.7,32.8,39.8,46.6,51.4,50.1,41.8,32.4,24.9,16.7]
TMAX_MISSOULA = [33.2,38.8,49.8,58.5,67.3,75.2,85.9,84.9,73.1,57.8,41.5,31.0]

PRCP_WGLAC = [3.23,1.98,2.08,1.93,2.64,3.47,1.70,1.30,2.05,2.49,3.27,3.01]
TMIN_WGLAC = [18.3,18.9,24.6,30.6,38.0,44.3,48.5,47.1,39.3,32.0,25.5,17.8]
TMAX_WGLAC = [30.5,35.0,43.2,54.0,64.5,71.7,80.0,79.3,67.5,52.3,37.3,28.8]

def barPlotNormals():
     
    n_groups = 12
    
    index = np.arange(n_groups)
    bar_width = 0.25
    colors = ['#A6CEE3','#1F78B4','#B2DF8A']
    
    
    plt.bar(index,TMAX_MISSOULA,bar_width,color=colors[0],label="Missoula")
    plt.bar(index+ bar_width,TMAX_BOZEMAN,bar_width,color=colors[1],label="Bozeman")
    plt.bar(index+ bar_width*2,TMAX_WGLAC,bar_width,color=colors[2],label="West Glacier")
    plt.xticks(index + bar_width + bar_width/2.0, ('JAN', 'FEB', 'MAR', 'APR', 'MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'),fontsize=17)
    #plt.ylabel("Inches",fontsize=17)
    plt.ylabel("$^\circ$F",fontsize=17)
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.setp(plt.gca().get_yticklabels(), fontsize=17)

    fig = plt.gcf()
    fig.set_size_inches(8*1.5,6*1.5)
    
    #plt.legend()
    plt.savefig("/projects/daymet2/docs/ccs_lecture/barplot_normals_tmax.png",dpi=150)
    plt.show()
    
#    rects1 = plt.bar(index, [tAnnTmin,tWinterTmin,tSpringTmin,tSummerTmin,tFallTmin], bar_width,color=colors,label='Tmin')
#        
#    rects2 = plt.bar(index + bar_width, [tAnnTmax,tWinterTmax,tSpringTmax,tSummerTmax,tFallTmax], bar_width,
#                     color=colors,
#                     label='Tmax',
#                     hatch="/")
#    ax = plt.gca()
#    ax.set_axisbelow(True)
#    ax.yaxis.grid(color='gray', linestyle='dashed')
##    
##    plt.xlabel('Climate Division',fontsize=17)
#    plt.ylabel('$^\circ$C',fontsize=17)
#    plt.setp(plt.gca().get_yticklabels(), fontsize=17)
#    plt.xticks(index + bar_width, ('Annual', 'Winter', 'Spring', 'Summer', 'Fall'),fontsize=17)
#    p = mpl.patches.Rectangle((0,0),1,1,fc="white")
#    p2 = mpl.patches.Rectangle((0,0),1,1,fc="white",hatch="//")
#    plt.legend([p,p2], ["Tmin","Tmax"],fontsize=17,loc=1)
#    fig = plt.gcf()
#    fig.set_size_inches(8*1.5,6*1.5)
#    plt.savefig("/projects/daymet2/docs/ccs_lecture/montana_seasonal_tair_trends.png",dpi=150)
#    plt.show()

def plotUshcnStations():
    
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn1895_2012.nc')
    
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
            lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    m.drawcountries()
    m.drawcoastlines()
    m.drawstates()
    m.scatter(stndaUS.stns[LON],stndaUS.stns[LAT],latlon=True)
    plt.show()

def calcMontanaAnomsMthly():
    
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn1895_2012.nc')

    strMth = stndaUS.mths[DATE][0]
    endMth = stndaUS.mths[DATE][-1]
    
    mths = np.arange(1,13)
    uYrs = np.unique(stndaUS.mths[YEAR])
    baseMaskAnn = np.logical_and(uYrs >= 1961,uYrs <= 1990)
    baseMaskMths = []
    maskMths = []
    for mth in mths:
        baseMaskMths.append(np.logical_and(np.logical_and(stndaUS.mths[YEAR] >= 1961,
                                                          stndaUS.mths[YEAR] <= 1990),
                                           stndaUS.mths[MONTH]==mth))
        maskMths.append(stndaUS.mths[MONTH]==mth)
    
    days = utld.get_days_metadata(datetime(strMth.year,1,1), datetime(endMth.year,12,31))
    tairAgg = ushcn.TairAggregate(days)
    stns = stndaUS.stns
    
    anomAnnAllTmin = np.ma.masked_array(np.zeros((uYrs.size,stns.size)))
    anomAnnAllTmax = np.ma.masked_array(np.zeros((uYrs.size,stns.size)))
    
    anomMthlyAllTmin = np.ma.masked_array(np.zeros((12,uYrs.size,stns.size)))
    anomMthlyAllTmax = np.ma.masked_array(np.zeros((12,uYrs.size,stns.size)))
        
    schk = status_check(stns.size, 50)
    
    blankAnoms = np.ma.masked_equal(np.zeros(uYrs.size)*MISSING_VAL,MISSING_VAL)
    
    for astn,x in zip(stns,np.arange(stns.size)):
        
        if x == 21:
            print astn[STN_ID]
        
        obsTmin = stndaUS.loadObs(astn[STN_ID], 'FLs.52i_tmin')
        obsTmax = stndaUS.loadObs(astn[STN_ID], 'FLs.52i_tmax')
                
        #calc annual anomalies
        annTmin = tairAgg.mthlyToAnn(obsTmin,0)
        annTmax = tairAgg.mthlyToAnn(obsTmax,0)
        
        blanksSet = False
        
        if np.sum(annTmin[baseMaskAnn].mask) > 9:
            anomTmin = blankAnoms
            blanksSet = True
        else:
            anomTmin = annTmin - np.ma.mean(annTmin[baseMaskAnn],dtype=np.float)
            
        if np.sum(annTmax[baseMaskAnn].mask) > 9:
            anomTmin = blankAnoms
            blanksSet = True
        else:
            anomTmax = annTmax - np.ma.mean(annTmax[baseMaskAnn],dtype=np.float)
        
        anomAnnAllTmin[:,x] = anomTmin
        anomAnnAllTmax[:,x] = anomTmax
        
        for mth in mths:
            mthTmin = obsTmin[maskMths[mth-1]]
            mthTmax = obsTmax[maskMths[mth-1]]
            
            if np.sum(obsTmin[baseMaskMths[mth-1]].mask) > 9:
                mthAnomTmin = blankAnoms
                blanksSet = True
            else:
                mthAnomTmin = mthTmin - np.mean(obsTmin[baseMaskMths[mth-1]],dtype=np.float)
            
            if np.sum(obsTmax[baseMaskMths[mth-1]].mask) > 9:
                mthAnomTmax = blankAnoms
                blanksSet = True
            else:
                mthAnomTmax = mthTmax - np.mean(obsTmax[baseMaskMths[mth-1]],dtype=np.float)
            
            anomMthlyAllTmin[mth-1,:,x] = mthAnomTmin
            anomMthlyAllTmax[mth-1,:,x] = mthAnomTmax
        
        if blanksSet:
            print "%s: removed from a record due to not enough base period obs"%astn[STN_ID]
        
        schk.increment()

    np.save('/projects/daymet2/montana_trends/anom_ann_tmin.npy',np.ma.filled(anomAnnAllTmin,MISSING_VAL)) 
    np.save('/projects/daymet2/montana_trends/anom_ann_tmax.npy',np.ma.filled(anomAnnAllTmax,MISSING_VAL)) 
    np.save('/projects/daymet2/montana_trends/anom_mthly_tmin.npy',np.ma.filled(anomMthlyAllTmin,MISSING_VAL)) 
    np.save('/projects/daymet2/montana_trends/anom_mthly_tmax.npy',np.ma.filled(anomMthlyAllTmax,MISSING_VAL)) 


def calcGriddedMontanaTrends(anomPath,tairVar,name,mth=None,startYr=1895,endYr=2012):
    
    uYrs = np.arange(startYr,endYr+1)
    
    anom = np.ma.masked_equal(np.load(anomPath),MISSING_VAL)
    if mth is not None:
        anom = anom[mth-1,:,:]
    
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    stns = stndaUS.stns
    
    dsGrid = RasterDataset('/projects/daymet2/docs/ccs_lecture/maskQtrDegMT.tif')
    gridMask = dsGrid.gdalDs.ReadAsArray() != 19    
    yGrid,xGrid = dsGrid.getCoordGrid1d()
    yGrid = np.sort(yGrid)
    
    allAnomGrid = np.zeros((uYrs.size,yGrid.size,xGrid.size))
    allAnomGrid = np.ma.masked_array(allAnomGrid,np.isnan(allAnomGrid))
    
    for i in np.arange(uYrs.size):
            
        anomYr = anom[i,:]
        maskStns = ~anomYr.mask
        stnsLon = stns[LON][maskStns]
        stnsLat = stns[LAT][maskStns]
        anomYr = np.ma.compressed(anomYr)
        
        anomGrid = griddata(stnsLon,stnsLat,anomYr,xGrid,yGrid)
        anomGrid = np.flipud(anomGrid)
        
        anomGrid.mask = np.logical_or(gridMask,anomGrid.mask)
        
        allAnomGrid[i,:,:] = anomGrid
        
    trendsGrid = np.zeros((allAnomGrid.shape[1],allAnomGrid.shape[2]))
    
    schk = status_check(trendsGrid.size,1000)
    for r in np.arange(allAnomGrid.shape[1]):
    
        for c in np.arange(allAnomGrid.shape[2]):
            
            if np.ma.is_masked(allAnomGrid[0,r,c]):
                trendsGrid[r,c] = -9999
            else:
                trendsGrid[r,c] = stats.linregress(uYrs,allAnomGrid[:,r,c])[0]
            
            schk.increment()
            
    np.save('/projects/daymet2/montana_trends/grid_anoms_%s_%s.npy'%(name,tairVar), np.ma.filled(allAnomGrid, MISSING_VAL))
    np.save('/projects/daymet2/montana_trends/grid_trends_%s_%s.npy'%(name,tairVar), trendsGrid)

def calcAnnGridAnoms(latGrid,anomGrid):
    
    rad = 4.0*np.arctan(1.0)/180.0
    
    lat1d = np.ravel(latGrid)
    
    anoms = np.zeros(anomGrid.shape[0])
    
    for x in np.arange(anoms.size):
        
        anomYr = anomGrid[x,:,:]
        anomYr = np.ma.ravel(anomYr)
        
        mask = ~anomYr.mask
        lat1dYr = lat1d[mask]
        wgts = np.cos(lat1dYr*rad)
        
        anomYr = anomYr.compressed()
        
        anoms[x] = np.average(anomYr,weights=wgts)

    return anoms

def outputMontanaAnoms():
    
    yrs = np.arange(1895,2013)
    mths = np.arange(1,13)

    tminAnn = np.ma.masked_equal(np.load('/projects/daymet2/montana_trends/grid_anoms_ann_tmin.npy'),MISSING_VAL)
    tmaxAnn = np.ma.masked_equal(np.load('/projects/daymet2/montana_trends/grid_anoms_ann_tmax.npy'),MISSING_VAL)
    
    tminMthly = []
    tmaxMthly = []
    for mth in mths:
        tminMthly.append(np.ma.masked_equal(np.load('/projects/daymet2/montana_trends/grid_anoms_mth%02d_tmin.npy'%mth),MISSING_VAL))
        tmaxMthly.append(np.ma.masked_equal(np.load('/projects/daymet2/montana_trends/grid_anoms_mth%02d_tmax.npy'%mth),MISSING_VAL))
    
    dsGrid = RasterDataset('/projects/daymet2/docs/ccs_lecture/maskQtrDegMT.tif')
    lat,lon = dsGrid.getCoordGrid1d()
    latG,lonG = dsGrid.getCoordMeshGrid()
    
    tminAnnAnoms = calcAnnGridAnoms(latG, tminAnn)
    tmaxAnnAnoms = calcAnnGridAnoms(latG, tmaxAnn)
    
    tminMthlyAnom = np.zeros((yrs.size,12))
    tmaxMthlyAnom = np.zeros((yrs.size,12))
    
    for mth in mths:
        tminMthlyAnom[:,mth-1] = calcAnnGridAnoms(latG, tminMthly[mth-1])
        tmaxMthlyAnom[:,mth-1] = calcAnnGridAnoms(latG, tmaxMthly[mth-1])
    
    def writeAnoms(fpath,annAnoms,mthlyAnoms):
        
        fout = open(fpath,'w')
        fout.write(",".join(['YEAR','ANN','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC\n']))
        
        for x in np.arange(yrs.size):
            
            line = []
            line.append(str(yrs[x]))
            line.append("%.2f"%annAnoms[x])
            line.extend(["%.2f"%mthlyAnoms[x,mth-1] for mth in mths])
            line[-1] = "".join([line[-1],"\n"])
            
            fout.write(",".join(line))
        
        fout.close()
        
    writeAnoms('/projects/daymet2/montana_trends/montana_ann_anomalies_tmin.csv', tminAnnAnoms, tminMthlyAnom)
    writeAnoms('/projects/daymet2/montana_trends/montana_ann_anomalies_tmax.csv', tmaxAnnAnoms, tmaxMthlyAnom)

def barPlotMthlyTrends():
    
    tminAnoms = np.loadtxt('/projects/daymet2/montana_trends/montana_ann_anomalies_tmin.csv',skiprows=1,delimiter=',')
    tmaxAnoms = np.loadtxt('/projects/daymet2/montana_trends/montana_ann_anomalies_tmax.csv',skiprows=1,delimiter=',')
    
    yrs = tminAnoms[:,0]
    tminAnoms = tminAnoms[:,1:]
    tmaxAnoms = tmaxAnoms[:,1:]
    
    #plt.plot(runningMean(tminAnoms[:,2],10))
    #plt.plot(tminAnoms[:,2])
    print stats.linregress(yrs,tminAnoms[:,0])[0]*10
    print stats.linregress(yrs,tmaxAnoms[:,0])[0]*10
    #plt.show()
    
    tminTrends = np.array([stats.linregress(yrs,tminAnoms[:,x])[0] for x in np.arange(tminAnoms.shape[1])])*10
    tmaxTrends = np.array([stats.linregress(yrs,tmaxAnoms[:,x])[0] for x in np.arange(tmaxAnoms.shape[1])])*10
    
    bwidth = .25
#        
#    plt.sca(ax1)
    plt.bar(np.arange(tminTrends.size),tminTrends,width=bwidth,color="k")
    plt.bar(np.arange(tmaxTrends.size)+bwidth,tmaxTrends,width=bwidth,color="grey")
    plt.legend(("Tmin","Tmax"),fontsize=10)
    plt.xticks(np.arange(tminTrends.size) + (bwidth*2)/2.0, ('Ann','Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.xlim(-.25,tminTrends.size-1+.75)
    plt.hlines(0, -.25, tminTrends.size-1+.75)
    
    ax1 =plt.gca()
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    plt.ylabel(r'$^\circ$C decade$^{-1}$')
    plt.title("Montana Temperature Trends: 1895-2012")
    plt.show()

if __name__ == '__main__':
    
    barPlotMthlyTrends()
    #outputMontanaAnoms()
    
#    print "TMIN"
#    calcGriddedMontanaTrends('/projects/daymet2/montana_trends/anom_ann_tmin.npy','tmin', 'ann',mth=None)
#    print "TMAX"
#    calcGriddedMontanaTrends('/projects/daymet2/montana_trends/anom_ann_tmax.npy','tmax', 'ann',mth=None)
##    
#    for mth in np.arange(1,13):
#        print "MONTH "+str(mth)
#        calcGriddedMontanaTrends('/projects/daymet2/montana_trends/anom_mthly_tmin.npy','tmin', 'mth%02d'%mth,mth=mth)
#        calcGriddedMontanaTrends('/projects/daymet2/montana_trends/anom_mthly_tmax.npy','tmax', 'mth%02d'%mth,mth=mth)



    
    #calcMontanaAnomsMthly()
    #plotUshcnStations()
    #barPlotNormals()
    #barPlotMtSeasonalTrends()
    
#    calcConusTrendsSeason(np.array([12,1,2]), "Winter", 'tmin')
#    calcConusTrendsSeason(np.array([12,1,2]), "Winter", 'tmax')
    
#    calcConusTrendsSeason(np.array([3,4,5]), "Spring", 'tmin')
#    calcConusTrendsSeason(np.array([3,4,5]), "Spring", 'tmax')
#    
#    calcConusTrendsSeason(np.array([6,7,8]), "Summer", 'tmin')
#    calcConusTrendsSeason(np.array([6,7,8]), "Summer", 'tmax')
#    
#    calcConusTrendsSeason(np.array([9,10,11]), "Fall", 'tmin')
#    calcConusTrendsSeason(np.array([9,10,11]), "Fall", 'tmax')
    
    #plotConusTrends()
    #plotConusTrendsMap()
    #plotConusTrends()
    
#    mask_to_shp('/projects/daymet2/dem/st_bounds/montanaWGS84.shp', 'montanaWGS84', 
#                '/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif',
#                '/projects/daymet2/docs/ccs_lecture/maskQtrDegMT.tif')
    
    #plotConusTrends()