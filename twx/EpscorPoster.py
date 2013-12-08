'''
Created on Aug 15, 2013

@author: jared.oyler
'''
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from db.station_data import station_data_infill,STN_ID,LON,LAT,NEON,YEAR,BAD
import brewer2mpl
import db.station_data as stnData
from copy import copy
from matplotlib.mlab import griddata
from utils.input_raster import RasterDataset
import db.ushcn as ushcn
from scipy import stats
from matplotlib.colors import Normalize
from osgeo import gdal,gdalconst,osr,ogr
import os
from matplotlib import cm

llcrnrlat=41.675096
urcrnrlat=50.35
llcrnrlon=-118.004842
urcrnrlon=-99.481707
lon_0 = (llcrnrlon+urcrnrlon)/2.0
lat_0 = (llcrnrlat+urcrnrlat)/2.0

MT_CLIMDIVS = np.arange(2401,2408)

def mask_stns(stns,climDivs=MT_CLIMDIVS):
    
    if climDivs is not None and not isinstance(climDivs, np.ndarray):
        climDivs = np.array([climDivs])
    
    lonMask = np.logical_and(stns[LON]>=llcrnrlon-2,stns[LON]<=urcrnrlon+2)
    latMask = np.logical_and(stns[LAT]>=llcrnrlat-2,stns[LAT]<=urcrnrlat+2)
    fnlMask = np.logical_and(np.logical_and(lonMask,latMask),np.isnan(stns[BAD]))
    
    if climDivs is not None:
        fnlMask = np.logical_and(np.in1d(stns[NEON], climDivs, False),fnlMask)
        
    stns = stns[fnlMask]
    return stns

def getAnnAnoms(stnda,stns):
    print "loading obs....."
    obs = stnda.load_obs(stns[STN_ID])
    print "aggregating to ann....."
    agg = ushcn.TairAggregate(stnda.days)
    obsAnn = agg.dailyToAnn(obs)
    
    uYrs = np.unique(stnda.days[YEAR])
    baseMask = np.logical_and(uYrs >= 1961,uYrs <= 1990)
    
    obsAnom = obsAnn-np.mean(obsAnn[baseMask,:],axis=0)
    
    return obsAnom
  
def calcMontanaTrends():
    
    stndaRawTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serial_tmin.nc', 'tmin',stn_dtype=stnData.DTYPE_STN_MEAN_LST_TDI)
    stndaRawTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serial_tmax.nc', 'tmax',stn_dtype=stnData.DTYPE_STN_MEAN_LST_TDI)
    stndaHomogTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin',stn_dtype=stnData.DTYPE_STN_MEAN_LST_TDI)
    stndaHomogTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax',stn_dtype=stnData.DTYPE_STN_MEAN_LST_TDI)
    days = stndaRawTmin.days
    
    
    stnsHomogTmin = mask_stns(stndaHomogTmin.stns, True)
    stnsHomogTmax = mask_stns(stndaHomogTmax.stns, True)
    stnsRawTmin = stndaRawTmin.stns[np.in1d(stndaRawTmin.stns[STN_ID], stnsHomogTmin[STN_ID], True)]
    stnsRawTmax = stndaRawTmax.stns[np.in1d(stndaRawTmax.stns[STN_ID], stnsHomogTmax[STN_ID], True)]
    
    tminRawTrends = []
    tmaxRawTrends = []
    tminHomogTrends = []
    tmaxHomogTrends = []
    
    uYrs = np.unique(days[YEAR])
    
    climDivs = [MT_CLIMDIVS]
    climDivs.extend(MT_CLIMDIVS)
    
    for aDiv in climDivs:
        
        stnsHomogTmin = mask_stns(stndaHomogTmin.stns, aDiv)
        stnsHomogTmax = mask_stns(stndaHomogTmax.stns, aDiv)
        stnsRawTmin = stndaRawTmin.stns[np.in1d(stndaRawTmin.stns[STN_ID], stnsHomogTmin[STN_ID], True)]
        stnsRawTmax = stndaRawTmax.stns[np.in1d(stndaRawTmax.stns[STN_ID], stnsHomogTmax[STN_ID], True)]
        
        obsAnomRawTmin = getAnnAnoms(stndaRawTmin, stnsRawTmin)
        obsAnomRawTmax = getAnnAnoms(stndaRawTmax, stnsRawTmax)
        obsAnomHomogTmin = getAnnAnoms(stndaHomogTmin, stnsHomogTmin)
        obsAnomHomogTmax = getAnnAnoms(stndaHomogTmax, stnsHomogTmax)
    
        obsMeanAnomRawTmin = np.mean(obsAnomRawTmin,axis=1)
        obsMeanAnomRawTmax = np.mean(obsAnomRawTmax,axis=1)
        obsMeanAnomHomogTmin = np.mean(obsAnomHomogTmin,axis=1)
        obsMeanAnomHomogTmax = np.mean(obsAnomHomogTmax,axis=1)
    
        trendRawTmin = stats.linregress(uYrs,obsMeanAnomRawTmin)[0]*60
        trendRawTmax = stats.linregress(uYrs,obsMeanAnomRawTmax)[0]*60
        trendHomogTmin = stats.linregress(uYrs,obsMeanAnomHomogTmin)[0]*60
        trendHomogTmax = stats.linregress(uYrs,obsMeanAnomHomogTmax)[0]*60
        
        tminRawTrends.append(trendRawTmin)
        tmaxRawTrends.append(trendRawTmax)
        tminHomogTrends.append(trendHomogTmin)
        tmaxHomogTrends.append(trendHomogTmax)
    
    for x in np.arange(len(climDivs)):
        print "ClimDiv %d TMIN: %.3f,%.3f,%.3f"%(x,tminRawTrends[x],tminHomogTrends[x],tminHomogTrends[x]-tminRawTrends[x])
    print "########################"
    for x in np.arange(len(climDivs)):
        print "ClimDiv %d TMAX: %.3f,%.3f,%.3f"%(x,tmaxRawTrends[x],tmaxHomogTrends[x],tmaxHomogTrends[x]-tmaxRawTrends[x])
    
    trends = np.array((tminRawTrends,tminHomogTrends,tmaxRawTrends,tmaxHomogTrends))
    print trends.shape
    np.save('/projects/daymet2/docs/epscor_summit_2013_poster/trendsData.npy',trends)

def plotMontanaTrends():
    trends = np.load('/projects/daymet2/docs/epscor_summit_2013_poster/trendsData.npy')
    n_groups = 7
    
    index = np.arange(n_groups)
    bar_width = 0.25
    colors = ['#E41A1C','#377EB8','#4DAF4A','#984EA3','#FF7F00','#FFFF33','#A65628']
    
    plt.subplot(121) 
    rects1 = plt.bar(index, trends[0,1:], bar_width,
                     color=colors,
                     label='Non-Homogenized')
        
    rects2 = plt.bar(index + bar_width, trends[1,1:], bar_width,
                     color=colors,
                     label='Homogenized',
                     hatch="/")
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    
    plt.xlabel('Climate Division',fontsize=17)
    plt.ylabel(r'$^\circ$C / 60yrs',fontsize=17)
    plt.setp(plt.gca().get_yticklabels(), fontsize=17)
    plt.xticks(index + bar_width, ('1', '2', '3', '4', '5','6','7'),fontsize=17)
    p = mpl.patches.Rectangle((0,0),1,1,fc="white")
    p2 = mpl.patches.Rectangle((0,0),1,1,fc="white",hatch="//")
    plt.legend([p,p2], ["Non-Homog.","Homog."],fontsize=17,loc=2)
    plt.title("Tmin",fontsize=17)
    
    plt.subplot(122) 
    rects1 = plt.bar(index, trends[2,1:], bar_width,
                     color=colors,
                     label='Non-Homogenized')
        
    rects2 = plt.bar(index + bar_width, trends[3,1:], bar_width,
                     color=colors,
                     label='Homogenized',
                     hatch="/")
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    
    plt.xlabel('Climate Division',fontsize=17)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.xticks(index + bar_width, ('1', '2', '3', '4', '5','6','7'),fontsize=17)
    plt.title("Tmax",fontsize=17)
    
    plt.subplots_adjust(wspace=0.02)
    cf = plt.gcf()
    cf.set_size_inches(10,5)
    plt.tight_layout()
    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/trendBarPlot.png',dpi=300)
    plt.show()
    
def plotStnTrendMaps():
    
    stndaRawTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serial_tmin.nc', 'tmin',stn_dtype=stnData.DTYPE_STN_MEAN_LST_TDI)
    stndaRawTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serial_tmax.nc', 'tmax',stn_dtype=stnData.DTYPE_STN_MEAN_LST_TDI)
    stndaHomogTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin',stn_dtype=stnData.DTYPE_STN_MEAN_LST_TDI)
    stndaHomogTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax',stn_dtype=stnData.DTYPE_STN_MEAN_LST_TDI)
    days = stndaRawTmin.days
    
    stnsRawTmin = mask_stns(stndaRawTmin.stns,climDivs=None)
    stnsRawTmax = mask_stns(stndaRawTmax.stns,climDivs=None)
    stnsHomogTmin = mask_stns(stndaHomogTmin.stns,climDivs=None)
    stnsHomogTmax = mask_stns(stndaHomogTmax.stns,climDivs=None)
        
    obsAnomRawTmin = getAnnAnoms(stndaRawTmin, stnsRawTmin)
    obsAnomRawTmax = getAnnAnoms(stndaRawTmax, stnsRawTmax)
    obsAnomHomogTmin = getAnnAnoms(stndaHomogTmin, stnsHomogTmin)
    obsAnomHomogTmax = getAnnAnoms(stndaHomogTmax, stnsHomogTmax)
    
    def getTrends(obsAnom,days):
        
        uYrs = np.unique(days[YEAR])
        #slope, intercept, r_value, p_value, std_err = stats.linregress(uYrs,obsAnom[:,0])
        trends = np.zeros(obsAnom.shape[1])
        for x in np.arange(obsAnom.shape[1]):
            trends[x] = stats.linregress(uYrs,obsAnom[:,x])[0]*10
        
        return trends
    
    trendsRawTmin = getTrends(obsAnomRawTmin, days)
    trendsRawTmax = getTrends(obsAnomRawTmax, days)
    trendsHomogTmin = getTrends(obsAnomHomogTmin, days)
    trendsHomogTmax = getTrends(obsAnomHomogTmax, days)
    
#    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
#    stns = stndaUS.stns
#    lonMask = np.logical_and(stns[LON]>=llcrnrlon-2,stns[LON]<=urcrnrlon+2)
#    latMask = np.logical_and(stns[LAT]>=llcrnrlat-2,stns[LAT]<=urcrnrlat+2)
#    
#    stns = stns[np.logical_and(lonMask,latMask)]
#    obsFLs = stndaUS.loadObs(stns[STN_ID], 'FLs.52i_tmin')
#    obsFLsAnn = agg.mthlyToAnn(obsFLs)
#    obsFLsAnom = obsFLsAnn-np.mean(obsFLsAnn[baseMask,:],axis=0)
#    
#    trends = np.zeros(stns.size)
#    for x in np.arange(stns.size):
#        trends[x] = stats.linregress(uYrs,obsFLsAnom[:,x])[0]*10
#    
#    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
#    lat,lon = dsGrid.getCoordGrid1d()
#    lonMask = np.nonzero(np.logical_and(lon>=llcrnrlon-2,lon<=urcrnrlon+2))[0]
#    latMask = np.nonzero(np.logical_and(lat>=llcrnrlat-2,lat<=urcrnrlat+2))[0]
#    lon = lon[lonMask]
#    lat = lat[latMask]
#    lon = np.sort(lon)
#    lat = np.sort(lat)
    
    ds = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc')
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    lonMask = np.nonzero(np.logical_and(lon>=llcrnrlon-2,lon<=urcrnrlon+2))[0]
    latMask = np.nonzero(np.logical_and(lat>=llcrnrlat-2,lat<=urcrnrlat+2))[0]
    lon = lon[lonMask]
    lat = lat[latMask]
    lon = np.sort(lon)
    lat = np.sort(lat)
    
    print "Gridding data...."
    trendGridRawTmin = griddata(stnsRawTmin[LON], stnsRawTmin[LAT], trendsRawTmin, lon, lat)
    trendGridRawTmax = griddata(stnsRawTmax[LON], stnsRawTmax[LAT], trendsRawTmax, lon, lat)
    trendGridHomogTmin = griddata(stnsHomogTmin[LON], stnsHomogTmin[LAT], trendsHomogTmin, lon, lat)
    trendGridHomogTmax = griddata(stnsHomogTmax[LON], stnsHomogTmax[LAT], trendsHomogTmax, lon, lat)
    
    print "Mapping data...."
    m = Basemap(resolution='l',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
    
    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
    clrs = clrs.mpl_colors
    clrs[5] = "grey"
    levels = [-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]
    
    
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)#,cbar_size="3%")#,cbar_pad=0.02)#axes_pad=.625
    m.ax = grid[0]
    m.drawcountries(linewidth=2)
    m.drawstates(linewidth=2)

    cf = m.contourf(x, y, trendGridRawTmin,colors=clrs,levels=levels)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0],ticks=levels)#,label='$^\circ$C / decade')
    cbar.ax.tick_params(labelsize=17)
    grid[0].set_ylabel("Tmin",fontsize=17)
    grid[0].set_title("Non-homogenized",fontsize=17)
    #cbar = grid.cbar_axes.colorbar(cf)
    cbar.set_ticks(levels)
    cbar.set_label(r'$^\circ$C / decade',fontsize=17)
    #cbar.ax.xaxis.tick_top()
    
    m.ax = grid[1]
    m.drawcountries(linewidth=2)
    m.drawstates(linewidth=2)
    grid[1].set_title("Homogenized",fontsize=17)
    cf = m.contourf(x, y, trendGridHomogTmin,colors=clrs,levels=levels)
    
    m.ax = grid[2]
    m.drawcountries(linewidth=2)
    m.drawstates(linewidth=2)
    grid[2].set_ylabel("Tmax",fontsize=17)
    cf = m.contourf(x, y, trendGridRawTmax,colors=clrs,levels=levels)
    
    m.ax = grid[3]
    m.drawcountries(linewidth=2)
    m.drawstates(linewidth=2)
    cf = m.contourf(x, y, trendGridHomogTmax,colors=clrs,levels=levels)
    
    cf = plt.gcf()
    cf.set_size_inches(8*1.5,6*1.5)
    
    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/trendMaps.png',dpi=300)
    plt.show()
        
#    for i in np.arange(len(grid.axes_all)):
#        print i
#        gridCell = grid[i]
#        m.ax = gridCell
#        m.drawcountries()
#        m.drawstates()
#        
#        cf = m.contourf(x,y,bases[i],100,cmap=cmaps[i])
#        cbar = gridCell.cax.colorbar(cf)
#    
#    plt.show()
#    
#    
#    
#    
#    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
#    clrs = clrs.mpl_colors
#    clrs[5] = "grey"
#    
#    m.contourf(x, y, trendGrid,colors=clrs,levels=[-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50])
#    cbar = plt.colorbar()
#    cbar.set_ticks([-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50])
#    plt.show()


def plotStns():
    
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    
    stns = stndaTmax.stns
    stnsTmin = stndaTmin.stns[~np.in1d(stndaTmin.stns[STN_ID], stns[STN_ID], True)]
    
    stnids = np.concatenate((stns[STN_ID],stnsTmin[STN_ID]))
    sIdx = np.argsort(stnids)
    
    stnids = stnids[sIdx]
    lon = np.concatenate((stns[LON],stnsTmin[LON]))[sIdx]
    lat = np.concatenate((stns[LAT],stnsTmin[LAT]))[sIdx]
    
    #Setup map
    ax = plt.subplot(111)
    m = Basemap(resolution='h',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    m.drawcountries(linewidth=2)
    m.drawstates(linewidth=2)

    
    maskSntl = np.char.startswith(stnids, 'SNOTEL')
    maskGHCN = np.char.startswith(stnids, 'GHCN')
    maskRaws = np.char.startswith(stnids, 'RAW')
    
    ghcn = m.scatter(lon[maskGHCN],lat[maskGHCN],latlon=True,s=5,c='k',edgecolors='k',facecolors='k')#s=2c='#8C8C8C',edgecolors='#8C8C8C',facecolors='#8C8C8C')
    raws = m.scatter(lon[maskRaws],lat[maskRaws],latlon=True,s=5,c='#E41A1C',edgecolors='#E41A1C',facecolors='#E41A1C',marker='v')
    snotel = m.scatter(lon[maskSntl],lat[maskSntl],latlon=True,s=5,c='#377EB8',edgecolors='#377EB8',facecolors='#377EB8',marker='D')
    plt.legend([ghcn,raws,snotel],['GHCN-D','RAWS','SNOTEL'],loc=3,fontsize=17)
    #plt.legend([ghcn,raws,snotel],['GHCN-D','RAWS','SNOTEL'],fontsize=17,loc=3)
    fig = plt.gcf()
    #fig.set_size_inches(8*1.25,6*1.25)
    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/stnsMap.png',dpi=300)
    plt.show()


def climDivBoxPlot(climDivData):
    
    print [x.size for x in climDivData]
    
    climDivs = np.arange(2401,2408)
    colors = ['#E41A1C','#377EB8','#4DAF4A','#984EA3','#FF7F00','#FFFF33','#A65628']
    means = [np.mean(x) for x in climDivData]
    
    bp = plt.boxplot(climDivData,vert=True,sym="+",patch_artist=True)
    plt.setp(bp['boxes'], color='grey')
    plt.setp(bp['whiskers'], color='black',linestyle="-")
    plt.setp(bp['fliers'], color='black')
    plt.setp(bp['medians'], color='black')
    
    for x in np.arange(len(colors)):
        plt.setp(bp['boxes'][x], color=colors[x],edgecolor="black")
        
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    s = plt.scatter(np.arange(1,len(means)+1),means,color='black')
    s.set_zorder(20)
    locs,labs = plt.xticks(np.arange(1,8),[str(x) for x in np.arange(1,8)])
    plt.xlabel("Climate Division")

def homogAdjBoxplot():
    
    stndtype = copy(stnData.DTYPE_STN_DFLT)
    stndtype.append(('xval_overall_bias',np.float64))
    stndtype.append(('xval_overall_mae',np.float64))
    stndtype.append(('xval_overall_r2',np.float64))
    
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax',stn_dtype=stndtype)
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin',stn_dtype=stndtype)
    
    climDivs = np.arange(2401,2408)
    colors = ['#E41A1C','#377EB8','#4DAF4A','#984EA3','#FF7F00','#FFFF33','#A65628']
    
    databiasTmin = []
    databiasTmax = []
    
    dataMabTmin = []
    dataMabTmax = []

    dataMaeTmin = []
    dataMaeTmax = []
    
    dataR2Tmin = []
    dataR2Tmax = []
    
    #Error by ClimDiv
    for aDiv in climDivs:
        
        biasTminDiv = stndaTmin.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_bias']),stndaTmin.stns[NEON]==aDiv)]
        biasTmaxDiv = stndaTmax.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_bias']),stndaTmax.stns[NEON]==aDiv)]
        databiasTmin.append(biasTminDiv)
        databiasTmax.append(biasTmaxDiv)
        
        mabTminDiv = np.abs(stndaTmin.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_bias']),stndaTmin.stns[NEON]==aDiv)])
        mabTmaxDiv = np.abs(stndaTmax.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_bias']),stndaTmax.stns[NEON]==aDiv)])
        dataMabTmin.append(mabTminDiv)
        dataMabTmax.append(mabTmaxDiv)
        
        maeTminDiv = stndaTmin.stns['xval_overall_mae'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_mae']),stndaTmin.stns[NEON]==aDiv)]
        maeTmaxDiv = stndaTmax.stns['xval_overall_mae'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_mae']),stndaTmax.stns[NEON]==aDiv)]
        dataMaeTmin.append(maeTminDiv)
        dataMaeTmax.append(maeTmaxDiv)
        
        r2TminDiv = stndaTmin.stns['xval_overall_r2'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_r2']),stndaTmin.stns[NEON]==aDiv)]
        r2TmaxDiv = stndaTmax.stns['xval_overall_r2'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_r2']),stndaTmax.stns[NEON]==aDiv)]
        dataR2Tmin.append(r2TminDiv)
        dataR2Tmax.append(r2TmaxDiv)
        
    #Overall Montana Error
    biasTminMt = np.mean(stndaTmin.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_bias']),
                                                                     np.in1d(stndaTmin.stns[NEON], climDivs, False))])
    biasTmaxMt = np.mean(stndaTmax.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_bias']),
                                                                     np.in1d(stndaTmax.stns[NEON], climDivs, False))])
    mabTminMt = np.mean(np.abs(stndaTmin.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_bias']),
                                                                     np.in1d(stndaTmin.stns[NEON], climDivs, False))]))
    mabTmaxMt = np.mean(np.abs(stndaTmax.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_bias']),
                                                                     np.in1d(stndaTmax.stns[NEON], climDivs, False))]))  
    maeTminMt = np.mean(stndaTmin.stns['xval_overall_mae'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_mae']),
                                                                     np.in1d(stndaTmin.stns[NEON], climDivs, False))])
    maeTmaxMt = np.mean(stndaTmax.stns['xval_overall_mae'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_mae']),
                                                                     np.in1d(stndaTmax.stns[NEON], climDivs, False))])
    r2TminMt = np.mean(stndaTmin.stns['xval_overall_r2'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_r2']),
                                                                     np.in1d(stndaTmin.stns[NEON], climDivs, False))])
    r2TmaxMt = np.mean(stndaTmax.stns['xval_overall_r2'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_r2']),
                                                                     np.in1d(stndaTmax.stns[NEON], climDivs, False))])
    print "TMIN MT ERROR: ",biasTminMt,mabTminMt,maeTminMt,r2TminMt
    print "TMAX MT ERROR: ",biasTmaxMt,mabTmaxMt,maeTmaxMt,r2TmaxMt
    
    plt.subplot(221)
    climDivBoxPlot(databiasTmin)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), fontsize=17)
    plt.gca().set_xlabel("")
    plt.gca().set_ylabel("Bias ($^\circ$C)",fontsize=17)
    plt.title("Tmin",fontsize=17)
    ymin,ymax  = plt.ylim()
    plt.ylim(-4,3)
    plt.subplot(222)
    climDivBoxPlot(databiasTmax)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.gca().set_xlabel("")
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.title("Tmax",fontsize=17)
    plt.ylim(-4,3)
     
    plt.subplot(223)
    climDivBoxPlot(dataMabTmin)
    plt.setp(plt.gca().get_xticklabels(), fontsize=17)
    plt.gca().set_xlabel("Climate Division",fontsize=17)
    plt.setp(plt.gca().get_yticklabels(), fontsize=17)
    plt.setp(plt.gca().get_xticklabels(), fontsize=17)
    plt.gca().set_xlabel("Climate Division",fontsize=17)
    plt.gca().set_ylabel("MAE Mean Value ($^\circ$C)",fontsize=17)
    ymin,ymax  = plt.ylim()
    ymax = 3.5
    plt.ylim(0,ymax)
    plt.subplot(224)
    climDivBoxPlot(dataMabTmax)
    plt.setp(plt.gca().get_xticklabels(), fontsize=17)
    plt.gca().set_xlabel("Climate Division",fontsize=17)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.ylim(0,ymax)
    
    plt.subplots_adjust(wspace=0.005)
    plt.gcf().set_size_inches(6,8)
    plt.tight_layout()
    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/Boxplots1.png',dpi=300)
    plt.show()
    
    plt.clf()
    
    plt.subplot(221)
    climDivBoxPlot(dataMaeTmin)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), fontsize=17)
    plt.gca().set_xlabel("")
    plt.gca().set_ylabel("MAE Daily Values ($^\circ$C)",fontsize=17)
    ymin,ymax  = plt.ylim()
    plt.ylim(0,3.5)
    plt.title("Tmin",fontsize=17)
    plt.subplot(222)
    climDivBoxPlot(dataMaeTmax)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.gca().set_xlabel("")
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.ylim(0,3.5)
    plt.title("Tmax",fontsize=17)
    
    plt.subplot(223)
    climDivBoxPlot(dataR2Tmin)
    plt.setp(plt.gca().get_yticklabels(), fontsize=17)
    plt.setp(plt.gca().get_xticklabels(), fontsize=17)
    plt.gca().set_ylabel("R$^2$ Daily Values",fontsize=17)
    plt.gca().set_xlabel("Climate Division",fontsize=17)
    ymin,ymax  = plt.ylim()
    plt.ylim(0.92,1)
    plt.subplot(224)
    climDivBoxPlot(dataR2Tmax)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.setp(plt.gca().get_xticklabels(), fontsize=17)
    plt.gca().set_xlabel("Climate Division",fontsize=17)
    plt.ylim(0.92,1)
    
    plt.subplots_adjust(wspace=0.005)
    plt.gcf().set_size_inches(6,8)
    plt.tight_layout()
    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/Boxplots2.png',dpi=300)
    plt.show()
    
#    dataMabTmin.extend(dataMabTmax)
#    data = dataMabTmin
#    #data = data[::-1]
#    means = [np.mean(x) for x in data]
#    
#    bp = plt.boxplot(data,vert=True,sym="",patch_artist=True)
#    plt.setp(bp['boxes'], color='grey')
#    plt.setp(bp['whiskers'], color='black',linestyle="-")
#    plt.setp(bp['fliers'], color='black')
#    plt.setp(bp['medians'], color='black')
#    
#    colors.extend(colors)
#    for x in np.arange(len(colors)):
#        plt.setp(bp['boxes'][x], color=colors[x],edgecolor="black")
#    
#    
#    ax = plt.gca()
#    ax.set_axisbelow(True)
#    ax.yaxis.grid(color='gray', linestyle='dashed')
#    s = plt.scatter(np.arange(1,len(means)+1),means,color='black')
#    s.set_zorder(20)
    
    
    
#    top = 4
#    
#    nBoxes = len(data)
#    pos = np.arange(nBoxes)+1
#    upperLabels = [str(np.round(s, 2))+' $^\circ$C' for s in means]
#    weights = ['bold', 'semibold']
#    for tick,label in zip(range(nBoxes),ax.get_yticklabels()):
#        k = tick % 2
#        ax.text(top-(top*0.1),pos[tick], upperLabels[tick],
#            horizontalalignment='center',verticalalignment='center', size='x-small', weight='bold')
#    
#    plt.yticks([1,2,3,4,5,6],['RAWS','SNOTEL','GHCN-D','RAWS','SNOTEL','GHCN-D'])
#    ymin,ymax = plt.ylim()
#    plt.vlines(7.5, ymin, ymax)
#    plt.ylim(0,ymax)
#    plt.text(-3.9,6.25,'Tmin',weight='bold')
#    plt.text(-3.9,3.25,'Tmax',weight='bold')
#    plt.xlabel("Adjustment ($^\circ$C)")
#    plt.savefig('/projects/daymet2/docs/final_writeup/homogAdjBoxplots.png',dpi=300)

#    plt.subplots_adjust(wspace=0.005)
#    plt.gcf().set_size_inches(8,16)
#    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/Boxplots.png',dpi=300)
#    plt.show()

def xvalErrMaps():
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    lat,lon = dsGrid.getCoordGrid1d()
    lonMask = np.nonzero(np.logical_and(lon>=llcrnrlon-2,lon<=urcrnrlon+2))[0]
    latMask = np.nonzero(np.logical_and(lat>=llcrnrlat-2,lat<=urcrnrlat+2))[0]
    lon = lon[lonMask]
    lat = lat[latMask]
    lon = np.sort(lon)
    lat = np.sort(lat)
    
#    ds = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc')
#    lon = ds.variables['lon'][:]
#    lat = ds.variables['lat'][:]
#    lonMask = np.nonzero(np.logical_and(lon>=llcrnrlon-2,lon<=urcrnrlon+2))[0]
#    latMask = np.nonzero(np.logical_and(lat>=llcrnrlat-2,lat<=urcrnrlat+2))[0]
#    lon = lon[lonMask]
#    lat = lat[latMask]
#    lon = np.sort(lon)
#    lat = np.sort(lat)
    
    stndtype = copy(stnData.DTYPE_STN_DFLT)
    stndtype.append(('xval_overall_bias',np.float64))
    stndtype.append(('xval_overall_mae',np.float64))
    stndtype.append(('xval_overall_r2',np.float64))
    
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax',stn_dtype=stndtype)
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin',stn_dtype=stndtype)

    biasTminMask = np.isfinite(stndaTmin.stns['xval_overall_bias'])
    biasTminStns = stndaTmin.stns[biasTminMask]
    biasTmin = biasTminStns['xval_overall_bias']
    mabTmin = np.abs(biasTminStns['xval_overall_bias'])
    
    colors = np.zeros(mabTmin.size,dtype="<S16")
    colors[np.logical_and(mabTmin>=0.0,mabTmin<0.5)] = "blue"
    colors[np.logical_and(mabTmin>=0.5,mabTmin<1.0)] = "purple"
    colors[np.logical_and(mabTmin>=1.0,mabTmin<1.5)] = "green"
    colors[np.logical_and(mabTmin>=1.5,mabTmin<2.0)] = "orange"
    colors[mabTmin>=2.0] = "red"
    
    s = [20*1.4**n for n in range(5)]
    
    sizes = np.zeros(mabTmin.size)
    sizes[np.logical_and(mabTmin>=0.0,mabTmin<0.5)] = s[0]
    sizes[np.logical_and(mabTmin>=0.5,mabTmin<1.0)] = s[1]
    sizes[np.logical_and(mabTmin>=1.0,mabTmin<1.5)] = s[2]
    sizes[np.logical_and(mabTmin>=1.5,mabTmin<2.0)] = s[3]
    sizes[mabTmin>=2.0] = s[4]

    print "Gridding data...."
    #mabTminGrid = griddata(biasTminStns[LON], biasTminStns[LAT], biasTmin, lon, lat)
    
    print "Mapping data...."
    m = Basemap(resolution='l',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    m.drawcountries()
    m.drawstates()
    m.scatter(biasTminStns[LON],biasTminStns[LAT],s=sizes,latlon=True,facecolors="none",c=colors,edgecolors=list(colors))#,edgecolors=colors)
    #x, y = m(*np.meshgrid(lon,lat))
    
    #m.contourf(x, y, mabTminGrid,100,cmap=plt.cm.jet)
    #plt.colorbar()
    plt.show()

def montanaPredictorMaps():
    
    dsElev = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc')
    dsTdi = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc')
    dsLstTmin = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin.nc')
    dsLstTmax = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax.nc')
    
    lon = dsElev.variables['lon'][:]
    lat = dsElev.variables['lat'][:]
    lonMask = np.nonzero(np.logical_and(lon>=llcrnrlon-2,lon<=urcrnrlon+2))[0]
    latMask = np.nonzero(np.logical_and(lat>=llcrnrlat-2,lat<=urcrnrlat+2))[0]
    lon = lon[lonMask]
    lat = lat[latMask]
    
    elev = dsElev.variables['elev'][latMask,lonMask]
    tdi = dsTdi.variables['tdi'][latMask,lonMask]
    lstTmin = dsLstTmin.variables['lst_tmin'][latMask,lonMask]
    lstTmax = dsLstTmax.variables['lst_tmax'][latMask,lonMask]
    
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),axes_pad=.625,cbar_mode="each",cbar_location="right",cbar_pad=0.02)

    m = Basemap(resolution='l',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
    
    bases = [elev*10**-2,tdi,lstTmin,lstTmax]
    #cmaps = [plt.cm.gist_earth,plt.cm.gist_earth,plt.cm.jet,plt.cm.jet]
    cmaps = [plt.cm.gist_earth,#brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=False).mpl_colormap,
             plt.cm.gist_earth,#brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=False).mpl_colormap,
             plt.cm.jet,#brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=True).mpl_colormap,
             plt.cm.jet]#brewer2mpl.get_map('YlOrBr', 'Sequential', 9, reverse=False).mpl_colormap]
    titles = ["Elevation (m * 10$^{-2}$)","TDI (Ridge vs. Valley Index)","MODIS Tmin LST ($^\circ$C)","MODIS Tmax LST ($^\circ$C)"]
    
    
    for i in np.arange(len(grid.axes_all)):
        print i
        gridCell = grid[i]
        m.ax = gridCell
        m.drawcountries(linewidth=2)
        m.drawstates(linewidth=2)
        
        cf = m.contourf(x,y,bases[i],100,cmap=cmaps[i])
        cbar = gridCell.cax.colorbar(cf)
        cbar.ax.tick_params(labelsize=17)
        gridCell.set_title(titles[i],fontsize=17)
    
    cf = plt.gcf()
    cf.set_size_inches(8*1.5,6*1.5)
    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/predictorMaps.png',dpi=300)
    plt.show()

def readMaskedGtiff(fpath):
    ds = gdal.Open(fpath)
    a = ds.ReadAsArray()
    ndata = ds.GetRasterBand(1).GetNoDataValue()
    ndata = np.array([ndata])
    ndata = ndata.astype(a.dtype)[0]
    
    return np.ma.masked_array(a,a==ndata)

def exInterpMaps():
    os.chdir('/projects/daymet2/docs/ncar_workshop2013_poster/EgTile')
    elev = readMaskedGtiff('cropStPlane_tile17Elev1.tif')
    tdi = readMaskedGtiff('cropStPlane_tile17Tdi1.tif')
    lstTmin = readMaskedGtiff('cropStPlane_tile17LstTmin1.tif')
    lstTmax = readMaskedGtiff('cropStPlane_tile17LstTmax1.tif')
    meanTmin = readMaskedGtiff('cropStPlane_tminMean.tif')
    meanTmax = readMaskedGtiff('cropStPlane_tmaxMean.tif')
    seTmin = readMaskedGtiff('cropStPlane_tminSe.tif')
    seTmax = readMaskedGtiff('cropStPlane_tmaxSe.tif')
    
    cf = plt.gcf()
    grid = ImageGrid(cf,121,nrows_ncols=(2,2),axes_pad=.625,cbar_mode="each",cbar_location="right",label_mode = "1",cbar_pad=0.02)
    
    im = grid[0].imshow(elev*10.0**-2,cmap=cm.gist_earth)
    grid[0].set_xticks([])
    grid[0].set_yticks([])
    grid[0].set_title("Elevation (m * 10$^{-2}$)",fontsize=17)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    im = grid[1].imshow(tdi,cmap=cm.gist_earth)
    grid[1].set_xticks([])
    grid[1].set_yticks([])
    grid[1].set_title("TDI (Ridge vs. Valley Index)",fontsize=17)
    cbar = grid.cbar_axes[1].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    im = grid[2].imshow(lstTmin,cmap=cm.jet)
    grid[2].set_xticks([])
    grid[2].set_yticks([])
    grid[2].set_title("MODIS Tmin LST ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[2].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    im = grid[3].imshow(lstTmax,cmap=cm.jet)
    grid[3].set_xticks([])
    grid[3].set_yticks([])
    grid[3].set_title("MODIS Tmax LST ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[3].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    grid = ImageGrid(cf,122,nrows_ncols=(2,2),axes_pad=.625,cbar_mode="each",cbar_location="right",label_mode = "1",cbar_pad=0.02)
    
    im = grid[0].imshow(meanTmin,cmap=cm.jet)
    grid[0].set_xticks([])
    grid[0].set_yticks([])
    grid[0].set_title("Normal Daily Tmin ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    im = grid[1].imshow(meanTmax,cmap=cm.jet)
    grid[1].set_xticks([])
    grid[1].set_yticks([])
    grid[1].set_title("Normal Daily Tmax ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[1].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    norm = Normalize(vmin=np.min(np.array([np.min(seTmin),np.min(seTmax)])),vmax=np.max(np.array([np.max(seTmin),np.max(seTmax)])))
    
    im = grid[2].imshow(seTmin,cmap=cm.jet,norm=norm)
    grid[2].set_xticks([])
    grid[2].set_yticks([])
    grid[2].set_title("Std. Err. Tmin ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[2].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    im = grid[3].imshow(seTmax,cmap=cm.jet,norm=norm)
    grid[3].set_xticks([])
    grid[3].set_yticks([])
    grid[3].set_title("Std. Err. Tmax ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[3].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    
    cf.set_size_inches(8*2,6*2)
    
    
    #fig.subplots_adjust(hspace=0.1)
#    grid[4].imshow(meanTmin)
#    grid[5].imshow(meanTmax)
#    grid[6].imshow(seTmin)
#    grid[7].imshow(seTmax)
    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/exInterpMaps.png',dpi=300)
    plt.show()

def plotMerraVsTopoWx():
    os.chdir('/projects/daymet2/docs/ncar_workshop2013_poster/EgTile')
    tminMerra = readMaskedGtiff('/projects/daymet2/docs/epscor_summit_2013_poster/merraTminStPlaneFnl.tif')
    tminTwx = readMaskedGtiff('/projects/daymet2/docs/ncar_workshop2013_poster/EgTile/cropStPlane_tminMean.tif')
    
#    maskFnl = np.logical_or(tminMerra.mask,tminTwx.mask)
#    tminMerra[maskFnl] = np.ma.masked
#    tminTwx[maskFnl] = np.ma.masked
    
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),axes_pad=.8,cbar_mode="each",cbar_location="right",label_mode = "1",cbar_pad=0.02)
    
    norm = Normalize(np.min(tminTwx),np.max(tminTwx))
    #norm = Normalize(np.min(stnCnts), np.max(stnCnts))

    
    
    im = grid[0].imshow(tminMerra,cmap=cm.jet,norm=norm)
    grid[0].set_xticks([])
    grid[0].set_yticks([])
    grid[0].set_title("MODIS ET Input\nNormal Daily Tmin ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    im = grid[1].imshow(tminTwx,cmap=cm.jet,norm=norm)
    grid[1].set_xticks([])
    grid[1].set_yticks([])
    grid[1].set_title("TopoWx\nNormal Daily Tmin ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[1].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    #cf.set_size_inches(8*2,6*2)
    
    
    #fig.subplots_adjust(hspace=0.1)
#    grid[4].imshow(meanTmin)
#    grid[5].imshow(meanTmax)
#    grid[6].imshow(seTmin)
#    grid[7].imshow(seTmax)
    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/merraVsTwx.png',dpi=300)
    plt.show()

def tileOverviewMap():
    m = Basemap(resolution='l',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
            llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    m.drawcountries(linewidth=2)
    m.drawstates(linewidth=2)
    m.readshapefile('/projects/daymet2/docs/ncar_workshop2013_poster/EgTile/Tile17', 'tile', drawbounds=True,linewidth=2,color='red')
    plt.savefig('/projects/daymet2/docs/epscor_summit_2013_poster/tileOverview.png',dpi=300)
    plt.show()
if __name__ == '__main__':
    #plotMerraVsTopoWx()
    #tileOverviewMap()
    #exInterpMaps()
    #plotStnTrendMaps()
    #plotMontanaTrends()
    #homogAdjBoxplot()
    #xvalErrMaps()
    plotStns()
    #montanaPredictorMaps()