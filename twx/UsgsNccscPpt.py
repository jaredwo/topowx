'''
Created on Sep 4, 2013

@author: jared.oyler
'''
import infill.obs_por as op
import numpy as np
from twx.db.station_data import StationDataDb, LON, LAT,CLIMDIV, StationSerialDataDb,VARIO_NUG,NEON,BAD,YEAR,\
    STN_ID,DTYPE_STN_BASIC, MASK, DATE, YMD,DTYPE_STN_MEAN_LST_TDI,BAD, MEAN_OBS
import twx.db.station_data as stnData
import twx.utils.util_geo as utlg
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from netCDF4 import Dataset
from twx.utils.input_raster import input_raster,RasterDataset
import shapefile
import pickle
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
from matplotlib import cm
from copy import copy
from twx.utils.status_check import StatusCheck
from matplotlib.mlab import amap
import twx.db.ushcn as ushcn
import twx.utils.util_dates as utld
from datetime import datetime
from matplotlib.mlab import griddata 
from twx.infill.inhomo_pha import HomogRawDaily,parsePhaAdj
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import stats
from modis.clip_raster import crop_nodata
import os
from osgeo import gdal,gdalconst,osr,ogr
import brewer2mpl
import sys

def plotStns():
    
    #Load the climate division Line Collections for matplotlib
    lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
    stndaTmax = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
    stndaTmin = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    
    stns = stndaTmax.stns
    stnsTmin = stndaTmin.stns[~np.in1d(stndaTmin.stns[STN_ID], stns[STN_ID], True)]
    
    stnids = np.concatenate((stns[STN_ID],stnsTmin[STN_ID]))
    sIdx = np.argsort(stnids)
    
    stnids = stnids[sIdx]
    lon = np.concatenate((stns[LON],stnsTmin[LON]))[sIdx]
    lat = np.concatenate((stns[LAT],stnsTmin[LAT]))[sIdx]
    
    #Setup map
    ax = plt.subplot(111)
    m = Basemap(resolution='i',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5,area_thresh=10000)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.readshapefile('/projects/daymet2/dem/climate_divisions/NCCSCExtent_roughWGS84', 'NCCSCExtent_roughWGS84', 
                    drawbounds=True,linewidth=2,color="#FF00FF")
    
#    #Put Climate Divisions on map
#    for x in np.arange(lineCollect.size):
#    
#        lines = lineCollect[x]
#        lines.set_edgecolors('k')
#        lines.set_linewidth(0.5)
#        ax.add_collection(lines)
    
    maskSntl = np.char.startswith(stnids, 'SNOTEL')
    maskGHCN = np.char.startswith(stnids, 'GHCN')
    maskRaws = np.char.startswith(stnids, 'RAW')
    
    ghcn = m.scatter(lon[maskGHCN],lat[maskGHCN],latlon=True,s=2,c='#8C8C8C',edgecolors='#8C8C8C',facecolors='#8C8C8C')
    raws = m.scatter(lon[maskRaws],lat[maskRaws],latlon=True,s=5,c='#E41A1C',edgecolors='#E41A1C',facecolors='#E41A1C',marker='v')
    snotel = m.scatter(lon[maskSntl],lat[maskSntl],latlon=True,s=5,c='#377EB8',edgecolors='#377EB8',facecolors='#377EB8',marker='D')
    plt.legend([ghcn,raws,snotel],['GHCN-D','RAWS','SNOTEL'],loc=3)
    #plt.legend([ghcn,raws,snotel],['GHCN-D','RAWS','SNOTEL'],fontsize=17,loc=3)
    fig = plt.gcf()
    #fig.set_size_inches(8*1.25,6*1.25)
    plt.savefig('/projects/daymet2/docs/UsgsConfCall201309/stnsMap.png',dpi=300)
    plt.show()
    
def plotConusTrends():
    m = Basemap(resolution='i',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5,area_thresh= 10000)
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    lat,lon = dsGrid.getCoordGrid1d()
    x, y = m(*np.meshgrid(lon,lat))

    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
    clrs = clrs.mpl_colors
    clrs[5] = "grey"
    levels = [-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]

    tmaxUshcn = np.ma.masked_equal(np.load('/projects/daymet2/docs/final_writeup/conus_trends/trendsHomogUshcnTmax.npy'),-9999)*10
    tmaxTopoWx = np.ma.masked_equal(np.load('/projects/daymet2/docs/final_writeup/conus_trends/trendsHomogTopoWxTmax.npy'),-9999)*10
    tmaxTopoWx[tmaxUshcn.mask] = np.ma.masked
    tminUshcn = np.ma.masked_equal(np.load('/projects/daymet2/docs/final_writeup/conus_trends/trendsHomogUshcnTmin.npy'),-9999)*10
    tminTopoWx = np.ma.masked_equal(np.load('/projects/daymet2/docs/final_writeup/conus_trends/trendsHomogTopoWxTmin.npy'),-9999)*10
    tminTopoWx[tminUshcn.mask] = np.ma.masked
            
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05,cbar_size="2%")#,cbar_pad=0.02)#axes_pad=.625
    m.ax = grid[0]
#    m.drawcountries()
#    m.drawstates()
#    m.drawcoastlines()
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    m.readshapefile('/projects/daymet2/dem/climate_divisions/NCCSCExtent_roughWGS84', 'NCCSCExtent_roughWGS84', 
                    drawbounds=True,linewidth=2,color="#FF00FF")

    cf = m.contourf(x, y, tminUshcn,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0],ticks=levels)#,label='$^\circ$C / decade')
    grid[0].set_ylabel("Tmin")
    grid[0].set_title("USHCN Stations")
    #cbar = grid.cbar_axes.colorbar(cf)
    cbar.set_ticks(levels)
    cbar.set_label(r'$^\circ$C decade$^{-1}$')
    #cbar.ax.xaxis.tick_top()
    
    m.ax = grid[1]
#    m.drawcountries()
#    m.drawstates()
#    m.drawcoastlines()
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    m.readshapefile('/projects/daymet2/dem/climate_divisions/NCCSCExtent_roughWGS84', 'NCCSCExtent_roughWGS84', 
                    drawbounds=True,linewidth=2,color="#FF00FF")
    grid[1].set_title("TopoWx Stations")
    cf = m.contourf(x, y, tminTopoWx,colors=clrs,levels=levels,extend='both')
    
    m.ax = grid[2]
#    m.drawcountries()
#    m.drawstates()
#    m.drawcoastlines()
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    m.readshapefile('/projects/daymet2/dem/climate_divisions/NCCSCExtent_roughWGS84', 'NCCSCExtent_roughWGS84', 
                    drawbounds=True,linewidth=2,color="#FF00FF")
    grid[2].set_ylabel("Tmax")
    cf = m.contourf(x, y, tmaxUshcn,colors=clrs,levels=levels,extend='both')
    
    m.ax = grid[3]
#    m.drawcountries()
#    m.drawstates()
#    m.drawcoastlines()
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    m.readshapefile('/projects/daymet2/dem/climate_divisions/NCCSCExtent_roughWGS84', 'NCCSCExtent_roughWGS84', 
                    drawbounds=True,linewidth=2,color="#FF00FF")
    cf = m.contourf(x, y, tmaxTopoWx,colors=clrs,levels=levels,extend='both')
    
    #cf = plt.gcf()
    #cf.set_size_inches(8*1.5,6*1.5)
    
    plt.savefig('/projects/daymet2/docs/UsgsConfCall201309/trendMaps.png',dpi=150)
    plt.show()

def plotPredictors():
            
    dsElev = RasterDataset('/projects/daymet2/compare/predictors/cce_elev.tif')
    dsTdi = RasterDataset('/projects/daymet2/compare/predictors/cce_tdi.tif')
    dsTmin = RasterDataset('/projects/daymet2/compare/predictors/cce_lst_tmin.tif')
    dsTmax = RasterDataset('/projects/daymet2/compare/predictors/cce_lst_tmax.tif')
    
    elev = dsElev.readAsArray()
    tdi = dsTdi.readAsArray()
    lstTmin = dsTmin.readAsArray()
    lstTmax = dsTmax.readAsArray()
    
    bases = [elev*10**-2,tdi,lstTmin,lstTmax]
    #cmaps = [plt.cm.gist_earth,plt.cm.gist_earth,plt.cm.jet,plt.cm.jet]
    cmaps = [plt.cm.gist_earth,#brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=False).mpl_colormap,
             plt.cm.gist_earth,#brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=False).mpl_colormap,
             plt.cm.jet,#brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=True).mpl_colormap,
             plt.cm.jet]#brewer2mpl.get_map('YlOrBr', 'Sequential', 9, reverse=False).mpl_colormap]
    titles = ["Elevation (m * 10$^{-2}$)","TDI (Ridge vs. Valley Index)","MODIS Tmin LST ($^\circ$C)","MODIS Tmax LST ($^\circ$C)"]
    
        
    dsGrid = dsElev
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
    lon_0 = (llcrnrlon+urcrnrlon)/2.0
    lat_0 = (llcrnrlat+urcrnrlat)/2.0
        
    print "Mapping data...."
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),axes_pad=.625,cbar_mode="each",cbar_location="right",cbar_pad=0.02)
    
    m = Basemap(resolution='i',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
    
    for i in np.arange(len(grid.axes_all)):
        print i
        gridCell = grid[i]
        m.ax = gridCell
        m.drawcountries()
        m.drawstates()
        m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
        
        cf = m.contourf(x,y,bases[i],100,cmap=cmaps[i])
        cbar = gridCell.cax.colorbar(cf)
        gridCell.set_title(titles[i])
    
    cf = plt.gcf() 
    cf.set_size_inches(8*1.5,6*1.5)
    plt.savefig('/projects/daymet2/docs/UsgsConfCall201309/predictorMaps.png',dpi=150)
    plt.show()

def plotCceInterp():
            
    dsTmin = RasterDataset('/projects/daymet2/docs/UsgsConfCall201309/cce_output/tmin_norm/cce_final.tif')
    dsTmax = RasterDataset('/projects/daymet2/docs/UsgsConfCall201309/cce_output/tmax_norm/cce_final.tif')
    dsTminSe = RasterDataset('/projects/daymet2/docs/UsgsConfCall201309/cce_output/tmin_se/cce_final.tif')
    dsTmaxSe = RasterDataset('/projects/daymet2/docs/UsgsConfCall201309/cce_output/tmax_se/cce_final.tif')
    
    tmin = dsTmin.readAsArray()
    tmax = dsTmax.readAsArray()
    tminSe = dsTminSe.readAsArray()
    tmaxSe = dsTmaxSe.readAsArray()
    
    bases = [tmin,tmax,tminSe,tmaxSe]
    #cmaps = [plt.cm.gist_earth,plt.cm.gist_earth,plt.cm.jet,plt.cm.jet]
    cmaps = [plt.cm.jet,#brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=False).mpl_colormap,
             plt.cm.jet,#brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=False).mpl_colormap,
             plt.cm.jet,#brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=True).mpl_colormap,
             plt.cm.jet]#brewer2mpl.get_map('YlOrBr', 'Sequential', 9, reverse=False).mpl_colormap]
    titles = ["Normal Daily Tmin ($^\circ$C)",
              "Normal Daily Tmax ($^\circ$C)",
              "Std. Err. Tmin ($^\circ$C)",
              "Std. Err. Tmax ($^\circ$C)"]
    
#    norm = Normalize(vmin=np.min(np.array([np.min(tminSe),np.min(tmaxSe)])),vmax=np.max(np.array([np.max(tminSe),np.max(tmaxSe)])))
#    norms = [None,None,norm,norm]
    
    seMin = np.min(np.array([np.min(tminSe),np.min(tmaxSe)]))
    seMax = np.max(np.array([np.max(tminSe),np.max(tmaxSe)]))
    #levels = np.linspace(seMin,seMax,100)
    levels = [None,None,np.linspace(seMin,seMax,100),np.linspace(seMin,seMax,100)]
      
    dsGrid = dsTmin
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
    lon_0 = (llcrnrlon+urcrnrlon)/2.0
    lat_0 = (llcrnrlat+urcrnrlat)/2.0
        
    print "Mapping data...."
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),axes_pad=.625,cbar_mode="each",cbar_location="right",cbar_pad=0.02)
    
    m = Basemap(resolution='i',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
    
    for i in np.arange(len(grid.axes_all)):
        print i
        gridCell = grid[i]
        m.ax = gridCell
        m.drawcountries()
        m.drawstates()
        m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
        
        if levels[i] is None:
            cf = m.contourf(x,y,bases[i],100,cmap=cmaps[i])
        else:
            cf = m.contourf(x,y,bases[i],cmap=cmaps[i],levels=levels[i])
            
        cbar = gridCell.cax.colorbar(cf)
        gridCell.set_title(titles[i])
    
    cf = plt.gcf() 
    cf.set_size_inches(8*1.5,6*1.5)
    plt.savefig('/projects/daymet2/docs/UsgsConfCall201309/cceInterp.png',dpi=150)
    plt.show()

def tminTmaxClimDivMap(errSet,subPltNum,sidx,norm=None,cmap=cm.hot_r):
        
    cf = plt.gcf()
    
    grid = ImageGrid(cf,subPltNum,nrows_ncols=(1,2),axes_pad=0.1,cbar_mode="single",cbar_location="right")

    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)

    errAll = np.concatenate(errSet)

    if norm is None:
        norm = Normalize(np.min(errAll), np.max(errAll))

    sm = cm.ScalarMappable(norm, cmap)
    sm.set_array(errAll)

    for i in np.arange(len(grid.axes_all)):
        
        gridCell = grid[i]
        
        m.ax = gridCell
        m.drawcountries(linewidth=0.5,color='white')
        
        lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
        lineCollect = lineCollect[sidx]
    
        #Put Climate Divisions on map
        for x in np.arange(lineCollect.size):
            
            lines = lineCollect[x]
            lines.set_facecolors(sm.to_rgba(errSet[i][x]))
            lines.set_edgecolors('#8C8C8C')
            lines.set_linewidth(0.5)
            gridCell.add_collection(lines)
        m.readshapefile('/projects/daymet2/dem/climate_divisions/NCCSCExtent_roughWGS84', 'NCCSCExtent_roughWGS84', 
                    drawbounds=True,linewidth=2,color="#FF00FF")
    cbar = grid[0].cax.colorbar(sm)
    #cbar.ax.tick_params(labelsize=17) 
    return grid,cbar

def plotInterpErrorMaps():
    
    fontsize=12
    #Load station counts per climate division into dictionary
    stndtype = copy(stnData.DTYPE_STN_DFLT)
    stndtype.append(('xval_overall_bias',np.float64))
    stndtype.append(('xval_overall_mae',np.float64))
    stndtype.append(('xval_overall_r2',np.float64))
    
    stndaTmax = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax',stn_dtype=stndtype)
    stndaTmin = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin',stn_dtype=stndtype)
    climDivs = np.concatenate((stndaTmax.stns[NEON],stndaTmin.stns[NEON]))
    climDivs = np.unique(climDivs[np.isfinite(climDivs)])[2:]
    
    biasTmin = []
    biasTmax = []
    mabTmin = []
    mabTmax = []
    maeTmin = []
    maeTmax = []
    r2Tmin = []
    r2Tmax = []
    
    for div in climDivs:

        biasTmin.append(np.mean(stndaTmin.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_bias']),stndaTmin.stns[NEON]==div)]))
        biasTmax.append(np.mean(stndaTmax.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_bias']),stndaTmax.stns[NEON]==div)]))

        mabTmin.append(np.mean(np.abs(stndaTmin.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_bias']),stndaTmin.stns[NEON]==div)])))
        mabTmax.append(np.mean(np.abs(stndaTmax.stns['xval_overall_bias'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_bias']),stndaTmax.stns[NEON]==div)])))
        
        maeTmin.append(np.mean(stndaTmin.stns['xval_overall_mae'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_mae']),stndaTmin.stns[NEON]==div)]))
        maeTmax.append(np.mean(stndaTmax.stns['xval_overall_mae'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_mae']),stndaTmax.stns[NEON]==div)]))

        r2Tmin.append(np.mean(stndaTmin.stns['xval_overall_r2'][np.logical_and(np.isfinite(stndaTmin.stns['xval_overall_r2']),stndaTmin.stns[NEON]==div)]))
        r2Tmax.append(np.mean(stndaTmax.stns['xval_overall_r2'][np.logical_and(np.isfinite(stndaTmax.stns['xval_overall_r2']),stndaTmax.stns[NEON]==div)]))

    biasTmin = np.array(biasTmin)
    biasTmax = np.array(biasTmax)
    mabTmin = np.array(mabTmin)
    mabTmax = np.array(mabTmax)
    maeTmin = np.array(maeTmin)
    maeTmax = np.array(maeTmax)
    r2Tmin = np.array(r2Tmin)
    r2Tmax = np.array(r2Tmax)
    
    overallTminBias = np.mean(stndaTmin.stns['xval_overall_bias'][np.isfinite(stndaTmin.stns['xval_overall_bias'])])
    overallTmaxBias = np.mean(stndaTmax.stns['xval_overall_bias'][np.isfinite(stndaTmax.stns['xval_overall_bias'])])
    overallTminMab = np.mean(np.abs(stndaTmin.stns['xval_overall_bias'][np.isfinite(stndaTmin.stns['xval_overall_bias'])]))
    overallTmaxMab = np.mean(np.abs(stndaTmax.stns['xval_overall_bias'][np.isfinite(stndaTmax.stns['xval_overall_bias'])]))
    
    overallTminMae = np.mean(stndaTmin.stns['xval_overall_mae'][np.isfinite(stndaTmin.stns['xval_overall_mae'])])
    overallTmaxMae = np.mean(stndaTmax.stns['xval_overall_mae'][np.isfinite(stndaTmax.stns['xval_overall_mae'])])
    
    overallTminR2 = np.mean(stndaTmin.stns['xval_overall_r2'][np.isfinite(stndaTmin.stns['xval_overall_r2'])])
    overallTmaxR2 = np.mean(stndaTmax.stns['xval_overall_r2'][np.isfinite(stndaTmax.stns['xval_overall_r2'])])
    #errSets = [(biasTmin,biasTmax),(mabTmin,mabTmax),(maeTmin,maeTmax),(r2Tmin,r2Tmax)]
    #errSets = [(biasTmin,biasTmax),(mabTmin,mabTmax),(maeTmin,maeTmax),(r2Tmin,r2Tmax)]
    #errSets = [(mabTmin,mabTmax)]
    
    #Get idxs to sort climate divisions by #
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
    
    
    norm = Normalize(vmin=-0.5,vmax=0.5)
    grid,cbar = tminTmaxClimDivMap((biasTmin,biasTmax), 411, sidx,norm=norm,cmap=cm.RdBu_r)
    grid[0].set_title("Tmin")
    grid[0].set_ylabel('a.  ', rotation='horizontal')
    grid[1].set_title("Tmax")
    cbar.set_label_text("Bias ($^\circ$C)",fontsize=fontsize)
    grid[0].text(0.025,0.075,"Overall\nBias: %.2f$^\circ$C"%(overallTminBias,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nBias: %.2f$^\circ$C"%(overallTmaxBias,),transform=grid[1].transAxes,fontsize=10)
    
    grid,cbar = tminTmaxClimDivMap((mabTmin,mabTmax), 412, sidx)
    cbar.set_label_text("MAB ($^\circ$C)",fontsize=fontsize)
    grid[0].set_ylabel('b.  ', rotation='horizontal')
    grid[0].text(0.025,0.075,"Overall\nMAB: %.2f$^\circ$C"%(overallTminMab,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAB: %.2f$^\circ$C"%(overallTmaxMab,),transform=grid[1].transAxes,fontsize=10)
    cbar.ax.set_yticks([.3,.6,.9,1.2,1.5])
    
    grid,cbar = tminTmaxClimDivMap((maeTmin,maeTmax), 413, sidx)
    cbar.set_label_text("MAE ($^\circ$C)",fontsize=fontsize)
    grid[0].set_ylabel('c.  ', rotation='horizontal')
    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(overallTminMae,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(overallTmaxMae,),transform=grid[1].transAxes,fontsize=10)
    
    grid,cbar = tminTmaxClimDivMap((r2Tmin,r2Tmax), 414, sidx)
    cbar.set_label_text("mR$^2$",fontsize=fontsize)
    grid[0].set_ylabel('d.  ', rotation='horizontal')
    grid[0].text(0.025,0.075,"Overall\nmR$^2$: %.2f"%(overallTminR2,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nmR$^2$: %.2f"%(overallTmaxR2,),transform=grid[1].transAxes,fontsize=10)
    
    fig =plt.gcf()
    #fig.set_size_inches(8*2,6*3)
    fig.set_size_inches(8,6*1.5)
    fig.subplots_adjust(hspace=0.05)
    #plt.tight_layout()
    plt.savefig('/projects/daymet2/docs/UsgsConfCall201309/climDivErrMaps.png',dpi=150)
    plt.show()

def plotTwxVsPRISMTmaxTrend():
            
    dsTwxTmin = RasterDataset('/projects/daymet2/compare/topowx_files/trends/cce_topowx_tmin19482012trend.tif')
    dsPrismTmin = RasterDataset('/projects/daymet2/compare/prism_files/trends/cce_prism4km_tmin_trend1948-2012.tif')
    dsTwxTmax = RasterDataset('/projects/daymet2/compare/topowx_files/trends/cce_topowx_tmax19482012trend.tif')
    dsPrismTmax = RasterDataset('/projects/daymet2/compare/prism_files/trends/cce_prism4km_tmax_trend1948-2012.tif')
    
    twxTmin = dsTwxTmin.gdalDs.GetRasterBand(1).ReadAsArray()
    twxTmin = np.ma.masked_equal(twxTmin, dsTwxTmin.gdalDs.GetRasterBand(1).GetNoDataValue())*65
    twxTmax = dsTwxTmax.gdalDs.GetRasterBand(1).ReadAsArray()
    twxTmax = np.ma.masked_equal(twxTmax, dsTwxTmax.gdalDs.GetRasterBand(1).GetNoDataValue())*65
    
    prismTmin = dsPrismTmin.gdalDs.GetRasterBand(1).ReadAsArray()
    prismTmin = np.ma.masked_equal(prismTmin, dsPrismTmin.gdalDs.GetRasterBand(1).GetNoDataValue())*65
    prismTmax = dsPrismTmax.gdalDs.GetRasterBand(1).ReadAsArray()
    prismTmax = np.ma.masked_equal(prismTmax, dsPrismTmax.gdalDs.GetRasterBand(1).GetNoDataValue())*65
    
#    print np.min(twxTmin),np.max(twxTmin)
#    print np.min(prismTmin),np.max(prismTmin)
    
    dsGrid = dsTwxTmin
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
    lon_0 = (llcrnrlon+urcrnrlon)/2.0
    lat_0 = (llcrnrlat+urcrnrlat)/2.0
    
    levelsRed = np.arange(0,6.5,.5)
    levelsBlue = np.arange(-2.5,0.5,.5)
    
    #levels = [-2,-1,0,1,2,3,4,5,6]
    clrs = brewer2mpl.get_map('RdYlBu', 'Diverging', 11, reverse=True)
    clrs = clrs.mpl_colors[2:]
    
    norm = Normalize(0,6)
    smReds = cm.ScalarMappable(norm, cm.Reds)
    midPtsReds = (levelsRed + np.diff(levelsRed, 1)[0]/2.0)[0:-1]
    
    norm = Normalize(-2.5,0)
    smBlues = cm.ScalarMappable(norm, cm.Blues_r)
    midPtsBlues = (levelsBlue + np.diff(levelsBlue, 1)[0]/2.0)[0:-1]
    
    clrsRed = [smReds.to_rgba(x) for x in midPtsReds]
    clrsBlu = [smBlues.to_rgba(x) for x in midPtsBlues]
    clrsBlu.extend(clrsRed)
    clrs = clrsBlu
    levels = np.arange(-2.5,6.5,.5)
        
    print "Mapping data...."
    
    m = Basemap(resolution='i',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
    #,cbar_size="3%")
    
#    m.ax = grid[1]
#    cf = m.contourf(x,y,elev,100,cmap=cm.gist_earth)
#    cbar = grid[1].cax.colorbar(cf)
#    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)

    #plt.subplot(212)
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)#,cbar_size="8%")
    
    m.ax = grid[0]
    cf = m.contourf(x,y,twxTmin,levels=levels,colors=clrs)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_label(r'$^\circ$C / 65 yrs')
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[0].set_title("TopoWx")
    grid[0].set_ylabel("Tmin")
    m.drawcountries()
    m.drawstates()
    
    m.ax = grid[1]
    cf = m.contourf(x,y,prismTmin,levels=levels,colors=clrs)
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[1].set_title("PRISM")
    m.drawcountries()
    m.drawstates()
    
    m.ax = grid[2]
    cf = m.contourf(x,y,twxTmax,levels=levels,colors=clrs)
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[2].set_ylabel("Tmax")
    m.drawcountries()
    m.drawstates()
    
    m.ax = grid[3]
    cf = m.contourf(x,y,prismTmax,levels=levels,colors=clrs)
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    m.drawcountries()
    m.drawstates()

    cf = plt.gcf() 
    cf.set_size_inches(8*1.5,6*1.5)
    plt.savefig('/projects/daymet2/docs/UsgsConfCall201309/tairTrends.png',dpi=150)
    plt.show()

def plotTwxVsPRISMTrend():
            
    dsTwxTmin = RasterDataset('/projects/daymet2/compare/topowx_files/trends/cce_topowx_tmin19482012trend.tif')
    dsPrismTmin = RasterDataset('/projects/daymet2/compare/prism_files/trends/cce_prism4km_tmin_trend1948-2012.tif')
    
    twxTmin = dsTwxTmin.gdalDs.GetRasterBand(1).ReadAsArray()
    twxTmin = np.ma.masked_equal(twxTmin, dsTwxTmin.gdalDs.GetRasterBand(1).GetNoDataValue())*65
    
    prismTmin = dsPrismTmin.gdalDs.GetRasterBand(1).ReadAsArray()
    prismTmin = np.ma.masked_equal(prismTmin, dsPrismTmin.gdalDs.GetRasterBand(1).GetNoDataValue())*65
    
    dsGrid = dsTwxTmin
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
    lon_0 = (llcrnrlon+urcrnrlon)/2.0
    lat_0 = (llcrnrlat+urcrnrlat)/2.0
    
    levelsRed = np.arange(0,6.5,.5)
    levelsBlue = np.arange(-2,0.5,.5)
    
    #levels = [-2,-1,0,1,2,3,4,5,6]
    clrs = brewer2mpl.get_map('RdYlBu', 'Diverging', 11, reverse=True)
    clrs = clrs.mpl_colors[3:]
    
    norm = Normalize(0,6)
    smReds = cm.ScalarMappable(norm, cm.Reds)
    midPtsReds = (levelsRed + np.diff(levelsRed, 1)[0]/2.0)[0:-1]
    
    norm = Normalize(-2,0)
    smBlues = cm.ScalarMappable(norm, cm.Blues_r)
    midPtsBlues = (levelsBlue + np.diff(levelsBlue, 1)[0]/2.0)[0:-1]
    
    clrsRed = [smReds.to_rgba(x) for x in midPtsReds]
    clrsBlu = [smBlues.to_rgba(x) for x in midPtsBlues]
    clrsBlu.extend(clrsRed)
    clrs = clrsBlu
    levels = np.arange(-2,6.5,.5)
        
    print "Mapping data...."
    
    m = Basemap(resolution='i',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
    #,cbar_size="3%")
    
#    m.ax = grid[1]
#    cf = m.contourf(x,y,elev,100,cmap=cm.gist_earth)
#    cbar = grid[1].cax.colorbar(cf)
#    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)

    #plt.subplot(212)
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05,cbar_size="8%")
    
    m.ax = grid[0]
    cf = m.contourf(x,y,twxTmin,levels=levels,colors=clrs)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_label(r'$^\circ$C / 65 yrs')
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[0].set_title("TopoWx")
    
    m.ax = grid[1]
    cf = m.contourf(x,y,prismTmin,levels=levels,colors=clrs)
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[1].set_title("PRISM")
    
    plt.savefig('/projects/daymet2/docs/UsgsConfCall201309/tminTrends.png')
    plt.show()

def plotTwxVsPrismDaymetNorms():
            
    dsTwxTmin = RasterDataset('/projects/daymet2/compare/topowx_files/normals/cce_topowx_tmin19812010norm.tif')
    dsPrismTmin = RasterDataset('/projects/daymet2/compare/prism_files/normals/cce_prism_tmin1981_2010norm.tif')
    dsDaymetTmin = RasterDataset('/projects/daymet2/compare/daymet_files/normals/cce_daymet_tmin19812010norm.tif')
    
    dsTwxTmax = RasterDataset('/projects/daymet2/compare/topowx_files/normals/cce_topowx_tmax19812010norm.tif')
    dsPrismTmax = RasterDataset('/projects/daymet2/compare/prism_files/normals/cce_prism_tmax1981_2010norm.tif')
    dsDaymetTmax = RasterDataset('/projects/daymet2/compare/daymet_files/normals/cce_daymet_tmax19812010norm.tif')
    
    twxTmin = dsTwxTmin.readAsArray()
    prismTmin = dsPrismTmin.readAsArray()/100
    daymetTmin = dsDaymetTmin.readAsArray()
    
    twxTmax = dsTwxTmax.readAsArray()
    prismTmax = dsPrismTmax.readAsArray()/100
    daymetTmax = dsDaymetTmax.readAsArray()
        
    dsGrid = dsTwxTmin
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
    lon_0 = (llcrnrlon+urcrnrlon)/2.0
    lat_0 = (llcrnrlat+urcrnrlat)/2.0
        
    print "Mapping data...."
    m = Basemap(resolution='i',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
    
    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 8, reverse=True)
    clrs = clrs.mpl_colors
    clrs[3] = "grey"
    clrs[4] = "grey"
    minTmin = np.min([np.min(twxTmin-prismTmin),np.min(twxTmin-daymetTmin)])
    maxTmin = np.max([np.max(twxTmin-prismTmin),np.max(twxTmin-daymetTmin)])
    print minTmin,maxTmin
    levels = np.arange(-4,5)
    
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),cbar_mode="single",cbar_location="right",axes_pad=0.5,cbar_pad=0.2)
    
    m.ax = grid[0]
    cf = m.contourf(x,y,twxTmin-prismTmin,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_label(r'$^\circ$C')
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[0].set_title("TopoWx Minus PRISM")
    grid[0].set_ylabel("Tmin")
    
    m.ax = grid[1]
    cf = m.contourf(x,y,twxTmin-daymetTmin,colors=clrs,levels=levels,extend='both')
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[1].set_title("TopoWx Minus Daymet")
    
    m.ax = grid[2]
    cf = m.contourf(x,y,twxTmax-prismTmax,colors=clrs,levels=levels,extend='both')
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[2].set_title("TopoWx Minus PRISM")
    grid[2].set_ylabel("Tmax")
    
    m.ax = grid[3]
    cf = m.contourf(x,y,twxTmax-daymetTmax,colors=clrs,levels=levels,extend='both')
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[3].set_title("TopoWx Minus Daymet")
    
    plt.savefig('/projects/daymet2/docs/UsgsConfCall201309/normsCompare.png',dpi=150)
    plt.show()

if __name__ == '__main__':
    #plotTwxVsPrismDaymetNorms()
    plotTwxVsPRISMTmaxTrend()
    #plotTwxVsPRISMTrend()
    #plotCceInterp()
    #plotInterpErrorMaps()
    #plotPredictors()
    #plotStns()
    #plotConusTrends()