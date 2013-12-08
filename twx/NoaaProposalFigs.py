'''
Created on Nov 11, 2013

@author: jared.oyler
'''
from osgeo import gdal
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from PaperResultsMetrics import getCceHillshade,drawCceBnds, CMAP_ESRI_PRCP
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import Normalize
from matplotlib import cm
from utils.input_raster import RasterDataset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Polygon
import brewer2mpl
import shapefile
from matplotlib.collections import LineCollection
from pyhdf.SD import SD, SDC


def plotDsiSeries():
    '/MODIS/NTSG_Products/DSI/Annual/DSI_05deg_2000.hdf'
    
    pathDsi = '/MODIS/NTSG_Products/DSI/Annual/DSI_05deg_'
    
    dsGrd = RasterDataset('/projects/daymet2/docs/noaa_proposal/DSI_05deg_20111.tif')
    dsMask = RasterDataset('/projects/daymet2/docs/noaa_proposal/montana_mask.tif')
    dsLc = RasterDataset('/projects/daymet2/docs/noaa_proposal/UMD_Landcover.tif')
    
    mask = dsMask.readAsArray().data == 1
    lat,lon = dsGrd.getCoordGrid1d()
    maskNoData = ~mask
    
    lc = dsLc.readAsArray()
    
    nonzero_rows,nonzero_cols = np.nonzero(mask)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    nonzero_rows = np.arange(nonzero_rows[0],nonzero_rows[-1]+1)
    nonzero_cols = np.arange(nonzero_cols[0],nonzero_cols[-1]+1)
    lat = lat[nonzero_rows]
    lon = lon[nonzero_cols]
    
    yrs = np.arange(2000,2012)
    
    sd = SD(pathDsi+str(2000)+".hdf",SDC.READ)
    fval = sd.select('ET_1km').attributes()['_FillValue']
    sd.end()
    
#    dsiAll = np.zeros((yrs.size,nonzero_rows.size,nonzero_cols.size))
#    

#    
#    for yr,x in zip(yrs,np.arange(yrs.size)):
#        
#        sd = SD(pathDsi+str(yr)+".hdf",SDC.READ)
#        dsi = sd.select('ET_1km')[:]
#        dsi[maskNoData] = fval
#        
#        dsiCrp = dsi[nonzero_rows,:]
#        dsiCrp = dsiCrp[:,nonzero_cols]
#        dsiAll[x,:,:] = dsiCrp
#        print yr
        
    #np.save('/projects/daymet2/docs/noaa_proposal/dsiMontana.npy', dsiAll)    
    
    dsiAll = np.load('/projects/daymet2/docs/noaa_proposal/dsiMontana.npy')    
    dsiAll = np.ma.masked_equal(dsiAll, fval)
    
    
    
    lc[maskNoData] = np.ma.masked
    lcCrp = lc[nonzero_rows,:]
    lcCrp = lcCrp[:,nonzero_cols]
    
    #lcCrp[lcCrp!=10] = np.ma.masked 
    
    forestDsi = np.mean(dsiAll[:,lcCrp==1],axis=1)
    grassDsi =  np.mean(dsiAll[:,lcCrp==10],axis=1)
    allDsi =  np.mean(dsiAll[:,lcCrp!=0],axis=1)
    
    plt.subplot(211)
    plt.plot(allDsi,color="#377EB8",lw=1.5)
    plt.plot(forestDsi,color="#4DAF4A",lw=1.5)
    plt.plot(grassDsi,color="#FF7F00",lw=1.5)
    plt.legend(["All","Evergreen Needleleaf Forest","Grassland"],fontsize=10,loc='upper center',bbox_to_anchor=(0.5, 1.15),shadow=True)
    xmin,xmax = plt.xlim()
    xmax = xmax -1
    plt.hlines(0, xmin, xmax)
    plt.xlim(xmin,xmax)
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.xticks(np.arange(yrs.size),[str(yr) for yr in yrs])
    plt.xlabel("Year")
    plt.ylabel("DSI")
    
    
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf*4)

    print "Mapping data...."
    m = Basemap(resolution='h',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=-111,lat_0=0)
    xMap, yMap = m(*np.meshgrid(lon,lat))
    
    elev = getCceHillshade(m, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon)
  
    cf = plt.gcf()
    grid = ImageGrid(cf,212,nrows_ncols=(1,2),cbar_mode="each",cbar_location="right",axes_pad=0.3,cbar_pad=0.05)#,
    lcCrp[np.logical_and(lcCrp!=10,lcCrp!=1)] = np.ma.masked
    
    m.ax = grid[0]
    clrs = ['#4DAF4A','#FF7F00']
    alpha = .5
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,lcCrp,colors=clrs,levels=[0,5,11],alpha=alpha,antialiased=True)
    cf = m.contourf(xMap,yMap,lcCrp,colors=clrs,levels=[0,5,11],alpha=alpha,antialiased=True)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_ticks([ 2.5,  8. ])
    cbar.set_ticklabels(["EF","GL"])
    #cbar.set_ticklabels(["Normal/\nWet","D0","D1","D2","D3","D4"][::-1])
    cbar.set_alpha(1)
    cbar.draw_all()
    
    m.drawcountries()
    m.drawstates()
    plt.sca(grid[0])
    plt.title("Land Cover",fontsize=12)
    
    m.ax = grid[1]
    
    norm = Normalize(-1.5,1.5)
    
    clrs = brewer2mpl.get_map('RdYlGn', 'Diverging', 11, reverse=False)
    cmap = clrs.get_mpl_colormap()
    clrs = clrs.mpl_colors
    levels = np.linspace(-1.5,1.5,11)
    alpha = .5
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,dsiAll[-1,:,:],levels=levels,cmap=cmap,alpha=alpha,antialiased=True,extend='both')
    cf = m.contourf(xMap,yMap,dsiAll[-1,:,:],levels=levels,cmap=cmap,alpha=alpha,antialiased=True,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[1])
    cbar.set_alpha(1)
    cbar.draw_all()
    cbar.set_ticks(levels)
    m.drawcountries()
    m.drawstates()
    
    plt.sca(grid[1])
    plt.title("2011 Annual DSI",fontsize=12)
    cbar.ax.set_ylabel("DSI",fontsize=9)
    
    plt.savefig('/projects/daymet2/docs/noaa_proposal/AnnDsiLandCover.png',dpi=300) 
    plt.show()
    


def plotUdmVsDSI():
    
    dsDsi = RasterDataset('/projects/daymet2/docs/noaa_proposal/DSI_2007225.tif')
    dsi = dsDsi.readAsArray()
    
    dsUdm = RasterDataset('/projects/daymet2/docs/noaa_proposal/USDM_raster3.tif')
    udm = dsUdm.readAsArray()
    
    maskSetWet = np.logical_and(udm.mask,~dsi.mask)
        
    udm = udm.data.astype(np.int)
    udm[maskSetWet] = -1
    udm = np.ma.masked_array(udm,dsi.mask)
    
#    plt.imshow(udm)
#    plt.colorbar()
#    plt.show()
    
    #levels = [-0.3,-0.7,-9,-1.2,-1.5]
    levels = np.array([np.min(dsi),-1.5,-1.2,-.9,-0.7,-0.3,np.max(dsi)])
    midPts = (levels[0:-1] + np.diff(levels, 1)/2.0)
    
    clrs = brewer2mpl.get_map('YlOrBr', 'Sequential', 6, reverse=True)
    cmap = clrs.get_mpl_colormap()
    clrs = clrs.mpl_colors
    clrs[-1] = "#31A354"
    
    dsGrid = dsDsi
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf*4)

    print "Mapping data...."
    m = Basemap(resolution='h',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=-111,lat_0=0)
    xMap, yMap = m(*np.meshgrid(lon,lat))
    
  
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)#,cbar_size="2%")
    
    #elev = getCceHillshade(m, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon)
    
    alpha = 0.5
   
    m.ax = grid[0]
    #m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,dsi,colors=clrs,levels=levels,alpha=alpha,antialiased=True)
    
    
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_ticks(midPts)
    cbar.set_ticklabels(["Normal/\nWet","D0","D1","D2","D3","D4"][::-1])
    cbar.set_alpha(1)
    cbar.draw_all()
    m.drawcountries()
    m.drawstates()
    #cbar.set_label(r'Tmin ($^\circ$C)')
    grid[0].set_title("DSI")
    
    #axin = inset_axes(grid[0],width="25%",height="25%",loc=3)
    
#    m_inset = Basemap(resolution='h',projection='aea', llcrnrlat=39,urcrnrlat=52,llcrnrlon=-118,urcrnrlon=-96,
#                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5,ax=axin,area_thresh=100000)
#    m_inset.drawcountries()
#    m_inset.drawcoastlines()
#    m_inset.drawstates()
#    s = m_inset.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,color='k')               
#    s[4].set_facecolors('red')
#    for x in np.arange(lineCollect.size):
#        
#        lines = lineCollect[x]
#        lines.set_facecolors(sm.to_rgba(errSet[i][x]))
#        lines.set_edgecolors('#8C8C8C')
#        lines.set_linewidth(0.5)
#        gridCell.add_collection(lines)
#    
#    for xy in m.CCE_CMP_US_Only:
#        poly = Polygon(xy, facecolor='red')
#        m_inset.ax.add_patch(poly)
    
#    bx, by = omap(bmap.boundarylons, bmap.boundarylats)
#    xy = list(zip(bx,by))
#    mapboundary = Polygon(xy,edgecolor='red',linewidth=2,fill=False)
#    omap.ax.add_patch(mapboundary)
    
#    grid[0].set_ylabel("Tmin")
#    grid[0].set_title("TopoWx",fontsize=12)
#    cbar.set_label(r'$^\circ$C / 65 yrs')
    
    m.ax = grid[1]
    
    #m.imshow(elev,cmap=cm.gray)
    #s = m.readshapefile('/projects/daymet2/docs/noaa_proposal/USDM_20070814', 'USDM_20070814', drawbounds=True)
    
#    r = shapefile.Reader(r"/projects/daymet2/docs/noaa_proposal/USDM_20070814")
#    shapes = r.shapes()
#    records = r.records()
#    
#    clrsShp = clrs[0:-1][::-1]
#    
#    for record, shape in zip(records,shapes):
#        lons,lats = zip(*shape.points)
#        data = np.array(m(lons, lats)).T
#     
#        if len(shape.parts) == 1:
#            segs = [data,]
#        else:
#            segs = []
#            for i in range(1,len(shape.parts)):
#                index = shape.parts[i-1]
#                index2 = shape.parts[i]
#                segs.append(data[index:index2])
#            segs.append(data[index2:])
#     
#        lines = LineCollection(segs,antialiaseds=(1,))
#        lines.set_facecolors(clrsShp[record[1]])
#        lines.set_edgecolors('k')
#        lines.set_linewidth(0.1)
#        grid[1].add_collection(lines)
    
    
    #s[4].set_facecolors('red')
    
    #grid[1].set_title("TopoWx")#,fontsize=12)
    
    levels = np.arange(-2,5)    
    clrs = brewer2mpl.get_map('YlOrBr', 'Sequential', 6, reverse=False)
    cmap = clrs.get_mpl_colormap()
    clrs = clrs.mpl_colors
    clrs[0] = "#31A354"
    
    print np.unique(udm)
    print levels
    
    cf = m.contourf(xMap,yMap,udm,colors=clrs,levels=levels,alpha=alpha,antialiased=True)
    m.drawcountries()
    m.drawstates()
    grid[1].set_title("USDM")
#    m.ax = grid[2]
#    m.imshow(elev,cmap=cm.gray)
#    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
#    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
#    grid[2].set_ylabel("Tmax")
#    drawCceBnds(m)
#    
#    m.ax = grid[3]
#    m.imshow(elev,cmap=cm.gray)
#    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
#    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
#    drawCceBnds(m)

    #fig =plt.gcf()
    #fig.set_size_inches(8*2,6*3)
    #fig.set_size_inches(6.5,4.88)

    plt.savefig('/projects/daymet2/docs/noaa_proposal/DsiVsUSDM_20070815.png',dpi=300) 
    #plt.savefig()
    plt.show()


def plotMerraVsTwx():
    
    dsTwxTmin = gdal.Open('/projects/daymet2/cce_case_study/topowx_files/annual/cce_topowx_tmin19482012ann.tif')
    dsMerraTmin = Dataset('/projects/daymet2/mod16_aguposter/crown_merra.nc')
    
    yrs = np.arange(1948,2012)
    yrMask = np.logical_and(yrs>=2000,yrs<=2009)
    
    twxTmin = dsTwxTmin.ReadAsArray()
    ndata = dsTwxTmin.GetRasterBand(1).GetNoDataValue()
    twxTmin = np.ma.masked_equal(twxTmin, ndata)
    
    twxTmin = np.ma.mean(twxTmin[yrMask,:,:],axis=0)    
    merraTmin = np.ma.mean(dsMerraTmin.variables['tmin'][:],axis=0)
    
    maskTwx = ~twxTmin.mask
    maskMerra = ~merraTmin.mask
    maskFnl = ~np.logical_and(maskTwx,maskMerra)
    
    twxTmin = np.ma.masked_array(twxTmin.data,maskFnl)
    merraTmin = np.ma.masked_array(merraTmin.data,maskFnl)
    
    
    minVal = np.min([np.min(twxTmin),np.min(merraTmin)])
    maxVal = np.max([np.max(twxTmin),np.max(merraTmin)])
    print minVal,maxVal
    
    dsTwxTmin = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/annual/cce_topowx_tmin19482012ann.tif')
    dsGrid = dsTwxTmin
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)

    print "Mapping data...."
    m = Basemap(resolution='h',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=-111,lat_0=0)
    xMap, yMap = m(*np.meshgrid(lon,lat))
    
#    levelsRed = np.arange(0,6.5,.5)
#    levelsBlue = np.arange(-2,0.5,.5)
##    clrs = brewer2mpl.get_map('RdYlBu', 'Diverging', 11, reverse=True)
##    clrs = clrs.mpl_colors[3:]
#    
#    norm = Normalize(0,6)
#    smReds = cm.ScalarMappable(norm, cm.Reds)
#    midPtsReds = (levelsRed + np.diff(levelsRed, 1)[0]/2.0)[0:-1]
#    
#    norm = Normalize(-2,0)
#    smBlues = cm.ScalarMappable(norm, cm.Blues_r)
#    midPtsBlues = (levelsBlue + np.diff(levelsBlue, 1)[0]/2.0)[0:-1]
#    
#    clrsRed = [smReds.to_rgba(x) for x in midPtsReds]
#    clrsBlu = [smBlues.to_rgba(x) for x in midPtsBlues]
#    clrsBlu.extend(clrsRed)
#    clrs = clrsBlu
#    levels = np.arange(-2,6.5,.5)

    acmap = CMAP_ESRI_PRCP
    norm = Normalize(minVal,maxVal)
    levels = np.linspace(minVal, maxVal, 100)
    
    elev = getCceHillshade(m, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon)
  
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)#,cbar_size="2%")
   
    alpha = .5
   
    m.ax = grid[0]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,merraTmin,cmap=acmap,norm=norm,levels=levels,alpha=alpha,antialiased=True)
    cf = m.contourf(xMap,yMap,merraTmin,cmap=acmap,norm=norm,levels=levels,alpha=alpha,antialiased=True)

    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_ticks(np.arange(-5.0,4.0))
    cbar.set_alpha(1)
    cbar.draw_all()
    cbar.set_label(r'Tmin ($^\circ$C)')
    grid[0].set_title("MERRA")
    
    axin = inset_axes(grid[0],width="25%",height="25%",loc=3)
    
    m_inset = Basemap(resolution='h',projection='aea', llcrnrlat=39,urcrnrlat=52,llcrnrlon=-118,urcrnrlon=-96,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5,ax=axin,area_thresh=100000)
    m_inset.drawcountries()
    m_inset.drawcoastlines()
    m_inset.drawstates()
    s = m_inset.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,color='k')               
    s[4].set_facecolors('red')
#    for x in np.arange(lineCollect.size):
#        
#        lines = lineCollect[x]
#        lines.set_facecolors(sm.to_rgba(errSet[i][x]))
#        lines.set_edgecolors('#8C8C8C')
#        lines.set_linewidth(0.5)
#        gridCell.add_collection(lines)
#    
#    for xy in m.CCE_CMP_US_Only:
#        poly = Polygon(xy, facecolor='red')
#        m_inset.ax.add_patch(poly)
    
#    bx, by = omap(bmap.boundarylons, bmap.boundarylats)
#    xy = list(zip(bx,by))
#    mapboundary = Polygon(xy,edgecolor='red',linewidth=2,fill=False)
#    omap.ax.add_patch(mapboundary)
    
#    grid[0].set_ylabel("Tmin")
#    grid[0].set_title("TopoWx",fontsize=12)
#    cbar.set_label(r'$^\circ$C / 65 yrs')
    drawCceBnds(m)
    
    m.ax = grid[1]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,twxTmin,cmap=acmap,norm=norm,levels=levels,alpha=alpha,antialiased=True)
    cf = m.contourf(xMap,yMap,twxTmin,cmap=acmap,norm=norm,levels=levels,alpha=alpha,antialiased=True)
    grid[1].set_title("TopoWx")#,fontsize=12)
    drawCceBnds(m)
    
#    m.ax = grid[2]
#    m.imshow(elev,cmap=cm.gray)
#    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
#    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
#    grid[2].set_ylabel("Tmax")
#    drawCceBnds(m)
#    
#    m.ax = grid[3]
#    m.imshow(elev,cmap=cm.gray)
#    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
#    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
#    drawCceBnds(m)

    fig =plt.gcf()
    #fig.set_size_inches(8*2,6*3)
    fig.set_size_inches(6.5,4.88)

    plt.savefig('/projects/daymet2/docs/noaa_proposal/MerraVsTopoWx.png',dpi=300) 
    #plt.savefig()
    plt.show()

if __name__ == '__main__':
    #plotMerraVsTwx()
    #plotUdmVsDSI()
    plotDsiSeries()