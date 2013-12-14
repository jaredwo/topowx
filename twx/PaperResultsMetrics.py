'''
Created on Apr 5, 2013

@author: jared.oyler
'''
import infill.obs_por as op
import numpy as np
from twx.db.station_data import station_data_ncdb, LON, LAT,CLIMDIV, station_data_infill,VARIO_NUG,NEON,BAD,YEAR,\
    STN_ID,DTYPE_STN_BASIC, MASK, DATE, YMD,DTYPE_STN_MEAN_LST_TDI,BAD, MEAN_OBS,get_norm_varname,\
    ELEV,get_lst_varname, STN_NAME, LST,TDI, MONTH, DTYPE_INTERP, DTYPE_OPTIM,\
    get_optim_varname
import twx.db.station_data as stnData
import twx.utils.util_geo as utlg
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from netCDF4 import Dataset
from twx.utils.input_raster import input_raster,RasterDataset, OutsideExtent
import shapefile
import pickle
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize,BoundaryNorm
from matplotlib import cm
from copy import copy
from twx.utils.status_check import status_check
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
import matplotlib
from DatasetCompare import MultiDaymetTileRaster, PrismTileRaster,\
    MultiTwxTileRaster
from qa.qa_temp import imposs_value_mask


CONUS_US_BOUNDS = (-126.0,-64.0,22.0,53.0)


cdict = {'red':[(0.0,   194/255.0,      194/255.0),
                (0.2,   237.0/255.0,    237.0/255.0),
                (0.4,   255.0/255.0,    255.0/255.0),
                (0.6,   0.0/255.0,      0.0/255.0),
                (0.8,   32.0/255.0,     32.0/255.0),
                (1.0,   11.0/255.0,     11.0/255.0)],
     'green': [ (0.0,   80/255.0,       80/255.0),
                (0.2,   161/255.0,      161/255.0),
                (0.4,   255.0/255.0,    255.0/255.0),
                (0.6,   219.0/255.0,    219.0/255.0),
                (0.8,   153.0/255.0,    153.0/255.0),
                (1.0,   44.0/255.0,     44.0/255.0)],
     'blue': [  (0.0,   60/255.0,       60/255.0),
                (0.2,   19/255.0,       19/255.0),
                (0.4,   0.0/255.0,      0.0/255.0),
                (0.6,   0.0/255.0,      0.0/255.0),
                (0.8,   143.0/255.0,    143.0/255.0),
                (1.0,   122.0/255.0,    122.0/255.0)]}
    
def reverseCmapDict(cdict):
    cdict_r = {}
    
    r = np.array(cdict['red'])
    r[:,1] = r[:,1][::-1]
    r[:,2] = r[:,2][::-1]
    
    g = np.array(cdict['green'])
    g[:,1] = g[:,1][::-1]
    g[:,2] = g[:,2][::-1]
    
    b = np.array(cdict['blue'])
    b[:,1] = b[:,1][::-1]
    b[:,2] = b[:,2][::-1]
    
    cdict_r['red'] = [tuple(x) for x in tuple(r)]
    cdict_r['green'] = [tuple(x) for x in tuple(g)]
    cdict_r['blue'] = [tuple(x) for x in tuple(b)]
    
    return cdict_r

CMAP_ESRI_PRCP = matplotlib.colors.LinearSegmentedColormap('prcp',reverseCmapDict(cdict),256)

TRANSECT_CCE_LON = (-114.6,-112.5)
TRANSECT_CCE_LAT = (47.78,47.78)
   

def potential_stn_cnt(nYrsOfData):
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    stns = db.stns
    stn_ids = db.stn_ids
    
    path_por = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    por = op.load_por_csv(path_por)
    
    #Need a least a few Tair observations per month
    mask_por_tmin,mask_por_tmax,mask_por_prcp = op.build_valid_por_masks(por,nYrsOfData)
    
    maskConus = np.logical_and(np.logical_and(stns[LON] >= CONUS_US_BOUNDS[0], stns[LON] <= CONUS_US_BOUNDS[1]),
                              np.logical_and(stns[LAT] >= CONUS_US_BOUNDS[2], stns[LAT] <= CONUS_US_BOUNDS[3])) 
    
    maskFnl = np.logical_and(np.logical_or(mask_por_tmin,mask_por_tmax),maskConus)
    
    print 'GHCN:',np.sum(np.logical_and(maskFnl,np.char.startswith(stn_ids, 'GHCN')))
    print 'RAWS:',np.sum(np.logical_and(maskFnl,np.char.startswith(stn_ids, 'RAWS')))
    print 'SNOTEL:',np.sum(np.logical_and(maskFnl,np.char.startswith(stn_ids, 'SNOTEL')))
    print 'OVERALL',np.sum(maskFnl)

def get_nValid_obs(stn_prefix,tair_var):
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc')
    stns = db.stns
    db.ds.close()
    db = None
    
    stnIdx = np.nonzero(np.char.startswith(stns[STN_ID], stn_prefix))[0]
    
    ds = Dataset('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc')
    tair = ds.variables[tair_var][:,stnIdx]
    return np.sum(np.isfinite(tair))

def qa_flag_cnts(stn_prefix,tair_var):
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    stns = db.stns
    stn_ids = db.stn_ids
    db.ds.close()
    db = None
    
    ds = Dataset('/projects/daymet2/station_data/all/all_1948_2012.nc')
    tairVar = ds.variables[tair_var]
    qflagVar = ds.variables["".join(["qflag_",tair_var])]
    qflagVar.set_auto_maskandscale(False)
    tairVar.set_auto_maskandscale(False)
    
    path_por = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    por = op.load_por_csv(path_por)
    
    maskConus = np.logical_and(np.logical_and(stns[LON] >= CONUS_US_BOUNDS[0], stns[LON] <= CONUS_US_BOUNDS[1]),
                              np.logical_and(stns[LAT] >= CONUS_US_BOUNDS[2], stns[LAT] <= CONUS_US_BOUNDS[3])) 
    
    #Need a least a few Tair observations per month
    mask_por_tmin,mask_por_tmax = op.build_valid_por_masks(por,.01)[0:2]
    masks_tair = {'tmin':mask_por_tmin,'tmax':mask_por_tmax}
    
    maskPrefix = np.char.startswith(stn_ids, stn_prefix)
    
    maskFnl = np.logical_and(np.logical_and(maskConus,maskPrefix),masks_tair[tair_var])
    idxFnl = np.nonzero(maskFnl)[0]
    
    valsTair = tairVar[:,idxFnl]
    ttl_obs = np.sum(valsTair != -9999)
    valsTair = None
    
    flgs = qflagVar[:,idxFnl]
    ttl_flg = np.sum(flgs != '')
    flgs = None
    
    print "#####################################"
    print " ".join([stn_prefix,tair_var])
    print "#####################################"
    print "Total flagged: ",ttl_flg
    print "Total observations: ",ttl_obs
    print "% flagged: ",(ttl_flg/float(ttl_obs))*100.
    print ""

def homogChgPtSntl():
    adjTmin = parsePhaAdj('/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/output/PhaAdjTmin.log')
    adjTmin['adj'] = -adjTmin['adj']
    
    #maskSntlTmin = np.logical_and(np.char.startswith(adjTmin['stn_id'],'SNT'),adjTmin['adj'] != 0)
    maskSntlTmin = np.char.startswith(adjTmin['stn_id'],'SNT')
    adjSntl = adjTmin[maskSntlTmin]
    
    dates = np.array([utld.ymdL_to_date(ymd) for ymd in adjSntl['ymdStr']])
    yrs = np.array([adate.year for adate in dates])
    
    yrsCnts = np.arange(1990,2013)
    cnts = np.zeros(yrsCnts.size)
    
    for x in np.arange(yrsCnts.size):
        cnts[x] = np.mean(adjSntl['adj'][yrs==yrsCnts[x]])
    
    plt.plot(cnts)
    plt.show()

def homogChgPtStats():
#    nTminSntl = get_nValid_obs('SNOTEL','tmin')
#    nTmaxSntl = get_nValid_obs('SNOTEL','tmax')
#    print nTminSntl,nTmaxSntl
    nTminSntl = 5572690
    nTmaxSntl = 5564052
    
#    nTminRaws = get_nValid_obs('RAWS','tmin')
#    nTmaxRaws = get_nValid_obs('RAWS','tmax')
#    print nTminRaws,nTmaxRaws
    nTminRaws = 7230359
    nTmaxRaws = 7237182
    
#    nTminGhcn = get_nValid_obs('GHCN','tmin')
#    nTmaxGhcn = get_nValid_obs('GHCN','tmax')
#    print nTminGhcn,nTmaxGhcn
    nTminGhcn = 150445193
    nTmaxGhcn = 150920223
    
    adjTmin = parsePhaAdj('/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/output/PhaAdjTmin.log')
    adjTmin['adj'] = -adjTmin['adj']
    
    adjTmax = parsePhaAdj('/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/output/PhaAdjTmax.log')
    adjTmax['adj'] = -adjTmax['adj']
    
    maskSntlTmin = np.logical_and(np.char.startswith(adjTmin['stn_id'],'SNT'),adjTmin['adj'] != 0)
    maskRawTmin = np.logical_and(np.char.startswith(adjTmin['stn_id'],'RAW'),adjTmin['adj'] != 0)
    maskGhcnTmin = np.logical_and(np.logical_and(~np.char.startswith(adjTmin['stn_id'],'SNT'),~np.char.startswith(adjTmin['stn_id'],'RAW')),adjTmin['adj'] != 0)

    maskSntlTmax = np.logical_and(np.char.startswith(adjTmax['stn_id'],'SNT'),adjTmax['adj'] != 0)
    maskRawTmax = np.logical_and(np.char.startswith(adjTmax['stn_id'],'RAW'),adjTmax['adj'] != 0)
    maskGhcnTmax = np.logical_and(np.logical_and(~np.char.startswith(adjTmax['stn_id'],'SNT'),~np.char.startswith(adjTmax['stn_id'],'RAW')),adjTmax['adj'] != 0)
    
    print "SNOTEL TMIN",np.sum(maskSntlTmin),np.sum(maskSntlTmin)/(nTminSntl/365.25),1.0/(np.sum(maskSntlTmin)/(nTminSntl/365.25))
    print "SNOTEL TMAX",np.sum(maskSntlTmax),np.sum(maskSntlTmax)/(nTmaxSntl/365.25),1.0/(np.sum(maskSntlTmax)/(nTmaxSntl/365.25))
    
    print "RAWS TMIN",np.sum(maskRawTmin),np.sum(maskRawTmin)/(nTminRaws/365.25),1.0/(np.sum(maskRawTmin)/(nTminRaws/365.25))
    print "RAWS TMAX",np.sum(maskRawTmax),np.sum(maskRawTmax)/(nTmaxRaws/365.25),1.0/(np.sum(maskRawTmax)/(nTmaxRaws/365.25))
    
    print "GHCN TMIN",np.sum(maskGhcnTmin),np.sum(maskGhcnTmin)/(nTminGhcn/365.25),1.0/(np.sum(maskGhcnTmin)/(nTminGhcn/365.25))
    print "GHCN TMAX",np.sum(maskGhcnTmax),np.sum(maskGhcnTmax)/(nTmaxGhcn/365.25),1.0/(np.sum(maskGhcnTmax)/(nTmaxGhcn/365.25))
    
    print "OVERALL CNTS TMIN:",np.sum(adjTmin['adj'] != 0)
    print "OVERALL CNTS TMAX:",np.sum(adjTmax['adj'] != 0)
    
    nTminAll = nTminSntl + nTminGhcn + nTminRaws
    nTmaxAll = nTmaxSntl + nTmaxGhcn + nTmaxRaws
    maskTminAdj = adjTmin['adj'] != 0
    maskTmaxAdj = adjTmax['adj'] != 0
    
    print "OVERALL TMIN:",np.sum(maskTminAdj),np.sum(maskTminAdj)/(nTminAll/365.25),1.0/(np.sum(maskTminAdj)/(nTminAll/365.25))
    print "OVERALL TMAX:",np.sum(maskTmaxAdj),np.sum(maskTmaxAdj)/(nTmaxAll/365.25),1.0/(np.sum(maskTmaxAdj)/(nTmaxAll/365.25))
    
def homogAdjBoxplot():
    adjTmin = parsePhaAdj('/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/output/PhaAdjTmin.log')
    adjTmin['adj'] = -adjTmin['adj']
    
    adjTmax = parsePhaAdj('/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/output/PhaAdjTmax.log')
    adjTmax['adj'] = -adjTmax['adj']
    
    maskSntl = np.logical_and(np.char.startswith(adjTmin['stn_id'],'SNT'),adjTmin['adj'] != 0)
    maskRaw = np.logical_and(np.char.startswith(adjTmin['stn_id'],'RAW'),adjTmin['adj'] != 0)
    maskGhcn = np.logical_and(np.logical_and(~np.char.startswith(adjTmin['stn_id'],'SNT'),~np.char.startswith(adjTmin['stn_id'],'RAW')),adjTmin['adj'] != 0)

    maskSntlTmax = np.logical_and(np.char.startswith(adjTmax['stn_id'],'SNT'),adjTmax['adj'] != 0)
    maskRawTmax = np.logical_and(np.char.startswith(adjTmax['stn_id'],'RAW'),adjTmax['adj'] != 0)
    maskGhcnTmax = np.logical_and(np.logical_and(~np.char.startswith(adjTmax['stn_id'],'SNT'),~np.char.startswith(adjTmax['stn_id'],'RAW')),adjTmax['adj'] != 0)
    
    data = [adjTmin['adj'][maskGhcn],adjTmin['adj'][maskSntl],adjTmin['adj'][maskRaw],
            adjTmax['adj'][maskGhcnTmax],adjTmax['adj'][maskSntlTmax],adjTmax['adj'][maskRawTmax]]
    data = data[::-1]
    
    means = [np.mean(x) for x in data]
    
    bp = plt.boxplot(data,vert=False,sym="",patch_artist=True)
    print bp
    plt.setp(bp['boxes'], color='grey')
    plt.setp(bp['whiskers'], color='black',linestyle="-")
    plt.setp(bp['fliers'], color='black')
    plt.setp(bp['medians'], color='black')
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    s = plt.scatter(means,[1,2,3,4,5,6],color='black')
    s.set_zorder(20)
    
    top = 4
    
    nBoxes = len(data)
    pos = np.arange(nBoxes)+1
    upperLabels = [str(np.round(s, 2))+' $^\circ$C' for s in means]
    weights = ['bold', 'semibold']
    for tick,label in zip(range(nBoxes),ax.get_yticklabels()):
        k = tick % 2
        ax.text(top-(top*0.1),pos[tick], upperLabels[tick],
            horizontalalignment='center',verticalalignment='center', size='x-small', weight='bold')
    
    plt.yticks([1,2,3,4,5,6],['RAWS','SNOTEL','GHCN-D','RAWS','SNOTEL','GHCN-D'])
    xmin,xmax = plt.xlim()
    plt.hlines(3.5, xmin, xmax)
    plt.xlim(xmin,xmax)
    plt.text(-3.9,6.25,'Tmin',weight='bold')
    plt.text(-3.9,3.25,'Tmax',weight='bold')
    plt.xlabel("Adjustment ($^\circ$C)")
    plt.savefig('/projects/daymet2/docs/final_writeup/homogAdjBoxplots.png',dpi=300)
    plt.show()

def qa_loc_cnts(path_locqa):
    
    afile = open(path_locqa)
    afile.readline()
    
    cnt = 0
    dists = []
    n0dist = 0
    
    for line in afile.readlines():
        
        #STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
        vals = line.split(",")
        
        if vals[7] != '' and vals[8] != '':
            
            dist = utlg.grt_circle_dist(float(vals[2]),float(vals[3]),float(vals[7]),float(vals[8]))
            
            if not dist > 1000:
            
                dists.append(dist)
                cnt+=1
            else:
                print line
            
            if dist == 0:
                n0dist+=1
                print line
    
    dists = np.array(dists)
    print np.percentile(dists[dists != 0], 75)
    plt.boxplot(dists[dists != 0])
    plt.show()
    
    print "Total stations locations adjusted: "+str(cnt)
    print "Total stations elevation adjusted only: "+str(n0dist)
    print "Avg. dist (km) moved: "+str(np.mean(dists[dists != 0]))

def plotClimDivStnDensity():
    
    #Load station counts per climate division into dictionary
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc', 'tmax')
    stns = stnda.stns
    #stns = stnda.stns[np.char.startswith(stnda.stns[STN_ID], 'GHCN')]
    climDivs = np.unique(stnda.stns[CLIMDIV][np.isfinite(stnda.stns[CLIMDIV])])#[2:]
    stnCnts = []
    for div in climDivs:
        stnCnts.append(np.sum(stns[CLIMDIV]==div))
    stnCnts = np.array(stnCnts)
    
    #Load the climate division areas
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    areas = np.array([float(aRec[-1]) * 1.0e-6 for aRec in recs]) #in square km
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    areas = areas[sidx]
    climDivsShp = climDivsShp[sidx]
    
    #Calculate density
    print np.min(stnCnts)
    stnDensity = (stnCnts/areas)*1000 #per 100 km2
    #stnDensity = stnCnts
    #Create a normalized colormap
    norm = Normalize(0,5)
    #norm = Normalize(np.min(stnCnts), np.max(stnCnts))
    cmap = cm.Greys#jet
    sm = cm.ScalarMappable(norm, cmap)
    sm.set_array(stnDensity)
    cmap.set_over(sm.to_rgba(5))
    cmap.set_under("white")
    #Load the climate division Line Collections for matplotlib
    lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
    lineCollect = lineCollect[sidx]
    
    #Setup map
    ax = plt.subplot(111)
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    m.drawcountries(linewidth=0.5,color='white')
    
    #Put Climate Divisions on map
    for x in np.arange(lineCollect.size):
    
        lines = lineCollect[x]
        if stnDensity[x] > 0:
            lines.set_facecolors(sm.to_rgba(stnDensity[x]))
        lines.set_edgecolors('k')
        lines.set_linewidth(0.5)
        ax.add_collection(lines)
    
    m.colorbar(sm,extend='max')
    #m.colorbar(sm)
    plt.show()
    

def plotStns():
    
    #Load the climate division Line Collections for matplotlib
    lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
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
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    #m.drawcountries(linewidth=0.5,color='white')
    
    #Put Climate Divisions on map
    for x in np.arange(lineCollect.size):
    
        lines = lineCollect[x]
        lines.set_edgecolors('k')
        lines.set_linewidth(0.5)
        ax.add_collection(lines)
    
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
    plt.savefig('/projects/daymet2/docs/final_writeup/stnsMap.png',dpi=300)
    plt.show()

def plotImpExample():

    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/infill_tmax.nc', 'tmax',stn_dtype=stnData.DTYPE_STN_BASIC)
    x = np.nonzero(stnda.stn_ids=='SNOTEL_13C01S')[0][0]#'SNOTEL_13C01S'
    tair = stnda.ds.variables['tmax'][:,x]
    impFlg = stnda.ds.variables['flag_impute'][:,x].astype(np.bool)
    tairObs = np.copy(tair)
    tairObs[impFlg] = np.nan
    tair[~impFlg] = np.nan
    
    plt.plot(tairObs,color='#E41A1C')
    plt.plot(tair,color='#377EB8')
    
    yrs = (1950,1960,1970,1980,1990,2000,2010)
    idxs = [np.nonzero(stnda.days[YEAR]==x)[0][0] for x in yrs]
    
    plt.xticks(idxs,[str(x) for x in yrs])
    plt.xlabel('Day',fontsize=17)
    plt.ylabel('$^\circ$C',fontsize=17)
    plt.legend(['Observed','Imputed'],loc=3,fontsize=17)
    ylocs,ylabs = plt.yticks()
    plt.yticks(ylocs,fontsize=17)
    xlocs,xlabs = plt.xticks()
    plt.xticks(xlocs,fontsize=17)
    plt.savefig('/projects/daymet2/docs/ncar_workshop2013_poster/impEg.png',dpi=300)
    plt.show()
    

def plotmaeVsStdErr():
    
    errVar = 'xval_overall_bias'
    #Load station counts per climate division into dictionary
    stndtype = copy(stnData.DTYPE_STN_DFLT)
    stndtype.append(('xval_overall_bias',np.float64))
    stndtype.append(('xval_overall_mae',np.float64))
    stndtype.append(('xval_stderr',np.float64))
    
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin',stn_dtype=stndtype)
    climDivs = stnda.stns[NEON]
    climDivs = np.unique(climDivs[np.isfinite(climDivs)])[2:]
    
    mab = []
    stderr = []
    
    for div in climDivs:
        mab.append(np.mean(np.abs(stnda.stns[errVar][np.logical_and(np.isfinite(stnda.stns[errVar]),stnda.stns[NEON]==div)])))
        stderr.append(np.mean(np.abs(stnda.stns['xval_overall_mae'][np.logical_and(np.isfinite(stnda.stns[errVar]),stnda.stns[NEON]==div)])))
    
    slope,incpt,r_value = stats.linregress(mab, stderr)[:3]
    r2 = r_value**2 #r-squared value; variance explained
    print r2
    plt.subplot(111)
    plt.plot(mab,stderr,'.',color='#17375E',label='_nolegend_')#"_nolegend_")
    xmin,xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = incpt + x*slope
    plt.plot(x,y,color="#C0504D",label="Regress Line",lw=2)
    plt.xlim((xmin,xmax))
    plt.xlabel("MAB ($^\circ$C)",fontsize=17)
    plt.ylabel("Std. Err. ($^\circ$C)",fontsize=17)
    plt.legend(loc=2,fontsize=17)
    ylocs,ylabs = plt.yticks()
    plt.yticks(ylocs,fontsize=17)
    xlocs,xlabs = plt.xticks()
    plt.xticks(xlocs,fontsize=17)
    plt.text(.70, .1, "".join(["R$^2$=%.2f"%(r2,)]), fontsize=17, transform=plt.gca().transAxes)
    
    fig = plt.gcf()
    fig.set_size_inches(8*.65,6*.65)
    plt.subplots_adjust(top=0.95,bottom=0.15,left=0.2,right=0.95)
    #plt.savefig('/projects/daymet2/docs/ncar_workshop2013_poster/stdErrVsMABTmin.png',dpi=300)
    plt.show()  

def plotBiasMaps():
    
    errVar = 'xval_overall_bias'
    #Load station counts per climate division into dictionary
    stndtype = copy(stnData.DTYPE_STN_DFLT)
    stndtype.append((errVar,np.float64))
    
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmin.nc', 'tmin',stn_dtype=stndtype)
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc', 'tmax',stn_dtype=stndtype)
    climDivs = stndaTmin.stns[NEON]
    climDivs = np.unique(climDivs[np.isfinite(climDivs)])[2:]
    
    mabTmin = []
    mabTmax = []
    
    for div in climDivs:
        #mabTmin.append(np.mean(stndaTmin.stns[errVar][np.logical_and(np.isfinite(stndaTmin.stns[errVar]),stndaTmin.stns[NEON]==div)]))
        #mabTmax.append(np.mean(stndaTmax.stns[errVar][np.logical_and(np.isfinite(stndaTmax.stns[errVar]),stndaTmax.stns[NEON]==div)]))
        mabTmin.append(np.mean(np.abs(stndaTmin.stns[errVar][np.logical_and(np.isfinite(stndaTmin.stns[errVar]),stndaTmin.stns[NEON]==div)])))
        mabTmax.append(np.mean(np.abs(stndaTmax.stns[errVar][np.logical_and(np.isfinite(stndaTmax.stns[errVar]),stndaTmax.stns[NEON]==div)])))
        
    mabTmin = np.array(mabTmin)
    mabTmax = np.array(mabTmax)
    errSet = (mabTmin,mabTmax)
    mabAll = np.concatenate(errSet)
    
    overallMabTmin = np.mean(np.abs(stndaTmin.stns[errVar][np.isfinite(stndaTmin.stns[errVar])]))
    overallMabTmax = np.mean(np.abs(stndaTmax.stns[errVar][np.isfinite(stndaTmax.stns[errVar])]))
    
    #Get idxs to sort climate divisions by #
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
    
    cf = plt.gcf()
    
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="5%")
    
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    
    norm = Normalize(np.min(mabAll), np.max(mabAll))
    #norm = Normalize(0.2, 1.5)
    cmap = cm.hot_r
    sm = cm.ScalarMappable(norm, cmap)
    sm.set_array(mabAll)
    
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

    cbar = grid[0].cax.colorbar(sm)
    cbar.set_label_text("MAE ($^\circ$C)",fontsize=17)
    #grid[0].set_yticks([.3,.6,.9,1.2,1.5])
    cbar.ax.set_yticks([.3,.6,.9,1.2,1.5])
    cbar.ax.tick_params(labelsize=17) 
    
    grid[0].text(0.025,0.075,"Overall MAE \n%.2f$^\circ$C"%(overallMabTmin,),transform=grid[0].transAxes,fontsize=17)
    grid[1].text(0.025,0.075,"Overall MAE \n%.2f$^\circ$C"%(overallMabTmax,),transform=grid[1].transAxes,fontsize=17)
    
    fig =plt.gcf()
    fig.set_size_inches(8*2,6*2)
    
    plt.savefig('/projects/daymet2/docs/ncar_workshop2013_poster/mabMaps.png',dpi=300)
    
    plt.show()     

#    m.drawcountries(linewidth=0.5,color='white')
#    
#    cax = plt.gca()
#    
#    lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
#    lineCollect = lineCollect[sidx]
#    
#    #Put Climate Divisions on map
#    for x in np.arange(lineCollect.size):
#        
#        lines = lineCollect[x]
#        lines.set_facecolors(sm.to_rgba(mab[x]))
#        lines.set_edgecolors('#8C8C8C')
#        lines.set_linewidth(0.5)
#        cax.add_collection(lines)
#    
#    
#    
#    cbar = m.colorbar(sm)
#    cbar.set_label("$^\circ$C",fontsize=17)
#    cbar.ax.tick_params(labelsize=17) 
#    plt.show() 

def plotMabMaps():
    
    errVar = 'xval_overall_bias'
    #Load station counts per climate division into dictionary
    stndtype = copy(stnData.DTYPE_STN_DFLT)
    stndtype.append((errVar,np.float64))
    
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmin.nc', 'tmin',stn_dtype=stndtype)
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc', 'tmax',stn_dtype=stndtype)
    climDivs = stndaTmin.stns[NEON]
    climDivs = np.unique(climDivs[np.isfinite(climDivs)])[2:]
    
    mabTmin = []
    mabTmax = []
    
    for div in climDivs:
        #mabTmin.append(np.mean(stndaTmin.stns[errVar][np.logical_and(np.isfinite(stndaTmin.stns[errVar]),stndaTmin.stns[NEON]==div)]))
        #mabTmax.append(np.mean(stndaTmax.stns[errVar][np.logical_and(np.isfinite(stndaTmax.stns[errVar]),stndaTmax.stns[NEON]==div)]))
        mabTmin.append(np.mean(np.abs(stndaTmin.stns[errVar][np.logical_and(np.isfinite(stndaTmin.stns[errVar]),stndaTmin.stns[NEON]==div)])))
        mabTmax.append(np.mean(np.abs(stndaTmax.stns[errVar][np.logical_and(np.isfinite(stndaTmax.stns[errVar]),stndaTmax.stns[NEON]==div)])))
        
    mabTmin = np.array(mabTmin)
    mabTmax = np.array(mabTmax)
    errSet = (mabTmin,mabTmax)
    mabAll = np.concatenate(errSet)
    
    overallMabTmin = np.mean(np.abs(stndaTmin.stns[errVar][np.isfinite(stndaTmin.stns[errVar])]))
    overallMabTmax = np.mean(np.abs(stndaTmax.stns[errVar][np.isfinite(stndaTmax.stns[errVar])]))
    
    #Get idxs to sort climate divisions by #
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
    
    cf = plt.gcf()
    
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="5%")
    
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    
    norm = Normalize(np.min(mabAll), np.max(mabAll))
    #norm = Normalize(0.2, 1.5)
    cmap = cm.hot_r
    sm = cm.ScalarMappable(norm, cmap)
    sm.set_array(mabAll)
    
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

    cbar = grid[0].cax.colorbar(sm)
    cbar.set_label_text("MAE ($^\circ$C)",fontsize=17)
    #grid[0].set_yticks([.3,.6,.9,1.2,1.5])
    cbar.ax.set_yticks([.3,.6,.9,1.2,1.5])
    cbar.ax.tick_params(labelsize=17) 
    
    grid[0].text(0.025,0.075,"Overall MAE \n%.2f$^\circ$C"%(overallMabTmin,),transform=grid[0].transAxes,fontsize=17)
    grid[1].text(0.025,0.075,"Overall MAE \n%.2f$^\circ$C"%(overallMabTmax,),transform=grid[1].transAxes,fontsize=17)
    
    fig =plt.gcf()
    fig.set_size_inches(8*2,6*2)
    
    plt.savefig('/projects/daymet2/docs/ncar_workshop2013_poster/mabMaps.png',dpi=300)
    
    plt.show()     

#    m.drawcountries(linewidth=0.5,color='white')
#    
#    cax = plt.gca()
#    
#    lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
#    lineCollect = lineCollect[sidx]
#    
#    #Put Climate Divisions on map
#    for x in np.arange(lineCollect.size):
#        
#        lines = lineCollect[x]
#        lines.set_facecolors(sm.to_rgba(mab[x]))
#        lines.set_edgecolors('#8C8C8C')
#        lines.set_linewidth(0.5)
#        cax.add_collection(lines)
#    
#    
#    
#    cbar = m.colorbar(sm)
#    cbar.set_label("$^\circ$C",fontsize=17)
#    cbar.ax.tick_params(labelsize=17) 
#    plt.show() 


def tminTmaxClimDivMap(errSet,subPltNum,sidx,norm=None,cmap=cm.hot_r):
        
    cf = plt.gcf()
    
    grid = ImageGrid(cf,subPltNum,nrows_ncols=(1,2),axes_pad=0.1,cbar_mode="single",cbar_location="right")

    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)

    errAll = np.concatenate(errSet)

    if norm is None:
        norm = Normalize(np.min(errAll[np.isfinite(errAll)]), np.max(errAll[np.isfinite(errAll)]))

    sm = cm.ScalarMappable(norm, cmap)
    sm.set_array(errAll[np.isfinite(errAll)])

    for i in np.arange(len(grid.axes_all)):
        
        gridCell = grid[i]
        
        m.ax = gridCell
        m.drawcountries(linewidth=0.5,color='white')
        
        lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
        lineCollect = lineCollect[sidx]
        
        #Put Climate Divisions on map
        for x in np.arange(lineCollect.size):
            
            lines = lineCollect[x]
            
            if np.isnan(errSet[i][x]):
                lines.set_facecolors('grey')
            else:
                lines.set_facecolors(sm.to_rgba(errSet[i][x]))
            lines.set_edgecolors('#8C8C8C')
            lines.set_linewidth(0.5)
            gridCell.add_collection(lines)
    cbar = grid[0].cax.colorbar(sm)
    #cbar.ax.tick_params(labelsize=17) 
    return grid,cbar

def analyzeInterpErr():
    
    #Load station counts per climate division into dictionary
    stndtype = copy(stnData.DTYPE_STN_DFLT)
    stndtype.append(('xval_overall_bias',np.float64))
    stndtype.append(('xval_overall_mae',np.float64))
    stndtype.append(('xval_overall_r2',np.float64))
    
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax',stn_dtype=stndtype)
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin',stn_dtype=stndtype)
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
    
    return biasTmin,biasTmax,mabTmin,mabTmax,maeTmin,maeTmax,r2Tmin,r2Tmax

def plotNcdcNormsErrorMaps():
    
    stnsNorm = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    maskStns = np.isfinite(stnsNorm['neon'])
    stnsNorm = stnsNorm[maskStns]
    climDivs = np.unique(stnsNorm['neon'])
    
    normsTmin = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmin.npy')[:,maskStns]
    normsTmax = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmax.npy')[:,maskStns]
    
    normsTminTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmin.npy')
    normsTmaxTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmax.npy')
    
    normsTminDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmin.npy')
    normsTmaxDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmax.npy')
    
    normsTminPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmin.npy')
    normsTmaxPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmax.npy')
        
    normsTmin = np.mean(normsTmin,axis=0)
    normsTmax = np.mean(normsTmax,axis=0)
    normsTminTwx = np.mean(normsTminTwx,axis=0)
    normsTmaxTwx = np.mean(normsTmaxTwx,axis=0)
    normsTminDaymet = np.mean(normsTminDaymet,axis=0)
    normsTmaxDaymet = np.mean(normsTmaxDaymet,axis=0)
    normsTminPrism = np.mean(normsTminPrism,axis=0)
    normsTmaxPrism = np.mean(normsTmaxPrism,axis=0)
#    mth = 0
#    normsTmin = normsTmin[mth,:]
#    normsTmax = normsTmax[mth,:]
#    normsTminTwx = normsTminTwx[mth,:]
#    normsTmaxTwx = normsTmaxTwx[mth,:]
#    normsTminDaymet = normsTminDaymet[mth,:]
#    normsTmaxDaymet = normsTmaxDaymet[mth,:]
#    normsTminPrism = normsTminPrism[mth,:]
#    normsTmaxPrism = normsTmaxPrism[mth,:]
    
    normsAll = [normsTminTwx,normsTmaxTwx,normsTminDaymet,normsTmaxDaymet,normsTminPrism,normsTmaxPrism]
    
    maskBadNorms = np.zeros(normsTminDaymet.size,dtype=np.bool)
    for aInterpNorm in normsAll:
        maskBadNorms = np.logical_or(maskBadNorms,imposs_value_mask(aInterpNorm))

    print "Removing a total of %d stations because of bad interpolated norms."%np.sum(maskBadNorms)
    normsTminTwx[maskBadNorms] = np.nan
    normsTmaxTwx[maskBadNorms]  = np.nan
    normsTminDaymet[maskBadNorms] = np.nan
    normsTmaxDaymet[maskBadNorms]  = np.nan
    normsTminPrism[maskBadNorms]  = np.nan
    normsTmaxPrism[maskBadNorms]  = np.nan
    
    difNormsTminTwx = np.abs(normsTminTwx-normsTmin)
    difNormsTmaxTwx = np.abs(normsTmaxTwx-normsTmax)
    difNormsTminDaymet = np.abs(normsTminDaymet-normsTmin)
    difNormsTmaxDaymet = np.abs(normsTmaxDaymet-normsTmax)
    difNormsTminPrism = np.abs(normsTminPrism-normsTmin)
    difNormsTmaxPrism = np.abs(normsTmaxPrism-normsTmax)
    
    difNormsTminTwx = np.ma.masked_array(difNormsTminTwx,np.isnan(difNormsTminTwx))
    difNormsTmaxTwx = np.ma.masked_array(difNormsTmaxTwx,np.isnan(difNormsTmaxTwx))
    difNormsTminDaymet = np.ma.masked_array(difNormsTminDaymet,np.isnan(difNormsTminDaymet))
    difNormsTmaxDaymet = np.ma.masked_array(difNormsTmaxDaymet,np.isnan(difNormsTmaxDaymet))
    difNormsTminPrism = np.ma.masked_array(difNormsTminPrism,np.isnan(difNormsTminPrism))
    difNormsTmaxPrism = np.ma.masked_array(difNormsTmaxPrism,np.isnan(difNormsTmaxPrism))
            
    fontsize=12
    
    maeNormDivsTminTwx = []
    maeNormDivsTmaxTwx = []  
    maeNormDivsTminDaymet = []
    maeNormDivsTmaxDaymet = []
    maeNormDivsTminPrism = []
    maeNormDivsTmaxPrism = []
    
    for div in climDivs:
        
        maskDivStns = stnsNorm[NEON]==div
        
        maeNormDivsTminTwx.append(np.ma.mean(difNormsTminTwx[maskDivStns]))
        maeNormDivsTmaxTwx.append(np.ma.mean(difNormsTmaxTwx[maskDivStns]))
        maeNormDivsTminDaymet.append(np.ma.mean(difNormsTminDaymet[maskDivStns]))
        maeNormDivsTmaxDaymet.append(np.ma.mean(difNormsTmaxDaymet[maskDivStns]))
        maeNormDivsTminPrism.append(np.ma.mean(difNormsTminPrism[maskDivStns]))
        maeNormDivsTmaxPrism.append(np.ma.mean(difNormsTmaxPrism[maskDivStns]))

    maeNormDivsTminTwx = np.array(maeNormDivsTminTwx)
    maeNormDivsTmaxTwx = np.array(maeNormDivsTmaxTwx)
    maeNormDivsTminDaymet = np.array(maeNormDivsTminDaymet)
    maeNormDivsTmaxDaymet = np.array(maeNormDivsTmaxDaymet)
    maeNormDivsTminPrism = np.array(maeNormDivsTminPrism)
    maeNormDivsTmaxPrism = np.array(maeNormDivsTmaxPrism)
    
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
    errAll = np.concatenate([maeNormDivsTminTwx,maeNormDivsTmaxTwx,maeNormDivsTminPrism,maeNormDivsTmaxPrism,maeNormDivsTminDaymet,maeNormDivsTmaxDaymet])
    #norm = Normalize(np.min(errAll[np.isfinite(errAll)]), np.max(errAll[np.isfinite(errAll)]))
    
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
    cmapB = brewer2mpl.get_map('YlOrRd', 'Sequential', 5, reverse=False)
    cmap = cmapB.get_mpl_colormap()
    cmapcolors = cmapB.mpl_colors
    
    cmapFnl = cm.hot_r.from_list('Custom cmap', cmapcolors[0:4], cmap.N)
    cmapFnl.set_over(cmapcolors[-1])
    norm = BoundaryNorm([0.0,0.25,0.5,0.75,1.0], cmapFnl.N)
    
    sm = cm.ScalarMappable(norm, cmapFnl)
    sm.set_array(errAll[np.isfinite(errAll)])
    
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
    
    drawErrorMap(0,maeNormDivsTminTwx)
    m.ax = grid[0]
    m.drawmeridians([-103])
    #cbar = grid[0].cax.colorbar(sm,extend='max')
    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0],extend='max')
    cbar.set_label("MAE ($^\circ$C)",fontsize=fontsize)
    grid[0].set_title(r"$\overline{TN}_a$")
    grid[1].set_title(r"$\overline{TX}_a$")
    grid[0].set_ylabel('TopoWx')
    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminTwx),),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxTwx),),transform=grid[1].transAxes,fontsize=10)
    drawErrorMap(1,maeNormDivsTmaxTwx)
    
    drawErrorMap(2,maeNormDivsTminPrism)
    drawErrorMap(3,maeNormDivsTmaxPrism)
    grid[2].set_ylabel('PRISM')
    grid[2].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminPrism),),transform=grid[2].transAxes,fontsize=10)
    grid[3].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxPrism),),transform=grid[3].transAxes,fontsize=10)
    
    drawErrorMap(4,maeNormDivsTminDaymet)
    drawErrorMap(5,maeNormDivsTmaxDaymet)
    grid[4].set_ylabel('Daymet')
    grid[4].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminDaymet),),transform=grid[4].transAxes,fontsize=10)
    grid[5].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxDaymet),),transform=grid[5].transAxes,fontsize=10)
    
    


    fig =plt.gcf()
#    #fig.set_size_inches(8*2,6*3)
    fig.set_size_inches(8,6)
#    fig.subplots_adjust(hspace=0.05)
    #plt.tight_layout()
    plt.savefig('/projects/daymet2/docs/final_writeup/dsCompareNormMaeMap.png',dpi=300)
    plt.show()
    
    
def plotNcdcNormsBiasMaps():
    
    stnsNorm = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    maskStns = np.isfinite(stnsNorm['neon'])
    stnsNorm = stnsNorm[maskStns]
    climDivs = np.unique(stnsNorm['neon'])
    
    normsTmin = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmin.npy')[:,maskStns]
    normsTmax = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmax.npy')[:,maskStns]
    
    normsTminTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmin.npy')
    normsTmaxTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmax.npy')
    
    normsTminDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmin.npy')
    normsTmaxDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmax.npy')
    
    normsTminPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmin.npy')
    normsTmaxPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmax.npy')
        
    normsTmin = np.mean(normsTmin,axis=0)
    normsTmax = np.mean(normsTmax,axis=0)
    normsTminTwx = np.mean(normsTminTwx,axis=0)
    normsTmaxTwx = np.mean(normsTmaxTwx,axis=0)
    normsTminDaymet = np.mean(normsTminDaymet,axis=0)
    normsTmaxDaymet = np.mean(normsTmaxDaymet,axis=0)
    normsTminPrism = np.mean(normsTminPrism,axis=0)
    normsTmaxPrism = np.mean(normsTmaxPrism,axis=0)
#    print normsTmin.shape
#    mth = 7
#    normsTmin = normsTmin[mth,:]
#    normsTmax = normsTmax[mth,:]
#    normsTminTwx = normsTminTwx[mth,:]
#    normsTmaxTwx = normsTmaxTwx[mth,:]
#    normsTminDaymet = normsTminDaymet[mth,:]
#    normsTmaxDaymet = normsTmaxDaymet[mth,:]
#    normsTminPrism = normsTminPrism[mth,:]
#    normsTmaxPrism = normsTmaxPrism[mth,:]
    
    normsAll = [normsTminTwx,normsTmaxTwx,normsTminDaymet,normsTmaxDaymet,normsTminPrism,normsTmaxPrism]
    
    maskBadNorms = np.zeros(normsTminDaymet.size,dtype=np.bool)
    for aInterpNorm in normsAll:
        maskBadNorms = np.logical_or(maskBadNorms,imposs_value_mask(aInterpNorm))

    print "Removing a total of %d stations because of bad interpolated norms."%np.sum(maskBadNorms)
    normsTminTwx[maskBadNorms] = np.nan
    normsTmaxTwx[maskBadNorms]  = np.nan
    normsTminDaymet[maskBadNorms] = np.nan
    normsTmaxDaymet[maskBadNorms]  = np.nan
    normsTminPrism[maskBadNorms]  = np.nan
    normsTmaxPrism[maskBadNorms]  = np.nan
    
    difNormsTminTwx = normsTminTwx-normsTmin
    difNormsTmaxTwx = normsTmaxTwx-normsTmax
    difNormsTminDaymet = normsTminDaymet-normsTmin
    difNormsTmaxDaymet = normsTmaxDaymet-normsTmax
    difNormsTminPrism = normsTminPrism-normsTmin
    difNormsTmaxPrism = normsTmaxPrism-normsTmax
    
    difNormsTminTwx = np.ma.masked_array(difNormsTminTwx,np.isnan(difNormsTminTwx))
    difNormsTmaxTwx = np.ma.masked_array(difNormsTmaxTwx,np.isnan(difNormsTmaxTwx))
    difNormsTminDaymet = np.ma.masked_array(difNormsTminDaymet,np.isnan(difNormsTminDaymet))
    difNormsTmaxDaymet = np.ma.masked_array(difNormsTmaxDaymet,np.isnan(difNormsTmaxDaymet))
    difNormsTminPrism = np.ma.masked_array(difNormsTminPrism,np.isnan(difNormsTminPrism))
    difNormsTmaxPrism = np.ma.masked_array(difNormsTmaxPrism,np.isnan(difNormsTmaxPrism))
            
    fontsize=12
    
    maeNormDivsTminTwx = []
    maeNormDivsTmaxTwx = []  
    maeNormDivsTminDaymet = []
    maeNormDivsTmaxDaymet = []
    maeNormDivsTminPrism = []
    maeNormDivsTmaxPrism = []
    
    for div in climDivs:
        
        maskDivStns = stnsNorm[NEON]==div
        
        maeNormDivsTminTwx.append(np.ma.mean(difNormsTminTwx[maskDivStns]))
        maeNormDivsTmaxTwx.append(np.ma.mean(difNormsTmaxTwx[maskDivStns]))
        maeNormDivsTminDaymet.append(np.ma.mean(difNormsTminDaymet[maskDivStns]))
        maeNormDivsTmaxDaymet.append(np.ma.mean(difNormsTmaxDaymet[maskDivStns]))
        maeNormDivsTminPrism.append(np.ma.mean(difNormsTminPrism[maskDivStns]))
        maeNormDivsTmaxPrism.append(np.ma.mean(difNormsTmaxPrism[maskDivStns]))

    maeNormDivsTminTwx = np.array(maeNormDivsTminTwx)
    maeNormDivsTmaxTwx = np.array(maeNormDivsTmaxTwx)
    maeNormDivsTminDaymet = np.array(maeNormDivsTminDaymet)
    maeNormDivsTmaxDaymet = np.array(maeNormDivsTmaxDaymet)
    maeNormDivsTminPrism = np.array(maeNormDivsTminPrism)
    maeNormDivsTmaxPrism = np.array(maeNormDivsTmaxPrism)
    
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
    errAll = np.concatenate([maeNormDivsTminTwx,maeNormDivsTmaxTwx,maeNormDivsTminPrism,maeNormDivsTmaxPrism,maeNormDivsTminDaymet,maeNormDivsTmaxDaymet])
    print np.min(errAll[np.isfinite(errAll)]),np.max(errAll[np.isfinite(errAll)])
    #norm = Normalize(np.min(errAll[np.isfinite(errAll)]), np.max(errAll[np.isfinite(errAll)]))
    
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
    
    
    clrsRed = brewer2mpl.get_map('Reds', 'Sequential', 9, reverse=False).mpl_colors
    clrsBlue = brewer2mpl.get_map('Blues', 'Sequential', 9, reverse=True).mpl_colors
    
    clrsBlue.append("grey")
    clrsBlue.append("grey")
    clrsBlue.extend(clrsRed)
    clrs = clrsBlue
    
    #clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
    #clrs = clrs.mpl_colors
    #clrs[5] = "grey"
    #levels = [-.9,-8,-.7,-.6,-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]
    levels = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.,0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9 ]
    #levels = np.linspace(-1,1,21)
    
    cmapFnl = cm.hot_r.from_list('Custom cmap', clrs[1:-1], 256)
    cmapFnl.set_over(clrs[-1])
    cmapFnl.set_under(clrs[0])
    norm = BoundaryNorm(levels, cmapFnl.N)
    
    sm = cm.ScalarMappable(norm, cmapFnl)
    sm.set_array(errAll[np.isfinite(errAll)])
    
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
    
    drawErrorMap(0,maeNormDivsTminTwx)
    m.ax = grid[0]
    m.drawmeridians([-103])
    #cbar = grid[0].cax.colorbar(sm,extend='max')
    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0],extend='both')
    cbar.set_label("MAE ($^\circ$C)",fontsize=fontsize)
    grid[0].set_title(r"$\overline{TN}_a$")
    grid[1].set_title(r"$\overline{TX}_a$")
    grid[0].set_ylabel('TopoWx')
    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminTwx),),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxTwx),),transform=grid[1].transAxes,fontsize=10)
    drawErrorMap(1,maeNormDivsTmaxTwx)
    
    drawErrorMap(2,maeNormDivsTminPrism)
    drawErrorMap(3,maeNormDivsTmaxPrism)
    grid[2].set_ylabel('PRISM')
    grid[2].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminPrism),),transform=grid[2].transAxes,fontsize=10)
    grid[3].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxPrism),),transform=grid[3].transAxes,fontsize=10)
    
    drawErrorMap(4,maeNormDivsTminDaymet)
    drawErrorMap(5,maeNormDivsTmaxDaymet)
    grid[4].set_ylabel('Daymet')
    grid[4].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminDaymet),),transform=grid[4].transAxes,fontsize=10)
    grid[5].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxDaymet),),transform=grid[5].transAxes,fontsize=10)
    
    


    fig =plt.gcf()
#    #fig.set_size_inches(8*2,6*3)
    fig.set_size_inches(8,6)
#    fig.subplots_adjust(hspace=0.05)
    #plt.tight_layout()
    #plt.savefig('/projects/daymet2/docs/final_writeup/dsCompareNormMaeMap.png',dpi=300)
    plt.show()

def plotNcdcNormsErrorMapsDifs():
    
    stnsNorm = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    maskStns = np.isfinite(stnsNorm['neon'])
    stnsNorm = stnsNorm[maskStns]
    climDivs = np.unique(stnsNorm['neon'])
    
    normsTmin = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmin.npy')[:,maskStns]
    normsTmax = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmax.npy')[:,maskStns]
    
    normsTminTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmin.npy')
    normsTmaxTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmax.npy')
    
    normsTminDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmin.npy')
    normsTmaxDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmax.npy')
    
    normsTminPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmin.npy')
    normsTmaxPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmax.npy')
        
    normsTmin = np.mean(normsTmin,axis=0)
    normsTmax = np.mean(normsTmax,axis=0)
    normsTminTwx = np.mean(normsTminTwx,axis=0)
    normsTmaxTwx = np.mean(normsTmaxTwx,axis=0)
    normsTminDaymet = np.mean(normsTminDaymet,axis=0)
    normsTmaxDaymet = np.mean(normsTmaxDaymet,axis=0)
    normsTminPrism = np.mean(normsTminPrism,axis=0)
    normsTmaxPrism = np.mean(normsTmaxPrism,axis=0)
#    mth = 0
#    normsTmin = normsTmin[mth,:]
#    normsTmax = normsTmax[mth,:]
#    normsTminTwx = normsTminTwx[mth,:]
#    normsTmaxTwx = normsTmaxTwx[mth,:]
#    normsTminDaymet = normsTminDaymet[mth,:]
#    normsTmaxDaymet = normsTmaxDaymet[mth,:]
#    normsTminPrism = normsTminPrism[mth,:]
#    normsTmaxPrism = normsTmaxPrism[mth,:]
    
    normsAll = [normsTminTwx,normsTmaxTwx,normsTminDaymet,normsTmaxDaymet,normsTminPrism,normsTmaxPrism]
    
    maskBadNorms = np.zeros(normsTminDaymet.size,dtype=np.bool)
    for aInterpNorm in normsAll:
        maskBadNorms = np.logical_or(maskBadNorms,imposs_value_mask(aInterpNorm))

    print "Removing a total of %d stations because of bad interpolated norms."%np.sum(maskBadNorms)
    normsTminTwx[maskBadNorms] = np.nan
    normsTmaxTwx[maskBadNorms]  = np.nan
    normsTminDaymet[maskBadNorms] = np.nan
    normsTmaxDaymet[maskBadNorms]  = np.nan
    normsTminPrism[maskBadNorms]  = np.nan
    normsTmaxPrism[maskBadNorms]  = np.nan
    
    print np.sum(np.isfinite(normsTminTwx))
    
    difNormsTminTwx = np.abs(normsTminTwx-normsTmin)
    difNormsTmaxTwx = np.abs(normsTmaxTwx-normsTmax)
    difNormsTminDaymet = np.abs(normsTminDaymet-normsTmin)
    difNormsTmaxDaymet = np.abs(normsTmaxDaymet-normsTmax)
    difNormsTminPrism = np.abs(normsTminPrism-normsTmin)
    difNormsTmaxPrism = np.abs(normsTmaxPrism-normsTmax)
    
    difNormsTminTwx = np.ma.masked_array(difNormsTminTwx,np.isnan(difNormsTminTwx))
    difNormsTmaxTwx = np.ma.masked_array(difNormsTmaxTwx,np.isnan(difNormsTmaxTwx))
    difNormsTminDaymet = np.ma.masked_array(difNormsTminDaymet,np.isnan(difNormsTminDaymet))
    difNormsTmaxDaymet = np.ma.masked_array(difNormsTmaxDaymet,np.isnan(difNormsTmaxDaymet))
    difNormsTminPrism = np.ma.masked_array(difNormsTminPrism,np.isnan(difNormsTminPrism))
    difNormsTmaxPrism = np.ma.masked_array(difNormsTmaxPrism,np.isnan(difNormsTmaxPrism))
            
    fontsize=12
    
    maeNormDivsTminTwx = []
    maeNormDivsTmaxTwx = []  
    maeNormDivsTminDaymet = []
    maeNormDivsTmaxDaymet = []
    maeNormDivsTminPrism = []
    maeNormDivsTmaxPrism = []
    
    for div in climDivs:
        
        maskDivStns = stnsNorm[NEON]==div
        
        maeNormDivsTminTwx.append(np.ma.mean(difNormsTminTwx[maskDivStns]))
        maeNormDivsTmaxTwx.append(np.ma.mean(difNormsTmaxTwx[maskDivStns]))
        maeNormDivsTminDaymet.append(np.ma.mean(difNormsTminDaymet[maskDivStns]))
        maeNormDivsTmaxDaymet.append(np.ma.mean(difNormsTmaxDaymet[maskDivStns]))
        maeNormDivsTminPrism.append(np.ma.mean(difNormsTminPrism[maskDivStns]))
        maeNormDivsTmaxPrism.append(np.ma.mean(difNormsTmaxPrism[maskDivStns]))

    maeNormDivsTminTwx = np.array(maeNormDivsTminTwx)
    maeNormDivsTmaxTwx = np.array(maeNormDivsTmaxTwx)
    maeNormDivsTminDaymet = np.array(maeNormDivsTminDaymet)
    maeNormDivsTmaxDaymet = np.array(maeNormDivsTmaxDaymet)
    maeNormDivsTminPrism = np.array(maeNormDivsTminPrism)
    maeNormDivsTmaxPrism = np.array(maeNormDivsTmaxPrism)
    
    #Get idxs to sort climate divisions by #
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
    
    print "Starting to plot...."
    cf = plt.gcf()
    
    grid = ImageGrid(cf,212,nrows_ncols=(2,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")

    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    #cmap = cm.hot_r
    #errAll = np.concatenate([maeNormDivsTminTwx,maeNormDivsTmaxTwx,maeNormDivsTminPrism,maeNormDivsTmaxPrism,maeNormDivsTminDaymet,maeNormDivsTmaxDaymet])
    #norm = Normalize(np.min(errAll[np.isfinite(errAll)]), np.max(errAll[np.isfinite(errAll)]))
    
    maeNormDivsTminPrism = maeNormDivsTminPrism - maeNormDivsTminTwx
    maeNormDivsTmaxPrism = maeNormDivsTmaxPrism - maeNormDivsTmaxTwx
    maeNormDivsTminDaymet = maeNormDivsTminDaymet - maeNormDivsTminTwx
    maeNormDivsTmaxDaymet = maeNormDivsTmaxDaymet - maeNormDivsTmaxTwx
    
    errAll = np.concatenate([maeNormDivsTminPrism,maeNormDivsTmaxPrism,maeNormDivsTminDaymet,maeNormDivsTmaxDaymet])
    print np.min(errAll[np.isfinite(errAll)]),np.max(errAll[np.isfinite(errAll)])
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
    
    clrsRed = brewer2mpl.get_map('Reds', 'Sequential', 6, reverse=False).mpl_colors
    clrsBlue = brewer2mpl.get_map('Blues', 'Sequential', 6, reverse=True).mpl_colors
    N = brewer2mpl.get_map('Reds', 'Sequential', 6, reverse=False).get_mpl_colormap().N 
    clrsBlue.append("grey")
    clrsBlue.extend(clrsRed)
    clrs = clrsBlue
#    
#    #clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
#    #clrs = clrs.mpl_colors
#    #clrs[5] = "grey"
#    levels = [-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]
    
    cmapFnl = cm.hot_r.from_list('Custom cmap', clrs[1:-1], N)
    cmapFnl.set_over(clrs[-1])
    cmapFnl.set_under(clrs[0])
    levels = [-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]
    norm = BoundaryNorm(levels, cmapFnl.N)
    
    sm = cm.ScalarMappable(norm, cmapFnl)
    sm.set_array(errAll[np.isfinite(errAll)])
    
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
    
    drawErrorMap(0,maeNormDivsTminPrism)
    #cbar = grid[0].cax.colorbar(sm,extend='max')
    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0],extend='both')
    cbar.set_label("MAE Difference ($^\circ$C)",fontsize=fontsize)
    cbar.set_ticks(levels)
    grid[0].set_title(r"$\overline{TN}_a$")
    grid[1].set_title(r"$\overline{TX}_a$")
    grid[0].set_ylabel('PRISM Minus TopoWx')
    #grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminTwx),),transform=grid[0].transAxes,fontsize=10)
    #grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxTwx),),transform=grid[1].transAxes,fontsize=10)
    drawErrorMap(1,maeNormDivsTmaxPrism)
    
    drawErrorMap(2,maeNormDivsTminDaymet)
    drawErrorMap(3,maeNormDivsTmaxDaymet)
    grid[2].set_ylabel('Daymet Minus TopoWx')
    #grid[2].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminPrism),),transform=grid[2].transAxes,fontsize=10)
    #grid[3].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxPrism),),transform=grid[3].transAxes,fontsize=10)
    
    #drawErrorMap(4,maeNormDivsTminDaymet)
    #drawErrorMap(5,maeNormDivsTmaxDaymet)
    #grid[4].set_ylabel('Daymet')
    #grid[4].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTminDaymet),),transform=grid[4].transAxes,fontsize=10)
    #grid[5].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(np.ma.mean(difNormsTmaxDaymet),),transform=grid[5].transAxes,fontsize=10)
    
    


    fig =plt.gcf()
#    #fig.set_size_inches(8*2,6*3)
    fig.set_size_inches(6,6)
#    fig.subplots_adjust(hspace=0.05)
    #plt.tight_layout()
    #plt.savefig('/projects/daymet2/docs/final_writeup/climDivErrMaps.png',dpi=150)
    plt.show()
  
def plotNcdcNormsErrorBars():
    
    stnsNorm = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    maskStns = np.isfinite(stnsNorm['neon'])
    
    stnsNorm = stnsNorm[maskStns]
    climDivs = np.unique(stnsNorm['neon'])
    
    normsTmin = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmin.npy')[:,maskStns]
    normsTmax = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmax.npy')[:,maskStns]
    
    normsTminTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmin.npy')
    normsTmaxTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmax.npy')
    
    normsTminDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmin.npy')
    normsTmaxDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmax.npy')
    
    normsTminPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmin.npy')
    normsTmaxPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmax.npy')
    
    maskStns2 = stnsNorm[LON] <= -103.0#stnsNorm[LON] > -99999#stnsNorm[LON] <= -104.0
    print np.sum(maskStns2)
    stnsNorm = stnsNorm[maskStns2]
    normsTmin = normsTmin[:,maskStns2]
    normsTmax = normsTmax[:,maskStns2]
    normsTminTwx = normsTminTwx[:,maskStns2]
    normsTmaxTwx = normsTmaxTwx[:,maskStns2]
    normsTminDaymet = normsTminDaymet[:,maskStns2]
    normsTmaxDaymet = normsTmaxDaymet[:,maskStns2]
    normsTminPrism = normsTminPrism[:,maskStns2]
    normsTmaxPrism = normsTmaxPrism[:,maskStns2]
        
#    normsTmin = np.mean(normsTmin,axis=0)
#    normsTmax = np.mean(normsTmax,axis=0)
#    normsTminTwx = np.mean(normsTminTwx,axis=0)
#    normsTmaxTwx = np.mean(normsTmaxTwx,axis=0)
#    normsTminDaymet = np.mean(normsTminDaymet,axis=0)
#    normsTmaxDaymet = np.mean(normsTmaxDaymet,axis=0)
#    normsTminPrism = np.mean(normsTminPrism,axis=0)
#    normsTmaxPrism = np.mean(normsTmaxPrism,axis=0)
#    mth = 0
#    normsTmin = normsTmin[mth,:]
#    normsTmax = normsTmax[mth,:]
#    normsTminTwx = normsTminTwx[mth,:]
#    normsTmaxTwx = normsTmaxTwx[mth,:]
#    normsTminDaymet = normsTminDaymet[mth,:]
#    normsTmaxDaymet = normsTmaxDaymet[mth,:]
#    normsTminPrism = normsTminPrism[mth,:]
#    normsTmaxPrism = normsTmaxPrism[mth,:]
    
    maeTminTwx = np.zeros(12)
    maeTminPrism = np.zeros(12)
    maeTminDaymet = np.zeros(12)
    
    maeTmaxTwx = np.zeros(12)
    maeTmaxPrism = np.zeros(12)
    maeTmaxDaymet = np.zeros(12)
    
    for mth in np.arange(12):

        normsMthTmin = normsTmin[mth,:]
        normsMthTmax = normsTmax[mth,:]
        
        normsMthTminTwx = normsTminTwx[mth,:]
        normsMthTmaxTwx = normsTmaxTwx[mth,:]
        
        normsMthTminDaymet = normsTminDaymet[mth,:]
        normsMthTmaxDaymet = normsTmaxDaymet[mth,:]
        
        normsMthTminPrism = normsTminPrism[mth,:]
        normsMthTmaxPrism = normsTmaxPrism[mth,:]
        
        normsAll = [normsMthTminTwx,normsMthTmaxTwx,normsMthTminDaymet,normsMthTmaxDaymet,normsMthTminPrism,normsMthTmaxPrism]
    
        maskBadNorms = np.zeros(normsMthTminDaymet.size,dtype=np.bool)
        for aInterpNorm in normsAll:
            maskBadNorms = np.logical_or(maskBadNorms,imposs_value_mask(aInterpNorm))

        normsMthTminTwx[maskBadNorms] = np.nan
        normsMthTmaxTwx[maskBadNorms]  = np.nan
        normsMthTminDaymet[maskBadNorms] = np.nan
        normsMthTmaxDaymet[maskBadNorms]  = np.nan
        normsMthTminPrism[maskBadNorms]  = np.nan
        normsMthTmaxPrism[maskBadNorms]  = np.nan
    
        difNormsTminTwx = np.abs(normsMthTminTwx-normsMthTmin)
        difNormsTmaxTwx = np.abs(normsMthTmaxTwx-normsMthTmax)
        difNormsTminDaymet = np.abs(normsMthTminDaymet-normsMthTmin)
        difNormsTmaxDaymet = np.abs(normsMthTmaxDaymet-normsMthTmax)
        difNormsTminPrism = np.abs(normsMthTminPrism-normsMthTmin)
        difNormsTmaxPrism = np.abs(normsMthTmaxPrism-normsMthTmax)
    
        difNormsTminTwx = np.ma.masked_array(difNormsTminTwx,np.isnan(difNormsTminTwx))
        difNormsTmaxTwx = np.ma.masked_array(difNormsTmaxTwx,np.isnan(difNormsTmaxTwx))
        difNormsTminDaymet = np.ma.masked_array(difNormsTminDaymet,np.isnan(difNormsTminDaymet))
        difNormsTmaxDaymet = np.ma.masked_array(difNormsTmaxDaymet,np.isnan(difNormsTmaxDaymet))
        difNormsTminPrism = np.ma.masked_array(difNormsTminPrism,np.isnan(difNormsTminPrism))
        difNormsTmaxPrism = np.ma.masked_array(difNormsTmaxPrism,np.isnan(difNormsTmaxPrism))
        
        maeTminTwx[mth] = np.ma.mean(difNormsTminTwx)
        maeTminPrism[mth] = np.ma.mean(difNormsTminPrism)
        maeTminDaymet[mth] = np.ma.mean(difNormsTminDaymet)
        maeTmaxTwx[mth] = np.ma.mean(difNormsTmaxTwx)
        maeTmaxPrism[mth] = np.ma.mean(difNormsTmaxPrism)
        maeTmaxDaymet[mth] = np.ma.mean(difNormsTmaxDaymet)
                
##############################################################################

    print "Starting to plot...."

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    bwidth = .25
    xlim = (-.25,12.90)
        
    plt.sca(ax1)
    plt.bar(np.arange(maeTminTwx.size),maeTminTwx,width=bwidth,color="k")
    plt.bar(np.arange(maeTminPrism.size)+bwidth,maeTminPrism,width=bwidth,color="grey")
    plt.bar(np.arange(maeTminDaymet.size)+(bwidth*2),maeTminDaymet,width=bwidth,color="w",hatch="//")
    xlim = plt.xlim(-.2,12)
    plt.xticks(np.arange(maeTminTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    plt.ylim((0,1.3))
    plt.yticks(plt.yticks()[0],fontsize=10)
    plt.ylabel("MAE ($^\circ$C)",fontsize=10)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("a.",loc="left")
    
    plt.sca(ax2)
    plt.bar(np.arange(maeTmaxTwx.size),maeTmaxTwx,width=bwidth,color="k")
    plt.bar(np.arange(maeTmaxPrism.size)+bwidth,maeTmaxPrism,width=bwidth,color="grey")
    plt.bar(np.arange(maeTmaxDaymet.size)+(bwidth*2),maeTmaxDaymet,width=bwidth,color="w",hatch="//")
    plt.legend(("TopoWx","PRISM","Daymet"),fontsize=10)
    plt.ylim((0,1.3))
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(maeTmaxTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("b.",loc="left")
    
    plt.tight_layout()
    
    f.set_size_inches(8,3.2)
    plt.savefig('/projects/daymet2/docs/final_writeup/dsCompareNormalsMaeBar.png',dpi=250)
    plt.show()


def plotNcdcNormsBiasBars():
    
    stnsNorm = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    maskStns = np.isfinite(stnsNorm['neon'])
    
    stnsNorm = stnsNorm[maskStns]
    climDivs = np.unique(stnsNorm['neon'])
    
    normsTmin = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmin.npy')[:,maskStns]
    normsTmax = np.load('/projects/daymet2/station_data/ncdc_normals/norm_tmax.npy')[:,maskStns]
    
    normsTminTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmin.npy')
    normsTmaxTwx = np.load('/projects/daymet2/ds_compare/normals/twx_norms_tmax.npy')
    
    normsTminDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmin.npy')
    normsTmaxDaymet = np.load('/projects/daymet2/ds_compare/normals/daymet_norms_tmax.npy')
    
    normsTminPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmin.npy')
    normsTmaxPrism = np.load('/projects/daymet2/ds_compare/normals/prism_norms_tmax.npy')
    
    maskStns2 = stnsNorm[LON] > -99999#stnsNorm[LON] > -99999#stnsNorm[LON] <= -104.0
    print np.sum(maskStns2)
    stnsNorm = stnsNorm[maskStns2]
    normsTmin = normsTmin[:,maskStns2]
    normsTmax = normsTmax[:,maskStns2]
    normsTminTwx = normsTminTwx[:,maskStns2]
    normsTmaxTwx = normsTmaxTwx[:,maskStns2]
    normsTminDaymet = normsTminDaymet[:,maskStns2]
    normsTmaxDaymet = normsTmaxDaymet[:,maskStns2]
    normsTminPrism = normsTminPrism[:,maskStns2]
    normsTmaxPrism = normsTmaxPrism[:,maskStns2]
        
#    normsTmin = np.mean(normsTmin,axis=0)
#    normsTmax = np.mean(normsTmax,axis=0)
#    normsTminTwx = np.mean(normsTminTwx,axis=0)
#    normsTmaxTwx = np.mean(normsTmaxTwx,axis=0)
#    normsTminDaymet = np.mean(normsTminDaymet,axis=0)
#    normsTmaxDaymet = np.mean(normsTmaxDaymet,axis=0)
#    normsTminPrism = np.mean(normsTminPrism,axis=0)
#    normsTmaxPrism = np.mean(normsTmaxPrism,axis=0)
#    mth = 0
#    normsTmin = normsTmin[mth,:]
#    normsTmax = normsTmax[mth,:]
#    normsTminTwx = normsTminTwx[mth,:]
#    normsTmaxTwx = normsTmaxTwx[mth,:]
#    normsTminDaymet = normsTminDaymet[mth,:]
#    normsTmaxDaymet = normsTmaxDaymet[mth,:]
#    normsTminPrism = normsTminPrism[mth,:]
#    normsTmaxPrism = normsTmaxPrism[mth,:]
    
    maeTminTwx = np.zeros(12)
    maeTminPrism = np.zeros(12)
    maeTminDaymet = np.zeros(12)
    
    maeTmaxTwx = np.zeros(12)
    maeTmaxPrism = np.zeros(12)
    maeTmaxDaymet = np.zeros(12)
    
    for mth in np.arange(12):

        normsMthTmin = normsTmin[mth,:]
        normsMthTmax = normsTmax[mth,:]
        
        normsMthTminTwx = normsTminTwx[mth,:]
        normsMthTmaxTwx = normsTmaxTwx[mth,:]
        
        normsMthTminDaymet = normsTminDaymet[mth,:]
        normsMthTmaxDaymet = normsTmaxDaymet[mth,:]
        
        normsMthTminPrism = normsTminPrism[mth,:]
        normsMthTmaxPrism = normsTmaxPrism[mth,:]
        
        normsAll = [normsMthTminTwx,normsMthTmaxTwx,normsMthTminDaymet,normsMthTmaxDaymet,normsMthTminPrism,normsMthTmaxPrism]
    
        maskBadNorms = np.zeros(normsMthTminDaymet.size,dtype=np.bool)
        for aInterpNorm in normsAll:
            maskBadNorms = np.logical_or(maskBadNorms,imposs_value_mask(aInterpNorm))

        normsMthTminTwx[maskBadNorms] = np.nan
        normsMthTmaxTwx[maskBadNorms]  = np.nan
        normsMthTminDaymet[maskBadNorms] = np.nan
        normsMthTmaxDaymet[maskBadNorms]  = np.nan
        normsMthTminPrism[maskBadNorms]  = np.nan
        normsMthTmaxPrism[maskBadNorms]  = np.nan
    
        difNormsTminTwx = normsMthTminTwx-normsMthTmin
        difNormsTmaxTwx = normsMthTmaxTwx-normsMthTmax
        difNormsTminDaymet = normsMthTminDaymet-normsMthTmin
        difNormsTmaxDaymet = normsMthTmaxDaymet-normsMthTmax
        difNormsTminPrism = normsMthTminPrism-normsMthTmin
        difNormsTmaxPrism = normsMthTmaxPrism-normsMthTmax
    
        difNormsTminTwx = np.ma.masked_array(difNormsTminTwx,np.isnan(difNormsTminTwx))
        difNormsTmaxTwx = np.ma.masked_array(difNormsTmaxTwx,np.isnan(difNormsTmaxTwx))
        difNormsTminDaymet = np.ma.masked_array(difNormsTminDaymet,np.isnan(difNormsTminDaymet))
        difNormsTmaxDaymet = np.ma.masked_array(difNormsTmaxDaymet,np.isnan(difNormsTmaxDaymet))
        difNormsTminPrism = np.ma.masked_array(difNormsTminPrism,np.isnan(difNormsTminPrism))
        difNormsTmaxPrism = np.ma.masked_array(difNormsTmaxPrism,np.isnan(difNormsTmaxPrism))
        
        maeTminTwx[mth] = np.ma.mean(difNormsTminTwx)
        maeTminPrism[mth] = np.ma.mean(difNormsTminPrism)
        maeTminDaymet[mth] = np.ma.mean(difNormsTminDaymet)
        maeTmaxTwx[mth] = np.ma.mean(difNormsTmaxTwx)
        maeTmaxPrism[mth] = np.ma.mean(difNormsTmaxPrism)
        maeTmaxDaymet[mth] = np.ma.mean(difNormsTmaxDaymet)
                
##############################################################################

    print "Starting to plot...."

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    bwidth = .25
    xlim = (-.25,12.90)
        
    plt.sca(ax1)
    plt.bar(np.arange(maeTminTwx.size),maeTminTwx,width=bwidth,color="k")
    plt.bar(np.arange(maeTminPrism.size)+bwidth,maeTminPrism,width=bwidth,color="grey")
    plt.bar(np.arange(maeTminDaymet.size)+(bwidth*2),maeTminDaymet,width=bwidth,color="w",hatch="//")
    xlim = plt.xlim(-.2,12)
    plt.xticks(np.arange(maeTminTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    #plt.ylim((0,1.3))
    #plt.yticks(plt.yticks()[0],fontsize=10)
    plt.ylabel("MAE ($^\circ$C)",fontsize=10)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("a.",loc="left")
    
    plt.sca(ax2)
    plt.bar(np.arange(maeTmaxTwx.size),maeTmaxTwx,width=bwidth,color="k")
    plt.bar(np.arange(maeTmaxPrism.size)+bwidth,maeTmaxPrism,width=bwidth,color="grey")
    plt.bar(np.arange(maeTmaxDaymet.size)+(bwidth*2),maeTmaxDaymet,width=bwidth,color="w",hatch="//")
    plt.legend(("TopoWx","PRISM","Daymet"),fontsize=10)
    #plt.ylim((0,1.3))
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(maeTmaxTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("b.",loc="left")
    
    plt.tight_layout()
    
    f.set_size_inches(8,3.2)
    #plt.savefig('/projects/daymet2/docs/final_writeup/dsCompareNormalsMaeBar.png',dpi=250)
    plt.show()

def plotInterpErrorMaps():
    
    fontsize=12
    
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    climDivs = np.concatenate((stndaTmax.stns[NEON],stndaTmin.stns[NEON]))
    climDivs = np.unique(climDivs[np.isfinite(climDivs)])[2:]
    
    mth = 12
    
    maeNormTmin = stndaTmin.ds.variables['xvalfnl_mae_norm'][mth,:]
    maeNormTmax = stndaTmax.ds.variables['xvalfnl_mae_norm'][mth,:]
    maeDlyTmin = stndaTmin.ds.variables['xvalfnl_mae_dly'][mth,:]
    maeDlyTmax = stndaTmax.ds.variables['xvalfnl_mae_dly'][mth]
    r2DlyTmin = stndaTmin.ds.variables['xvalfnl_r2_dly'][mth,:]
    r2DlyTmax = stndaTmax.ds.variables['xvalfnl_r2_dly'][mth,:]
    
    biasTmin = stndaTmin.ds.variables['xvalfnl_bias_dly'][mth,:]
    biasTmax = stndaTmax.ds.variables['xvalfnl_bias_dly'][mth,:]
    
    maeNormDivsTmin = []
    maeNormDivsTmax = []
    maeDlyDivsTmin = []
    maeDlyDivsTmax = []
    r2DlyDivsTmin = []
    r2DlyDivsTmax = []
    
    for div in climDivs:
        
        maskStnsTmin = stndaTmin.stns[NEON]==div
        maskStnsTmax = stndaTmax.stns[NEON]==div
        
        maeNormDivsTmin.append(np.ma.mean(maeNormTmin[maskStnsTmin]))
        maeNormDivsTmax.append(np.ma.mean(maeNormTmax[maskStnsTmax]))
        
        maeDlyDivsTmin.append(np.ma.mean(maeDlyTmin[maskStnsTmin]))
        maeDlyDivsTmax.append(np.ma.mean(maeDlyTmax[maskStnsTmax]))
        
        r2DlyDivsTmin.append(np.ma.mean(r2DlyTmin[maskStnsTmin]))
        r2DlyDivsTmax.append(np.ma.mean(r2DlyTmax[maskStnsTmax]))
     
    maeNormDivsTmin = np.array(maeNormDivsTmin)
    maeNormDivsTmax = np.array(maeNormDivsTmax)
    maeDlyDivsTmin = np.array(maeDlyDivsTmin)
    maeDlyDivsTmax = np.array(maeDlyDivsTmax)
    r2DlyDivsTmin = np.array(r2DlyDivsTmin)
    r2DlyDivsTmax = np.array(r2DlyDivsTmax)
    
    climDivsMaskTmin = np.in1d(stndaTmin.stns[NEON], climDivs)
    climDivsMaskTmax = np.in1d(stndaTmax.stns[NEON], climDivs)
    
    print "TMIN Max Daily MAE: "+str(np.max(maeDlyDivsTmin))
    print "TMAX Max Daily MAE: "+str(np.max(maeDlyDivsTmax))
    
    print "90th Percentil Tmax Normal MAE: "+str(np.percentile(maeNormDivsTmax,90))
    
    print "TMIN BIAS: ",str(np.mean(biasTmin[climDivsMaskTmin]))
    print "TMAX BIAS: ",str(np.mean(biasTmax[climDivsMaskTmax]))
    
    allMaeNormTmin = np.ma.mean(maeNormTmin[climDivsMaskTmin])
    allMaeNormTmax = np.ma.mean(maeNormTmax[climDivsMaskTmax])
    
    allMaeDlyTmin = np.ma.mean(maeDlyTmin[climDivsMaskTmin])
    allMaeDlyTmax = np.ma.mean(maeDlyTmax[climDivsMaskTmax])
    
    allR2DlyTmin = np.ma.mean(r2DlyTmin[climDivsMaskTmin])
    allR2DlyTmax = np.ma.mean(r2DlyTmax[climDivsMaskTmax])
    
    #Get idxs to sort climate divisions by #
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
        
    grid,cbar = tminTmaxClimDivMap((maeNormDivsTmin,maeNormDivsTmax), 311, sidx)
    cbar.set_label_text("Normal MAE ($^\circ$C)",fontsize=fontsize)
    grid[0].set_title("Tmin")
    grid[1].set_title("Tmax")
    grid[0].set_ylabel('a.  ', rotation='horizontal')
    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allBiasNormTmin,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeNormTmax,),transform=grid[1].transAxes,fontsize=10)
    #cbar.ax.set_yticks([.3,.6,.9,1.2,1.5])
    
    grid,cbar = tminTmaxClimDivMap((maeDlyDivsTmin,maeDlyDivsTmax), 312, sidx)
    cbar.set_label_text("Daily MAE ($^\circ$C)",fontsize=fontsize)
    grid[0].set_ylabel('b.  ', rotation='horizontal')
    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmin,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmax,),transform=grid[1].transAxes,fontsize=10)
    
    grid,cbar = tminTmaxClimDivMap((r2DlyDivsTmin,r2DlyDivsTmax), 313, sidx,cmap=cm.hot)
    cbar.set_label_text(r"$\bar{R}^2$",fontsize=fontsize)
    grid[0].set_ylabel('c.  ', rotation='horizontal')
    grid[0].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmin,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmax,),transform=grid[1].transAxes,fontsize=10)
    
    fig =plt.gcf()
    #fig.set_size_inches(8*2,6*3)
    fig.set_size_inches(8,6*1.25)
    fig.subplots_adjust(hspace=0.05)
    #plt.tight_layout()
    #plt.savefig('/projects/daymet2/docs/final_writeup/climDivErrMaps.png',dpi=150)
    plt.show()

def plotInterpErrorMaps2():
    
    fontsize=12
    
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    climDivs = np.concatenate((stndaTmax.stns[NEON],stndaTmin.stns[NEON]))
    climDivs = np.unique(climDivs[np.isfinite(climDivs)])[2:]
    
    mth = 2#12
    
    maeNormTmin = stndaTmin.ds.variables['xvalfnl_mae_norm'][mth,:]
    maeNormTmax = stndaTmax.ds.variables['xvalfnl_mae_norm'][mth,:]
    maeDlyTmin = stndaTmin.ds.variables['xvalfnl_mae_dly'][mth,:]
    maeDlyTmax = stndaTmax.ds.variables['xvalfnl_mae_dly'][mth]
    r2DlyTmin = stndaTmin.ds.variables['xvalfnl_r2_dly'][mth,:]
    r2DlyTmax = stndaTmax.ds.variables['xvalfnl_r2_dly'][mth,:]
    
    biasTmin = stndaTmin.ds.variables['xvalfnl_bias_norm'][mth,:]
    biasTmax = stndaTmax.ds.variables['xvalfnl_bias_norm'][mth,:]
    
    maeNormDivsTmin = []
    maeNormDivsTmax = []
    maeDlyDivsTmin = []
    maeDlyDivsTmax = []
    r2DlyDivsTmin = []
    r2DlyDivsTmax = []
    biasDivsTmin = []
    biasDivsTmax = []
    
    for div in climDivs:
        
        maskStnsTmin = stndaTmin.stns[NEON]==div
        maskStnsTmax = stndaTmax.stns[NEON]==div
        
        maeNormDivsTmin.append(np.ma.mean(maeNormTmin[maskStnsTmin]))
        maeNormDivsTmax.append(np.ma.mean(maeNormTmax[maskStnsTmax]))
        
        maeDlyDivsTmin.append(np.ma.mean(maeDlyTmin[maskStnsTmin]))
        maeDlyDivsTmax.append(np.ma.mean(maeDlyTmax[maskStnsTmax]))
        
        r2DlyDivsTmin.append(np.ma.mean(r2DlyTmin[maskStnsTmin]))
        r2DlyDivsTmax.append(np.ma.mean(r2DlyTmax[maskStnsTmax]))
        
        biasDivsTmin.append(np.ma.mean(biasTmin[maskStnsTmin]))
        biasDivsTmax.append(np.ma.mean(biasTmax[maskStnsTmax]))
     
    maeNormDivsTmin = np.array(maeNormDivsTmin)
    maeNormDivsTmax = np.array(maeNormDivsTmax)
    maeDlyDivsTmin = np.array(maeDlyDivsTmin)
    maeDlyDivsTmax = np.array(maeDlyDivsTmax)
    r2DlyDivsTmin = np.array(r2DlyDivsTmin)
    r2DlyDivsTmax = np.array(r2DlyDivsTmax)
    biasDivsTmin = np.array(biasDivsTmin)
    biasDivsTmax = np.array(biasDivsTmax)
    
    climDivsMaskTmin = np.in1d(stndaTmin.stns[NEON], climDivs)
    climDivsMaskTmax = np.in1d(stndaTmax.stns[NEON], climDivs)
    
    print "TMIN Max Daily MAE: "+str(np.max(maeDlyDivsTmin))
    print "TMAX Max Daily MAE: "+str(np.max(maeDlyDivsTmax))
    
    print "90th Percentil Tmax Normal MAE: "+str(np.percentile(maeNormDivsTmax,90))
    
    print "TMIN BIAS: ",str(np.mean(biasTmin[climDivsMaskTmin]))
    print "TMAX BIAS: ",str(np.mean(biasTmax[climDivsMaskTmax]))
    
    allMaeNormTmin = np.ma.mean(maeNormTmin[climDivsMaskTmin])
    allMaeNormTmax = np.ma.mean(maeNormTmax[climDivsMaskTmax])
    
    allMaeDlyTmin = np.ma.mean(maeDlyTmin[climDivsMaskTmin])
    allMaeDlyTmax = np.ma.mean(maeDlyTmax[climDivsMaskTmax])
    
    allR2DlyTmin = np.ma.mean(r2DlyTmin[climDivsMaskTmin])
    allR2DlyTmax = np.ma.mean(r2DlyTmax[climDivsMaskTmax])
    
    allBiasTmin = np.ma.mean(biasTmin[climDivsMaskTmin])
    allBiasTmax = np.ma.mean(biasTmax[climDivsMaskTmax])
    
    #Get idxs to sort climate divisions by #
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
    
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
            #lines.set_edgecolors('#8C8C8C')
            lines.set_edgecolors('k')
            lines.set_linewidth(0.5)
            gridCell.add_collection(lines)
    
    print "Starting to plot bias..."
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    errAll = np.concatenate([biasDivsTmin,biasDivsTmax])
    
    clrsRed = brewer2mpl.get_map('Reds', 'Sequential', 9, reverse=False).mpl_colors
    clrsBlue = brewer2mpl.get_map('Blues', 'Sequential', 9, reverse=True).mpl_colors
    
    clrsBlue.append("grey")
    clrsBlue.append("grey")
    clrsBlue.extend(clrsRed)
    clrs = clrsBlue
    
    #clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
    #clrs = clrs.mpl_colors
    #clrs[5] = "grey"
    #levels = [-.9,-8,-.7,-.6,-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]
    levels = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.,0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9 ]
    #levels = np.linspace(-1,1,21)
    
    cmapFnl = cm.hot_r.from_list('Custom cmap', clrs[1:-1], 256)
    cmapFnl.set_over(clrs[-1])
    cmapFnl.set_under(clrs[0])
    norm = BoundaryNorm(levels, cmapFnl.N)
    
    sm = cm.ScalarMappable(norm, cmapFnl)
    sm.set_array(errAll[np.isfinite(errAll)])
    

    drawErrorMap(0,biasDivsTmin)
    m.ax = grid[0]
    m.drawmeridians([-103])
    #cbar = grid[0].cax.colorbar(sm,extend='max')
    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0],extend='both')
    cbar.set_label("MAE ($^\circ$C)",fontsize=fontsize)
    grid[0].set_title(r"$\overline{TN}_a$")
    grid[1].set_title(r"$\overline{TX}_a$")
    grid[0].set_ylabel('TopoWx')
    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allBiasTmin,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allBiasTmax,),transform=grid[1].transAxes,fontsize=10)
    drawErrorMap(1,biasDivsTmin)
    plt.show()
    
    
    
    print "Starting to plot..."
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    
    errAll = np.concatenate([maeNormDivsTmin,maeNormDivsTmax,maeDlyDivsTmin,maeDlyDivsTmax])
    print np.max(errAll)
    cmapB = brewer2mpl.get_map('Greys', 'Sequential', 8, reverse=False)#'YlOrRd'
    cmap = cmapB.get_mpl_colormap()
    cmapcolors = cmapB.mpl_colors
    
    cmapFnl = cm.hot_r.from_list('Custom cmap', cmapcolors[0:8], cmap.N)
    #cmapFnl.set_over(cmapcolors[-1])
    norm = BoundaryNorm([0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0], cmapFnl.N)
    
    sm = cm.ScalarMappable(norm, cmapFnl)
    sm.set_array(errAll[np.isfinite(errAll)])
    
    
    drawErrorMap(0,maeNormDivsTmin)
    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0])#,extend='max')
    cbar.set_label("MAE ($^\circ$C)",fontsize=fontsize)
    grid[0].set_title(r"$TN_a$")
    grid[1].set_title(r"$TX_a$")
    grid[0].set_ylabel('Normals')
    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeNormTmin,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeNormTmax,),transform=grid[1].transAxes,fontsize=10)
    drawErrorMap(1,maeNormDivsTmax)
    
    drawErrorMap(2,maeDlyDivsTmin)
    drawErrorMap(3,maeDlyDivsTmax)
    grid[2].set_ylabel('Daily')
    grid[2].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmin,),transform=grid[2].transAxes,fontsize=10)
    grid[3].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmax,),transform=grid[3].transAxes,fontsize=10)
    
#    grid = ImageGrid(cf,212,nrows_ncols=(1,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")
#    
#    r2All = np.concatenate([r2DlyDivsTmin,r2DlyDivsTmax])
#
#    cmapB = brewer2mpl.get_map('YlOrRd', 'Sequential', 9, reverse=False)
#    cmap = cmapB.get_mpl_colormap()
#    cmapcolors = cmapB.mpl_colors
#    
#    cmapFnl = cm.hot_r.from_list('Custom cmap', cmapcolors[0:9], cmap.N)
#    #cmapFnl.set_over(cmapcolors[-1])
#    norm = BoundaryNorm(np.linspace(0.92,1.0,9), cmapFnl.N)
#    
#    sm = cm.ScalarMappable(norm, cmapFnl)
#    sm.set_array(r2All[np.isfinite(r2All)])
#    
#    drawErrorMap(0,r2DlyDivsTmin)
#    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0])#,extend='max')
#    cbar.set_label(r"$\bar{R}^2$",fontsize=fontsize)
#    #grid[0].set_ylabel('Normals')
#    grid[0].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmin,),transform=grid[0].transAxes,fontsize=10)
#    grid[1].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmax,),transform=grid[1].transAxes,fontsize=10)
#    drawErrorMap(1,r2DlyDivsTmax)
      
#    grid,cbar = tminTmaxClimDivMap((maeNormDivsTmin,maeNormDivsTmax), 311, sidx)
#    cbar.set_label_text("Normal MAE ($^\circ$C)",fontsize=fontsize)
#    grid[0].set_title("Tmin")
#    grid[1].set_title("Tmax")
#    grid[0].set_ylabel('a.  ', rotation='horizontal')
#    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeNormTmin,),transform=grid[0].transAxes,fontsize=10)
#    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeNormTmax,),transform=grid[1].transAxes,fontsize=10)
#    #cbar.ax.set_yticks([.3,.6,.9,1.2,1.5])
#    
#    grid,cbar = tminTmaxClimDivMap((maeDlyDivsTmin,maeDlyDivsTmax), 312, sidx)
#    cbar.set_label_text("Daily MAE ($^\circ$C)",fontsize=fontsize)
#    grid[0].set_ylabel('b.  ', rotation='horizontal')
#    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmin,),transform=grid[0].transAxes,fontsize=10)
#    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmax,),transform=grid[1].transAxes,fontsize=10)
#    
#    grid,cbar = tminTmaxClimDivMap((r2DlyDivsTmin,r2DlyDivsTmax), 313, sidx,cmap=cm.hot)
#    cbar.set_label_text(r"$\bar{R}^2$",fontsize=fontsize)
#    grid[0].set_ylabel('c.  ', rotation='horizontal')
#    grid[0].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmin,),transform=grid[0].transAxes,fontsize=10)
#    grid[1].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmax,),transform=grid[1].transAxes,fontsize=10)
    
    fig =plt.gcf()
    #fig.set_size_inches(8*2,6*3)
    #fig.set_size_inches(8,6*1.25)
    #fig.subplots_adjust(hspace=0.05)
    #plt.tight_layout()
    #plt.savefig('/projects/daymet2/docs/final_writeup/climDivErrMaps.png',dpi=300)
    plt.show()

def plotInterpErrorMapsNcdcNormsTest():
    
    fontsize=12
    
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
    climDivs = np.concatenate((stndaTmax.stns[NEON],stndaTmin.stns[NEON]))
    climDivs = np.unique(climDivs[np.isfinite(climDivs)])[2:]
    
    stnsNorm = np.load('/projects/daymet2/station_data/ncdc_normals/norm_stns.npy')
    stnsMaskTmin = np.in1d(stndaTmin.stn_ids, stnsNorm[STN_ID], True)
    stnsMaskTmax = np.in1d(stndaTmax.stn_ids, stnsNorm[STN_ID], True)
    
    mth = 7#12
    
    maeNormTmin = stndaTmin.ds.variables['xvalfnl_mae_norm'][mth,:]
    maeNormTmax = stndaTmax.ds.variables['xvalfnl_mae_norm'][mth,:]
    maeDlyTmin = stndaTmin.ds.variables['xvalfnl_mae_dly'][mth,:]
    maeDlyTmax = stndaTmax.ds.variables['xvalfnl_mae_dly'][mth]
    r2DlyTmin = stndaTmin.ds.variables['xvalfnl_r2_dly'][mth,:]
    r2DlyTmax = stndaTmax.ds.variables['xvalfnl_r2_dly'][mth,:]
    
    biasTmin = stndaTmin.ds.variables['xvalfnl_bias_norm'][mth,:]
    biasTmax = stndaTmax.ds.variables['xvalfnl_bias_norm'][mth,:]
    
    maeNormDivsTmin = []
    maeNormDivsTmax = []
    maeDlyDivsTmin = []
    maeDlyDivsTmax = []
    r2DlyDivsTmin = []
    r2DlyDivsTmax = []
    biasDivsTmin = []
    biasDivsTmax = []
    
    for div in climDivs:
        
        maskStnsTmin = np.logical_and(stndaTmin.stns[NEON]==div,stnsMaskTmin)
        maskStnsTmax = np.logical_and(stndaTmax.stns[NEON]==div,stnsMaskTmax)
        
        maeNormDivsTmin.append(np.ma.mean(maeNormTmin[maskStnsTmin]))
        maeNormDivsTmax.append(np.ma.mean(maeNormTmax[maskStnsTmax]))
        
        maeDlyDivsTmin.append(np.ma.mean(maeDlyTmin[maskStnsTmin]))
        maeDlyDivsTmax.append(np.ma.mean(maeDlyTmax[maskStnsTmax]))
        
        r2DlyDivsTmin.append(np.ma.mean(r2DlyTmin[maskStnsTmin]))
        r2DlyDivsTmax.append(np.ma.mean(r2DlyTmax[maskStnsTmax]))
        
        biasDivsTmin.append(np.ma.mean(biasTmin[maskStnsTmin]))
        biasDivsTmax.append(np.ma.mean(biasTmax[maskStnsTmax]))
     
    maeNormDivsTmin = np.array(maeNormDivsTmin)
    maeNormDivsTmax = np.array(maeNormDivsTmax)
    maeDlyDivsTmin = np.array(maeDlyDivsTmin)
    maeDlyDivsTmax = np.array(maeDlyDivsTmax)
    r2DlyDivsTmin = np.array(r2DlyDivsTmin)
    r2DlyDivsTmax = np.array(r2DlyDivsTmax)
    biasDivsTmin = np.array(biasDivsTmin)
    biasDivsTmax = np.array(biasDivsTmax)
    
    climDivsMaskTmin = np.logical_and(np.in1d(stndaTmin.stns[NEON], climDivs),stnsMaskTmin)
    climDivsMaskTmax = np.logical_and(np.in1d(stndaTmax.stns[NEON], climDivs),stnsMaskTmax)
    
    print "TMIN Max Daily MAE: "+str(np.max(maeDlyDivsTmin))
    print "TMAX Max Daily MAE: "+str(np.max(maeDlyDivsTmax))
    
    print "90th Percentil Tmax Normal MAE: "+str(np.percentile(maeNormDivsTmax,90))
    
    print "TMIN BIAS: ",str(np.mean(biasTmin[climDivsMaskTmin]))
    print "TMAX BIAS: ",str(np.mean(biasTmax[climDivsMaskTmax]))
    
    allMaeNormTmin = np.ma.mean(maeNormTmin[climDivsMaskTmin])
    allMaeNormTmax = np.ma.mean(maeNormTmax[climDivsMaskTmax])
    
    allMaeDlyTmin = np.ma.mean(maeDlyTmin[climDivsMaskTmin])
    allMaeDlyTmax = np.ma.mean(maeDlyTmax[climDivsMaskTmax])
    
    allR2DlyTmin = np.ma.mean(r2DlyTmin[climDivsMaskTmin])
    allR2DlyTmax = np.ma.mean(r2DlyTmax[climDivsMaskTmax])
    
    allBiasTmin = np.ma.mean(biasTmin[climDivsMaskTmin])
    allBiasTmax = np.ma.mean(biasTmax[climDivsMaskTmax])
    
    #Get idxs to sort climate divisions by #
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
    
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
            #lines.set_edgecolors('#8C8C8C')
            lines.set_edgecolors('k')
            lines.set_linewidth(0.5)
            gridCell.add_collection(lines)
    
    print "Starting to plot bias..."
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(1,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    errAll = np.concatenate([biasDivsTmin,biasDivsTmax])
    
    clrsRed = brewer2mpl.get_map('Reds', 'Sequential', 9, reverse=False).mpl_colors
    clrsBlue = brewer2mpl.get_map('Blues', 'Sequential', 9, reverse=True).mpl_colors
    
    clrsBlue.append("grey")
    clrsBlue.append("grey")
    clrsBlue.extend(clrsRed)
    clrs = clrsBlue
    
    #clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
    #clrs = clrs.mpl_colors
    #clrs[5] = "grey"
    #levels = [-.9,-8,-.7,-.6,-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]
    levels = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.,0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9 ]
    #levels = np.linspace(-1,1,21)
    
    cmapFnl = cm.hot_r.from_list('Custom cmap', clrs[1:-1], 256)
    cmapFnl.set_over(clrs[-1])
    cmapFnl.set_under(clrs[0])
    norm = BoundaryNorm(levels, cmapFnl.N)
    
    sm = cm.ScalarMappable(norm, cmapFnl)
    sm.set_array(errAll[np.isfinite(errAll)])
    

    drawErrorMap(0,biasDivsTmin)
    m.ax = grid[0]
    m.drawmeridians([-103])
    #cbar = grid[0].cax.colorbar(sm,extend='max')
    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0],extend='both')
    cbar.set_label("MAE ($^\circ$C)",fontsize=fontsize)
    grid[0].set_title(r"$\overline{TN}_a$")
    grid[1].set_title(r"$\overline{TX}_a$")
    grid[0].set_ylabel('TopoWx')
    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allBiasTmin,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allBiasTmax,),transform=grid[1].transAxes,fontsize=10)
    drawErrorMap(1,biasDivsTmin)
    plt.show()
    
    
    
    print "Starting to plot..."
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    
    errAll = np.concatenate([maeNormDivsTmin,maeNormDivsTmax,maeDlyDivsTmin,maeDlyDivsTmax])
    print np.max(errAll)
    cmapB = brewer2mpl.get_map('Greys', 'Sequential', 8, reverse=False)#'YlOrRd'
    cmap = cmapB.get_mpl_colormap()
    cmapcolors = cmapB.mpl_colors
    
    cmapFnl = cm.hot_r.from_list('Custom cmap', cmapcolors[0:8], cmap.N)
    #cmapFnl.set_over(cmapcolors[-1])
    norm = BoundaryNorm([0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0], cmapFnl.N)
    
    sm = cm.ScalarMappable(norm, cmapFnl)
    sm.set_array(errAll[np.isfinite(errAll)])
    
    
    drawErrorMap(0,maeNormDivsTmin)
    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0])#,extend='max')
    cbar.set_label("MAE ($^\circ$C)",fontsize=fontsize)
    grid[0].set_title(r"$TN_a$")
    grid[1].set_title(r"$TX_a$")
    grid[0].set_ylabel('Normals')
    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeNormTmin,),transform=grid[0].transAxes,fontsize=10)
    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeNormTmax,),transform=grid[1].transAxes,fontsize=10)
    drawErrorMap(1,maeNormDivsTmax)
    
    drawErrorMap(2,maeDlyDivsTmin)
    drawErrorMap(3,maeDlyDivsTmax)
    grid[2].set_ylabel('Daily')
    grid[2].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmin,),transform=grid[2].transAxes,fontsize=10)
    grid[3].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmax,),transform=grid[3].transAxes,fontsize=10)
    
#    grid = ImageGrid(cf,212,nrows_ncols=(1,2),axes_pad=0.1,cbar_mode="single",cbar_location="right",cbar_size="3%")
#    
#    r2All = np.concatenate([r2DlyDivsTmin,r2DlyDivsTmax])
#
#    cmapB = brewer2mpl.get_map('YlOrRd', 'Sequential', 9, reverse=False)
#    cmap = cmapB.get_mpl_colormap()
#    cmapcolors = cmapB.mpl_colors
#    
#    cmapFnl = cm.hot_r.from_list('Custom cmap', cmapcolors[0:9], cmap.N)
#    #cmapFnl.set_over(cmapcolors[-1])
#    norm = BoundaryNorm(np.linspace(0.92,1.0,9), cmapFnl.N)
#    
#    sm = cm.ScalarMappable(norm, cmapFnl)
#    sm.set_array(r2All[np.isfinite(r2All)])
#    
#    drawErrorMap(0,r2DlyDivsTmin)
#    cbar = plt.colorbar(sm, cax=grid[0].cax, ax=grid[0])#,extend='max')
#    cbar.set_label(r"$\bar{R}^2$",fontsize=fontsize)
#    #grid[0].set_ylabel('Normals')
#    grid[0].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmin,),transform=grid[0].transAxes,fontsize=10)
#    grid[1].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmax,),transform=grid[1].transAxes,fontsize=10)
#    drawErrorMap(1,r2DlyDivsTmax)
      
#    grid,cbar = tminTmaxClimDivMap((maeNormDivsTmin,maeNormDivsTmax), 311, sidx)
#    cbar.set_label_text("Normal MAE ($^\circ$C)",fontsize=fontsize)
#    grid[0].set_title("Tmin")
#    grid[1].set_title("Tmax")
#    grid[0].set_ylabel('a.  ', rotation='horizontal')
#    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeNormTmin,),transform=grid[0].transAxes,fontsize=10)
#    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeNormTmax,),transform=grid[1].transAxes,fontsize=10)
#    #cbar.ax.set_yticks([.3,.6,.9,1.2,1.5])
#    
#    grid,cbar = tminTmaxClimDivMap((maeDlyDivsTmin,maeDlyDivsTmax), 312, sidx)
#    cbar.set_label_text("Daily MAE ($^\circ$C)",fontsize=fontsize)
#    grid[0].set_ylabel('b.  ', rotation='horizontal')
#    grid[0].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmin,),transform=grid[0].transAxes,fontsize=10)
#    grid[1].text(0.025,0.075,"Overall\nMAE: %.2f$^\circ$C"%(allMaeDlyTmax,),transform=grid[1].transAxes,fontsize=10)
#    
#    grid,cbar = tminTmaxClimDivMap((r2DlyDivsTmin,r2DlyDivsTmax), 313, sidx,cmap=cm.hot)
#    cbar.set_label_text(r"$\bar{R}^2$",fontsize=fontsize)
#    grid[0].set_ylabel('c.  ', rotation='horizontal')
#    grid[0].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmin,),transform=grid[0].transAxes,fontsize=10)
#    grid[1].text(0.025,0.075,"Overall\n"+r"$\bar{R}^2$: %.2f"%(allR2DlyTmax,),transform=grid[1].transAxes,fontsize=10)
    
    fig =plt.gcf()
    #fig.set_size_inches(8*2,6*3)
    #fig.set_size_inches(8,6*1.25)
    #fig.subplots_adjust(hspace=0.05)
    #plt.tight_layout()
    #plt.savefig('/projects/daymet2/docs/final_writeup/climDivErrMaps.png',dpi=300)
    plt.show()

def pickleClimDivsMpl():
    
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDiv')
    shapes = shpClimDivArea.shapes()
    records = shpClimDivArea.records()
    
    ax = plt.subplot(111)
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    m.drawcountries(linewidth=0.5,color='white')
    
    
    lineCollects = pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle'))
    for lines in lineCollects:
        lines.set_facecolors(cm.jet(np.random.rand(1)))
        lines.set_edgecolors('k')
        lines.set_linewidth(0.1)
        ax.add_collection(lines)
    plt.show()
        
    
#    lineCollects = []
#    for record, shape in zip(records,shapes):
#        lons,lats = zip(*shape.points)
#        data = np.array(m(lons, lats)).T
#        #data = np.array([lons, lats]).T
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
#        lineCollects.append(lines)
#    pickle.dump(lineCollects,open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle','wb'))
    

def plotKrigParams():
    
    #Load station counts per climate division into dictionary
    stndtype = copy(stnData.DTYPE_STN_DFLT)
    stndtype.append(('xval_overall_mae',np.float64))
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc', 'tmax',stn_dtype=stndtype)
    climDivs = np.unique(stnda.stns[NEON][np.isfinite(stnda.stns[NEON])])[2:]
    krigParams = []
    for div in climDivs:
        #krigParams.append(np.mean(np.abs(stnda.stns['xval_overall_bias'][np.logical_and(np.isfinite(stnda.stns['xval_overall_bias']),stnda.stns[NEON]==div)])))
        krigParams.append(np.mean(stnda.stns['xval_overall_mae'][np.logical_and(np.isfinite(stnda.stns['xval_overall_mae']),stnda.stns[NEON]==div)]))
    krigParams = np.array(krigParams)
    
    #Load the climate division areas
    shpClimDivArea = shapefile.Reader(r'/projects/daymet2/dem/climate_divisions/ClimDivAlbersArea')
    recs = shpClimDivArea.records()
    climDivsShp = np.array([float(aRec[5]) for aRec in recs])
    sidx = np.argsort(climDivsShp)
    climDivsShp = climDivsShp[sidx]
        
    #Create a normalized colormap
    norm = Normalize(np.min(krigParams), np.max(krigParams))
    #norm = Normalize(0, 2)
    cmap = cm.jet
    sm = cm.ScalarMappable(norm, cmap)
    sm.set_array(krigParams)
    #cmap.set_over(sm.to_rgba(4))
    #Load the climate division Line Collections for matplotlib
    lineCollect = np.array(pickle.load(open('/projects/daymet2/dem/climate_divisions/ClimDivLineCollections.pickle')))
    lineCollect = lineCollect[sidx]
    
    #Setup map
    ax = plt.subplot(111)
    m = Basemap(resolution='c',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5)
    m.drawcountries(linewidth=0.5,color='white')
    
    #Put Climate Divisions on map
    for x in np.arange(lineCollect.size):
    
        lines = lineCollect[x]
        lines.set_facecolors(sm.to_rgba(krigParams[x]))
        lines.set_edgecolors('k')
        lines.set_linewidth(0.5)
        ax.add_collection(lines)
    
    m.colorbar(sm,extend='neither')
    plt.show()   

def impResiduals():
    dbPathInfill = '/projects/daymet2/station_data/infill/infill_20130518/infill_tmax.nc'
    dbPathSerial = '/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc'
    
    dsI = Dataset(dbPathInfill)
    dsS = Dataset(dbPathSerial)
    
    stnidS = dsS.variables['stn_id'][:].astype("<S16")
    maskBad = dsS.variables['bad'][:] == 1
    maskOut = dsS.variables['mask'][:].mask
    maskCA = np.char.startswith(stnidS, "GHCN_CA")
    
    stnidSkip = stnidS[np.logical_or(np.logical_or(maskBad,maskOut),maskCA)]
    
    stnidI = dsI.variables['stn_id'][:].astype("<S16")
    
    fnlMask = np.logical_not(np.in1d(stnidI, stnidSkip, True))
    
    stnPfxs = ['GHCN','RAWS','SNOTEL',]
    
    for apfx in stnPfxs:
        
        aMask = np.logical_and(np.char.startswith(stnidI, apfx),fnlMask)
        print apfx,np.sum(aMask)
        print "Bias: "+str(np.mean(dsI.variables['bias'][aMask]))
        print "MAB: "+str(np.mean(np.abs(dsI.variables['bias'][aMask])))
        print "MAE: "+str(np.mean(dsI.variables['mae'][aMask]))
        print "R2: "+str(np.mean(dsI.variables['r2'][aMask]))

def runningMean(a,n=5):
    sideWin = (n-1)/2
    
    rm = np.ma.masked_array(np.zeros(a.size))
    for x in np.arange(a.size):
        rm[x]=np.ma.mean(a[x-sideWin:x+sideWin+1])
    rm[0:sideWin] = np.ma.masked
    rm[-sideWin:] = np.ma.masked
    
    return rm
      
def plotSntlHomogDifs():
    tairVar = 'tmax'
    stndaHomog = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc',stnDtype=DTYPE_STN_BASIC)
    stndaRaw = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    
    sntlIds = np.loadtxt('/projects/daymet2/docs/final_writeup/stnsForHomogDifPlot/tminSNTL.txt', dtype=np.str)
    
    tairAggGHCN = ushcn.TairAggregate(stndaHomog.days)
    uYrs = np.unique(stndaHomog.days[YEAR])
    baseMaskRS = uYrs >= 1990
#        
    print "SNOTEL Homog..."
    obsHomogSNTL = stndaHomog.load_all_stn_obs_var(sntlIds, tairVar)[0]
    obsHomogSNTL = np.ma.masked_array(obsHomogSNTL,np.isnan(obsHomogSNTL))
    obsHomogSNTL = tairAggGHCN.dailyToAnn(obsHomogSNTL)
    obsHomogSNTL = obsHomogSNTL - np.ma.mean(obsHomogSNTL[baseMaskRS,:],axis=0)
    obsHomogSNTL[~baseMaskRS,:] = np.ma.masked
    print "SNOTEL Raw..."
    obsRawSNTL = stndaRaw.load_all_stn_obs_var(sntlIds, tairVar)[0]
    obsRawSNTL = np.ma.masked_array(obsRawSNTL,np.isnan(obsRawSNTL))
    obsRawSNTL = tairAggGHCN.dailyToAnn(obsRawSNTL)
    obsRawSNTL = obsRawSNTL - np.ma.mean(obsRawSNTL[baseMaskRS,:],axis=0)
    obsRawSNTL[~baseMaskRS,:] = np.ma.masked
    
    obsHomogSNTL = obsHomogSNTL[baseMaskRS,:]
    obsRawSNTL = obsRawSNTL[baseMaskRS,:]
    annDifsSNTL = obsHomogSNTL - obsRawSNTL
    
    yrs = np.arange(1990,2013)
    print runningMean(np.ma.mean(obsRawSNTL,axis=1))
    print stats.linregress(yrs,np.ma.mean(obsRawSNTL,axis=1))
    print stats.linregress(yrs,np.ma.mean(obsHomogSNTL,axis=1))
    plt.plot(np.ma.mean(obsRawSNTL,axis=1))
    plt.plot(np.ma.mean(obsHomogSNTL,axis=1))
    plt.show()
    
#    plt.plot(np.ma.mean(annDifsUSHCN,axis=1),color='#E41A1C',lw=2)
#    plt.plot(np.ma.mean(annDifsGHCN,axis=1),color='#377EB8',lw=2)
#    plt.plot(np.ma.mean(annDifsSNTL,axis=1),color='#4DAF4A',lw=2)
#    print np.ma.mean(annDifsSNTL,axis=1)
#    plt.plot(np.ma.mean(annDifsRAW,axis=1),color='#984EA3',lw=2)
##    plt.plot(np.ma.mean(obsUsRaw,axis=1))
##    plt.plot(np.ma.mean(obsAllRawGHCN,axis=1))
##    plt.plot(np.ma.mean(obsRawSNTL,axis=1))
##    plt.plot(np.ma.mean(obsRawRAWS,axis=1))
#    ax = plt.gca()
#    ax.set_axisbelow(True)
#    ax.yaxis.grid(color='gray', linestyle='dashed')
#    ax.xaxis.grid(color='gray', linestyle='dashed')
#    
#    xtickYrs = np.array([1950,1960,1970,1980,1990,2000,2010])
#    plt.xticks(xtickYrs-1948,[str(x) for x in xtickYrs],fontsize=17)#fontsize=17
#    ylocs,ylabs = plt.yticks()
#    plt.yticks(ylocs,fontsize=17)
#    plt.xlabel('Year',fontsize=17)
#    plt.ylabel('Anomaly Difference ($^\circ$C)',fontsize=17)
#    plt.legend(('USHCN (NCDC)','GHCN-D','SNOTEL','RAWS'),loc=2,fontsize=17)
#    #fig = plt.gcf()
#    #fig.set_size_inches(8*1.5,6*1.5)
#    #plt.savefig('/projects/daymet2/docs/final_writeup/homogDiftmin.png',dpi=300)
#    plt.show()

def rawVsHomogTrends():
    tairVar = 'tmax'
    stndaHomog = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc',stnDtype=DTYPE_STN_BASIC)
    stndaRaw = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    
    ghcnIds = np.loadtxt('/projects/daymet2/docs/final_writeup/stnsForHomogDifPlot/tmaxGHCN.txt', dtype=np.str)
    rawsIds = np.loadtxt('/projects/daymet2/docs/final_writeup/stnsForHomogDifPlot/tmaxRAWS.txt', dtype=np.str)
    sntlIds = np.loadtxt('/projects/daymet2/docs/final_writeup/stnsForHomogDifPlot/tmaxSNTL.txt', dtype=np.str)
    maskHCN = ushcn.buildGhcnUShcnMask(ghcnIds,'/projects/daymet2/station_data/ghcn/ghcnd-stations.txt')
    ghcnIds = ghcnIds[maskHCN]
   
    ghcnStns = stndaHomog.stns[np.in1d(stndaHomog.stn_ids, ghcnIds, True)]
    matchUSIds = ushcn.matchGhcnToUshcn(ghcnStns, stndaUS.stns)
    ghcnStns = ghcnStns[matchUSIds != "NONE"]
    ushcnIds = matchUSIds[matchUSIds != "NONE"]
    ghcnIds = ghcnStns[STN_ID]
    
    tairAggGHCN = ushcn.TairAggregate(stndaHomog.days)
    uYrs = np.unique(stndaHomog.days[YEAR])
    baseMask = np.logical_and(uYrs >= 1948,uYrs <= 2012)
    baseMaskRS = uYrs >= 1990
#        
    print "GHCN Homog...."
    obsAllHomogGHCN = stndaHomog.load_all_stn_obs_var(ghcnIds, tairVar)[0]
    obsAllHomogGHCN = np.ma.masked_array(obsAllHomogGHCN,np.isnan(obsAllHomogGHCN))
    obsAllHomogGHCN = tairAggGHCN.dailyToAnn(obsAllHomogGHCN)
    obsAllHomogGHCN = obsAllHomogGHCN - np.ma.mean(obsAllHomogGHCN[baseMask,:],axis=0)
    print "GHCN Raw...."
    obsAllRawGHCN = stndaRaw.load_all_stn_obs_var(ghcnIds, tairVar)[0]
    obsAllRawGHCN = np.ma.masked_array(obsAllRawGHCN,np.isnan(obsAllRawGHCN))
    obsAllRawGHCN = tairAggGHCN.dailyToAnn(obsAllRawGHCN)
    obsAllRawGHCN = obsAllRawGHCN - np.ma.mean(obsAllRawGHCN[baseMask,:],axis=0) 
    print "USHCN Homog..."
    obsFLs = stndaUS.loadObs(ushcnIds, 'FLs.52i_tmax')
    obsUsRaw = stndaUS.loadObs(ushcnIds, 'raw_tmax')
    obsFLs = np.ma.masked_array(obsFLs,obsUsRaw.mask)
    obsFLs = tairAggGHCN.mthlyToAnn(obsFLs)
    obsFLs = obsFLs - np.ma.mean(obsFLs[baseMask,:],axis=0)
    print "USHCN Raw..."
    obsUsRaw = tairAggGHCN.mthlyToAnn(obsUsRaw)
    obsUsRaw = obsUsRaw - np.ma.mean(obsUsRaw[baseMask,:],axis=0)
    print "SNOTEL Homog..."
    obsHomogSNTL = stndaHomog.load_all_stn_obs_var(sntlIds, tairVar)[0]
    obsHomogSNTL = np.ma.masked_array(obsHomogSNTL,np.isnan(obsHomogSNTL))
    obsHomogSNTL = tairAggGHCN.dailyToAnn(obsHomogSNTL)
    obsHomogSNTL = obsHomogSNTL - np.ma.mean(obsHomogSNTL[baseMaskRS,:],axis=0)
    obsHomogSNTL[~baseMaskRS,:] = np.ma.masked
    print "SNOTEL Raw..."
    obsRawSNTL = stndaRaw.load_all_stn_obs_var(sntlIds, tairVar)[0]
    obsRawSNTL = np.ma.masked_array(obsRawSNTL,np.isnan(obsRawSNTL))
    obsRawSNTL = tairAggGHCN.dailyToAnn(obsRawSNTL)
    obsRawSNTL = obsRawSNTL - np.ma.mean(obsRawSNTL[baseMaskRS,:],axis=0)
    obsRawSNTL[~baseMaskRS,:] = np.ma.masked
    print "RAWS Homog..."
    obsHomogRAWS = stndaHomog.load_all_stn_obs_var(rawsIds, tairVar)[0]
    obsHomogRAWS = np.ma.masked_array(obsHomogRAWS,np.isnan(obsHomogRAWS))
    obsHomogRAWS = tairAggGHCN.dailyToAnn(obsHomogRAWS)
    obsHomogRAWS = obsHomogRAWS - np.ma.mean(obsHomogRAWS[baseMaskRS,:],axis=0)
    obsHomogRAWS[~baseMaskRS,:] = np.ma.masked
    print "RAWS Raw..."
    obsRawRAWS = stndaRaw.load_all_stn_obs_var(rawsIds, tairVar)[0]
    obsRawRAWS = np.ma.masked_array(obsRawRAWS,np.isnan(obsRawRAWS))
    obsRawRAWS = tairAggGHCN.dailyToAnn(obsRawRAWS)
    obsRawRAWS = obsRawRAWS - np.ma.mean(obsRawRAWS[baseMaskRS,:],axis=0)
    obsRawRAWS[~baseMaskRS,:] = np.ma.masked
        
    #USHCN GHCN Trends
    yrs = np.arange(1948,2013)
    print "###########################################"
    print "USHCN"
    print "RAW",stats.linregress(yrs,np.ma.mean(obsUsRaw,axis=1))
    print "HOMOG",stats.linregress(yrs,np.ma.mean(obsFLs,axis=1))
    print "###########################################"
    print "GHCN-D"
    print "RAW",stats.linregress(yrs,np.ma.mean(obsAllRawGHCN,axis=1))
    print "HOMOG",stats.linregress(yrs,np.ma.mean(obsAllHomogGHCN,axis=1))
    
    yrs = np.arange(1990,2013)
    obsHomogSNTL = obsHomogSNTL[baseMaskRS,:]
    obsRawSNTL = obsRawSNTL[baseMaskRS,:]
    obsHomogRAWS = obsHomogRAWS[baseMaskRS,:]
    obsRawRAWS = obsRawRAWS[baseMaskRS,:]
    print "###########################################"
    print "SNOTEL"
    print "RAW",stats.linregress(yrs,np.ma.mean(obsRawSNTL,axis=1))
    print "HOMOG",stats.linregress(yrs,np.ma.mean(obsHomogSNTL,axis=1))
    print "###########################################"
    print "RAWS"
    print "RAW",stats.linregress(yrs,np.ma.mean(obsRawRAWS,axis=1))
    print "HOMOG",stats.linregress(yrs,np.ma.mean(obsHomogRAWS,axis=1))    

def plotHomogDifs():
    tairVar = 'tmin'
    stndaHomog = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc',stnDtype=DTYPE_STN_BASIC)
    stndaRaw = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    
    ghcnIds = np.loadtxt('/projects/daymet2/docs/final_writeup/stnsForHomogDifPlot/tminGHCN.txt', dtype=np.str)
    rawsIds = np.loadtxt('/projects/daymet2/docs/final_writeup/stnsForHomogDifPlot/tminRAWS.txt', dtype=np.str)
    sntlIds = np.loadtxt('/projects/daymet2/docs/final_writeup/stnsForHomogDifPlot/tminSNTL.txt', dtype=np.str)
    maskHCN = ushcn.buildGhcnUShcnMask(ghcnIds,'/projects/daymet2/station_data/ghcn/ghcnd-stations.txt')
    ghcnIds = ghcnIds[maskHCN]
   
    ghcnStns = stndaHomog.stns[np.in1d(stndaHomog.stn_ids, ghcnIds, True)]
    matchUSIds = ushcn.matchGhcnToUshcn(ghcnStns, stndaUS.stns)
    ghcnStns = ghcnStns[matchUSIds != "NONE"]
    print ghcnStns.size
    ushcnIds = matchUSIds[matchUSIds != "NONE"]
    ghcnIds = ghcnStns[STN_ID]
    
    tairAggGHCN = ushcn.TairAggregate(stndaHomog.days)
    uYrs = np.unique(stndaHomog.days[YEAR])
    baseMask = np.logical_and(uYrs >= 1948,uYrs <= 2012)
    baseMaskRS = uYrs >= 1990
#        
    print "GHCN Homog...."
    obsAllHomogGHCN = stndaHomog.load_all_stn_obs_var(ghcnIds, tairVar)[0]
    obsAllHomogGHCN = np.ma.masked_array(obsAllHomogGHCN,np.isnan(obsAllHomogGHCN))
    obsAllHomogGHCN = tairAggGHCN.dailyToAnn(obsAllHomogGHCN)
    obsAllHomogGHCN = obsAllHomogGHCN - np.ma.mean(obsAllHomogGHCN[baseMask,:],axis=0)
    print "GHCN Raw...."
    obsAllRawGHCN = stndaRaw.load_all_stn_obs_var(ghcnIds, tairVar)[0]
    obsAllRawGHCN = np.ma.masked_array(obsAllRawGHCN,np.isnan(obsAllRawGHCN))
    obsAllRawGHCN = tairAggGHCN.dailyToAnn(obsAllRawGHCN)
    obsAllRawGHCN = obsAllRawGHCN - np.ma.mean(obsAllRawGHCN[baseMask,:],axis=0) 
    print "USHCN Homog..."
    obsFLs = stndaUS.loadObs(ushcnIds, 'FLs.52i_tmin')
    obsUsRaw = stndaUS.loadObs(ushcnIds, 'raw_tmin')
    obsFLs = np.ma.masked_array(obsFLs,obsUsRaw.mask)
    obsFLs = tairAggGHCN.mthlyToAnn(obsFLs)
    obsFLs = obsFLs - np.ma.mean(obsFLs[baseMask,:],axis=0)
    print "USHCN Raw..."
    obsUsRaw = tairAggGHCN.mthlyToAnn(obsUsRaw)
    obsUsRaw = obsUsRaw - np.ma.mean(obsUsRaw[baseMask,:],axis=0)
    print "SNOTEL Homog..."
    obsHomogSNTL = stndaHomog.load_all_stn_obs_var(sntlIds, tairVar)[0]
    obsHomogSNTL = np.ma.masked_array(obsHomogSNTL,np.isnan(obsHomogSNTL))
    obsHomogSNTL = tairAggGHCN.dailyToAnn(obsHomogSNTL)
    obsHomogSNTL = obsHomogSNTL - np.ma.mean(obsHomogSNTL[baseMaskRS,:],axis=0)
    obsHomogSNTL[~baseMaskRS,:] = np.ma.masked
    print "SNOTEL Raw..."
    obsRawSNTL = stndaRaw.load_all_stn_obs_var(sntlIds, tairVar)[0]
    obsRawSNTL = np.ma.masked_array(obsRawSNTL,np.isnan(obsRawSNTL))
    obsRawSNTL = tairAggGHCN.dailyToAnn(obsRawSNTL)
    obsRawSNTL = obsRawSNTL - np.ma.mean(obsRawSNTL[baseMaskRS,:],axis=0)
    obsRawSNTL[~baseMaskRS,:] = np.ma.masked
    print "RAWS Homog..."
    obsHomogRAWS = stndaHomog.load_all_stn_obs_var(rawsIds, tairVar)[0]
    obsHomogRAWS = np.ma.masked_array(obsHomogRAWS,np.isnan(obsHomogRAWS))
    obsHomogRAWS = tairAggGHCN.dailyToAnn(obsHomogRAWS)
    obsHomogRAWS = obsHomogRAWS - np.ma.mean(obsHomogRAWS[baseMaskRS,:],axis=0)
    obsHomogRAWS[~baseMaskRS,:] = np.ma.masked
    print "RAWS Raw..."
    obsRawRAWS = stndaRaw.load_all_stn_obs_var(rawsIds, tairVar)[0]
    obsRawRAWS = np.ma.masked_array(obsRawRAWS,np.isnan(obsRawRAWS))
    obsRawRAWS = tairAggGHCN.dailyToAnn(obsRawRAWS)
    obsRawRAWS = obsRawRAWS - np.ma.mean(obsRawRAWS[baseMaskRS,:],axis=0)
    obsRawRAWS[~baseMaskRS,:] = np.ma.masked
    
    annDifsGHCN = obsAllHomogGHCN - obsAllRawGHCN
    annDifsUSHCN = obsFLs - obsUsRaw
    annDifsSNTL = obsHomogSNTL - obsRawSNTL
    annDifsRAW = obsHomogRAWS - obsRawRAWS
    
    plt.plot(np.ma.mean(obsRawSNTL,axis=1))
    plt.plot(np.ma.mean(obsHomogSNTL,axis=1))
    plt.show()
    
    plt.plot(np.ma.mean(annDifsUSHCN,axis=1),color='#E41A1C',lw=2)
    plt.plot(np.ma.mean(annDifsGHCN,axis=1),color='#377EB8',lw=2)
    plt.plot(np.ma.mean(annDifsSNTL,axis=1),color='#4DAF4A',lw=2)
    print np.ma.mean(annDifsSNTL,axis=1)
    plt.plot(np.ma.mean(annDifsRAW,axis=1),color='#984EA3',lw=2)
#    plt.plot(np.ma.mean(obsUsRaw,axis=1))
#    plt.plot(np.ma.mean(obsAllRawGHCN,axis=1))
#    plt.plot(np.ma.mean(obsRawSNTL,axis=1))
#    plt.plot(np.ma.mean(obsRawRAWS,axis=1))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    
    xtickYrs = np.array([1950,1960,1970,1980,1990,2000,2010])
    plt.xticks(xtickYrs-1948,[str(x) for x in xtickYrs],fontsize=17)#fontsize=17
    ylocs,ylabs = plt.yticks()
    plt.yticks(ylocs,fontsize=17)
    plt.xlabel('Year',fontsize=17)
    plt.ylabel('Anomaly Difference ($^\circ$C)',fontsize=17)
    plt.legend(('USHCN (NCDC)','GHCN-D','SNOTEL','RAWS'),loc=2,fontsize=17)
    #fig = plt.gcf()
    #fig.set_size_inches(8*1.5,6*1.5)
    #plt.savefig('/projects/daymet2/docs/final_writeup/homogDiftmin.png',dpi=300)
    plt.show()

def topoWxVsUSHCN():
    
    tairVar = 'tmin'
    
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    stnsUS = stndaUS.stns
        
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmin.nc','tmin',stn_dtype=DTYPE_STN_BASIC)
    stndaRAW = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serial_tmin.nc','tmin',stn_dtype=DTYPE_STN_BASIC)
    uYrs = np.unique(stnda.days[YEAR])
    hcnMask = ushcn.buildGhcnUShcnMask(stnda.stn_ids,'/projects/daymet2/station_data/ghcn/ghcnd-stations.txt')
    stnsG = stnda.stns[hcnMask]
    
    tairAgg = ushcn.TairAggregate(stnda.days)
    
    matchUSIds = ushcn.matchGhcnToUshcn(stnsG, stnsUS)
    print "# of stations that couldn't match: "+str(np.sum(matchUSIds=='NONE'))
    
    stns = stnsG[matchUSIds != 'NONE']
    matchUSIds = matchUSIds[matchUSIds != 'NONE']
    
#    stns = stns[0:300]
#    matchUSIds = matchUSIds[0:300]
    
    annDifsHCN = np.zeros((uYrs.size,stns.size))
    annDifsHCN = np.ma.masked_array(annDifsHCN,np.isnan(annDifsHCN))
    
    annDifsTWX = np.zeros((uYrs.size,stns.size))
    annDifsTWX = np.ma.masked_array(annDifsTWX,np.isnan(annDifsTWX))
    
    baseMask = np.logical_and(uYrs >= 1961,uYrs <= 1990)
    
    stchk = status_check(stns.size,50)
    for astn,usId,x in zip(stns,matchUSIds,np.arange(stns.size)):
        
        obsTwx = tairAgg.dailyToAnn(stnda.load_obs(astn[STN_ID]))
        #obsTwxRAW = tairAgg.dailyToAnn(stndaRAW.load_obs(astn[STN_ID]))
        
        obsFLs = stndaUS.loadObs(usId, 'FLs.52i_tmin')
        obsFLs = tairAgg.mthlyToAnn(obsFLs)
        
        obsRaw = stndaUS.loadObs(usId, 'raw_tmin')
        obsRaw = tairAgg.mthlyToAnn(obsRaw)
        
        
        obsTwx = obsTwx - np.mean(obsTwx[baseMask])
        obsFLs = obsFLs - np.mean(obsFLs[baseMask])
        obsRaw = obsRaw - np.mean(obsRaw[baseMask])
#        
#        obsTob = stndaUS.loadObs(usId, 'tob_tmax')
#        obsTob = tairAgg.mthlyToAnn(obsTob)
#             
#        difTobRaw = obsTob - obsRaw
#           
#        #difFlsTob = obsFLs - obsTob
#        #obsUS = obsRaw+difFlsTob
#        
#        obsUS = obsFLs-difTobRaw
        
        annDifsHCN[:,x] = obsFLs-obsRaw#obsFLs-obsRaw#obsTwx-obsFLs
        annDifsTWX[:,x] = obsTwx-obsRaw
        stchk.increment()


    annDifsHCN = annDifsHCN[:,np.sum(annDifsHCN.mask,axis=0)==0]
    annDifsTWX = annDifsTWX[:,np.sum(annDifsTWX.mask,axis=0)==0]
    print annDifsHCN.shape,annDifsTWX.shape  

    avgAnnDifHCN = np.ma.mean(annDifsHCN,axis=1)
    avgAnnDifTWX = np.ma.mean(annDifsTWX,axis=1)
    
    plt.plot(avgAnnDifHCN)
    plt.plot(avgAnnDifTWX)
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.legend(('HCN','TWX'))
    plt.show()


def anomalyMap():
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
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
    
    anomRaw = np.zeros((uYrs.size,stns.size))
    anomRaw = np.ma.masked_array(anomRaw,np.isnan(anomRaw))
    
    for astn,x in zip(stns,np.arange(stns.size)):
        obsFLs = stndaUS.loadObs(astn[STN_ID], 'FLs.52i_tmax')
        obsFLs = tairAgg.mthlyToAnn(obsFLs)
        
        obsRaw = stndaUS.loadObs(astn[STN_ID], 'raw_tmax')
        obsRaw = tairAgg.mthlyToAnn(obsRaw)
        
        obsFLs = obsFLs - np.ma.mean(obsFLs[baseMask])
        obsRaw = obsRaw - np.ma.mean(obsRaw[baseMask])
        
        anomFLs[:,x] = obsFLs
        anomRaw[:,x] = obsRaw
        
    ###########################################
    stndaTWX = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    maskHCN = ushcn.buildGhcnUShcnMask(stndaTWX.stns[STN_ID],'/projects/daymet2/station_data/ghcn/ghcnd-stations.txt')
    stnsTWX = stndaTWX.stns[maskHCN]
    
    homog = HomogRawDaily(stndaTWX, '/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/monthly/FLs.r00/',
                          '/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/output/PhaAdjTmax.log','tmax')
    
    anomTWX = np.zeros((uYrs.size,stnsTWX.size))
    anomTWX = np.ma.masked_array(anomTWX,np.isnan(anomTWX))
    
    for astn,x in zip(stnsTWX,np.arange(stnsTWX.size)):
        obsTWX = homog.getFLs(astn[STN_ID])
        obsTWX = tairAgg.mthlyToAnn(obsTWX)
        obsTWX = obsTWX - np.ma.mean(obsTWX[baseMask])
        anomTWX[:,x] = obsTWX
    ###########################################
        
#    stndaTWX = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc','tmax')
#    obsTWX = stndaTWX.ds.variables['tmax_ann'][:]
#    anomTWX = obsTWX - np.mean(obsTWX,axis=0)
#    
#    maskHCN = ushcn.buildGhcnUShcnMask(stndaTWX.stns[STN_ID],'/projects/daymet2/station_data/ghcn/ghcnd-stations.txt')
#    stn_mask = np.logical_and(np.logical_and(np.isfinite(stndaTWX.stns[MASK]),np.isnan(stndaTWX.stns[BAD])),maskHCN)  
#    stnsTWX = stndaTWX.stns[stn_mask]
#    anomTWX = anomTWX[:,stn_mask]
    
    rad = 4.0*np.arctan(1.0)/180.0
    re = 6371220.0
    rr  = re*rad

    latGrid,lonGrid = dsGrid.getCoordMeshGrid()
    latGrid = latGrid.ravel()
    lonGrid = lonGrid.ravel()
    
    wgts = np.cos(latGrid*rad)
    
    yGrid,xGrid = dsGrid.getCoordGrid1d()
    yGrid = np.sort(yGrid)
    
    difsAnom = np.zeros(uYrs.size)
    difsAnom2 = np.zeros(uYrs.size)
    
    avgAnomFLs = np.zeros(uYrs.size)
    avgAnomTWX = np.zeros(uYrs.size)
    avgAnomRAW = np.zeros(uYrs.size)
    
    for i in np.arange(uYrs.size):
    
        print uYrs[i]
        anomGrid = griddata(stns[LON],stns[LAT],anomFLs[i,:],xGrid,yGrid)
        anomGrid = np.flipud(anomGrid)
        anomGrid.mask = np.logical_or(gridMask,anomGrid.mask)
        anomFlsYr = np.ma.average(anomGrid.ravel(),weights=wgts)
        
        anomGrid = griddata(stns[LON],stns[LAT],anomRaw[i,:],xGrid,yGrid)
        anomGrid = np.flipud(anomGrid)
        anomGrid.mask = np.logical_or(gridMask,anomGrid.mask)
        anomRawYr = np.ma.average(anomGrid.ravel(),weights=wgts)
        
        gridMask = anomGrid.mask
        anomGrid = griddata(stnsTWX[LON],stnsTWX[LAT],anomTWX[i,:],xGrid,yGrid)
        anomGrid = np.flipud(anomGrid)
        anomGrid = np.ma.masked_array(anomGrid,mask=gridMask)
        anomTwxYr = np.ma.average(anomGrid.ravel(),weights=wgts)
        
        avgAnomFLs[i] = anomFlsYr
        avgAnomTWX[i] = anomTwxYr
        avgAnomRAW[i] = anomRawYr
        
        difsAnom[i] = anomFlsYr - anomRawYr
        #difsAnom2[i] = np.ma.mean(anomFLs[i,:]) - np.ma.mean(anomRaw[i,:])
        difsAnom2[i] = anomTwxYr - anomRawYr
        #print difsAnom[i],difsAnom2[i]
#        if  difsAnom[i] > .3:
#            plt.imshow(anomGrid)
#            plt.show()
    
    plt.plot(difsAnom)
    plt.plot(difsAnom2)
    plt.legend(('HCN','TWX'))
    plt.show()
    
    plt.plot(avgAnomFLs)
    plt.plot(avgAnomTWX)
    plt.plot(avgAnomRAW)
    plt.legend(('HCN','TWX','RAW'))
    plt.show()
#        
#    dlon   = np.abs(dsGrid.geoT[1])*rr
#    dx     = dlon*np.cos(latGrid*rad)
#    
#    dy = np.ones(latGrid.size)*np.abs(dsGrid.geoT[5])*rr
##    dy[0] = 
##    dy[1:latGrid.size-1] = np.abs(latGrid[2:latGrid.size]-latGrid[0:latGrid.size-2])*rr*0.5 
##    dy[latGrid.size-1] = np.abs(latGrid[latGrid.size-1]-latGrid[latGrid.size-2])*rr
#    
#    area = dx*dy
#    area = np.cos(latGrid*rad)
#
#    print np.ma.average(anomGrid.ravel(),weights=area)
#    print np.ma.mean(anomGrid)
#    
##    plt.imshow(anomGrid)
##    plt.show()
#    
#    m = Basemap(projection='cyl',llcrnrlat=np.min(yGrid),urcrnrlat=np.max(yGrid),
#                llcrnrlon=np.min(xGrid),urcrnrlon=np.max(xGrid),resolution='l')
#    m.drawcoastlines()
#    m.drawstates()
#    m.drawcountries()
#    #m.contourf(stndaUS.stns[LON],stndaUS.stns[LAT], anom,latlon=True,tri=True)
#    m.imshow(np.flipud(anomGrid))
#    m.colorbar()
#    
#    plt.show()


def getPciAccuracyStats(stnda):
    
    mask_stns = np.logical_and(np.isnan(stnda.stns[BAD]),np.isfinite(stnda.stns[MASK])) 
    mask_stns = np.logical_and(stnda.stns[NEON] > 20,mask_stns)
    
    climDivs = stnda.stns[NEON]
    climDivs = np.unique(climDivs[np.isfinite(climDivs)])[2:]
    
    #Standard Error vs Mean Value MAE
    ###################################################
    mab = []
    stderr = []
    
    for div in climDivs:
        mab.append(np.mean(np.abs(stnda.stns['xval_overall_bias'][np.logical_and(mask_stns,stnda.stns[NEON]==div)])))
        stderr.append(np.mean(np.abs(stnda.stns['xval_stderr'][np.logical_and(mask_stns,stnda.stns[NEON]==div)])))
    ###################################################

    
    #Confidence interval stats
    ###################################################
    xvalErr = stnda.ds.variables['xval_err'][mask_stns]
    xvalStd = stnda.ds.variables['xval_stderr'][mask_stns]
    obs = stnda.stns[MEAN_OBS][mask_stns]
    interp = obs + xvalErr
    
    pRng = np.linspace(0.7,0.99,30)

    pAct = []
    pciWidth = []
    
    for p in pRng:
        
        pci_critval = stats.norm.ppf((1.0-p)/2.0)
        pci_r = np.abs(xvalStd*pci_critval) 
        
        pciU = interp + pci_r
        pciL = interp - pci_r
        
        inPci = np.logical_and(obs >= pciL, obs <= pciU)
        
        pAct.append(np.sum(inPci)/np.float(interp.size))
        pciWidth.append(np.mean(pciU-pciL))
    ###################################################
    
    return np.array(stderr),np.array(mab),np.array(pRng),np.array(pAct),np.array(pciWidth) 

def getPciAccuracyStats2(stndaDs):
    
    maskGoodStns = stndaDs.variables['bad'][:].mask
    maskDomain = ~stndaDs.variables['mask'][:].mask
    maskClimDivs = stndaDs.variables['neon'][:].data >= 101
    maskStns = np.logical_and(np.logical_and(maskGoodStns,maskDomain),maskClimDivs)
    
    climDivs = stndaDs.variables['neon'][:]
    climDivs = np.unique(np.ma.compressed(climDivs))
    uClimDivs = climDivs[climDivs >= 101]
    climDivs = stndaDs.variables['neon'][:].data
    
    xval_err_mthly = stndaDs.variables['xval_err_mthly'][0:12,:].data
    xval_stderr_mthly = stndaDs.variables['xval_stderr_mthly'][0:12,:].data
    
    #Standard Error vs Normals MAE
    ###################################################
    normMae = []
    stderr = []
    
    for div in uClimDivs:
        
        climDivMask = np.logical_and(climDivs == div,maskStns)
        
        maeClimDiv = np.abs(xval_err_mthly[:,climDivMask])
        seClimDiv = xval_stderr_mthly[:,climDivMask]
        
        mMae = np.mean(maeClimDiv,axis=1)
        mSe = np.mean(seClimDiv,axis=1)
        
        normMae.extend(mMae)
        stderr.extend(mSe)
    ###################################################

    #Confidence interval stats
    ###################################################
    xvalErr = xval_err_mthly[:,maskStns]
    xvalStd = xval_stderr_mthly[:,maskStns]
    
    obs = []
    for mth in np.arange(1,13):
        obs.append(stndaDs.variables[get_norm_varname(mth)][maskStns])
    obs = np.vstack(obs)
    
    interp = obs + xvalErr
    
    xvalErr = np.ravel(xvalErr)
    xvalStd = np.ravel(xvalStd)
    obs = np.ravel(obs)
    interp = np.ravel(interp)
    
    pRng = np.linspace(0.7,0.99,30)

    pAct = []
    pciWidth = []
    
    for p in pRng:
        
        pci_critval = stats.norm.ppf((1.0-p)/2.0)
        pci_r = np.abs(xvalStd*pci_critval) 
        
        pciU = interp + pci_r
        pciL = interp - pci_r
        
        inPci = np.logical_and(obs >= pciL, obs <= pciU)
        
        pAct.append(np.sum(inPci)/np.float(interp.size))
        pciWidth.append(np.mean(pciU-pciL))
    ###################################################
    
    return np.array(stderr),np.array(normMae),np.array(pRng),np.array(pAct),np.array(pciWidth) 

def plotPCIAccuracy():
    
#    stndtype = copy(stnData.DTYPE_STN_DFLT)
#    stndtype.append(('xval_overall_bias',np.float64))
#    stndtype.append(('xval_overall_mae',np.float64))
#    stndtype.append(('xval_err',np.float64))
#    stndtype.append(('xval_stderr',np.float64))
#    
#    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin',stn_dtype=stndtype)
#    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax',stn_dtype=stndtype)
#    
    dsTmin = Dataset('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc')
    dsTmax = Dataset('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
    
    stderrTmin,maeTmin,pRngTmin,pActTmin,pciWidthTmin = getPciAccuracyStats2(dsTmin)
    stderrTmax,maeTmax,pRngTmax,pActTmax,pciWidthTmax = getPciAccuracyStats2(dsTmax)

    print pRngTmin
    print pActTmin
    print ""
    print pRngTmax
    print pActTmax
    print ""
    print pciWidthTmin
    print pciWidthTmax
#    print pciWidthTmin[pRngTmin==0.70],pciWidthTmin[pRngTmin==0.99]
#    print pciWidthTmax[pRngTmax==0.70],pciWidthTmax[pRngTmax==0.99]
#    
#    
#    print stderrTmin
    #Tmin std vs. mae
    plt.subplot(221)
    slope,incpt,r_value = stats.linregress(stderrTmin, maeTmin)[:3]
    r2 = r_value**2 #r-squared value; variance explained
    plt.plot(stderrTmin,maeTmin,'.',color='grey',label='_nolegend_')#"_nolegend_")'#17375E'
    xmin,xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = incpt + x*slope
    plt.plot(x,y,color="k",label="Regression Line")#,lw=2)"#C0504D"
    plt.xlim((xmin,xmax))
    plt.xlabel("MAE ($^\circ$C)",fontsize=12)
    plt.ylabel(r"$\bar \sigma_k$"+"($^\circ$C)",fontsize=12)
    plt.legend(loc=2,fontsize=10)
    ylocs,ylabs = plt.yticks()
    plt.yticks(ylocs,fontsize=12)
    xlocs,xlabs = plt.xticks()
    plt.xticks(xlocs,fontsize=12)
    plt.text(.70, .1, "".join(["R$^2$=%.2f"%(r2,)]), fontsize=10, transform=plt.gca().transAxes)
    plt.title("a.",loc="left")
#    ax = plt.gca()
#    ax.set_axisbelow(True)
#    ax.xaxis.grid(color='gray', linestyle='dashed')
#    ax.yaxis.grid(color='gray', linestyle='dashed')
    
    #Tmax std vs. mae
    plt.subplot(222)
    slope,incpt,r_value = stats.linregress(stderrTmax, maeTmax)[:3]
    r2 = r_value**2 #r-squared value; variance explained
    plt.plot(stderrTmax,maeTmax,'.',color='grey',label='_nolegend_')#"_nolegend_")
    xmin,xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = incpt + x*slope
    plt.plot(x,y,color="k",label="Regression Line")#,lw=2)
    plt.xlim((xmin,xmax))
    plt.xlabel("MAE ($^\circ$C)",fontsize=12)
    plt.ylabel(r"$\bar \sigma_k$"+"($^\circ$C)",fontsize=12)
    plt.legend(loc=2,fontsize=10)
    ylocs,ylabs = plt.yticks()
    plt.yticks(ylocs,fontsize=12)
    xlocs,xlabs = plt.xticks()
    plt.xticks(xlocs,fontsize=12)
    plt.text(.70, .1, "".join(["R$^2$=%.2f"%(r2,)]), fontsize=10, transform=plt.gca().transAxes)
    plt.title("b.",loc="left")
    plt.xlim((xmin,xmax))
#    ax = plt.gca()
#    ax.set_axisbelow(True)
#    ax.xaxis.grid(color='gray', linestyle='dashed')
#    ax.yaxis.grid(color='gray', linestyle='dashed')
    
    
    plt.subplot(223)
    plt.plot(pRngTmin*100,pRngTmax*100,'k')
    plt.plot(pRngTmin*100,pActTmin*100,'.',color="k")#,markersize=3)markerfacecolor="none"
    plt.plot(pRngTmax*100,pActTmax*100,'+',color="k")
    plt.xlabel('Prediction CI (%)')
    plt.ylabel('Actual in Interval (%)')
    plt.legend(("1:1 line",r"$\overline{TN}_a$",r"$\overline{TX}_a$"),loc=2,fontsize=10)
    plt.title("c.",loc="left")
    
    plt.xlim((pRngTmin[0]*100,pRngTmin[-1]*100))
    plt.ylim((pRngTmin[0]*100,pRngTmin[-1]*100))
#    ax = plt.gca()
#    ax.set_axisbelow(True)
#    ax.xaxis.grid(color='gray', linestyle='dashed')
#    ax.yaxis.grid(color='gray', linestyle='dashed') 
    
    
    plt.subplot(224)
    plt.plot(pRngTmin*100,pciWidthTmin,'.',color="k")
    plt.plot(pRngTmax*100,pciWidthTmax,'+',color="k")
    plt.xlabel('Prediction CI (%)') 
    plt.ylabel('CI Width ($^\circ$C)')
    plt.legend((r"$\overline{TN}_a$",r"$\overline{TX}_a$"),loc=2,fontsize=10)
    plt.subplots_adjust(hspace=.35,wspace=.35)
    plt.title("d.",loc="left")
#    ax = plt.gca()
#    ax.set_axisbelow(True)
#    ax.xaxis.grid(color='gray', linestyle='dashed')
#    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.savefig('/projects/daymet2/docs/final_writeup/pciAccuracy.png',dpi=300) 
    plt.show()

def anomalyMapHCNvsTopoWx():
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
    strMth = stndaUS.mths[DATE][0]
    endMth = stndaUS.mths[DATE][-1]
    
    uYrs = np.unique(stndaUS.mths[YEAR])
    baseMask = np.logical_and(uYrs >= 1961,uYrs <= 1990)
    
    days = utld.get_days_metadata(datetime(strMth.year,1,1), datetime(endMth.year,12,31))
    tairAgg = ushcn.TairAggregate(days)
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    gridMask = dsGrid.gdalDs.ReadAsArray() != 19
    
    #stns = stndaUS.stns[np.sum(stndaUS.data['raw_tmax'].mask,axis=0)==0]
    stns = stndaUS.stns
    
    anomFLs = np.zeros((uYrs.size,stns.size))
    anomFLs = np.ma.masked_array(anomFLs,np.isnan(anomFLs))
    
    for astn,x in zip(stns,np.arange(stns.size)):
        
        obsFLs = stndaUS.loadObs(astn[STN_ID], 'FLs.52i_tmax')
        obsFLs = tairAgg.mthlyToAnn(obsFLs)
        obsFLs = obsFLs - np.ma.mean(obsFLs[baseMask])
        
        anomFLs[:,x] = obsFLs
 
    ###########################################
    stndaTWX = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax',stn_dtype=DTYPE_STN_MEAN_LST_TDI)
    stnsTwxMask = np.logical_and(np.isnan(stndaTWX.stns[BAD]),np.isfinite(stndaTWX.stns[MASK]))
    stnsTWX = stndaTWX.stns[stnsTwxMask]
    stnsTwxIdx = np.nonzero(stnsTwxMask)[0]
        
    anomTWX = np.zeros((uYrs.size,stnsTWX.size))
    anomTWX = np.ma.masked_array(anomTWX,np.isnan(anomTWX))
    
    for astn,x,i in zip(stnsTWX,stnsTwxIdx,np.arange(stnsTwxIdx.size)):
        obsTWX = stndaTWX.ds.variables['tmax_ann'][:,x]
        obsTWX = obsTWX - np.ma.mean(obsTWX[baseMask])
        anomTWX[:,i] = obsTWX
    ###########################################
        
    
    rad = 4.0*np.arctan(1.0)/180.0
    re = 6371220.0
    rr  = re*rad

    latGrid,lonGrid = dsGrid.getCoordMeshGrid()
    latGrid = latGrid.ravel()
    lonGrid = lonGrid.ravel()
    
    wgts = np.cos(latGrid*rad)
    
    yGrid,xGrid = dsGrid.getCoordGrid1d()
    yGrid = np.sort(yGrid)
        
    avgAnomFLs = np.zeros(uYrs.size)
    avgAnomTWX = np.zeros(uYrs.size)
    
    for i in np.arange(uYrs.size):
    
        print uYrs[i]
        
        anomGrid = griddata(stns[LON],stns[LAT],anomFLs[i,:],xGrid,yGrid)
        anomGrid = np.flipud(anomGrid)
        anomGrid.mask = np.logical_or(gridMask,anomGrid.mask)
        anomFlsYr = np.ma.average(anomGrid.ravel(),weights=wgts)
        avgAnomFLs[i] = anomFlsYr
        
        anomGridTwx = griddata(stnsTWX[LON],stnsTWX[LAT],anomTWX[i,:],xGrid,yGrid)
        anomGridTwx = np.flipud(anomGridTwx)
        anomGridTwx.mask = np.logical_or(np.logical_or(gridMask,anomGridTwx.mask),anomGrid.mask)
        anomTWXYr = np.ma.average(anomGridTwx.ravel(),weights=wgts)
        avgAnomTWX[i] = anomTWXYr
        
#        plt.subplot(121)
#        m = Basemap(projection='cyl',llcrnrlat=np.min(yGrid),urcrnrlat=np.max(yGrid),
#                    llcrnrlon=np.min(xGrid),urcrnrlon=np.max(xGrid),resolution='l')
#        m.drawcoastlines()
#        m.drawstates()
#        m.drawcountries()
#        #m.contourf(stndaUS.stns[LON],stndaUS.stns[LAT], anom,latlon=True,tri=True)
#        m.imshow(np.flipud(anomGrid))
#        m.colorbar()
#        
#        plt.subplot(122)
#        m = Basemap(projection='cyl',llcrnrlat=np.min(yGrid),urcrnrlat=np.max(yGrid),
#                    llcrnrlon=np.min(xGrid),urcrnrlon=np.max(xGrid),resolution='l')
#        m.drawcoastlines()
#        m.drawstates()
#        m.drawcountries()
#        #m.contourf(stndaUS.stns[LON],stndaUS.stns[LAT], anom,latlon=True,tri=True)
#        m.imshow(np.flipud(anomGridTwx))
#        m.colorbar()
#        
#        plt.show()
    
    plt.plot(avgAnomTWX-avgAnomFLs)
    #plt.plot(avgAnomTWX)
    plt.show()
    
#    plt.plot(avgAnomFLs)
#    plt.plot(avgAnomTWX)
#    plt.plot(avgAnomRAW)
#    plt.legend(('HCN','TWX','RAW'))
#    plt.show()
#        
#    dlon   = np.abs(dsGrid.geoT[1])*rr
#    dx     = dlon*np.cos(latGrid*rad)
#    
#    dy = np.ones(latGrid.size)*np.abs(dsGrid.geoT[5])*rr
##    dy[0] = 
##    dy[1:latGrid.size-1] = np.abs(latGrid[2:latGrid.size]-latGrid[0:latGrid.size-2])*rr*0.5 
##    dy[latGrid.size-1] = np.abs(latGrid[latGrid.size-1]-latGrid[latGrid.size-2])*rr
#    
#    area = dx*dy
#    area = np.cos(latGrid*rad)
#
#    print np.ma.average(anomGrid.ravel(),weights=area)
#    print np.ma.mean(anomGrid)
#    
##    plt.imshow(anomGrid)
##    plt.show()
#    
#    m = Basemap(projection='cyl',llcrnrlat=np.min(yGrid),urcrnrlat=np.max(yGrid),
#                llcrnrlon=np.min(xGrid),urcrnrlon=np.max(xGrid),resolution='l')
#    m.drawcoastlines()
#    m.drawstates()
#    m.drawcountries()
#    #m.contourf(stndaUS.stns[LON],stndaUS.stns[LAT], anom,latlon=True,tri=True)
#    m.imshow(np.flipud(anomGrid))
#    m.colorbar()
#    
#    plt.show()

def avgAnnDifsHomog():
    stndaH = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc','tmax')
    stndaS = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serial_tmax.nc','tmax',stn_dtype=DTYPE_STN_BASIC)
    
    mask_stns1 = np.logical_and(np.isfinite(stndaH.stns[MASK]),np.isnan(stndaH.stns[BAD]))
    
    mask_stns2 = np.logical_or(np.logical_or(np.char.startswith(stndaH.stn_ids, 'SNOTEL'),
                                np.char.startswith(stndaH.stn_ids, 'RAWS')),np.char.startswith(stndaH.stn_ids, 'GHCN_US'))
    
    mask_stns = np.logical_and(mask_stns1,mask_stns2)
    
    
    
    stns = stndaH.stns[mask_stns]
    maskSNOTEL = np.char.startswith(stns[STN_ID], 'SNOTEL')
    maskRAWS = np.char.startswith(stns[STN_ID], 'RAWS')
    maskGHCN = np.char.startswith(stns[STN_ID], 'GHCN')
    maskHCN = ushcn.buildGhcnUShcnMask(stns[STN_ID],'/projects/daymet2/station_data/ghcn/ghcnd-stations.txt')
    
    uYrs = np.unique(stndaH.days[YEAR])
    
    annDifs = np.zeros((uYrs.size,stns.size))
    
    yrMasks = []
    for yr in uYrs:
        yrMasks.append(stndaH.days[YEAR] == yr)
    
    stchk = status_check(stns.size,100)
    for aId,x in zip(stns[STN_ID],np.arange(stns[STN_ID].size)):
        
        obsH = stndaH.load_obs(aId) 
        obsS = stndaS.load_obs(aId)
        
        annH = np.array([np.mean(obsH[aMask],dtype=np.float) for aMask in yrMasks])
        annS = np.array([np.mean(obsS[aMask],dtype=np.float) for aMask in yrMasks])
        
#        anomH = annH-np.mean(annH)
#        anomS = annS-np.mean(annS)
        
        annDifs[:,x] = annH - annS
        
#        for aMask,i in zip(yrMasks,np.arange(len(yrMasks))):
#            
#            annH = np.mean(obsH[aMask],dtype=np.float)
#            annS = np.mean(obsS[aMask],dtype=np.float)
#            annDifs[i,x] = annH - annS
        
        stchk.increment()
    
    #avgAnnDif = np.mean(annDifs,axis=1)
    
    #np.save('/projects/daymet2/docs/final_writeup/homogAnnAnomDifsTmax.npy', annDifs)
    #np.save('/projects/daymet2/docs/final_writeup/homogAnnAnomDifsTmaxStnIds.npy', stns[STN_ID])
    
    plt.plot(np.mean(annDifs[:,maskHCN],axis=1))
    plt.plot(np.mean(annDifs[:,maskGHCN],axis=1))
    plt.plot(np.mean(annDifs[:,maskRAWS],axis=1))
    plt.plot(np.mean(annDifs[:,maskSNOTEL],axis=1))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.legend(('USHCN','GHCN','RAWS','SNOTEL'))
    plt.show()
    
def runs_of_ones_array(bits):
    #http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts

def getStnsForHomogAnomCompare(ds,stnIdsToChk,tairVar,dayMask=None,outFpath=None):
       
    stnIds = ds.variables['stn_id'][:].astype("<S16")
    
    stat_chk = status_check(stnIdsToChk.size,500)
    
    USE_ALL_IMP_THRESHOLD = np.round(365.25 * 5.0)
    
    fnlIds = []
    
    for aId in stnIdsToChk:
        
        x = np.nonzero(stnIds==aId)[0][0]
        
        if dayMask is None:
            stnTair = ds.variables[tairVar][:,x]
        else:
            stnTair = ds.variables[tairVar][dayMask,x]
        
        try:   
            maskMiss = stnTair.mask
        except AttributeError:
            maskMiss = np.isnan(stnTair)
        
        missRuns = runs_of_ones_array(maskMiss)
        
        if missRuns.size > 0:
            maxRun = np.max(missRuns)
        else:
            maxRun = 0
        
        if maxRun < USE_ALL_IMP_THRESHOLD:
            fnlIds.append(aId)
            
        stat_chk.increment()
        
    fnlIds = np.array(fnlIds)
    
    if outFpath is not None:
        np.savetxt(outFpath, fnlIds,fmt="%s")
    
    return fnlIds

def createPredictorGrids():
    os.chdir('/projects/daymet2/docs/ncar_workshop2013_poster/EgTile')
    crop_nodata('tmin_mean_Layer1.tif','crop_tmin_mean_Layer1.tif')
    crop_nodata('tmin_se_Layer1.tif', 'crop_tmin_se_Layer1.tif')
    crop_nodata('tmax_mean_Layer1.tif','crop_tmax_mean_Layer1.tif')
    crop_nodata('tmax_se_Layer1.tif','crop_tmax_se_Layer1.tif')
    crop_nodata('tile17Elef1.tif','crop_tile17Elev1.tif')
    crop_nodata('tile17LstTmax1.tif','crop_tile17LstTmax1.tif')
    crop_nodata('tile17LstTmin1.tif','crop_tile17LstTmin1.tif')
    crop_nodata('tile17Tdi1.tif','crop_tile17Tdi1.tif')

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
    
    im = grid[2].imshow(lstTmin,cmap=cm.spectral)
    grid[2].set_xticks([])
    grid[2].set_yticks([])
    grid[2].set_title("MODIS Tmin LST ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[2].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    im = grid[3].imshow(lstTmax,cmap=cm.spectral)
    grid[3].set_xticks([])
    grid[3].set_yticks([])
    grid[3].set_title("MODIS Tmax LST ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[3].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    grid = ImageGrid(cf,122,nrows_ncols=(2,2),axes_pad=.625,cbar_mode="each",cbar_location="right",label_mode = "1",cbar_pad=0.02)
    
    im = grid[0].imshow(meanTmin,cmap=cm.spectral)
    grid[0].set_xticks([])
    grid[0].set_yticks([])
    grid[0].set_title("Normal Daily Tmin ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    im = grid[1].imshow(meanTmax,cmap=cm.spectral)
    grid[1].set_xticks([])
    grid[1].set_yticks([])
    grid[1].set_title("Normal Daily Tmax ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[1].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    norm = Normalize(vmin=np.min(np.array([np.min(seTmin),np.min(seTmax)])),vmax=np.max(np.array([np.max(seTmin),np.max(seTmax)])))
    
    im = grid[2].imshow(seTmin,cmap=cm.spectral,norm=norm)
    grid[2].set_xticks([])
    grid[2].set_yticks([])
    grid[2].set_title("Std. Err. Tmin ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[2].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    im = grid[3].imshow(seTmax,cmap=cm.spectral,norm=norm)
    grid[3].set_xticks([])
    grid[3].set_yticks([])
    grid[3].set_title("Std. Err. Tmax ($^\circ$C)",fontsize=17)
    cbar = grid.cbar_axes[3].colorbar(im)
    cbar.ax.tick_params(labelsize=17)
    
    
    cf.set_size_inches(8*2*1.25,6*2*1.25)
    
    
    #fig.subplots_adjust(hspace=0.1)
#    grid[4].imshow(meanTmin)
#    grid[5].imshow(meanTmax)
#    grid[6].imshow(seTmin)
#    grid[7].imshow(seTmax)
    plt.savefig('/projects/daymet2/docs/ncar_workshop2013_poster/exInterpMaps.png',dpi=300)
    plt.show()


def getCceHillshade(m,llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon):
    dsElev = RasterDataset('/projects/daymet2/dem/hillshade30-wgs84-ds.tif')
    latElev,lonElev = dsElev.getCoordGrid1d()
    latElev = np.sort(latElev)
    nx = np.sum(np.logical_and(latElev>=llcrnrlat,latElev<=urcrnrlat))
    ny = np.sum(np.logical_and(lonElev>=llcrnrlon,lonElev<=urcrnrlon))
    #xElev, yElev = m(*np.meshgrid(lonElev,latElev))
    elev = dsElev.readAsArray()
    elev = np.flipud(elev)
    elev = m.transform_scalar(elev, lonElev, latElev, nx, ny)
    return elev

def plotTwxVsPrismDaymetNorms():
    '''
    /projects/daymet2/cce_case_study/prism_files/normals_mthly/cce_tmax_normal_11.tif 
    /projects/daymet2/cce_case_study/topowx_files/normals/mosaics
    /projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics
    '''
    def getMeanNorm(dataPath,varName,mths):
        
        mthNorms = []
        
        for mth in mths:
            
            ds = RasterDataset("".join([dataPath,"cce_%s_normal_%02d.tif"%(varName,mth)]))
            a = ds.readAsArray()
            a.shape = (1,a.shape[0],a.shape[1])
            mthNorms.append(a)
            
        mthNorms = np.ma.vstack(mthNorms)
        return np.ma.mean(mthNorms,axis=0)
    
    tairVar = 'tmin'
    #djf = np.array([12,1,2])
    #djf = np.arange(1,13)
    jja = np.array([12])
    #jja = np.arange(1,13)
    djf = jja
    
    sTwx = getMeanNorm('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/',tairVar,jja)
    sDaymet = getMeanNorm('/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/',tairVar,jja)
    sPrism = getMeanNorm('/projects/daymet2/cce_case_study/prism_files/normals_mthly/',tairVar,jja)
    tairVar = 'tmax'
    wTwx = getMeanNorm('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/',tairVar,djf)
    wDaymet = getMeanNorm('/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/',tairVar,djf)
    wPrism = getMeanNorm('/projects/daymet2/cce_case_study/prism_files/normals_mthly/',tairVar,djf)
    
    
    #CS = plt.contourf(wTwx,alpha=.4,ls=None,antialiased=True)
    
    #CS = contourf(Z) 
#    for c in CS.collections: 
#        c.set_antialiased(False) 
    
    #plt.contour(wTwx,alpha=.4)
    #plt.savefig('/projects/daymet2/docs/final_writeup/testAlpha.png',dpi=300)
    #plt.show()
    
    sDaymetDif = sDaymet - sTwx
    wDaymetDif = wDaymet - wTwx
    sPrismDif = sPrism - sTwx
    wPrismDif = wPrism - wTwx
    
    dsEx = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/cce_%s_normal_%02d.tif'%(tairVar,1))
         
    dsGrid = dsEx
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
    lon_0 = (llcrnrlon+urcrnrlon)/2.0
    lat_0 = (llcrnrlat+urcrnrlat)/2.0

    def drawCceBnds(m):
        m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)#,linewidth=1)
        m.drawcountries()
        m.drawstates()

    print "Mapping data...."
    m = Basemap(resolution='h',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=-111,lat_0=0)
    x, y = m(*np.meshgrid(lon,lat))
    
    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 8, reverse=True)
    clrs = clrs.mpl_colors
    clrs[3] = "grey"
    clrs[4] = "grey"
    
#    cc = matplotlib.colors.ColorConverter()
#    grey = cc.to_rgb("grey")
#    grey = [x*255 for x in grey]
#    
#    cmap = brewer2mpl.get_map('RdBu', 'Diverging', 8, reverse=True)
#    #cmap.colors[3] = grey
#    #cmap.colors[4] = grey
#    #cmap.mpl_colors[3] = 'grey'
#    #cmap.mpl_colors[4] = 'grey'
#    #print cmap.mpl_colors
#    cmap = cmap.get_mpl_colormap()
#    cmap.set_over('red')
#    cmap.set_under('blue')
    
    levels = np.arange(-4,5)

    dsElev = RasterDataset('/projects/daymet2/dem/hillshade30-wgs84-ds.tif')
    latElev,lonElev = dsElev.getCoordGrid1d()
    latElev = np.sort(latElev)
    nx = np.sum(np.logical_and(latElev>=llcrnrlat,latElev<=urcrnrlat))
    ny = np.sum(np.logical_and(lonElev>=llcrnrlon,lonElev<=urcrnrlon))
    #xElev, yElev = m(*np.meshgrid(lonElev,latElev))
    elev = dsElev.readAsArray()
    elev = np.flipud(elev)
    elev = m.transform_scalar(elev, lonElev, latElev, nx, ny)#, returnxy, checkbounds, order, masked)
    
    #lonAll = lon
    #latAll = np.sort(lat)
    #wTwx = np.flipud(wTwx)
    
    #m.imshow(elev,cmap=cm.gray)
    #wTwx2 = m.transform_scalar(wTwx, lonAll, latAll, lonAll.size, latAll.size)
    #wTwx2 = np.ma.masked_array(wTwx2,wTwx.mask)
    
    #m.imshow(wTwx2,cmap=CMAP_ESRI_PRCP,alpha=.55)
    #cf = m.contourf(x,y,sTwx,20,cmap=CMAP_ESRI_PRCP,alpha=.4,antialiased=True)
    #cf = m.contourf(x,y,sTwx,20,cmap=CMAP_ESRI_PRCP,alpha=.4,antialiased=True)

    # plt.show()
  
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05,cbar_size="3%")
   
    alpha = .4
   
    transLon = (-114.6,-112.5)
    transLat = (47.78,47.78)
   
    m.ax = grid[0]
    m.imshow(elev,cmap=cm.gray)
    #cf = m.contourf(x,y,sDaymetDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend="both")#,extend='both')
    #cf = m.contourf(x,y,sDaymetDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(x,y,sDaymetDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend='both')#,extend='both')
    cf = m.contourf(x,y,sDaymetDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend='both')
    m.plot(transLon,transLat,'k--',latlon=True,lw=.5)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_alpha(1)
    cbar.draw_all()
    grid[0].set_ylabel("Tmin")
    grid[0].set_title("Daymet Minus TopoWx",fontsize=12)
    #cbar = grid.cbar_axes.colorbar(cf)
    #cbar.set_ticks(levels)
    cbar.set_label(r'$^\circ$C')
    
    drawCceBnds(m)
    #m.etopo(scale=10)
    plt.sca(grid[0])
    tsize = 6
    plt.text(107088,140951-5000,"1",fontsize=tsize)
    plt.text(119947,140951-5000,"2",fontsize=tsize)
    plt.text(129132,140951-5000,"3",fontsize=tsize)
    plt.text(140154,140951-5000,"4",fontsize=tsize)
    plt.text(195877,140338-5000,"5",fontsize=tsize)
    plt.text(6051,11135,"1=Flathead Lake\n2=Mission Mountains\n3=Seeley-Swan Valley\n4=Swan Crest\n5=Gates Park",fontsize=6,bbox=dict(facecolor='white',alpha=.7))
    
    
    
    m.ax = grid[1]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(x,y,sPrismDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend='both')#,extend='both')
    cf = m.contourf(x,y,sPrismDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend='both')
    m.plot(transLon,transLat,'k--',latlon=True,lw=.5)
    grid[1].set_title("PRISM Minus TopoWx",fontsize=12)
    drawCceBnds(m)
    plt.sca(grid[1])
    tsize = 6
    plt.text(107088,140951-5000,"1",fontsize=tsize)
    plt.text(119947,140951-5000,"2",fontsize=tsize)
    plt.text(129132,140951-5000,"3",fontsize=tsize)
    plt.text(140154,140951-5000,"4",fontsize=tsize)
    plt.text(195877,140338-5000,"5",fontsize=tsize)
    #plt.text(6051,11135,"1=Flathead Lake\n2=Mission Mountains\n3=Seeley-Swan Valley\n4=Swan Crest\n5=Gates Park",fontsize=6,bbox=dict(facecolor='white',alpha=.7))
    
    
    
    m.ax = grid[2]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(x,y,wDaymetDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend='both')#,extend='both')
    cf = m.contourf(x,y,wDaymetDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend='both')
    m.plot(transLon,transLat,'k--',latlon=True,lw=.5)
    grid[2].set_ylabel("Tmax")
    drawCceBnds(m)
    plt.sca(grid[2])
    tsize = 6
    plt.text(107088,140951-5000,"1",fontsize=tsize)
    plt.text(119947,140951-5000,"2",fontsize=tsize)
    plt.text(129132,140951-5000,"3",fontsize=tsize)
    plt.text(140154,140951-5000,"4",fontsize=tsize)
    plt.text(195877,140338-5000,"5",fontsize=tsize)
    #plt.text(6051,11135,"1=Flathead Lake\n2=Mission Mountains\n3=Seeley-Swan Valley\n4=Swan Crest\n5=Gates Park",fontsize=6,bbox=dict(facecolor='white',alpha=.7))
    
    
    m.ax = grid[3]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(x,y,wPrismDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend='both')#,extend='both')
    cf = m.contourf(x,y,wPrismDif,colors=clrs,levels=levels,alpha=alpha,antialiased=True,extend='both')
    m.plot(transLon,transLat,'k--',latlon=True,lw=.5)
    drawCceBnds(m)
    plt.sca(grid[3])
    tsize = 6
    plt.text(107088,140951-5000,"1",fontsize=tsize)
    plt.text(119947,140951-5000,"2",fontsize=tsize)
    plt.text(129132,140951-5000,"3",fontsize=tsize)
    plt.text(140154,140951-5000,"4",fontsize=tsize)
    plt.text(195877,140338-5000,"5",fontsize=tsize)
    #plt.text(6051,11135,"1=Flathead Lake\n2=Mission Mountains\n3=Seeley-Swan Valley\n4=Swan Crest\n5=Gates Park",fontsize=6,bbox=dict(facecolor='white',alpha=.7))
    
    
    cfig = plt.gcf()
    plt.tight_layout()
    cfig.set_size_inches(7,5.5)

    plt.savefig('/projects/daymet2/docs/final_writeup/cce_normdifs.png',dpi=250) 
    #plt.savefig()
    plt.show()

def plotTwxDaymetPrismNorms():
    
    def getMeanNorm(dataPath,varName,mths):
        
        mthNorms = []
        
        for mth in mths:
            
            ds = RasterDataset("".join([dataPath,"cce_%s_normal_%02d.tif"%(varName,mth)]))
            a = ds.readAsArray()
            a.shape = (1,a.shape[0],a.shape[1])
            mthNorms.append(a)
            
        mthNorms = np.ma.vstack(mthNorms)
        print mthNorms.shape
        return np.ma.mean(mthNorms,axis=0)
    
    
    jja = np.array([6,7,8])
    tairVar = 'tmin'
    sTwxTmin = getMeanNorm('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/',tairVar,jja)
    sDaymetTmin = getMeanNorm('/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/',tairVar,jja)
    sPrismTmin = getMeanNorm('/projects/daymet2/cce_case_study/prism_files/normals_mthly/',tairVar,jja)
    tairVar = 'tmax'
    sTwxTmax = getMeanNorm('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/',tairVar,jja)
    sDaymetTmax = getMeanNorm('/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/',tairVar,jja)
    sPrismTmax = getMeanNorm('/projects/daymet2/cce_case_study/prism_files/normals_mthly/',tairVar,jja)
    
    vminNormTmin = np.min([np.min(sTwxTmin),np.min(sPrismTmin),np.min(sDaymetTmin)])
    vmaxNormTmin = np.max([np.max(sTwxTmin),np.max(sPrismTmin),np.max(sDaymetTmin)])
    levelsNormTmin = np.arange(np.floor(vminNormTmin*1)/1, (np.ceil(vmaxNormTmin*1)/1)+1,.1)
    
    vminNormTmax = np.min([np.min(sTwxTmax),np.min(sPrismTmax),np.min(sDaymetTmax)])
    vmaxNormTmax = np.max([np.max(sTwxTmax),np.max(sPrismTmax),np.max(sDaymetTmax)])
    levelsNormTmax = np.arange(np.floor(vminNormTmax*1)/1, (np.ceil(vmaxNormTmax*1)/1)+1,.1)
    
    dsEx = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/cce_%s_normal_%02d.tif'%(tairVar,1))
         
    dsGrid = dsEx
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
    lon_0 = (llcrnrlon+urcrnrlon)/2.0
    lat_0 = (llcrnrlat+urcrnrlat)/2.0

    def drawCceBnds(m):
        m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)#,linewidth=1)

    print "Mapping data...."
    m = Basemap(resolution='i',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
    
#    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 8, reverse=True)
#    clrs = clrs.mpl_colors
#    clrs[3] = "grey"
#    clrs[4] = "grey"
#    levels = np.arange(-4,5)
#    
    cfig = plt.gcf()
    grid = ImageGrid(cfig,121,nrows_ncols=(3,1),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)
   
    m.ax = grid[0]
    cf = m.contourf(x,y,sTwxTmin,cmap=CMAP_ESRI_PRCP,levels=levelsNormTmin)#,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    drawCceBnds(m)
    
    m.ax = grid[1]
    cf = m.contourf(x,y,sPrismTmin,cmap=CMAP_ESRI_PRCP,levels=levelsNormTmin)#,extend='both')
    drawCceBnds(m)
    
    m.ax = grid[2]
    cf = m.contourf(x,y,sDaymetTmin,cmap=CMAP_ESRI_PRCP,levels=levelsNormTmin)#,extend='both')
    drawCceBnds(m)
    
    grid = ImageGrid(cfig,122,nrows_ncols=(3,1),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)
    
    m.ax = grid[0]
    cf = m.contourf(x,y,sTwxTmax,cmap=CMAP_ESRI_PRCP,levels=levelsNormTmax)#,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    drawCceBnds(m)
    
    m.ax = grid[1]
    cf = m.contourf(x,y,sPrismTmax,cmap=CMAP_ESRI_PRCP,levels=levelsNormTmax)#,extend='both')
    drawCceBnds(m)
    
    m.ax = grid[2]
    cf = m.contourf(x,y,sDaymetTmax,cmap=CMAP_ESRI_PRCP,levels=levelsNormTmax)#,extend='both')
    drawCceBnds(m)

    plt.show()

def plotTwxVariationsVsPrismDaymetNorms():
            
    dsTwxTmin = RasterDataset('/projects/daymet2/compare/topowx_files/normals/cce_topowx_tmax19812010norm.tif')
    dsTwxNoLstTmin = RasterDataset('/projects/daymet2/compare/topowx_files/normals/no_lst/cce_topowx_tmax19812010norm.tif')
    dsTwxNoHomogTmin = RasterDataset('/projects/daymet2/compare/topowx_files/normals/no_homog/cce_topowx_tmax19812010norm.tif')
    dsTwxNoHomogLstTmin = RasterDataset('/projects/daymet2/compare/topowx_files/normals/no_homog_lst/cce_topowx_tmax19812010norm.tif')
    
    dsPrismTmin = RasterDataset('/projects/daymet2/compare/prism_files/normals/cce_prism_tmax1981_2010norm.tif')
    dsDaymetTmin = RasterDataset('/projects/daymet2/compare/daymet_files/normals/cce_daymet_tmax19812010norm.tif')
    
    twxTmin = dsTwxTmin.readAsArray()
    tmxNoLstTmin = dsTwxNoLstTmin.readAsArray()
    tmxNoHomogTmin = dsTwxNoHomogTmin.readAsArray()
    tmxNoHomogLstTmin = dsTwxNoHomogLstTmin.readAsArray()
    
    prismTmin = dsPrismTmin.readAsArray()/100
    daymetTmin = dsDaymetTmin.readAsArray()
    prismTmin = daymetTmin
    
    lsAllDs = [twxTmin,tmxNoLstTmin,tmxNoHomogTmin,tmxNoHomogLstTmin,prismTmin,daymetTmin]
        
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
    
#    cmap = cm.jet
#    minTmin = np.min([np.min(aDs) for aDs in lsAllDs])
#    maxTmin = np.max([np.max(aDs) for aDs in lsAllDs])
#    print minTmin,maxTmin
#    levels = np.arange(-8,5.1,0.1)
    
    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 8, reverse=True)
    clrs = clrs.mpl_colors
    #clrs[3] = "grey"
    #clrs[4] = "grey"
#    minTmin = np.min([np.min(twxTmin-prismTmin),np.min(twxTmin-daymetTmin)])
#    maxTmin = np.max([np.max(twxTmin-prismTmin),np.max(twxTmin-daymetTmin)])
#    print minTmin,maxTmin
    levels = np.arange(-4,5)
    
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)
    
    m.ax = grid[0]
    cf = m.contourf(x,y,twxTmin-prismTmin,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[0].set_title("TopoWx-PRISM")
    
    m.ax = grid[1]
    cf = m.contourf(x,y,tmxNoLstTmin-prismTmin,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[1])
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[1].set_title("TopoWxNoLst-PRISM")
    
    m.ax = grid[2]
    cf = m.contourf(x,y,tmxNoHomogTmin-prismTmin,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[2])
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[2].set_title("TopoWxNoHomog-PRISM")
    
    m.ax = grid[3]
    cf = m.contourf(x,y,tmxNoHomogLstTmin-prismTmin,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[3])
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[3].set_title("TopoWxNoHomogLst-PRISM")
    
    plt.suptitle("Daymet Tmax Difs")
    
    plt.show()
    
def plotTwxVsPrismDaymetTmaxNorms():
            
    dsTwxTmax = RasterDataset('/projects/daymet2/compare/topowx_files/normals/cce_topowx_tmax19812010norm.tif')
    dsPrismTmax = RasterDataset('/projects/daymet2/compare/prism_files/normals/cce_prism_tmax1981_2010norm.tif')
    dsDaymetTmax = RasterDataset('/projects/daymet2/compare/daymet_files/normals/cce_daymet_tmax19812010norm.tif')
    
    twxTmax = dsTwxTmax.readAsArray()
    prismTmax = dsPrismTmax.readAsArray()/100
    daymetTmax = dsDaymetTmax.readAsArray()
        
    dsGrid = dsTwxTmax
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
    
    cmap = cm.jet
    minTmax = np.min([np.min(twxTmax),np.min(prismTmax),np.min(daymetTmax)])
    maxTmax = np.max([np.max(twxTmax),np.max(prismTmax),np.max(daymetTmax)])
    print minTmax,maxTmax
    levels = np.arange(1,16,0.1)
    
    cf = plt.gcf()
    grid = ImageGrid(cf,211,nrows_ncols=(1,3),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)#,cbar_size="3%")
    
    m.ax = grid[0]
    cf = m.contourf(x,y,twxTmax,cmap=cmap,levels=levels)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    #cbar.set_label(r'$^\circ$C / 65 yrs')
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[0].set_title("TopoWx")
    
    m.ax = grid[1]
    cf = m.contourf(x,y,prismTmax,cmap=cmap,levels=levels)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[1])
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[1].set_title("PRISM")
    
    m.ax = grid[2]
    cf = m.contourf(x,y,daymetTmax,cmap=cmap,levels=levels)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[2])
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[2].set_title("Daymet")
    
    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 8, reverse=True)
    clrs = clrs.mpl_colors
    clrs[3] = "grey"
    clrs[4] = "grey"
    minTmax = np.min([np.min(twxTmax-prismTmax),np.min(twxTmax-daymetTmax)])
    maxTmax = np.max([np.max(twxTmax-prismTmax),np.max(twxTmax-daymetTmax)])
    print minTmax,maxTmax
    levels = np.arange(-4,5)
    
    cf = plt.gcf()
    grid = ImageGrid(cf,212,nrows_ncols=(1,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)
    
    m.ax = grid[0]
    cf = m.contourf(x,y,twxTmax-prismTmax,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[0].set_title("TopoWx-PRISM")
    
    m.ax = grid[1]
    cf = m.contourf(x,y,twxTmax-daymetTmax,colors=clrs,levels=levels,extend='both')
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[1])
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    grid[1].set_title("TopoWx-Daymet")
    
    plt.show()

def drawCceBnds(m):
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)#,linewidth=1)
    m.drawcountries()
    m.drawstates()

def plotTwxVsDaymetPRISMTrend():
    
    dsTwxTmin = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmin19812010trend.tif')
    dsDaymetTmin = RasterDataset('/projects/daymet2/cce_case_study/daymet_files/trends/cce_daymet_tmin19812010trend.tif')
    dsPrismTmin = RasterDataset('/projects/daymet2/cce_case_study/prism_files/trends/cce_prism4km_tmin_trend1981-2010.tif')
    dsTwxTmax = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmax19812010trend.tif')
    dsDaymetTmax = RasterDataset('/projects/daymet2/cce_case_study/daymet_files/trends/cce_daymet_tmax19812010trend.tif')
    dsPrismTmax = RasterDataset('/projects/daymet2/cce_case_study/prism_files/trends/cce_prism4km_tmax_trend1981-2010.tif')
    
    twxTmin = dsTwxTmin.readAsArray()*30
    prismTmin = dsPrismTmin.readAsArray()*30
    daymetTmin = dsDaymetTmin.readAsArray()*30
    
    twxTmax = dsTwxTmax.readAsArray()*30
    prismTmax = dsPrismTmax.readAsArray()*30
    daymetTmax = dsDaymetTmax.readAsArray()*30
    
    print np.min([np.min(twxTmin),np.min(twxTmax),np.min(prismTmin),np.min(prismTmax),np.min(daymetTmin),np.min(daymetTmax)])
    print np.max([np.max(twxTmin),np.max(twxTmax),np.max(prismTmin),np.max(prismTmax),np.max(daymetTmin),np.max(daymetTmax)])
    
    twxTmaxC = np.ma.compressed(twxTmax)
    prismTmaxC = np.ma.compressed(prismTmax)
    daymetTmaxC = np.ma.compressed(daymetTmax)
    
    def percentTrend(a,dname):
        
        maskNeg = a < 0
        maskPos = a >= 0
        print dname
        print "Neg|%.5f|%.5f"%(np.sum(maskNeg)/np.float(maskNeg.size),np.mean(a[maskNeg]))
        print "Pos|%.5f|%.5f"%(np.sum(maskPos)/np.float(maskPos.size),np.mean(a[maskPos]))
    
    percentTrend(twxTmaxC,'twx')
    percentTrend(prismTmaxC,'prism')
    percentTrend(daymetTmaxC,'daymet')
    
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
    m = Basemap(resolution='h',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=-111,lat_0=0)
    xMap, yMap = m(*np.meshgrid(lon,lat))
    
#    levelsRed = np.arange(0,6.5,.5)
#    levelsBlue = np.arange(-2,0.5,.5)

    levelsRed = np.arange(0,5.5,.5)
    levelsBlue = np.arange(-3,0.5,.5)
    
    norm = Normalize(0,5)
    smReds = cm.ScalarMappable(norm, cm.Reds)
    midPtsReds = (levelsRed + np.diff(levelsRed, 1)[0]/2.0)[0:-1]
    
    norm = Normalize(-3,0)
    smBlues = cm.ScalarMappable(norm, cm.Blues_r)
    midPtsBlues = (levelsBlue + np.diff(levelsBlue, 1)[0]/2.0)[0:-1]
    
    clrsRed = [smReds.to_rgba(x) for x in midPtsReds]
    clrsBlu = [smBlues.to_rgba(x) for x in midPtsBlues]
    clrsBlu.extend(clrsRed)
    clrs = clrsBlu
    levels = np.arange(-3,5.5,.5)
    
    elev = getCceHillshade(m, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon)
  
    cfig = plt.gcf()
    grid = ImageGrid(cfig,111,nrows_ncols=(2,3),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)#,cbar_size="2%")
   
    alpha = .5
    cbarLab = r'$^\circ$C / 30 yrs'
   
    m.ax = grid[0]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,twxTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,twxTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_alpha(1)
    cbar.draw_all()
    grid[0].set_ylabel("Tmin")
    grid[0].set_title("TopoWx",fontsize=12)
    cbar.set_label(cbarLab)
    
    drawCceBnds(m)
    #m.etopo(scale=10)
    
    m.ax = grid[1]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,prismTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,prismTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    grid[1].set_title("PRISM",fontsize=12)
    drawCceBnds(m)
    
    m.ax = grid[2]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,daymetTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,daymetTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    grid[2].set_title("Daymet",fontsize=12)
    drawCceBnds(m)
    
    m.ax = grid[3]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    grid[3].set_ylabel("Tmax")
    drawCceBnds(m)
    
    m.ax = grid[4]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    drawCceBnds(m)
    
    m.ax = grid[5]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,daymetTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,daymetTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    drawCceBnds(m)

    #plt.savefig('/projects/daymet2/docs/final_writeup/map_trends_19812010.png',dpi=150) 
    #plt.savefig()
    cfig.set_size_inches(8,7)
    plt.savefig('/projects/daymet2/docs/final_writeup/map_trends_19812010.png',dpi=250) 
    plt.show()

def plotTwxVsPRISMTrend2():
    
    dsTwxTmin = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmin19482012trend.tif')
    dsPrismTmin = RasterDataset('/projects/daymet2/cce_case_study/prism_files/trends/cce_prism4km_tmin_trend1948-2012.tif')
    dsTwxTmax = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmax19482012trend.tif')
    dsPrismTmax = RasterDataset('/projects/daymet2/cce_case_study/prism_files/trends/cce_prism4km_tmax_trend1948-2012.tif')
    
    twxTmin = dsTwxTmin.readAsArray()*65
    prismTmin = dsPrismTmin.readAsArray()*65
    twxTmax = dsTwxTmax.readAsArray()*65
    prismTmax = dsPrismTmax.readAsArray()*65
    
    print np.min([np.min(twxTmin),np.min(twxTmax),np.min(prismTmin),np.min(prismTmax)])
    print np.max([np.max(twxTmin),np.max(twxTmax),np.max(prismTmin),np.max(prismTmax)])
    
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
    
    levelsRed = np.arange(0,6.5,.5)
    levelsBlue = np.arange(-2,0.5,.5)
#    clrs = brewer2mpl.get_map('RdYlBu', 'Diverging', 11, reverse=True)
#    clrs = clrs.mpl_colors[3:]
    
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
    
    elev = getCceHillshade(m, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon)
  
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)#,cbar_size="2%")
   
    alpha = .5
   
    m.ax = grid[0]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,twxTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
    cf = m.contourf(xMap,yMap,twxTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_alpha(1)
    cbar.draw_all()
    grid[0].set_ylabel("Tmin")
    grid[0].set_title("TopoWx",fontsize=12)
    cbar.set_label(r'$^\circ$C / 65 yrs')
    
    drawCceBnds(m)
    #m.etopo(scale=10)
    
    m.ax = grid[1]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,prismTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
    cf = m.contourf(xMap,yMap,prismTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
    grid[1].set_title("PRISM",fontsize=12)
    drawCceBnds(m)
    
    m.ax = grid[2]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
    grid[2].set_ylabel("Tmax")
    drawCceBnds(m)
    
    m.ax = grid[3]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True)
    drawCceBnds(m)

    plt.savefig('/projects/daymet2/docs/final_writeup/map_trends_19482012.png',dpi=150) 
    #plt.savefig()
    plt.show()
    

def plotTwxVsPRISMTrend():
            
    dsTwxTmin = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmin19482012trend.tif')
    dsPrismTmin = RasterDataset('/projects/daymet2/cce_case_study/prism_files/trends/cce_prism4km_tmin_trend1948-2012.tif')
    
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
    
    m1 = Basemap(resolution='i',projection='tmerc', llcrnrlat=44,urcrnrlat=50,
        llcrnrlon=-116,urcrnrlon=-103,lon_0=(-116+-103)/2.0,lat_0=(44+50)/2.0)
    x1, y1 = m1(*np.meshgrid(lon,lat))
    
    dsElev = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_elev.tif')
    elev = dsElev.gdalDs.GetRasterBand(1).ReadAsArray()
    elev = np.ma.masked_equal(elev, dsElev.gdalDs.GetRasterBand(1).GetNoDataValue())
    print np.sum(~elev.mask)
    #plt.subplot(211)
    cf = plt.gcf()
    grid = ImageGrid(cf,211,nrows_ncols=(1,1),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)#,cbar_size="3%")
    
#    m.ax = grid[1]
#    cf = m.contourf(x,y,elev,100,cmap=cm.gist_earth)
#    cbar = grid[1].cax.colorbar(cf)
#    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)

    m1.ax = grid[0]
    m1.drawcountries(linewidth=1)
    m1.drawstates(linewidth=1)
    cf = m1.contourf(x1,y1,elev,100,cmap=cm.gist_earth)
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    #cbar = grid[0].cax.colorbar(cf)
    cbar.set_label(r'Elevation (m)')
    grid[0].set_title("Crown of the Contintent Ecosystem\nTmin Trends 1948-2012")
    m1.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1,color='black')
        
    
    #plt.subplot(212)
    cf = plt.gcf()
    grid = ImageGrid(cf,212,nrows_ncols=(1,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05,cbar_size="8%")
    
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
    
    #plt.savefig('/projects/daymet2/compare/CCE_Tmin_Trends.png')
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
    grid[1].set_title("TopoWx Stations")
    cf = m.contourf(x, y, tminTopoWx,colors=clrs,levels=levels,extend='both')
    
    m.ax = grid[2]
#    m.drawcountries()
#    m.drawstates()
#    m.drawcoastlines()
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    grid[2].set_ylabel("Tmax")
    cf = m.contourf(x, y, tmaxUshcn,colors=clrs,levels=levels,extend='both')
    
    m.ax = grid[3]
#    m.drawcountries()
#    m.drawstates()
#    m.drawcoastlines()
    m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
    cf = m.contourf(x, y, tmaxTopoWx,colors=clrs,levels=levels,extend='both')
    
    #cf = plt.gcf()
    #cf.set_size_inches(8*1.5,6*1.5)
    
    #plt.savefig('/projects/daymet2/docs/final_writeup/conus_trends/trendMaps.png',dpi=150)
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

def calcConusTrendsTopoWx():
    
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stns = stnda.stns
    stn_mask = np.nonzero(np.isnan(stns[BAD]))[0]
    stns = stns[stn_mask]
    tairAnn = stnda.ds.variables['tmin_ann'][:,stn_mask]
    
    uYrs = np.unique(stnda.days[YEAR])
    baseMask = np.logical_and(uYrs >= 1961,uYrs <= 1990)
    
    normsAnn = np.mean(tairAnn[baseMask,:],axis=0)
    tairAnom = tairAnn - normsAnn
        
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    gridMask = dsGrid.gdalDs.ReadAsArray() != 19

    yGrid,xGrid = dsGrid.getCoordGrid1d()
    yGrid = np.sort(yGrid)
    
    allAnomGridHomog = np.zeros((uYrs.size,yGrid.size,xGrid.size))
    allAnomGridHomog = np.ma.masked_array(allAnomGridHomog,np.isnan(allAnomGridHomog))
       
    for i in np.arange(uYrs.size):
    
        print uYrs[i]
        anomGridHomog = griddata(stns[LON],stns[LAT],tairAnom[i,:],xGrid,yGrid)
        anomGridHomog = np.flipud(anomGridHomog)
        anomGridHomog.mask = np.logical_or(gridMask,anomGridHomog.mask)
        allAnomGridHomog[i,:,:] = anomGridHomog
            
    np.save('/projects/daymet2/docs/final_writeup/conus_trends/annAnomHomogTopoWxTmin.npy', np.ma.filled(allAnomGridHomog, -9999))
    
    trendsHomog = np.zeros((allAnomGridHomog.shape[1],allAnomGridHomog.shape[2]))
    
    schk = status_check(trendsHomog.size,10000)
    for r in np.arange(allAnomGridHomog.shape[1]):
    
        for c in np.arange(allAnomGridHomog.shape[2]):
            
            if np.ma.is_masked(allAnomGridHomog[0,r,c]):
                trendsHomog[r,c] = -9999
            else:
                trendsHomog[r,c] = stats.linregress(uYrs,allAnomGridHomog[:,r,c])[0]
            
            schk.increment()
    np.save('/projects/daymet2/docs/final_writeup/conus_trends/trendsHomogTopoWxTmin.npy', trendsHomog)

def calcConusTrends():
    stndaUS = ushcn.StationDataUSHCN('/projects/daymet2/station_data/ushcn/ushcn.nc')
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
    
    anomRaw = np.zeros((uYrs.size,stns.size))
    anomRaw = np.ma.masked_array(anomRaw,np.isnan(anomRaw))
    
    for astn,x in zip(stns,np.arange(stns.size)):
        obsFLs = stndaUS.loadObs(astn[STN_ID], 'FLs.52i_tmin')
        obsFLs = tairAgg.mthlyToAnn(obsFLs)
        
        obsRaw = stndaUS.loadObs(astn[STN_ID], 'raw_tmin')
        obsRaw = tairAgg.mthlyToAnn(obsRaw)
        
        obsFLs = obsFLs - np.ma.mean(obsFLs[baseMask])
        obsRaw = obsRaw - np.ma.mean(obsRaw[baseMask])
        
        anomFLs[:,x] = obsFLs
        anomRaw[:,x] = obsRaw
    
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
        
        anomGridRaw = griddata(stns[LON],stns[LAT],anomRaw[i,:],xGrid,yGrid)
        anomGridRaw = np.flipud(anomGridRaw)
        anomGridRaw.mask = np.logical_or(gridMask,anomGridRaw.mask)
        allAnomGridRaw[i,:,:] = anomGridRaw
    
    np.save('/projects/daymet2/docs/final_writeup/conus_trends/annAnomHomogUshcnTmin.npy', np.ma.filled(allAnomGridHomog, -9999))
    np.save('/projects/daymet2/docs/final_writeup/conus_trends/annAnomRawUshcnTmin.npy', np.ma.filled(allAnomGridRaw, -9999))
    
    trendsHomog = np.zeros((allAnomGridHomog.shape[1],allAnomGridHomog.shape[2]))
    trendsRaw = np.zeros((allAnomGridHomog.shape[1],allAnomGridHomog.shape[2]))
    
    schk = status_check(trendsHomog.size,10000)
    for r in np.arange(allAnomGridHomog.shape[1]):
    
        for c in np.arange(allAnomGridHomog.shape[2]):
            
            if np.ma.is_masked(allAnomGridHomog[0,r,c]):
                trendsHomog[r,c] = -9999
                trendsRaw[r,c] = -9999
            else:
                trendsHomog[r,c] = stats.linregress(uYrs,allAnomGridHomog[:,r,c])[0]
                trendsRaw[r,c] = stats.linregress(uYrs,allAnomGridRaw[:,r,c])[0]
            
            schk.increment()
    np.save('/projects/daymet2/docs/final_writeup/conus_trends/trendsHomogUshcnTmin.npy', trendsHomog)
    np.save('/projects/daymet2/docs/final_writeup/conus_trends/trendsRawUshcnTmin.npy', trendsRaw)


def plotLSTvsAir():
            
    dsTmin = Dataset('/projects/daymet2/interp_output/h05v01/h05v01_tmin.nc')
    dsTmax = Dataset('/projects/daymet2/interp_output/h05v01/h05v01_tmax.nc')
    dsTminLst = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin08.nc')
    dsTmaxLst = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax08.nc')
    
    tmin = dsTmin.variables['tmin_normal'][7,:,:]
    tmax = dsTmax.variables['tmax_normal'][7,:,:]
    
    lat = dsTmin.variables['lat'][:]
    lon = dsTmax.variables['lon'][:]
    latLst = dsTminLst.variables['lat'][:]
    lonLst = dsTmaxLst.variables['lon'][:]
    
    latMask = np.logical_and(np.round(latLst,5) >= np.round(lat[-1],5),np.round(latLst,5) <= np.round(lat[0],5))
    lonMask = np.logical_and(np.round(lonLst,5) >= np.round(lon[0],5),np.round(lonLst,5) <= np.round(lon[-1],5))
    
    tminLst = dsTminLst.variables['tmin08'][latMask,lonMask]
    tmaxLst = dsTmaxLst.variables['tmax08'][latMask,lonMask]
        
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
    lon_0 = (llcrnrlon+urcrnrlon)/2.0
    lat_0 = (llcrnrlat+urcrnrlat)/2.0
#        
#    print "Mapping data...."
    m = Basemap(resolution='i',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
   
    cmap = cm.jet
    
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(2,2),cbar_mode="each",cbar_location="right",axes_pad=0.5,cbar_pad=0.05)#,axes_pad=0.05,cbar_pad=0.05)#,cbar_size="3%")
    
    grid[0].set_ylabel("Tmin")
    grid[0].set_title("USHCN Stations")
    
    m.ax = grid[0]
    img = m.imshow(tminLst,origin='upper')
    cbar = plt.colorbar(img,cax = grid.cbar_axes[0])
    cbar.set_label("$^\circ$C")
    m.ax.set_ylabel("Tmin")
    m.ax.set_title("Surface",fontsize=12)
    
    m.ax = grid[1]
    img = m.imshow(tmin,origin='upper')
    cbar = plt.colorbar(img,cax = grid.cbar_axes[1])
    cbar.set_label("$^\circ$C")
    m.ax.set_title("Air",fontsize=12)
    
    m.ax = grid[2]
    img = m.imshow(tmaxLst,origin='upper')
    cbar = plt.colorbar(img,cax = grid.cbar_axes[2])
    cbar.set_label("$^\circ$C")
    m.ax.set_ylabel("Tmax")
    
    m.ax = grid[3]
    img = m.imshow(tmax,origin='upper')
    cbar = plt.colorbar(img,cax = grid.cbar_axes[3])
    cbar.set_label("$^\circ$C")
    plt.suptitle("August Normal: 1981-2010")
    plt.show()
    
    #cbar.set_label(r'$^\circ$C / 65 yrs')
    #m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    #grid[0].set_title("TopoWx")
    
#    
#    cf = plt.gcf()
#    grid = ImageGrid(cf,211,nrows_ncols=(1,3),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)#,cbar_size="3%")
#    
#    m.ax = grid[0]
#    cf = m.contourf(x,y,twxTmin,cmap=cmap,levels=levels)
#    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
#    #cbar.set_label(r'$^\circ$C / 65 yrs')
#    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
#    grid[0].set_title("TopoWx")
#    
#    m.ax = grid[1]
#    cf = m.contourf(x,y,prismTmin,cmap=cmap,levels=levels)
#    cbar = plt.colorbar(cf, cax = grid.cbar_axes[1])
#    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
#    grid[1].set_title("PRISM")
#    
#    m.ax = grid[2]
#    cf = m.contourf(x,y,daymetTmin,cmap=cmap,levels=levels)
#    cbar = plt.colorbar(cf, cax = grid.cbar_axes[2])
#    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
#    grid[2].set_title("Daymet")
#    
#    clrs = brewer2mpl.get_map('RdBu', 'Diverging', 8, reverse=True)
#    clrs = clrs.mpl_colors
#    clrs[3] = "grey"
#    clrs[4] = "grey"
#    minTmin = np.min([np.min(twxTmin-prismTmin),np.min(twxTmin-daymetTmin)])
#    maxTmin = np.max([np.max(twxTmin-prismTmin),np.max(twxTmin-daymetTmin)])
#    print minTmin,maxTmin
#    levels = np.arange(-4,5)
#    
#    cf = plt.gcf()
#    grid = ImageGrid(cf,212,nrows_ncols=(1,2),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)
#    
#    m.ax = grid[0]
#    cf = m.contourf(x,y,twxTmin-prismTmin,colors=clrs,levels=levels,extend='both')
#    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
#    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
#    grid[0].set_title("TopoWx-PRISM")
#    
#    m.ax = grid[1]
#    cf = m.contourf(x,y,twxTmin-daymetTmin,colors=clrs,levels=levels,extend='both')
#    cbar = plt.colorbar(cf, cax = grid.cbar_axes[1])
#    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
#    grid[1].set_title("TopoWx-Daymet")
#    
#    plt.show()


def caCoast_grtPlns_MaeVsSE():
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
    
    climDivsGP = np.sort(np.array([1407,1404,1401,3401,2508,2507,2502,2501,3201,3204,3207]))
    stn_maskGP = np.nonzero(np.logical_and(np.in1d(stnda.stns[NEON], climDivsGP, False),np.isnan(stnda.stns[BAD])))[0]   
    
    stn_maskCA = np.nonzero(np.logical_and(stnda.stns[NEON]==404,np.isnan(stnda.stns[BAD])))[0]    
    
    maeDlyCA = np.mean(np.abs(stnda.ds.variables['xvalfnl_mae_dly'][0:12,stn_maskCA]),axis=1)
    maeCA = np.mean(np.abs(stnda.ds.variables['xval_err_mthly'][0:12,stn_maskCA]),axis=1)
    seCA = np.mean(stnda.ds.variables['xval_stderr_mthly'][0:12,stn_maskCA],axis=1)
    
    maeDlyGP = np.mean(np.abs(stnda.ds.variables['xvalfnl_mae_dly'][0:12,stn_maskGP]),axis=1)
    maeGP = np.mean(np.abs(stnda.ds.variables['xval_err_mthly'][0:12,stn_maskGP]),axis=1)
    seGP = np.mean(stnda.ds.variables['xval_stderr_mthly'][0:12,stn_maskGP],axis=1)
    
#    def plotMaeSe(mae,maeDly,se,ax):
#    
#        line1, = ax.plot(mae,'o-',color='#377EB8')
#        line2, = ax.plot(maeDly,'s-',color='#377EB8')
#        ax.set_ylabel("MAE ($^\circ$C)",color='#377EB8')
#        ax.set_xticks(np.arange(12))
#        ax.set_xlim(-0.5,11.5)
#        ax.set_xticklabels(('JAN', 'FEB', 'MAR', 'APR', 'MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'))
#        # Make the y-axis label and tick labels match the line color.
#        for tl in ax.get_yticklabels():
#            tl.set_color('#377EB8')
#    
#        axTwin = ax.twinx()
#        line3, = axTwin.plot(se,'ro-',color='#E41A1C')
#        axTwin.set_ylabel(r"$\bar \sigma_k$"+"($^\circ$C)",color='#E41A1C')
#        axTwin.set_xlim(-0.5,11.5)
#        for tl in axTwin.get_yticklabels():
#            tl.set_color('#E41A1C')
#        return line1,line2,line3
    
    def plotMaeSe(mae,maeDly,se,ax):
    
        line1, = ax.plot(mae,'o-',color='k')#color='#377EB8')
        line2, = ax.plot(maeDly,'s-',color='k')#color='#377EB8')
        line3, = ax.plot(se,'v--',color='k')#color='#E41A1C')
        ax.set_ylabel("$^\circ$C")
        ax.set_xticks(np.arange(12))
        ax.set_xlim(-0.5,11.5)
        ax.set_xticklabels(('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
        ax.set_xlabel("Month")
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        return line1,line2,line3
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #plt.subplot(121)
    line1,line2,line3 = plotMaeSe(maeCA, maeDlyCA, seCA, ax1)
    ax1.set_ylim(0.4,2.1)
    plt.sca(ax1)
    plt.title("a.",loc="left")
    line1,line2,line3 = plotMaeSe(maeGP, maeDlyGP, seGP, ax2)
    ax2.set_ylim(0.4,2.1)
    ax2.set_ylabel("")
    plt.sca(ax2)
    plt.title("b.",loc="left")
    plt.tight_layout()
    plt.legend((line1,line2,line3),("Normals MAE","Daily MAE",r"$\bar \sigma_k$"),fontsize=12,loc=0)
    #w (horizontal), h (vertical 
    f.set_size_inches(9,4.5)
    plt.savefig('/projects/daymet2/docs/final_writeup/caCoast_GrtPlns.png',dpi=300)
    plt.show()

def plotCcePredictors():
            
    dsElev = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_elev.tif')
    dsTdi = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_tdi.tif')
    dsTmin1 = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_tmin01.tif')
    dsTmin2 = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_tmin08.tif')
    dsTmax1 = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_tmax01.tif')
    dsTmax2 = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_tmax08.tif')
    
    elev = dsElev.readAsArray()
    tdi = dsTdi.readAsArray()
    tmin1 = dsTmin1.readAsArray()
    tmin2 = dsTmin2.readAsArray()
    tmax1 = dsTmax1.readAsArray()
    tmax2 = dsTmax2.readAsArray()
    
    
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
    m = Basemap(resolution='i',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=lon_0,lat_0=lat_0)
    x, y = m(*np.meshgrid(lon,lat))
    
    
    cf = plt.gcf()
    
    def plotData(adata,cmap,cbar_label,gridCell,m,title):
        m.ax = gridCell
        plt.sca(gridCell)
        #m.drawcountries()
        #m.drawstates()
        m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)#,linewidth=1)
        cf = m.contourf(x,y,adata,150,cmap=cmap)
        cbar = gridCell.cax.colorbar(cf)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.set_ylabel(cbar_label,fontsize=9)
        
        plt.title(title,loc="left",fontsize=9)
        
        
    axes_pad = .7
    grid = ImageGrid(cf,311,nrows_ncols=(1,2),axes_pad=axes_pad,cbar_mode="each",cbar_location="right",cbar_pad=0.02)
    plotData(elev*10**-3, plt.cm.gist_earth, "km", grid[0], m,'a.')
    plotData(tdi, plt.cm.gist_earth, "unitless", grid[1], m,'b.')    
    
    grid = ImageGrid(cf,312,nrows_ncols=(1,2),axes_pad=axes_pad,cbar_mode="each",cbar_location="right",cbar_pad=0.02)
    plotData(tmin1, CMAP_ESRI_PRCP, "($^\circ$C)", grid[0], m,'c.')
    plotData(tmin2, CMAP_ESRI_PRCP, "($^\circ$C)", grid[1], m,'d.')
    
    grid = ImageGrid(cf,313,nrows_ncols=(1,2),axes_pad=axes_pad,cbar_mode="each",cbar_location="right",cbar_pad=0.02)
    plotData(tmax1, CMAP_ESRI_PRCP, "($^\circ$C)", grid[0], m,'e.')
    plotData(tmax2, CMAP_ESRI_PRCP, "($^\circ$C)", grid[1], m,'f.')       
    
    plt.tight_layout(h_pad=0.1)
    plt.savefig('/projects/daymet2/docs/final_writeup/cce_predictor_maps.png',dpi=150)
    plt.show()


def plotCcePredictors2():
            
    dsElev = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_elev.tif')
    dsTdi = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_tdi.tif')
    dsTmin = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_tmin08.tif')
    dsTmax = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_tmax01.tif')
    
    elev = dsElev.readAsArray()
    tdi = dsTdi.readAsArray()
    tmin = dsTmin.readAsArray()
    tmax = dsTmax.readAsArray()
    
    dsGrid = dsElev
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
            
    print "Mapping data...."
    m = Basemap(resolution='h',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=-111,lat_0=0)
    x, y = m(*np.meshgrid(lon,lat))
    
    
    hillshd = getCceHillshade(m, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon)
    
    cfig = plt.gcf()
    
    def plotData(adata,cmap,cbar_label,grid,gridCellNum,m,title):
        m.ax = grid[gridCellNum]
        plt.sca(grid[gridCellNum])
        m.imshow(hillshd,cmap=cm.gray)
        m.drawcountries()
        m.drawstates()
        m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)#,linewidth=1)
        cf = m.contourf(x,y,adata,150,cmap=cmap,alpha=.5,antialiased=True)
        cf = m.contourf(x,y,adata,150,cmap=cmap,alpha=.5,antialiased=True)
        cbar = grid[gridCellNum].cax.colorbar(cf)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.set_ylabel(cbar_label,fontsize=9)
        
        #cbar = plt.colorbar(cf, cax = grid.cbar_axes[gridCellNum])
        cbar.set_alpha(1)
        #cbar.draw_all()
        
        plt.title(title,loc="left",fontsize=12)
    
    axes_pad = .7
    grid = ImageGrid(cfig,211,nrows_ncols=(1,2),axes_pad=axes_pad,cbar_mode="each",cbar_location="right",cbar_pad=0.02)
    plotData(elev, plt.cm.gist_earth, "meters", grid,0, m,'Elevation')#elev*10**-3
    plotData(tdi, plt.cm.gist_earth, "unitless", grid,1, m,'TDI: Ridge vs. Valley Index')    
    
    grid = ImageGrid(cfig,212,nrows_ncols=(1,2),axes_pad=axes_pad,cbar_mode="each",cbar_location="right",cbar_pad=0.02)
    plotData(tmin, CMAP_ESRI_PRCP, "$^\circ$C", grid,0, m,'MODIS LST: Aug. Normal Tmin')
    plotData(tmax, CMAP_ESRI_PRCP, "$^\circ$C", grid,1, m,'MODIS LST: Jan. Normal Tmax')     
    
    plt.tight_layout()
    cfig.set_size_inches(9,7)
    plt.savefig('/projects/daymet2/docs/nws_workshop/cce_predictor_maps.png',dpi=250)
    plt.show()
 
def plotCcePredictorR2():
    dsMask = RasterDataset('/projects/daymet2/dem/interp_grids/cce/crp_cce_us_mask.tif')
    cceMask = dsMask.readAsArray().data
    
#    dsMask = input_raster('/projects/daymet2/dem/interp_grids/cce/crp_cce_us_mask.tif')
#    cceMask = dsMask.readEntireRaster()
        
    def maskStnsToCCE(stnDa):
    
        stns = stnDa.stns[np.isnan(stnDa.stns[BAD])]
        
        cceStnMask = np.zeros(stns.size,dtype=np.bool)
        
        for stn,x in zip(stns,np.arange(stns.size)):
            
            try:
                row,col = dsMask.getRowCol(stn[LON], stn[LAT])
                #col,row = dsMask.getGridCellOffset(stn[LON], stn[LAT])
                
                cceStnMask[x] = cceMask[row,col]
            except OutsideExtent:
                pass

        return stns[cceStnMask]
    
    stnDaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stnsCceTmin = maskStnsToCCE(stnDaTmin)
    
    stnDaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
    stnsCceTmax = maskStnsToCCE(stnDaTmax)
    
    def getR2(stns):
    
        r2Elev = []
        r2Lst = []
        lapseRate = []
        
        for mth in np.arange(1,13):
            
            elev = stns[ELEV]
            lst = stns[get_lst_varname(mth)]
            tair = stns[get_norm_varname(mth)]
            
            slope,intercept,rval,pval,stderr = stats.linregress(elev, tair)
            r2Elev.append(rval**2)
            lapseRate.append(slope*1000)
            
            r2Lst.append((stats.linregress(lst, tair)[2])**2)
        
        
        return r2Elev,lapseRate,r2Lst
        
    r2ElevTmin,lrTmin,r2LstTmin = getR2(stnsCceTmin)
    r2ElevTmax,lrTmax,r2LstTmax = getR2(stnsCceTmax)
    
    print lrTmin
    print lrTmax
    
    blueCol = '#377EB8'
    redCol = '#E41A1C'
    
    print np.min(r2ElevTmax),np.arange(1,13)[np.argmin(r2ElevTmax)]
    
    plt.plot(r2ElevTmin,'o-',color=blueCol)
    plt.plot(r2LstTmin,'s--',color=blueCol,lw=1.5)
    plt.plot(r2ElevTmax,'o-',color=redCol)
    plt.plot(r2LstTmax,'s--',color=redCol,lw=1.5)
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    
    plt.xlim((-0.5,11.5))
    plt.xticks(np.arange(12),('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),fontsize=9)
        
    r2_str = r"$R^2$"
    plt.ylabel(r2_str,fontsize=10)
    
    cf = plt.gcf()
    cf.set_size_inches(8*.6,6*.6)
    
    ax.tick_params(labelsize=9)
    plt.legend(("Tmin Elev.","Tmin LST","Tmax Elev.","Tmax LST"),fontsize=9,ncol=2,loc='upper center',bbox_to_anchor=(0.5, 1.14))
    #plt.savefig('/projects/daymet2/docs/final_writeup/cce_predictor_r2.png',dpi=150)
    plt.show()

def cceMae():
    dsMask = RasterDataset('/projects/daymet2/dem/interp_grids/cce/crp_cce_us_mask.tif')
    cceMask = dsMask.readAsArray().data
            
    def maskStnsToCCE(stnDa):
            
        stns = stnDa.stns
        cceStnMask = np.zeros(stns.size,dtype=np.bool)
        
        for stn,x in zip(stns,np.arange(stns.size)):
            
            try:
                row,col = dsMask.getRowCol(stn[LON], stn[LAT])
                #col,row = dsMask.getGridCellOffset(stn[LON], stn[LAT])
                
                cceStnMask[x] = cceMask[row,col]
            except OutsideExtent:
                pass

        return np.logical_and(cceStnMask,np.isnan(stnDa.stns[BAD]))#stns[cceStnMask]
    
    stnDaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stnsCceTminMask = maskStnsToCCE(stnDaTmin)
    
    stnDaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
    stnsCceTmaxMask = maskStnsToCCE(stnDaTmax)
    
    print np.sum(stnsCceTminMask),np.sum(stnsCceTmaxMask)
    
    maeDlyTmin = np.mean(np.abs(stnDaTmin.ds.variables['xvalfnl_mae_dly'][0:12,stnsCceTminMask]),axis=1)
    maeTmin = np.mean(np.abs(stnDaTmin.ds.variables['xvalfnl_mae_norm'][0:12,stnsCceTminMask]),axis=1)
    seTmin = np.mean(stnDaTmin.ds.variables['xval_stderr_mthly'][0:12,stnsCceTminMask],axis=1)
    
    maeDlyTmax = np.mean(np.abs(stnDaTmax.ds.variables['xvalfnl_mae_dly'][0:12,stnsCceTmaxMask]),axis=1)
    maeTmax = np.mean(np.abs(stnDaTmax.ds.variables['xvalfnl_mae_norm'][0:12,stnsCceTmaxMask]),axis=1)
    seTmax = np.mean(stnDaTmax.ds.variables['xval_stderr_mthly'][0:12,stnsCceTmaxMask],axis=1)
    
    #########################################################
    #Annual Error stats
    #########################################################
    maeDlyAnnTmin = np.mean(np.abs(stnDaTmin.ds.variables['xvalfnl_mae_dly'][12,stnsCceTminMask]))
    maeAnnTmin = np.mean(np.abs(stnDaTmin.ds.variables['xval_err_mthly'][12,stnsCceTminMask]))
    
    maeDlyAnnTmax = np.mean(np.abs(stnDaTmax.ds.variables['xvalfnl_mae_dly'][12,stnsCceTmaxMask]))
    maeAnnTmax = np.mean(np.abs(stnDaTmax.ds.variables['xval_err_mthly'][12,stnsCceTmaxMask]))
    
    print maeAnnTmin,maeDlyAnnTmin
    print maeAnnTmax,maeDlyAnnTmax
    
    def plotMaeSe(mae,maeDly,se,ax):
    
        line1, = ax.plot(mae,'o-',color='#377EB8')
        line2, = ax.plot(maeDly,'s-',color='#377EB8')
        line3, = ax.plot(se,'ro-',color='#E41A1C')
        ax.set_ylabel("($^\circ$C)")
        ax.set_xticks(np.arange(12))
        ax.set_xlim(-0.5,11.5)
        ax.set_xticklabels(('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
        ax.set_xlabel("Month")
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        return line1,line2,line3
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #plt.subplot(121)
    line1,line2,line3 = plotMaeSe(maeTmin, maeDlyTmin, seTmin, ax1)
    ax1.set_ylim(0.5,1.8)
    plt.sca(ax1)
    plt.title("a.",loc="left")
    line1,line2,line3 = plotMaeSe(maeTmax, maeDlyTmax, seTmax, ax2)
    ax2.set_ylim(0.5,1.8)
    ax2.set_ylabel("")
    plt.sca(ax2)
    plt.title("b.",loc="left")
    plt.tight_layout()
    plt.legend((line1,line2,line3),("Normal MAE","Daily MAE",r"$\bar \sigma_k$"),fontsize=12,loc=0)
    #w (horizontal), h (vertical 
    f.set_size_inches(9,4.5)
    #plt.savefig('/projects/daymet2/docs/final_writeup/caCoast_GrtPlns.png',dpi=150)
    plt.show()

def cceBiasVs():
    
    dbHomog = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc',startend_ymd=(19810101,20101231))
    dbRaw = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc',startend_ymd=(19810101,20101231))
    
    stnDaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stnsTmin = stnDaTmin.stns[maskStnsToCCE(stnDaTmin)]
    stnDaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
    stnsTmax = stnDaTmax.stns[maskStnsToCCE(stnDaTmax)]
    
    delStns = np.array(['GHCN_USC00241202','GHCN_USC00242576','GHCN_USC00242812','GHCN_USC00245038','GHCN_USC00245164','GHCN_USC00246302','GHCN_USC00246307','GHCN_USC00244563'])
    stnsTmin = stnsTmin[~np.in1d(stnsTmin[STN_ID], delStns, True)]
    stnsTmax = stnsTmax[~np.in1d(stnsTmax[STN_ID], delStns, True)]
    
    dRastTmin = MultiDaymetTileRaster('/projects/daymet2/daymet_oakridge/multi_yr_tiles/','tmin')
    dRastTmax = MultiDaymetTileRaster('/projects/daymet2/daymet_oakridge/multi_yr_tiles/','tmax')
    
    pRastTmin = PrismTileRaster('/projects/daymet2/prism/4km_daily/netcdf/prism4km_tmin.nc', 'tmin')
    pRastTmax = PrismTileRaster('/projects/daymet2/prism/4km_daily/netcdf/prism4km_tmax.nc', 'tmax')
    
    tRastTmin = MultiTwxTileRaster('/stage/climate/topowx_tile_output/', 'tmin')
    tRastTmax = MultiTwxTileRaster('/stage/climate/topowx_tile_output/', 'tmax')

    startYr = 1981
    endYr = 2010
    
    dMaskDays = np.logical_and(dRastTmin.days[YEAR]>=startYr,dRastTmin.days[YEAR]<=endYr)
    pMaskDays = np.logical_and(pRastTmin.days[YEAR]>=startYr,pRastTmin.days[YEAR]<=endYr)
    tMaskDays = np.logical_and(tRastTmin.days[YEAR]>=startYr,tRastTmin.days[YEAR]<=endYr)
    
    def shiftTair(tair):
        tairShift = np.roll(tair,-1)
        tairShift[-1] = np.nan
        return tairShift
        
    
    def getErrorStats(aRast,aDb,stns,dayMask,varname,useDbStn=True,shiftBack=False):
        
        bias = []
        mae = []
        
        mthBias = {}
        mthMae = {}
        
        mths = np.arange(1,13)
        for mth in mths:
            mthBias[mth] = []
            mthMae[mth] = []
        
        for stn in stns:
            
            if useDbStn:
                stn = aDb.stns[aDb.stn_ids==stn[STN_ID]][0]
            tairStn = aDb.load_all_stn_obs_var(stn[STN_ID], varname)[0]
            tairInterp = aRast.getTimeSeries(stn[LON], stn[LAT])[dayMask]
            
            if shiftBack:
                tairInterp = shiftTair(tairInterp)
            
            maskOverlap = np.logical_and(np.isfinite(tairStn),np.isfinite(tairInterp))
            
            if np.sum(maskOverlap) == 0:
                raise Exception("No obs overlap for "+stn[STN_ID])
        
            tairStn = tairStn[maskOverlap]
            tairInterp = tairInterp[maskOverlap]
            days = aDb.days[maskOverlap]
            difs = tairInterp - tairStn
            
            bias.append(np.mean(difs))
            mae.append(np.mean(np.abs(difs)))
            
            for mth in mths:
                
                if np.sum(days[MONTH]==mth) == 0:
                    raise Exception("No obs overlap for "+str(mth)+" "+stn[STN_ID])
                
                mthBias[mth].append(np.mean(difs[days[MONTH]==mth]))
                mthMae[mth].append(np.mean(np.abs(difs[days[MONTH]==mth])))
        
        mthBias[13] = bias
        mthMae[13] = mae
        
#        mbias = np.mean(bias)
#        mab = np.mean(np.abs(bias))
#        mmae = np.mean(mae)
        
        mbias = [np.mean(mthBias[mth]) for mth in np.arange(1,14)]
        mab = [np.mean(np.abs(mthBias[mth])) for mth in np.arange(1,14)]
        mmae = [np.mean(np.abs(mthMae[mth])) for mth in np.arange(1,14)]
        
        return mbias,mab,mmae
    
    print "Calculating Twx Stats..."
    biasTminTwx,mabTminTwx,maeTminTwx = getErrorStats(tRastTmin, dbHomog, stnsTmin, tMaskDays, 'tmin', useDbStn=False)
    biasTmaxTwx,mabTmaxTwx,maeTmaxTwx = getErrorStats(tRastTmax, dbHomog, stnsTmax, tMaskDays, 'tmax', useDbStn=False)
    print "Calculating Daymet Stats..."
    biasTminDaymet,mabTminDaymet,maeTminDaymet = getErrorStats(dRastTmin, dbRaw, stnsTmin, dMaskDays, 'tmin', useDbStn=True,shiftBack=True)
    biasTmaxDaymet,mabTmaxDaymet,maeTmaxDaymet = getErrorStats(dRastTmax, dbRaw, stnsTmax, dMaskDays, 'tmax', useDbStn=True,shiftBack=True)
    print "Calculating Prism Stats..."
    biasTminPrism,mabTminPrism,maeTminPrism = getErrorStats(pRastTmin, dbRaw, stnsTmin, pMaskDays, 'tmin', useDbStn=True,shiftBack=True)
    biasTmaxPrism,mabTmaxPrism,maeTmaxPrism = getErrorStats(pRastTmax, dbRaw, stnsTmax, pMaskDays, 'tmax', useDbStn=True,shiftBack=True)
    
    biasTmin = [biasTminTwx,biasTminDaymet,biasTminPrism]
    mabTmin = [mabTminTwx,mabTminDaymet,mabTminPrism]
    maeTmin = [maeTminTwx,maeTminDaymet,maeTminPrism]
    
    biasTmax = [biasTmaxTwx,biasTmaxDaymet,biasTmaxPrism]
    mabTmax = [mabTmaxTwx,mabTmaxDaymet,mabTmaxPrism]
    maeTmax = [maeTmaxTwx,maeTmaxDaymet,maeTmaxPrism]
    
    allStats = (biasTmin,mabTmin,maeTmin,biasTmax,mabTmax,maeTmax)
    pickle.dump(allStats,open('/projects/daymet2/cce_case_study/stn_compare_shiftall.pickle','wb'))
    
def cceBiasVsPlot():
    errStats = pickle.load(open('/projects/daymet2/cce_case_study/stn_compare.pickle'))
    #errStats = pickle.load(open('/projects/daymet2/cce_case_study/stn_compare_shiftprism.pickle'))
    errStatsShift = pickle.load(open('/projects/daymet2/cce_case_study/stn_compare_shiftall.pickle'))
    
    #errStats = [np.array(aStat) for aStat in errStats]
    biasTmin,mabTmin,maeTmin,biasTmax,mabTmax,maeTmax = errStats
    biasShiftTmin,mabShiftTmin,maeShiftTmin,biasShiftTmax,mabShiftTmax,maeShiftTmax = errStatsShift
    
    statTminTwx,statTminDaymet,statTminPrism = [np.array(aStat) for aStat in maeTmin]
    statTmaxTwx,statTmaxDaymet,statTmaxPrism = [np.array(aStat) for aStat in maeTmax]
    
    statShiftTminTwx,statShiftTminDaymet,statShiftTminPrism = [np.array(aStat) for aStat in maeShiftTmin]
    statShiftTmaxTwx,statShiftTmaxDaymet,statShiftTmaxPrism = [np.array(aStat) for aStat in maeShiftTmax]
    
    statTminPrism = statShiftTminPrism
    statTmaxPrism = statShiftTmaxPrism
#    statTminDaymet = statShiftTminDaymet
#    statTmaxDaymet = statShiftTmaxDaymet

    print statTminTwx[-1]
    print statTminPrism[-1]
    print statTminDaymet[-1]
    
    print statTmaxTwx[-1]
    print statTmaxPrism[-1]
    print statTmaxDaymet[-1]
    
    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, sharey='row',sharex=False)
    
    bwidth = .25
    xlim = (-.25,12.90)
        
    plt.sca(ax1)
    plt.bar(np.arange(statTminTwx.size),statTminTwx,width=bwidth,color="k")
    plt.bar(np.arange(statTminPrism.size)+bwidth,statTminPrism,width=bwidth,color="grey")
    plt.bar(np.arange(statTminDaymet.size)+(bwidth*2),statTminDaymet,width=bwidth,color="w",hatch="//")
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(statTminTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    plt.ylim((0,3.0))
    plt.yticks(np.arange(0,3.25,.25),fontsize=10)
    plt.ylabel("$^\circ$C",fontsize=10)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("a.",loc="left")
    
    plt.sca(ax2)
    plt.bar(np.arange(statTmaxTwx.size),statTmaxTwx,width=bwidth,color="k")
    plt.bar(np.arange(statTmaxPrism.size)+bwidth,statTmaxPrism,width=bwidth,color="grey")
    plt.bar(np.arange(statTmaxDaymet.size)+(bwidth*2),statTmaxDaymet,width=bwidth,color="w",hatch="//")
    plt.ylim((0,3.0))
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(statTmaxTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("b.",loc="left")
    
    ###################################
    statTminTwx,statTminDaymet,statTminPrism = [np.array(aStat) for aStat in mabTmin]
    statTmaxTwx,statTmaxDaymet,statTmaxPrism = [np.array(aStat) for aStat in mabTmax]
    
    statShiftTminTwx,statShiftTminDaymet,statShiftTminPrism = [np.array(aStat) for aStat in mabShiftTmin]
    statShiftTmaxTwx,statShiftTmaxDaymet,statShiftTmaxPrism = [np.array(aStat) for aStat in mabShiftTmax]
    
    statTminPrism = statShiftTminPrism
    statTmaxPrism = statShiftTmaxPrism
#    statTminDaymet = statShiftTminDaymet
#    statTmaxDaymet = statShiftTmaxDaymet

#    print statTminTwx
#    print statTminPrism
#    print statTminDaymet
    
    plt.sca(ax3)
    plt.bar(np.arange(statTminTwx.size),statTminTwx,width=bwidth,color="k")
    plt.bar(np.arange(statTminPrism.size)+bwidth,statTminPrism,width=bwidth,color="grey")
    plt.bar(np.arange(statTminDaymet.size)+(bwidth*2),statTminDaymet,width=bwidth,color="w",hatch="//")
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(statTminTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    plt.ylim((0,1.50))
    plt.yticks(np.arange(0,1.6,.1),fontsize=10)
    plt.ylabel("$^\circ$C",fontsize=10)
    ax3.set_axisbelow(True)
    ax3.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("c.",loc="left")
    
    plt.sca(ax4)
    plt.bar(np.arange(statTmaxTwx.size),statTmaxTwx,width=bwidth,color="k")
    plt.bar(np.arange(statTmaxPrism.size)+bwidth,statTmaxPrism,width=bwidth,color="grey")
    plt.bar(np.arange(statTmaxDaymet.size)+(bwidth*2),statTmaxDaymet,width=bwidth,color="w",hatch="//")
    plt.legend(("TopoWx","PRISM","Daymet"),fontsize=10)
    plt.ylim((0,1.50))
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(statTmaxTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    ax4.set_axisbelow(True)
    ax4.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("d.",loc="left")
    
    plt.tight_layout()
    
    f.set_size_inches(8,4.5)
    #plt.savefig('/projects/daymet2/docs/final_writeup/cce_stn_compare.png',dpi=250)
    plt.show()

def cceBiasVsPlot2():
    errStats = pickle.load(open('/projects/daymet2/cce_case_study/stn_compare.pickle'))
    #errStats = [np.array(aStat) for aStat in errStats]
    biasTmin,mabTmin,maeTmin,biasTmax,mabTmax,maeTmax = errStats
    
    statTminTwx,statTminDaymet,statTminPrism = [np.array(aStat) for aStat in biasTmin]
    statTmaxTwx,statTmaxDaymet,statTmaxPrism = [np.array(aStat) for aStat in biasTmax]
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    
    lt = 'ko-'
    lp = 'ks-.'
    ld = 'k*--'
    
    statTminTwx,statTminDaymet,statTminPrism = [np.array(aStat) for aStat in maeTmin]
    statTmaxTwx,statTmaxDaymet,statTmaxPrism = [np.array(aStat) for aStat in maeTmax]
    
    ax1.plot(statTminTwx[0:12],lt)
    ax1.plot(statTminPrism[0:12],lp)
    ax1.plot(statTminDaymet[0:12],ld)
    
    ax2.plot(statTmaxTwx[0:12],lt)
    ax2.plot(statTmaxPrism[0:12],lp)
    ax2.plot(statTmaxDaymet[0:12],ld)
    
    statTminTwx,statTminDaymet,statTminPrism = [np.array(aStat) for aStat in mabTmin]
    statTmaxTwx,statTmaxDaymet,statTmaxPrism = [np.array(aStat) for aStat in mabTmax]
    
    ax3.plot(statTminTwx[0:12],lt)
    ax3.plot(statTminPrism[0:12],lp)
    ax3.plot(statTminDaymet[0:12],ld)
    
    ax4.plot(statTmaxTwx[0:12],lt)
    ax4.plot(statTmaxPrism[0:12],lp)
    ax4.plot(statTmaxDaymet[0:12],ld)
    
    plt.show()

def maskStnsToCCE(stnDa):
    
    dsMask = RasterDataset('/projects/daymet2/dem/interp_grids/cce/crp_cce_us_mask.tif')
    cceMask = dsMask.readAsArray().data
    
    stns = stnDa.stns
    cceStnMask = np.zeros(stns.size,dtype=np.bool)
    
    for stn,x in zip(stns,np.arange(stns.size)):
        
        try:
            row,col = dsMask.getRowCol(stn[LON], stn[LAT])
            #col,row = dsMask.getGridCellOffset(stn[LON], stn[LAT])
            
            cceStnMask[x] = cceMask[row,col]
        except OutsideExtent:
            pass

    return np.logical_and(cceStnMask,np.isnan(stnDa.stns[BAD]))#stns[cceStnMask]

def cceXvalStats():
    
    stnDaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stnsCceTminMask = maskStnsToCCE(stnDaTmin)
    stnDaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
    stnsCceTmaxMask = maskStnsToCCE(stnDaTmax)
    
    biasTminXval = np.mean(stnDaTmin.ds.variables['xvalfnl_bias_norm'][:,stnsCceTminMask],axis=1)
    biasTmaxXval = np.mean(stnDaTmax.ds.variables['xvalfnl_bias_norm'][:,stnsCceTmaxMask],axis=1)
    
    maeTminXval = np.mean(stnDaTmin.ds.variables['xvalfnl_mae_norm'][:,stnsCceTminMask],axis=1)
    maeTmaxXval = np.mean(stnDaTmax.ds.variables['xvalfnl_mae_norm'][:,stnsCceTmaxMask],axis=1)
    
    maeDlyTminXval = np.mean(stnDaTmin.ds.variables['xvalfnl_mae_dly'][:,stnsCceTminMask],axis=1)
    maeDlyTmaxXval = np.mean(stnDaTmax.ds.variables['xvalfnl_mae_dly'][:,stnsCceTmaxMask],axis=1)
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    bwidth = .25
    xlim = (-.25,12.90)
    
    print "TMIN"
    print maeTminXval
    print maeDlyTminXval
    print biasTminXval
    print "TMAX"
    print maeTmaxXval
    print maeDlyTmaxXval
    print biasTmaxXval
    
    print "TMIN"
    print np.mean(maeTminXval[0:12])
    print maeDlyTminXval
    print biasTminXval
    print "TMAX"
    print np.mean(maeTminXval[0:12])
    print maeTmaxXval
    print maeDlyTmaxXval
    print biasTmaxXval
    
    plt.sca(ax1)
    plt.bar(np.arange(maeTminXval.size),maeTminXval,width=bwidth,color="grey")
    plt.bar(np.arange(maeDlyTminXval.size)+bwidth,maeDlyTminXval,width=bwidth,color="w",hatch="//")
    plt.bar(np.arange(biasTminXval.size)+(bwidth*2),biasTminXval,width=bwidth,color="k")
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(maeTminXval.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    plt.ylim((-.1,1.7))
    plt.yticks(np.arange(-.1,1.7+.1,.1),fontsize=10)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("a.",loc="left")
    
    plt.sca(ax2)
    plt.bar(np.arange(maeTmaxXval.size),maeTmaxXval,width=bwidth,color="grey")
    plt.bar(np.arange(maeDlyTmaxXval.size)+bwidth,maeDlyTmaxXval,width=bwidth,color="w",hatch="//")
    plt.bar(np.arange(biasTmaxXval.size)+(bwidth*2),biasTmaxXval,width=bwidth,color="k")
    plt.ylim((-.1,1.7))
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(maeTminXval.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.legend(("Normal MAE","Daily MAE","Bias"),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("b.",loc="left")
    
    plt.tight_layout()
    
    f.set_size_inches(8,3.5)
    plt.savefig('/projects/daymet2/docs/final_writeup/cce_xvalstats.png',dpi=250)
    plt.show()


def plotCceNormSeExamples():
            
    cceDataPath = '/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/'
    
    def getMinMaxMth(varName,minMax):
        
        mthMeans = []
        
        for mth in np.arange(1,13):
            
            ds = RasterDataset("".join([cceDataPath,"cce_",varName,"_%02d.tif"%mth]))
            a = ds.readAsArray()
            mthMeans.append(np.ma.mean(a))
        
        mthMeans = np.array(mthMeans)
        if minMax == 'min':
            mth = np.argmin(mthMeans) + 1
        else:
            mth = np.argmax(mthMeans) + 1
        
        return mth
    
    mthTmin = getMinMaxMth("tmin_se",'max')
    mthTmax = getMinMaxMth("tmax_se",'min')
        
    dsTminSe = RasterDataset("".join([cceDataPath,"cce_tmin_se_%02d.tif"%mthTmin]))
    dsTmaxSe = RasterDataset("".join([cceDataPath,"cce_tmax_se_%02d.tif"%mthTmax]))
    dsTminNorm = RasterDataset("".join([cceDataPath,"cce_tmin_normal_%02d.tif"%mthTmin]))
    dsTmaxNorm = RasterDataset("".join([cceDataPath,"cce_tmax_normal_%02d.tif"%mthTmax]))
        
    tminSe = dsTminSe.readAsArray()
    tmaxSe = dsTmaxSe.readAsArray()
    tminNorm = dsTminNorm.readAsArray()
    tmaxNorm = dsTmaxNorm.readAsArray()
    
    vminSe = np.min([np.min(tminSe),np.min(tmaxSe)])
    vmaxSe = np.max([np.max(tminSe),np.max(tmaxSe)])
    #levels = np.arange(np.floor(vminSe*10)/10, (np.ceil(vmaxSe*10)/10)+.05,0.1)
    levelsSe = np.arange(np.floor(vminSe*10)/10, (np.ceil(vmaxSe*10)/10)+.05,0.05)
    #levels = np.arange(np.floor(vminSe*100)/100, (np.ceil(vmaxSe*100)/100)+.01,0.01)
    
    vminNorm = np.min([np.min(tminNorm),np.min(tmaxNorm)])
    vmaxNorm = np.max([np.max(tminNorm),np.max(tmaxNorm)])
    #levels = np.arange(np.floor(vminSe*10)/10, (np.ceil(vmaxSe*10)/10)+.05,0.1)
    levelsNorm = np.arange(np.floor(vminNorm*1)/1, (np.ceil(vmaxNorm*1)/1)+1,1)
    
#    normCmapSe = matplotlib.colors.Normalize(vmin=np.min([np.min(tminSe),np.min(tmaxSe)]),
#                                             vmax=np.max([np.max(tminSe),np.max(tmaxSe)]))
#    normCmapNormal = matplotlib.colors.Normalize(vmin=np.min([np.min(tminNorm),np.min(tmaxNorm)]),
#                                             vmax=np.max([np.max(tminNorm),np.max(tmaxNorm)]))
#     
#    cmap = CMAP_ESRI_PRCP
#    smSe = cm.ScalarMappable(normCmapSe, cmap)
#    smSe.set_array(np.concatenate((np.ma.compressed(tminSe),np.ma.compressed(tmaxSe))))
#    
#    smNormal = cm.ScalarMappable(normCmapNormal, cmap)
#    smNormal.set_array(np.concatenate((np.ma.compressed(tminNorm),np.ma.compressed(tmaxNorm))))
        
    dsGrid = dsTminSe
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
    
    
    cf = plt.gcf()
        
    axes_pad = .2
    grid = ImageGrid(cf,211,nrows_ncols=(1,2),axes_pad=axes_pad,cbar_mode="single",cbar_location="right",cbar_pad=0.02)
    
    m.ax = grid[0]
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)#
    contf = m.contourf(x, y, tminSe,cmap=CMAP_ESRI_PRCP,levels=levelsSe)
    m.contour(x, y, tminSe,levels=levelsSe,colors='k',linewidths=.2)
    cbar = grid[0].cax.colorbar(contf)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylabel("$^\circ$C",fontsize=9)
    plt.sca(grid[0])
    plt.title("a.",loc="left",fontsize=9)
    
    m.ax = grid[1]
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)#
    contf = m.contourf(x, y, tmaxSe,cmap=CMAP_ESRI_PRCP,levels=levelsSe)
    m.contour(x, y, tmaxSe,levels=levelsSe,colors='k',linewidths=.2)
    plt.sca(grid[1])
    plt.title("b.",loc="left",fontsize=9)
    
    grid = ImageGrid(cf,212,nrows_ncols=(1,2),axes_pad=axes_pad,cbar_mode="single",cbar_location="right",cbar_pad=0.02)
    m.ax = grid[0]
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)
    contf = m.contourf(x, y, tminNorm,cmap=CMAP_ESRI_PRCP,levels=levelsNorm)
    #m.contour(x, y, tminNorm,levels=levelsNorm,colors='k',linewidths=.2)
    cbar = grid[0].cax.colorbar(contf)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylabel("$^\circ$C",fontsize=9)
    plt.sca(grid[0])
    plt.title("c.",loc="left",fontsize=9)
    
    m.ax = grid[1]
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)#
    contf = m.contourf(x, y, tmaxNorm,cmap=CMAP_ESRI_PRCP,levels=levelsNorm)
    plt.sca(grid[1])
    plt.title("d.",loc="left",fontsize=9)
    #m.contour(x, y, tmaxNorm,colors='gray',levels=levelsNorm)#,colors='k',linewidths=.2)
    plt.tight_layout(h_pad=0.1)
    plt.savefig('/projects/daymet2/docs/final_writeup/cce_interp_ex_maps.png',dpi=150)
    plt.show()      

def plotCceNormSeExamples2():
            
    cceDataPath = '/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/'
        
    mthTmin = 8
    mthTmax = 8
    
    dsTminSe = RasterDataset("".join([cceDataPath,"cce_tmin_se_%02d.tif"%mthTmin]))
    dsTmaxSe = RasterDataset("".join([cceDataPath,"cce_tmax_se_%02d.tif"%mthTmax]))
    dsTminNorm = RasterDataset("".join([cceDataPath,"cce_tmin_normal_%02d.tif"%mthTmin]))
    dsTmaxNorm = RasterDataset("".join([cceDataPath,"cce_tmax_normal_%02d.tif"%mthTmax]))
        
    tminSe = dsTminSe.readAsArray()
    tmaxSe = dsTmaxSe.readAsArray()
    tminNorm = dsTminNorm.readAsArray()
    tmaxNorm = dsTmaxNorm.readAsArray()
    
    print np.max(tmaxSe),np.min(tminSe)
    
    dsGrid = dsTminSe
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)
            
    print "Mapping data...."
    m = Basemap(resolution='h',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=-111,lat_0=0)
    x, y = m(*np.meshgrid(lon,lat))
    
    
    hillshd = getCceHillshade(m, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon)
    
    cfig = plt.gcf()
    
    def plotData(adata,cmap,cbar_label,grid,gridCellNum,m,title,levels=None,ylabel=None):
        m.ax = grid[gridCellNum]
        plt.sca(grid[gridCellNum])
        m.imshow(hillshd,cmap=cm.gray)
        m.drawcountries()
        m.drawstates()
        m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)#,linewidth=1)
        
        if levels is None:
            levels = np.linspace(np.min(adata), np.max(adata), 150)
        
        cf = m.contourf(x,y,adata,cmap=cmap,levels=levels,alpha=.5,antialiased=True)
        cf = m.contourf(x,y,adata,cmap=cmap,levels=levels,alpha=.5,antialiased=True)
#        if levels:
#            cf = m.contourf(x,y,adata,150,cmap=cmap,alpha=.5,antialiased=True)
#            cf = m.contourf(x,y,adata,150,cmap=cmap,alpha=.5,antialiased=True)
#        else:
#            cf = m.contourf(x,y,adata,150,cmap=cmap,alpha=.5,antialiased=True)
#            cf = m.contourf(x,y,adata,150,cmap=cmap,alpha=.5,antialiased=True)
            
        cbar = grid[gridCellNum].cax.colorbar(cf)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.set_ylabel(cbar_label,fontsize=9)
        
        #cbar = plt.colorbar(cf, cax = grid.cbar_axes[gridCellNum])
        cbar.set_alpha(1)
        #cbar.draw_all()
        
        #plt.title(title,loc="left",fontsize=12)
        if title is not None:
            plt.title(title,fontsize=12)
        if ylabel is not None:
            plt.ylabel(ylabel,fontsize=12)
    
    axes_pad = .7
    grid = ImageGrid(cfig,211,nrows_ncols=(1,2),axes_pad=axes_pad,cbar_mode="each",cbar_location="right",cbar_pad=0.02)
    
    vminNorm = np.min([np.min(tminNorm),np.min(tmaxNorm)])
    vmaxNorm = np.max([np.max(tminNorm),np.max(tmaxNorm)])
    #levels = np.arange(np.floor(vminSe*10)/10, (np.ceil(vmaxSe*10)/10)+.05,0.1)
    #levelsNorm = np.arange(np.floor(vminNorm), np.ceil(vmaxNorm)+1)
    #levelsNorm = np.linspace(vminNorm, vmaxNorm, 50)
    levelsNorm = None#np.arange(np.floor(np.min(tminNorm)), np.ceil(np.max(tminNorm))+1)
    plotData(tminNorm, CMAP_ESRI_PRCP, "$^\circ$C", grid,0, m,'Tmin',levelsNorm,'Interpolated Aug. Normal')
    plt.sca(grid[0])
    tsize = 6
    m.plot(TRANSECT_CCE_LON,TRANSECT_CCE_LAT,'k--',latlon=True,lw=.5)
    plt.text(107088,140951-5000,"1",fontsize=tsize)
    plt.text(119947,140951-5000,"2",fontsize=tsize)
    plt.text(129132,140951-5000,"3",fontsize=tsize)
    plt.text(140154,140951-5000,"4",fontsize=tsize)
    plt.text(195877,140338-5000,"5",fontsize=tsize)
    plt.text(6051,11135,"1=Flathead Lake\n2=Mission Mountains\n3=Seeley-Swan Valley\n4=Swan Crest\n5=Gates Park",fontsize=6,bbox=dict(facecolor='white',alpha=.7))
    m.scatter([-114.1,-113.953,-113.833,-113.638,-112.929],[47.78]*5,latlon=True)
    
    levelsNorm = None#np.arange(np.floor(np.min(tmaxNorm)), np.ceil(np.max(tmaxNorm))+1)
    plotData(tmaxNorm, CMAP_ESRI_PRCP, "$^\circ$C", grid,1, m,'Tmax',levelsNorm)
    plt.sca(grid[1])
    tsize = 6
    m.plot(TRANSECT_CCE_LON,TRANSECT_CCE_LAT,'k--',latlon=True,lw=.5)
    plt.text(107088,140951-5000,"1",fontsize=tsize)
    plt.text(119947,140951-5000,"2",fontsize=tsize)
    plt.text(129132,140951-5000,"3",fontsize=tsize)
    plt.text(140154,140951-5000,"4",fontsize=tsize)
    plt.text(195877,140338-5000,"5",fontsize=tsize)    
    
    vminSe = np.min([np.min(tminSe),np.min(tmaxSe)])
    vmaxSe = np.max([np.max(tminSe),np.max(tmaxSe)])
    #levels = np.arange(np.floor(vminSe*10)/10, (np.ceil(vmaxSe*10)/10)+.05,0.1)
    #levelsSe = np.arange(np.floor(vminSe*10)/10, (np.ceil(vmaxSe*10)/10)+.01,0.01)
    levelsSe = None# np.linspace(vminSe, vmaxSe, 150)
    # levelsSe=None
    grid = ImageGrid(cfig,212,nrows_ncols=(1,2),axes_pad=axes_pad,cbar_mode="each",cbar_location="right",cbar_pad=0.02)
    plotData(tminSe, CMAP_ESRI_PRCP, "Std. Err. ($^\circ$C)", grid,0, m,None,levelsSe,'Uncertainty Aug. Normal')
    plotData(tmaxSe, CMAP_ESRI_PRCP, "Std. Err. ($^\circ$C)", grid,1, m,None,levelsSe)     
    
    plt.tight_layout()
    cfig.set_size_inches(7,5.5)
    plt.savefig('/projects/daymet2/docs/final_writeup/cce_aug_norms_se.png',dpi=250)
    plt.show()

def trendsBoxplot():
    dsTwxTmin = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmin19812010trend.tif')
    dsDaymetTmin = RasterDataset('/projects/daymet2/cce_case_study/daymet_files/trends/cce_daymet_tmin19812010trend.tif')
    dsPrismTmin = RasterDataset('/projects/daymet2/cce_case_study/prism_files/trends/cce_prism4km_tmin_trend1981-2010.tif')
    dsTwxTmax = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmax19812010trend.tif')
    dsDaymetTmax = RasterDataset('/projects/daymet2/cce_case_study/daymet_files/trends/cce_daymet_tmax19812010trend.tif')
    dsPrismTmax = RasterDataset('/projects/daymet2/cce_case_study/prism_files/trends/cce_prism4km_tmax_trend1981-2010.tif')
    
    twxTmin = dsTwxTmin.readAsArray()*30
    prismTmin = dsPrismTmin.readAsArray()*30
    daymetTmin = dsDaymetTmin.readAsArray()*30
    
    twxTmax = dsTwxTmax.readAsArray()*30
    prismTmax = dsPrismTmax.readAsArray()*30
    daymetTmax = dsDaymetTmax.readAsArray()*30
    
    ds = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_elev.tif')
    elev = ds.readAsArray()
    elev = np.ma.filled(elev, -1)
        
    twxTmin = twxTmin.data
    prismTmin = prismTmin.data
    daymetTmin = daymetTmin.data
    twxTmax = twxTmax.data
    prismTmax = prismTmax.data
    daymetTmax = daymetTmax.data
    
#    elevLevels = np.arange(750,2750,250)
#    elevLevels[-1] = 2751
#    elevMasks = [np.logical_and(elev >= elevLevels[x],elev < elevLevels[x+1]) for x in np.arange(elevLevels.size-1)]
#    elevLevels[-1] = 2750
#    
#    elevMasks[0] = elev < elevLevels[1]
#    elevMasks[-1] = elev >= 2500# elevLevels[-1]
    
    elevMasks = []
    elevMasks.append(np.logical_and(elev >= 734,elev < 1000)) #734
    elevMasks.append(np.logical_and(elev >= 1000,elev < 1250)) #1000
    elevMasks.append(np.logical_and(elev >= 1250,elev < 1500)) #1250
    elevMasks.append(np.logical_and(elev >= 1500,elev < 1750)) #1500
    elevMasks.append(np.logical_and(elev >= 1750,elev < 2000)) #1750
    elevMasks.append(np.logical_and(elev >= 2000,elev < 2500)) #2000
    elevMasks.append(elev >= 2500) #2500
    
    elevLevels = np.array([734,1000,1250,1500,1750,2000,2500])
    
    dataTwxTmin = [twxTmin[elevMask] for elevMask in elevMasks]
    dataPrismTmin = [prismTmin[elevMask] for elevMask in elevMasks]
    dataDaymetTmin = [daymetTmin[elevMask] for elevMask in elevMasks]
    
    dataTwxTmax = [twxTmax[elevMask] for elevMask in elevMasks]
    dataPrismTmax = [prismTmax[elevMask] for elevMask in elevMasks]
    dataDaymetTmax = [daymetTmax[elevMask] for elevMask in elevMasks]
    
    #print np.mean(dataTwxTmin[0]),np.mean(dataPrismTmin[0]),np.mean(dataDaymetTmin[0])
    print np.mean(dataTwxTmin[0]),np.mean(dataTwxTmin[-1])
    
    def drawBoxplot(data,ax):
        
        means = [np.mean(x) for x in data]
        
        bp = ax.boxplot(data,vert=False,sym="",patch_artist=True)#sym=""
        plt.setp(bp['boxes'], color='grey')
        plt.setp(bp['whiskers'], color='black',linestyle="-")
        plt.setp(bp['fliers'], color='black')
        plt.setp(bp['medians'], color='black')
        
        #ax = plt.gca()
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
        s = ax.scatter(means,np.arange(1,len(data)+1),color='black',s=4)
        s.set_zorder(20)
    
    f, ((ax1, ax2, ax3), (ax4, ax5,ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    #fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(2, 3, sharex=True, sharey=True)

    xlim = (-4,8)
    tairLobLoc = (-3.9,8)
    fsizeTairLab = 10
    
    ylabels = ["%d"%(aElev,) for aElev in elevLevels]
    ylabels[0] = "<%d"%(elevLevels[1],)
    ylabels[-1] = ">%d"%(elevLevels[-1],)
    
    #plt.subplot(131)
    drawBoxplot(dataTwxTmin,ax1)
    plt.sca(ax1)
    plt.xlim(xlim)
    plt.yticks(np.arange(1,9),ylabels,fontsize=10)
    plt.ylabel('meters',fontsize=10)
    plt.title("TopoWx",fontsize=12)
    #plt.text(tairLobLoc[0],tairLobLoc[1],'Tmin',weight='bold',fontsize=fsizeTairLab)
    
    #plt.subplot(132)
    drawBoxplot(dataPrismTmin,ax2)
    plt.sca(ax2)
    plt.xlim(xlim)
    plt.title("PRISM",fontsize=12)
    #plt.text(tairLobLoc[0],tairLobLoc[1],'Tmin',weight='bold',fontsize=fsizeTairLab)
    
    #plt.subplot(133)
    drawBoxplot(dataDaymetTmin,ax3)
    plt.sca(ax3)
    plt.xlim(xlim)
    plt.title("Daymet",fontsize=12)
    #plt.text(tairLobLoc[0],tairLobLoc[1],'Tmin',weight='bold',fontsize=fsizeTairLab)
    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel("Tmin")
    
    #plt.subplot(131)
    drawBoxplot(dataTwxTmax,ax4)
    plt.sca(ax4)
    plt.xlim(xlim)
    plt.xlabel(r'$^\circ$C / 30 yrs',fontsize=10)
    plt.yticks(np.arange(1,9),ylabels,fontsize=10)
    plt.ylabel('meters',fontsize=10)
    plt.tick_params(axis='x',labelsize=10)
    #plt.text(tairLobLoc[0],tairLobLoc[1],'Tmax',weight='bold',fontsize=fsizeTairLab)
    
    #plt.subplot(132)
    drawBoxplot(dataPrismTmax,ax5)
    plt.sca(ax5)
    plt.xlim(xlim)
    plt.xlabel(r'$^\circ$C / 30 yrs',fontsize=10)
    plt.tick_params(axis='x',labelsize=10)
    #plt.text(tairLobLoc[0],tairLobLoc[1],'Tmax',weight='bold',fontsize=fsizeTairLab)
    
    #plt.subplot(133)
    drawBoxplot(dataDaymetTmax,ax6)
    plt.sca(ax6)
    plt.xlim(xlim)
    plt.tick_params(axis='x',labelsize=10)
    plt.xlabel(r'$^\circ$C / 30 yrs',fontsize=10)
    ax6.yaxis.set_label_position("right")
    ax6.set_ylabel("Tmax")
    #plt.text(tairLobLoc[0],tairLobLoc[1],'Tmax',weight='bold',fontsize=fsizeTairLab)
    
    plt.tight_layout()
    
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
#    xmin,xmax = plt.xlim()
#    plt.hlines(3.5, xmin, xmax)
#    plt.xlim(xmin,xmax)
#    plt.text(-3.9,6.25,'Tmin',weight='bold')
#    plt.text(-3.9,3.25,'Tmax',weight='bold')
#    plt.xlabel("Adjustment ($^\circ$C)")
    f.set_size_inches(7,4)
    plt.savefig('/projects/daymet2/docs/final_writeup/trendsBoxplots.png',dpi=300)
    
    plt.show()


def plotTwxDaymetPrismAnom():
    dsElev = gdal.Open('/projects/daymet2/cce_case_study/predictors/cce_elev.tif')
    elev = dsElev.ReadAsArray()
    maskHighElev = elev>=2500#np.logical_and(elev>=2500,elev<=2750)
    maskLowElev = np.logical_and(elev>=1,elev<=1000)
    
    dsTwxTmin = gdal.Open('/projects/daymet2/cce_case_study/topowx_files/annual/cce_topowx_tmin19482012ann.tif')
    dsPrismTmin = gdal.Open('/projects/daymet2/cce_case_study/prism_files/annual/cce_prism4km_tmin_ann1948-2012.tif')
    dsDaymetTmin = gdal.Open('/projects/daymet2/cce_case_study/daymet_files/annual/cce_mosaic_daymet_tmin19802011ann.tif')
    
    dsTwxTmax = gdal.Open('/projects/daymet2/cce_case_study/topowx_files/annual/cce_topowx_tmax19482012ann.tif')
    dsPrismTmax = gdal.Open('/projects/daymet2/cce_case_study/prism_files/annual/cce_prism4km_tmax_ann1948-2012.tif')
    dsDaymetTmax = gdal.Open('/projects/daymet2/cce_case_study/daymet_files/annual/cce_mosaic_daymet_tmax19802011ann.tif')
    
    yrsTwx = np.arange(1948,2013)
    yrsDaymet = np.arange(1980,2012)
    normMaskTwx = np.logical_and(yrsTwx>=1981,yrsTwx<=2010)
    normMaskDaymet = np.logical_and(yrsDaymet>=1981,yrsDaymet<=2010)
    
    twxTmin = dsTwxTmin.ReadAsArray()
    prismTmin = dsPrismTmin.ReadAsArray()
    daymetTmin = dsDaymetTmin.ReadAsArray()
    
    twxTmax = dsTwxTmax.ReadAsArray()
    prismTmax = dsPrismTmax.ReadAsArray()
    daymetTmax = dsDaymetTmax.ReadAsArray()
    
    def getMeanAnom(tairAnn,mask,isDaymet=False):
        
        tairAnn = tairAnn[:,mask]
        
        if isDaymet:
            normMask = normMaskDaymet
        else:
            normMask = normMaskTwx
            
        annNorm = np.mean(tairAnn[normMask,:],axis=0)
        annAnom = tairAnn-annNorm
        mAnnAnom = np.mean(annAnom,axis=1)
        mAnnAnom = runningMean(mAnnAnom) 
        
        if isDaymet:
            a = np.ma.masked_equal(np.ones(yrsTwx.size)*-9999, -9999)   
            a[np.logical_and(yrsTwx>=1980,yrsTwx<=2011)] = mAnnAnom
            mAnnAnom = a
        
        return mAnnAnom
    
    twxTminHigh = getMeanAnom(twxTmin, maskHighElev)
    prismTminHigh = getMeanAnom(prismTmin, maskHighElev)
    daymetTminHigh = getMeanAnom(daymetTmin, maskHighElev,isDaymet=True)
    
    twxTminLow = getMeanAnom(twxTmin, maskLowElev)
    prismTminLow = getMeanAnom(prismTmin, maskLowElev)
    daymetTminLow = getMeanAnom(daymetTmin, maskLowElev,isDaymet=True)
    
    twxTmaxHigh = getMeanAnom(twxTmax, maskHighElev)
    prismTmaxHigh = getMeanAnom(prismTmax, maskHighElev)
    daymetTmaxHigh = getMeanAnom(daymetTmax, maskHighElev,isDaymet=True)
    
    twxTmaxLow = getMeanAnom(twxTmax, maskLowElev)
    prismTmaxLow = getMeanAnom(prismTmax, maskLowElev)
    daymetTmaxLow = getMeanAnom(daymetTmax, maskLowElev,isDaymet=True)
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    
    ylim = (-3,2.5)
        
    ax1.plot(yrsTwx,twxTminLow,color="#377EB8",lw=1.5)#,"k-")
    ax1.plot(yrsTwx,prismTminLow,color="#4DAF4A",lw=1.5)#,"k--",lw=2)
    ax1.plot(yrsTwx,daymetTminLow,color="#E41A1C",lw=1.5)#,"k-.",lw=2)
    plt.sca(ax1)
    plt.ylim(ylim)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax1.xaxis.grid(color='gray', linestyle='dashed')
    plt.ylabel('Anomaly ($^\circ$C)',fontsize=10)
    plt.tick_params(axis='y',labelsize=10)
    plt.title("Low Elevation",fontsize=12)
    plt.legend(("TopoWx","PRISM","Daymet"),fontsize=10,loc=2)
    
    ax2.plot(yrsTwx,twxTminHigh,color="#377EB8",lw=1.5)#,"k-")
    ax2.plot(yrsTwx,prismTminHigh,color="#4DAF4A",lw=1.5)#,"k--",lw=2)
    ax2.plot(yrsTwx,daymetTminHigh,color="#E41A1C",lw=1.5)#,"k-.",lw=2)
    plt.sca(ax2)
    plt.ylim(ylim)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray', linestyle='dashed')
    ax2.xaxis.grid(color='gray', linestyle='dashed')
    plt.title("High Elevation",fontsize=12)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Tmin")
    
    ax3.plot(yrsTwx,twxTmaxLow,color="#377EB8",lw=1.5)
    ax3.plot(yrsTwx,prismTmaxLow,color="#4DAF4A",lw=1.5)
    ax3.plot(yrsTwx,daymetTmaxLow,color="#E41A1C",lw=1.5)
    plt.sca(ax3)
    plt.ylim(ylim)
    ax3.set_axisbelow(True)
    ax3.yaxis.grid(color='gray', linestyle='dashed')
    ax3.xaxis.grid(color='gray', linestyle='dashed')
    plt.ylabel('Anomaly ($^\circ$C)',fontsize=10)
    plt.tick_params(axis='x',labelsize=10)
    plt.tick_params(axis='y',labelsize=10)
    
    ax4.plot(yrsTwx,twxTmaxHigh,color="#377EB8",lw=1.5)
    ax4.plot(yrsTwx,prismTmaxHigh,color="#4DAF4A",lw=1.5)
    ax4.plot(yrsTwx,daymetTmaxHigh,color="#E41A1C",lw=1.5)
    plt.sca(ax4)
    plt.ylim(ylim)
    ax4.set_axisbelow(True)
    ax4.yaxis.grid(color='gray', linestyle='dashed')
    ax4.xaxis.grid(color='gray', linestyle='dashed')
    plt.tick_params(axis='x',labelsize=10)
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel("Tmax")
    
    plt.tight_layout()
    plt.savefig('/projects/daymet2/docs/final_writeup/ann_anom_cce_compare.png',dpi=150)
    plt.show()
    

def plotCceTransectNorms():
    '''
    /projects/daymet2/cce_case_study/prism_files/normals_mthly/cce_tmax_normal_11.tif 
    /projects/daymet2/cce_case_study/topowx_files/normals/mosaics
    /projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics
    '''
    def getMeanNorm(dataPath,varName,mths):
        
        mthNorms = []
        
        for mth in mths:
            
            ds = RasterDataset("".join([dataPath,"cce_%s_normal_%02d.tif"%(varName,mth)]))
            a = ds.readAsArray()
            a.shape = (1,a.shape[0],a.shape[1])
            mthNorms.append(a)
            
        mthNorms = np.ma.vstack(mthNorms)
        return np.ma.mean(mthNorms,axis=0)
    
    jja = np.array([8])
    
    tairVar = 'tmin'
    tminTwx = getMeanNorm('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/',tairVar,jja)
    tminDaymet = getMeanNorm('/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/',tairVar,jja)
    tminPrism = getMeanNorm('/projects/daymet2/cce_case_study/prism_files/normals_mthly/',tairVar,jja)
    tairVar = 'tmax'
    tmaxTwx = getMeanNorm('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/',tairVar,jja)
    tmaxDaymet = getMeanNorm('/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/',tairVar,jja)
    tmaxPrism = getMeanNorm('/projects/daymet2/cce_case_study/prism_files/normals_mthly/',tairVar,jja)
    
    dsElev = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_elev.tif')
    dsTdi = RasterDataset('/projects/daymet2/cce_case_study/predictors/cce_tdi.tif')
    
    transLon = (-114.6,-112.5)
    transLat = (47.78,47.78)
    
    lats,lons = dsElev.getCoordGrid1d()
    
    elev = dsElev.readAsArray()
    tdi = dsTdi.readAsArray()
    
    row,col1 = dsElev.getRowCol(transLon[0], transLat[0])
    row,col2 = dsElev.getRowCol(transLon[1], transLat[1])
    
    lonsTrans =  np.around(lons[col1:col2],3)
    
    f, (ax1,ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
    
    lw=1.5
    line1, = ax1.plot(lonsTrans,runningMean(tminTwx[row,col1:col2],5),color="#377EB8",lw=lw)
    line2, = ax1.plot(lonsTrans,runningMean(tminPrism[row,col1:col2],5),color="#4DAF4A",lw=lw)
    line3, = ax1.plot(lonsTrans,runningMean(tminDaymet[row,col1:col2],5),color="#E41A1C",lw=lw)
    plt.sca(ax1)
    plt.ylabel('Tmin ($^\circ$C)',fontsize=10)
    plt.tick_params(axis='x',labelsize=10)
    plt.tick_params(axis='y',labelsize=10)
    plt.xlim((lonsTrans[0],lonsTrans[-1]))
    plt.xticks(np.arange(-114.5,-112.50,.25))
    
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(color='gray', linestyle='dashed')
    ymin,ymax = plt.ylim()
    plt.ylim(ymin,ymax+2)
    
    ax1twin = ax1.twinx()
    line4, = ax1twin.plot(lonsTrans,runningMean(elev[row,col1:col2],5),"k--",lw=lw)
    #ax2.plot(elev[row,col1:col2],color="k")
    ax1twin.invert_yaxis()
    plt.sca(ax1twin)
    plt.ylabel('Elevation (m; inverted)',fontsize=10)
    plt.tick_params(axis='y',labelsize=10)
    plt.legend([line1,line2,line3,line4],("TopoWx","PRISM","Daymet","Elevation"),fontsize=10,ncol=2)
    
    ymin,ymax = plt.ylim()
    plt.vlines([-114.1,-113.953,-113.833,-113.638,-112.929], ymin, ymax,lw=5,alpha=.3)
    yText = ymin
    shift = -.015
    plt.text(-114.1+shift, yText-20, "1")
    plt.text(-113.953+shift, yText-20, "2")
    plt.text(-113.833+shift, yText-20, "3")
    plt.text(-113.638+shift, yText-20, "4")
    plt.text(-112.929+shift, yText-20, "5")
    
    ax2.plot(lonsTrans,runningMean(tmaxTwx[row,col1:col2],5),color="#377EB8",lw=lw)
    ax2.plot(lonsTrans,runningMean(tmaxPrism[row,col1:col2],5),color="#4DAF4A",lw=lw)
    ax2.plot(lonsTrans,runningMean(tmaxDaymet[row,col1:col2],5),color="#E41A1C",lw=lw)
    plt.sca(ax2)
    plt.ylabel('Tmax ($^\circ$C)',fontsize=10)
    plt.tick_params(axis='x',labelsize=10)
    plt.tick_params(axis='y',labelsize=10)
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(color='gray', linestyle='dashed')
    ax2.set_xlabel('Longitude',fontsize=10)
    
    ax2twin = ax2.twinx()
    ax2twin.plot(lonsTrans,runningMean(elev[row,col1:col2],5),"k--",lw=lw)
    ax2twin.invert_yaxis()
    plt.sca(ax2twin)
    plt.ylabel('Elevation (m; inverted)',fontsize=10)
    plt.tick_params(axis='y',labelsize=10)
    plt.xlim((lonsTrans[0],lonsTrans[-1]))
    
    ymin,ymax = plt.ylim()
    plt.vlines([-114.1,-113.953,-113.833,-113.638,-112.929], ymin, ymax,lw=5,alpha=.3)
    yText = ymin
    shift = -.015
    plt.text(-114.1+shift, yText-20, "1")
    plt.text(-113.953+shift, yText-20, "2")
    plt.text(-113.833+shift, yText-20, "3")
    plt.text(-113.638+shift, yText-20, "4")
    plt.text(-112.929+shift, yText-20, "5")
    plt.ylim((ymin,ymax))
    
    #plt.xlabel('Longitude',fontsize=10)
    plt.savefig('/projects/daymet2/docs/final_writeup/cce_transect.png',dpi=150)
    plt.show()

def outputCceStnsToCsv():
    dsMask = RasterDataset('/projects/daymet2/dem/interp_grids/cce/crp_cce_us_mask.tif')
    cceMask = dsMask.readAsArray().data
        
#    dsMask = input_raster('/projects/daymet2/dem/interp_grids/cce/crp_cce_us_mask.tif')
#    cceMask = dsMask.readEntireRaster()
        
    def maskStnsToCCE(stnDa):
    
        stns = stnDa.stns[np.isnan(stnDa.stns[BAD])]
        
        cceStnMask = np.zeros(stns.size,dtype=np.bool)
        
        for stn,x in zip(stns,np.arange(stns.size)):
            
            try:
                row,col = dsMask.getRowCol(stn[LON], stn[LAT])
                #col,row = dsMask.getGridCellOffset(stn[LON], stn[LAT])
                
                cceStnMask[x] = cceMask[row,col]
            except OutsideExtent:
                pass

        return stns[cceStnMask]
    
    stnDaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stnsCceTmin = maskStnsToCCE(stnDaTmin)
    
    stnDaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
    stnsCceTmax = maskStnsToCCE(stnDaTmax)
    
    def outputCsv(pathOut,stns):
    
        fout = open(pathOut,'w')
        
        mths = np.arange(1,13)
        
        header = ['STNID','NAME','LON','LAT','ELEV']
        header.extend([get_lst_varname(mth) for mth in mths])
        header.extend([get_norm_varname(mth) for mth in mths])
        header[-1] = header[-1]+"\n"
        
        fout.write(",".join(header))

        for stn in stns:
            
            stnLine = [stn[STN_ID],stn[STN_NAME],"%.04f"%stn[LON],"%.04f"%stn[LAT],"%.02f"%stn[ELEV]]
            stnLine.extend(["%.04f"%stn[get_lst_varname(mth)] for mth in mths])
            stnLine.extend(["%.04f"%stn[get_norm_varname(mth)] for mth in mths])
            stnLine[-1] = stnLine[-1]+"\n"
            fout.write(",".join(stnLine))
    
    outputCsv("/projects/daymet2/docs/final_writeup/cce_stns_tmin.csv", stnsCceTmin)
    outputCsv("/projects/daymet2/docs/final_writeup/cce_stns_tmax.csv", stnsCceTmax)

def plotRelImp():
    dtype = [(LON,np.float), (LAT,np.float), (ELEV,np.float), (LST,np.float),('r2',np.float)]
    relImpTmax = np.loadtxt('/projects/daymet2/docs/final_writeup/cce_relimpr2_tmax.csv', dtype=dtype, delimiter=",", skiprows=1)
    relImpTmin = np.loadtxt('/projects/daymet2/docs/final_writeup/cce_relimpr2_tmin.csv', dtype=dtype, delimiter=",", skiprows=1)

    x = np.arange(12)
    bwidth = 0.6
    orient = 'vertical'
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    bar1 = ax1.bar(x, height=relImpTmax[ELEV],width=bwidth,color='#377EB8',orientation=orient)
    bar2 = ax1.bar(x, height=relImpTmax[LST],width=bwidth,bottom=relImpTmax[ELEV],color='#E41A1C',orientation=orient)
    bar3 = ax1.bar(x, height=relImpTmax[LAT],width=bwidth,bottom=relImpTmax[ELEV]+relImpTmax[LST],color='#4DAF4A',orientation=orient)
    bar4 = ax1.bar(x, height=relImpTmax[LON],width=bwidth,bottom=relImpTmax[ELEV]+relImpTmax[LST]+relImpTmax[LAT],color='#984EA3',orientation=orient)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax1.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.sca(ax1)
    plt.xticks(x + bwidth/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),fontsize=8)
    plt.ylim((0,1))
    plt.xlim((-0.5,12))
    plt.tick_params(axis='y',labelsize=9)
    plt.ylabel(r"$R^2$")
    plt.title('a.',loc="left",fontsize=12)
    
    ax2.bar(x, height=relImpTmin[ELEV],width=bwidth,color='#377EB8',orientation=orient)
    ax2.bar(x, height=relImpTmin[LST],width=bwidth,bottom=relImpTmin[ELEV],color='#E41A1C',orientation=orient)
    ax2.bar(x, height=relImpTmin[LAT],width=bwidth,bottom=relImpTmin[ELEV]+relImpTmin[LST],color='#4DAF4A',orientation=orient)
    ax2.bar(x, height=relImpTmin[LON],width=bwidth,bottom=relImpTmin[ELEV]+relImpTmin[LST]+relImpTmin[LAT],color='#984EA3',orientation=orient)    
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray', linestyle='dashed')
    plt.sca(ax2)
    plt.xticks(x + bwidth/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),fontsize=8)
    plt.xlim((-0.5,12))
    plt.tick_params(axis='y',labelsize=9)
    plt.legend([bar1,bar2,bar3,bar4],['Elev.','LST','Lat',"Lon"],fontsize=9,ncol=2)
    plt.title('b.',loc="left",fontsize=12)
    plt.tight_layout()
    
    f.set_size_inches(6.75,3.5)
    
    plt.savefig("/projects/daymet2/docs/final_writeup/cce_relimp.png",dpi=250)
    plt.show()

def flatheadLake():
    dbTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    dbTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
 
    xTmin = np.nonzero(dbTmin.stn_ids=='GHCN_USC00240755')[0][0]
    xTmax = np.nonzero(dbTmax.stn_ids=='GHCN_USC00240755')[0][0]
    
    stn = dbTmin.stns[dbTmin.stn_ids=='GHCN_USC00240755'][0]
    
#    xTmin = np.nonzero(dbTmin.stn_ids=='RAWS_MCON')[0][0]
#    xTmax = np.nonzero(dbTmax.stn_ids=='RAWS_MCON')[0][0]
    
    stnTmin = np.array([dbTmin.stns[xTmin][get_norm_varname(mth)] for mth in np.arange(1,13)])
    stnTmax = np.array([dbTmax.stns[xTmax][get_norm_varname(mth)] for mth in np.arange(1,13)])
    
    lon,lat = stn[LON],stn[LAT]
    #lon,lat = -114.119253,47.912605 #mid lake
    #lon,lat = -114.02853698,47.87532492
    
    #lon,lat = -113.717362,47.535698 #condon
    
    def getNorms(prefix):
    
        tmin = []
        tmax = []
        
        for mth in np.arange(1,13):
        
            ds = RasterDataset(prefix+'cce_%s_normal_%02d.tif'%('tmin',mth))
            tmin.append(ds.getDataValue(lon, lat))
            
            ds = RasterDataset(prefix+'cce_%s_normal_%02d.tif'%('tmax',mth))
            tmax.append(ds.getDataValue(lon, lat))
        
        tmin = np.array(tmin)
        tmax = np.array(tmax)
        return tmin,tmax
    
    def getSe(prefix):
    
        tmin = []
        tmax = []
        
        for mth in np.arange(1,13):
        
            ds = RasterDataset(prefix+'cce_%s_se_%02d.tif'%('tmin',mth))
            tmin.append(ds.getDataValue(lon, lat))
            
            ds = RasterDataset(prefix+'cce_%s_se_%02d.tif'%('tmax',mth))
            tmax.append(ds.getDataValue(lon, lat))
        
        tmin = np.array(tmin)
        tmax = np.array(tmax)
        return tmin,tmax
    
    twxTmin,twxTmax = getNorms('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/')
    #seTmin,seTmax = getSe('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/')
    prismTmin,prismTmax = getNorms('/projects/daymet2/cce_case_study/prism_files/normals_mthly/')
    daymetTmin,daymetTmax = getNorms('/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/')
#    wTwx = getMeanNorm('/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/',tairVar,djf)
#    wDaymet = getMeanNorm('/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/',tairVar,djf)
#    wPrism = getMeanNorm('/projects/daymet2/cce_case_study/prism_files/normals_mthly/',tairVar,djf)
    
    plt.plot(twxTmax-stnTmax)
    plt.plot(prismTmax-stnTmax)
    plt.plot(daymetTmax-stnTmax)
    plt.legend(['Twx','Prism','Daymet'])
    
#    plt.plot(stnTmax)
#    plt.plot(twxTmax)
#    plt.plot(prismTmax)
#    plt.plot(daymetTmax)
#    plt.legend(['Station','Twx','Prism','Daymet'])
    
#    plt.plot(twxTmax-stnTmax)
#    plt.plot(prismTmax-stnTmax)
#    plt.plot(daymetTmax-stnTmax)
    #plt.plot(tmax)
    plt.show()

def cceSpecificStnCompareBarPlot():
    
    def f_to_c(f):
        return (f-32)/1.8
    
    normsNcdc = np.loadtxt('/projects/daymet2/cce_case_study/normals_ncdc.csv',
                           dtype=[('stn_id',"<S50"),('name',"<S50"),('elev',np.float),
                                  ('lat',np.float),('lon',np.float),('tmax',np.float),('tmin',np.float)], 
                           delimiter=",", skiprows=1, usecols=[0,1,2,3,4,6,8])
    
    normsNcdc['tmin'] = f_to_c(normsNcdc['tmin']/10.0)
    normsNcdc['tmax'] = f_to_c(normsNcdc['tmax']/10.0)
    normsNcdc['stn_id'] = ['GHCN_'+aId.split("D:")[1] for aId in normsNcdc['stn_id']]
    
    def getNorms(prefix,lon,lat):
    
        tmin = []
        tmax = []
        
        for mth in np.arange(1,13):
        
            ds = RasterDataset(prefix+'cce_%s_normal_%02d.tif'%('tmin',mth))
            tmin.append(ds.getDataValue(lon, lat))
            
            ds = RasterDataset(prefix+'cce_%s_normal_%02d.tif'%('tmax',mth))
            tmax.append(ds.getDataValue(lon, lat))
        
        tmin = np.array(tmin)
        tmax = np.array(tmax)
        return tmin,tmax
    
    errStats = pickle.load(open('/projects/daymet2/cce_case_study/stn_compare.pickle'))
    #errStats = pickle.load(open('/projects/daymet2/cce_case_study/stn_compare_shiftprism.pickle'))
    errStatsShift = pickle.load(open('/projects/daymet2/cce_case_study/stn_compare_shiftall.pickle'))
    
    #errStats = [np.array(aStat) for aStat in errStats]
    biasTmin,mabTmin,maeTmin,biasTmax,mabTmax,maeTmax = errStats
    biasShiftTmin,mabShiftTmin,maeShiftTmin,biasShiftTmax,mabShiftTmax,maeShiftTmax = errStatsShift
    
    statTminTwx,statTminDaymet,statTminPrism = [np.array(aStat) for aStat in maeTmin]
    statTmaxTwx,statTmaxDaymet,statTmaxPrism = [np.array(aStat) for aStat in maeTmax]
    
    statShiftTminTwx,statShiftTminDaymet,statShiftTminPrism = [np.array(aStat) for aStat in maeShiftTmin]
    statShiftTmaxTwx,statShiftTmaxDaymet,statShiftTmaxPrism = [np.array(aStat) for aStat in maeShiftTmax]
    
    statTminPrism = statShiftTminPrism
    statTmaxPrism = statShiftTmaxPrism
#    statTminDaymet = statShiftTminDaymet
#    statTmaxDaymet = statShiftTmaxDaymet

    print statTminTwx[-1]
    print statTminPrism[-1]
    print statTminDaymet[-1]
    
    print statTmaxTwx[-1]
    print statTmaxPrism[-1]
    print statTmaxDaymet[-1]
    
    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, sharey='row',sharex=False)
    
    bwidth = .25
    xlim = (-.25,12.90)
        
    plt.sca(ax1)
    plt.bar(np.arange(statTminTwx.size),statTminTwx,width=bwidth,color="k")
    plt.bar(np.arange(statTminPrism.size)+bwidth,statTminPrism,width=bwidth,color="grey")
    plt.bar(np.arange(statTminDaymet.size)+(bwidth*2),statTminDaymet,width=bwidth,color="w",hatch="//")
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(statTminTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    plt.ylim((0,3.0))
    plt.yticks(np.arange(0,3.25,.25),fontsize=10)
    plt.ylabel("$^\circ$C",fontsize=10)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("a.",loc="left")
    
    plt.sca(ax2)
    plt.bar(np.arange(statTmaxTwx.size),statTmaxTwx,width=bwidth,color="k")
    plt.bar(np.arange(statTmaxPrism.size)+bwidth,statTmaxPrism,width=bwidth,color="grey")
    plt.bar(np.arange(statTmaxDaymet.size)+(bwidth*2),statTmaxDaymet,width=bwidth,color="w",hatch="//")
    plt.ylim((0,3.0))
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(statTmaxTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("b.",loc="left")
    
    ###################################
    statTminTwx,statTminDaymet,statTminPrism = [np.array(aStat) for aStat in mabTmin]
    statTmaxTwx,statTmaxDaymet,statTmaxPrism = [np.array(aStat) for aStat in mabTmax]
    
    statShiftTminTwx,statShiftTminDaymet,statShiftTminPrism = [np.array(aStat) for aStat in mabShiftTmin]
    statShiftTmaxTwx,statShiftTmaxDaymet,statShiftTmaxPrism = [np.array(aStat) for aStat in mabShiftTmax]
    
    statTminPrism = statShiftTminPrism
    statTmaxPrism = statShiftTmaxPrism
#    statTminDaymet = statShiftTminDaymet
#    statTmaxDaymet = statShiftTmaxDaymet

#    print statTminTwx
#    print statTminPrism
#    print statTminDaymet
    
    plt.sca(ax3)
    plt.bar(np.arange(statTminTwx.size),statTminTwx,width=bwidth,color="k")
    plt.bar(np.arange(statTminPrism.size)+bwidth,statTminPrism,width=bwidth,color="grey")
    plt.bar(np.arange(statTminDaymet.size)+(bwidth*2),statTminDaymet,width=bwidth,color="w",hatch="//")
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(statTminTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    plt.ylim((0,1.50))
    plt.yticks(np.arange(0,1.6,.1),fontsize=10)
    plt.ylabel("$^\circ$C",fontsize=10)
    ax3.set_axisbelow(True)
    ax3.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("c.",loc="left")
    
    plt.sca(ax4)
    plt.bar(np.arange(statTmaxTwx.size),statTmaxTwx,width=bwidth,color="k")
    plt.bar(np.arange(statTmaxPrism.size)+bwidth,statTmaxPrism,width=bwidth,color="grey")
    plt.bar(np.arange(statTmaxDaymet.size)+(bwidth*2),statTmaxDaymet,width=bwidth,color="w",hatch="//")
    plt.legend(("TopoWx","PRISM","Daymet"),fontsize=10)
    plt.ylim((0,1.50))
    xlim = plt.xlim(xlim)
    plt.xticks(np.arange(statTmaxTwx.size) + (bwidth*3)/2.0, ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann'),fontsize=10)
    plt.hlines(0, xlim[0], xlim[1])
    plt.xlim(xlim)
    ax4.set_axisbelow(True)
    ax4.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("d.",loc="left")
    
    plt.tight_layout()
    
    f.set_size_inches(8,4.5)
    #plt.savefig('/projects/daymet2/docs/final_writeup/cce_stn_compare.png',dpi=250)
    plt.show()

def plotOptimNnghsKriging():
    
    dtypeStns = copy(DTYPE_INTERP)
    dtypeStns.extend(DTYPE_OPTIM)
    db = station_data_infill('/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc', 'tmax',stn_dtype=dtypeStns)
    stns = db.stns[np.isnan(db.stns[BAD])]
    m = Basemap(resolution='i',projection='aea', llcrnrlat=22,urcrnrlat=49,llcrnrlon=-119,urcrnrlon=-64,
                lat_1=29.5,lat_2=45.5,lon_0=-96.0,lat_0=37.5,area_thresh= 10000)
    
    dsGrid = RasterDataset('/projects/daymet2/dem/interp_grids/ConusQtrDeg/maskQtrDeg.tif')
    lat,lon = dsGrid.getCoordGrid1d()
    x, y = m(*np.meshgrid(lon,lat))
    gridMask = dsGrid.gdalDs.ReadAsArray() != 19
    latS = np.sort(lat) 
    
    
#    clrsRed = brewer2mpl.get_map('Reds', 'Sequential', 6, reverse=False).mpl_colors
#    clrsBlue = brewer2mpl.get_map('Blues', 'Sequential', 6, reverse=True).mpl_colors
#    
#    clrsBlue.append("grey")
#    clrsBlue.extend(clrsRed)
#    clrs = clrsBlue
    
    #clrs = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True)
    #clrs = clrs.mpl_colors
    #clrs[5] = "grey"
    #levels = [-0.50,-0.40,-0.30,-0.20,-0.10,-0.05,0.05,0.10,0.20,0.30,0.40,0.50]
               
    cf = plt.gcf()
    grid = ImageGrid(cf,111,nrows_ncols=(4,3),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05,cbar_size="2%")#,cbar_pad=0.02)#axes_pad=.625
    
    for mth in np.arange(1,13):
        
        print mth
        
        stnsMth = stns[np.isfinite(stns[get_optim_varname(mth)])] 
    
        aGrid = griddata(stnsMth[LON],stnsMth[LAT],stnsMth[get_optim_varname(mth)],lon,latS)
        aGrid = np.flipud(aGrid)
        aGrid.mask = np.logical_or(gridMask,aGrid.mask)
    
        m.ax = grid[mth-1]
        m.readshapefile('/projects/daymet2/dem/st_bounds/statesp020','statesp020')
        cf = m.contourf(x, y, aGrid)

        if mth == 1:
            cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
        
        #grid[0].set_ylabel("USHCN")
        #grid[0].set_title(str(mth))
        #cbar = grid.cbar_axes.colorbar(cf)
        #cbar.set_ticks(levels)
        #cbar.set_label(r'$^\circ$C decade$^{-1}$')
        
    ##############################################################################
    plt.show()

if __name__ == '__main__':
    
    #plotOptimNnghsKriging()
    #plotInterpErrorMapsNcdcNormsTest()
    #plotNcdcNormsBiasBars()
    #plotNcdcNormsBiasMaps()
    plotInterpErrorMaps2()
    #plotNcdcNormsErrorBars()
    #plotNcdcNormsErrorMaps()
    #plotNcdcNormsErrorMapsDifs()
    #flatheadLake()
    #plotTwxDaymetPrismAnom()
    #cceBiasVsPlot()
    #cceXvalStats()
    #cceBiasVs()
    #flatheadLake()
    #plotRelImp()
    #outputCceStnsToCsv()
    #cceMae()
    #plotTwxVsPrismDaymetNorms()
    #plotCceTransectNorms()
    #plotTwxDaymetPrismAnom()
    #trendsBoxplot()
    #plotTwxVsDaymetPRISMTrend()
    #plotTwxVsPRISMTrend2()
    #plotTwxVsPrismDaymetNorms()
    #plotCceNormSeExamples2()
    #plotCcePredictorR2()
    #plotCcePredictors2()
    #plotTwxVsPRISMTrend()
    #caCoast_grtPlns_MaeVsSE()
    #greatPlainsDlyMaeVsSE()
    #caCoastMaeVsSE()
    #plotPCIAccuracy()
    #plotInterpErrorMaps()
    
    #calcConusTrendsTopoWx()
    #calcConusTrends()
    #plotConusTrends()
    #rawVsHomogTrends()
    #homogChgPtSntl()
    #plotSntlHomogDifs()
    #plotHomogDifs()
    #homogChgPtStats()
    
    #plotLSTvsAir()
    #plotTwxVsPrismDaymetNorms()
    #plotTwxVariationsVsPrismDaymetNorms()
    #plotTwxVsPrismDaymetNorms()
    #plotTwxVsPrismDaymetTmaxNorms()
    #plotTwxVsPRISMTrend()
    #The potential number of Tair input stations for GHCN, RAWS, SNOTEL
    #potential_stn_cnt(0.01)
    
    #The #/% of observations that were flagged
    #qa_flag_cnts('SNOTEL', 'tmin')
    #qa_flag_cnts('SNOTEL', 'tmax')
    #qa_flag_cnts('RAWS', 'tmin')
    #qa_flag_cnts('RAWS', 'tmax')
    #qa_flag_cnts('GHCN', 'tmin')
    #qa_flag_cnts('GHCN', 'tmax')
    
    #The final station counts after applying 5-years of data criteria
    #potential_stn_cnt(5)
    
    #Statistics on location QA and manual station moves
    #qa_loc_cnts('/projects/daymet2/station_data/all/qa_elev_final.csv')
    
    #Imputation Stats
    #impResiduals()
    
    #Homogenization Stats
#    stnda = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc')
#    ds = stnda.ds
#    stnIdsToChk = ds.variables['stn_id'][:].astype("<S16")
#    stnIdsToChk = stnIdsToChk[np.char.startswith(stnIdsToChk, 'SNOTEL')]
#    dayMask = stnda.days[YMD] >= 19900101
#    stnIds = getStnsForHomogAnomCompare(ds, stnIdsToChk, 'tmin', dayMask=dayMask, outFpath='/projects/daymet2/docs/final_writeup/stnsForHomogDifPlot/tminSNTL.txt')
#    stnIds = getStnsForHomogAnomCompare(ds, stnIdsToChk, 'tmax', dayMask=dayMask, outFpath='/projects/daymet2/docs/final_writeup/stnsForHomogDifPlot/tmaxSNTL.txt')
#    for aId in stnIds:
#        print aId
    #homogAdjBoxplot()
    #plotHomogDifs()
    #avgAnnDifsHomog()
    #topoWxVsUSHCN()
    
    #plotImpExample()
    #plotmaeVsStdErr()
    #plotMabMaps()
    
    #anomalyMap()
    #plotStns()
    #plotClimDivStnDensity()
    #plotInterpErrorMaps()
    #plotPCIAccuracy()
    #pickleClimDivsMpl()
    
    #createPredictorGrids()
    #exInterpMaps()
    
    #Plots
    #plotClimDivStnDensity()
    #plotKrigParams()
    
    #anomalyMapHCNvsTopoWx()
    