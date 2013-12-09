'''
Created on Aug 14, 2013

@author: jared.oyler
'''

import os
import numpy as np
from netCDF4 import Dataset,num2date,date2num
import twx.utils.util_dates as utld
from twx.utils.util_dates import YEAR, DATE, MONTH, DAY
from twx.db.ushcn import TairAggregate
import matplotlib.pyplot as plt
import osgeo.gdal as gdal
import osgeo.gdalconst as gdalconst
import osgeo.osr as osr
from scipy import stats
from twx.utils.status_check import status_check
from modis.clip_raster import resample_to_grd1,mask_to_rastmask
from datetime import datetime
from twx.db.station_data import BAD,LON,LAT,ELEV,BAD,LST,TDI,MASK,NEON,OPTIM_NNGH,VARIO_RNG,VARIO_NUG,VARIO_PSILL,OPTIM_NNGH_ANOM,station_data_ncdb,DTYPE_STN_BASIC,YMD,STN_ID
from twx.utils.input_raster import RasterDataset
from twx.utils.util_ncdf import NcdfRaster
import twx.interp.interp_tair as it

PROJ_GEO_WGS84 = 4326

CCE_TILES_DAYMET = np.array([12452,
                            12453,
                            12454,
                            12455,
                            12274,
                            12273,
                            12272,
                            12275,
                            12094,
                            12093])

CCE_TILES_TWX = ['h04v01','h05v01','h06v01','h05v02','h06v02']#16,17,18,37,38
PATH_CCE_MASK = '/projects/daymet2/dem/interp_grids/cce/crp_cce_us_mask.tif'

def getNorm(dsPath,varname,start_yr=1981,end_yr=2010):
    
    ds = Dataset(dsPath)
    days = utld.get_days_metadata_dates(num2date(ds.variables['time'][:],
                                                 units=ds.variables['time'].units,calendar='standard'))
    dayMask = np.nonzero(np.logical_and(days[YEAR] >= start_yr,days[YEAR] <= end_yr))[0]
    days = days[dayMask]
    agg = TairAggregate(days)
    tair = ds.variables[varname][dayMask,:,:]
    tairAnn = agg.dailyToAnn(tair)
    tairNorm = np.ma.mean(tairAnn,axis=0)
    
    ds.close()
    
    return tairNorm

def getNormsMth(dsPath,varname,start_yr=1981,end_yr=2010):
    
    ds = Dataset(dsPath)
    days = utld.get_days_metadata_dates(num2date(ds.variables['time'][:],
                                                 units=ds.variables['time'].units,calendar='standard'))
    agg = TairAggregate(days)
    tair = ds.variables[varname][:]
    tairMthNorms = agg.dailyToMthlyNorms(tair)    
    ds.close()
    
    return tairMthNorms

def getAnnTrend(dsPath,varname,start_yr=1948,end_yr=2012):
    
    ds = Dataset(dsPath)
    days = utld.get_days_metadata_dates(num2date(ds.variables['time'][:],
                                                 units=ds.variables['time'].units,calendar='standard'))
    dayMask = np.nonzero(np.logical_and(days[YEAR] >= start_yr,days[YEAR] <= end_yr))[0]
    days = days[dayMask]
    uYrs = np.unique(days[YEAR])
    tairAnn = ds.variables[varname][dayMask,:,:]
    ds.close()
    
    atrends = np.zeros((tairAnn.shape[1],tairAnn.shape[2]))
    
    schk = status_check(atrends.size,10000)
    for r in np.arange(tairAnn.shape[1]):
    
        for c in np.arange(tairAnn.shape[2]):
            
            if np.ma.is_masked(tairAnn[0,r,c]):
                atrends[r,c] = -9999
            else:
                atrends[r,c] = stats.linregress(uYrs,tairAnn[:,r,c])[0]
            
            schk.increment()
    
    atrends = np.ma.masked_equal(atrends, -9999)
    atrends.fill_value = -9999
    
    return atrends

def getTrend(dsPath,varname,start_yr=1948,end_yr=2012):
    
    ds = Dataset(dsPath)
    days = utld.get_days_metadata_dates(num2date(ds.variables['time'][:],
                                                 units=ds.variables['time'].units,calendar='standard'))
    dayMask = np.nonzero(np.logical_and(days[YEAR] >= start_yr,days[YEAR] <= end_yr))[0]
    days = days[dayMask]
    uYrs = np.unique(days[YEAR])
    agg = TairAggregate(days)
    tair = ds.variables[varname][dayMask,:,:]
    tairAnn = agg.dailyToAnn(tair)
    tair = None
    ds.close()
    
    atrends = np.zeros((tairAnn.shape[1],tairAnn.shape[2]))
    
    #schk = status_check(atrends.size,1000)
    for r in np.arange(tairAnn.shape[1]):
    
        for c in np.arange(tairAnn.shape[2]):
            
            if np.ma.is_masked(tairAnn[0,r,c]):
                atrends[r,c] = -9999
            else:
                atrends[r,c] = stats.linregress(uYrs,tairAnn[:,r,c])[0]
            
            #schk.increment()
    
    atrends = np.ma.masked_equal(atrends, -9999)
    atrends.fill_value = -9999
    
    return atrends

def getAnn(dsPath,varname,start_yr=1948,end_yr=2012):
    
    ds = Dataset(dsPath)
    days = utld.get_days_metadata_dates(num2date(ds.variables['time'][:],
                                                 units=ds.variables['time'].units,calendar='standard'))
    dayMask = np.nonzero(np.logical_and(days[YEAR] >= start_yr,days[YEAR] <= end_yr))[0]
    days = days[dayMask]
    agg = TairAggregate(days)
    tair = ds.variables[varname][dayMask,:,:]
    tairAnn = agg.dailyToAnn(tair)
    tair = None
    ds.close()
        
    tairAnn = np.ma.masked_equal(tairAnn, -9999)
    tairAnn.fill_value = -9999
    
    return tairAnn


'''
Mosaic Normals
gdalwarp -dstnodata -9999 12093_tmin_1981_2010norm.tif 12094_tmin_1981_2010norm.tif 12272_tmin_1981_2010norm.tif 12273_tmin_1981_2010norm.tif 12274_tmin_1981_2010norm.tif 12452_tmin_1981_2010norm.tif 12453_tmin_1981_2010norm.tif 12454_tmin_1981_2010norm.tif mosaic_tmin_1981_2010norm.tif
gdalwarp -dstnodata -9999 12093_tmax_1981_2010norm.tif 12094_tmax_1981_2010norm.tif 12272_tmax_1981_2010norm.tif 12273_tmax_1981_2010norm.tif 12274_tmax_1981_2010norm.tif 12452_tmax_1981_2010norm.tif 12453_tmax_1981_2010norm.tif 12454_tmax_1981_2010norm.tif mosaic_tmax_1981_2010norm.tif
'''
def outputAllNormsDaymet():
    
    pathMultiYr = "/projects/daymet2/daymet_oakridge/multi_yr_tiles/"
    outPath = "/projects/daymet2/daymet_oakridge/norms/"
    
    for tname in CCE_TILES_DAYMET:
        
        for vname in ['tmin','tmax']:
            
            dsPath = "".join([pathMultiYr,str(tname),"_",vname,"_1980_2011.nc"])
            tairNorm = getNorm(dsPath, vname)
            tairNorm.fill_value = -9999.
            outFpath = "".join([outPath,str(tname),"_",vname,"_1981_2010norm.tif"])
            print outFpath
            daymetTileToTiff(dsPath, tairNorm, outFpath)

def prism4kmToTiff(dsPath,a,pathOut):
    
    ds = Dataset(dsPath)
    
    nrow = len(ds.dimensions['lat'])
    ncol = len(ds.dimensions['lon'])
            
    driver = gdal.GetDriverByName("GTiff")
    
    if len(a.shape) == 3:
        raster = driver.Create(pathOut,ncol,nrow,a.shape[0],gdalconst.GDT_Float64) 
    else:
        raster = driver.Create(pathOut,ncol,nrow,1,gdalconst.GDT_Float64) 
    
    '''
    Create GDAL geotransform list to define resolution and bounds
    GeoTransform[0] /* top left x */
    GeoTransform[1] /* w-e pixel resolution */
    GeoTransform[2] /* rotation, 0 if image is "north up" */
    GeoTransform[3] /* top left y */
    GeoTransform[4] /* rotation, 0 if image is "north up" */
    GeoTransform[5] /* n-s pixel resolution */
    '''
    geotransform = [None]*6
    #n-s pixel height/resolution needs to be negative.
    geotransform[5] = -np.abs(ds.variables['lat'][0] - ds.variables['lat'][1])   
    geotransform[1] = np.abs(ds.variables['lon'][0] - ds.variables['lon'][1])
    geotransform[2],geotransform[4] = (0.0,0.0)
    geotransform[0] = ds.variables['lon'][0] - (geotransform[1]/2.0) 
    geotransform[3] = ds.variables['lat'][0] + np.abs(geotransform[5]/2.0)

    raster.SetGeoTransform(geotransform)

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(PROJ_GEO_WGS84)
    
    sr = osr.SpatialReference()
    sr.ImportFromProj4("+proj=latlong +ellps=WGS72 +towgs84=0,0,4.5,0,0,0.554,0.219")
    raster.SetProjection(sr.ExportToWkt())
    
    ndataVal = a.fill_value
    a = np.ma.filled(a)
    
    if len(a.shape) == 3:
        
        for x in np.arange(a.shape[0]):
            
            band = raster.GetRasterBand(int(x+1))
            band.SetNoDataValue(float(ndataVal))
            band.WriteArray(a[x,:,:],0,0) 
            band.FlushCache()
    else:
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(float(ndataVal))
        band.WriteArray(a,0,0) 
        band.FlushCache()
        
    ds.close() 
     

def topoWxTileToTiff(dsPath,a,pathOut):
    
    ds = Dataset(dsPath)
    
    nrow = len(ds.dimensions['lat'])
    ncol = len(ds.dimensions['lon'])
            
    driver = gdal.GetDriverByName("GTiff")
    
    if len(a.shape) == 3:
        raster = driver.Create(pathOut,ncol,nrow,a.shape[0],gdalconst.GDT_Float64) 
    else:
        raster = driver.Create(pathOut,ncol,nrow,1,gdalconst.GDT_Float64) 
    
    '''
    Create GDAL geotransform list to define resolution and bounds
    GeoTransform[0] /* top left x */
    GeoTransform[1] /* w-e pixel resolution */
    GeoTransform[2] /* rotation, 0 if image is "north up" */
    GeoTransform[3] /* top left y */
    GeoTransform[4] /* rotation, 0 if image is "north up" */
    GeoTransform[5] /* n-s pixel resolution */
    '''
    geotransform = [None]*6
    #n-s pixel height/resolution needs to be negative.
    geotransform[5] = -np.abs(ds.variables['lat'][0] - ds.variables['lat'][1])   
    geotransform[1] = np.abs(ds.variables['lon'][0] - ds.variables['lon'][1])
    geotransform[2],geotransform[4] = (0.0,0.0)
    geotransform[0] = ds.variables['lon'][0] - (geotransform[1]/2.0) 
    geotransform[3] = ds.variables['lat'][0] + np.abs(geotransform[5]/2.0)

    raster.SetGeoTransform(geotransform)

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(PROJ_GEO_WGS84)
    raster.SetProjection(sr.ExportToWkt())
    
    ndataVal = a.fill_value
    a = np.ma.filled(a)
    
    if len(a.shape) == 3:
        
        for x in np.arange(a.shape[0]):
            
            band = raster.GetRasterBand(int(x+1))
            band.SetNoDataValue(float(ndataVal))
            band.WriteArray(a[x,:,:],0,0) 
            band.FlushCache()
    else:
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(float(ndataVal))
        band.WriteArray(a,0,0) 
        band.FlushCache()
    
    ds.close() 

def daymetTileToTiff(dsPath,a,pathOut):
    
    ds = Dataset(dsPath)
    
    nrow = len(ds.dimensions['y'])
    ncol = len(ds.dimensions['x'])
            
    driver = gdal.GetDriverByName("GTiff")
    
    if len(a.shape) == 3:
        raster = driver.Create(pathOut,ncol,nrow,a.shape[0],gdalconst.GDT_Float64) 
    else:
        raster = driver.Create(pathOut,ncol,nrow,1,gdalconst.GDT_Float64) 
        
    '''
    Create GDAL geotransform list to define resolution and bounds
    GeoTransform[0] /* top left x */
    GeoTransform[1] /* w-e pixel resolution */
    GeoTransform[2] /* rotation, 0 if image is "north up" */
    GeoTransform[3] /* top left y */
    GeoTransform[4] /* rotation, 0 if image is "north up" */
    GeoTransform[5] /* n-s pixel resolution */
    '''
    geotransform = [None]*6
    #n-s pixel height/resolution needs to be negative.
    geotransform[5] = -np.abs(ds.variables['y'][0] - ds.variables['y'][1])   
    geotransform[1] = np.abs(ds.variables['x'][0] - ds.variables['x'][1])
    geotransform[2],geotransform[4] = (0.0,0.0)
    geotransform[0] = ds.variables['x'][0] - (geotransform[1]/2.0) 
    geotransform[3] = ds.variables['y'][0] + np.abs(geotransform[5]/2.0)

    raster.SetGeoTransform(geotransform)

    sr = osr.SpatialReference()
    sr.ImportFromProj4("+proj=lcc +datum=WGS84 +lat_1=25n +lat_2=60n +lat_0=42.5n +lon_0=100w")
    raster.SetProjection(sr.ExportToWkt())
    
    ndataVal = a.fill_value
    a = np.ma.filled(a)

    if len(a.shape) == 3:
        
        for x in np.arange(a.shape[0]):
            
            band = raster.GetRasterBand(int(x+1))
            band.SetNoDataValue(float(ndataVal))
            band.WriteArray(a[x,:,:],0,0) 
            band.FlushCache()
    else:
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(float(ndataVal))
        band.WriteArray(a,0,0) 
        band.FlushCache()
    
    ds.close()       

def mosaic(in_files,out_mosaic,ndata=-9999):

    cmd = ["".join(["gdalwarp -dstnodata ",str(ndata)])]
    cmd.extend(in_files)
    cmd.append(out_mosaic)
    
    print "running mosaic cmd..."+" ".join(cmd)
    os.system(" ".join(cmd))

def resample_prismwgs72_to_wgs84(fpath_dstgrid,fpath_srcgrid,fpath_out,resample_alg=gdalconst.GRA_NearestNeighbour):
    grid_dst = gdal.Open(fpath_dstgrid)
    grid_src = gdal.Open(fpath_srcgrid)
    
    dst_proj = grid_dst.GetProjection()
    dst_geot = grid_dst.GetGeoTransform()
        
    src_proj = "+proj=latlong +ellps=WGS72 +towgs84=0,0,4.5,0,0,0.554,0.219"
    sr_wgs72 = osr.SpatialReference()
    sr_wgs72.ImportFromProj4(src_proj)
    src_proj = sr_wgs72.ExportToWkt()
    
    src_band = grid_src.GetRasterBand(1)
    src_dtype = src_band.DataType
    src_ndata =  src_band.GetNoDataValue()
    
    grid_out = gdal.GetDriverByName('GTiff').Create(fpath_out, 
                                                    grid_dst.RasterXSize, 
                                                    grid_dst.RasterYSize, grid_src.RasterCount, src_dtype)
    
    band_out = grid_out.GetRasterBand(1)
    band_out.Fill(src_ndata)
    band_out.SetNoDataValue(src_ndata)
    
    grid_out.SetGeoTransform(dst_geot)
    grid_out.SetProjection(dst_proj)
    
    gdal.ReprojectImage(grid_src, grid_out, src_proj, dst_proj, resample_alg)
    
    grid_out.FlushCache()


def prism_ann4km_to_netcdf(prism_fpath,tair_var,out_fpath,start_yr=1948,end_yr=2012):
    
    ds_in = gdal.Open("".join([prism_fpath,"us_",tair_var,"_",str(start_yr),".14.asc"]))
    '''
    Get GDAL geotransform list to define resolution and bounds
    GeoTransform[0] /* top left x */
    GeoTransform[1] /* w-e pixel resolution */
    GeoTransform[2] /* rotation, 0 if image is "north up" */
    GeoTransform[3] /* top left y */
    GeoTransform[4] /* rotation, 0 if image is "north up" */
    GeoTransform[5] /* n-s pixel resolution */
    '''
    geoT = ds_in.GetGeoTransform()
    
    #Get the upper left and right point lon
    ulLon = geoT[0] + (geoT[1] / 2.0)
    urLon = (geoT[0] + (ds_in.RasterXSize-1)*geoT[1] + (ds_in.RasterYSize-1)*geoT[2]) + geoT[1] / 2.0
    
    #Get the upper left and lower left lat
    ulLat = geoT[3] + (geoT[5] / 2.0)
    llLat = (geoT[3] + (ds_in.RasterXSize-1)*geoT[4] + (ds_in.RasterYSize-1)*geoT[5]) + geoT[5] / 2.0
    
    #Build arrays of lat/lon
    lons = np.linspace(ulLon, urLon, ds_in.RasterXSize)
    lats = np.linspace(ulLat, llLat, ds_in.RasterYSize)
    
    
    ds_out = Dataset(out_fpath,'w')
    
    #Create lat/lon dimensions and variables
    ds_out.createDimension('lat',lats.size)
    ds_out.createDimension('lon',lons.size)

    latitudes = ds_out.createVariable('lat','f8',('lat',))
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ds_out.createVariable('lon','f8',('lon',))
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    yrs = np.arange(start_yr,end_yr+1)
    dates = [datetime(yr,1,1) for yr in yrs]
    
    ds_out.createDimension('time',yrs.size)
    time_var = ds_out.createVariable('time','f8',('time',))
    time_var.long_name = "time"
    time_var.calendar = "standard"
    time_var.units = "".join(["days since ",str(start_yr),"-1-1 0:0:0"])
    time_var[:] = date2num(dates,time_var.units)
    
    tair = ds_out.createVariable(tair_var,'f8',('time','lat','lon'),fill_value=-9999.)
    tair.long_name = "daily temperature" ;
    tair.units = "degrees C" ;
    tair.missing_value = -9999.
    
    for yr,x in zip(yrs,np.arange(yrs.size)):
        
        print yr
        
        ds_in = gdal.Open("".join([prism_fpath,"us_",tair_var,"_",str(yr),".14.asc"]))
        a = ds_in.GetRasterBand(1).ReadAsArray()
        ndata = ds_in.GetRasterBand(1).GetNoDataValue()
        a = np.ma.masked_equal(a, ndata)
        a = a/100.0 #PRISM scale factor
        
        tair[x,:,:] = a
        ds_out.sync()

def getStnVarTo(dsFrom,dsTo,varName):
    
    if varName not in dsTo.variables.keys():
        
        attrNames = dsFrom.variables[varName].ncattrs()
        
        fillValue = None
        if '_FillValue' in attrNames:
            fillValue = dsFrom.variables[varName].getncattr('_FillValue')
          
        avar = dsTo.createVariable(varName,dsFrom.variables[varName].dtype,('stn_id',),fill_value=fillValue)
        
        for aAttr in attrNames:
            
            if aAttr != '_FillValue':
                avar.setncattr(aAttr,dsFrom.variables[varName].getncattr(aAttr))
        
        dsTo.sync()
        
        return avar
        
    else:
        
        return dsTo.variables[varName]

def copyStnAttrs(fromNcPath,toNcPath):

    dsFrom = Dataset(fromNcPath)
    dsTo = Dataset(toNcPath,'r+')
    
    stnIdsFrom = dsFrom.variables['stn_id'][:].astype("<S16")
    stnIdsTo = dsTo.variables['stn_id'][:].astype("<S16")
    
    stnMaskTo = np.in1d(stnIdsTo, stnIdsFrom, True)
    stnMaskFrom = np.in1d(stnIdsFrom, stnIdsTo, True)
    print np.sum(stnMaskTo),np.sum(stnMaskFrom)
    print np.sum(stnIdsTo[stnMaskTo] != stnIdsFrom[stnMaskFrom])
    #Set stations not in dsFrom to bad in dsTo
    varBadTo = getStnVarTo(dsFrom, dsTo, BAD)
    varBadTo[~stnMaskTo] = 1
    
    VARS_TO_COPY = [LON,LAT,ELEV,BAD,LST,TDI,MASK,NEON,OPTIM_NNGH,VARIO_NUG,VARIO_RNG,VARIO_PSILL,OPTIM_NNGH_ANOM]
    
    for aVarName in VARS_TO_COPY:
        
        varTo = getStnVarTo(dsFrom, dsTo, aVarName)
        varTo[stnMaskTo] = dsFrom.variables[aVarName][stnMaskFrom]
        dsTo.sync()
        
    dsTo.close()
    dsFrom.close()
    

def calcSummaryStatTwx(tilePath,varName,tileNames,aStatFunc,outPath):

    for atile in tileNames:
        ds_path = "".join([tilePath,atile,"/",atile,"_",varName,".nc"])
        ds = Dataset(ds_path)
        tairStat = aStatFunc(ds)
        outFpath = "".join([outPath,atile,"_",varName,".tif"])
        topoWxTileToTiff(ds_path, tairStat,outFpath)
        ds.close()
        ds = None
        
def mosaicStatTilesTwx(tilePath):
    os.chdir(tilePath)
    files = np.array(os.listdir(tilePath))
    mosaic(files,"mosaic.tif")

def cropToCceTmx(tilePath):

    resample_to_grd1(PATH_CCE_MASK, 
                     "".join([tilePath,"mosaic.tif"]), 
                     "".join([tilePath,"temp_cce_mosaic.tif"]),
                     gdalconst.GRA_NearestNeighbour)
    mask_to_rastmask(PATH_CCE_MASK, "".join([tilePath,"temp_cce_mosaic.tif"]), "".join([tilePath,"cce_final.tif"]))
    os.system("".join(['rm ',tilePath,"temp_cce_mosaic.tif"]))

def outCceStat(tilePath,varName,tileNames,aStatFunc,outPath):
    calcSummaryStatTwx(tilePath, varName, tileNames, aStatFunc, outPath)
    mosaicStatTilesTwx(outPath)
    cropToCceTmx(outPath)

def prismTrendEg():
    
    start_yr = 1948
    end_yr = 2012
    
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
    
    tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpLonLatPt(-113.878048, 47.347526)
    print ptInterper.a_pt[LON],ptInterper.a_pt[LAT]
    
    ag = TairAggregate(ptInterper.days)
    twxAnn = ag.dailyToAnn(tmin_dly)
    
    
    ds = NcdfRaster('/projects/daymet2/prism/4km_annual/tmin/prism4km_tmin_ann1948-2012.nc')
    x,y = ds.getGridCellOffset(ptInterper.a_pt[LON], ptInterper.a_pt[LAT])  
    tair =  ds.ds.variables['tmin'][:,y,x]
    print ds.ds.variables['lat'][y],ds.ds.variables['lon'][x]
    
    days = utld.get_days_metadata_dates(num2date(ds.ds.variables['time'][:],units=ds.ds.variables['time'].units,calendar='standard'))
    dayMask = np.nonzero(np.logical_and(days[YEAR] >= start_yr,days[YEAR] <= end_yr))[0]
    days = days[dayMask]
    uYrs = np.unique(days[YEAR])
    
    plt.plot(uYrs,tair[dayMask]-np.mean(tair[dayMask]))
    plt.plot(uYrs,twxAnn-np.mean(twxAnn))
    plt.legend(['PRISM 4km','TopoWx'],loc=2)
    plt.ylabel('$^\circ$C')
    plt.xlabel("Year")
    plt.title("Annual Anomalies 1948-2012\n"+str(ptInterper.a_pt[LON])+", "+str(ptInterper.a_pt[LAT]))
    plt.xlim((1948,2012))
    plt.show()

class InterpObs():
    
    def __init__(self):
        pass

    def getObs(self,aPt,tairVar):
        pass
    
class InterpObsTwx(InterpObs):
    
    def __init__(self):
        
        auxFpaths = ['/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc',
                     '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc',
                     '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax.nc',
                     '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin.nc',
                     '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_climdiv.nc']
                
#        self.ptInterper = it.PtInterpTair('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc',
#                        '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc',
#                        '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
#                        '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
#                        auxFpaths)
        #Non-homog
        self.ptInterper = it.PtInterpTair('/projects/daymet2/station_data/infill/infill_20130518/serial_tmin.nc',
                        '/projects/daymet2/station_data/infill/infill_20130518/serial_tmax.nc',
                        '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                        '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                        auxFpaths)
        
    def getObs(self,aPt,tairVar):
        tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = self.ptInterper.interpLonLatPt(aPt[LON], aPt[LAT])
        
        if tairVar == 'tmin':
            return tmin_dly
        elif tairVar == 'tmax':
            return tmax_dly
        else:
            raise Exception("Unrecognized TairVar")

class InterpObsPRISM(InterpObs):
    
    def __init__(self):
        
        auxFpaths = ['/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc',
                     '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc',
                     '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax.nc',
                     '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin.nc',
                     '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_climdiv.nc']
                
        self.ptInterper = it.PtInterpTair('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc',
                        '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc',
                        '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                        '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                        auxFpaths)
        
        self.dsTwxTmin = RasterDataset('/projects/daymet2/compare/topowx_files/normals/cce_topowx_tmin19812010norm.tif')
        self.dsPrismTmin = RasterDataset('/projects/daymet2/compare/prism_files/normals/cce_prism_tmin1981_2010norm.tif')
        self.dsTwxTmax = RasterDataset('/projects/daymet2/compare/topowx_files/normals/cce_topowx_tmax19812010norm.tif')
        self.dsPrismTmax = RasterDataset('/projects/daymet2/compare/prism_files/normals/cce_prism_tmax1981_2010norm.tif')
        

    def getObs(self,aPt,tairVar):
        tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = self.ptInterper.interpLonLatPt(aPt[LON], aPt[LAT])
        
        if tairVar == 'tmin':
            
            normTwx = self.dsTwxTmin.getDataValue(aPt[LON], aPt[LAT])
            normPRISM = self.dsPrismTmin.getDataValue(aPt[LON], aPt[LAT])/100.0
            return tmin_dly+(normPRISM-normTwx)
        elif tairVar == 'tmax':
            normTwx = self.dsTwxTmax.getDataValue(aPt[LON], aPt[LAT])
            normPRISM = self.dsPrismTmax.getDataValue(aPt[LON], aPt[LAT])/100.0
            return tmax_dly+(normPRISM-normTwx)
        else:
            raise Exception("Unrecognized TairVar")
     
class InterpObsDaymet(InterpObs):
    
    def __init__(self):
        
        self.pathData = '/projects/daymet2/compare/daymet_files/daymet_pt_download/'
        a = np.loadtxt("".join([self.pathData,'GLAC_7.csv']), delimiter=",",skiprows=7)
        self.days = utld.get_days_metadata_daymet(a[:,0],a[:,1])
        

    def getObs(self,aPt,tairVar):
        
        a = np.loadtxt("".join([self.pathData,aPt[STN_ID],'.csv']), delimiter=",",skiprows=7)
        
        tmin_dly = a[:,3]
        tmax_dly = a[:,2]
        
        if tairVar == 'tmin':
            return tmin_dly
        elif tairVar == 'tmax':
            return tmax_dly
        else:
            raise Exception("Unrecognized TairVar")
        
class GlacStnValidate():

    STNIDS_RM = np.array(['GLAC_2','GLAC_8','GLAC_9','GLAC_10','GLAC_13'])
    
    def __init__(self,pathDsGlac,daysIn):

        self.stnda = station_data_ncdb(pathDsGlac,stnDtype=DTYPE_STN_BASIC)
        self.ymdMaskIn = np.in1d(daysIn[YMD], self.stnda.days[YMD], True)
        self.ymdMaskGlac = np.in1d(self.stnda.days[YMD], daysIn[YMD], True)
        self.stnsGlac = self.stnda.stns[np.logical_and(np.char.startswith(self.stnda.stn_ids, 'GLAC'),
                                                       ~np.in1d(self.stnda.stn_ids,self.STNIDS_RM))]
        self.obsTmin = {}
        self.obsTmax = {}
        
        for aStn in self.stnsGlac:
            
            self.obsTmin[aStn[STN_ID]] = self.stnda.load_all_stn_obs_var(aStn[STN_ID],'tmin')[0]
            self.obsTmax[aStn[STN_ID]] = self.stnda.load_all_stn_obs_var(aStn[STN_ID],'tmax')[0]

    def compareAll(self,aInterpObs,tairVar):
        
        maeAll = []
        biasAll = []
        r2All = []
        
        for aStn in self.stnsGlac:
            
            mae,bias,r2 = self.compareTair(aInterpObs.getObs(aStn,tairVar),tairVar,aStn[STN_ID])
            maeAll.append(mae)
            biasAll.append(bias)
            r2All.append(r2)
        
        return maeAll,biasAll,r2All
        

    def compareTair(self,tair,tairVar,stnId):
        
        if tairVar == 'tmin':
            tairObs = self.obsTmin[stnId]
        elif tairVar == 'tmax':
            tairObs = self.obsTmax[stnId]
        else:
            raise Exception("Unrecognized TairVar")
        
        tair = tair[self.ymdMaskIn]
        tairObs = tairObs[self.ymdMaskGlac]
        finMask = np.isfinite(tairObs)
        
        difs = tair[finMask]-tairObs[finMask]
        mae = np.mean(np.abs(difs))
        bias = np.mean(difs)
        r2 = stats.linregress(tair[finMask], tairObs[finMask])[2]**2
        
        return mae,bias,r2
        
def createGlacDaymetDownloadFile():
    interpObs = InterpObsTwx()
    glacValid = GlacStnValidate('/projects/daymet2/station_data/crown_stns_final.nc', interpObs.ptInterper.days)
    
    fileOut = open('/projects/daymet2/compare/daymet_files/daymet_pt_download/glaclatlon.txt','w')
    
    for aStn,x in zip(glacValid.stnsGlac,np.arange(glacValid.stnsGlac.size)):
        fileOut.write("".join([aStn[STN_ID],'.csv,',str(aStn[LAT]),",",str(aStn[LON]),", ingore stuff",str(x+1),"\n"]))
    
    fileOut.close()

class DaymetTileRaster():
    
    def __init__(self,fname,varname):
        #Coords are latlon or xy
        self.ds = Dataset(fname,"r")
        self.aVar = self.ds.variables[varname]
            
        self.x = self.ds.variables['x'][:]
        self.y = self.ds.variables['y'][:]
            
        '''
        Create GDAL geotransform list to define resolution and bounds
        GeoTransform[0] /* top left x */
        GeoTransform[1] /* w-e pixel resolution */
        GeoTransform[2] /* rotation, 0 if image is "north up" */
        GeoTransform[3] /* top left y */
        GeoTransform[4] /* rotation, 0 if image is "north up" */
        GeoTransform[5] /* n-s pixel resolution */
        '''
        self.geoTransform = [None]*6
        #n-s pixel height/resolution needs to be negative.  not sure why?
        self.geoTransform[5] = -np.abs(self.y[0] - self.y[1])   
        self.geoTransform[1] = np.abs(self.x[0] - self.x[1])
        self.geoTransform[2],self.geoTransform[4] = (0.0,0.0)
        self.geoTransform[0] = self.x[0] - (self.geoTransform[1]/2.0) 
        self.geoTransform[3] = self.y[0] + np.abs(self.geoTransform[5]/2.0)
                
        self.min_x = self.geoTransform[0]
        self.max_x = self.min_x + (self.x.size*self.geoTransform[1])
        self.max_y =  self.geoTransform[3]
        self.min_y =  self.max_y - (-self.y.size*self.geoTransform[5])
        
        sr = osr.SpatialReference()
        sr.ImportFromProj4("+proj=lcc +datum=WGS84 +lat_1=25n +lat_2=60n +lat_0=42.5n +lon_0=100w")
        self.sourceSR = sr
        
        self.targetSR = osr.SpatialReference()
        self.targetSR.ImportFromEPSG(PROJ_GEO_WGS84)
        
        self.coordTrans_src_to_wgs84 = osr.CoordinateTransformation(self.sourceSR, self.targetSR)
        self.coordTrans_wgs84_to_src = osr.CoordinateTransformation(self.targetSR, self.sourceSR)
        
        self.daysNoLeap = utld.get_days_metadata_dates(num2date(self.ds.variables['time'][:], self.ds.variables['time'].units))
        self.days = utld.get_days_metadata(self.daysNoLeap[DATE][0], self.daysNoLeap[DATE][-1])
        self.leapDayMask = ~np.logical_and(self.days[MONTH] == 2, self.days[DAY] == 29)
        
    def getGridCellOffset(self,lon,lat):
        
        x, y, z = self.coordTrans_wgs84_to_src.TransformPoint(lon,lat) 
        
        if not self.is_inbounds(x, y):
            raise Exception("Lon/Lat outside raster extent")
        
        originX = self.geoTransform[0]
        originY = self.geoTransform[3]
        pixelWidth = self.geoTransform[1]
        pixelHeight = self.geoTransform[5]
        
        xOffset = abs(int((x - originX) / pixelWidth))
        yOffset = abs(int((y - originY) / pixelHeight))
        return xOffset,yOffset
    
    def getTimeSeries(self,lon,lat):
        x,y = self.getGridCellOffset(lon, lat)
        ts = self.aVar[:,y,x]
        
        if np.ma.isMaskedArray(ts):
            ts = ts.data
            
        if ts[0] == self.aVar._FillValue:
            raise Exception("Lon/Lat has no data for this tile")
        
        tsFnl = np.ones(self.days.size)*np.nan
        tsFnl[self.leapDayMask] = ts
        
        return tsFnl
    
    def is_inbounds(self,x,y):
        return x >= self.min_x and x <= self.max_x and y >= self.min_y and y <= self.max_y

class MultiDaymetTileRaster():
    
    def __init__(self,tilePath,varname):
        
        fnames = np.array(os.listdir(tilePath))
        fnames = fnames[np.char.find(fnames, varname) != -1]
        self.dsTiles = [DaymetTileRaster("".join([tilePath,fname]),varname) for fname in fnames]
        self.days = self.dsTiles[0].days
              
    def getTimeSeries(self,lon,lat):
        
        ptFound = False
        for ds in self.dsTiles:
            try:
                ts = ds.getTimeSeries(lon,lat)
                ptFound = True
                break
            except Exception:
                continue
        
        if not ptFound:
            raise Exception("Could not find point in tiles.")
        return ts

class PrismTileRaster():
    
    def __init__(self,fname,varname):
        #Coords are latlon or xy
        self.ds = Dataset(fname)
        self.aVar = self.ds.variables[varname]
            
        self.x = self.ds.variables['lon'][:]
        self.y = self.ds.variables['lat'][:]
            
        '''
        Create GDAL geotransform list to define resolution and bounds
        GeoTransform[0] /* top left x */
        GeoTransform[1] /* w-e pixel resolution */
        GeoTransform[2] /* rotation, 0 if image is "north up" */
        GeoTransform[3] /* top left y */
        GeoTransform[4] /* rotation, 0 if image is "north up" */
        GeoTransform[5] /* n-s pixel resolution */
        '''
        self.geoTransform = [None]*6
        #n-s pixel height/resolution needs to be negative.  not sure why?
        self.geoTransform[5] = -np.abs(self.y[0] - self.y[1])   
        self.geoTransform[1] = np.abs(self.x[0] - self.x[1])
        self.geoTransform[2],self.geoTransform[4] = (0.0,0.0)
        self.geoTransform[0] = self.x[0] - (self.geoTransform[1]/2.0) 
        self.geoTransform[3] = self.y[0] + np.abs(self.geoTransform[5]/2.0)
                
        self.min_x = self.geoTransform[0]
        self.max_x = self.min_x + (self.x.size*self.geoTransform[1])
        self.max_y =  self.geoTransform[3]
        self.min_y =  self.max_y - (-self.y.size*self.geoTransform[5])
                
        self.days = utld.get_days_metadata_dates(num2date(self.ds.variables['time'][:], self.ds.variables['time'].units))
          
    def getGridCellOffset(self,lon,lat):
        
        x,y = lon,lat
        
        if not self.is_inbounds(x, y):
            raise Exception("Lon/Lat outside raster extent")
        
        originX = self.geoTransform[0]
        originY = self.geoTransform[3]
        pixelWidth = self.geoTransform[1]
        pixelHeight = self.geoTransform[5]
        
        xOffset = abs(int((x - originX) / pixelWidth))
        yOffset = abs(int((y - originY) / pixelHeight))
        return xOffset,yOffset
    
    def getTimeSeries(self,lon,lat):
        x,y = self.getGridCellOffset(lon, lat)
        return self.aVar[:,y,x]
    
    def is_inbounds(self,x,y):
        return x >= self.min_x and x <= self.max_x and y >= self.min_y and y <= self.max_y

class MultiTwxTileRaster():
    
    def __init__(self,tilePath,varname,tiles=CCE_TILES_TWX):
        
        self.dsTiles = []
        for atile in CCE_TILES_TWX:
            self.dsTiles.append(PrismTileRaster("".join([tilePath,atile,"/",atile,"_",varname,".nc"]), varname))
        self.days = self.dsTiles[0].days
              
    def getTimeSeries(self,lon,lat):
        
        ptFound = False
        for ds in self.dsTiles:
            try:
                ts = ds.getTimeSeries(lon,lat)
                ptFound = True
                break
            except Exception:
                continue
        
        if not ptFound:
            raise Exception("Could not find point in tiles.")
        return ts

if __name__ == '__main__':
    
    #PRISM Glac Validate
#    interpObs = InterpObsPRISM()
#    glacValid = GlacStnValidate('/projects/daymet2/station_data/crown_stns_final.nc', interpObs.ptInterper.days)
#    maeAll, biasAll, r2All = glacValid.compareAll(interpObs, 'tmin')
#    print 'TMIN',np.mean(maeAll),np.mean(biasAll),np.mean(r2All)
#    maeAll, biasAll, r2All = glacValid.compareAll(interpObs, 'tmax')
#    print 'TMAX',np.mean(maeAll),np.mean(biasAll),np.mean(r2All)
    
    #Daymet Glac Validate
#    interpObs = InterpObsDaymet()
#    glacValid = GlacStnValidate('/projects/daymet2/station_data/crown_stns_final.nc', interpObs.days)
#    maeAll, biasAll, r2All = glacValid.compareAll(interpObs, 'tmax')
#    print np.mean(maeAll),np.mean(biasAll),np.mean(r2All)
    
    #TopoWX Glac Validate
#    interpObs = InterpObsTwx()
#    glacValid = GlacStnValidate('/projects/daymet2/station_data/crown_stns_final.nc', interpObs.ptInterper.days)
#    maeAll, biasAll, r2All = glacValid.compareAll(interpObs, 'tmin')
#    print np.mean(maeAll),np.mean(biasAll),np.mean(r2All)
#    maeAll, biasAll, r2All = glacValid.compareAll(interpObs, 'tmax')
#    print np.mean(maeAll),np.mean(biasAll),np.mean(r2All)
    
    #prismTrendEg()
    
#    copyStnAttrs('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 
#                 '/projects/daymet2/station_data/infill/infill_20130518/serial_tmin.nc')
    
    ################################################
    #1.) Calculate 1981-2010 Normals for applicable TWX and Daymet Tiles
    ################################################
    #TWX
    #Normal Product Paths
#    tilepath = '/stage/climate/topowx_tile_output/'
#    outpath = '/projects/daymet2/compare/topowx_files/normals/'
    #No LST Paths
#    tilepath = '/stage/climate/topowx_nolst/'
#    outpath = '/projects/daymet2/compare/topowx_files/normals/no_lst/'
    #Non-Homog station data
#    tilepath = '/stage/climate/topowx_nohomog/'
#    outpath = '/projects/daymet2/compare/topowx_files/normals/no_homog/'
    #Non-Homog station data, no LST
#    tilepath = '/stage/climate/topowx_nohomog_nolst/'
#    outpath = '/projects/daymet2/compare/topowx_files/normals/no_homog_lst/'
#    for atile in CCE_TILES_TWX:
#        for atair in ['tmin','tmax']:
#            print atile,atair
#            ds_path = "".join([tilepath,atile,"/",atile,"_",atair,".nc"])
#            tairNorm = getNorm(ds_path,atair)
#            tairNorm.fill_value = -9999.
#            outFpath = "".join([outpath,atile,"_",atair,"_19812010norm.tif"])
#            print outFpath
#            topoWxTileToTiff(ds_path, tairNorm,outFpath)
#    
#    #Daymet
#    tilepath = "/projects/daymet2/daymet_oakridge/multi_yr_tiles/"
#    outpath = "/projects/daymet2/compare/daymet_files/normals/"
#    for tname in CCE_TILES_DAYMET:
#        
#        for vname in ['tmin','tmax']:
#            
#            dsPath = "".join([tilepath,str(tname),"_",vname,"_1980_2011.nc"])
#            tairNorm = getNorm(dsPath, vname)
#            tairNorm.fill_value = -9999.
#            outFpath = "".join([outpath,str(tname),"_",vname,"_1981_2010norm.tif"])
#            print outFpath
#            daymetTileToTiff(dsPath, tairNorm, outFpath)


#    #Daymet Monthly Norms
#    tilepath = "/projects/daymet2/daymet_oakridge/multi_yr_tiles/"
#    outpath = "/projects/daymet2/cce_case_study/daymet_files/normals_mthly/"
#    for tname in CCE_TILES_DAYMET:
#        
#        for vname in ['tmin','tmax']:
#            
#            dsPath = "".join([tilepath,str(tname),"_",vname,"_1980_2011.nc"])
#            tairMthNorms = getNormsMth(dsPath, vname)
#            tairMthNorms.fill_value = -9999.
#            
#            for mth in np.arange(1,13):
#            
#                outFpath = "".join([outpath,str(tname),"_",vname,"_1981_2010norm%02d.tif"%mth])
#                daymetTileToTiff(dsPath, tairMthNorms[mth-1,:,:], outFpath)
    
    ################################################
    #2.) Calculate 1981-2010 Trends for applicable TWX and Daymet Tiles, and PRISM
    ################################################
    #TWX
#    tilepath = '/stage/climate/topowx_tile_output/'
#    outpath = '/projects/daymet2/cce_case_study/topowx_files/trends/'
#    #No LST Paths
##    tilepath = '/stage/climate/topowx_nolst/'
##    outpath = '/projects/daymet2/compare/topowx_files/trends/no_lst/'
#    #Non-Homog station data
##    tilepath = '/stage/climate/topowx_nohomog/'
##    outpath = '/projects/daymet2/compare/topowx_files/trends/no_homog/'
##    #Non-Homog station data, no LST
##    tilepath = '/stage/climate/topowx_nohomog_nolst/'
##    outpath = '/projects/daymet2/compare/topowx_files/trends/no_homog_lst/'
#    for atile in CCE_TILES_TWX:
#        for atair in ['tmin','tmax']:
#            ds_path = "".join([tilepath,atile,"/",atile,"_",atair,".nc"])
#            tairTrend = getTrend(ds_path,atair,1981,2010)
#            outFpath =  "".join([outpath,atile,"_",atair,"_19812010trend.tif"])
#            print outFpath
#            topoWxTileToTiff(ds_path, tairTrend,outFpath)
#    
#    #Daymet     
#    tilepath = "/projects/daymet2/daymet_oakridge/multi_yr_tiles/"
#    outpath = "/projects/daymet2/compare/daymet_files/trends/"
#    for tname in CCE_TILES_DAYMET:
#        for vname in ['tmin','tmax']:
#            dsPath = "".join([tilepath,str(tname),"_",vname,"_1980_2011.nc"])
#            tairTrend = getTrend(dsPath, vname,1981,2010)
#            outFpath = "".join([outpath,str(tname),"_",vname,"_1981_2010trend.tif"])
#            daymetTileToTiff(dsPath, tairTrend, outFpath)
#    
#    #PRISM
#    dsPathTmax = '/projects/daymet2/prism/4km_annual/tmax/prism4km_tmax_ann1948-2012.nc'
#    tairTrend = getAnnTrend(dsPathTmax, 'tmax', 1981, 2010)
#    prism4kmToTiff(dsPathTmax, tairTrend, 
#                   '/projects/daymet2/compare/prism_files/trends/prism4km_tmax_trend1981-2010.tif')
#    dsPathTmin = '/projects/daymet2/prism/4km_annual/tmin/prism4km_tmin_ann1948-2012.nc'
#    tairTrend = getAnnTrend(dsPathTmin, 'tmin', 1981, 2010)
#    prism4kmToTiff(dsPathTmin, tairTrend, 
#                   '/projects/daymet2/compare/prism_files/trends/prism4km_tmin_trend1981-2010.tif')
#
#    ################################################
#    #3.) Calculate 1948-2012 Trends for applicable TWX Tiles, and PRISM
#    ################################################
    #TWX
#    tilepath = '/stage/climate/topowx_tile_output/'
#    outpath = '/projects/daymet2/cce_case_study/topowx_files/trends/'
#    #No LST Paths
##    tilepath = '/stage/climate/topowx_nolst/'
##    outpath = '/projects/daymet2/compare/topowx_files/trends/no_lst/'
#    #Non-Homog station data
##    tilepath = '/stage/climate/topowx_nohomog/'
##    outpath = '/projects/daymet2/compare/topowx_files/trends/no_homog/'
##    #Non-Homog station data, no LST
##    tilepath = '/stage/climate/topowx_nohomog_nolst/'
##    outpath = '/projects/daymet2/compare/topowx_files/trends/no_homog_lst/'
#    for atile in CCE_TILES_TWX:
#        for atair in ['tmin','tmax']:
#            ds_path = "".join([tilepath,atile,"/",atile,"_",atair,".nc"])
#            tairTrend = getTrend(ds_path,atair,1948,2012)
#            outFpath =  "".join([outpath,atile,"_",atair,"_19482012trend.tif"])
#            print outFpath
#            topoWxTileToTiff(ds_path, tairTrend,outFpath)
#            
#    #PRISM
#    dsPathTmax = '/projects/daymet2/prism/4km_annual/tmax/prism4km_tmax_ann1948-2012.nc'
#    tairTrend = getAnnTrend(dsPathTmax, 'tmax', 1948, 2012)
#    prism4kmToTiff(dsPathTmax, tairTrend, 
#                   '/projects/daymet2/compare/prism_files/trends/prism4km_tmax_trend1948-2012.tif')
#    dsPathTmin = '/projects/daymet2/prism/4km_annual/tmin/prism4km_tmin_ann1948-2012.nc'
#    tairTrend = getAnnTrend(dsPathTmin, 'tmin', 1948, 2012)
#    prism4kmToTiff(dsPathTmin, tairTrend, 
#                   '/projects/daymet2/compare/prism_files/trends/prism4km_tmin_trend1948-2012.tif')
#    
#    ################################################
#    #4.) Mosaic 1981-2010 Normals of TMX and Daymet tiles
#    ################################################
#    #TWX
#    normals_path = '/projects/daymet2/compare/topowx_files/normals/'
    #normals_path = '/projects/daymet2/compare/topowx_files/normals/no_lst/'
    #normals_path = '/projects/daymet2/compare/topowx_files/normals/no_homog/'
#    normals_path = '/projects/daymet2/compare/topowx_files/normals/no_homog_lst/'
#
#    os.chdir(normals_path)
#    files = np.array(os.listdir(normals_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmin') != -1,
#                                 np.char.endswith(files, '19812010norm.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_topowx_tmin19812010norm.tif")
#    
#    files = np.array(os.listdir(normals_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmax') != -1,
#                                 np.char.endswith(files, '19812010norm.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_topowx_tmax19812010norm.tif")
#    
#    #Daymet
#    normals_path = '/projects/daymet2/compare/daymet_files/normals/'  
#    os.chdir(normals_path)
#    files = np.array(os.listdir(normals_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmin') != -1,
#                                 np.char.endswith(files, '1981_2010norm.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_daymet_tmin19812010norm.tif")
#    
#    files = np.array(os.listdir(normals_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmax') != -1,
#                                 np.char.endswith(files, '1981_2010norm.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_daymet_tmax19812010norm.tif")
  

    #Daymet Monthly
#    tilePath = '/projects/daymet2/cce_case_study/daymet_files/normals_mthly/'
#    pathOut = '/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/'
#    
#    os.chdir(tilePath)
#    
#    for mth in np.arange(1,13):
#        
#        fnames = ["%d_tmin_1981_2010norm%02d.tif"%(tileName,mth) for tileName in CCE_TILES_DAYMET]
#        mosaic(fnames,"".join([pathOut,"/tmin_normal_%02d.tif"%mth]))
#        
#        fnames = ["%d_tmax_1981_2010norm%02d.tif"%(tileName,mth) for tileName in CCE_TILES_DAYMET]
#        mosaic(fnames,"".join([pathOut,"/tmax_normal_%02d.tif"%mth]))
    
#    ################################################
#    #5.) Mosaic 1981-2010 Trends for applicable TWX and Daymet Tiles
#    ################################################
#    #TWX
#    trends_path = '/projects/daymet2/cce_case_study/topowx_files/trends/'
#    #normals_path = '/projects/daymet2/compare/topowx_files/trends/no_lst/'
#    #normals_path = '/projects/daymet2/compare/topowx_files/trends/no_homog/'
##    normals_path = '/projects/daymet2/compare/topowx_files/trends/no_homog_lst/'
##
##
#    os.chdir(trends_path)
#    files = np.array(os.listdir(trends_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmin') != -1,
#                                 np.char.endswith(files, '19812010trend.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_topowx_tmin19812010trend.tif")
#    
#    files = np.array(os.listdir(trends_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmax') != -1,
#                                 np.char.endswith(files, '19812010trend.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_topowx_tmax19812010trend.tif")
#
#    #Daymet
#    normals_path = '/projects/daymet2/compare/daymet_files/trends/'  
#    os.chdir(normals_path)
#    files = np.array(os.listdir(normals_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmin') != -1,
#                                 np.char.endswith(files, '1981_2010trend.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_daymet_tmin19812010trend.tif")
#    
#    files = np.array(os.listdir(normals_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmax') != -1,
#                                 np.char.endswith(files, '1981_2010trend.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_daymet_tmax19812010trend.tif")
#    
#    ################################################
#    #6.) Mosaic 1948-2012 Trends for applicable TWX Tiles
#    ################################################
#    #TWX
#    trends_path = '/projects/daymet2/cce_case_study/topowx_files/trends/'
#    #normals_path = '/projects/daymet2/compare/topowx_files/trends/no_lst/'
#    #normals_path = '/projects/daymet2/compare/topowx_files/trends/no_homog/'
##    normals_path = '/projects/daymet2/compare/topowx_files/trends/no_homog_lst/'
##
#    os.chdir(trends_path)
#    files = np.array(os.listdir(trends_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmin') != -1,
#                                 np.char.endswith(files, '19482012trend.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_topowx_tmin19482012trend.tif")
#    
#    files = np.array(os.listdir(trends_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmax') != -1,
#                                 np.char.endswith(files, '19482012trend.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_topowx_tmax19482012trend.tif")
#    
#    ################################################
#    #7.) Resample, mask and crop 1981-2010 normal mosaics to CCE domain
#    ################################################
#    #TWX

#    normals_path = '/projects/daymet2/compare/topowx_files/normals/'
    #normals_path = '/projects/daymet2/compare/topowx_files/normals/no_lst/'
    #normals_path = '/projects/daymet2/compare/topowx_files/normals/no_homog/'
#    normals_path = '/projects/daymet2/compare/topowx_files/normals/no_homog_lst/'
#    resample_to_grd1(PATH_CCE_MASK, 
#                     "".join([normals_path,'mosaic_topowx_tmin19812010norm.tif']), 
#                     "".join([normals_path,'cce_temp_mosaic_topowx_tmin19812010norm.tif']),
#                     gdalconst.GRA_NearestNeighbour)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     "".join([normals_path,'cce_temp_mosaic_topowx_tmin19812010norm.tif']), 
#                     "".join([normals_path,'cce_topowx_tmin19812010norm.tif']))
#    os.system("".join(['rm ',normals_path,'cce_temp_mosaic_topowx_tmin19812010norm.tif']))
#
#    resample_to_grd1(PATH_CCE_MASK, 
#                     "".join([normals_path,'mosaic_topowx_tmax19812010norm.tif']), 
#                     "".join([normals_path,'cce_temp_mosaic_topowx_tmax19812010norm.tif']),
#                     gdalconst.GRA_NearestNeighbour)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     "".join([normals_path,'cce_temp_mosaic_topowx_tmax19812010norm.tif']), 
#                     "".join([normals_path,'cce_topowx_tmax19812010norm.tif']))
#    os.system("".join(['rm ',normals_path,'cce_temp_mosaic_topowx_tmax19812010norm.tif']))
#    
#    #Daymet
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/daymet_files/normals/mosaic_daymet_tmin19812010norm.tif', 
#                     '/projects/daymet2/compare/daymet_files/normals/cce_temp_mosaic_daymet_tmin19812010norm.tif',
#                     gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/daymet_files/normals/cce_temp_mosaic_daymet_tmin19812010norm.tif', 
#                     '/projects/daymet2/compare/daymet_files/normals/cce_daymet_tmin19812010norm.tif')
#    os.system('rm /projects/daymet2/compare/daymet_files/normals/cce_temp_mosaic_daymet_tmin19812010norm.tif')
#    
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/daymet_files/normals/mosaic_daymet_tmax19812010norm.tif', 
#                     '/projects/daymet2/compare/daymet_files/normals/cce_temp_mosaic_daymet_tmax19812010norm.tif',
#                     gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/daymet_files/normals/cce_temp_mosaic_daymet_tmax19812010norm.tif', 
#                     '/projects/daymet2/compare/daymet_files/normals/cce_daymet_tmax19812010norm.tif')
#    os.system('rm /projects/daymet2/compare/daymet_files/normals/cce_temp_mosaic_daymet_tmax19812010norm.tif')


    #Daymet Monthly
#    mosaics_path = '/projects/daymet2/cce_case_study/daymet_files/normals_mthly_mosaics/'
#
#    for mth in np.arange(1,13):
#        
#        #########################
#        resample_to_grd1(PATH_CCE_MASK, 
#                         "".join([mosaics_path,"tmin_normal_%02d.tif"%mth]), 
#                         "".join([mosaics_path,"temp.tif"]),
#                         gdalconst.GRA_Bilinear)
#        mask_to_rastmask(PATH_CCE_MASK, 
#                         "".join([mosaics_path,'temp.tif']), 
#                         "".join([mosaics_path,"cce_tmin_normal_%02d.tif"%mth]))
#        
#        ##########################
#        resample_to_grd1(PATH_CCE_MASK, 
#                         "".join([mosaics_path,"tmax_normal_%02d.tif"%mth]), 
#                         "".join([mosaics_path,"temp.tif"]),
#                         gdalconst.GRA_Bilinear)
#        mask_to_rastmask(PATH_CCE_MASK, 
#                         "".join([mosaics_path,'temp.tif']), 
#                         "".join([mosaics_path,"cce_tmax_normal_%02d.tif"%mth]))
#
#    os.system("".join(['rm ',mosaics_path,'temp.tif']))

#    #PRISM
#    mosaics_path = '/projects/daymet2/prism/new_norms/'
#    path_out = '/projects/daymet2/cce_case_study/prism_files/normals_mthly/'
##PRISM_tmin_30yr_normal_800mM2_09_bil.bil
#    for mth in np.arange(1,13):
#        
#        #########################
#        resample_to_grd1(PATH_CCE_MASK, 
#                         "".join([mosaics_path,"PRISM_tmin_30yr_normal_800mM2_%02d_bil.bil"%mth]), 
#                         "".join([path_out,"temp.tif"]),
#                         gdalconst.GRA_NearestNeighbour)
#        mask_to_rastmask(PATH_CCE_MASK, 
#                         "".join([path_out,'temp.tif']), 
#                         "".join([path_out,"cce_tmin_normal_%02d.tif"%mth]))
#        
#        ##########################
#        resample_to_grd1(PATH_CCE_MASK, 
#                         "".join([mosaics_path,"PRISM_tmax_30yr_normal_800mM2_%02d_bil.bil"%mth]),  
#                         "".join([path_out,"temp.tif"]),
#                         gdalconst.GRA_NearestNeighbour)
#        mask_to_rastmask(PATH_CCE_MASK, 
#                         "".join([path_out,'temp.tif']), 
#                         "".join([path_out,"cce_tmax_normal_%02d.tif"%mth]))
#
#    os.system("".join(['rm ',path_out,'temp.tif']))
    
#    
#
#    ################################################
#    #8.) Resample, mask and crop 1981-2010 trend mosaics to CCE domain
#    ################################################
#    #TWX
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/mosaic_topowx_tmin19812010trend.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmin19812010trend.tif',
#                     gdalconst.GRA_NearestNeighbour)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmin19812010trend.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmin19812010trend.tif')
#    os.system('rm /projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmin19812010trend.tif')
#
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/mosaic_topowx_tmax19812010trend.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmax19812010trend.tif',
#                     gdalconst.GRA_NearestNeighbour)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmax19812010trend.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmax19812010trend.tif')
#    os.system('rm /projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmax19812010trend.tif')  
#    
#    #Daymet
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/daymet_files/trends/mosaic_daymet_tmin19812010trend.tif', 
#                     '/projects/daymet2/compare/daymet_files/trends/cce_temp_mosaic_daymet_tmin19812010trend.tif',
#                     gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/daymet_files/trends/cce_temp_mosaic_daymet_tmin19812010trend.tif', 
#                     '/projects/daymet2/compare/daymet_files/trends/cce_daymet_tmin19812010trend.tif')
#    os.system('rm /projects/daymet2/compare/daymet_files/trends/cce_temp_mosaic_daymet_tmin19812010trend.tif')
#    
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/daymet_files/trends/mosaic_daymet_tmax19812010trend.tif', 
#                     '/projects/daymet2/compare/daymet_files/trends/cce_temp_mosaic_daymet_tmax19812010trend.tif',
#                     gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/daymet_files/trends/cce_temp_mosaic_daymet_tmax19812010trend.tif', 
#                     '/projects/daymet2/compare/daymet_files/trends/cce_daymet_tmax19812010trend.tif')
#    os.system('rm /projects/daymet2/compare/daymet_files/trends/cce_temp_mosaic_daymet_tmax19812010trend.tif')
#    
#    #PRISM
#    resample_prismwgs72_to_wgs84(PATH_CCE_MASK, 
#                                 '/projects/daymet2/compare/prism_files/trends/prism4km_tmax_trend1981-2010.tif', 
#                                 '/projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmax_trend1981-2010.tif',
#                                 gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmax_trend1981-2010.tif', 
#                     '/projects/daymet2/compare/prism_files/trends/cce_prism4km_tmax_trend1981-2010.tif')
#    os.system('rm /projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmax_trend1981-2010.tif')
#    
#    resample_prismwgs72_to_wgs84(PATH_CCE_MASK, 
#                             '/projects/daymet2/compare/prism_files/trends/prism4km_tmin_trend1981-2010.tif', 
#                             '/projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmin_trend1981-2010.tif',
#                             gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmin_trend1981-2010.tif', 
#                     '/projects/daymet2/compare/prism_files/trends/cce_prism4km_tmin_trend1981-2010.tif')
#    os.system('rm /projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmin_trend1981-2010.tif')
#
#    ################################################
#    #9.) Resample, mask and crop 1948-2012 trend mosaics to CCE domain
#    ################################################
#    #TWX
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/mosaic_topowx_tmin19482012trend.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmin19482012trend.tif',
#                     gdalconst.GRA_NearestNeighbour)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmin19482012trend.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmin19482012trend.tif')
#    os.system('rm /projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmin19482012trend.tif')
#    
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/mosaic_topowx_tmax19482012trend.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmax19482012trend.tif',
#                     gdalconst.GRA_NearestNeighbour)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmax19482012trend.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/trends/cce_topowx_tmax19482012trend.tif')
#    os.system('rm /projects/daymet2/cce_case_study/topowx_files/trends/cce_temp_mosaic_topowx_tmax19482012trend.tif') 
#
#    #PRISM
#    resample_prismwgs72_to_wgs84(PATH_CCE_MASK, 
#                                 '/projects/daymet2/compare/prism_files/trends/prism4km_tmax_trend1948-2012.tif', 
#                                 '/projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmax_trend1948-2012.tif',
#                                 gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmax_trend1948-2012.tif', 
#                     '/projects/daymet2/compare/prism_files/trends/cce_prism4km_tmax_trend1948-2012.tif')
#    os.system('rm /projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmax_trend1948-2012.tif')
#    
#    resample_prismwgs72_to_wgs84(PATH_CCE_MASK, 
#                             '/projects/daymet2/compare/prism_files/trends/prism4km_tmin_trend1948-2012.tif', 
#                             '/projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmin_trend1948-2012.tif',
#                             gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmin_trend1948-2012.tif', 
#                     '/projects/daymet2/compare/prism_files/trends/cce_prism4km_tmin_trend1948-2012.tif')
#    os.system('rm /projects/daymet2/compare/prism_files/trends/cce_temp_prism4km_tmin_trend1948-2012.tif')
    
#    ################################################
#    #9.) Resample, mask and crop predictors to CCE domain
#    ################################################

#    path_predictor = '/projects/daymet2/dem/interp_grids/conus/tifs/crop_lst_tmin08.tif'
#    path_out = '/projects/daymet2/cce_case_study/predictors/'
#    name = "tmin08"
#
#    #TWX
#    resample_to_grd1(PATH_CCE_MASK,path_predictor,"".join([path_out,"temp_cce_",name,".tif"]),
#                     gdalconst.GRA_NearestNeighbour)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     "".join([path_out,"temp_cce_",name,".tif"]),
#                     "".join([path_out,"cce_",name,".tif"]))
#    os.system("".join(['rm ',path_out,"temp_cce_",name,".tif"]))     

#######################################################################################3

#    def aStatFunc(aDs):
#        
#        a = aDs.variables['tmin_se'][:,:]
#        a = np.ma.masked_equal(a,aDs.variables['tmin_se']._FillValue)
#        a.fill_value = aDs.variables['tmin_se']._FillValue
#        return a
#
#    outCceStat('/stage/climate/topowx_tile_output/','tmin',
#               CCE_TILES_TWX, aStatFunc, 
#               '/projects/daymet2/docs/UsgsConfCall201309/cce_output/tmin_se/')



################################################################################################
#Annual Datasets
################################################################################################
    #TWX
#    tilepath = '/stage/climate/topowx_tile_output/'
#    outpath = '/projects/daymet2/cce_case_study/topowx_files/annual/'
#
#    for atile in CCE_TILES_TWX:
#        for atair in ['tmin','tmax']:
#            ds_path = "".join([tilepath,atile,"/",atile,"_",atair,".nc"])            
#            tairAnn = getAnn(ds_path,atair, 1948,2012)
#            print tairAnn.shape
#            outFpath =  "".join([outpath,atile,"_",atair,"_19482012ann.tif"])
#            print outFpath
#            topoWxTileToTiff(ds_path, tairAnn,outFpath)
#    
#    #Daymet     
#    tilepath = "/projects/daymet2/daymet_oakridge/multi_yr_tiles/"
#    outpath = "/projects/daymet2/cce_case_study/daymet_files/annual/"
#    for tname in CCE_TILES_DAYMET:
#        for vname in ['tmin','tmax']:
#            dsPath = "".join([tilepath,str(tname),"_",vname,"_1980_2011.nc"])
#            tairAnn = getAnn(dsPath,vname, 1980,2011)
#            outFpath = "".join([outpath,str(tname),"_",vname,"_1980_2011ann.tif"])
#            print outFpath
#            daymetTileToTiff(dsPath, tairAnn, outFpath)
#    
#    #PRISM
#    dsPathTmax = '/projects/daymet2/prism/4km_annual/tmax/prism4km_tmax_ann1948-2012.nc'
#    tairTrend = getAnnTrend(dsPathTmax, 'tmax', 1981, 2010)
#    prism4kmToTiff(dsPathTmax, tairTrend, 
#                   '/projects/daymet2/compare/prism_files/trends/prism4km_tmax_trend1981-2010.tif')
#    dsPathTmin = '/projects/daymet2/prism/4km_annual/tmin/prism4km_tmin_ann1948-2012.nc'
#    tairTrend = getAnnTrend(dsPathTmin, 'tmin', 1981, 2010)
#    prism4kmToTiff(dsPathTmin, tairTrend, 
#                   '/projects/daymet2/compare/prism_files/trends/prism4km_tmin_trend1981-2010.tif')

#    ################################################
#    #5.) Mosaic Annual Datasets
#    ################################################
#    #TWX
#    anns_path = '/projects/daymet2/cce_case_study/topowx_files/annual/'
#
#    os.chdir(anns_path)
#    files = np.array(os.listdir(anns_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmin') != -1,
#                                 np.char.endswith(files, '19482012ann.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_topowx_tmin19482012ann.tif")
    
#    files = np.array(os.listdir(anns_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmax') != -1,
#                                 np.char.endswith(files, '19482012ann.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_topowx_tmax19482012ann.tif")
##
##    #Daymet
#    anns_path = '/projects/daymet2/cce_case_study/daymet_files/annual/'  
#    os.chdir(anns_path)
#    files = np.array(os.listdir(anns_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmin') != -1,
#                                 np.char.endswith(files, '1980_2011ann.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_daymet_tmin19802011ann.tif")
#    
#    files = np.array(os.listdir(anns_path))
#    files = files[np.logical_and(np.logical_and(np.char.find(files,'tmax') != -1,
#                                 np.char.endswith(files, '1980_2011ann.tif')),
#                                 ~np.char.startswith(files, 'mosaic'))]
#    mosaic(files,"mosaic_daymet_tmax19802011ann.tif")
    
    
    #PRISM
#    dsPath = '/projects/daymet2/prism/4km_annual/tmin/prism4km_tmin_ann1948-2012.nc'
#    ds = Dataset(dsPath)
#    a = ds.variables['tmin'][:]
#    prism4kmToTiff(dsPath, a, '/projects/daymet2/cce_case_study/prism_files/annual/prism4km_tmin_ann1948-2012.tif')
    
#    dsPath = '/projects/daymet2/prism/4km_annual/tmax/prism4km_tmax_ann1948-2012.nc'
#    ds = Dataset(dsPath)
#    a = ds.variables['tmax'][:]
#    prism4kmToTiff(dsPath, a, '/projects/daymet2/cce_case_study/prism_files/annual/prism4km_tmax_ann1948-2012.tif')
    
    #RESAMPLE
    #TWX
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/annual/mosaic_topowx_tmin19482012ann.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/annual/cce_temp_mosaic_topowx_tmin19482012ann.tif',
#                     gdalconst.GRA_NearestNeighbour)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/annual/cce_temp_mosaic_topowx_tmin19482012ann.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/annual/cce_topowx_tmin19482012ann.tif')
#    os.system('rm /projects/daymet2/cce_case_study/topowx_files/annual/cce_temp_mosaic_topowx_tmin19482012ann.tif')

#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/annual/mosaic_topowx_tmax19482012ann.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/annual/cce_temp_mosaic_topowx_tmax19482012ann.tif',
#                     gdalconst.GRA_NearestNeighbour)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/topowx_files/annual/cce_temp_mosaic_topowx_tmax19482012ann.tif', 
#                     '/projects/daymet2/cce_case_study/topowx_files/annual/cce_topowx_tmax19482012ann.tif')
#    os.system('rm /projects/daymet2/cce_case_study/topowx_files/annual/cce_temp_mosaic_topowx_tmax19482012ann.tif')

    #Daymet
    
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/daymet_files/annual/mosaic_daymet_tmin19802011ann.tif', 
#                     '/projects/daymet2/cce_case_study/daymet_files/annual/cce_temp_mosaic_daymet_tmin19802011ann.tif',
#                     gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/daymet_files/annual/cce_temp_mosaic_daymet_tmin19802011ann.tif', 
#                     '/projects/daymet2/cce_case_study/daymet_files/annual/cce_mosaic_daymet_tmin19802011ann.tif')
#    os.system('rm /projects/daymet2/cce_case_study/daymet_files/annual/cce_temp_mosaic_daymet_tmin19802011ann.tif')
    ####################################################
#    resample_to_grd1(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/daymet_files/annual/mosaic_daymet_tmax19802011ann.tif', 
#                     '/projects/daymet2/cce_case_study/daymet_files/annual/cce_temp_mosaic_daymet_tmax19802011ann.tif',
#                     gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/daymet_files/annual/cce_temp_mosaic_daymet_tmax19802011ann.tif', 
#                     '/projects/daymet2/cce_case_study/daymet_files/annual/cce_mosaic_daymet_tmax19802011ann.tif')
#    os.system('rm /projects/daymet2/cce_case_study/daymet_files/annual/cce_temp_mosaic_daymet_tmax19802011ann.tif')


    #PRISM
#    resample_prismwgs72_to_wgs84(PATH_CCE_MASK, 
#                                 '/projects/daymet2/cce_case_study/prism_files/annual/prism4km_tmin_ann1948-2012.tif', 
#                                 '/projects/daymet2/cce_case_study/prism_files/annual/cce_temp_prism4km_tmin_ann1948-2012.tif',
#                                 gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/prism_files/annual/cce_temp_prism4km_tmin_ann1948-2012.tif', 
#                     '/projects/daymet2/cce_case_study/prism_files/annual/cce_prism4km_tmin_ann1948-2012.tif')
#    os.system('rm /projects/daymet2/cce_case_study/prism_files/annual/cce_temp_prism4km_tmin_ann1948-2012.tif')
    
#    resample_prismwgs72_to_wgs84(PATH_CCE_MASK, 
#                                 '/projects/daymet2/cce_case_study/prism_files/annual/prism4km_tmax_ann1948-2012.tif', 
#                                 '/projects/daymet2/cce_case_study/prism_files/annual/cce_temp_prism4km_tmax_ann1948-2012.tif',
#                                 gdalconst.GRA_Bilinear)
#    mask_to_rastmask(PATH_CCE_MASK, 
#                     '/projects/daymet2/cce_case_study/prism_files/annual/cce_temp_prism4km_tmax_ann1948-2012.tif', 
#                     '/projects/daymet2/cce_case_study/prism_files/annual/cce_prism4km_tmax_ann1948-2012.tif')
#    os.system('rm /projects/daymet2/cce_case_study/prism_files/annual/cce_temp_prism4km_tmax_ann1948-2012.tif')


    ########################################################################################################
    #Station Compare
    ########################################################################################################
    dRast = MultiDaymetTileRaster('/projects/daymet2/daymet_oakridge/multi_yr_tiles/','tmin')
    plt.plot(dRast.getTimeSeries(-113.8394,47.9203))
    plt.show()
    
#    pRast = PrismTileRaster('/projects/daymet2/prism/4km_daily/netcdf/prism4km_tmax.nc', 'tmax')
#    plt.plot(pRast.getTimeSeries(-113.82,47.69))
#    plt.show()
    
#    tRast = MultiTwxTileRaster('/stage/climate/topowx_tile_output/', 'tmin')
#    plt.plot(tRast.getTimeSeries(-113.82,47.69))
#    plt.show()
    
    #GHCN_USC00248087 9998.04572334