'''
Created on Nov 9, 2012

@author: jared.oyler
'''
import os
import numpy as np
from netCDF4 import Dataset,num2date,date2num
import twx.utils.util_dates as utld
from twx.utils.util_dates import YEAR
from twx.db.ushcn import TairAggregate
import matplotlib.pyplot as plt
import osgeo.gdal as gdal
import osgeo.gdalconst as gdalconst
import osgeo.osr as osr

RPATH_DAYMET = 'http://daymet.ornl.gov/thredds/fileServer/allcf/'
YEARS = np.arange(1980,2012)
#MONTANA_TILES = np.concatenate((np.arange(12452,12459),
#                                np.arange(12272,12279),
#                                np.arange(12093,12099)))

MONTANA_TILES = np.array([12452,
                        12453,
                        12454,
                        12274,
                        12273,
                        12272,
                        12094,
                        12093])

OUT_PATH = "/projects/daymet2/daymet_oakridge/orig_tiles/"
NCDF_CHK_COLS = 50

class TileInfo():
    
    def __init__(self,tile_path,tile_num,yrs,tair_var):
        self.tile_path = tile_path
        self.tile_num = tile_num
        self.yrs = yrs
        self.tair_var = tair_var

        self.dates = None
        self.yr_dates = None
        self.ymd = None

class DaymetDataset(Dataset):

    def create_proj_var(self):
        print "creating projection variable..."
        proj_var = self.createVariable('lambert_conformal_conic','f4')
        proj_var.grid_mapping_name = "lambert_conformal_conic"
        proj_var.longitude_of_central_meridian = -100.
        proj_var.latitude_of_projection_origin = 42.5
        proj_var.false_easting = 0.
        proj_var.false_northing = 0.
        proj_var.standard_parallel = np.array([25., 60.])
        self.sync()
        print "done creating projection variable."

    def set_tmin_var(self,tile_info):
        
        tmin_var = self.createVariable('tmin','f4',('time','y','x'),fill_value=-9999.)#,zlib=True,chunksizes=(tile_info.ymd.size,NCDF_CHK_COLS,NCDF_CHK_COLS))
        
        tmin_var.long_name = "daily minimum temperature" ;
        tmin_var.units = "degrees C" ;
        tmin_var.missing_value = -9999.
        tmin_var.valid_range = np.array([-50., 50.])
        tmin_var.coordinates = "lat lon"
        tmin_var.grid_mapping = "lambert_conformal_conic" ;
        tmin_var.cell_methods = "area: mean time: minimum" ;
        
        print "setting tmin data..."
        
        for x in np.arange(tile_info.yrs.size):
        
            yr = tile_info.yrs[x]
            
            ds_in = Dataset("".join([tile_info.tile_path,str(tile_info.tile_num),"_",str(yr),"_",tile_info.tair_var,".nc"]))
            tmin_in = ds_in.variables['tmin'][:]
            yr_ymd = utld.get_ymd_array(tile_info.yr_dates[x])            
            yr_mask = np.in1d(tile_info.ymd, yr_ymd, True)

            tmin_var[yr_mask,:,:] = tmin_in
            
            self.sync()
            
            print yr
        
        print "done setting tmin data."
        
    def set_tmax_var(self,tile_info):
        
        tmax_var = self.createVariable('tmax','f4',('time','y','x'),fill_value=-9999.)#,zlib=True,chunksizes=(tile_info.ymd.size,NCDF_CHK_COLS,NCDF_CHK_COLS))
        
        tmax_var.long_name = "daily maximum temperature" ;
        tmax_var.units = "degrees C" ;
        tmax_var.missing_value = -9999.
        tmax_var.valid_range = np.array([-50., 50.])
        tmax_var.coordinates = "lat lon"
        tmax_var.grid_mapping = "lambert_conformal_conic" ;
        tmax_var.cell_methods = "area: mean time: maximum" ;
        
        print "setting tmax data..."
        
        for x in np.arange(tile_info.yrs.size):
        
            yr = tile_info.yrs[x]
            
            ds_in = Dataset("".join([tile_info.tile_path,str(tile_info.tile_num),"_",str(yr),"_",tile_info.tair_var,".nc"]))
            tmax_in = ds_in.variables['tmax'][:]
            yr_ymd = utld.get_ymd_array(tile_info.yr_dates[x])            
            yr_mask = np.in1d(tile_info.ymd, yr_ymd, True)

            tmax_var[yr_mask,:,:] = tmax_in
            
            self.sync()
            
            print yr
        
        print "done setting tmax data."

    def create_time_dimvars(self,tile_info):
    
        dates = None
        
        yr_dates = []
        
        print "setting time data..."
        
        for yr in tile_info.yrs:
            
            ds_in = Dataset("".join([tile_info.tile_path,str(tile_info.tile_num),"_",str(yr),"_",tile_info.tair_var,".nc"]))
            yr_dates.append(num2date(ds_in.variables['time'][:],units=ds_in.variables['time'].units))
            #yr_dates = ds_in.variables['time'][:]
            if dates is None:
                dates = yr_dates
            else:
                dates = np.concatenate((dates,yr_dates))
    
            ds_in.close()
            
            print yr
        
        dates = np.concatenate(yr_dates)
        
        self.createDimension('time',dates.size)
        time_var = self.createVariable('time','f8',('time',))
        time_var.long_name = "time"
        time_var.calendar = "standard"
        time_var.units = "days since 1980-01-01 00:00:00 UTC"
        time_var[:] = date2num(dates,time_var.units)
        #time_var[:] = dates
        #time_var.bounds = "time_bnds"
        tile_info.dates = dates
        tile_info.yr_dates = yr_dates
        tile_info.ymd = utld.get_ymd_array(dates)
        
        self.sync()
        
        print "done setting time data..."

    def create_spatial_dimvars(self,tile_info):
        
        print "creating spatial dims and variables..."
        
        ds_in = Dataset("".join([tile_info.tile_path,str(tile_info.tile_num),"_",str(tile_info.yrs[0]),"_",tile_info.tair_var,".nc"]))
        
        x = ds_in.variables['x'][:]
        y = ds_in.variables['y'][:]
        lon = ds_in.variables['lon'][:]
        lat = ds_in.variables['lat'][:]
        
        self.createDimension('x',x.size)
        self.createDimension('y',y.size)
        
        x_var = self.createVariable('x','f8',('x',))
        x_var.units = "m"
        x_var.long_name = "x coordinate of projection"
        x_var.standard_name = "projection_x_coordinate"
        x_var[:] = x 
    
        y_var = self.createVariable('y','f8',('y',))
        y_var.units = "m"
        y_var.long_name = "y coordinate of projection"
        y_var.standard_name = "projection_y_coordinate"
        y_var[:] = y
        
        lon_var = self.createVariable('lon','f8',('y','x'))
        lon_var.units = "degrees_east"
        lon_var.long_name = "longitude coordinate"
        lon_var.standard_name = "longitude"
        lon_var[:] = lon
        
        lat_var = self.createVariable('lat','f8',('y','x'))
        lat_var.units = "degrees_north"
        lat_var.long_name = "latitude coordinate"
        lat_var.standard_name = "latitude"
        lat_var[:] = lat
        
        self.sync()
        
        print "done creating spatial dims and variables..."

def build_multi_yr_tiles(tiles=MONTANA_TILES):
    PATH_MULTIYR = "/projects/daymet2/daymet_oakridge/multi_yr_tiles/"
    
    for tile in tiles:
        
        for tair_var in ['tmin','tmax']:
            
            
            ds_path = "".join([PATH_MULTIYR,str(tile),"_",tair_var,"_",str(YEARS[0]),"_",str(YEARS[-1]),".nc"])
            print "saving multiyear dataset: "+ds_path
            ds = DaymetDataset(ds_path,'w')
            tinfo = TileInfo("".join([OUT_PATH,str(tile),"/"]), tile, YEARS, tair_var)

            ds.create_proj_var()
            ds.create_spatial_dimvars(tinfo)
            ds.create_time_dimvars(tinfo)
            
            if tair_var == 'tmin':
                ds.set_tmin_var(tinfo)
            else:
                ds.set_tmax_var(tinfo)
                
            ds.close()


class NcdfRaster():
    
    def __init__(self,fname,coords='latlon'):
        #Coords are latlon or xy
        self.ds = Dataset(fname,"r")
        
        if coords == 'latlon':
            
            self.x = self.ds.variables[VAR_LON][:]
            #make sure no lons > 180
            self.x[self.x > 180] = self.x[self.x > 180]-360.0
            self.y = self.ds.variables[VAR_LAT][:]
        
        elif coords== 'xy':
            
            self.x = self.ds.variables['x']
            self.y = self.ds.variables['y']
            
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
        self.geoTransform[1] = np.abs(self.x [0] - self.x[1])
        self.geoTransform[2],self.geoTransform[4] = (0.0,0.0)
        self.geoTransform[0] = self.x[0] - (self.geoTransform[1]/2.0) 
        self.geoTransform[3] = self.y[0] + np.abs(self.geoTransform[5]/2.0)
                
        self.min_x = self.geoT[0]
        self.max_x = self.min_x + (self.gdalDs.RasterXSize*self.geoT[1])
        self.max_y =  self.geoT[3]
        self.min_y =  self.max_y - (-self.gdalDs.RasterYSize*self.geoT[5])
        
    def toGTiff(self,fpathOut,a,proj4='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'):
        
        drvr =  gdal.GetDriverByName('GTiff')
        dsOut = drvr.Create(fpathOut, a.shape[1], a.shape[0], 1, DTYPES_MAP[a.dtype])
        
        sr = osr.SpatialReference()
        sr.ImportFromProj4(proj4)
        dsOut.SetProjection(sr.ExportToWkt())
        dsOut.SetGeoTransform(self.geoTransform)
        
        bandOut = dsOut.GetRasterBand(1)
        
        if np.ma.isMA(a):
            bandOut.Fill(a.fill_value)
            bandOut.SetNoDataValue(a.fill_value)
            a = np.ma.filled(a)
    
        bandOut.WriteArray(np.ma.filled(a),0,0) 
        bandOut.FlushCache()
        
    def getGridCellOffset(self,x,y):
        
        if not self.is_inbounds(x, y):
            raise Exception("Lon/Lat outside raster extent")
        
        originX = self.geoTransform[0]
        originY = self.geoTransform[3]
        pixelWidth = self.geoTransform[1]
        pixelHeight = self.geoTransform[5]
        
        xOffset = abs(int((x - originX) / pixelWidth))
        yOffset = abs(int((y - originY) / pixelHeight))
        return xOffset,yOffset
    
    def is_inbounds(self,x,y):
        return x >= self.min_x and x <= self.max_x and y >= self.min_y and y <= self.max_y

def download_tiles(out_path,tiles,yrs):

    for tile in tiles:
        
        out_path = "".join([out_path,str(tile)])
        
        try:
        
            os.chdir(out_path)
        
        except OSError:
            #path doesn't exist
            os.mkdir(out_path)
            os.chdir(out_path)
        
        for yr in yrs:
            
            os.system("".join(["wget ",RPATH_DAYMET,str(yr),"/",str(tile),"_",str(yr),"/tmin.nc -O ",str(tile),"_",str(yr),"_tmin.nc"]))
            os.system("".join(["wget ",RPATH_DAYMET,str(yr),"/",str(tile),"_",str(yr),"/tmax.nc -O ",str(tile),"_",str(yr),"_tmax.nc"]))

def getNorm1981_2010(dsPath,varname):
    
    ds = Dataset(dsPath)
    days = utld.get_days_metadata_dates(num2date(ds.variables['time'][:],
                                                 units=ds.variables['time'].units,calendar='standard'))
    dayMask = np.nonzero(np.logical_and(days[YEAR] >= 1981,days[YEAR] <= 2010))[0]
    days = days[dayMask]
    agg = TairAggregate(days)
    tair = ds.variables[varname][:]
    tairAnn = agg.dailyToAnn(tair)
    tairNorm = np.ma.mean(tairAnn,axis=0)
    
    ds.close()
    
    return tairNorm


def daymetTileToTiff(dsPath,a,pathOut):
    
    ds = Dataset(dsPath)
    
    nrow = len(ds.dimensions['y'])
    ncol = len(ds.dimensions['x'])
            
    driver = gdal.GetDriverByName("GTiff")
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

    band = raster.GetRasterBand(1)
    band.SetNoDataValue(a.fill_value)
    band.WriteArray(np.ma.filled(a),0,0) 
    band.FlushCache()
    
    ds.close()

'''
Mosaic Normals
gdalwarp -dstnodata -9999 12093_tmin_1981_2010norm.tif 12094_tmin_1981_2010norm.tif 12272_tmin_1981_2010norm.tif 12273_tmin_1981_2010norm.tif 12274_tmin_1981_2010norm.tif 12452_tmin_1981_2010norm.tif 12453_tmin_1981_2010norm.tif 12454_tmin_1981_2010norm.tif 12455_tmin_1981_2010norm.tif 12275_tmin_1981_2010norm.tif mosaic_tmin_1981_2010norm.tif
gdalwarp -dstnodata -9999 12093_tmax_1981_2010norm.tif 12094_tmax_1981_2010norm.tif 12272_tmax_1981_2010norm.tif 12273_tmax_1981_2010norm.tif 12274_tmax_1981_2010norm.tif 12452_tmax_1981_2010norm.tif 12453_tmax_1981_2010norm.tif 12454_tmax_1981_2010norm.tif 12455_tmax_1981_2010norm.tif 12275_tmax_1981_2010norm.tif mosaic_tmax_1981_2010norm.tif
'''
def outputAllNorms():
    
    pathMultiYr = "/projects/daymet2/daymet_oakridge/multi_yr_tiles/"
    outPath = "/projects/daymet2/daymet_oakridge/norms/"
    
    for tname in [12455,12275]:#MONTANA_TILES:
        
        for vname in ['tmin','tmax']:
            
            dsPath = "".join([pathMultiYr,str(tname),"_",vname,"_1980_2011.nc"])
            tairNorm = getNorm1981_2010(dsPath, vname)
            tairNorm.fill_value = -9999.
            outFpath = "".join([outPath,str(tname),"_",vname,"_1981_2010norm.tif"])
            print outFpath
            daymetTileToTiff(dsPath, tairNorm, outFpath)
            
  
if __name__ == '__main__':
    
    outputAllNorms()
    
    #build_multi_yr_tiles(tiles=[12455,12275])
    #download_tiles(OUT_PATH, MONTANA_TILES, YEARS)
