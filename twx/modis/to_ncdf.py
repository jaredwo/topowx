'''
Created on Jan 27, 2012

@author: jared.oyler
'''
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import osgeo.osr as osr
from netCDF4 import Dataset
import netCDF4
import sys,os
from datetime import datetime,date
import utils.util_dates as utld
from utils.util_dates import YDAY,YEAR
from utils.input_raster import input_raster, RasterDataset
TDAYCOEF = 0.45

'''
1/2 lat * 2/3 lon (MERRA resolution)

/MODIS/Mirror/NTSG_Only/MOD16_NCEP2_1kmALB/MOD16A2.105.NCEP2_1kmALB
'''


UPPER_LEFT_PT = (-8895604.157333,5559752.598333)
LOWER_RIGHT_PT = (-7783653.637667,4447802.078667)
XDIM = 1200
YDIM = 1200
DATAFIELD_ET_1KM = 'ET_1km'
SCALE_FACTOR = 0.1
MAX_VALID_VALUE = 32700
MAX_VALID_VALUE_ANN = 65500
EPSG_WGS84 = 4326 #EPSG Code
EPSG_NAD83 = 4269 #EPSG Code
PROJ4_MODIS = "+proj=sinu +R=6371007.181 +nadgrids=@null +no_defs +wktext"
MOD16_PATH = "/MODIS/Mirror/MOD16/MOD16A2.105_MERRAGMAO/"
MOD16_PATH_ANN = "/MODIS/Mirror/MOD16/MOD16A3.105_MERRAGMAO/"
MOD16_TOPOMET_PATH = "/projects/daymet2/modis/topomet_hdf4/ann_files/MOD16_DAYMET/MOD16A2/"
MOD16_TOPOMET_PATH_ANN = "/projects/daymet2/modis/topomet_hdf4/ann_files/MOD16_DAYMET/MOD16A3/"
MERRA_DATA_PATH = "/projects/daymet2/mod16_aguposter/merra_met/"
TILE = "h10v04"
MODIS_8DAYS = np.array([1, 9, 17, 25, 33, 41, 49, 57,
                       65, 73, 81, 89, 97, 105, 113, 121,
                       129, 137, 145, 153, 161, 169, 177, 185,
                       193, 201, 209, 217, 225, 233, 241, 249,
                       257, 265, 273, 281, 289, 297, 305, 313,
                       321, 329, 337, 345, 353, 361])

TIME_START_INDEX = 18262 #Index to start reading on 20000101
TIME_COUNT = 3653 #The total of days to read (20000101 - 20091231)


class modis_sin_latlon_transform():

    def __init__(self):
        
        self.sr_sin = osr.SpatialReference()
        self.sr_sin.ImportFromProj4(PROJ4_MODIS)
        self.sr_wgs84 = osr.SpatialReference()
        self.sr_wgs84.ImportFromEPSG(EPSG_WGS84)
        self.sr_nad83 = osr.SpatialReference()
        self.sr_nad83.ImportFromEPSG(EPSG_NAD83)
        
        self.trans_sin_to_wgs84 = osr.CoordinateTransformation(self.sr_sin,self.sr_wgs84)
        self.trans_wgs84_to_sin = osr.CoordinateTransformation(self.sr_wgs84,self.sr_sin)
        self.trans_nad83_to_sin = osr.CoordinateTransformation(self.sr_nad83,self.sr_sin)

class modis_et_dataset(object):
    
    def __init__(self,file_path,max_valid_val=MAX_VALID_VALUE):
        
        et_sd = SD(file_path, SDC.READ)
        et_sds = et_sd.select(DATAFIELD_ET_1KM)
        self.et_a = np.array(et_sds[:],dtype=np.float64)
        
        self.et_a = np.ma.masked_greater_equal(self.et_a, max_valid_val)
        self.et_a = self.et_a*SCALE_FACTOR
        
#        plt.imshow(self.et_a)
#        plt.show()
        
        
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
        geotransform[0] = UPPER_LEFT_PT[0]
        geotransform[3] = UPPER_LEFT_PT[1]
        geotransform[2],geotransform[4] = (0.0,0.0)
        geotransform[1] = np.abs((UPPER_LEFT_PT[0]-LOWER_RIGHT_PT[0])/XDIM)
        #n-s pixel height/resolution needs to be negative
        geotransform[5] = -np.abs((UPPER_LEFT_PT[1]-LOWER_RIGHT_PT[1])/YDIM)
        self.geotransform = geotransform
        
        self.trans_coord = modis_sin_latlon_transform()
        
        self.min_x = UPPER_LEFT_PT[0]
        self.max_x = LOWER_RIGHT_PT[0]
        self.max_y = UPPER_LEFT_PT[1]
        self.min_y = LOWER_RIGHT_PT[1]
        self.rows = YDIM
        self.cols = XDIM
        
    def row_col_grids(self):
        
        rcgrid = np.mgrid[0:self.rows,0:self.cols]
        rows = rcgrid[0]
        cols = rcgrid[1]
        
        return rows,cols
    
    def geo_coord_grids(self):
        
        rows,cols = self.row_col_grids()
        return self.geo_location(cols, rows)
    
    def geo_location(self,col,row):
        '''Affine Transfrom: Converts pixel row and column to spatially referenced coordinates (in native projection)'''
        x_geo = (self.geotransform[0] + col*self.geotransform[1] + row*self.geotransform[2]) + self.geotransform[1] / 2.0
        y_geo = (self.geotransform[3] + col*self.geotransform[4] + row*self.geotransform[5]) + self.geotransform[5] / 2.0
        return x_geo,y_geo
    
    def gridcell_offset_lonlat(self,lon,lat):
        '''Returns the grid cell offset for this raster based on the input wgs84/nad83 lon/lat'''
        x_sin,y_sin,z_sin = self.trans_coord.trans_nad83_to_sin.TransformPoint(lon,lat)
        
        if not self.is_inbounds(x_sin, y_sin):
            raise Exception("Lon/Lat outside raster extent")
        
        originX = self.geotransform[0]
        originY = self.geotransform[3]
        pixelWidth = self.geotransform[1]
        pixelHeight = self.geotransform[5]
        
        xOffset = abs(int((x_sin - originX) / pixelWidth))
        yOffset = abs(int((y_sin - originY) / pixelHeight))
        return xOffset,yOffset
    
    def data_value_lonlat(self,lon,lat):
        
        x,y = self.gridcell_offset_lonlat(lon, lat)
        return self.et_a[y,x]
    
    def is_inbounds(self,x_sin,y_sin):
        return x_sin >= self.min_x and x_sin <= self.max_x and y_sin >= self.min_y and y_sin <= self.max_y


class modis_hdf_dataset(modis_et_dataset):
    
    def __init__(self,file_path,ds_name,sf=None):
        
        sd = SD(file_path, SDC.READ)
        sds = sd.select(ds_name)
        self.a = sds[:]
        self.a = np.ma.masked_equal(self.a,sds.getfillvalue())
        self.et_a = self.a
        
        if sf == None:
        
            try:
                
                sf = sds.scale_factor
                self.a = self.a*sf
                self.et_a = self.et_a*sf
                
            except:
                pass # no scale factor
        
        else:
            
            self.a = self.a*sf
            self.et_a = self.et_a*sf
            
        
        
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
        geotransform[0] = UPPER_LEFT_PT[0]
        geotransform[3] = UPPER_LEFT_PT[1]
        geotransform[2],geotransform[4] = (0.0,0.0)
        geotransform[1] = np.abs((UPPER_LEFT_PT[0]-LOWER_RIGHT_PT[0])/XDIM)
        #n-s pixel height/resolution needs to be negative
        geotransform[5] = -np.abs((UPPER_LEFT_PT[1]-LOWER_RIGHT_PT[1])/YDIM)
        self.geotransform = geotransform
        
        self.trans_coord = modis_sin_latlon_transform()
        
        self.min_x = UPPER_LEFT_PT[0]
        self.max_x = LOWER_RIGHT_PT[0]
        self.max_y = UPPER_LEFT_PT[1]
        self.min_y = LOWER_RIGHT_PT[1]
        self.rows = YDIM
        self.cols = XDIM


def init_ncdf_ann(fpath,lons,lats,yr_str,yr_end):

    ncdf_file = Dataset(fpath,'w')
    #Set global attributes
    title = "".join(["MOD16 Evapotranspiration Annual Crown of the Continent (subset of tile h10v04) ",str(yr_str),"-",str(yr_end)])
    ncdf_file.title = title
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.source = "MOD16"
    ncdf_file.history = "".join(["Created on: ",datetime.strftime(date.today(),"%Y-%m-%d")]) 
    ncdf_file.references = "http://www.ntsg.umt.edu/project/mod16"
    ncdf_file.comment = "30-arcsec geographic projection"
    
    yrs = np.arange(yr_str,yr_end+1)
    
    #Create 3-dimensions
    ncdf_file.createDimension('time',yrs.size)
    ncdf_file.createDimension('lat',lats.size)
    ncdf_file.createDimension('lon',lons.size)
    
    #Create dimension variables and fill with values
    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    #times.units = "days since 1950-1-1 0:0:0"
    times.standard_name = "time"
    times[:] = yrs
    
    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    chnk_size = (1,lats.size,lons.size)
    ncdf_var = ncdf_file.createVariable('et','i2',('time','lat','lon'),chunksizes=chnk_size)
    ncdf_var.long_name = "evapotranspiration"
    ncdf_var.units = "kg/m^2/yr"
    ncdf_var.standard_name = "evapotranspiration"
    ncdf_var.missing_value = netCDF4.default_fillvals['i2']
    ncdf_var.scale_factor = np.float32(0.1)
    
    return ncdf_file
    

def init_ncdf(fpath,lons,lats,yr_str,yr_end):

    ncdf_file = Dataset(fpath,'w')
    #Set global attributes
    title = "".join(["MOD16 Evapotranspiration 8-day Composite Crown of the Continent (subset of tile h10v04) ",str(yr_str),"-",str(yr_end)])
    ncdf_file.title = title
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.source = "MOD16"
    ncdf_file.history = "".join(["Created on: ",datetime.strftime(date.today(),"%Y-%m-%d")]) 
    ncdf_file.references = "http://www.ntsg.umt.edu/project/mod16"
    ncdf_file.comment = "30-arcsec geographic projection"
    
    yrs = np.arange(yr_str,yr_end+1)
    days = np.zeros(yrs.size*MODIS_8DAYS.size)
    
    x = 0
    for yr in yrs:
        
        i = 0
        for day in MODIS_8DAYS:
            
            days[i+x] = long("".join([str(yr),'%03d' % day]))
            i+=1
            
        x+=MODIS_8DAYS.size 
    
    #Create 3-dimensions
    ncdf_file.createDimension('time',days.size)
    ncdf_file.createDimension('lat',lats.size)
    ncdf_file.createDimension('lon',lons.size)
    
    #Create dimension variables and fill with values
    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    #times.units = "days since 1950-1-1 0:0:0"
    times.standard_name = "time modis 8-day composite"
    times[:] = days
    
    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    row_chksize = 50
    col_chksize = 50
    chnk_size = (1,lats.size,lons.size)
    ncdf_var = ncdf_file.createVariable('et','i2',('time','lat','lon'),chunksizes=chnk_size)
    ncdf_var.long_name = "evapotranspiration"
    ncdf_var.units = "kg/m2/8day"
    ncdf_var.standard_name = "evapotranspiration"
    ncdf_var.missing_value = netCDF4.default_fillvals['i2']
    ncdf_var.scale_factor = np.float32(0.1)
    
    return ncdf_file

def create_ncdf_topomet_crown_8day():
    
    ds_tm = Dataset('/projects/daymet2/modis/crown_topomet.nc')
    tmin = ds_tm.variables['tmin'][:]
    tmax = ds_tm.variables['tmax'][:]
    tmean = (tmax + tmin)/2.0;
    tday = ((tmax - tmean)*TDAYCOEF) + tmean;
    tmean = None
    tmax = None
    srad = ds_tm.variables['srad'][:]
    vpd = ds_tm.variables['vpd'][:]
    
    lats = ds_tm.variables['lat'][:]
    lons = ds_tm.variables['lon'][:]
    
    days_daily = utld.get_days_metadata(datetime(2000,1,1), datetime(2009,12,31))
    ydays = days_daily[YDAY]
    years_daily = days_daily[YEAR]
    
    yr_str = 2000
    yr_end = 2009
    
    yrs = np.arange(yr_str,yr_end+1)
    days = np.zeros(yrs.size*MODIS_8DAYS.size)
    
    x = 0
    for yr in yrs:
        
        i = 0
        for day in MODIS_8DAYS:
            
            days[i+x] = long("".join([str(yr),'%03d' % day]))
            i+=1
            
        x+=MODIS_8DAYS.size
        

    ncdf_file = Dataset('/projects/daymet2/modis/crown_topomet_8day.nc','w')
    #Set global attributes
    ncdf_file.title = "Crown of the Continent Met Data"
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.source = "TopoMet"
    ncdf_file.history = "".join(["Created on: ",datetime.strftime(date.today(),"%Y-%m-%d")]) 
    ncdf_file.comment = "30-arcsec geographic projection"
    
    ncdf_file.createDimension('lat',lats.size)
    ncdf_file.createDimension('lon',lons.size)
    ncdf_file.createDimension('time',days.size)
    
    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = days
    
    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    chnk_size = (1,lats.size,lons.size)
    tmin_var = ncdf_file.createVariable('tmin','f4',('time','lat','lon',),chunksizes=chnk_size)
    tmin_var.long_name = "minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "minimum_air_temperature"
    tmin_var.missing_value = netCDF4.default_fillvals['f4']
    
    tday_var = ncdf_file.createVariable('tday','f4',('time','lat','lon',),chunksizes=chnk_size)
    tday_var.long_name = "daytime air temperature"
    tday_var.units = "C"
    tday_var.standard_name = "daytime_air_temperature"
    tday_var.missing_value = netCDF4.default_fillvals['f4']
    
    vpd_var = ncdf_file.createVariable('vpd','f4',('time','lat','lon',),chunksizes=chnk_size)
    vpd_var.long_name = "water vapor saturation deficit"
    vpd_var.units = "Pa"
    vpd_var.standard_name = "water_vapor_saturation_deficit"
    vpd_var.missing_value = netCDF4.default_fillvals['f4']
    
    srad_var = ncdf_file.createVariable('srad','f4',('time','lat','lon',),chunksizes=chnk_size)
    srad_var.long_name = "surface net downward shortwave flux"
    srad_var.units = "W m-2"
    srad_var.standard_name = "surface_net_downward_shortwave_flux"
    srad_var.missing_value = netCDF4.default_fillvals['f4']
    
    i = 0
    
    for year in yrs:
        
        print year
        
        for day in MODIS_8DAYS:
            
            str_yday = day
            end_yday = day+8
            
            if str_yday == 361:
                
                day_mask = np.logical_and(years_daily == year, ydays >= str_yday)
            
            else:
            
                day_mask = np.logical_and(years_daily == year, np.logical_and(ydays >= str_yday, ydays < end_yday))
            
             
            tmin_var[i,:,:] = np.mean(tmin[day_mask,:,:],axis=0)
            tday_var[i,:,:] = np.mean(tday[day_mask,:,:],axis=0)
            srad_var[i,:,:] = np.mean(srad[day_mask,:,:],axis=0)
            vpd_var[i,:,:] = np.mean(vpd[day_mask,:,:],axis=0)
            i+=1
    
    
    ncdf_file.close()
    

def create_ncdf_topomet_crown():
    
    ncdf_cmask = Dataset("/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc","r")
    ncdf_tmin = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_tmin.nc","r")
    ncdf_tmax = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_tmax.nc","r")
    ncdf_vpd = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_vpd.nc","r")
    ncdf_srad = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_srad.nc","r")
    
    cmask = ncdf_cmask.variables['mask'][:,:]
    nonzero_rows_cmask,nonzero_cols_cmask = np.nonzero(cmask)
    nonzero_rows_cmask = np.unique(nonzero_rows_cmask)
    nonzero_cols_cmask = np.unique(nonzero_cols_cmask)
    
    lons = ncdf_cmask.variables['lon'][:]
    lats = ncdf_cmask.variables['lat'][:]
    lons = lons[nonzero_cols_cmask]
    lats = lats[nonzero_rows_cmask]
    
    time = ncdf_tmin.variables['time'][TIME_START_INDEX:]
    
    ncdf_file = Dataset('/projects/daymet2/modis/crown_topomet.nc','w')
    #Set global attributes
    ncdf_file.title = "Crown of the Continent Met Data"
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.source = "TopoMet"
    ncdf_file.history = "".join(["Created on: ",datetime.strftime(date.today(),"%Y-%m-%d")]) 
    ncdf_file.comment = "30-arcsec geographic projection"
    
    ncdf_file.createDimension('lat',lats.size)
    ncdf_file.createDimension('lon',lons.size)
    ncdf_file.createDimension('time',time.size)
    
    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.units = "days since 1950-1-1 0:0:0"
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = time
    
    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    tmin_var = ncdf_file.createVariable('tmin','f4',('time','lat','lon',),chunksizes=(time.size,50,50),zlib=True)
    tmin_var.long_name = "minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "minimum_air_temperature"
    tmin_var.missing_value = netCDF4.default_fillvals['f4']
    tmin_var[:,:,:] = ncdf_tmin.variables['tmin'][TIME_START_INDEX:,nonzero_rows_cmask,nonzero_cols_cmask]
    ncdf_file.sync()
    ncdf_tmin.close()
    
    tmax_var = ncdf_file.createVariable('tmax','f4',('time','lat','lon',),chunksizes=(time.size,50,50),zlib=True)
    tmax_var.long_name = "maximum air temperature"
    tmax_var.units = "C"
    tmax_var.standard_name = "maximum_air_temperature"
    tmax_var.missing_value = netCDF4.default_fillvals['f4']
    tmax_var[:,:,:] = ncdf_tmax.variables['tmax'][TIME_START_INDEX:,nonzero_rows_cmask,nonzero_cols_cmask]
    ncdf_file.sync()
    ncdf_tmax.close()
    
    vpd_var = ncdf_file.createVariable('vpd','f4',('time','lat','lon',),chunksizes=(time.size,50,50),zlib=True)
    vpd_var.long_name = "water vapor saturation deficit"
    vpd_var.units = "Pa"
    vpd_var.standard_name = "water_vapor_saturation_deficit"
    vpd_var.missing_value = netCDF4.default_fillvals['f4']
    vpd_var[:,:,:] = ncdf_vpd.variables['vpd'][TIME_START_INDEX:,nonzero_rows_cmask,nonzero_cols_cmask]
    ncdf_file.sync()
    ncdf_vpd.close()
    
    srad_var = ncdf_file.createVariable('srad','f4',('time','lat','lon',),chunksizes=(time.size,50,50),zlib=True)
    srad_var.long_name = "surface net downward shortwave flux"
    srad_var.units = "W m-2"
    srad_var.standard_name = "surface_net_downward_shortwave_flux"
    srad_var.missing_value = netCDF4.default_fillvals['f4']
    srad_var[:,:,:] = ncdf_srad.variables['srad'][TIME_START_INDEX:,nonzero_rows_cmask,nonzero_cols_cmask]
    ncdf_file.sync()
    ncdf_srad.close()
    
    ncdf_file.close()

def create_ncdf_merra_crown():
    
    ds = RasterDataset('/projects/daymet2/cce_case_study/topowx_files/annual/cce_topowx_tmin19482012ann.tif')
    cmask = ~ds.readAsArray().mask
    
    
#    ncdf_cmask = Dataset("/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc","r")
#    
#    cmask = ncdf_cmask.variables['mask'][:,:]
    nonzero_rows_cmask,nonzero_cols_cmask = np.nonzero(cmask)
    nonzero_rows_cmask = np.unique(nonzero_rows_cmask)
    nonzero_cols_cmask = np.unique(nonzero_cols_cmask)
    nonzero_mask = cmask[nonzero_rows_cmask,:]
    nonzero_mask = nonzero_mask[:,nonzero_cols_cmask]
    cmask = nonzero_mask
    
    lats,lons = ds.getCoordGrid1d()
    
#    lons = ncdf_cmask.variables['lon'][:]
#    lats = ncdf_cmask.variables['lat'][:]
    lons = lons[nonzero_cols_cmask]
    lats = lats[nonzero_rows_cmask]
    
    yr_str = 2000
    yr_end = 2009
    
    yrs = np.arange(yr_str,yr_end+1)
    days = np.zeros(yrs.size*MODIS_8DAYS.size)
    
    x = 0
    for yr in yrs:
        
        i = 0
        for day in MODIS_8DAYS:
            
            days[i+x] = long("".join([str(yr),'%03d' % day]))
            i+=1
            
        x+=MODIS_8DAYS.size 

    #################################################
    # init ncdf
    #################################################

    ncdf_file = Dataset('/projects/daymet2/mod16_aguposter/crown_merra.nc','w')
    #Set global attributes
    ncdf_file.title = "Crown of the Continent Merra Smoothed Met Data"
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.source = "GMAO Merra and MOD16"
    ncdf_file.history = "".join(["Created on: ",datetime.strftime(date.today(),"%Y-%m-%d")]) 
    ncdf_file.comment = "30-arcsec geographic projection"
    
    ncdf_file.createDimension('lat',lats.size)
    ncdf_file.createDimension('lon',lons.size)
    ncdf_file.createDimension('time',days.size)
    
    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.units = "days since 1950-1-1 0:0:0"
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = days
    
    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    chnk_size = (1,lats.size,lons.size)
    tmin_var = ncdf_file.createVariable('tmin','f4',('time','lat','lon',),chunksizes=chnk_size)
    tmin_var.long_name = "minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "minimum_air_temperature"
    tmin_var.missing_value = netCDF4.default_fillvals['f4']
    
    tday_var = ncdf_file.createVariable('tday','f4',('time','lat','lon',),chunksizes=chnk_size)
    tday_var.long_name = "daytime air temperature"
    tday_var.units = "C"
    tday_var.standard_name = "daytime_air_temperature"
    tday_var.missing_value = netCDF4.default_fillvals['f4']
    
    vpd_var = ncdf_file.createVariable('vpd','f4',('time','lat','lon',),chunksizes=chnk_size)
    vpd_var.long_name = "water vapor saturation deficit"
    vpd_var.units = "Pa"
    vpd_var.standard_name = "water_vapor_saturation_deficit"
    vpd_var.missing_value = netCDF4.default_fillvals['f4']
    
    srad_var = ncdf_file.createVariable('srad','f4',('time','lat','lon',),chunksizes=chnk_size)
    srad_var.long_name = "surface net downward shortwave flux"
    srad_var.units = "W m-2"
    srad_var.standard_name = "surface_net_downward_shortwave_flux"
    srad_var.missing_value = netCDF4.default_fillvals['f4']

    #################################################

    yr_dirs = np.sort(os.listdir(MERRA_DATA_PATH))
    #yr_dirs = [yr_dirs[0]]
    
    i = 0
    for yr in yr_dirs:
        
        print yr
        
        yr_path =  "".join([MERRA_DATA_PATH,yr,"/"])
        day_files = np.sort(os.listdir(yr_path))
        
        for fname in day_files:
            
            fpath = "".join([yr_path,fname])
            
            sds_tmin = modis_hdf_dataset(fpath,'Tmin_1km')
            sds_srad = modis_hdf_dataset(fpath,'SWrad_1km')
            sds_tday = modis_hdf_dataset(fpath,'Tday_1km')
            sds_vpd = modis_hdf_dataset(fpath,'VPDday_1km',0.1)
            
            a_out_tmin = np.zeros(cmask.shape)
            a_out_srad = np.zeros(cmask.shape)
            a_out_tday = np.zeros(cmask.shape)
            a_out_vpd = np.zeros(cmask.shape)
            
            rng_rows = np.arange(lats.size)
            rng_cols = np.arange(lons.size)
            
            for r in rng_rows:
                
                for c in rng_cols:
                    
                    x,y = sds_tmin.gridcell_offset_lonlat(lons[c], lats[r])
       
                    a_out_tmin[r,c] =  sds_tmin.a[y,x]
                    a_out_srad[r,c] =  sds_srad.a[y,x]
                    a_out_tday[r,c] =  sds_tday.a[y,x]
                    a_out_vpd[r,c] =  sds_vpd.a[y,x]

            a_out_tmin[np.isnan(a_out_tmin)] = netCDF4.default_fillvals['f4']
            a_out_srad[np.isnan(a_out_srad)] = netCDF4.default_fillvals['f4']
            a_out_tday[np.isnan(a_out_tday)] = netCDF4.default_fillvals['f4']
            a_out_vpd[np.isnan(a_out_vpd)] = netCDF4.default_fillvals['f4']

            tmin_var[i,:,:] = a_out_tmin
            srad_var[i,:,:] = a_out_srad
            tday_var[i,:,:] = a_out_tday
            vpd_var[i,:,:] = a_out_vpd
            
            i+=1
    
    ncdf_file.close()


def create_ncdf_crown_mask():
    
    fmf = input_raster('/projects/daymet2/dem/interp_mask_crown.tif')

    ds_lc = Dataset("/projects/daymet2/modis/crown_landcover.nc")
    
    lats = ds_lc.variables['lat'][:]
    lons = ds_lc.variables['lon'][:]
    
    mask = np.zeros((lats.size,lons.size))
    
    for x in np.arange(lats.size):
        
        for i in np.arange(lons.size):
            
            mask[x,i] = fmf.getDataValue(lons[i], lats[x])
    

    ncdf_file = Dataset('/projects/daymet2/modis/crown_mask_crown.nc','w')

    #Create 2-dimensions
    ncdf_file.createDimension('lat',lats.size)
    ncdf_file.createDimension('lon',lons.size)

    latitudes = ncdf_file.createVariable('lat','f8',('lat',))
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',))
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    ncdf_var = ncdf_file.createVariable('mask','i1',('lat','lon',),fill_value=False)
    ncdf_var[:,:] = mask
    ncdf_file.close()
    
    
    plt.imshow(mask)
    plt.show()

def create_ncdf_fmf_mask():
    
    fmf = input_raster('/projects/daymet2/dem/interp_mask_fmf.tif')

    ds_lc = Dataset("/projects/daymet2/modis/crown_landcover.nc")
    
    lats = ds_lc.variables['lat'][:]
    lons = ds_lc.variables['lon'][:]
    
    mask = np.zeros((lats.size,lons.size))
    
    for x in np.arange(lats.size):
        
        for i in np.arange(lons.size):
            
            mask[x,i] = fmf.getDataValue(lons[i], lats[x])
    

    ncdf_file = Dataset('/projects/daymet2/modis/crown_mask_fmf.nc','w')

    #Create 2-dimensions
    ncdf_file.createDimension('lat',lats.size)
    ncdf_file.createDimension('lon',lons.size)

    latitudes = ncdf_file.createVariable('lat','f8',('lat',))
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',))
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    ncdf_var = ncdf_file.createVariable('mask','i1',('lat','lon',),fill_value=False)
    ncdf_var[:,:] = mask
    ncdf_file.close()
    
    
    plt.imshow(mask)
    plt.show()
    
def create_ncdf_glac_mask():
    
    fmf = input_raster('/projects/daymet2/dem/interp_mask_glac2_clipped.tif')

    ds_lc = Dataset("/projects/daymet2/modis/crown_landcover.nc")
    
    lats = ds_lc.variables['lat'][:]
    lons = ds_lc.variables['lon'][:]
    
    mask = np.zeros((lats.size,lons.size))
    
    for x in np.arange(lats.size):
        
        for i in np.arange(lons.size):
            
            mask[x,i] = fmf.getDataValue(lons[i], lats[x])
    

    ncdf_file = Dataset('/projects/daymet2/modis/crown_mask_glac.nc','w')

    #Create 2-dimensions
    ncdf_file.createDimension('lat',lats.size)
    ncdf_file.createDimension('lon',lons.size)

    latitudes = ncdf_file.createVariable('lat','f8',('lat',))
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',))
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    ncdf_var = ncdf_file.createVariable('mask','i1',('lat','lon',),fill_value=False)
    ncdf_var[:,:] = mask
    ncdf_file.close()
    
    
    plt.imshow(mask)
    plt.show()

def create_ncdf_dem():
    
    ncdf_conusmask = Dataset("/projects/daymet2/dem/smoothed/ncdf/interp_mask_conus.nc","r")
    ncdf_cmask = Dataset("/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc","r")
    ncdf_dem = Dataset("/projects/daymet2/dem/smoothed/ncdf/dem_orig.nc","r")
    
    mask = ncdf_conusmask.variables['mask'][:,:]
    nonzero_rows,nonzero_cols = np.nonzero(mask)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    nonzero_mask = mask[nonzero_rows,:]
    nonzero_mask = nonzero_mask[:,nonzero_cols]
    mask = None
    
    lons = ncdf_conusmask.variables['lon'][:]
    lats = ncdf_conusmask.variables['lat'][:]
    nonzero_lons = lons[nonzero_cols]
    nonzero_lats = lats[nonzero_rows]
    ncdf_conusmask.close()
    
    dem = ncdf_dem.variables['elev'][:,:]
    nonzero_dem = dem[nonzero_rows,:]
    nonzero_dem = nonzero_dem[:,nonzero_cols]
    dem = None
    ncdf_dem.close()
    
    cmask = ncdf_cmask.variables['mask'][:,:]
    nonzero_rows_cmask,nonzero_cols_cmask = np.nonzero(cmask)
    nonzero_rows_cmask = np.unique(nonzero_rows_cmask)
    nonzero_cols_cmask = np.unique(nonzero_cols_cmask)
    final_mask = nonzero_mask[nonzero_rows_cmask,:]
    final_mask = final_mask[:,nonzero_cols_cmask]

    final_dem = nonzero_dem[nonzero_rows_cmask,:]
    final_dem = final_dem[:,nonzero_cols_cmask]
    
    final_lons = nonzero_lons[nonzero_cols_cmask]
    final_lats = nonzero_lats[nonzero_rows_cmask]
    
    ncdf_file = Dataset('/projects/daymet2/modis/crown_dem.nc','w')
    #Set global attributes
    ncdf_file.title = "Crown of the Continent DEM"
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.source = "PRISM Normals DEM"
    ncdf_file.history = "".join(["Created on: ",datetime.strftime(date.today(),"%Y-%m-%d")]) 
    ncdf_file.comment = "30-arcsec geographic projection"
    
    ncdf_file.createDimension('lat',final_lats.size)
    ncdf_file.createDimension('lon',final_lons.size)
    
    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = final_lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = final_lons
    
    ncdf_var = ncdf_file.createVariable("elev",'f4',('lat','lon',),fill_value=False)
    ncdf_var[:,:] = final_dem
    
    ncdf_file.close()
    
    print final_dem.dtype
    plt.imshow(final_dem)
    plt.show()
    
def create_ncdf_landcover():
    
    ncdf_mask = Dataset("/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc","r")

    mask = ncdf_mask.variables['mask'][:,:]
    nonzero_rows,nonzero_cols = np.nonzero(mask)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    nonzero_mask = mask[nonzero_rows,:]
    nonzero_mask = nonzero_mask[:,nonzero_cols]
    mask = nonzero_mask
    
    lons = ncdf_mask.variables['lon'][:]
    lats = ncdf_mask.variables['lat'][:]
    nonzero_lons = lons[nonzero_cols]
    nonzero_lats = lats[nonzero_rows]
    lons = nonzero_lons
    lats = nonzero_lats
    ncdf_mask.close()
    
    #Create ncdf file
    ########################################################################
    ncdf_file = Dataset("/projects/daymet2/modis/crown_landcover.nc",'w')
    #Set global attributes
    ncdf_file.title = "MOD12Q1 v004 Land Cover Type 2 (UMD) Crown of the Continent (subset of tile h10v04)"
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.source = "MOD12Q1"
    ncdf_file.history = "".join(["Created on: ",datetime.strftime(date.today(),"%Y-%m-%d")]) 
    ncdf_file.references = "http://www.ntsg.umt.edu/project/mod16"
    ncdf_file.comment = "30-arcsec geographic projection"
    
    ncdf_file.createDimension('lat',lats.size)
    ncdf_file.createDimension('lon',lons.size)
    ########################################################################
    
    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    ncdf_var = ncdf_file.createVariable('lc','i2',('lat','lon'))
    ncdf_var.long_name = "land cover"
    ncdf_var.units = "land cover type 2 (UMD)"
    ncdf_var.standard_name = "land cover"
    ncdf_var.missing_value = 255
    ncdf_var.land_cover_classes = "water = 0\n \
evergreen needleleaf forest = 1\n \
evergreen broadleaf forest = 2\n \
deciduous needleleaf forest = 3\n \
deciduous broadleaf forest = 4\n \
mixed forests = 5\n \
closed shrublands = 6\n \
open shrublands = 7\n \
woody savannas = 8\n \
savannas = 9\n \
grasslands = 10\n \
croplands = 12\n \
urban and built-up = 13\n \
barren or sparsely vegetated = 16\n \
unclassfied = 254"
    
    ds_lc = modis_hdf_dataset("/projects/daymet2/modis/MOD12Q1.A2001001.h10v04.004.2004358134106.hdf",'Land_Cover_Type_2')
            
    a_out = np.zeros(mask.shape)
            
    rng_rows = np.arange(lats.size)
    rng_cols = np.arange(lons.size)
    
    for r in rng_rows:
        
        for c in rng_cols:
                       
            a_out[r,c] =  ds_lc.data_value_lonlat(lons[c],lats[r])

    ncdf_var[:,:] = a_out
            
    plt.imshow(ncdf_var[:,:])
    plt.show()
    
    ncdf_file.close()
        

def create_ncdf_gmao_et():
    
    ncdf_mask = Dataset("/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc","r")

    mask = ncdf_mask.variables['mask'][:,:]
    nonzero_rows,nonzero_cols = np.nonzero(mask)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    nonzero_mask = mask[nonzero_rows,:]
    nonzero_mask = nonzero_mask[:,nonzero_cols]
    mask = nonzero_mask
    
    lons = ncdf_mask.variables['lon'][:]
    lats = ncdf_mask.variables['lat'][:]
    nonzero_lons = lons[nonzero_cols]
    nonzero_lats = lats[nonzero_rows]
    lons = nonzero_lons
    lats = nonzero_lats
    ncdf_mask.close()
    
    #ncdf_et = None
    ncdf_et = init_ncdf('/projects/daymet2/modis/mod16_test.nc', lons, lats,2000,2009)
    var_et = ncdf_et.variables['et']
    
    yr_dirs = np.sort(os.listdir(MOD16_PATH))
    yr_dirs = yr_dirs[0:-1] #don't include 2010
    
    i = 0
    for yr in yr_dirs:
        
        yr_path =  "".join([MOD16_PATH,yr,"/"])
        day_dirs = np.sort(os.listdir(yr_path))
        
        for day in day_dirs:
            
            day_path = "".join([yr_path,day,"/"])
            files = os.listdir(day_path)
            fname =  [x for x in files if TILE in x][0]
            
            file_path = "".join([day_path,fname])
            
            ds_et = modis_et_dataset(file_path)
            trans_pt = modis_sin_latlon_transform()
            
            a_out = np.zeros(mask.shape)
            
            rng_rows = np.arange(lats.size)
            rng_cols = np.arange(lons.size)
            
            for r in rng_rows:
                
                for c in rng_cols:
                               
                    a_out[r,c] =  ds_et.data_value_lonlat(lons[c],lats[r])
                    #print trans_pt.trans_nad83_to_sin.TransformPoint(lon,lat)

            #a_out = np.ma.masked_greater(a_out, MAX_VALID_VALUE)
            a_out[np.isnan(a_out)] = netCDF4.default_fillvals['i2']
            #a_out = np.ma.filled(a_out, netCDF4.default_fillvals['i2'])
            var_et[i,:,:] = a_out
            
#            plt.imshow(var_et[i,:,:])
#            plt.colorbar()
#            plt.show()
            #print i
            i+=1
        print yr
    ncdf_et.close()
    

def create_ncdf_gmao_et_ann():
    
    ncdf_mask = Dataset("/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc","r")

    mask = ncdf_mask.variables['mask'][:,:]
    nonzero_rows,nonzero_cols = np.nonzero(mask)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    nonzero_mask = mask[nonzero_rows,:]
    nonzero_mask = nonzero_mask[:,nonzero_cols]
    mask = nonzero_mask
    
    lons = ncdf_mask.variables['lon'][:]
    lats = ncdf_mask.variables['lat'][:]
    nonzero_lons = lons[nonzero_cols]
    nonzero_lats = lats[nonzero_rows]
    lons = nonzero_lons
    lats = nonzero_lats
    ncdf_mask.close()
    
    #ncdf_et = None
    ncdf_et = init_ncdf_ann('/projects/daymet2/modis/mod16ann_merragmao.nc', lons, lats,2000,2009)
    var_et = ncdf_et.variables['et']
    
    yr_dirs = np.sort(os.listdir(MOD16_PATH_ANN))
    yr_dirs = yr_dirs[0:-1] #don't include 2010
    
    i = 0
    for yr in yr_dirs:
        
        yr_path =  "".join([MOD16_PATH_ANN,yr,"/"])
        files = os.listdir(yr_path)
        fname =  [x for x in files if TILE in x][0]
            
        file_path = "".join([yr_path,fname])
        
        ds_et = modis_et_dataset(file_path,MAX_VALID_VALUE_ANN)
        
        a_out = np.zeros(mask.shape)
        
        rng_rows = np.arange(lats.size)
        rng_cols = np.arange(lons.size)
            
        for r in rng_rows:
            
            for c in rng_cols:
                           
                a_out[r,c] =  ds_et.data_value_lonlat(lons[c],lats[r])

        a_out[np.isnan(a_out)] = netCDF4.default_fillvals['i2']
        var_et[i,:,:] = a_out
        
        i+=1
        print yr
    ncdf_et.close()

def create_ncdf_topomet_et():
    
    ncdf_mask = Dataset("/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc","r")

    mask = ncdf_mask.variables['mask'][:,:]
    nonzero_rows,nonzero_cols = np.nonzero(mask)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    nonzero_mask = mask[nonzero_rows,:]
    nonzero_mask = nonzero_mask[:,nonzero_cols]
    mask = nonzero_mask
    
    lons = ncdf_mask.variables['lon'][:]
    lats = ncdf_mask.variables['lat'][:]
    nonzero_lons = lons[nonzero_cols]
    nonzero_lats = lats[nonzero_rows]
    lons = nonzero_lons
    lats = nonzero_lats
    ncdf_mask.close()
    
    ncdf_et = init_ncdf('/projects/daymet2/modis/mod16_topomet.nc', lons, lats,2000,2009)
    var_et = ncdf_et.variables['et']
    
    files = np.sort(os.listdir(MOD16_TOPOMET_PATH))
    
    i = 0
    for fname in files:
        
        fpath =  "".join([MOD16_TOPOMET_PATH,fname])
        
        ds_et = modis_et_dataset(fpath)
        
        a_out = np.zeros(mask.shape)
            
        rng_rows = np.arange(lats.size)
        rng_cols = np.arange(lons.size)
        
        for r in rng_rows:
            
            for c in rng_cols:
                           
                a_out[r,c] =  ds_et.data_value_lonlat(lons[c],lats[r])

        a_out[np.isnan(a_out)] = netCDF4.default_fillvals['i2']
        var_et[i,:,:] = a_out
        
        i+=1
        

        
        print fname
            
    ncdf_et.close()


def create_ncdf_topomet_et_ann():
    
    ncdf_mask = Dataset("/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc","r")

    mask = ncdf_mask.variables['mask'][:,:]
    nonzero_rows,nonzero_cols = np.nonzero(mask)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    nonzero_mask = mask[nonzero_rows,:]
    nonzero_mask = nonzero_mask[:,nonzero_cols]
    mask = nonzero_mask
    
    lons = ncdf_mask.variables['lon'][:]
    lats = ncdf_mask.variables['lat'][:]
    nonzero_lons = lons[nonzero_cols]
    nonzero_lats = lats[nonzero_rows]
    lons = nonzero_lons
    lats = nonzero_lats
    ncdf_mask.close()
    
    #ncdf_et = None
    ncdf_et = init_ncdf_ann('/projects/daymet2/modis/mod16ann_topomet.nc', lons, lats,2000,2009)
    var_et = ncdf_et.variables['et']
    
    files = np.sort(os.listdir(MOD16_TOPOMET_PATH_ANN))
    
    i = 0
    for fname in files:
        
        fpath =  "".join([MOD16_TOPOMET_PATH_ANN,fname])
        
        ds_et = modis_et_dataset(fpath,MAX_VALID_VALUE_ANN)
        
        a_out = np.zeros(mask.shape)
            
        rng_rows = np.arange(lats.size)
        rng_cols = np.arange(lons.size)
        
        for r in rng_rows:
            
            for c in rng_cols:
                           
                a_out[r,c] =  ds_et.data_value_lonlat(lons[c],lats[r])

        a_out[np.isnan(a_out)] = netCDF4.default_fillvals['i2']
        var_et[i,:,:] = a_out
        
        i+=1
        
        print fname
            
    ncdf_et.close()

if __name__ == '__main__':
    create_ncdf_merra_crown()
    #create_ncdf_crown_mask()
    #create_ncdf_glac_mask()
    #create_ncdf_fmf_mask()
    #create_ncdf_topomet_crown_8day()
    #create_ncdf_merra_crown()
    #create_ncdf_topomet_crown()
    #create_ncdf_topomet_et_ann()
    #create_ncdf_gmao_et_ann()
    #create_ncdf_topomet_et()
    
    #create_ncdf_landcover()
    #sys.exit()
    #create_ncdf_dem()
    #sys.exit()
    

            
            
        
#    ds_et = modis_et_dataset('/projects/daymet2/modis/MOD16A2.A2009177.h10v04.105.2010355183415.hdf')
#    trans_pt = modis_sin_latlon_transform()
#    
#    a_out = np.zeros(mask.shape)
#    
#    rng_rows = np.arange(lats.size)
#    rng_cols = np.arange(lons.size)
#    
#    for r in rng_rows:
#        
#        for c in rng_cols:
#            
#            
#            a_out[r,c] =  ds_et.data_value_lonlat(lons[c],lats[r])
#            
#            #print trans_pt.trans_nad83_to_sin.TransformPoint(lon,lat)
#            
#            
#    a_out = np.ma.masked_greater(a_out, MAX_VALID_VALUE)
#    a_out = a_out*SCALE_FACTOR
#    
#    plt.imshow(a_out)
#    plt.show()
    