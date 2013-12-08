'''
Created on Nov 11, 2011
Utilities for using NCEP/NCAR reanalysis data

@author: jared.oyler
'''
from netCDF4 import Dataset
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import utils.util_dates as utld
from utils.util_dates import YMD,DATE
from datetime import datetime

#min lon, max lon, min lat, max lat
REANALYSIS_BOUNDS = (-130,-60,20,52.5)
YEARS = np.arange(1948,2012)

def create_single_reanalysis_file(path_renalysis,out_path,bnds=REANALYSIS_BOUNDS,yrs=YEARS):
    '''
    Merges separate annual NCEP/NCAR reanalysis tair/hgt files into a single netcdf file
    @param path_renalysis:
    @param out_path:
    @param bnds:
    @param yrs:
    '''
    
    ds_air = Dataset("".join([path_renalysis,"air.",str(yrs[0]),".nc"]))
        
    lon = ds_air.variables['lon'][:]
    lon[lon>180] = lon[lon>180]-360.0
    lat = ds_air.variables['lat'][:]
    level = ds_air.variables['level'][:]
    ds_air.close()
    
    mask_lat = np.logical_and(lat<=bnds[3],lat>=bnds[2])
    mask_lon = np.logical_and(lon<=bnds[1],lon>=bnds[0])
    mask_level = level >= 300
    
    lon = lon[mask_lon]
    lat = lat[mask_lat]
    level = level[mask_level]
    
    airs = []
    hgts = []
    dates = []
    
    for year in yrs:
        
        ds_air = Dataset("".join([path_renalysis,"air.",str(year),".nc"]))
        airs.append(ds_air.variables['air'][:,mask_level,mask_lat,mask_lon]-273.15) #convert K->C
        time = ds_air.variables['time']
        dates.append(netCDF4.num2date(time[:],units=time.units))
        ds_air.close()
        
        ds_hgt = Dataset("".join([path_renalysis,"hgt.",str(year),".nc"]))
        hgts.append(ds_hgt.variables['hgt'][:,mask_level,mask_lat,mask_lon])
        ds_hgt.close()
        print year

    air = np.vstack(airs)
    hgt = np.vstack(hgts)
    date = np.concatenate(dates)
    
    ncdf_file = Dataset(out_path,'w')
    #Set global attributes
    ncdf_file.title = "NCEP/NCAR Reanalysis CONUS Tair and Hgt 1948-2011"
    ncdf_file.references = "http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.html"
    
    #Create 4-dimensions
    dim_time = ncdf_file.createDimension('time',date.size)
    dim_level = ncdf_file.createDimension('level',level.size)
    dim_lat = ncdf_file.createDimension('lat',lat.size)
    dim_lon = ncdf_file.createDimension('lon',lon.size)
    
    #Create dimension variables and fill with values
    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.units = "days since 1948-1-1 0:0:0"
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = netCDF4.date2num(date,times.units)
    
    levels = ncdf_file.createVariable('level','f4',('level',),fill_value=False)
    levels.long_name = "level"
    levels.units = "millibar"
    levels.standard_name = "level"
    levels.calendar = "standard"
    levels[:] = level
    
    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lat

    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lon
    
    ncdf_var = ncdf_file.createVariable('air','f4',('time','level','lat','lon'))
    ncdf_var.long_name = "mean Daily Air temperature"
    ncdf_var.units = "C"
    ncdf_var.missing_value = 32766
    ncdf_var[:,:,:,:] = air
    
    ncdf_var = ncdf_file.createVariable('hgt','f4',('time','level','lat','lon'))
    ncdf_var.long_name = "mean Daily Geopotential height"
    ncdf_var.units = "m"
    ncdf_var.missing_value = 32766
    ncdf_var[:,:,:,:] = hgt
    
    ncdf_file.close()

class free_air_interp():
    '''
    Class for vertical interpolation of daily avg. NCEP/NCAR reanalysis tair to a specific pt elevation 
    '''
    
    def __init__(self,path_nc_file,str_ymd=19480101,end_ymd=20110930):
        
        ds = Dataset(path_nc_file)
        
        var_time = ds.variables['time']
        str_date,end_date = netCDF4.num2date(var_time[0], var_time.units),netCDF4.num2date(var_time[-1], var_time.units)
        days = utld.get_days_metadata(str_date, end_date)
        mask_time = np.logical_and(days[YMD]>=str_ymd,days[YMD]<=end_ymd)
        str_date,end_date = days[DATE][mask_time][[0,-1]]
        print str_date,end_date
        self.days = utld.get_days_metadata(str_date, end_date)
        self.day_rng = np.arange(self.days[YMD].size)
        self.mask_time = mask_time
        
        self.lon = ds.variables['lon'][:]
        self.lat = ds.variables['lat'][:]
        self.level = ds.variables['level'][:]
        print "loading tair..."
        self.air = ds.variables['air'][:,:,:,:]
        print "loading hgt..."
        self.hgt = ds.variables['hgt'][:,:,:,:]
        
        '''
        Create GDAL-like geotransform list to define resolution and bounds
        GeoTransform[0] /* top left x */
        GeoTransform[1] /* w-e pixel resolution */
        GeoTransform[2] /* rotation, 0 if image is "north up" */
        GeoTransform[3] /* top left y */
        GeoTransform[4] /* rotation, 0 if image is "north up" */
        GeoTransform[5] /* n-s pixel resolution */
        '''
        self.geoTransform = [None]*6
        #n-s pixel height/resolution needs to be negative.  not sure why?
        self.geoTransform[5] = -np.abs(self.lat[0] - self.lat[1])   
        self.geoTransform[1] = np.abs(self.lon[0] - self.lon[1])
        self.geoTransform[2],self.geoTransform[4] = (0.0,0.0)
        self.geoTransform[0] = self.lon[0] - (self.geoTransform[1]/2.0) 
        self.geoTransform[3] = self.lat[0] + np.abs(self.geoTransform[5]/2.0)
    
    def get_grid_offset(self,lon,lat):
        
        originX = self.geoTransform[0]
        originY = self.geoTransform[3]
        pixelWidth = self.geoTransform[1]
        pixelHeight = self.geoTransform[5]
        
        xOffset = abs(int((lon - originX) / pixelWidth))
        yOffset = abs(int((lat - originY) / pixelHeight))
        return xOffset,yOffset
    
    def get_free_air_hgt(self,lon,lat,hgt):
        hgt_index = np.nonzero(self.level == hgt)[0][0]
        col,row = self.get_grid_offset(lon, lat)
        return np.array(self.air[self.mask_time,hgt_index,row,col],dtype=np.float64)
        
    def get_free_air(self,lon,lat,elev):
        
        col,row = self.get_grid_offset(lon, lat)    
        
        pt_air = self.air[:,:,row,col]
        pt_hgt = self.hgt[:,:,row,col]
        
        free_air = np.zeros(self.day_rng.size)
        
        for day in self.day_rng:
            
            day_air = pt_air[day,:]
            day_hgt = pt_hgt[day,:]
            
            try:
                
                lwr_bnd = np.nonzero(day_hgt<=elev)[0][-1]
            
            except IndexError:
                
                #pt elev is below lowest hgt. need to extrapolate
                #print "|".join(["WARNING: Extrapolating below lowest hgt ",str(lon),str(lat),str(elev),str(day)])
                free_air[day] = linear_extrap(elev,np.float64(day_hgt[1]),np.float64(day_hgt[0]),np.float64(day_air[1]),np.float64(day_air[0]))
                continue
            
            try:
                
                up_bnd = np.nonzero(day_hgt>=elev)[0][0]
            
            except IndexError:
                
                #pt elev is above highest hgt. need to extrapolate
                #print "WARNING: Extrapolating above highest hgt"
                free_air[day] = linear_extrap(elev,np.float64(day_hgt[-2]),np.float64(day_hgt[-1]),np.float64(day_air[-2]),np.float64(day_air[-1]))
                continue
            
            if lwr_bnd == up_bnd:
                #The elevation of the point is at an exact pressure level--do not interpolate
                free_air[day] = day_air[lwr_bnd]
            else:
                free_air[day] = linear_interp(elev,np.float64(day_hgt[lwr_bnd]),np.float64(day_hgt[up_bnd]),np.float64(day_air[lwr_bnd]),np.float64(day_air[up_bnd]))
        
        return free_air
            
def linear_interp(x,x0,x1,y0,y1):
    
    return y0 + ((x-x0)*((y1-y0)/(x1-x0)))

def linear_extrap(x,x0,x1,y0,y1):
    
    return y0 + (((x - x0)/(x1 - x0))*(y1-y0))




if __name__ == '__main__':
    
#    create_single_reanalysis_file('/projects/daymet2/reanalysis_data/ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/',
#                                  '/projects/daymet2/reanalysis_data/ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/conus.air.hgt.1948.2011.nc')
#    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    
    fair_interp = free_air_interp('/projects/daymet2/reanalysis_data/ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/conus.air.hgt.1948.2011.nc')
    #print fair_interp.get_grid_offset(-113.430344,47.230488)
    #fair = fair_interp.get_free_air(-113.430344,47.230488,1405.469727)
    fair = fair_interp.get_free_air(-108.7,47.6167,705.0)
    print np.nonzero(np.isnan(fair))[0].size
    
    plt.plot(fair)
    #plt.plot(fair2)
    plt.show()
    
    