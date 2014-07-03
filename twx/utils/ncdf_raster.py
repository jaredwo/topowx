'''
Created on Jul 10, 2012

@author: jared.oyler
'''
from netCDF4 import Dataset
import numpy as np

class ncdf_raster():
    
    def __init__(self,path,var):

        dataset = Dataset(path,"r",format=format)
        self.lons = dataset.variables['lon'][:]
        #make sure no lons > 180
        self.lons[self.lons>180] = self.lons[self.lons>180]-360.0
        self.lats = dataset.variables['lat'][:]
        self.vals = dataset.variables[var][:]
        dataset.close()
        
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
        self.geoTransform[5] = -np.abs(self.lats[0] - self.lats[1])   
        self.geoTransform[1] = np.abs(self.lons[0] - self.lons[1])
        self.geoTransform[2],self.geoTransform[4] = (0.0,0.0)
        self.geoTransform[0] = self.lons[0] - (self.geoTransform[1]/2.0) 
        self.geoTransform[3] = self.lats[0] + np.abs(self.geoTransform[5]/2.0)
        
        self.cols = self.lons.size
        self.rows = self.lats.size
        
        self.min_x = self.geoTransform[0]
        self.max_x = self.min_x + (self.cols*self.geoTransform[1])
        self.max_y =  self.geoTransform[3]
        self.min_y =  self.max_y - (-self.rows*self.geoTransform[5])
    
    def getGridCellOffset(self,lon,lat):
        
        if not self.is_inbounds(lon, lat):
            raise Exception("Lon/Lat outside raster extent")
        
        originX = self.geoTransform[0]
        originY = self.geoTransform[3]
        pixelWidth = self.geoTransform[1]
        pixelHeight = self.geoTransform[5]
        
        xOffset = abs(int((lon - originX) / pixelWidth))
        yOffset = abs(int((lat - originY) / pixelHeight))
        return xOffset,yOffset
    
    def is_inbounds(self,x_geo,y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y
    
    def get_data_value(self,lon,lat,useCache=False):
        
        col,row = self.getGridCellOffset(lon,lat)
        return self.vals[row,col]