'''
Created on Jul 2, 2014

@author: jaredwo
'''
import osgeo.gdal as gdal
import osgeo.gdalconst as gdalconst
import osgeo.osr as osr
import numpy as np

PROJECTION_GEO_WGS84 = 4326 #EPSG Code
PROJECTION_GEO_NAD83 = 4269 #EPSG Code
PROJECTION_GEO_WGS72 = 4322 #EPSG Code

class OutsideExtent(Exception):
    pass

class RasterDataset(object):

    def __init__(self,ds_path):
        
        self.gdal_ds = gdal.Open(ds_path)
        
        #GDAL GeoTransform. 
        #Top left x,y are for the upper left corner of upper left pixel
        #GeoTransform[0] /* top left x */
        #GeoTransform[1] /* w-e pixel resolution */
        #GeoTransform[2] /* rotation, 0 if image is "north up" */
        #GeoTransform[3] /* top left y */
        #GeoTransform[4] /* rotation, 0 if image is "north up" */
        #GeoTransform[5] /* n-s pixel resolution */
        self.geo_t = np.array(self.gdal_ds.GetGeoTransform())
        
        self.projection = self.gdal_ds.GetProjection()
        self.source_sr = osr.SpatialReference()
        self.source_sr.ImportFromWkt(self.projection)
        self.target_sr = osr.SpatialReference()
        self.target_sr.ImportFromEPSG(PROJECTION_GEO_WGS84)
        self.coordTrans_src_to_wgs84 = osr.CoordinateTransformation(self.source_sr, self.target_sr)
        self.coordTrans_wgs84_to_src = osr.CoordinateTransformation(self.target_sr, self.source_sr)
                
        self.min_x = self.geo_t[0]
        self.max_x = self.min_x + (self.gdal_ds.RasterXSize*self.geo_t[1])
        self.max_y =  self.geo_t[3]
        self.min_y =  self.max_y - (-self.gdal_ds.RasterYSize*self.geo_t[5])
        
        self.rows = self.gdal_ds.RasterYSize
        self.cols = self.gdal_ds.RasterXSize
        self.ndata = self.gdal_ds.GetRasterBand(1).GetNoDataValue()
        
    
    def get_coord_mesh_grid(self):
        
        #Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geo_t[0] + (self.geo_t[1] / 2.0)
        urX = self.get_coord(0, self.gdal_ds.RasterXSize-1)[1]
        
        #Get the upper left and lower left y coordinates
        ulY = self.geo_t[3] + (self.geo_t[5] / 2.0)
        llY = self.get_coord(self.gdal_ds.RasterYSize-1, 0)[0]
        
        #Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdal_ds.RasterXSize)
        y = np.linspace(ulY, llY, self.gdal_ds.RasterYSize)
        
        xGrid, yGrid = np.meshgrid(x, y)
        
        return yGrid, xGrid
    
    def get_coord_grid_1d(self):
        
        #Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geo_t[0] + (self.geo_t[1] / 2.0)
        urX = self.get_coord(0, self.gdal_ds.RasterXSize-1)[1]
        
        #Get the upper left and lower left y coordinates
        ulY = self.geo_t[3] + (self.geo_t[5] / 2.0)
        llY = self.get_coord(self.gdal_ds.RasterYSize-1, 0)[0]
        
        #Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdal_ds.RasterXSize)
        y = np.linspace(ulY, llY, self.gdal_ds.RasterYSize)
        
        return y, x
        
    def get_coord(self, row, col):
        
        xCoord = (self.geo_t[0] + col*self.geo_t[1] + row*self.geo_t[2]) + self.geo_t[1] / 2.0
        yCoord = (self.geo_t[3] + col*self.geo_t[4] + row*self.geo_t[5]) + self.geo_t[5] / 2.0
        
        return yCoord,xCoord

    def get_row_col(self,lon,lat):
        '''Returns the grid cell offset for this raster based on the input wgs84 lon/lat'''
        xGeo, yGeo, zGeo = self.coordTrans_wgs84_to_src.TransformPoint(lon,lat) 
        
        if not self.is_inbounds(xGeo, yGeo):
            raise OutsideExtent("lat/lon outside raster extent: "+str(lat)+","+str(lon))
        
        originX = self.geo_t[0]
        originY = self.geo_t[3]
        pixelWidth = self.geo_t[1]
        pixelHeight = self.geo_t[5]
        
        xOffset = np.abs(np.int((xGeo - originX) / pixelWidth))
        yOffset = np.abs(np.int((yGeo - originY) / pixelHeight))
        
        row = int(yOffset)
        col = int(xOffset)
        
        return row,col

    def get_data_value(self,lon,lat):
        
        row,col = self.get_row_col(lon,lat)
        data_val = self.gdal_ds.ReadAsArray(col,row,1,1)[0,0] 
        #data_val = self.readDataArray(col,row,1,1)[0,0]        
        return data_val

    def read_as_array(self):
        
        a = self.gdal_ds.GetRasterBand(1).ReadAsArray()
        a = np.ma.masked_equal(a, self.gdal_ds.GetRasterBand(1).GetNoDataValue())
        return a
    
    def is_inbounds(self,x_geo,y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y
