import osgeo.gdal as gdal
import osgeo.gdalconst as gdalconst
import osgeo.osr as osr
import numpy as np
import netCDF4 as ncdf


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
        
        #y,x = self.get_coord_grid_1d()
        
#        self.min_x = x[0]-np.abs(self.geo_t[1])
#        self.max_x = x[-1]-np.abs(self.geo_t[1])
#        self.max_y = y[0]+np.abs(self.geo_t[5])
#        self.min_y =  y[-1]-np.abs(self.geo_t[5])
        
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
        
        if not self.__is_inbounds(xGeo, yGeo):
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
    
    def __is_inbounds(self,x_geo,y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y

class input_raster(object):
    '''
    Encapsulates an input gdal raster (e.g.-dem,slope,aspect raster datasets)
    GeoTransform[0] /* top left x */
    GeoTransform[1] /* w-e pixel resolution */
    GeoTransform[2] /* rotation, 0 if image is "north up" */
    GeoTransform[3] /* top left y */
    GeoTransform[4] /* rotation, 0 if image is "north up" */
    GeoTransform[5] /* n-s pixel resolution */

    '''

    def __init__(self, filePath,bandNum=1):
        '''
        Params
        filePath: the path to the dataset.  If the path is invalid, an error
        will be raised
        '''
        # ds.RasterCount
        self.raster = gdal.Open(filePath, gdalconst.GA_ReadOnly)
        if self.raster is None:
            raise Exception ('the raster file could not be opened')
        self.geoTransform = self.raster.GetGeoTransform()
        self.projection = self.raster.GetProjection()
        self.rows = self.raster.RasterYSize
        self.cols = self.raster.RasterXSize
        self.source_sr = osr.SpatialReference()
        self.source_sr.ImportFromWkt(self.projection)
        self.target_sr = osr.SpatialReference()
        self.target_sr.ImportFromEPSG(PROJECTION_GEO_WGS84)
        self.coordTrans_src_to_wgs84 = osr.CoordinateTransformation(self.source_sr, self.target_sr)
        self.coordTrans_wgs84_to_src = osr.CoordinateTransformation(self.target_sr, self.source_sr)
        self.data = self.raster.GetRasterBand(bandNum)
        self.dataCache = dict()
        self.min_x = self.geoTransform[0]
        self.max_x = self.min_x + (self.cols*self.geoTransform[1])
        self.max_y =  self.geoTransform[3]
        self.min_y =  self.max_y - (-self.rows*self.geoTransform[5])
        self.ndata = self.data.GetNoDataValue()
    
    def __del__(self):
        '''closes the gdal object'''
        self.raster = None
    
    def res(self):
        return self.geoTransform[1],self.geoTransform[1]
    
    def getGeoLocation(self, xPixel,yLine):
        '''Affine Transfrom: Converts pixel row and column to spatially referenced coordinates (in native projection)'''
        Xgeo = (self.geoTransform[0] + xPixel*self.geoTransform[1] + yLine*self.geoTransform[2]) + self.geoTransform[1] / 2.0
        Ygeo = (self.geoTransform[3] + xPixel*self.geoTransform[4] + yLine*self.geoTransform[5]) + self.geoTransform[5] / 2.0
        return Xgeo,Ygeo
    
    def getGridCellOffset(self,lon,lat):
        '''Returns the grid cell offset for this raster based on the input wgs84 lon/lat'''
        xGeo, yGeo, zGeo = self.coordTrans_wgs84_to_src.TransformPoint(lon,lat) 
        
        if not self.__is_inbounds(xGeo, yGeo):
            raise OutsideExtent("lat/lon outside raster extent: "+str(lat)+","+str(lon))
        
        originX = self.geoTransform[0]
        originY = self.geoTransform[3]
        pixelWidth = self.geoTransform[1]
        pixelHeight = self.geoTransform[5]
        
        xOffset = abs(int((xGeo - originX) / pixelWidth))
        yOffset = abs(int((yGeo - originY) / pixelHeight))
        return xOffset,yOffset
    
    def __is_inbounds(self,x_geo,y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y
    
    def get_data_value(self,lon,lat,useCache=False):
        
        col,row = self.getGridCellOffset(lon,lat)
        
        data_val = self.readDataArray(col,row,1,1)[0,0]
        
#        if data_val == self.ndata:
#            raise Exception("Coords have no data")
        
        return data_val
    
    def col_row_arrays(self):
        
        rcgrid = np.mgrid[0:self.rows,0:self.cols]
        rows = rcgrid[0]
        cols = rcgrid[1]
        
#        cols = np.empty((self.rows,self.cols),dtype=np.int64)
#        for col in np.arange(self.cols):
#            cols[:,col] = col
#        rows = np.empty((self.rows,self.cols),dtype=np.int64)
#        for row in np.arange(self.rows):
#            rows[row,:] = row
        
        return cols,rows
    
    def x_y_arrays(self):
        
        cols,rows = self.col_row_arrays()
        return self.getGeoLocation(cols, rows)
               
    def getLatLon(self, aColNum,aRowNum,transform=True):
        '''Converts pixel row and column to WGS 84 Lat/Lon'''
        #First get the spatial referenced coordinates
        x,y = self.getGeoLocation(aColNum, aRowNum)
        #Convert to WGS84
        if transform:
            lon, lat, Z = self.coordTrans_src_to_wgs84.TransformPoint(x,y)
        else:
            lon,lat = x,y
        return  lat, lon
    
    def readEntireRaster(self):
        return self.readDataArray(0,0,self.cols,self.rows)
    
    def readEntireRasterBand(self,bandNum):
        band = self.raster.GetRasterBand(bandNum)
        return band.ReadAsArray(0,0,self.cols,self.rows)
    
    def to_ncdf(self,path,var_name,dtype):
        
        lat = self.getLatLon(0.0,np.arange(self.rows),transform=False)[0]
        lon = self.getLatLon(np.arange(self.cols),0.0,transform=False)[1]
        
        val = self.readEntireRaster()
        
        ncdf_file = ncdf.Dataset(path,'w')
    
        #Create 2-dimensions
        ncdf_file.createDimension('lat',lat.size)
        ncdf_file.createDimension('lon',lon.size)
    
        latitudes = ncdf_file.createVariable('lat','f8',('lat',))
        latitudes.long_name = "latitude"
        latitudes.units = "degrees_north"
        latitudes.standard_name = "latitude"
        latitudes[:] = lat
    
        longitudes = ncdf_file.createVariable('lon','f8',('lon',))
        longitudes.long_name = "longitude"
        longitudes.units = "degrees_east"
        longitudes.standard_name = "longitude"
        longitudes[:] = lon
        
        ncdf_var = ncdf_file.createVariable(var_name,dtype,('lat','lon',),fill_value=False)
        ncdf_var[:,:] = val
        if self.data.GetNoDataValue() is not None:
            ncdf_var.missing_value = self.data.GetNoDataValue()
        
        ncdf_file.close()
        
    def readDataArray(self,aColNumStart,aRowNumStart,numCols=1,numRows=1):
        '''
        Read a subset of the dataset.  Provides the same functionality as
        the GDAL ReadAsArray function
        
        Return
        A numpy array
        '''
        return self.data.ReadAsArray(aColNumStart,aRowNumStart,numCols,numRows)