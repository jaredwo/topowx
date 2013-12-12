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

    def __init__(self,dsPath):
        
        self.gdalDs = gdal.Open(dsPath)
        
        #GDAL GeoTransform. 
        #Top left x,y are for the upper left corner of upper left pixel
        #GeoTransform[0] /* top left x */
        #GeoTransform[1] /* w-e pixel resolution */
        #GeoTransform[2] /* rotation, 0 if image is "north up" */
        #GeoTransform[3] /* top left y */
        #GeoTransform[4] /* rotation, 0 if image is "north up" */
        #GeoTransform[5] /* n-s pixel resolution */
        self.geoT = np.array(self.gdalDs.GetGeoTransform())
        
        self.projection = self.gdalDs.GetProjection()
        self.sourceSR = osr.SpatialReference()
        self.sourceSR.ImportFromWkt(self.projection)
        self.targetSR = osr.SpatialReference()
        self.targetSR.ImportFromEPSG(PROJECTION_GEO_WGS84)
        self.coordTrans_src_to_wgs84 = osr.CoordinateTransformation(self.sourceSR, self.targetSR)
        self.coordTrans_wgs84_to_src = osr.CoordinateTransformation(self.targetSR, self.sourceSR)
        
        #y,x = self.getCoordGrid1d()
        
#        self.min_x = x[0]-np.abs(self.geoT[1])
#        self.max_x = x[-1]-np.abs(self.geoT[1])
#        self.max_y = y[0]+np.abs(self.geoT[5])
#        self.min_y =  y[-1]-np.abs(self.geoT[5])
        
        self.min_x = self.geoT[0]
        self.max_x = self.min_x + (self.gdalDs.RasterXSize*self.geoT[1])
        self.max_y =  self.geoT[3]
        self.min_y =  self.max_y - (-self.gdalDs.RasterYSize*self.geoT[5])
        
        self.rows = self.gdalDs.RasterYSize
        self.cols = self.gdalDs.RasterXSize
        self.ndata = self.gdalDs.GetRasterBand(1).GetNoDataValue()
        
    
    def getCoordMeshGrid(self):
        
        #Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geoT[0] + (self.geoT[1] / 2.0)
        urX = self.getCoord(0, self.gdalDs.RasterXSize-1)[1]
        
        #Get the upper left and lower left y coordinates
        ulY = self.geoT[3] + (self.geoT[5] / 2.0)
        llY = self.getCoord(self.gdalDs.RasterYSize-1, 0)[0]
        
        #Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdalDs.RasterXSize)
        y = np.linspace(ulY, llY, self.gdalDs.RasterYSize)
        
        xGrid, yGrid = np.meshgrid(x, y)
        
        return yGrid, xGrid
    
    def getCoordGrid1d(self):
        
        #Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geoT[0] + (self.geoT[1] / 2.0)
        urX = self.getCoord(0, self.gdalDs.RasterXSize-1)[1]
        
        #Get the upper left and lower left y coordinates
        ulY = self.geoT[3] + (self.geoT[5] / 2.0)
        llY = self.getCoord(self.gdalDs.RasterYSize-1, 0)[0]
        
        #Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdalDs.RasterXSize)
        y = np.linspace(ulY, llY, self.gdalDs.RasterYSize)
        
        return y, x
        
    def getCoord(self, row, col):
        
        xCoord = (self.geoT[0] + col*self.geoT[1] + row*self.geoT[2]) + self.geoT[1] / 2.0
        yCoord = (self.geoT[3] + col*self.geoT[4] + row*self.geoT[5]) + self.geoT[5] / 2.0
        
        return yCoord,xCoord

    def getRowCol(self,lon,lat):
        '''Returns the grid cell offset for this raster based on the input wgs84 lon/lat'''
        xGeo, yGeo, zGeo = self.coordTrans_wgs84_to_src.TransformPoint(lon,lat) 
        
        if not self.isInbounds(xGeo, yGeo):
            raise OutsideExtent("lat/lon outside raster extent: "+str(lat)+","+str(lon))
        
        originX = self.geoT[0]
        originY = self.geoT[3]
        pixelWidth = self.geoT[1]
        pixelHeight = self.geoT[5]
        
        xOffset = np.abs(np.int((xGeo - originX) / pixelWidth))
        yOffset = np.abs(np.int((yGeo - originY) / pixelHeight))
        
        row = int(yOffset)
        col = int(xOffset)
        
        return row,col

    def getDataValue(self,lon,lat):
        
        row,col = self.getRowCol(lon,lat)
        data_val = self.gdalDs.ReadAsArray(col,row,1,1)[0,0] 
        #data_val = self.readDataArray(col,row,1,1)[0,0]        
        return data_val

    def readAsArray(self):
        
        a = self.gdalDs.GetRasterBand(1).ReadAsArray()
        a = np.ma.masked_equal(a, self.gdalDs.GetRasterBand(1).GetNoDataValue())
        return a
    
    def isInbounds(self,x_geo,y_geo):
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
        self.sourceSR = osr.SpatialReference()
        self.sourceSR.ImportFromWkt(self.projection)
        self.targetSR = osr.SpatialReference()
        self.targetSR.ImportFromEPSG(PROJECTION_GEO_WGS84)
        self.coordTrans_src_to_wgs84 = osr.CoordinateTransformation(self.sourceSR, self.targetSR)
        self.coordTrans_wgs84_to_src = osr.CoordinateTransformation(self.targetSR, self.sourceSR)
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
        
        if not self.is_inbounds(xGeo, yGeo):
            raise OutsideExtent("lat/lon outside raster extent: "+str(lat)+","+str(lon))
        
        originX = self.geoTransform[0]
        originY = self.geoTransform[3]
        pixelWidth = self.geoTransform[1]
        pixelHeight = self.geoTransform[5]
        
        xOffset = abs(int((xGeo - originX) / pixelWidth))
        yOffset = abs(int((yGeo - originY) / pixelHeight))
        return xOffset,yOffset
    
    def is_inbounds(self,x_geo,y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y
    
    def getDataValue(self,lon,lat,useCache=False):
        
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