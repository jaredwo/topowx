import osgeo.gdalconst as gdalconst
import osgeo.gdal as gdal

NO_DATA = -32767.

class output_raster(object):
    '''
    Encapsulates an output gdal raster
    '''

    def __init__(self, aFileName,inputRaster,numbands = 1,reuseExisting=False,noDataVal=NO_DATA,datatype=gdalconst.GDT_Float32):
        '''
        Constructor
        If the output dataset already exists on the file system, it will simply
        point to this dataset instead of creating a new one
        
        Params
        aFileName: the file name for the dataset.  
        aConfig: the loaded config object
        
        '''
        driver = gdal.GetDriverByName("GTiff")
        fullOutputPath = aFileName
        self.numbands = numbands
        self.raster = gdal.Open(fullOutputPath, gdalconst.GA_Update)
        self.noDataVal = noDataVal
        if self.raster is None and not reuseExisting:
            self.raster = driver.Create(fullOutputPath, 
                                        inputRaster.cols,
                                        inputRaster.rows,
                                        numbands, datatype) 
                                        #numbands, gdalconst.GDT_Float32)
            self.raster.SetGeoTransform(inputRaster.geo_t)
            self.raster.SetProjection(inputRaster.projection)
            [self.initBand(bandNum) for bandNum in range(1,numbands+1)]
        elif reuseExisting:
            if self.raster is None:
                raise(Exception("There is no existing output raster to reuse"+fullOutputPath))
            if self.numbands != self.raster.RasterCount:
                raise(Exception("Existing raster file does not have the num of bands specified"+fullOutputPath))
        else:
            raise(Exception("Raster file already exists"+fullOutputPath))
    def initBand(self,bandNum):
        band = self.raster.GetRasterBand(bandNum)
        #band.Fill(Constants.NO_DATA)
        band.SetNoDataValue(self.noDataVal)
    
    def readDataArray(self,aColNumStart,aRowNumStart,numCols,numRows,bandNum=1):
        band = self.raster.GetRasterBand(bandNum)
        return band.ReadAsArray(aColNumStart,aRowNumStart,numCols,numRows)
        
    def writeDataArray(self,aArray,aColNumStart,aRowNumStart,bandNum=1):
        '''
        Write to a subset of the dataset.  Provides the same functionality as
        the GDAL WriteArray function
        '''
        band = self.raster.GetRasterBand(bandNum)
        band.WriteArray(aArray,aColNumStart,aRowNumStart)
        band.FlushCache()
        band.GetStatistics(0, 1)