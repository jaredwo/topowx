'''
Utility functions for NetCDF files

Copyright 2014, Jared Oyler.

This file is part of TopoWx.

TopoWx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TopoWx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.
'''
import osgeo.gdalconst as gdalconst
import osgeo.gdal as gdal
import osgeo.osr as osr
import numpy as np
from netCDF4 import Dataset
from twx.raster import RasterDataset

PROJ_GEO_WGS84 = 4326 #EPSG Code
PROJ_GEO_NAD83 = 4269 #EPSG Code
PROJ_GEO_WGS72 = 4322 #EPSG Code

DTYPES_NP_TO_GDAL = {   np.dtype(np.float32):gdalconst.GDT_Float32,
                        np.dtype(np.float64):gdalconst.GDT_Float64,
                        np.dtype(np.int16):gdalconst.GDT_Int16,
                        np.dtype(np.int32):gdalconst.GDT_Int32,
                        np.dtype(np.uint16):gdalconst.GDT_UInt16,
                        np.dtype(np.int8):gdalconst.GDT_Byte,
                        np.dtype(np.uint8):gdalconst.GDT_UInt16}
DTYPES_GDAL_TO_NP = {a_value:a_key for a_key,a_value in DTYPES_NP_TO_GDAL.items()}

VAR_LON = "lon"
VAR_LAT = "lat"

TIME_SUM = "SUM"
TIME_AVG = "AVG"

ATTRS_NODATA = ('_FillValue','missing_value')


def expand_grid(ds,varname,expand_dims,outpath,mask,ndata = None):
            
    #Crop each raster dataset to only include cols,rows that are within the actual mask
    data = ds.variables[varname][:]    
    nonzero_rows,nonzero_cols = np.nonzero(mask)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    nonzero_rows = np.arange(nonzero_rows[0],nonzero_rows[-1]+1)
    nonzero_cols = np.arange(nonzero_cols[0],nonzero_cols[-1]+1)
    nonzero_data = data[nonzero_rows,:]
    nonzero_data = nonzero_data[:,nonzero_cols]
    data = nonzero_data
    
    lons = ds.variables['lon'][:]
    lats = ds.variables['lat'][:]
    lons = lons[nonzero_cols]
    lats = lats[nonzero_rows]
    
    res = np.abs(lons[0] - lons[1])
    nrows,ncols = data.shape
    var_dtype = ds.variables[varname].dtype
    
    if ndata is None:
    
        for ndata_attr in ATTRS_NODATA:
            
            try:
                ndata = np.asscalar(ds.variables[varname].getncattr(ndata_attr))
                break
            except AttributeError:
                continue
    
    if ndata is None:
        raise Exception('Could not determine no data values for input dataset variable: '
                        +varname)
    
    
    #################################################################
    #Determine # of rows and columns to add to top/bottom,left/right
    #################################################################
    add_rows = expand_dims[0] - nrows
    add_cols = expand_dims[1] - ncols
    
    if add_rows < 0 or add_cols < 0:
        raise Exception("Number of rows/cols to expand by must be greater than current number of rows/cols")
     
    if add_rows%2.0 == 0.0:
        
        nadd = np.int(add_rows/2.0)
        add_row_top = nadd
        add_row_bot = nadd
    
    else:
        
        add_row_top = np.int(np.round(add_rows/2.0))
        add_row_bot = add_rows - add_row_top
        
    if add_cols%2.0 == 0.0:
        
        nadd = np.int(add_cols/2.0)
        add_col_left = nadd
        add_col_right = nadd
    
    else:
        
        add_col_left = np.int(np.round(add_cols/2.0))
        add_col_right = add_cols - add_col_left
    #################################################################

    #################################################################
    # Lats to add to top
    #################################################################        
    add_top_lats = np.empty(add_row_top)
    start_lat = lats[0]
    for x in np.arange(add_row_top-1,-1,-1):
        add_top_lats[x] = start_lat + res
        start_lat = add_top_lats[x]
    #################################################################
    
    #################################################################
    # Lats to add to bottom
    #################################################################        
    add_bot_lats = np.empty(add_row_bot)
    start_lat = lats[-1]
    for x in np.arange(add_row_bot):
        add_bot_lats[x] = start_lat - res
        start_lat = add_bot_lats[x]
    #################################################################
    
    #################################################################
    # Lons to add to left
    #################################################################        
    add_left_lons = np.empty(add_col_left)
    start_lon = lons[0]
    for x in np.arange(add_col_left-1,-1,-1):
        add_left_lons[x] = start_lon - res
        start_lon = add_left_lons[x] 
    #################################################################
    
    #################################################################
    # Lons to add to right
    #################################################################        
    add_right_lons = np.empty(add_col_right)
    start_lon = lons[-1]
    for x in np.arange(add_col_right):
        add_right_lons[x] = start_lon + res
        start_lon = add_right_lons[x] 
    #################################################################

    new_lats = np.concatenate((add_top_lats,lats,add_bot_lats))
    new_lons = np.concatenate((add_left_lons,lons,add_right_lons))
    
    new_data = np.vstack((np.ones((add_row_top,data.shape[1]),dtype=var_dtype)*ndata,data,np.ones((add_row_bot,data.shape[1]),dtype=var_dtype)*ndata))
    new_data = np.hstack((np.ones((new_data.shape[0],add_col_left),dtype=var_dtype)*ndata,new_data,np.ones((new_data.shape[0],add_col_right),dtype=var_dtype)*ndata))
    
    ds_new = Dataset(outpath,'w')

    #Create 2-dimensions
    ds_new.createDimension('lat',new_lats.size)
    ds_new.createDimension('lon',new_lons.size)

    latitudes = ds_new.createVariable('lat','f8',('lat',))
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = new_lats

    longitudes = ds_new.createVariable('lon','f8',('lon',))
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = new_lons
    
    ncdf_var = ds_new.createVariable(varname,ds.variables[varname].dtype,('lat','lon',),fill_value=False)
    ncdf_var[:] = new_data
    for attrname in ds.variables[varname].ncattrs():
        ncdf_var.setncattr(attrname,ds.variables[varname].getncattr(attrname))
    
    ds_new.close()

def grid_wgs84_to_raster(fpath, a, lon, lat, gdal_dtype, ndata=None, gdal_driver="GTiff"):
    
    '''
    Create GDAL geotransform list to define resolution and bounds
    GeoTransform[0] /* top left x */
    GeoTransform[1] /* w-e pixel resolution */
    GeoTransform[2] /* rotation, 0 if image is "north up" */
    GeoTransform[3] /* top left y */
    GeoTransform[4] /* rotation, 0 if image is "north up" */
    GeoTransform[5] /* n-s pixel resolution */
    '''
    geo_t = [None]*6
    #n-s pixel height/resolution needs to be negative.  not sure why?
    geo_t[5] = -np.abs(lat[0] - lat[1])   
    geo_t[1] = np.abs(lon[0] - lon[1])
    geo_t[2],geo_t[4] = (0.0,0.0)
    geo_t[0] = lon[0] - (geo_t[1]/2.0) 
    geo_t[3] = lat[0] + np.abs(geo_t[5]/2.0)
    
    ds_out = gdal.GetDriverByName(gdal_driver).Create(fpath, int(a.shape[1]),
                                                      int(a.shape[0]),
                                                      1, gdal_dtype)
    ds_out.SetGeoTransform(geo_t)
    
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(PROJ_GEO_WGS84)
    ds_out.SetProjection(sr.ExportToWkt())
    
    band_out = ds_out.GetRasterBand(1)
    if ndata is not None:
        band_out.SetNoDataValue(ndata)
    band_out.WriteArray(np.ma.filled(a, ndata))
                
    ds_out.FlushCache()
    ds_out = None
    
def raster_wgs84_to_ncdf(rast_path,varname,out_path,attdict=None):
    
    rast = RasterDataset(rast_path)
    a = np.ma.getdata(rast.read_as_array())
    nodata = rast.gdal_ds.GetRasterBand(1).GetNoDataValue()
    
    lat,lon = rast.get_coord_grid_1d()
    
    ncdf_file = Dataset(out_path,'w')

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
    
    ncdf_var = ncdf_file.createVariable(varname,a.dtype,('lat','lon',),fill_value=False)
    #if nodata is not None:
    ncdf_var.missing_value = nodata
    ncdf_var.setncatts(attdict)
        
    ncdf_var[:,:] = a
    ncdf_file.close()
    
class GeoNc():
    
    def __init__(self,ds):
        
        self.ds = ds
        
        lons = ds.variables['lon'][:]
        #make sure no lons > 180
        lons[lons>180] = lons[lons>180]-360.0
        lats = ds.variables['lat'][:]
        
        '''
        Create GDAL geotransform list to define resolution and bounds
        GeoTransform[0] /* top left x */
        GeoTransform[1] /* w-e pixel resolution */
        GeoTransform[2] /* rotation, 0 if image is "north up" */
        GeoTransform[3] /* top left y */
        GeoTransform[4] /* rotation, 0 if image is "north up" */
        GeoTransform[5] /* n-s pixel resolution */
        '''
        self.geo_t = [None]*6
        #n-s pixel height/resolution needs to be negative.
        self.geo_t[5] = -np.abs(lats[0] - lats[1])   
        self.geo_t[1] = np.abs(lons[0] - lons[1])
        self.geo_t[2],self.geo_t[4] = (0.0,0.0)
        self.geo_t[0] = lons[0] - (self.geo_t[1]/2.0) 
        self.geo_t[3] = lats[0] + np.abs(self.geo_t[5]/2.0)
        
        self.lons = lons
        self.lats = lats
        
    
    def get_row_col(self,lon,lat):
                
        originX = self.geo_t[0]
        originY = self.geo_t[3]
        pixelWidth = self.geo_t[1]
        pixelHeight = self.geo_t[5]
        
        col = abs(int((lon - originX) / pixelWidth))
        row = abs(int((lat - originY) / pixelHeight))
        return row,col,self.lons[col],self.lats[row] 
    
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
        self.geo_t = [None]*6
        #n-s pixel height/resolution needs to be negative.  not sure why?
        self.geo_t[5] = -np.abs(self.y[0] - self.y[1])   
        self.geo_t[1] = np.abs(self.x [0] - self.x[1])
        self.geo_t[2],self.geo_t[4] = (0.0,0.0)
        self.geo_t[0] = self.x[0] - (self.geo_t[1]/2.0) 
        self.geo_t[3] = self.y[0] + np.abs(self.geo_t[5]/2.0)
                
        self.min_x = self.geo_t[0]
        self.max_x = self.min_x + (self.x.size*self.geo_t[1])
        self.max_y =  self.geo_t[3]
        self.min_y =  self.max_y - (-self.y.size*self.geo_t[5])
        
    def toGTiff(self,fpathOut,a,proj4='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'):
        
        drvr =  gdal.GetDriverByName('GTiff')
        dsOut = drvr.Create(fpathOut, a.shape[1], a.shape[0], 1, DTYPES_NP_TO_GDAL[a.dtype])
        
        sr = osr.SpatialReference()
        sr.ImportFromProj4(proj4)
        dsOut.SetProjection(sr.ExportToWkt())
        dsOut.SetGeoTransform(self.geo_t)
        
        bandOut = dsOut.GetRasterBand(1)
        
        if np.ma.isMA(a):
            bandOut.Fill(float(a.fill_value))
            bandOut.SetNoDataValue(float(a.fill_value))
            a = np.ma.filled(a)
    
        bandOut.WriteArray(np.ma.filled(a),0,0) 
        bandOut.FlushCache()
        
    def getGridCellOffset(self,x,y):
        
        if not self.is_inbounds(x, y):
            raise Exception("Lon/Lat outside raster extent")
        
        originX = self.geo_t[0]
        originY = self.geo_t[3]
        pixelWidth = self.geo_t[1]
        pixelHeight = self.geo_t[5]
        
        xOffset = abs(int((x - originX) / pixelWidth))
        yOffset = abs(int((y - originY) / pixelHeight))
        return xOffset,yOffset
    
    def is_inbounds(self,x,y):
        return x >= self.min_x and x <= self.max_x and y >= self.min_y and y <= self.max_y
    


class ncdf_raster():
    
    def __init__(self, filename,var,format='NETCDF4'):

        dataset = Dataset(filename,"r",format=format)
        self.lons = dataset.variables[VAR_LON][:]
        #make sure no lons > 180
        self.lons[self.lons>180] = self.lons[self.lons>180]-360.0
        self.lats = dataset.variables[VAR_LAT][:]
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
        self.geo_t = [None]*6
        #n-s pixel height/resolution needs to be negative.  not sure why?
        self.geo_t[5] = -np.abs(self.lats[0] - self.lats[1])   
        self.geo_t[1] = np.abs(self.lons[0] - self.lons[1])
        self.geo_t[2],self.geo_t[4] = (0.0,0.0)
        self.geo_t[0] = self.lons[0] - (self.geo_t[1]/2.0) 
        self.geo_t[3] = self.lats[0] + np.abs(self.geo_t[5]/2.0)
        
        self.cols = self.lons.size
        self.rows = self.lats.size
        
        self.min_x = self.geo_t[0]
        self.max_x = self.min_x + (self.cols*self.geo_t[1])
        self.max_y =  self.geo_t[3]
        self.min_y =  self.max_y - (-self.rows*self.geo_t[5])
    
    def getGridCellOffset(self,lon,lat):
        
        if not self.is_inbounds(lon, lat):
            raise Exception("Lon/Lat outside raster extent")
        
        originX = self.geo_t[0]
        originY = self.geo_t[3]
        pixelWidth = self.geo_t[1]
        pixelHeight = self.geo_t[5]
        
        xOffset = abs(int((lon - originX) / pixelWidth))
        yOffset = abs(int((lat - originY) / pixelHeight))
        return xOffset,yOffset
    
    def is_inbounds(self,x_geo,y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y
    
    def get_data_value(self,lon,lat,useCache=False):
        
        col,row = self.getGridCellOffset(lon,lat)
        return self.vals[row,col]

def create_nc_mask(rast_in,a_in,mask,out_path):
    
    lat = rast_in.getLatLon(0.0,np.arange(rast_in.rows),transform=False)[0]
    lon = rast_in.getLatLon(np.arange(rast_in.cols),0.0,transform=False)[1]
    
    ncdf_file = Dataset(out_path,'w')

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
    
    a_out = np.zeros(a_in.shape,dtype=np.int)
    a_out[mask] = 1
    
    ncdf_var = ncdf_file.createVariable('mask','i1',('lat','lon',),fill_value=False)
    ncdf_var[:,:] = a_out
    ncdf_file.close()


def toGTiff(a,ds,pathOut,fill=None):
    
    nrow = len(ds.dimensions['lat'])
    ncol = len(ds.dimensions['lon'])
    
    #Get the GDAL datatype
    try:
        dtype = DTYPES_NP_TO_GDAL[a.dtype]
    except KeyError:
        print "".join(["Warning: Numpy Datatype ",str(a.dtype),
                       " not supported. Defaulting to float64."])
        dtype = gdalconst.GDT_Float64
        
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(pathOut,ncol,nrow,1,dtype) 
    
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

    band = raster.GetRasterBand(1)
    if fill is not None:
        band.SetNoDataValue(fill)
    band.WriteArray(a,0,0) 
    band.FlushCache()
    
def to_geotiff(ncdf_file,var,path,time_index=None,dtype=None,nodata_val=None,proj=PROJ_GEO_WGS84):
    
    rows = len(ncdf_file.dimensions['lat'])
    cols = len(ncdf_file.dimensions['lon'])
    
    try:
        dtype = dtype if dtype != None else DTYPES_NP_TO_GDAL[ncdf_file.variables[var].dtype]
    except KeyError:
        raise Exception("Didn't recognize dtype in ncdf variable.  Pass dtype explicitly.")
    
    try:
        nodata_val = nodata_val if nodata_val != None else ncdf_file.variables[var]._FillValue
    except AttributeError:
        raise Exception("Couldn't find missing_value variable attribute.  Pass nodata_val explicitly")
    
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(path,cols,rows,1,dtype) 
    
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
    #n-s pixel height/resolution needs to be negative.  not sure why?
    geotransform[5] = -np.abs(ncdf_file.variables[VAR_LAT][0] - ncdf_file.variables[VAR_LAT][1])   
    geotransform[1] = np.abs(ncdf_file.variables[VAR_LON][0] - ncdf_file.variables[VAR_LON][1])
    geotransform[2],geotransform[4] = (0.0,0.0)
    geotransform[0] = ncdf_file.variables[VAR_LON][0] - (geotransform[1]/2.0) 
    geotransform[3] = ncdf_file.variables[VAR_LAT][0] + np.abs(geotransform[5]/2.0)

    raster.SetGeoTransform(geotransform)

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(proj)
    raster.SetProjection(sr.ExportToWkt())

    band = raster.GetRasterBand(1)
    
    band.SetNoDataValue(float(nodata_val))
    
    if time_index is None:
        band.WriteArray(ncdf_file.variables[var][:,:],0,0) 
    elif time_index == TIME_AVG:
        vals = ncdf_file.variables[var][:,:,:]
        vals_m = np.ma.mean(vals,axis=0)
        #vals_m[vals.mask[0,:,:]] = nodata_val
        vals_m = np.ma.filled(vals_m, nodata_val)
        band.WriteArray(vals_m,0,0)
    elif time_index == TIME_SUM:
        vals = ncdf_file.variables[var][:,:,:]
        vals_s = np.sum(vals,axis=0)
        vals_s[vals[0,:,:]==nodata_val] = nodata_val
        band.WriteArray(vals_s,0,0)
    else:
        band.WriteArray(ncdf_file.variables[var][time_index,:,:],0,0)   
    band.FlushCache()


def to_geotiffa(ncdf_file,a,path,nodata_val,proj=PROJ_GEO_WGS84):
    
    rows = len(ncdf_file.dimensions['lat'])
    cols = len(ncdf_file.dimensions['lon'])
    
    try:
        dtype = DTYPES_NP_TO_GDAL[a.dtype]
    except KeyError:
        raise Exception("Didn't recognize dtype in ncdf variable.  Pass dtype explicitly.")
    
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(path,cols,rows,1,dtype) 
    
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
    #n-s pixel height/resolution needs to be negative.  not sure why?
    geotransform[5] = -np.abs(ncdf_file.variables[VAR_LAT][0] - ncdf_file.variables[VAR_LAT][1])   
    geotransform[1] = np.abs(ncdf_file.variables[VAR_LON][0] - ncdf_file.variables[VAR_LON][1])
    geotransform[2],geotransform[4] = (0.0,0.0)
    geotransform[0] = ncdf_file.variables[VAR_LON][0] - (geotransform[1]/2.0) 
    geotransform[3] = ncdf_file.variables[VAR_LAT][0] + np.abs(geotransform[5]/2.0)

    raster.SetGeoTransform(geotransform)

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(proj)
    raster.SetProjection(sr.ExportToWkt())

    band = raster.GetRasterBand(1)
    
    band.SetNoDataValue(float(nodata_val))
    band.WriteArray(a,0,0)   
    band.FlushCache()


if __name__ == '__main__':
    
#    ds_dem = Dataset('/projects/daymet2/dem/smoothed/ncdf/dem_orig_expand.nc')
#    to_geotiff(ds_dem, 'elev','/projects/daymet2/dem/dem_orig_expand.tif',nodata_val=netCDF4.default_fillvals['f4'])
    
#    ds_tile = Dataset('/projects/daymet2/interp_output/krig_test/h06v02/h06v02_tmin.nc')
#    to_geotiff(ds_tile, 'tmin_mean', '/projects/daymet2/interp_output/krig_test/h06v02/tmin_mean.tif')
#    to_geotiff(ds_tile, 'tmin_se', '/projects/daymet2/interp_output/krig_test/h06v02/tmin_se.tif')
    
#    ds = Dataset('/projects/daymet2/modis/crown_merra.nc')
#    to_geotiff(ds, 'tmin', '/projects/daymet2/docs/usgs_hydro_presentation/merra_tmin.tif', TIME_AVG) #dtype, nodata_val, proj)
    
#    ds = Dataset('/projects/daymet2/modis/crown_merra.nc')
#    to_geotiff(ds, 'srad', '/projects/daymet2/docs/usgs_hydro_presentation/merra_srad.tif', TIME_AVG)
    
#    ds = Dataset('/projects/daymet2/modis/crown_merra.nc')
#    to_geotiff(ds, 'vpd', '/projects/daymet2/docs/usgs_hydro_presentation/merra_vpd.tif', TIME_AVG)
    
    ds = Dataset('/home/jared.oyler/MOD17A3_Science_GPP_2012.nc')
    to_geotiff(ds, 'GPP', '/home/jared.oyler/MOD17A3_Science_GPP_2012-2.tif',nodata_val=65535)
    
#    ds = Dataset('/projects/daymet2/interp_output/montana/h05v01/h05v01_tmax.nc')
#    to_geotiff(ds, 'tmax_mean', '/projects/daymet2/interp_output/montana/h05v01/mean_tmax.tif',nodata_val=9.96921e+36)
#    ds_mask = Dataset('/projects/daymet2/dem/interp_grids/ncdf/interp_mask_conus_expand.nc')
#    to_geotiff(ds_mask, 'mask','/projects/daymet2/dem/interp_mask_conus_expand.tif')
    
    #ds_mask = Dataset('/projects/daymet2/dem/smoothed/ncdf/interp_mask_conus.nc') 
    #mask = ds_mask.variables['mask'][:]
    
    #expand_grid(ds_mask, 'mask', (3000,7000), '/projects/daymet2/dem/smoothed/ncdf/interp_mask_conus_expand.nc',0,mask)
    
    #ds = Dataset('/projects/daymet2/dem/smoothed/ncdf/dem_orig.nc') 
    #expand_grid(ds, 'elev', (3000,7000), '/projects/daymet2/dem/smoothed/ncdf/dem_orig_expand.nc',ds.variables['elev'].missing,mask)
    
#    r_neon = input_raster('/projects/daymet2/dem/interp_mask_crown.tif')
#    print r_neon.geo_t
#    #print r_neon.getGridCellOffset(-188.0,46.8)
#    print r_neon.readEntireRaster().shape
#    sys.exit()
#    a_glac = r_neon.readEntireRaster()
#    mask = a_glac == 1
#    print a_glac[mask].size
#    create_nc_mask(r_neon,a_glac, mask, '/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc')

#    ncdf_file = Dataset('/projects/daymet2/interp_output/fullrun20110916/prism_norm/topomet_prism_norm_tmin.nc')
#    ndata = float(ncdf_file.variables['tmin'].missing_value)
#    to_geotiff(ncdf_file,"tmin",'/projects/daymet2/interp_output/fullrun20110916/prism_norm/topomet_prism_norm_tmin_jan.tif',time_index=0,nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)

    #ncdf_file = Dataset('/projects/daymet2/dem/smoothed/ncdf/interp_mask_conus.nc')
    #to_geotiff(ncdf_file,"mask",'/projects/daymet2/dem/smoothed/ncdf/interp_mask_conus.tif',nodata_val=-999,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)

#    ncdf_file = Dataset('/projects/daymet2/interp_output/bootstrap/topomet_stderr_tmin.nc')
#    ndata = float(ncdf_file.variables['tmin'].missing_value)
#    to_geotiff(ncdf_file,"tmin",'/projects/daymet2/interp_output/bootstrap/topomet_stderr_tmin.tif',nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)

#    ncdf_file = Dataset("/projects/daymet2/interp_output/neon_rgn_12/prism_norm/topomet_prism_norm_tmax.nc","r")
#    ndata = float(ncdf_file.variables['tmax'].missing_value)
#    to_geotiff(ncdf_file,"tmax", "/projects/daymet2/interp_output/neon_rgn_12/prism_norm/topomet_prism_norm_tmax.tif",nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
    
#    ncdf_file = Dataset("/projects/daymet2/interp_output/neon_rgn_12/prism_norm/topomet_prism_norm_prcp.nc","r")
#    ndata = float(ncdf_file.variables['prcp'].missing_value)
#    to_geotiff(ncdf_file,"prcp", "/projects/daymet2/interp_output/neon_rgn_12/prism_norm/topomet_prism_norm_prcp.tif",nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)

#    ncdf_file = Dataset("/projects/daymet2/interp_output/neon_rgn_12/prism_norm/topomet_prism_norm_tmin.nc","r")
#    ndata = float(ncdf_file.variables['tmin'].missing_value)
#    to_geotiff(ncdf_file,"tmin", "/projects/daymet2/interp_output/neon_rgn_12/prism_norm/topomet_prism_norm_tmin.tif",nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
    
#    ncdf_dem = Dataset("/projects/daymet2/dem/smoothed/ncdf/dem_1_0.nc","r")
#    to_geotiff(ncdf_dem,"elev", "/projects/daymet2/dem/smoothed/ncdf/ncdf_to_geotiff_test.tif",proj=PROJ_GEO_NAD83)
    
#    ncdf_file = Dataset('/projects/daymet2/interp_output/fullrun20110916/prism_norm/topomet_prism_norm_prcp.nc',"r")
#    ndata = float(ncdf_file.variables['prcp'].missing_value)
#    to_geotiff(ncdf_file,"prcp", "/projects/daymet2/nasa_debug/topomet_prism_norm_prcp.tif",time_index=12,nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
    
#    ncdf_file = Dataset('/projects/daymet2/interp_output/fullrun20110916/prism_norm/topomet_prism_norm_tmin.nc',"r")
#    ndata = float(ncdf_file.variables['tmin'].missing_value)
#    to_geotiff(ncdf_file,"tmin", "/projects/daymet2/nasa_debug/topomet_prism_norm_tmin.tif",time_index=12,nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
#    
#    ncdf_file = Dataset('/projects/daymet2/interp_output/fullrun20110916/prism_norm/topomet_prism_norm_tmax.nc',"r")
#    ndata = float(ncdf_file.variables['tmax'].missing_value)
#    to_geotiff(ncdf_file,"tmax", "/projects/daymet2/nasa_debug/topomet_prism_norm_tmax.tif",time_index=12,nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
    
#    ncdf_file = Dataset("/projects/daymet2/nasa_debug/topomet_avg_tmin.nc","r")
#    ndata = float(ncdf_file.variables['tmin'].missing_value)
#    to_geotiff(ncdf_file,"tmin", "/projects/daymet2/nasa_debug/topomet_avg_tmin.tif",nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
#    
#    ncdf_file = Dataset("/projects/daymet2/nasa_debug/topomet_avg_tmax.nc","r")
#    ndata = float(ncdf_file.variables['tmax'].missing_value)
#    to_geotiff(ncdf_file,"tmax", "/projects/daymet2/nasa_debug/topomet_avg_tmax.tif",nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
#    
#    ncdf_file = Dataset("/projects/daymet2/nasa_debug/topomet_avg_prcp.nc","r")
#    ndata = float(ncdf_file.variables['prcp'].missing_value)
#    to_geotiff(ncdf_file,"prcp", "/projects/daymet2/nasa_debug/topomet_avg_prcp.tif",nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
#    
#    ncdf_file = Dataset("/projects/daymet2/nasa_debug/topomet_avg_vpd.nc","r")
#    ndata = float(ncdf_file.variables['vpd'].missing_value)
#    to_geotiff(ncdf_file,"vpd", "/projects/daymet2/nasa_debug/topomet_avg_vpd.tif",nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
#    
#    ncdf_file = Dataset("/projects/daymet2/nasa_debug/topomet_avg_srad.nc","r")
#    ndata = float(ncdf_file.variables['srad'].missing_value)
#    to_geotiff(ncdf_file,"srad", "/projects/daymet2/nasa_debug/topomet_avg_srad.tif",nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
    
    
#    ncdf_file = Dataset("/projects/daymet2/interp_output/oneshot_tst/output_smry/topomet_avg_prcp.nc","r")
#    ndata = float(ncdf_file.variables['prcp'].missing_value)
#    to_geotiff(ncdf_file,"prcp", "/projects/daymet2/interp_output/oneshot_tst/output_smry/topomet_avg_prcp.tif",nodata_val=ndata,dtype=gdalconst.GDT_Float32,proj=PROJ_GEO_NAD83)
#    
