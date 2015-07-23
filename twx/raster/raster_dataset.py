'''
Utilities for working with basic raster datasets loaded
through GDAL.

Copyright 2014,2015, Jared Oyler.

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

import osgeo.gdal as gdal
import osgeo.osr as osr
import numpy as np

PROJECTION_GEO_WGS84 = 4326  # EPSG Code
PROJECTION_GEO_NAD83 = 4269  # EPSG Code
PROJECTION_GEO_WGS72 = 4322  # EPSG Code

class OutsideExtent(Exception):
    pass

class RasterDataset(object):
    '''
    Encapsulates a basic 2-D GDAL raster object with
    common utility functions.
    '''

    def __init__(self, ds_path):
        '''
        Parameters
        ----------
        ds_path : str
            A file path to a raster dataset (eg GeoTIFF file)
        '''

        self.gdal_ds = gdal.Open(ds_path)
        self.ds_path = ds_path
        # GDAL GeoTransform.
        # Top left x,y are for the upper left corner of upper left pixel
        # GeoTransform[0] /* top left x */
        # GeoTransform[1] /* w-e pixel resolution */
        # GeoTransform[2] /* rotation, 0 if image is "north up" */
        # GeoTransform[3] /* top left y */
        # GeoTransform[4] /* rotation, 0 if image is "north up" */
        # GeoTransform[5] /* n-s pixel resolution */
        self.geo_t = np.array(self.gdal_ds.GetGeoTransform())

        self.projection = self.gdal_ds.GetProjection()
        self.source_sr = osr.SpatialReference()
        self.source_sr.ImportFromWkt(self.projection)
        self.target_sr = osr.SpatialReference()
        self.target_sr.ImportFromEPSG(PROJECTION_GEO_WGS84)
        self.coordTrans_src_to_wgs84 = osr.CoordinateTransformation(self.source_sr, self.target_sr)
        self.coordTrans_wgs84_to_src = osr.CoordinateTransformation(self.target_sr, self.source_sr)

        self.min_x = self.geo_t[0]
        self.max_x = self.min_x + (self.gdal_ds.RasterXSize * self.geo_t[1])
        self.max_y = self.geo_t[3]
        self.min_y = self.max_y - (-self.gdal_ds.RasterYSize * self.geo_t[5])

        self.rows = self.gdal_ds.RasterYSize
        self.cols = self.gdal_ds.RasterXSize
        self.ndata = self.gdal_ds.GetRasterBand(1).GetNoDataValue()


    def get_coord_mesh_grid(self):
        '''
        Build a native projection coordinate mesh grid for the raster
        
        Returns
        ----------
        yGrid : ndarray
            A 2-D ndarray with the same shape as the raster containing
            the y-coordinate of the center of each grid cell.
        xGrid : ndarray
            A 2-D ndarray with the same shape as the raster containing
            the x-coordinate of the center of each grid cell.
        '''

        # Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geo_t[0] + (self.geo_t[1] / 2.0)
        urX = self.get_coord(0, self.gdal_ds.RasterXSize - 1)[1]

        # Get the upper left and lower left y coordinates
        ulY = self.geo_t[3] + (self.geo_t[5] / 2.0)
        llY = self.get_coord(self.gdal_ds.RasterYSize - 1, 0)[0]

        # Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdal_ds.RasterXSize)
        y = np.linspace(ulY, llY, self.gdal_ds.RasterYSize)

        xGrid, yGrid = np.meshgrid(x, y)

        return yGrid, xGrid

    def get_coord_grid_1d(self):
        '''
        Build 1-D native projection coordinates for y and x dimensions
        of the raster.
        
        Returns
        ----------
        y : ndarray
            A 1-D array of the y-coordinates of each row.
        x : ndarray
            A 1-D array of the x-coordinates of each column.
        '''

        # Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geo_t[0] + (self.geo_t[1] / 2.0)
        urX = self.get_coord(0, self.gdal_ds.RasterXSize - 1)[1]

        # Get the upper left and lower left y coordinates
        ulY = self.geo_t[3] + (self.geo_t[5] / 2.0)
        llY = self.get_coord(self.gdal_ds.RasterYSize - 1, 0)[0]

        # Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdal_ds.RasterXSize)
        y = np.linspace(ulY, llY, self.gdal_ds.RasterYSize)

        return y, x

    def get_coord(self, row, col):
        '''
        Get the native projection coordinates for a specific grid cell
        
        Parameters
        ----------
        row : int
            The row of the grid cell (zero-based)
        col : int
            The column of the grid cell (zero-based)
            
        Returns
        ----------
        yCoord : float
            The y projection coordinate of the grid cell.
        xCoord : float
            The x projection coordinate of the grid cell.   
        '''

        xCoord = (self.geo_t[0] + col * self.geo_t[1] + row * self.geo_t[2]) + self.geo_t[1] / 2.0
        yCoord = (self.geo_t[3] + col * self.geo_t[4] + row * self.geo_t[5]) + self.geo_t[5] / 2.0

        return yCoord, xCoord

    def get_row_col(self, lon, lat, check_bounds=True):
        '''
        Get the row, column grid cell offset for the raster based on an input
        WGS84 longitude, latitude point.
        
        Parameters
        ----------
        lon : float
            The longitude of the point
        lat : float
            The latitude of the point
        check_bounds : bool
            If True, will check if the point is within the bounds
            of the raster and raise a ValueError if not. If set to False and
            point is outside raster, the returned row, col will be clipped to
            the raster edge. 
        Returns
        ----------
        row : int
            The row of the closet grid cell to the lon,lat point (zero-based)
        col : int
            The column of the closet grid cell to the lon,lat point (zero-based) 
        '''

        xGeo, yGeo, zGeo = self.coordTrans_wgs84_to_src.TransformPoint(lon, lat)
        
        if check_bounds:

            if not self.is_inbounds(xGeo, yGeo):
                raise ValueError("lat/lon outside raster bounds: " + str(lat) + "," + str(lon))

        originX = self.geo_t[0]
        originY = self.geo_t[3]
        pixelWidth = self.geo_t[1]
        pixelHeight = self.geo_t[5]

        xOffset = np.abs(np.int((xGeo - originX) / pixelWidth))
        yOffset = np.abs(np.int((yGeo - originY) / pixelHeight))
        
        # clip row,col if outside raster bounds
        row = self._check_cellxy_valid(int(yOffset), self.rows)
        col = self._check_cellxy_valid(int(xOffset), self.cols)
            
        return row, col

    def get_data_value(self, lon, lat):
        '''
        Get the nearest grid cell data value to an input
        WGS84 longitude, latitude point. Will raise an 
        OutsideExtent exception if the longitude, latitude 
        is outside the bounds of the raster.
        
        Parameters
        ----------
        lon : float
            The longitude of the point
        lat : float
            The latitude of the point
            
        Returns
        ----------
        data_val : dtype of raster
            The data value of the closet grid cell to the lon,lat point         
        '''

        row, col = self.get_row_col(lon, lat)
        data_val = self.gdal_ds.ReadAsArray(col, row, 1, 1)[0, 0]

        return data_val

    def read_as_array(self):
        '''
        Read in the entire raster as a 2-D array
        
        Returns
        ----------
        a : MaskedArray
            A 2-D MaskedArray of the raster data. No data values
            are masked.      
        '''

        a = self.gdal_ds.GetRasterBand(1).ReadAsArray()
        a = np.ma.masked_equal(a, self.gdal_ds.GetRasterBand(1).GetNoDataValue())
        return a
    
    def output_new_ds(self, fpath, a, gdal_dtype, ndata=None, gdal_driver="GTiff"):
        '''
        Output a new raster with same geotransform and projection as this RasterDataset
        
        Parameters
        ----------
        fpath : str
            The filepath for the new raster.
        a : ndarray or MaskedArray
            A 2-D array of the raster data to be output.
            If MaskedArray, masked values are set to ndata
        gdal_dtype : str
            A gdal datatype from gdalconst.GDT_*.
        ndata : num, optional
            The no data value for the raster
        gdal_driver : str, optional
            The GDAL driver for the output raster data format
        '''
        
        ds_out = gdal.GetDriverByName(gdal_driver).Create(fpath, int(a.shape[1]), int(a.shape[0]), 1, gdal_dtype)
        ds_out.SetGeoTransform(self.gdal_ds.GetGeoTransform())
        ds_out.SetProjection(self.gdal_ds.GetProjection())
        
        band_out = ds_out.GetRasterBand(1)
        if ndata is not None:
            band_out.SetNoDataValue(ndata)
        band_out.WriteArray(np.ma.filled(a, ndata))
                    
        ds_out.FlushCache()
        ds_out = None

    def resample_to_ds(self, fpath, ds_src, gdal_gra, gdal_driver="GTiff"):
        '''
        Resample a different RasterDataset to this RasterDataset's grid
        
        Parameters
        ----------
        fpath : str
            The filepath for the new resampled raster.
        ds_src : RasterDataset
            The RasterDataset to  be resampled
        gdal_gra : str
            A gdal resampling algorithm from gdalconst.GRA_*.
        gdal_driver : str, optional
            The GDAL driver for the output raster data format
            
        Returns
        ----------
        grid_out : RasterDataset
            A RasterDataset pointing to the resampled raster  
        '''
        
        grid_src = ds_src.gdal_ds
        grid_dst = self.gdal_ds
        
        proj_src = grid_src.GetProjection()
        dtype_src = grid_src.GetRasterBand(1).DataType
        ndata_src = grid_src.GetRasterBand(1).GetNoDataValue()
        
        proj_dst = grid_dst.GetProjection()
        geot_dst = grid_dst.GetGeoTransform()
            
        grid_out = gdal.GetDriverByName(gdal_driver).Create(fpath, grid_dst.RasterXSize,
                                                        grid_dst.RasterYSize, 1, dtype_src)
        
        if ndata_src is not None:
            band = grid_out.GetRasterBand(1)
            band.Fill(ndata_src)
            band.SetNoDataValue(ndata_src)
        
        grid_out.SetGeoTransform(geot_dst)
        grid_out.SetProjection(proj_dst)
        grid_out.FlushCache()
        
        gdal.ReprojectImage(grid_src, grid_out, proj_src, proj_dst, gdal_gra)
        grid_out.FlushCache()
        # Make sure entire grid is written by setting to None. 
        # FlushCache doesn't seem to write the entire grid after resampling?
        grid_out = None
        
        # return as RasterDataset
        grid_out = RasterDataset(fpath)
        
        return grid_out
        
    def is_inbounds(self, x_geo, y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y
    
    def _check_cellxy_valid(self, i, n):
        
        if i < 0:
            i = 0
        elif i >= n:
            i = n - 1
        
        return i
        
        
