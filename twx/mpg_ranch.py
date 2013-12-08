'''
Created on Nov 15, 2013

@author: jared.oyler
'''
from DatasetCompare import topoWxTileToTiff
from netCDF4 import Dataset,num2date
import numpy as np
from modis.clip_raster import mask_to_shp, crop_nodata
from osgeo import gdal
import utils.util_dates as utld

def nc_to_tiff():
    dsPath = '/stage/climate/topowx_tile_output/h05v02/h05v02_tmin.nc'
    ds = Dataset(dsPath)
    norms = ds.variables['tmin_se'][:]
    norms = np.ma.array(norms)
    norms.fill_value = ds.variables['tmin_se']._FillValue
    topoWxTileToTiff(dsPath, norms, '/projects/daymet2/mpg_ranch/topowx/se_tmin.tif')


def avgDlyVals():
    ds = gdal.Open('/projects/daymet2/mpg_ranch/topowx/crop_norms_tmin.tif')
    b = ds.GetRasterBand(1)
    a = b.ReadAsArray()
    nodata = b.GetNoDataValue()
    mask = a != nodata
    
    print "TMIN"
    ds = Dataset('/stage/climate/topowx_tile_output/h05v02/h05v02_tmin.nc')
    dates = num2date(ds.variables['time'][:], ds.variables['time'].units)
    ymd = utld.get_ymd_array(dates)
    
    tmin = ds.variables['tmin'][:]
    tmin = tmin[:,mask]
    tmin = np.mean(tmin,axis=1,dtype=np.float)
    ds.close()
    
    print "TMAX"
    ds = Dataset('/stage/climate/topowx_tile_output/h05v02/h05v02_tmax.nc')
    tmax = ds.variables['tmax'][:]
    tmax = tmax[:,mask]
    tmax = np.mean(tmax,axis=1,dtype=np.float)
    ds.close()
    
    fout = open("/projects/daymet2/mpg_ranch/mgh_dly_19482012.csv",'w')
    fout.write("YMD,TMIN,TMAX\n")
    
    for x in np.arange(ymd.size):     
        fout.write("%d,%0.2f,%0.2f\n"%(ymd[x],tmin[x],tmax[x]))
    fout.close()
    
    
    
    
if __name__ == '__main__':

    avgDlyVals()

#    nc_to_tiff()
#    mask_to_shp('/projects/daymet2/mpg_ranch/mpgShp/ranch_boundary_Apr2012_wgs84buf2.shp',
#                'ranch_boundary_Apr2012_wgs84buf2', '/projects/daymet2/mpg_ranch/topowx/se_tmin.tif', 
#                '/projects/daymet2/mpg_ranch/topowx/crop_se_tmin.tif')
    
#    crop_nodata('/projects/daymet2/mpg_ranch/topowx/crop_norms_tmin.tif',
#                '/projects/daymet2/mpg_ranch/topowx/trim_norms_tmin.tif')

