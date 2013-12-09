'''
Created on Feb 3, 2012

@author: jared.oyler
'''

from pyhdf.SD import SD, SDC
import numpy as np
from to_ncdf import modis_et_dataset
from twx.utils.util_ncdf import ncdf_raster
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from twx.utils.status_check import status_check
import sys

#NetCDF4 chunk caching settings
CHK_CACHE_SIZE = 1095750000*5
CHK_CACHE_ELEMS = 1009
CHK_CACHE_PREEMP = 0.75

#The time series index at which to start reading data and
#the count of how many days to read.
TIME_START_INDEX = 18262 #Index to start reading on 20000101
TIME_COUNT = 3653 #The total of days to read (20000101 - 20091231)

#Default file paths to topomet netcdf datasets
PATH_MASK= "/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc"
PATH_TOPOMET_DATA= "/projects/daymet2/interp_output/fullrun20110916/"
FNAME_TMIN="topomet_tmin.nc"
FNAME_TMAX= "topomet_tmax.nc"
FNAME_PRCP= "topomet_prcp.nc"
FNAME_SRAD= "topomet_srad.nc"
FNAME_VPD= "topomet_vpd.nc"

DATAFIELD_ET_1KM = 'ET_1km'

def test_merra_crown():
    
    ds_merra = Dataset('/projects/daymet2/modis/crown_merra.nc')
    tmin = ds_merra.variables['vpd'][:,:,:]
    
    plt.imshow(np.mean(tmin,axis=0))
    plt.show()

if __name__ == '__main__':
    
    test_merra_crown()

    #ds_et = modis_et_dataset('/projects/daymet2/modis/MOD16A2.A2009177.h10v04.105.2010355183415.hdf')
    #x_sin,y_sin = ds_et.geo_coord_grids()
    
#    plt.imshow(ds_et.et_a)
#    plt.show()
    
    
#    tile_mask = np.zeros(ds_et.et_a.shape)
    #trans_pt = modis_sin_latlon_transform()
#    
#    ds_mask = ncdf_raster("/projects/daymet2/dem/smoothed/ncdf/interp_mask_crown.nc","mask")
##    mask = ds_mask.vals
##    nonzero_rows,nonzero_cols = np.nonzero(mask)
##    nonzero_rows,nonzero_cols = np.unique(nonzero_rows),np.unique(nonzero_cols)
##    min_row,max_row = np.min(nonzero_rows),np.max(nonzero_rows)
##    min_col,max_col = np.min(nonzero_cols),np.max(nonzero_cols)
##    
##    for r in np.arange(ds_et.et_a.shape[0]):
##        
##        for c in np.arange(ds_et.et_a.shape[1]):
##            
##            x_geo,y_geo = ds_et.geo_location(c,r)
##            lon,lat,z = trans_pt.trans_sin_to_wgs84.TransformPoint(x_geo,y_geo)
##            try:
##                col,row = ds_mask.getGridCellOffset(lon, lat)
##                
##                if row >= min_row and row <= max_row and col >= min_col and col <= max_col:
##                    tile_mask[r,c] =  1
##            except:
##                pass
#    #np.save('/projects/daymet2/modis/crown_mask_h10v04.npy',tile_mask)
#    #print x_sin[0,0],y_sin[0,0]
#
#    tile_mask = np.load('/projects/daymet2/modis/crown_mask_h10v04.npy')
#    
#
#
#    tile_mask2 = np.zeros_like(tile_mask)
#    nonzero_rows,nonzero_cols = np.nonzero(tile_mask)
#    nonzero_rows,nonzero_cols = np.unique(nonzero_rows),np.unique(nonzero_cols)
#
#    tile_mask2[nonzero_rows[0]:nonzero_rows[-1]+1,nonzero_cols[0]:nonzero_cols[-1]+1] = 1
#
#    plt.imshow(tile_mask2)
#    plt.show()
#   
#    print nonzero_rows[0],nonzero_rows[-1]
#    print nonzero_cols[0],nonzero_cols[-1]
#    sys.exit()
#
#
#    x_geo_ul,y_geo_ul = ds_et.geo_location(nonzero_cols[0],nonzero_rows[0])
#    x_geo_lr,y_geo_lr = ds_et.geo_location(nonzero_cols[-1],nonzero_rows[-1])
#    #print nonzero_rows.size
#    #print nonzero_cols.size
#    
#    #plt.imshow(tile_mask)
#    #plt.show()
#    
#    '''
#    Create GDAL geotransform list to define resolution and bounds
#    GeoTransform[0] /* top left x */
#    GeoTransform[1] /* w-e pixel resolution */
#    GeoTransform[2] /* rotation, 0 if image is "north up" */
#    GeoTransform[3] /* top left y */
#    GeoTransform[4] /* rotation, 0 if image is "north up" */
#    GeoTransform[5] /* n-s pixel resolution */
#    '''
#    geotransform = [None]*6
#    geotransform[0] = x_geo_ul - np.abs(ds_et.geotransform[1]/2.0)
#    geotransform[1] = ds_et.geotransform[1]
#    geotransform[2] = ds_et.geotransform[2]
#    geotransform[3] = y_geo_ul + np.abs(ds_et.geotransform[5]/2.0)
#    geotransform[4] = ds_et.geotransform[4]
#    geotransform[5] = ds_et.geotransform[5]
#    
#    print geotransform
#    print "upper left, lower right coords"
#    print geotransform[0],geotransform[3]
#    print x_geo_lr + np.abs(ds_et.geotransform[1]/2.0), y_geo_lr - np.abs(ds_et.geotransform[5]/2.0)
#    print nonzero_rows[0],nonzero_rows[-1]
#    print nonzero_cols[0],nonzero_cols[-1]
#    
#    TIME_COUNT_HALF = TIME_COUNT/2
#    print TIME_COUNT_HALF
#    print TIME_COUNT-TIME_COUNT_HALF
#    
#    rcgrid = np.mgrid[nonzero_rows[0]:nonzero_rows[-1]+1,nonzero_cols[0]:nonzero_cols[-1]+1]
#    rgrid = rcgrid[0]
#    cgrid = rcgrid[1]
#    
#    tmax = np.zeros((TIME_COUNT,rgrid.shape[0],rgrid.shape[1]),dtype=np.float32)
#    ds_tmax = Dataset("".join([PATH_TOPOMET_DATA,FNAME_TMAX]))
#    var_tmax = ds_tmax.variables['tmax']
#    var_tmax.set_var_chunk_cache(CHK_CACHE_SIZE,CHK_CACHE_ELEMS,CHK_CACHE_PREEMP)
#    
#    schk = status_check(nonzero_rows.size*nonzero_cols.size,1000)
#    for r in np.arange(nonzero_rows.size):
#        
#        for c in np.arange(nonzero_cols.size):
#    
#            x_geo,y_geo = ds_et.geo_location(cgrid[r,c], rgrid[r,c])
#            lon,lat,z = trans_pt.trans_sin_to_wgs84.TransformPoint(x_geo,y_geo)
#            col,row = ds_mask.getGridCellOffset(lon, lat)
#            tmax[:,r,c] = var_tmax[TIME_START_INDEX:,row,col]
#            schk.increment()
##            plt.plot(tmax[:,r,c])
##            plt.show()
#        print r
#    
#    
#    
##    sd1 = SD("/projects/daymet2/modis/topomet_hdf4/tmax1.hdf", SDC.READ)
##    sds = sd1.select("tmax")
##    tmax = sds[:]
##    print tmax.shape
##    sys.exit()
#    
#    sd1 = SD("/projects/daymet2/modis/topomet_hdf4/tmax1.hdf", SDC.WRITE | SDC.CREATE)
#    sd2 = SD("/projects/daymet2/modis/topomet_hdf4/tmax2.hdf", SDC.WRITE | SDC.CREATE)
#    
#    sds1 = sd1.create("tmax", SDC.FLOAT32,(TIME_COUNT_HALF,rgrid.shape[0],rgrid.shape[1]))
#    sds2 = sd2.create("tmax", SDC.FLOAT32,(TIME_COUNT-TIME_COUNT_HALF,rgrid.shape[0],rgrid.shape[1]))
#    
#    #Set dimension names
#    dim0 = sds1.dim(0)
#    dim0.setname("time")
#    dim1 = sds1.dim(1)
#    dim1.setname("row")
#    dim2 = sds1.dim(2)
#    dim2.setname("col")
#    ##################
#    dim0 = sds2.dim(0)
#    dim0.setname("time")
#    dim1 = sds2.dim(1)
#    dim1.setname("row")
#    dim2 = sds2.dim(2)
#    dim2.setname("col")
#    
#    sds1.setfillvalue(0)
#
#    sds1[:] = tmax[0:TIME_COUNT_HALF,:,:]
#    sds1.endaccess()
#    sd1.end()
#
#    sds2[:] = tmax[TIME_COUNT_HALF:,:,:]
#    sds2.endaccess()
#    sd2.end()
         
#    mask = ds_mask.vals
#    nonzero_rows,nonzero_cols = np.nonzero(mask)
#    
#    print np.unique(nonzero_rows).size,np.unique(nonzero_cols).size
    
