from netCDF4 import Dataset
from osgeo import gdal,gdalconst,osr
import numpy as np
import os
from modis.clip_raster import resample_to_grd1,mask_to_rastmask

PROJ_GEO_WGS84 = 4326
CCE_TILES_TWX = ['h04v01','h05v01','h06v01','h05v02','h06v02']#16,17,18,37,38

def topoWxTileToTiff(ds,a,pathOut):
        
    nrow = len(ds.dimensions['lat'])
    ncol = len(ds.dimensions['lon'])
            
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(pathOut,ncol,nrow,1,gdalconst.GDT_Float64) 
    
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
    band.SetNoDataValue(float(a.fill_value))
    band.WriteArray(np.ma.filled(a),0,0) 
    band.FlushCache()
    
def getMaskedArray(ds,varName):
    
    fval = ds.variables[varName]._FillValue
    
    a = ds.variables[varName][:]

    if np.ma.isMaskedArray(a):
        np.ma.set_fill_value(a, fval)
    else:
        a = np.ma.masked_array(a)
        np.ma.set_fill_value(a, fval)
        
    return a

def mosaic(in_files,out_mosaic,ndata=-9999):

    cmd = ["".join(["gdalwarp -dstnodata ",str(ndata)])]
    cmd.extend(in_files)
    cmd.append(out_mosaic)
    
    print "running mosaic cmd..."+" ".join(cmd)
    os.system(" ".join(cmd))


def mainTopoWxTileToTiff():
    tilePath = '/stage/climate/test_tile_output/'
    pathOut = '/projects/daymet2/cce_case_study/topowx_files/normals/'
    
    for tileName in CCE_TILES_TWX:
        
        tminDs = Dataset("".join([tilePath,tileName,"/",tileName,"_tmin.nc"]))
        tmaxDs = Dataset("".join([tilePath,tileName,"/",tileName,"_tmax.nc"]))
        
        tminNorms = getMaskedArray(tminDs,'tmin_normal')
        tmaxNorms = getMaskedArray(tmaxDs,'tmax_normal')
        tminSe = getMaskedArray(tminDs,'tmin_se')
        tmaxSe = getMaskedArray(tmaxDs,'tmax_se')
        
        for mth in np.arange(1,13):
            topoWxTileToTiff(tminDs, tminNorms[mth-1,:,:], "".join([pathOut,"tmin_normal_%02d_%s.tif"%(mth,tileName)]))
            topoWxTileToTiff(tmaxDs, tmaxNorms[mth-1,:,:], "".join([pathOut,"tmax_normal_%02d_%s.tif"%(mth,tileName)]))
            topoWxTileToTiff(tminDs, tminSe[mth-1,:,:], "".join([pathOut,"tmin_se_%02d_%s.tif"%(mth,tileName)]))
            topoWxTileToTiff(tmaxDs, tmaxSe[mth-1,:,:], "".join([pathOut,"tmax_se_%02d_%s.tif"%(mth,tileName)]))

def mainMosaic():
    tilePath = '/projects/daymet2/cce_case_study/topowx_files/normals/'
    pathOut = '/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/'
    
    os.chdir(tilePath)
    
    for mth in np.arange(1,13):
        
        fnames = ["tmin_normal_%02d_%s.tif"%(mth,tileName) for tileName in CCE_TILES_TWX]
        mosaic(fnames,"".join([pathOut,"/tmin_normal_%02d.tif"%mth]))
        
        fnames = ["tmax_normal_%02d_%s.tif"%(mth,tileName) for tileName in CCE_TILES_TWX]
        mosaic(fnames,"".join([pathOut,"/tmax_normal_%02d.tif"%mth]))
        
        fnames = ["tmin_se_%02d_%s.tif"%(mth,tileName) for tileName in CCE_TILES_TWX]
        mosaic(fnames,"".join([pathOut,"/tmin_se_%02d.tif"%mth]))
        
        fnames = ["tmax_se_%02d_%s.tif"%(mth,tileName) for tileName in CCE_TILES_TWX]
        mosaic(fnames,"".join([pathOut,"/tmax_se_%02d.tif"%mth]))

def mainResampleCrop():
    PATH_CCE_MASK = '/projects/daymet2/dem/interp_grids/cce/crp_cce_us_mask.tif'
    mosaics_path = '/projects/daymet2/cce_case_study/topowx_files/normals/mosaics/'

    
    for mth in np.arange(1,13):
        
        #########################
        resample_to_grd1(PATH_CCE_MASK, 
                         "".join([mosaics_path,"tmin_normal_%02d.tif"%mth]), 
                         "".join([mosaics_path,"temp.tif"]),
                         gdalconst.GRA_NearestNeighbour)
        mask_to_rastmask(PATH_CCE_MASK, 
                         "".join([mosaics_path,'temp.tif']), 
                         "".join([mosaics_path,"cce_tmin_normal_%02d.tif"%mth]))
        
        ##########################
        resample_to_grd1(PATH_CCE_MASK, 
                         "".join([mosaics_path,"tmax_normal_%02d.tif"%mth]), 
                         "".join([mosaics_path,"temp.tif"]),
                         gdalconst.GRA_NearestNeighbour)
        mask_to_rastmask(PATH_CCE_MASK, 
                         "".join([mosaics_path,'temp.tif']), 
                         "".join([mosaics_path,"cce_tmax_normal_%02d.tif"%mth]))
        
        #########################
        resample_to_grd1(PATH_CCE_MASK, 
                         "".join([mosaics_path,"tmin_se_%02d.tif"%mth]), 
                         "".join([mosaics_path,"temp.tif"]),
                         gdalconst.GRA_NearestNeighbour)
        mask_to_rastmask(PATH_CCE_MASK, 
                         "".join([mosaics_path,'temp.tif']), 
                         "".join([mosaics_path,"cce_tmin_se_%02d.tif"%mth]))
        
        ##########################
        resample_to_grd1(PATH_CCE_MASK, 
                         "".join([mosaics_path,"tmax_se_%02d.tif"%mth]), 
                         "".join([mosaics_path,"temp.tif"]),
                         gdalconst.GRA_NearestNeighbour)
        mask_to_rastmask(PATH_CCE_MASK, 
                         "".join([mosaics_path,'temp.tif']), 
                         "".join([mosaics_path,"cce_tmax_se_%02d.tif"%mth]))
    
    os.system("".join(['rm ',mosaics_path,'temp.tif']))

if __name__ == '__main__':
    #mainTopoWxTileToTiff()
    #mainMosaic()
    mainResampleCrop()

        
         
