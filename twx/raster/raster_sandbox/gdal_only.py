'''
Created on Apr 22, 2013

@author: jared.oyler
'''
from osgeo import gdal,gdalconst,osr,ogr
import numpy as np
from osgeo.gdal import FillNodata
import matplotlib.pyplot as plt
from numpy.ma.core import masked_less

def resample_modis_sinu_to_grid(fpath_dstgrid,fpath_modis_sinu_grid,fpath_out,resample_alg=gdalconst.GRA_Bilinear):
    grid_dst = gdal.Open(fpath_dstgrid)
    grid_src = gdal.Open(fpath_modis_sinu_grid)

    src_proj = "+proj=sinu +R=6371007.181 +nadgrids=@null +no_defs +wktext"
    #src_proj = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    sr_sin = osr.SpatialReference()
    sr_sin.ImportFromProj4(src_proj)
    src_proj = sr_sin.ExportToWkt()
    src_dtype = grid_src.GetRasterBand(1).DataType
    src_ndata =  grid_src.GetRasterBand(1).GetNoDataValue()
    
    dst_proj = grid_dst.GetProjection()
    dst_geot = grid_dst.GetGeoTransform()
    
    grid_out = gdal.GetDriverByName('GTiff').Create(fpath_out, grid_dst.RasterXSize,
                                                    grid_dst.RasterYSize, 1, src_dtype)
    band = grid_out.GetRasterBand(1)
    band.Fill(src_ndata)
    band.SetNoDataValue(src_ndata)
    
    grid_out.SetGeoTransform(dst_geot)
    grid_out.SetProjection(dst_proj)
    
    gdal.ReprojectImage(grid_src, grid_out, src_proj, dst_proj, resample_alg)
    grid_out.FlushCache()


def ascii_to_geotiff(fpath_ascii,fpath_out):
    
    targetSR = osr.SpatialReference()
    targetSR.ImportFromEPSG(4326)
    targetSR = targetSR.ExportToWkt()
    
    print "Opening ASCII file and reading data..."
    ds = gdal.Open(fpath_ascii)
    b = ds.GetRasterBand(1)
    a = b.ReadAsArray()
    geoT = ds.GetGeoTransform()
    ndata = b.GetNoDataValue()
    
    dsout = gdal.GetDriverByName('GTiff').Create(fpath_out, ds.RasterXSize,
                                                    ds.RasterYSize, 1, gdalconst.GDT_Int32)
    
    bout = dsout.GetRasterBand(1)
    bout.SetNoDataValue(ndata)
    
    dsout.SetGeoTransform(geoT)
    dsout.SetProjection(targetSR)
    
    print "Writing data to GTiff"
    bout.WriteArray(a,0,0)  
    bout.FlushCache()
    print "Done"

def interpWaterLst():

    dsMask = gdal.Open('/projects/daymet2/climate_office/modis/MOD44B/mosaic_vcf_mask2.tif')
    maskWater = dsMask.ReadAsArray()
    
    dsLst = gdal.Open('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Night_1km.08.C.tif')
    bndLst = dsLst.GetRasterBand(1)
    lstNoData = bndLst.GetNoDataValue()
    lst = dsLst.ReadAsArray()
    lstMax = np.max(lst[lst!=lstNoData])
    
    lst = np.roll(lst,1, axis=0)
    lst = np.roll(lst,-1,axis=1)
    
    lst[0,:] = lstNoData
    lst[:,-1] = lstNoData
    
    maskFnl = ~np.logical_and(lst != lstNoData, maskWater != 1)
    #maskFnl = maskWater

    #Create the lst band in memory
    mem_drv = gdal.GetDriverByName( 'GTiff' )
    dsLstMem = mem_drv.Create('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/[test]MOSAIC.LST_Night_1km.08.C.shift.tif',lst.shape[1],lst.shape[0],1,gdalconst.GDT_Float32)
    bndLstMem = dsLstMem.GetRasterBand(1)
    bndLstMem.SetNoDataValue(lstNoData)
    dsLstMem.SetGeoTransform(dsLst.GetGeoTransform())
    dsLstMem.SetProjection(dsLst.GetProjection())
    bndLstMem.WriteArray(lst,0,0)
    bndLstMem.FlushCache()
    
    #Create the mask band in memory
    mem_drv = gdal.GetDriverByName( 'MEM' )
    dsMaskMem = mem_drv.Create('',lst.shape[1],lst.shape[0],1,gdalconst.GDT_Int32)
    bndMaskMem = dsMaskMem.GetRasterBand(1)
    dsMaskMem.SetGeoTransform(dsLst.GetGeoTransform())
    dsMaskMem.SetProjection(dsLst.GetProjection())
    bndMaskMem.SetNoDataValue(lstNoData)
    bndMaskMem.WriteArray(maskFnl.astype(np.int16),0,0)
    bndMaskMem.FlushCache()
    
    gdal.FillNodata(bndLstMem,bndMaskMem,100,1)
    
    lstFilled = bndLstMem.ReadAsArray()
    #lstFilled = np.ma.masked_equal(lstFilled, lstNoData)
    lstFilled = np.ma.masked_greater(lstFilled, lstMax)
    
    plt.imshow(lstFilled)
    plt.colorbar()
    plt.show()
 
if __name__ == '__main__':
    interpWaterLst()
    #ascii_to_geotiff('/net/tops/home/jared.oyler/nima/soil.asc', '/net/tops/home/jared.oyler/nima/soil.tif')
    #ascii_to_geotiff('/net/tops/home/jared.oyler/nima/bio12.asc', '/net/tops/home/jared.oyler/nima/bio12.tif')
    #ascii_to_geotiff('/net/tops/home/jared.oyler/nima/bio1.asc', '/net/tops/home/jared.oyler/nima/bio1.tif')
    #ascii_to_geotiff('/net/tops/home/jared.oyler/nima/biome.asc', '/net/tops/home/jared.oyler/nima/biome.tif')
    #ascii_to_geotiff('/net/tops/home/jared.oyler/nima/ecoregion.asc', '/net/tops/home/jared.oyler/nima/ecoregion.tif')
    #ascii_to_geotiff('/net/tops/home/jared.oyler/nima/elev.asc', '/net/tops/home/jared.oyler/nima/elev.tif')
#    resample_modis_sinu_to_grid('/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.MRTTest.LST_Night_1km.tif', 
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.tif', 
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.GDALTest2.tif',
#                                gdalconst.GRA_NearestNeighbour)
    
#    resample_modis_sinu_to_grid('/projects/daymet2/climate_office/modis/resample_test/MOD15A2.A2013049.h09v04.005.2013058070810.nn.Lai_1km.tif', 
#                                '/projects/daymet2/climate_office/modis/resample_test/lai.gdal.int.tif', 
#                                '/projects/daymet2/climate_office/modis/resample_test/lai.gdal.resample.nn.tif',
#                                gdalconst.GRA_NearestNeighbour)
    
#    resample_modis_sinu_to_grid('/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.MRT.AEA.LST_Night_1km.tif', 
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.tif', 
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.GDALTest.AEA.tif',
#                                gdalconst.GRA_Bilinear)