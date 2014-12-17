'''
Created on Nov 29, 2012

@author: jared.oyler
'''
import numpy as np
import os
from osgeo import gdal,gdalconst,osr,ogr
from twx.utils.input_raster import input_raster,RasterDataset
from twx.utils.util_ncdf import to_ncdf,expand_grid
import twx.utils.util_geo as utlg
from netCDF4 import Dataset
from twx.utils.output_raster import output_raster
from twx.utils.status_check import StatusCheck
import matplotlib.pyplot as plt
import scipy.interpolate as si

def create_bounds_poly(lwrr,uprl,out_path,layer_name):
    
    lwrl =  (uprl[0],lwrr[1])
    uprr = (lwrr[0],uprl[1])
    
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(uprl[0],uprl[1])
    ring.AddPoint(uprr[0],uprr[1])
    ring.AddPoint(lwrr[0],lwrr[1])
    ring.AddPoint(lwrl[0],lwrl[1])
    ring.CloseRings()
    
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    fpath = "".join([out_path,layer_name,".shp"])
    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    ds = driver.CreateDataSource(fpath)
    layer = ds.CreateLayer(layer_name, geom_type=ogr.wkbPolygon)
    
    # add an id field to the output
    fieldDefn = ogr.FieldDefn('id', ogr.OFTInteger)
    layer.CreateField(fieldDefn)
    
    # get the FeatureDefn for the output layer
    featureDefn = layer.GetLayerDefn()
    
    # create a new feature
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(poly)
    feature.SetField('id', 1)
    
    # add the feature to the output layer
    layer.CreateFeature(feature)
    
    ring.Destroy()
    poly.Destroy()
    feature.Destroy()
    ds.Destroy()
    
    sr = osr.SpatialReference() 
    #sr.ImportFromEPSG(4326) #WGS84
    sr.ImportFromEPSG(4269) #NAD83
    sr.MorphToESRI()
    
    afile = open("".join([out_path,layer_name,".prj"]), 'w')
    afile.write(sr.ExportToWkt())
    afile.close()

def clip_rasters():
    pathout = '/projects/daymet2/climate_office/modis/MOD13A3/jja_mt_tifs/'
    path = '/projects/daymet2/climate_office/modis/MOD13A3/jja_tifs_wgs84/'
    os.chdir(path)
    fnames = np.array(os.listdir(path))
    fnames = fnames[np.char.endswith(fnames, ".tif")]
    fnames = fnames[np.char.find(fnames, "mean") != -1]
    
    
    for fname in fnames:
        
        cmd = 'gdalwarp -cutline "/projects/crown_ws/montanaboundary_GIS/states_WGS84.shp" -cl states_WGS84 -crop_to_cutline -dstnodata -9999 '
        cmd = cmd + fname + " " + "".join([pathout,fname])
        print cmd
        os.system(cmd)

def mask_to_shp(fpath_shp,shp_layer,fpath_in_ds,fpath_out_ds,nodata=-9999):
    
    #cmd = "".join(['gdalwarp -cutline "',fpath_shp,'" -cl ',shp_layer,' -crop_to_cutline -dstnodata -9999 ',fpath_in_ds,' ',fpath_out_ds])
    cmd = "".join(['gdalwarp -cutline "',fpath_shp,'" -cl ',shp_layer,' -dstnodata ',str(nodata),' ',fpath_in_ds,' ',fpath_out_ds])
    print cmd
    os.system(cmd)
    
def mask_to_rastmask(fpath_mask,fpath_in_ds,fpath_out_ds):
    
    ds_mask = gdal.Open(fpath_mask)    
    mask = ds_mask.ReadAsArray() == 0
    
    ds = gdal.Open(fpath_in_ds)
    geot = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ndata = ds.GetRasterBand(1).GetNoDataValue()
    dtype = ds.GetRasterBand(1).DataType
    a = ds.ReadAsArray()
    
    if len(a.shape) == 3:
        a[:,mask] = ndata
        nCols = a.shape[2]
        nRows = a.shape[1]
    else:
        a[mask] = ndata
        nCols = a.shape[1]
        nRows = a.shape[0]
        
     
        
    ds_out = gdal.GetDriverByName('GTiff').Create(  fpath_out_ds, 
                                                    nCols, 
                                                    nRows, ds.RasterCount, dtype)
    
    ds_out.SetGeoTransform(geot)
    ds_out.SetProjection(proj)
    
    if ds.RasterCount == 1:
    
        band_out = ds_out.GetRasterBand(1)
        band_out.Fill(ndata)
        band_out.SetNoDataValue(ndata)
        band_out.WriteArray(a)
        
    else:
        
        for x in np.arange(ds.RasterCount):
            
            band_out = ds_out.GetRasterBand(int(x+1))
            band_out.Fill(ndata)
            band_out.SetNoDataValue(ndata)
            band_out.WriteArray(a[x,:,:])
    
    ds_out.FlushCache()

def crop_nodata(fpath,fpath_out,data_mask=None):
    
    ds = gdal.Open(fpath)
    geot = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ndata = ds.GetRasterBand(1).GetNoDataValue()
    dtype = ds.GetRasterBand(1).DataType
    
    a = ds.ReadAsArray()

    if data_mask is None:
        ndata = np.array([ndata])
        ndata = ndata.astype(a.dtype)[0]
        mask_nd = a != ndata
    else:
        mask_nd = data_mask
        a[~data_mask] = ndata
    
    nonzero_rows,nonzero_cols = np.nonzero(mask_nd)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    nonzero_rows = np.arange(nonzero_rows[0],nonzero_rows[-1]+1)
    nonzero_cols = np.arange(nonzero_cols[0],nonzero_cols[-1]+1)
    
    min_row,max_row = nonzero_rows[0],nonzero_rows[-1]
    min_col,max_col = nonzero_cols[0],nonzero_cols[-1]
    
    a_crp = a[nonzero_rows,:]
    a_crp = a_crp[:,nonzero_cols]
    
    tl_lon,tl_lat = get_lonlat(geot, min_col, min_row)
    tl_lon = tl_lon - (geot[1]/2.0)
    tl_lat = tl_lat + np.abs(geot[5]/2.0)

    geot = list(geot)
    geot[0] = tl_lon
    geot[3] = tl_lat
    
    ds_out = gdal.GetDriverByName('GTiff').Create(  fpath_out, 
                                                    a_crp.shape[1], 
                                                    a_crp.shape[0], 1, dtype)
    
    band_out = ds_out.GetRasterBand(1)
    
    if ndata is not None:
        ndata = float(ndata)
        band_out.Fill(ndata)
        band_out.SetNoDataValue(ndata)
    
    ds_out.SetGeoTransform(geot)
    ds_out.SetProjection(proj)
    band_out.WriteArray(a_crp)
    ds_out.FlushCache()
    
def crop_to_bbox(fpath_rast,fpath_out,bbox):
    
    r_in = input_raster(fpath_rast)
    
    lat = r_in.getLatLon(0.0,np.arange(r_in.rows),transform=False)[0]
    lon = r_in.getLatLon(np.arange(r_in.cols),0.0,transform=False)[1]
    
    #params[P_STN_LOC_BNDS] = (-126.0,-64.0,22.0,53.0) #CONUS
    
    mask_lon = np.logical_and(lon >= bbox[0],lon <= bbox[1])
    mask_lat = np.logical_and(lat >= bbox[2],lat <= bbox[3])
    lat = lat[mask_lat]
    lon = lon[mask_lon]
    
    a_in = r_in.readEntireRaster()
    
    a_in = a_in[mask_lat,:]
    a_in = a_in[:,mask_lon]
    
    '''
    Create GDAL geotransform list to define resolution and bounds
    GeoTransform[0] /* top left x */
    GeoTransform[1] /* w-e pixel resolution */
    GeoTransform[2] /* rotation, 0 if image is "north up" */
    GeoTransform[3] /* top left y */
    GeoTransform[4] /* rotation, 0 if image is "north up" */
    GeoTransform[5] /* n-s pixel resolution */
    '''
    in_geot = list(r_in.geo_t)
    in_geot[0] = lon[0] - (in_geot[1]/2.0) 
    in_geot[3] = lat[0] + np.abs(in_geot[5]/2.0)
    
    
    in_proj = r_in.raster.GetProjection()
    in_dtype = r_in.data.DataType
    in_ndata = r_in.data.GetNoDataValue()
    nrows = int(np.sum(mask_lat))
    ncols = int(np.sum(mask_lon))
    
    driver = gdal.GetDriverByName("GTiff")
    r_out = driver.Create(fpath_out,ncols,nrows,1,in_dtype)
        
    r_out.SetGeoTransform(in_geot)
    r_out.SetProjection(in_proj)
    band = r_out.GetRasterBand(1)
    band.SetNoDataValue(in_ndata)
    band.WriteArray(a_in,0,0)
    band.FlushCache() 
    

def get_lonlat(geot,c,r):
    '''Affine Transfrom: Converts pixel row and column to spatially referenced coordinates (in native projection)'''
    lon = (geot[0] + c*geot[1] + r*geot[2]) + geot[1] / 2.0
    lat = (geot[3] + c*geot[4] + r*geot[5]) + geot[5] / 2.0
    return lon,lat

def resample_to_grd1(fpath_dstgrid,fpath_srcgrid,fpath_out,resample_alg=gdalconst.GRA_NearestNeighbour,out_driver_name='GTiff'):
    grid_dst = gdal.Open(fpath_dstgrid)
    grid_src = gdal.Open(fpath_srcgrid)
    
    dst_proj = grid_dst.GetProjection()
    dst_geot = grid_dst.GetGeoTransform()
    
    src_proj = grid_src.GetProjection()
    src_band = grid_src.GetRasterBand(1)
    src_dtype = src_band.DataType
    src_ndata =  src_band.GetNoDataValue()
    
    grid_out = gdal.GetDriverByName(out_driver_name).Create(fpath_out, 
                                                            grid_dst.RasterXSize, 
                                                            grid_dst.RasterYSize, grid_src.RasterCount, src_dtype)
    
    band_out = grid_out.GetRasterBand(1)
    band_out.Fill(src_ndata)
    band_out.SetNoDataValue(src_ndata)
    
    grid_out.SetGeoTransform(dst_geot)
    grid_out.SetProjection(dst_proj)
    
    gdal.ReprojectImage(grid_src, grid_out, src_proj, dst_proj, resample_alg)
    
    grid_out.FlushCache()
    
    return grid_out

def resample_modis_sinu_to_grid(fpath_dstgrid,fpath_modis_sinu_grid,fpath_out,resample_alg=gdalconst.GRA_Bilinear):
    grid_dst = gdal.Open(fpath_dstgrid)
    grid_src = gdal.Open(fpath_modis_sinu_grid)

    #src_proj = "+proj=sinu +R=6371007.181 +nadgrids=@null +no_defs +wktext"
    src_proj = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
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

def new_mask():
    ds_mask = gdal.Open('/projects/daymet2/dem/interp_grids/tifs/interp_mask_conus_expand.tif')
    ds_lst = gdal.Open('/projects/daymet2/dem/interp_grids/tifs/lst_tmax.tif')
    ndata = ds_lst.GetRasterBand(1).GetNoDataValue()
    
    a_mask = ds_mask.ReadAsArray()
    a_lst = ds_lst.ReadAsArray()
    
    a_lst[a_mask==0] = ndata
    
    a_mask[np.logical_and(a_mask==1,a_lst==ndata)] = 0
    
    grid_out = gdal.GetDriverByName('GTiff').Create('/projects/daymet2/dem/interp_grids/tifs/interp_mask_conus_expand_clip.tif', 
                                                    ds_mask.RasterXSize, ds_mask.RasterYSize, 1, gdalconst.GDT_Float32)
    grid_out.SetGeoTransform(ds_mask.GetGeoTransform())
    grid_out.SetProjection(ds_mask.GetProjection())
    grid_out.GetRasterBand(1).WriteArray(a_mask)
    grid_out.FlushCache()
    del grid_out
    
    grid_out = gdal.GetDriverByName('GTiff').Create('/projects/daymet2/dem/interp_grids/tifs/lst_tmin_clip.tif', 
                                                    ds_mask.RasterXSize, ds_mask.RasterYSize, 1, gdalconst.GDT_Float32)
    
    grid_out.SetGeoTransform(ds_mask.GetGeoTransform())
    grid_out.SetProjection(ds_mask.GetProjection())
    grid_out.GetRasterBand(1).SetNoDataValue(-9999.)
    grid_out.GetRasterBand(1).WriteArray(a_lst)
    grid_out.FlushCache()
    del grid_out
    
def create_binary_mask(fpath,fpath_out):
    
    ds = gdal.Open(fpath)
    geot = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ndata = ds.GetRasterBand(1).GetNoDataValue()
    
    a = ds.ReadAsArray()
    
    mask_nd = a == ndata
    
    a[mask_nd] = 0
    a[np.logical_not(mask_nd)] = 1
        
    ds_out = gdal.GetDriverByName('GTiff').Create(  fpath_out, 
                                                    a.shape[1], 
                                                    a.shape[0], 1, gdalconst.GDT_Byte)
    band_out = ds_out.GetRasterBand(1)
#    band_out.Fill(ndata)
#    band_out.SetNoDataValue(ndata)
    
    ds_out.SetGeoTransform(geot)
    ds_out.SetProjection(proj)
    band_out.WriteArray(a)
    ds_out.FlushCache()

def create_fnl_mask(rast_fpaths,ndata,fpath_out):
    
    ds = gdal.Open(rast_fpaths[0])
    geot = ds.GetGeoTransform()
    proj = ds.GetProjection()
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    ds = None
    a = np.ones((rows,cols),np.int8)
    
    for x in np.arange(len(rast_fpaths)):
        
        ds = gdal.Open(rast_fpaths[x])
        vals = ds.ReadAsArray()
        
        for n in ndata[x]:
            
            a[vals==n] = 0
            
    ds_out = gdal.GetDriverByName('GTiff').Create(  fpath_out, 
                                                    a.shape[1], 
                                                    a.shape[0], 1, gdalconst.GDT_Byte)
    band_out = ds_out.GetRasterBand(1)    
    ds_out.SetGeoTransform(geot)
    ds_out.SetProjection(proj)
    band_out.WriteArray(a)
    ds_out.FlushCache()

def replace_with_nn_vals():

    #rmask = input_raster('/projects/daymet2/dem/interp_grids/conus/tifs/mask.tif')
    rmask = input_raster('/projects/daymet2/dem/interp_grids/tifs/elev.tif')
    rlcc = input_raster('/projects/daymet2/dem/interp_grids/tifs/fwpusgs_lcc.tif')
    src_dtype = rlcc.data.DataType
    src_ndata =  rlcc.data.GetNoDataValue()

    rout = output_raster('/projects/daymet2/dem/interp_grids/tifs/lcc3.tif', rlcc,noDataVal=src_ndata, datatype=src_dtype)
    
    amask = rmask.readEntireRaster()
    alcc = rlcc.readEntireRaster()
    alcc_new = np.copy(alcc)
        
    idxs = np.nonzero(np.logical_and(amask != rlcc.ndata,alcc==rlcc.ndata))

    schk = StatusCheck(idxs[0].size, 1000)
    
    for y,x in zip(idxs[0],idxs[1]):
                    
        r = 1
        nn = []
        nn_vals = []
        
        while len(nn) == 0 and r < 10:
            
            lcol = x-r
            rcol = x+r
            trow = y-r
            brow = y+r
            
            #top ring
            if trow > 0 and trow < rlcc.rows:
                
                for i in np.arange(lcol,rcol+1):
                    
                    if i > 0 and i < rlcc.cols:
                                               
                        if alcc[trow,i] != rlcc.ndata:                                
                            nn.append((trow,i))
                            nn_vals.append(alcc[trow,i])
            
            #left ring
            if lcol > 0 and lcol < rlcc.cols:
                
                for i in np.arange(trow,brow+1):
                    
                    if i > 0 and i < rlcc.rows:
                        
                        if alcc[i,lcol] != rlcc.ndata:                                                              
                            nn.append((i,lcol))
                            nn_vals.append(alcc[i,lcol])
                        
            #bottom ring
            if brow > 0 and brow < rlcc.rows:
                
                for i in np.arange(rcol,lcol,-1):
                    
                    if i > 0 and i < rlcc.cols:
                        
                        if alcc[brow,i] != rlcc.ndata:                                                                                          
                            nn.append((brow,i))
                            nn_vals.append(alcc[brow,i])
                            
            #right ring
            if rcol > 0 and rcol < rlcc.cols:
                
                for i in np.arange(brow,trow,-1):
                    
                    if i > 0 and i < rlcc.rows:
                        
                        if alcc[i,rcol] != rlcc.ndata:   
                            nn.append((i,rcol))
                            nn_vals.append(alcc[i,rcol])
            
            r+=1
        
        if len(nn) > 0:
            nn = np.array(nn)
            nn_vals = np.array(nn_vals)
            lats,lons = rlcc.getLatLon(nn[:,1], nn[:,0], transform=False)
            pt_lat,pt_lon = rlcc.getLatLon(x, y, transform=False)
            d = utlg.grt_circle_dist(pt_lon,pt_lat, lons, lats)
            j = np.argsort(d)[0]
            nlat,nlon = lats[j],lons[j]
            nval = nn_vals[j]
            alcc_new[y,x] = nval
            schk.increment()
          
    rout.writeDataArray(alcc_new,0,0)

def nnInterpClimDiv():
    
    ds = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/climdiv.tif')
    a = ds.gdal_ds.ReadAsArray()
    lat,lon = ds.get_coord_mesh_grid()
    ndata = ds.gdal_ds.GetRasterBand(1).GetNoDataValue()
    ndataMask = a != ndata
    proj = ds.gdal_ds.GetProjection()
    dtype = ds.gdal_ds.GetRasterBand(1).DataType
    
    ptVals = a[ndataMask]
    ptLat = lat[ndataMask]
    ptLon = lon[ndataMask]
    
    newA = si.griddata((ptLon,ptLat), ptVals.astype(np.float), (lon,lat), method='nearest')
    
    ds_out = gdal.GetDriverByName('GTiff').Create('/projects/daymet2/dem/interp_grids/tifs/climdivNN.tif', 
                                                    newA.shape[1], 
                                                    newA.shape[0], 1, dtype)
    
    band_out = ds_out.GetRasterBand(1)
    band_out.Fill(ndata)
    band_out.SetNoDataValue(ndata)
    ds_out.SetGeoTransform(ds.geo_t)
    ds_out.SetProjection(proj)
    band_out.WriteArray(newA.astype(np.uint16))
    ds_out.FlushCache()
    
    
    plt.imshow(newA)
    plt.colorbar()
    plt.show()

def maskNnClimDiv():
    
    dsClimDiv = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/climdivNN.tif')
    a = dsClimDiv.gdal_ds.ReadAsArray()
    ndata = dsClimDiv.gdal_ds.GetRasterBand(1).GetNoDataValue()
    proj = dsClimDiv.gdal_ds.GetProjection()
    dtype = dsClimDiv.gdal_ds.GetRasterBand(1).DataType
    
    dsMask = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/mask_all.tif')
    mask = dsMask.gdal_ds.ReadAsArray()
    
    a[mask==0] = ndata
    
    ds_out = gdal.GetDriverByName('GTiff').Create('/projects/daymet2/dem/interp_grids/tifs/climdivNnMasked.tif', 
                                                    a.shape[1], 
                                                    a.shape[0], 1, dtype)
    
    band_out = ds_out.GetRasterBand(1)
    band_out.Fill(ndata)
    band_out.SetNoDataValue(ndata)
    ds_out.SetGeoTransform(dsClimDiv.geo_t)
    ds_out.SetProjection(proj)
    band_out.WriteArray(a.astype(np.uint16))
    ds_out.FlushCache()

def mergeClimDivLCC():
    
    dsClimDiv = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/climdivNnMasked.tif')
    a = dsClimDiv.gdal_ds.ReadAsArray()
    ndata = dsClimDiv.gdal_ds.GetRasterBand(1).GetNoDataValue()
    proj = dsClimDiv.gdal_ds.GetProjection()
    dtype = dsClimDiv.gdal_ds.GetRasterBand(1).DataType
    
    dsLCC = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/fwpusgs_lcc.tif')
    lcc = dsLCC.gdal_ds.ReadAsArray()
    
    a[0:470,500:3300] = lcc[0:470,500:3300]
    
    dsMask = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/mask_all.tif')
    mask = dsMask.gdal_ds.ReadAsArray()
    
    a[mask==0] = ndata
    
    ds_out = gdal.GetDriverByName('GTiff').Create('/projects/daymet2/dem/interp_grids/tifs/climdivLccMerge.tif', 
                                                    a.shape[1], 
                                                    a.shape[0], 1, dtype)
    
    band_out = ds_out.GetRasterBand(1)
    band_out.Fill(ndata)
    band_out.SetNoDataValue(ndata)
    ds_out.SetGeoTransform(dsClimDiv.geo_t)
    ds_out.SetProjection(proj)
    band_out.WriteArray(a.astype(np.uint16))
    ds_out.FlushCache()
    
if __name__ == '__main__':
    
    #Building of ClimDiv Raster
    #mergeClimDivLCC()
    #maskNnClimDiv()
    #nnInterpClimDiv()
    
    #replace_with_nn_vals()
#    mask_to_rastmask('/projects/daymet2/dem/interp_grids/conus/tifs/mask_all_nd.tif',
#                     '/projects/daymet2/dem/interp_grids/tifs/fwpusgs_lcc.tif', 
#                     '/projects/daymet2/dem/interp_grids/conus/tifs/fwpusgs_lcc_masked.tif')
    
    ########################################################################
    #Montana AOI
    ########################################################################
    
#    mask_to_shp('/projects/daymet2/dem/interp_grids/montana_interp_bbox/AOI_Polygon.shp', 
#                'AOI_Polygon','/projects/daymet2/dem/interp_grids/tifs/elev.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/elev.tif')
    
#    mask_to_shp('/projects/daymet2/dem/interp_grids/montana_interp_bbox/AOI_Polygon.shp', 
#                'AOI_Polygon','/projects/daymet2/dem/interp_grids/tifs/modis_lc_type1.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/modis_lc_type1.tif',nodata=255)
#    
#    mask_to_shp('/projects/daymet2/dem/interp_grids/montana_interp_bbox/AOI_Polygon.shp', 
#                'AOI_Polygon','/projects/daymet2/dem/interp_grids/tifs/lst_tmax.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/lst_tmax.tif')
#    
#    mask_to_shp('/projects/daymet2/dem/interp_grids/montana_interp_bbox/AOI_Polygon.shp', 
#                'AOI_Polygon','/projects/daymet2/dem/interp_grids/tifs/lst_tmin.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/lst_tmin.tif')
#    
#    mask_to_shp('/projects/daymet2/dem/interp_grids/montana_interp_bbox/AOI_Polygon.shp', 
#                'AOI_Polygon','/projects/daymet2/dem/interp_grids/tifs/vcf.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/vcf.tif')
#    
#    mask_to_shp('/projects/daymet2/dem/interp_grids/montana_interp_bbox/AOI_Polygon.shp', 
#                'AOI_Polygon','/projects/daymet2/dem/interp_grids/tifs/tdi.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/tdi.tif')

#    mask_to_shp('/projects/daymet2/dem/interp_grids/montana_interp_bbox/AOI_Polygon.shp', 
#                'AOI_Polygon','/projects/daymet2/dem/interp_grids/tifs/lcc.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/lcc.tif',nodata=255)
    
    ########################################################################
#    crop_nodata('/projects/daymet2/dem/interp_grids/montana/tifs/elev.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/crop_elev.tif')
#    
#    crop_nodata('/projects/daymet2/dem/interp_grids/montana/tifs/modis_lc_type1.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/crop_modis_lc_type1.tif')
#    
#    crop_nodata('/projects/daymet2/dem/interp_grids/montana/tifs/lst_tmax.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/crop_lst_tmax.tif')
#    
#    crop_nodata('/projects/daymet2/dem/interp_grids/montana/tifs/lst_tmin.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/crop_lst_tmin.tif')
#    
#    crop_nodata('/projects/daymet2/dem/interp_grids/montana/tifs/vcf.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/crop_vcf.tif')
#    
#    crop_nodata('/projects/daymet2/dem/interp_grids/montana/tifs/tdi.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/crop_tdi.tif')

#    crop_nodata('/projects/daymet2/dem/interp_grids/montana/tifs/lcc.tif', 
#                '/projects/daymet2/dem/interp_grids/montana/tifs/crop_lcc.tif')

########################################################################
#    create_binary_mask('/projects/daymet2/dem/interp_grids/montana/tifs/crop_elev.tif',
#                       '/projects/daymet2/dem/interp_grids/montana/tifs/mask.tif')
#    
#########################################################################    
#    rasts = []
#    rasts.append('/projects/daymet2/dem/interp_grids/montana/tifs/crop_elev.tif')
#    rasts.append('/projects/daymet2/dem/interp_grids/montana/tifs/crop_tdi.tif')
#    rasts.append('/projects/daymet2/dem/interp_grids/montana/tifs/crop_lst_tmax.tif')
#    rasts.append('/projects/daymet2/dem/interp_grids/montana/tifs/crop_lst_tmin.tif')
#    rasts.append('/projects/daymet2/dem/interp_grids/montana/tifs/crop_modis_lc_type1.tif')
#    rasts.append('/projects/daymet2/dem/interp_grids/montana/tifs/mask.tif')
#    
#    ndata = []
#    ndata.append((-9999,))
#    ndata.append((-9999,))
#    ndata.append((-9999,))
#    ndata.append((-9999,))
#    ndata.append((255,))
#    ndata.append((0,))
#########################################################################    
#    create_fnl_mask(rasts, ndata, '/projects/daymet2/dem/interp_grids/montana/tifs/mask_fnl.tif')
 ########################################################################   
#    to_ncdf('/projects/daymet2/dem/interp_grids/montana/tifs/crop_elev.tif',
#            'elev', '/projects/daymet2/dem/interp_grids/montana/ncdf/elev.nc', np.float32, -9999)
#    
#    to_ncdf('/projects/daymet2/dem/interp_grids/montana/tifs/crop_tdi.tif',
#            'tdi', '/projects/daymet2/dem/interp_grids/montana/ncdf/tdi.nc', np.float32, -9999)
#    
#    to_ncdf('/projects/daymet2/dem/interp_grids/montana/tifs/crop_lst_tmax.tif',
#            'lst_tmax', '/projects/daymet2/dem/interp_grids/montana/ncdf/lst_tmax.nc', np.float32, -9999)
#    
#    to_ncdf('/projects/daymet2/dem/interp_grids/montana/tifs/crop_lst_tmin.tif',
#            'lst_tmin', '/projects/daymet2/dem/interp_grids/montana/ncdf/lst_tmin.nc', np.float32, -9999)
#    
#    to_ncdf('/projects/daymet2/dem/interp_grids/montana/tifs/crop_modis_lc_type1.tif',
#            'lc', '/projects/daymet2/dem/interp_grids/montana/ncdf/modis_lc_type1.nc', np.int8, 255)
#    
#    to_ncdf('/projects/daymet2/dem/interp_grids/montana/tifs/crop_vcf.tif',
#            'vcf', '/projects/daymet2/dem/interp_grids/montana/ncdf/vcf.nc', np.float32, -9999)
#    
#    to_ncdf('/projects/daymet2/dem/interp_grids/montana/tifs/mask_fnl.tif',
#            'mask', '/projects/daymet2/dem/interp_grids/montana/ncdf/mask.nc', np.int8)
########################################################################    
#    ds = Dataset('/projects/daymet2/dem/interp_grids/montana/ncdf/mask.nc')
#    mask = ds.variables['mask'][:]
#    expand_grid(ds, 'mask', (1250,2500), '/projects/daymet2/dem/interp_grids/montana/ncdf/fnl_mask.nc',0,mask)
#    ds.close()
#    ds = None
#    
#    ds = Dataset('/projects/daymet2/dem/interp_grids/montana/ncdf/elev.nc')
#    expand_grid(ds, 'elev', (1250,2500), '/projects/daymet2/dem/interp_grids/montana/ncdf/fnl_elev.nc',
#                ds.variables['elev'].missing_value,mask)
#    ds.close()
#    ds = None
#    
#    ds = Dataset('/projects/daymet2/dem/interp_grids/montana/ncdf/tdi.nc')
#    expand_grid(ds, 'tdi', (1250,2500), '/projects/daymet2/dem/interp_grids/montana/ncdf/fnl_tdi.nc',
#                ds.variables['tdi'].missing_value,mask)
#    ds.close()
#    ds = None
#    
#    ds = Dataset('/projects/daymet2/dem/interp_grids/montana/ncdf/lst_tmax.nc')
#    expand_grid(ds, 'lst_tmax', (1250,2500), '/projects/daymet2/dem/interp_grids/montana/ncdf/fnl_lst_tmax.nc',
#                ds.variables['lst_tmax'].missing_value,mask)
#    ds.close()
#    ds = None
#    
#    ds = Dataset('/projects/daymet2/dem/interp_grids/montana/ncdf/lst_tmin.nc')
#    expand_grid(ds, 'lst_tmin', (1250,2500), '/projects/daymet2/dem/interp_grids/montana/ncdf/fnl_lst_tmin.nc',
#                ds.variables['lst_tmin'].missing_value,mask)
#    ds.close()
#    ds = None
#    
#    ds = Dataset('/projects/daymet2/dem/interp_grids/montana/ncdf/modis_lc_type1.nc')
#    expand_grid(ds, 'lc', (1250,2500), '/projects/daymet2/dem/interp_grids/montana/ncdf/fnl_modis_lc_type1.nc',
#                ds.variables['lc'].missing_value,mask)
#    ds.close()
#    ds = None
#    
#    ds = Dataset('/projects/daymet2/dem/interp_grids/montana/ncdf/vcf.nc')
#    expand_grid(ds, 'vcf', (1250,2500), '/projects/daymet2/dem/interp_grids/montana/ncdf/fnl_vcf.nc',
#                ds.variables['vcf'].missing_value,mask)
#    ds.close()
#    ds = None



    ########################################################################
    #CONUS AOI
    ########################################################################
    
#    resample_to_grd1('/projects/daymet2/dem/interp_grids/tifs/elev.tif', '/projects/daymet2/dem/interp_mask.tif', 
#                     '/projects/daymet2/dem/interp_grids/conus/tifs/mask.tif')
        

#########################################################################    
#    rasts = []
#    
#    rasts.append('/projects/daymet2/dem/interp_grids/tifs/elev.tif')
#    rasts.append('/projects/daymet2/dem/interp_grids/tifs/tdi.tif')
#    rasts.append('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Day_1km.01.C.tif')
#    rasts.append('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Night_1km.01.C.tif')
#    rasts.append('/projects/daymet2/dem/interp_grids/tifs/mask_all.tif')
#    
#    ndata_vals = []
#    ndata_vals.append((-999,)) #elev
#    ndata_vals.append((-999,)) #tdi
#    ndata_vals.append((65535,)) #lst_tmax
#    ndata_vals.append((65535,)) #lst_tmin
#    ndata_vals.append((0,)) #mask
#
#    create_fnl_mask(rasts, ndata_vals, '/projects/daymet2/dem/interp_grids/conus/tifs/mask_all_nd.tif')

#    ds = gdal.Open('/projects/daymet2/dem/interp_grids/conus/tifs/mask_all_nd.tif')    
#    mask = ds.ReadAsArray() != 0
#
#    for mth in np.arange(1,13):
#        fpathIn = '/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Night_1km.%02d.C.tif'%mth
#        crop_nodata(fpathIn, '/projects/daymet2/dem/interp_grids/conus/tifs/crop_lst_tmin%02d.tif'%mth, mask)
#        
#        fpathIn = '/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Day_1km.%02d.C.tif'%mth
#        crop_nodata(fpathIn, '/projects/daymet2/dem/interp_grids/conus/tifs/crop_lst_tmax%02d.tif'%mth, mask)
#
#    crop_nodata('/projects/daymet2/dem/interp_grids/tifs/elev.tif', 
#                '/projects/daymet2/dem/interp_grids/conus/tifs/crop_elev.tif',mask)
#    
#    crop_nodata('/projects/daymet2/dem/interp_grids/tifs/fwpusgs_lcc.tif', 
#                '/projects/daymet2/dem/interp_grids/conus/tifs/crop_fwpusgs_lcc.tif',mask)

#    crop_nodata('/projects/daymet2/dem/interp_grids/tifs/tdi.tif', 
#                '/projects/daymet2/dem/interp_grids/conus/tifs/crop_tdi.tif',mask)
#
#    crop_nodata('/projects/daymet2/dem/interp_grids/conus/tifs/mask_all_nd.tif', 
#                '/projects/daymet2/dem/interp_grids/conus/tifs/crop_mask.tif',mask)
##
#    crop_nodata('/projects/daymet2/dem/interp_grids/tifs/climdivLccMerge.tif', 
#                '/projects/daymet2/dem/interp_grids/conus/tifs/crop_climdiv.tif',mask)
    
########################################################################   
#    to_ncdf('/projects/daymet2/dem/interp_grids/conus/tifs/crop_elev.tif',
#            'elev', '/projects/daymet2/dem/interp_grids/conus/ncdf/elev.nc', np.float32, -999.0)  
#    to_ncdf('/projects/daymet2/dem/interp_grids/conus/tifs/crop_tdi.tif',
#            'tdi', '/projects/daymet2/dem/interp_grids/conus/ncdf/tdi.nc', np.float32, -999.0)
#    to_ncdf('/projects/daymet2/dem/interp_grids/conus/tifs/crop_climdiv.tif',
#            'neon', '/projects/daymet2/dem/interp_grids/conus/ncdf/climdiv.nc', np.uint16, np.uint16(65535))
#    to_ncdf('/projects/daymet2/dem/interp_grids/conus/tifs/crop_mask.tif',
#            'mask', '/projects/daymet2/dem/interp_grids/conus/ncdf/mask.nc', np.int8)
#    for mth in np.arange(1,13):
#        fpathIn = '/projects/daymet2/dem/interp_grids/conus/tifs/crop_lst_tmax%02d.tif'%mth
#        fpathOut = '/projects/daymet2/dem/interp_grids/conus/ncdf/lst_tmax%02d.nc'%mth
#        to_ncdf(fpathIn, 'tmax%02d'%mth, fpathOut, np.float32, 65535.0)
#        
#        fpathIn = '/projects/daymet2/dem/interp_grids/conus/tifs/crop_lst_tmin%02d.tif'%mth
#        fpathOut = '/projects/daymet2/dem/interp_grids/conus/ncdf/lst_tmin%02d.nc'%mth
#        to_ncdf(fpathIn, 'tmin%02d'%mth, fpathOut, np.float32, 65535.0)
        
#########################################################################    
    ds = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/mask.nc')
    mask = ds.variables['mask'][:]
    dims = (3250,7000)
    
    expand_grid(ds, 'mask', dims, '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_mask.nc',0,mask)
    ds.close()
    ds = None
 
    ds = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/elev.nc')
    expand_grid(ds, 'elev', dims, '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc',
                ds.variables['elev'].missing_value,mask)
    ds.close()
    ds = None
    
    ds = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/tdi.nc')
    expand_grid(ds, 'tdi', dims, '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc',
                ds.variables['tdi'].missing_value,mask)
    ds.close()
    ds = None

    ds = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/climdiv.nc')
    expand_grid(ds, 'neon', dims, '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_climdiv.nc',
                ds.variables['neon'].missing_value,mask)
    ds.close()
    ds = None
    
    for mth in np.arange(1,13):
        
        dsIn = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/lst_tmax%02d.nc'%mth)
        varname = 'tmax%02d'%mth
        fpathOut = '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax%02d.nc'%mth
        expand_grid(dsIn, varname, dims, fpathOut, dsIn.variables[varname].missing_value, mask)
        dsIn.close()
        dsIn = None
        
        dsIn = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/lst_tmin%02d.nc'%mth)
        varname = 'tmin%02d'%mth
        fpathOut = '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin%02d.nc'%mth
        expand_grid(dsIn, varname, dims, fpathOut, dsIn.variables[varname].missing_value, mask)
        dsIn.close()
        dsIn = None
    
##########################################################
##########################################################
    
    
#    crop_to_bbox('/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge_fnl_wgs84.tif', 
#                 '/projects/daymet2/dem/interp_grids/tifs/elev.tif', bbox=(-126.0,-64.0,22.0,53.0))
    
#    crop_to_bbox('/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge_tdi.tif', 
#                 '/projects/daymet2/dem/interp_grids/tifs/tdi.tif', bbox=(-126.0,-64.0,22.0,53.0))
    
    #Resampling of modis grids to the interpolation grid
    
    #Land Cover
#    resample_modis_sinu_to_grid('/projects/daymet2/dem/interp_grids/tifs/elev.tif', 
#                                '/projects/daymet2/climate_office/modis/MOD12Q1/mosaic_lc.tif', 
#                                '/projects/daymet2/dem/interp_grids/tifs/modis_lc_type1.tif',gdalconst.GRA_NearestNeighbour)
    
    #Tmax LST
#    resample_modis_sinu_to_grid('/projects/daymet2/dem/interp_grids/tifs/elev.tif', 
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs3/day/mosaic_mean_lst_tmax.tif', 
#                                '/projects/daymet2/dem/interp_grids/tifs/lst_tmax.tif',
#                                gdalconst.GRA_Bilinear)
    
#    resample_modis_sinu_to_grid('/projects/daymet2/dem/interp_grids/tifs/elev.tif', 
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.tif', 
#                                '/projects/daymet2/dem/interp_grids/tifs/lst_tmin.tif',
#                                gdalconst.GRA_NearestNeighbour)
    
#    resample_modis_sinu_to_grid('/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.MRT.LST_Night_1km.tif', 
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.tif', 
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.GDALTest.tif',
#                                gdalconst.GRA_Bilinear)

    
    #Tmin LST
#    resample_modis_sinu_to_grid('/projects/daymet2/dem/interp_grids/tifs/elev.tif', 
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs3/night/mosaic_mean_lst_tmin.tif', 
#                                '/projects/daymet2/dem/interp_grids/tifs/lst_tmin.tif',
#                                gdalconst.GRA_Bilinear)
    
    #VCF
#    resample_modis_sinu_to_grid('/projects/daymet2/dem/interp_grids/tifs/elev.tif', 
#                                '/projects/daymet2/climate_office/modis/MOD44B/mosaic_vcf_reclass1km.tif', 
#                                '/projects/daymet2/dem/interp_grids/tifs/vcf.tif',
#                                gdalconst.GRA_Bilinear)
    
    
    #create_bounds_poly((-108.2,43.0),(-119.5,53.6),"/projects/daymet2/dem/cce/","tdi_bnds")
    
#    clip_to_shp('/projects/daymet2/dem/cce/CCE_StudyArea/cce_buffer.shp', 'cce_buffer', 
#                '/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge_fnl.tif', '/projects/daymet2/dem/cce/cce_elev.tif')

#    mask_to_shp('/projects/daymet2/dem/cce/tdi_bnds/tdi_bnds.shp', 'tdi_bnds', 
#                '/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge_fnl.tif', '/projects/daymet2/dem/cce/cce_elev_for_tdi.tif')

    #crop_nodata('/projects/daymet2/dem/cce/cce_elev_for_tdi.tif', '/projects/daymet2/dem/cce/cce_elev_for_tdi_crop.tif')

#    resample_modis_sinu_to_grid('/projects/daymet2/dem/cce/cce_bbox_elev.tif',
#                                '/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs_day/mosaic_mean_lst_tmax.tif', 
#                                '/projects/daymet2/dem/cce/cce_bbox_lst_tmax.tif')

#    mask_to_shp('/projects/daymet2/dem/cce/CCE_StudyArea/cce_buffer.shp','cce_buffer',
#                '/projects/daymet2/dem/cce/cce_bbox_elev.tif', '/projects/daymet2/dem/cce/cce_elev.tif')

#    mask_to_shp('/projects/daymet2/dem/cce/CCE_StudyArea/cce_buffer.shp','cce_buffer',
#                '/projects/daymet2/dem/cce/cce_bbox_lst_tmin.tif', '/projects/daymet2/dem/cce/cce_lst_tmin.tif')

#    mask_to_shp('/projects/daymet2/dem/cce/CCE_StudyArea/cce_buffer.shp','cce_buffer',
#                '/projects/daymet2/dem/cce/cce_bbox_lst_tmax.tif', '/projects/daymet2/dem/cce/cce_lst_tmax.tif')

#    mask_to_shp('/projects/daymet2/dem/cce/CCE_StudyArea/cce_buffer.shp','cce_buffer',
#                '/projects/daymet2/dem/cce/cce_bbox_tdi.tif', '/projects/daymet2/dem/cce/cce_tdi.tif')

#    crop_nodata('/projects/daymet2/dem/cce/cce_elev.tif', 
#                '/projects/daymet2/dem/cce/cce_elev2.tif')
#    
#    crop_nodata('/projects/daymet2/dem/cce/cce_lst_tmin.tif', 
#                '/projects/daymet2/dem/cce/cce_lst_tmin2.tif')
#    
#    crop_nodata('/projects/daymet2/dem/cce/cce_lst_tmax.tif', 
#                '/projects/daymet2/dem/cce/cce_lst_tmax2.tif')

#    crop_nodata('/projects/daymet2/dem/cce/cce_tdi.tif', 
#                '/projects/daymet2/dem/cce/cce_tdi2.tif')


#    mask_to_shp('/projects/daymet2/climate_office/cnty_shp/choteau_cnty.shp', 'choteau_cnty', 
#                '/projects/daymet2/climate_office/modis/MOD13Q1.005/aug_gtiff/wgs84_aug_ndvi_2010.tif', 
#                '/projects/daymet2/climate_office/modis/MOD13Q1.005/aug_gtiff/cc_wgs84_aug_ndvi_2010.tif')

    #new_mask()
    #resample_to_grid()
    #clip_rasters()