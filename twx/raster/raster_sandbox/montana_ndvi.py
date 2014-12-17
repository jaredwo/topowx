'''
Created on Nov 13, 2012

@author: jared.oyler
'''
import numpy as np
import os
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import osgeo.osr as osr
import osgeo.gdal as gdal
import osgeo.gdalconst as gdalconst
from downmodis import downModis
from twx.utils.input_raster import input_raster
import scipy.spatial.distance as scpydist
import osgeo.ogr as ogr
import sys
from netCDF4 import Dataset
import netCDF4
import datetime

TILES_MONTANA = ['h09v04','h10v04','h11v04']

PATH_MOD13A2 = '/MODIS/Mirror/MOD13A2.005/'

MYD11A2_MTH_DAYS = {1:['001','009','017','025'],
                    2:['033','041','049','057'],
                    3:['065','073','081','089'],
                    4:['097','105','113','121'],
                    5:['121','129','137','145'],
                    6:['153','161','169','177'],
                    7:['185','193','201','209'],
                    8:['217','225','233','241'],
                    9:['249','257','265','273'],
                    10:['281','289','297','305'],
                    11:['305','313','321','329'],
                    12:['337','345','353','361']}

MTH_ABBRV = {1:'JAN',
            2:'FEB',
            3:'MAR',
            4:'APR',
            5:'MAY',
            6:'JUN',
            7:'JUL',
            8:'AUG',
            9:'SEP',
            10:'OCT',
            11:'NOV',
            12:'DEC'}

TILE_DATE_DIRS = ['2000.07.27','2000.08.12','2000.08.28','2000.09.13','2000.09.29',
                  '2001.07.28','2001.08.13','2001.08.29','2001.09.14','2001.09.30',
                  '2002.07.28','2002.08.13','2002.08.29','2002.09.14','2002.09.30',
                  '2003.07.28','2003.08.13','2003.08.29','2003.09.14','2003.09.30',
                  '2004.07.27','2004.08.12','2004.08.28','2004.09.13','2004.09.29',
                  '2005.07.28','2005.08.13','2005.08.29','2005.09.14','2005.09.30',
                  '2006.07.28','2006.08.13','2006.08.29','2006.09.14','2006.09.30',
                  '2007.07.28','2007.08.13','2007.08.29','2007.09.14','2007.09.30',
                  '2008.07.27','2008.08.12','2008.08.28','2008.09.13','2008.09.29',
                  '2009.07.28','2009.08.13','2009.08.29','2009.09.14','2009.09.30',
                  '2010.07.28','2010.08.13','2010.08.29','2010.09.14','2010.09.30',
                  '2011.07.28','2011.08.13','2011.08.29','2011.09.14','2011.09.30',
                  '2012.07.27','2012.08.12']#,'2012.08.28','2012.09.13','2012.09.29']

PREFIX_MOD13A3 = 'MOD13A3.A'
YEARS = np.arange(2000,2013)
PATH_MOD13A3 = '/projects/daymet2/climate_office/modis/sept_mod13a3/'
PATH_MYD11A2 = '/projects/daymet2/climate_office/modis/MYD11A2.005.h10v04/'

FILL_VALUE_NDVI = -3000
SCALE_NDVI = 0.0001

EPSG_WGS84 = 4326 #EPSG Code
EPSG_NAD83 = 4269 #EPSG Code
#PROJ4_MODIS = "+proj=sinu +R=6371007.181 +nadgrids=@null +no_defs +wktext"
PROJ4_MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"

UPPER_LEFT_PT = (-10007554.677000,5559752.598333)
LOWER_RIGHT_PT = (-6671703.118000,4447802.078667)
XDIM=3600
YDIM=1200

'''
Euclidian distance
import numpy as np
single_point = [3, 4]
points = np.arange(20).reshape((10,2))

dist = (points - single_point)**2
dist = np.sum(dist, axis=1)
dist = np.sqrt(dist)


gdalwarp -dstnodata -9999 lst_mean_ngt_h07v05.tif lst_mean_ngt_h07v06.tif lst_mean_ngt_h08v04.tif lst_mean_ngt_h08v05.tif lst_mean_ngt_h09v04.tif lst_mean_ngt_h09v05.tif lst_mean_ngt_h09v06.tif lst_mean_ngt_h10v04.tif lst_mean_ngt_h10v05.tif lst_mean_ngt_h10v06.tif lst_mean_ngt_h11v04.tif lst_mean_ngt_h11v05.tif lst_mean_ngt_h11v06.tif lst_mean_ngt_h12v04.tif lst_mean_ngt_h12v05.tif lst_mean_ngt_h13v04.tif lst_mean_ngt_h08v06.tif mosaic_mean_gdal2.tif

'''

class modis_sin_rast(input_raster):

    def __init__(self, filePath,bandNum=1):
        
        input_raster.__init__(self, filePath, bandNum)
        sr_sin = osr.SpatialReference()
        sr_sin.ImportFromProj4(PROJ4_MODIS)
        sr_wgs84 = osr.SpatialReference()
        sr_wgs84.ImportFromEPSG(EPSG_WGS84)
        trans_wgs84_to_sin = osr.CoordinateTransformation(sr_wgs84,sr_sin)
        self.coordTrans_wgs84_to_src = trans_wgs84_to_sin
        
        self.a = self.readEntireRaster()
        x_grid,y_grid = self.x_y_arrays()
        self.xs = x_grid.ravel()
        self.ys = y_grid.ravel()
        self.a_flat = self.a.ravel()
        ndata_mask = self.a_flat != self.ndata
        self.xs = self.xs[ndata_mask]
        self.ys = self.ys[ndata_mask]
        self.pts = np.column_stack((self.xs,self.ys))
        self.a_flat = self.a_flat[ndata_mask]
        
    
    def __euclid_dist(self,pt,pts):
        
        #single_point = [3, 4]
        #points = np.arange(20).reshape((10,2))
        
        dist = (pts - pt)**2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        return dist
    
    def get_data_value(self,lon,lat,interp_ndata = False):
        
        data_val = input_raster.get_data_value(self, lon, lat, False)
        
        if data_val == self.ndata and interp_ndata:

            x_pt, y_pt = self.coordTrans_wgs84_to_src.TransformPoint(lon,lat)[0:2] 
            dists = self.__euclid_dist([x_pt,y_pt],self.pts)
            dist_sort = np.argsort(dists)
            
            nn_dists = dists[dist_sort][0:4]
            nn_vals = self.a_flat[dist_sort][0:4]
            
            radius = np.max(nn_dists)+10
        
            wgt = (1.0+np.cos(np.pi*(nn_dists/radius)))/2.0
            #wgt = np.concatenate((np.ones(nnr_dists.size),wgt))
            wgt = wgt/np.sum(wgt)
            
            data_val = np.average(nn_vals, weights=wgt)
        
        return data_val

def download(path_fftp,out_path):
    
    os.chdir(out_path)
    fftp = open(path_fftp)
    for line in fftp.readlines():
        
        line = line.strip()
        if line[-3] == "xml":
            continue
        else:
            os.system("".join(["wget ",line]))

class modis_sin_latlon_transform():

    def __init__(self):
        
        self.sr_sin = osr.SpatialReference()
        self.sr_sin.ImportFromProj4(PROJ4_MODIS)
        self.sr_wgs84 = osr.SpatialReference()
        self.sr_wgs84.ImportFromEPSG(EPSG_WGS84)
        self.sr_nad83 = osr.SpatialReference()
        self.sr_nad83.ImportFromEPSG(EPSG_NAD83)
        
        self.trans_sin_to_wgs84 = osr.CoordinateTransformation(self.sr_sin,self.sr_wgs84)
        self.trans_wgs84_to_sin = osr.CoordinateTransformation(self.sr_wgs84,self.sr_sin)
        self.trans_nad83_to_sin = osr.CoordinateTransformation(self.sr_nad83,self.sr_sin)

def output_raster(a,fpath):
    
    driver = gdal.GetDriverByName("GTiff")
    nrows = a.shape[0]
    ncols = a.shape[1]
    
    projs = modis_sin_latlon_transform()
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
    geotransform[0] = UPPER_LEFT_PT[0]
    geotransform[3] = UPPER_LEFT_PT[1]
    geotransform[2],geotransform[4] = (0.0,0.0)
    geotransform[1] = np.abs((UPPER_LEFT_PT[0]-LOWER_RIGHT_PT[0])/XDIM)
    #n-s pixel height/resolution needs to be negative
    geotransform[5] = -np.abs((UPPER_LEFT_PT[1]-LOWER_RIGHT_PT[1])/YDIM)
    
    raster = driver.Create(fpath,ncols,nrows,1, gdalconst.GDT_Float32)
    raster.SetGeoTransform(geotransform)
    raster.SetProjection(projs.sr_sin.ExportToWkt()) 
    
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    
    a_d = a.data
    a_d[a.mask] = -9999
    
    band.WriteArray(a_d,0,0)  
    band.FlushCache()

def process_stack_MYD11A2_mth(in_path,out_fpath,mth,data_varname='LST_Night_1km',qc_varname='QC_Night'):
    os.chdir(in_path)
    qc_bitmask = 0b00000011
    
    fnames = np.array(os.listdir(in_path))
    fnames = np.sort(fnames[np.char.endswith(fnames,".hdf")])
    
    yrs = np.arange(2003,2012)
    mth_days = MYD11A2_MTH_DAYS[mth]
    mth_mask = None
    
    
    for yr in yrs:
       
        for yday in mth_days:
        
            yday_mask = np.char.startswith(fnames, "".join(["MYD11A2.A",str(yr),yday])) 
        
            if mth_mask is None:
                mth_mask = yday_mask
            else:
                mth_mask = np.logical_or(mth_mask,yday_mask)
        
    fnames = fnames[mth_mask]
    
#    if fnames.size != 414:
#        raise Exception("Incorrect # of files for "+in_path+" | ",str(fnames.size)," total files")
    
    lst_ls = []
    for fname in fnames:
        sd = SD(fname,SDC.READ)
        sds_lst = sd.select(data_varname)
        lst_scale = sds_lst.attributes()['scale_factor']
        lst_fill = sds_lst.attributes()['_FillValue']
        lst = np.array(sds_lst[:],dtype = np.float64)
        fill_mask = lst==lst_fill
        
        sds_qc = sd.select(qc_varname)[:]
        
        #mask out all but bits 0 - 1
        qc_masked = sds_qc & qc_bitmask
        #mask out values that are 10, or 11 in bits 0-1
        qc_mask = np.logical_or(qc_masked==0b10,qc_masked==0b11)
        
        #lst = np.ma.masked_array(lst,np.logical_or(fill_mask,qc_mask),fill_value=np.nan)
        lst[np.logical_or(fill_mask,qc_mask)] = np.nan
        lst = lst*lst_scale
        lst_ls.append(lst)
        #print fname,np.sum(qc_mask)
#        plt.imshow(lst)
#        plt.colorbar()
#        plt.show()
    
    lst_stack = np.dstack(lst_ls)
    lst_stack = np.ma.masked_array(lst_stack,np.isnan(lst_stack),fill_value=np.nan)
    lst_mean = np.ma.mean(lst_stack,axis=2)
    lst_mean = KtoC(lst_mean)
    
    grid_sd = EOSGridSD(fnames[0])
    grid_sd.to_geotiff(lst_mean,out_fpath)
    
    lst_mean = KtoC(lst_mean)
    
    grid_sd = EOSGridSD(fnames[0])
    grid_sd.to_geotiff(lst_mean,out_fpath)


def process_MOD12Q1(tiles,in_path,out_path,varname='Land_Cover_Type_1'):
    
    fnames = np.array(os.listdir(in_path))
    os.chdir(in_path)
    
    for fname in fnames:
        
        tile = fname.split('.')[2]
        
        if tile in tiles:
            
            sd = SD(fname,SDC.READ)
            sds_lc = sd.select(varname)
            lc = sds_lc[:]
            sd.end()

            grid_sd = EOSGridSD(fname)
            grid_sd.to_geotiff(lc,"".join([out_path,"lc_",tile,'.tif']),gdalconst.GDT_Byte,255)
            print tile
    

def stack_MOD44B(tile,in_path,out_path):
    
    yr_dirs = os.listdir(in_path)
    
    vcf_stack = []
    
    for yrdir in yr_dirs:
        
        os.chdir("".join([in_path,yrdir]))
        fnames = np.array(os.listdir("".join([in_path,yrdir])))
        x = np.nonzero(np.char.find(fnames,tile) != -1)[0][0]
        fname = fnames[x]
        sd = SD(fname,SDC.READ)
        sds_vcf = sd.select('Percent_Tree_Cover')
        vcf = sds_vcf[:].astype(np.float)
        vcf = np.ma.masked_array(vcf,mask=vcf==253)
        sd.end()
        
#        plt.imshow(vcf)
#        plt.colorbar()
#        plt.show()
        vcf_stack.append(vcf)

    vcf_stack = np.ma.dstack(vcf_stack)
    vcf_mean = np.ma.mean(vcf_stack,axis=2)
    vcf_mean = np.ma.round(vcf_mean)
    
    grid_sd = EOSGridSD(fname)
    grid_sd.to_geotiff(vcf_mean,"".join([out_path,"vcf_",tile,".tif"]),gdalconst.GDT_Byte,253)

def stack_MYD11A2_hdf(in_path,out_ncdf,out_fpath,data_varname='LST_Night_1km',qc_varname='QC_Night',va_varname='Night_view_angl'):
    
    qc_bitmask1 = 0b00000011
    qc_bitmask2 = 0b00001100
    qc_bitmask3 = 0b00110000
    qc_bitmask4 = 0b11000000
    
    #Bit1 values: Basic QA flags
    QC_BIT1_GOOD = 0b00000000
    QC_BIT1_OTHER = 0b00000001
    QC_BIT1_CLOUD = 0b00000010
    QC_BIT1_NA = 0b00000011
    
    #Bit3 values: Emis Error
    QC_BIT3_1 = 0b00000000 #Average emissivity error <= 0.01
    QC_BIT3_2 = 0b00010000 #Average emissivity error <= 0.02
    QC_BIT3_3 = 0b00100000 #Average emissivity error <= 0.04
    QC_BIT3_4 = 0b00110000 #Average emissivity error > 0.04
    
    #Bit4 values: LST Error
    QC_BIT4_1 = 0b00000000 #Average LST error <= 1K
    QC_BIT4_2 = 0b01000000 #Average LST error <= 2K
    QC_BIT4_3 = 0b10000000 #Average LST error <= 3K
    QC_BIT4_4 = 0b11000000 #Average LST error > 3K
    
    os.chdir(in_path)
    fnames = np.array(os.listdir(in_path))
    fnames = np.sort(fnames[np.char.endswith(fnames,".hdf")])
    
    yrs = np.arange(2003,2013)
    yrs_mask = None
    for yr in yrs:
       
        yr_mask = np.char.startswith(fnames, "".join(["MYD11A2.A",str(yr)])) 
        
        if yrs_mask is None:
            yrs_mask = yr_mask
        else:
            yrs_mask = np.logical_or(yrs_mask,yr_mask)
        
    fnames = fnames[yrs_mask]
    sd = SD(fnames[0],SDC.READ)
    x,y = sd.datasets()[data_varname][1]
    fval = sd.select(data_varname).attributes()['_FillValue']
    sd.end()
    
    os.system("".join(["cp ",fnames[0]," ",out_fpath]))
    
    ds = Dataset(out_ncdf,'w')
    ds.createDimension('time',fnames.size)
    ds.createDimension('y',y)
    ds.createDimension('x',x)
    ds.createVariable(data_varname,np.int16,('time','y','x',),fill_value=fval)
    ds.createVariable('mth','i4',('time',))
    ds.createVariable('yr','i4',('time',))
    ds.sync()
    
    i = 0
    for fname in fnames:
        
        print fname
        
        sd = SD(fname,SDC.READ)
        sds_lst = sd.select(data_varname)
        lst_scale = sds_lst.attributes()['scale_factor']
        lst_offset = sds_lst.attributes()['add_offset']
        lst_fill = sds_lst.attributes()['_FillValue']
        lst = sds_lst[:]
        
        qc = sd.select(qc_varname)[:]
        qc1 = qc & qc_bitmask1
        qc3 = qc & qc_bitmask3
        qc4 = qc & qc_bitmask4
                
        qc_mask1 =  np.logical_or(qc1==QC_BIT1_CLOUD,qc1==QC_BIT1_NA)
        qc_mask2 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_3)
        qc_mask3 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_4)
        qc_mask4 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_3)
        qc_mask5 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_4)
        #qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5),dva_mask)
        qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5)
        
#        lst[np.logical_or(qc_fnl_mask,lst==lst_fill)] = np.nan
#        lst = (lst*lst_scale) + lst_offset
        #lst[qc_fnl_mask] = netCDF4.default_fillvals['f8']
        lst[qc_fnl_mask] = fval
        
        date = datetime.datetime.strptime(fname.split('.')[1][1:], '%Y%j')
        
        if np.ma.isMaskedArray(lst):
            lst = lst.data
        
        ds.variables[data_varname][i,:,:] = lst
        ds.variables['mth'][i] = date.month
        ds.variables['yr'][i] = date.year
        ds.sync()
        sd.end()
        
        i+=1
        
    #Read back data and calculate/output mean
    mth = ds.variables['mth'][:]
        
    lst_fnl = None
    for x in np.arange(1,13):
        
        lst_mth = ds.variables[data_varname][mth==x,:,:]
        lst_mth_mean = np.ma.mean(lst_mth,axis=0,dtype=np.float)
    
        if lst_fnl is None:
            lst_fnl = lst_mth_mean
        else:
            lst_fnl = np.ma.dstack((lst_fnl,lst_mth_mean))
            
    lst_fnl = np.ma.mean(lst_fnl,axis=2)
    
    #Remove pixels that are missing more than 75% of their obs
    lst = ds.variables[data_varname][:]
    mask = np.logical_not(lst.mask)
    pct_obs = np.sum(mask,axis=0)/np.float(mask.shape[0])
    lst_fnl.mask = np.logical_or(lst_fnl.mask,pct_obs < 0.25)
    
    lst_fnl_a = lst_fnl.data
    lst_fnl_a[lst_fnl.mask] = fval
    lst_fnl_a = np.ma.round(lst_fnl_a).astype(np.uint16)
    
#    plt.imshow(np.ma.MaskedArray(lst_fnl_a,mask=lst_fnl.mask))
#    plt.colorbar()
#    plt.show()
    
    print "WRITING TO: "+out_fpath
    sd = SD(out_fpath,SDC.WRITE)
    sds_lst = sd.select(data_varname)
    sds_lst[:] = lst_fnl_a
    sd.end()
    
#    lst_fnl = KtoC(lst_fnl)
#    
#    grid_sd = EOSGridSD(fnames[0])
#    grid_sd.to_geotiff(lst_fnl,out_fpath)
    
    ds.close()     

def stack_MYD11A2_hdf_8day(in_path,out_ncdf,out_fpath,ydayPeriod,data_varname='LST_Night_1km',qc_varname='QC_Night',va_varname='Night_view_angl'):
    
    qc_bitmask1 = 0b00000011
    qc_bitmask2 = 0b00001100
    qc_bitmask3 = 0b00110000
    qc_bitmask4 = 0b11000000
    
    #Bit1 values: Basic QA flags
    QC_BIT1_GOOD = 0b00000000
    QC_BIT1_OTHER = 0b00000001
    QC_BIT1_CLOUD = 0b00000010
    QC_BIT1_NA = 0b00000011
    
    #Bit3 values: Emis Error
    QC_BIT3_1 = 0b00000000 #Average emissivity error <= 0.01
    QC_BIT3_2 = 0b00010000 #Average emissivity error <= 0.02
    QC_BIT3_3 = 0b00100000 #Average emissivity error <= 0.04
    QC_BIT3_4 = 0b00110000 #Average emissivity error > 0.04
    
    #Bit4 values: LST Error
    QC_BIT4_1 = 0b00000000 #Average LST error <= 1K
    QC_BIT4_2 = 0b01000000 #Average LST error <= 2K
    QC_BIT4_3 = 0b10000000 #Average LST error <= 3K
    QC_BIT4_4 = 0b11000000 #Average LST error > 3K
    
    os.chdir(in_path)
    fnames = np.array(os.listdir(in_path))
    fnames = np.sort(fnames[np.char.endswith(fnames,".hdf")])
    fnamesYday = np.array([x[13:16] for x in fnames],dtype=np.int)
    
    yrs = np.arange(2003,2013)
    yrs_mask = None
    for yr in yrs:
       
        yr_mask = np.logical_and(np.char.startswith(fnames, "".join(["MYD11A2.A",str(yr)])),fnamesYday==ydayPeriod) 
        
        if yrs_mask is None:
            yrs_mask = yr_mask
        else:
            yrs_mask = np.logical_or(yrs_mask,yr_mask)
        
    fnames = fnames[yrs_mask]
    sd = SD(fnames[0],SDC.READ)
    x,y = sd.datasets()[data_varname][1]
    fval = sd.select(data_varname).attributes()['_FillValue']
    sd.end()
    
    os.system("".join(["cp ",fnames[0]," ",out_fpath]))
    
    ds = Dataset(out_ncdf,'w')
    ds.createDimension('time',fnames.size)
    ds.createDimension('y',y)
    ds.createDimension('x',x)
    ds.createVariable(data_varname,np.int16,('time','y','x',),fill_value=fval)
    ds.createVariable('mth','i4',('time',))
    ds.createVariable('yr','i4',('time',))
    ds.sync()
    
    i = 0
    for fname in fnames:
        
        print fname
        
        sd = SD(fname,SDC.READ)
        sds_lst = sd.select(data_varname)
        lst_scale = sds_lst.attributes()['scale_factor']
        lst_offset = sds_lst.attributes()['add_offset']
        lst_fill = sds_lst.attributes()['_FillValue']
        lst = sds_lst[:]
        
        qc = sd.select(qc_varname)[:]
        qc1 = qc & qc_bitmask1
        qc3 = qc & qc_bitmask3
        qc4 = qc & qc_bitmask4
                
        qc_mask1 =  np.logical_or(qc1==QC_BIT1_CLOUD,qc1==QC_BIT1_NA)
        qc_mask2 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_3)
        qc_mask3 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_4)
        qc_mask4 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_3)
        qc_mask5 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_4)
        #qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5),dva_mask)
        qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5)
        
#        lst[np.logical_or(qc_fnl_mask,lst==lst_fill)] = np.nan
#        lst = (lst*lst_scale) + lst_offset
        #lst[qc_fnl_mask] = netCDF4.default_fillvals['f8']
        lst[qc_fnl_mask] = fval
        
        date = datetime.datetime.strptime(fname.split('.')[1][1:], '%Y%j')
        
        if np.ma.isMaskedArray(lst):
            lst = lst.data
        
        ds.variables[data_varname][i,:,:] = lst
        ds.variables['mth'][i] = date.month
        ds.variables['yr'][i] = date.year
        ds.sync()
        sd.end()
        
        i+=1
        
    #Read back data and calculate/output mean
    mth = ds.variables['mth'][:]
        
    lst_fnl = None
    for x in np.arange(1,13):
        
        lst_mth = ds.variables[data_varname][mth==x,:,:]
        lst_mth_mean = np.ma.mean(lst_mth,axis=0,dtype=np.float)
    
        if lst_fnl is None:
            lst_fnl = lst_mth_mean
        else:
            lst_fnl = np.ma.dstack((lst_fnl,lst_mth_mean))
            
    lst_fnl = np.ma.mean(lst_fnl,axis=2)
    
    #Remove pixels that are missing more than 75% of their obs
    lst = ds.variables[data_varname][:]
    mask = np.logical_not(lst.mask)
    pct_obs = np.sum(mask,axis=0)/np.float(mask.shape[0])
    lst_fnl.mask = np.logical_or(lst_fnl.mask,pct_obs < 0.25)
    
    lst_fnl_a = lst_fnl.data
    lst_fnl_a[lst_fnl.mask] = fval
    lst_fnl_a = np.ma.round(lst_fnl_a).astype(np.uint16)
    
#    plt.imshow(np.ma.MaskedArray(lst_fnl_a,mask=lst_fnl.mask))
#    plt.colorbar()
#    plt.show()
    
    print "WRITING TO: "+out_fpath
    sd = SD(out_fpath,SDC.WRITE)
    sds_lst = sd.select(data_varname)
    sds_lst[:] = lst_fnl_a
    sd.end()
    
#    lst_fnl = KtoC(lst_fnl)
#    
#    grid_sd = EOSGridSD(fnames[0])
#    grid_sd.to_geotiff(lst_fnl,out_fpath)
    
    ds.close()  

def reset_MYD11A2_fillval(fpath,varname,new_fill=1):
    #varname: LST_Night_1km or LST_Day_1km
    sd = SD(fpath,SDC.WRITE)
    sds_lst = sd.select(varname)
    cur_fill = sds_lst.attributes()['_FillValue']
    
    sds_lst.setfillvalue(new_fill)
    a = sds_lst[:]
    a[a==cur_fill] = new_fill
    sds_lst[:] = a
    sd.end()
    
def stack_MYD11A2_netcdf(in_path,out_ncdf,out_fpath,data_varname='LST_Night_1km',qc_varname='QC_Night',va_varname='Night_view_angl'):
    
    qc_bitmask1 = 0b00000011
    qc_bitmask2 = 0b00001100
    qc_bitmask3 = 0b00110000
    qc_bitmask4 = 0b11000000
    
    #Bit1 values: Basic QA flags
    QC_BIT1_GOOD = 0b00000000
    QC_BIT1_OTHER = 0b00000001
    QC_BIT1_CLOUD = 0b00000010
    QC_BIT1_NA = 0b00000011
    
    #Bit3 values: Emis Error
    QC_BIT3_1 = 0b00000000 #Average emissivity error <= 0.01
    QC_BIT3_2 = 0b00010000 #Average emissivity error <= 0.02
    QC_BIT3_3 = 0b00100000 #Average emissivity error <= 0.04
    QC_BIT3_4 = 0b00110000 #Average emissivity error > 0.04
    
    #Bit4 values: LST Error
    QC_BIT4_1 = 0b00000000 #Average LST error <= 1K
    QC_BIT4_2 = 0b01000000 #Average LST error <= 2K
    QC_BIT4_3 = 0b10000000 #Average LST error <= 3K
    QC_BIT4_4 = 0b11000000 #Average LST error > 3K
    
    os.chdir(in_path)
    fnames = np.array(os.listdir(in_path))
    fnames = np.sort(fnames[np.char.endswith(fnames,".hdf")])
    
    yrs = np.arange(2003,2013)
    yrs_mask = None
    for yr in yrs:
       
        yr_mask = np.char.startswith(fnames, "".join(["MYD11A2.A",str(yr)])) 
        
        if yrs_mask is None:
            yrs_mask = yr_mask
        else:
            yrs_mask = np.logical_or(yrs_mask,yr_mask)
        
    fnames = fnames[yrs_mask]
    sd = SD(fnames[0],SDC.READ)
    x,y = sd.datasets()[data_varname][1]
    sd.end()
    
    ds = Dataset(out_ncdf,'w')
    ds.createDimension('time',fnames.size)
    ds.createDimension('y',y)
    ds.createDimension('x',x)
    ds.createVariable(data_varname,'f8',('time','y','x',))
    ds.createVariable('mth','i4',('time',))
    ds.createVariable('yr','i4',('time',))
    ds.sync()
    
    i = 0
    for fname in fnames:
        
        print fname
        
        sd = SD(fname,SDC.READ)
        sds_lst = sd.select(data_varname)
        lst_scale = sds_lst.attributes()['scale_factor']
        lst_offset = sds_lst.attributes()['add_offset']
        lst_fill = sds_lst.attributes()['_FillValue']
        lst = np.array(sds_lst[:],dtype = np.float64)
        
        qc = sd.select(qc_varname)[:]
        qc1 = qc & qc_bitmask1
        qc3 = qc & qc_bitmask3
        qc4 = qc & qc_bitmask4
        
#        sd_dva = sd.select(va_varname)
#        dva_scale = sd_dva.attributes()['scale_factor']
#        dva_offset = sd_dva.attributes()['add_offset']
#        dva_fill = sd_dva.attributes()['_FillValue']
#        dva = sd_dva[:].astype(np.float64)
#        dva[dva==dva_fill]  = np.nan
#        dva = (dva*dva_scale) + dva_offset
#        dva = np.abs(dva)
#        dva_mask = dva > 40
        
        qc_mask1 =  np.logical_or(qc1==QC_BIT1_CLOUD,qc1==QC_BIT1_NA)
        qc_mask2 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_3)
        qc_mask3 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_4)
        qc_mask4 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_3)
        qc_mask5 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_4)
        #qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5),dva_mask)
        qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5)
        
        lst[np.logical_or(qc_fnl_mask,lst_fill)] = np.nan
        lst = (lst*lst_scale) + lst_offset
        lst[qc_fnl_mask] = netCDF4.default_fillvals['f8']
        
        date = datetime.datetime.strptime(fname.split('.')[1][1:], '%Y%j')
                
        ds.variables[data_varname][i,:,:] = lst
        ds.variables['mth'][i] = date.month
        ds.variables['yr'][i] = date.year
        ds.sync()
        sd.end()
        
        i+=1
        
    #Read back data and calculate/output mean
    mth = ds.variables['mth'][:]
        
    lst_fnl = None
    for x in np.arange(1,13):
        
        lst_mth = ds.variables[data_varname][mth==x,:,:]
        lst_mth_mean = np.ma.mean(lst_mth,axis=0)
    
        if lst_fnl is None:
            lst_fnl = lst_mth_mean
        else:
            lst_fnl = np.ma.dstack((lst_fnl,lst_mth_mean))
            
    lst_fnl = np.ma.mean(lst_fnl,axis=2)
    
    #Remove pixels that are missing more than 75% of their obs
    lst = ds.variables[data_varname][:]
    mask = np.logical_not(lst.mask)
    pct_obs = np.sum(mask,axis=0)/np.float(mask.shape[0])
    lst_fnl.mask = np.logical_or(lst_fnl.mask,pct_obs < 0.25)
    
    lst_fnl = KtoC(lst_fnl)
    
    grid_sd = EOSGridSD(fnames[0])
    grid_sd.to_geotiff(lst_fnl,out_fpath)
    
    ds.close()

def stack_MYD11A2_netcdf_all():
    
    varnames = [('LST_Night_1km','QC_Night','Night_view_angl'),
                ('LST_Day_1km','QC_Day','Day_view_angl')]
    
    path = '/projects/daymet2/climate_office/modis/MYD11A2/'
    ncdf_path = "".join([path,'temp_lst.nc'])
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/'
    
    fnames = np.array(os.listdir(path))
    fnames = fnames[np.char.startswith(fnames,'MYD11A2.005.')]
    #fnames = fnames[np.char.startswith(fnames,'MYD11A2.005.h14v03')]
    
    for fname in fnames:
        
        fpath = "".join([path,fname,"/"])
        print "###########################################"
        print fname
        print "###########################################"
        
        for varname in varnames:
            
            out_fpath = "".join([out_path,varname[0],"_",fname,".hdf"])
            #stack_MYD11A2_netcdf(fpath, ncdf_path, out_fpath, varname[0], varname[1], varname[2])
            stack_MYD11A2_hdf(fpath, ncdf_path, out_fpath, varname[0], varname[1], varname[2])
        
def process_stack_MYD11A2(in_path,out_fpath,data_varname='LST_Night_1km',qc_varname='QC_Night'):
    os.chdir(in_path)
    qc_bitmask = 0b00000011
    
    fnames = np.array(os.listdir(in_path))
    fnames = np.sort(fnames[np.char.endswith(fnames,".hdf")])
    
    yrs = np.arange(2003,2013)
    yrs_mask = None
    for yr in yrs:
       
        yr_mask = np.char.startswith(fnames, "".join(["MYD11A2.A",str(yr)])) 
        
        if yrs_mask is None:
            yrs_mask = yr_mask
        else:
            yrs_mask = np.logical_or(yrs_mask,yr_mask)
        
    fnames = fnames[yrs_mask]
    
#    if fnames.size != 414:
#        raise Exception("Incorrect # of files for "+in_path+" | ",str(fnames.size)," total files")
    
    lst_ls = []
    for fname in fnames:
        sd = SD(fname,SDC.READ)
        sds_lst = sd.select(data_varname)
        lst_scale = sds_lst.attributes()['scale_factor']
        lst_fill = sds_lst.attributes()['_FillValue']
        lst = np.array(sds_lst[:],dtype = np.float64)
        fill_mask = lst==lst_fill
        
        sds_qc = sd.select(qc_varname)[:]
        
        #mask out all but bits 0 - 1
        qc_masked = sds_qc & qc_bitmask
        #mask out values that are 10, or 11 in bits 0-1
        qc_mask = np.logical_or(qc_masked==0b10,qc_masked==0b11)
        
        #lst = np.ma.masked_array(lst,np.logical_or(fill_mask,qc_mask),fill_value=np.nan)
        lst[np.logical_or(fill_mask,qc_mask)] = np.nan
        lst = lst*lst_scale
        lst_ls.append(lst)
        #print fname,np.sum(qc_mask)
#        plt.imshow(lst)
#        plt.colorbar()
#        plt.show()
    
    lst_stack = np.dstack(lst_ls)
    lst_stack = np.ma.masked_array(lst_stack,np.isnan(lst_stack),fill_value=np.nan)
    lst_mean = np.ma.mean(lst_stack,axis=2)
    lst_mean = KtoC(lst_mean)
    
    grid_sd = EOSGridSD(fnames[0])
    grid_sd.to_geotiff(lst_mean,out_fpath)

def create_montana_mask():
    mt_shp = ogr.Open('/projects/crown_ws/montanaboundary_GIS/states_WGS84.shp')
    mt_lyr = mt_shp.GetLayer()
    
    rastin = gdal.Open('/projects/daymet2/climate_office/modis/MOD13A3/jja_tifs_wgs84/mask_test.tif')
    a = rastin.GetRasterBand(1).ReadAsArray()
    nrows,ncols = a.shape
    
    driver = gdal.GetDriverByName("GTiff")
    rastout = driver.Create('/projects/daymet2/climate_office/modis/MOD13A3/jja_tifs_wgs84/mt_mask.tif',ncols,nrows,1,gdal.GDT_Byte)
    rastout.SetGeoTransform(rastin.GetGeoTransform())
    rastout.SetProjection(rastin.GetProjection())
    band = rastout.GetRasterBand(1)
    band.Fill(0) #initialise raster with zeros
    #band.SetNoDataValue(0)
    rastout.FlushCache()
    gdal.RasterizeLayer(rastout, [1], mt_lyr)
    rastout.FlushCache()
    
    mask_arr=rastout.GetRasterBand(1).ReadAsArray()
    plt.imshow(mask_arr)
    plt.show()
    
def reproject_MOD13A3_gtiffs():
    path = '/projects/daymet2/climate_office/modis/MOD13A3/jja_tifs_mosaic/'
    path_out = '/projects/daymet2/climate_office/modis/MOD13A3/jja_tifs_wgs84/'
    
    os.chdir(path)
    fnames = np.array(os.listdir(path))
    fnames = fnames[np.char.endswith(fnames, ".tif")]
    
    for fname in fnames:
        cmd = 'gdalwarp -s_srs "+proj=sinu +R=6371007.181 +nadgrids=@null +no_defs +wktext" -t_srs EPSG:4326 -dstnodata -9999 '
        fname_out = "".join([path_out,fname])
        cmd = cmd + fname + " " + fname_out
        print cmd
        os.system(cmd)
        
        
def mosaic_MOD13A3_gtiffs():
    path = '/projects/daymet2/climate_office/modis/MOD13A3/jja_tifs/'
    path_out = '/projects/daymet2/climate_office/modis/MOD13A3/jja_tifs_mosaic/'
    
    os.chdir(path)
    fnames = np.array(os.listdir(path))
    fnames = fnames[np.char.endswith(fnames, ".tif")]
    
    yrs = np.arange(2000,2013)
    mths = ["june","july","august"]
    
    #for yr in yrs:
    yr = "mean"   
    for mth in mths:
        
        date_str = ".".join([str(yr),mth])
        mth_fnames = fnames[np.char.find(fnames, date_str) != -1]
        
        if len(mth_fnames) != 3:
            raise Exception("Incorrect # of files: "+str(yr)+"-",mth)
        
            
        cmd = ["".join(['gdalwarp -s_srs "+proj=sinu +R=6371007.181 +nadgrids=@null +no_defs +wktext" -t_srs "+proj=sinu +R=6371007.181 +nadgrids=@null +no_defs +wktext" -dstnodata -9999'])]
        cmd.extend(mth_fnames)
        cmd.append("".join([path_out,"mosaic.ndvi.",str(yr),".",mth,".tif"]))
    
        print "running mosaic cmd..."+" ".join(cmd)
        os.system(" ".join(cmd))
            
    
def process_stack_MOD13A3():
    tiles = ['h09v04','h10v04','h11v04']
    path = '/projects/daymet2/climate_office/modis/MOD13A3/jja/'
    path_out = '/projects/daymet2/climate_office/modis/MOD13A3/jja_tifs/'
    os.chdir(path)
    nyrs = 13
    
    months = {153:"june",183:"july",214:"august",
              152:"june",182:"july",213:"august",}
    
    fnames = np.array(os.listdir(path))
    fnames = fnames[np.char.endswith(fnames, ".hdf")]
    
    for tile in tiles:
        
        tile_files = fnames[np.char.find(fnames, tile) != -1]
        
        ndvi_ls = {"june":[],"july":[],"august":[]}
        for afile in tile_files:
            
            yr = afile[9:13]
            jday = int(afile[13:16])
            month = months[jday]
            
            sd = SD(afile,SDC.READ)
            sds_ndvi = sd.select('1 km monthly NDVI')
            sds_qa = sd.select('1 km monthly pixel raliability')
            
            ndvi = np.array(sds_ndvi[:],dtype=np.float64)
            qa = sds_qa[:]
            
            ndata_ndvi = ndvi == FILL_VALUE_NDVI
            ndata_qa = np.logical_and(qa != 0, qa != 1)
            
            ndvi[np.logical_or(ndata_ndvi,ndata_qa)] = np.nan
            
            ndvi = ndvi*SCALE_NDVI
            ndvi[np.isnan(ndvi)] = -9999
            
            fname_out = ".".join([tile,yr,month,"tif"])
            
            grid_sd = EOSGridSD(afile)
            grid_sd.to_geotiff(ndvi,"".join([path_out,fname_out]))
            
            ndvi_ls[month].append(ndvi)
            print fname_out
        
        for mth,als in ndvi_ls.items():
            if len(als) != nyrs:
                raise Exception("Incorrect # of years "+tile)
            
            ndvi_stack = np.dstack(als)
            ndvi_stackm = np.ma.masked_array(ndvi_stack,ndvi_stack==-9999)
            ndvi_mean = np.ma.mean(ndvi_stackm,axis=2)
            
            fname_out = ".".join([tile,"mean",mth,"tif"])
            grid_sd.to_geotiff(ndvi_mean,"".join([path_out,fname_out]))

def process_stack():
    os.chdir(PATH_MOD13A3)
    
    ndvi_ls = []
    for yr in YEARS:
        sd = SD("".join(["MOSAIC.MOD13A3.A",str(yr),".hdf"]),SDC.READ)
        sds_ndvi = sd.select('1 km monthly NDVI')
        sds_qa = sd.select('1 km monthly pixel raliability')
        
        ndvi = np.array(sds_ndvi[:],dtype=np.float64)
        qa = sds_qa[:]
        
        ndata_ndvi = ndvi == FILL_VALUE_NDVI
        ndata_qa = np.logical_and(qa != 0, qa != 1)
        
        ndvi[np.logical_or(ndata_ndvi,ndata_qa)] = np.nan
        
        ndvi = ndvi*SCALE_NDVI
        
        ndvi_ls.append(ndvi)
        
        print yr
    
    ndvi = np.dstack(ndvi_ls)
    
    ndvi_m = np.ma.masked_array(ndvi,np.isnan(ndvi))
    
    ndvi_mean = np.mean(ndvi_m,axis=2)
    
    ndvi_2012 = ndvi[:,:,-1]
    ndvi_2010 = ndvi[:,:,-3]
    
    ndvi_anom = (ndvi_2010-ndvi_mean)/ndvi_mean
    
    ndvi_anom = np.ma.masked_array(ndvi_anom,np.logical_or(ndvi_mean.mask,np.isnan(ndvi_2010)))
    
    ndvi_2012 = np.ma.masked_array(ndvi_2012,np.isnan(ndvi_2012))
    
    #output_raster(ndvi_2012, "ndvi_2012.tif")
    output_raster(ndvi_anom, "ndvi_2010_anom.tif")


    
#    print np.min(ndvi_anom),np.max(ndvi_anom)
#    plt.imshow(ndvi_anom)
#    plt.colorbar()
#    plt.show()

def reproject_mean_anom_h10v04_aug():
    path = '/projects/daymet2/climate_office/modis/MOD13Q1.005/aug_gtiff/'
    os.chdir(path)
    fnames = ["aug_ndvi_2007.tif","aug_ndvi_2010.tif","aug_ndvi_mean.tif"]
    for fname in fnames:
        cmd = 'gdalwarp -s_srs "+proj=sinu +R=6371007.181 +nadgrids=@null +no_defs +wktext" -r bilinear -tr 0.002946283 0.002946283 -t_srs EPSG:4326 -dstnodata -9999 '
        fname_out = "".join(["wgs84_",fname])
        cmd = cmd + fname + " " + fname_out
        print cmd
        os.system(cmd)

def calc_mean_anom_h10v04_aug():
    
    path = '/projects/daymet2/climate_office/modis/MOD13Q1.005/aug_gtiff/'
    fpath_out = '/projects/daymet2/climate_office/modis/MOD13Q1.005/aug_gtiff/aug_ndvi_mean.tif'
    yrs = np.arange(2000,2013)
    os.chdir(path)
    
    ndvi_stack = []
    for yr in yrs:
        
        print yr
        rastin = gdal.Open("".join(['aug_ndvi_',str(yr),".tif"]))
        a = np.array(rastin.GetRasterBand(1).ReadAsArray(),dtype=np.float64)
        ndvi_stack.append(a)
    
    ndvi_stack = np.dstack(ndvi_stack)
    
    ndvi_stackm = np.ma.masked_array(ndvi_stack,ndvi_stack==-9999)
    ndvi_mean = np.ma.mean(ndvi_stackm,axis=2)
    
    driver = gdal.GetDriverByName("GTiff")
    nrows,ncols = ndvi_mean.shape

    dtype = gdalconst.GDT_Float32
    ndata=-9999
    rastin = gdal.Open("".join(['aug_ndvi_',str(yrs[0]),".tif"]))

    raster = driver.Create(fpath_out,ncols,nrows,1, dtype)
    raster.SetGeoTransform(rastin.GetGeoTransform())
    raster.SetProjection(rastin.GetProjection()) 
    
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(ndata)
    
    a_d = ndvi_mean.data
    a_d[ndvi_mean.mask] = ndata
    
    band.WriteArray(a_d,0,0)  
    band.FlushCache()
            
def process_h10v04_august():
    
    path = '/projects/daymet2/climate_office/modis/MOD13Q1.005/'
    path_out = '/projects/daymet2/climate_office/modis/MOD13Q1.005/aug_gtiff/'
    tile = "h10v04"
        
    os.chdir(path)

    fnames = np.array(os.listdir(path))
    fnames = fnames[np.char.endswith(fnames, ".hdf")]
    
    yrs = np.arange(2000,2013)
    
    for yr in yrs:
        print yr
        fnames_yr = fnames[np.char.startswith(fnames, "".join(['MOD13Q1.A',str(yr)]))]
        ydays = np.array([int(x[13:16]) for x in fnames_yr])
        
        fnames_aug = fnames_yr[ydays >= 209]
        
        ndvi_stack = []
        
        for afile in fnames_aug:
            
            sd = SD(afile,SDC.READ)
            sds_ndvi = sd.select('250m 16 days NDVI')
            sds_qa = sd.select('250m 16 days pixel reliability')
            
            ndvi = np.array(sds_ndvi[:],dtype=np.float64)
            
            qa = sds_qa[:]
            
            ndata_ndvi = ndvi == FILL_VALUE_NDVI
            ndata_qa = np.logical_and(qa != 0, qa != 1)
            
            ndvi[np.logical_or(ndata_ndvi,ndata_qa)] = np.nan
            
            ndvi = ndvi*SCALE_NDVI
            
            ndvi_stack.append(ndvi)
        
        ndvi_stack = np.dstack(ndvi_stack)
        
        ndvi_stackm = np.ma.masked_array(ndvi_stack,np.isnan(ndvi_stack))
        ndvi_mean = np.ma.mean(ndvi_stackm,axis=2)
        
        grid_sd = EOSGridSD(fnames_aug[0])
        grid_sd.to_geotiff(ndvi_mean,"".join([path_out,"aug_ndvi_",str(yr),".tif"]))
            

def KtoC(k):
    return k - 273.15
        
def mosaic_modis():
    wdir = '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/'
    outdir = '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/'
    prefix = 'LST_Night'
    
    fnames = np.array(os.listdir(wdir))
    
    fnames = fnames[np.logical_and(np.char.endswith(fnames, '.hdf'),
                                   np.char.startswith(fnames, prefix))]
        
    fout = open("".join([outdir,"mosaic_file_list.txt"]),'w')
        
    fname_lines = ["".join([wdir,x,"\n"]) for x in fnames]
        
    fout.writelines(fname_lines)
    fout.close()
    
    os.chdir(outdir)
        
    os.system("".join(["mrtmosaic -i mosaic_file_list.txt -o","".join(["MOSAIC.",prefix,".hdf"])]))

def cp_modis():
    
    OUT_DIR = "/projects/daymet2/climate_office/modis/MOD13A2/"
    
    for date_dir in TILE_DATE_DIRS:
        
        dpath = "".join([PATH_MOD13A2,date_dir,"/"])
        
        fnames = np.array(os.listdir(dpath))
        
        for tname in TILES_MONTANA:
            
            fnames_tile = fnames[np.char.find(fnames,tname) != -1]
            
            for fname_tile in fnames_tile:
                
                fpath_in = "".join([dpath,fname_tile])
                fpath_out = "".join([OUT_DIR,fname_tile])
                
                cp_cmd = "".join(["cp ",fpath_in," ",fpath_out])
                print cp_cmd
                os.system(cp_cmd)

def mosaic_modis_txt():
    OUT_DIR = "/projects/daymet2/climate_office/modis/MOD13A2/"
    
    fnames = np.array(os.listdir(OUT_DIR))
    
    fout = open("".join([OUT_DIR,"mosaic_file_list.txt"]),'w')
    
    fnames = ["".join([OUT_DIR,x,"\n"]) for x in fnames]
    
    fout.writelines(fnames)
    fout.close()
    
class EOSGridSD():
    
    def __init__(self, path, mode=SDC.READ):
        
        self.sds = SD(path,mode)
        
        #Get struct metadata
        gridmeta = self.sds.attributes()['StructMetadata.0'].split("\n")
        gridmeta = np.array([x.strip() for x in gridmeta])
        
        xdim = int(gridmeta[np.char.startswith(gridmeta,"XDim")][0].split("=")[1])
        ydim = int(gridmeta[np.char.startswith(gridmeta,"YDim")][0].split("=")[1])
        
        ul_pt = gridmeta[np.char.startswith(gridmeta,"UpperLeftPointMtrs")][0].split("=")[1]
        ul_x,ul_y = ul_pt.split(",")
        ul_x,ul_y = float(ul_x[1:]),float(ul_y[:-1])
        
        lr_pt = gridmeta[np.char.startswith(gridmeta,"LowerRightMtrs")][0].split("=")[1]
        lr_x,lr_y = lr_pt.split(",")
        lr_x,lr_y = float(lr_x[1:]),float(lr_y[:-1])
            
        '''
        Create GDAL-style geotransform list to define resolution and bounds
        GeoTransform[0] /* top left x */
        GeoTransform[1] /* w-e pixel resolution */
        GeoTransform[2] /* rotation, 0 if image is "north up" */
        GeoTransform[3] /* top left y */
        GeoTransform[4] /* rotation, 0 if image is "north up" */
        GeoTransform[5] /* n-s pixel resolution */
        '''
        geotransform = [None]*6
        geotransform[0] = ul_x
        geotransform[3] = ul_y
        geotransform[2],geotransform[4] = (0.0,0.0)
        geotransform[1] = np.abs((ul_x-lr_x)/xdim)
        #n-s pixel height/resolution needs to be negative
        geotransform[5] = -np.abs((ul_y-lr_y)/ydim)
        self.geotransform = geotransform
        
        self.sr_sin = osr.SpatialReference()
        self.sr_sin.ImportFromProj4(PROJ4_MODIS)
        self.sr_wgs84 = osr.SpatialReference()
        self.sr_wgs84.ImportFromEPSG(EPSG_WGS84)
        
        self.trans_sin_to_wgs84 = osr.CoordinateTransformation(self.sr_sin, self.sr_wgs84)
        self.trans_wgs84_to_sin = osr.CoordinateTransformation(self.sr_wgs84, self.sr_sin)
        
    def get_xy(self,lon,lat):
        '''Returns the nearest-ngh grid cell offset for this raster based on the input wgs84 lon/lat'''
        
        x_sin, y_sin, z_sin = self.trans_wgs84_to_sin(lon,lat)
            
        originX = self.geo_t[0]
        originY = self.geo_t[3]
        pixelWidth = self.geo_t[1]
        pixelHeight = self.geo_t[5]
        
        xOffset = abs(int((x_sin - originX) / pixelWidth))
        yOffset = abs(int((y_sin - originY) / pixelHeight))
        return xOffset,yOffset
    
    def get_data_value(self,lon,lat,data_array):
        
        x,y = self.get_xy(lon,lat)
        return data_array[x,y]
    
    def to_geotiff(self,data_array,fpath,dtype=gdalconst.GDT_Float32,ndata=-9999):
        
        driver = gdal.GetDriverByName("GTiff")
        nrows = data_array.shape[0]
        ncols = data_array.shape[1]

        raster = driver.Create(fpath,ncols,nrows,1, dtype)
        raster.SetGeoTransform(self.geotransform)
        raster.SetProjection(self.sr_sin.ExportToWkt()) 
        
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(ndata)
        
        if np.ma.is_masked(data_array):
            a_d = data_array.data
            a_d[data_array.mask] = ndata
        else:
            a_d = data_array
        
        band.WriteArray(a_d,0,0)  
        band.FlushCache()

def stack_all_MYD11A2_mth():
            
    TILES = ['h07v05',
            'h07v06',
            'h08v04',
            'h08v05',
            'h08v06',
            'h09v04',
            'h09v05',
            'h09v06',
            'h10v03',
            'h10v04',
            'h10v05',
            'h10v06',
            'h11v03',
            'h11v04',
            'h11v05',
            'h11v06',
            'h12v04',
            'h12v05',
            'h13v04']
    
    for tile in TILES:
        
        in_path = "".join(["/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.",tile,"/"])
        
        for mth in np.sort(MYD11A2_MTH_DAYS.keys()):
             
        
            out_fpath = "".join(["/projects/daymet2/climate_office/modis/MYD11A2/mthmean_gtiffs_ngt/lst_mean_ngt_",MTH_ABBRV[mth],"_",tile,".tif"])
            #out_fpath = "".join(["/projects/daymet2/climate_office/modis/MYD11A2/mthmean_gtiffs_day/lst_mean_day_",MTH_ABBRV[mth],"_",tile,".tif"])
            try:
                #process_stack_MYD11A2_mth(in_path,out_fpath,mth,'LST_Day_1km','QC_Day')
                process_stack_MYD11A2_mth(in_path,out_fpath,mth)
            except Exception:
                print "Could not stack tile: "+tile
            
            print tile+"|"+MTH_ABBRV[mth]

def stack_all_MYD11A2():
        
    #TILES = ['h11v05','h09v06','h09v04','h07v06','h10v06','h09v05','h08v04','h12v05','h10v04','h11v04','h08v05','h12v04']#h10v05
    #TILES = ['h08v04']
    #TILES = ['h12v05','h10v04','h11v04','h08v05','h12v04']
    #TILES = ['h13v04','h07v05','h11v06']
    #TILES = ['h11v06']
    #TILES = ['h08v06']
    #TILES = ['h10v03','h11v03']
    #TILES = ['h11v03']
    
    TILES = ['h07v05',
            'h07v06',
            'h08v04',
            'h08v05',
            'h08v06',
            'h09v04',
            'h09v05',
            'h09v06',
            'h10v03',
            'h10v04',
            'h10v05',
            'h10v06',
            'h11v03',
            'h11v04',
            'h11v05',
            'h11v06',
            'h12v04',
            'h12v05',
            'h13v04']
    
#    TILES_CMPLT = ['h07v05',
#    'h07v06',
#    'h08v05',
#    'h08v06',
#    'h09v04',
#    'h09v05',
#    'h09v06',
#    'h10v03',
#    'h10v04',
#    'h10v05',
#    'h10v06',
#    'h11v03',
#    'h11v04',
#    'h11v05',
#    'h12v04',
#    'h12v05',
#    'h13v04']
#    
#    TILES = np.array(TILES)
#    TILES_CMPLT = np.array(TILES_CMPLT)
#    
#    TILES = TILES[np.logical_not(np.in1d(TILES, TILES_CMPLT, True))]
    #print TILES
    
    for tile in TILES:
        print tile
        in_path = "".join(["/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.",tile,"/"])
        #out_fpath = "".join(["/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs/lst_mean_ngt_",tile,".tif"])
        out_fpath = "".join(["/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs_ngt/lst_mean_ngt_",tile,".tif"])
        try:
            #process_stack_MYD11A2(in_path,out_fpath,'LST_Day_1km','QC_Day')
            process_stack_MYD11A2(in_path,out_fpath)
        except Exception:
            print "Could not stack tile: "+tile
            
def mask_hdf_filenames(fnames,yrs):

    fnames = np.sort(fnames[np.char.endswith(fnames,".hdf")])
    
    yrs_mask = None
    for yr in yrs:
       
        yr_mask = np.char.startswith(fnames, "".join(["MYD11A2.A",str(yr)])) 
        
        if yrs_mask is None:
            yrs_mask = yr_mask
        else:
            yrs_mask = np.logical_or(yrs_mask,yr_mask)
        
    fnames = fnames[yrs_mask]
    return fnames
    


def debug_missing_tiles():
    yrs = np.arange(2003,2012)
    fnames1 = mask_hdf_filenames(np.array(os.listdir("/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.h10v05/")),yrs)
    fnames2 = mask_hdf_filenames(np.array(os.listdir("/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.h11v03/")),yrs)
    
    fnames1 = np.char.split(fnames1, ".", 2)
    fnames1 = np.array([x[1] for x in fnames1])
    
    fnames2 = np.char.split(fnames2, ".", 2)
    fnames2 = np.array([x[1] for x in fnames2])
    
    print fnames1[np.logical_not(np.in1d(fnames1, fnames2, True))]
    
def mosaic_MYD11A2_tiffs():
    
    path = "/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs3/day/"
    os.chdir(path)
    fnames = os.listdir(path)
    cmd = "gdalwarp -dstnodata -9999"
    fin_str = " ".join(fnames)
    cmd = " ".join([cmd,fin_str,"mosaic_mean_lst_tmax.tif"])
    os.system(cmd)
    
def mosaic_MOD12Q1_tiffs():
    
    path = "/projects/daymet2/climate_office/modis/MCD12Q1.051/gtiffs/"
    os.chdir(path)
    fnames = os.listdir(path)
    cmd = "gdalwarp -dstnodata 255"
    fin_str = " ".join(fnames)
    cmd = " ".join([cmd,fin_str,"mosaic_lc.tif"])
    os.system(cmd)
    
def mosaic_MOD44B_tiffs():
    
    path = "/projects/daymet2/climate_office/modis/MOD44B/"
    os.chdir(path)
    fnames = os.listdir(path)
    cmd = "gdalwarp -dstnodata 253"
    fin_str = " ".join(fnames)
    cmd = " ".join([cmd,fin_str,"mosaic_vcf.tif"])
    os.system(cmd)
    
def mosaic_MYD11A2_tiffs_mth():
    
    inpath = "/projects/daymet2/climate_office/modis/MYD11A2/mthmean_gtiffs_day/"
    outpath = '/projects/daymet2/climate_office/modis/MYD11A2/mthmean_day_mosaics/'
    os.chdir(inpath)
    fnames = np.array(os.listdir(inpath))
    
    for mth in np.sort(MTH_ABBRV.keys()):
        
        fnames_mth = fnames[np.char.find(fnames, MTH_ABBRV[mth]) != -1]
        foutpath = "".join([outpath,"lst_day_mosaic_",MTH_ABBRV[mth],".tif"])
    
        cmd = "gdalwarp -dstnodata -9999"
        fin_str = " ".join(fnames_mth)
        cmd = " ".join([cmd,fin_str,foutpath])
        os.system(cmd)
        sys.exit()

def extract_MCD12Q1_gtiffs(inpath,outpath):
    
    fnames = np.array(os.listdir(inpath))
    fnames = fnames[np.char.endswith(fnames, ".hdf")]
    
    os.chdir(inpath)
    
    for fname in fnames:
        sd = EOSGridSD(fname)
        a_lc = sd.sds.select('Land_Cover_Type_1')[:]
        
        fpath_out = "".join([outpath,fname[0:-4],".tif"])
        
        sd.to_geotiff(a_lc, fpath_out, gdalconst.GDT_Byte,255)

def extract_MYD11A2_gtiff(inpath,outpath,varname):
    sd = EOSGridSD(inpath)
    lst = sd.sds.select(varname)[:]
    sd.to_geotiff(lst, outpath, gdalconst.GDT_UInt16, 0)
    
    
    
        
if __name__ == '__main__':
    
#    reset_MYD11A2_fillval('/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.hdf', 
#                          'LST_Night_1km')
    
    reset_MYD11A2_fillval('/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_day_mosaic/MOSAIC.LST_Day.hdf', 
                          'LST_Day_1km')
    
#    extract_MYD11A2_gtiff('/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.hdf', 
#                          '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_night_mosaic/MOSAIC.LST_Night.tif','LST_Night_1km')
    
#    extract_MYD11A2_gtiff('/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_day_mosaic/MOSAIC.LST_Day.hdf', 
#                          '/projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_day_mosaic/MOSAIC.LST_Day.tif','LST_Day_1km')
    #mosaic_MOD12Q1_tiffs()
    
#    extract_MCD12Q1_gtiffs('/projects/daymet2/climate_office/modis/MCD12Q1.051/',
#                           '/projects/daymet2/climate_office/modis/MCD12Q1.051/gtiffs/')
    
    #mosaic_MOD44B_tiffs()
    #mosaic_MOD12Q1_tiffs()
    
#    tiles = ['h07v05',
#            'h07v06',
#            'h08v04',
#            'h08v05',
#            'h08v06',
#            'h09v04',
#            'h09v05',
#            'h09v06',
#            'h10v03',
#            'h10v04',
#            'h10v05',
#            'h10v06',
#            'h11v03',
#            'h11v04',
#            'h11v05',
#            'h11v06',
#            'h12v04',
#            'h12v05',
#            'h13v04',
#            'h09v03',
#            'h12v03',
#            'h13v03',
#            'h14v03',
#            'h14v04']
#    #tiles = ['h10v03']
#    for tile in tiles:
#        print tile
#        stack_MOD44B(tile,'/MODIS/Mirror/MOD44B.005/','/projects/daymet2/climate_office/modis/MOD44B/')    
    
    
    #process_MOD12Q1(tiles, '/MODIS/Mirror/MOD12Q1.Special.004/','/projects/daymet2/climate_office/modis/MOD12Q1/')
    
    #mosaic_MYD11A2_tiffs()
    #stack_MYD11A2_netcdf_all()
    
#    stack_MYD11A2_netcdf('/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.h09v04/',
#                         '/projects/daymet2/climate_office/modis/MYD11A2/h09v04_tmin.nc',
#                         '/projects/daymet2/climate_office/modis/MYD11A2/h09v04_tmin.tif')
    #reproject_mean_anom_h10v04_aug()
    #calc_mean_anom_h10v04_aug()
    #process_h10v04_august()
    #mosaic_MYD11A2_tiffs()
    #mosaic_MYD11A2_tiffs_mth()
    #stack_all_MYD11A2_mth()
    #mosaic_MYD11A2_tiffs()
    #process_stack_MOD13A3()
    #create_montana_mask()
    #reproject_MOD13A3_gtiffs()
    #mosaic_MOD13A3_gtiffs()
    #process_stack_MOD13A3()
    #debug_missing_tiles()
    #stack_all_MYD11A2()
#    process_stack_MYD11A2("/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.h10v05/", 
#                          "/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs/lst_mean_ngt_h10v05.tif")
    #dmodis = downModis("jared.oyler@ntsg.umt.edu","/projects/daymet2/climate_office/modis/MYD11A2.005.h09v04",tiles="h09v04",path="MOLA/MYD11A2.005",debug=True)
    #dmodis.connectFTP()
    #dmodis.downloadsAllDay(allDays=True)
    #process_stack_MYD11A2()
    #download("/projects/daymet2/climate_office/modis/data_url_script_2012-11-16_180230.txt", "/projects/daymet2/climate_office/modis/MOD13Q1.005")
    #sd = EOSGridSD('/projects/daymet2/climate_office/modis/MOD11A2.A2011361.h10v04.005.2012025175244.hdf')
    #process_stack()
    #mosaic_modis()
    #mosaic_modis_txt() 
    
    '''
    h11v05
    h10v05
    h09v06
    h13v04*
    h09v04
    h07v06
    h10v06
    h09v05
    h07v05*
    h11v06*
    h08v04
    h12v05
    h10v04
    h11v04
    h08v05
    h12v04
    h08v04
    h10v05
    h08v06
    '''    
