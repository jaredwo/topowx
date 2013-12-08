'''
Created on Sep 10, 2013

@author: jared.oyler
'''

import numpy as np
import os
import osgeo.gdal as gdal
import osgeo.osr as osr
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt


import osgeo.gdalconst as gdalconst
from downmodis import downModis
from utils.input_raster import RasterDataset
import scipy.spatial.distance as scpydist
import osgeo.ogr as ogr
import sys
from netCDF4 import Dataset,date2num,num2date
import netCDF4
import datetime
from utils.status_check import status_check
from db.station_data import station_data_ncdb, station_data_infill,BAD,STN_ID,YEAR,LON,LAT,\
    DTYPE_STN_BASIC,DATE,MONTH
from db.all_create_db import dbDataset
import utils.util_dates as utld
from db.ushcn import TairAggregate
import utils.util_geo as utlg
from scipy import stats
from utils.status_check import status_check
import cProfile
from matplotlib.mlab import griddata 
#from modis.montana_ndvi import EOSGridSD

#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri
r = robjects.r

import osgeo.gdal as gdal
import osgeo.osr as osr

MYD11A2_MTH_DAYS = {1:[1,9,17,25],
                    2:[33,41,49,57],
                    3:[65,73,81,89],
                    4:[97,105,113,121],
                    5:[121,129,137,145],
                    6:[153,161,169,177],
                    7:[185,193,201,209],
                    8:[217,225,233,241],
                    9:[249,257,265,273],
                    10:[281,289,297,305],
                    11:[305,313,321,329],
                    12:[337,345,353,361]}

MYD11A2_MTH_DAYS8 = {1:[353,361,1,9,17,25,33,41],
                    2:[17,25,33,41,49,57,65,73],
                    3:[49,57,65,73,81,89,97,105],
                    4:[81,89,97,105,113,121,129,137],
                    5:[105,113,121,129,137,145,153,161],
                    6:[137,145,153,161,169,177,185,193],
                    7:[169,177,185,193,201,209,217,225],
                    8:[201,209,217,225,233,241,249,257],
                    9:[233,241,249,257,265,273,281,289],
                    10:[265,273,281,289,297,305,313,321],
                    11:[289,297,305,313,321,329,337,345],
                    12:[321,329,337,345,353,361,1,9]}

MODIS_8DAY = np.array([1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97,
                        105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
                        209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305,
                        313, 321, 329, 337, 345, 353, 361])
MODIS_TILESIZE = 1200

EPSG_WGS84 = 4326 #EPSG Code
EPSG_NAD83 = 4269 #EPSG Code
PROJ4_MODIS = "+proj=sinu +R=6371007.181 +nadgrids=@null +no_defs +wktext"
#PROJ4_MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"

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
        
        sd_dva = sd.select(va_varname)
        dva_scale = sd_dva.attributes()['scale_factor']
        dva_offset = sd_dva.attributes()['add_offset']
        dva_fill = sd_dva.attributes()['_FillValue']
        dva = sd_dva[:].astype(np.float64)
        dva[dva==dva_fill]  = np.nan
        dva = (dva*dva_scale) + dva_offset
        dva = np.abs(dva)
        dva_mask = dva > 40
        
        qc = sd.select(qc_varname)[:]
        qc1 = qc & qc_bitmask1
        qc3 = qc & qc_bitmask3
        qc4 = qc & qc_bitmask4
                
        qc_mask1 =  np.logical_or(qc1==QC_BIT1_CLOUD,qc1==QC_BIT1_NA)
        qc_mask2 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_3)
        qc_mask3 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_4)
        qc_mask4 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_3)
        qc_mask5 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_4)
        qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5),dva_mask)
        #qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5)
        
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
    lst_fnl = ds.variables[data_varname][:]   
    lst_fnl = np.ma.mean(lst_fnl,axis=0)
    
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
    
    ds.close()

def stack_MYD11A2_hdf_mth(in_path,out_ncdf,out_fpath,mth,data_varname='LST_Night_1km',qc_varname='QC_Night',va_varname='Night_view_angl'):
    
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
    fnames = fnames[np.char.startswith(fnames,"MYD")]
    fnamesYday = np.array([x[13:16] for x in fnames],dtype=np.int)
    fnamesYear = np.array([x[9:13] for x in fnames],dtype=np.int)
    
    mthYdays = MYD11A2_MTH_DAYS[mth]
    yrs = np.arange(2003,2013)
    
    mask_fnlfnames = np.logical_and(np.in1d(fnamesYday, mthYdays, False),np.in1d(fnamesYear, yrs, False))
        
    fnames = fnames[mask_fnlfnames]
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
        
#        sd_dva = sd.select(va_varname)
#        dva_scale = sd_dva.attributes()['scale_factor']
#        dva_offset = sd_dva.attributes()['add_offset']
#        dva_fill = sd_dva.attributes()['_FillValue']
#        dva = sd_dva[:].astype(np.float64)
#        dva[dva==dva_fill]  = np.nan
#        dva = (dva*dva_scale) + dva_offset
#        dva = np.abs(dva)
#        dva_mask = dva > 20
        
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
        #qc_fnl_mask = qc_mask1
        qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5)
        
#        lst[np.logical_or(qc_fnl_mask,lst==lst_fill)] = np.nan
#        lst = (lst*lst_scale) + lst_offset
        #lst[qc_fnl_mask] = netCDF4.default_fillvals['f8']
        lst[qc_fnl_mask] = fval
        
        if np.ma.isMaskedArray(lst):
            lst = lst.data
        
        ds.variables[data_varname][i,:,:] = lst
        ds.sync()
        sd.end()
        
        i+=1
        
    #Read back data and calculate/output mean
    lst = ds.variables[data_varname][:]   
    lst_fnl = np.ma.mean(lst,axis=0)
    
    #Remove pixels that are missing more than 75% of their obs
    mask = np.logical_not(lst.mask)
    pct_obs = np.sum(mask,axis=0)/np.float(mask.shape[0])
    lst_fnl.mask = np.logical_or(lst_fnl.mask,pct_obs < 0.25)
    
    lst_fnl_a = lst_fnl.data
    lst_fnl_a[lst_fnl.mask] = fval
    lst_fnl_a = np.ma.round(lst_fnl_a).astype(np.uint16)
    
    print "WRITING TO: "+out_fpath
    sd = SD(out_fpath,SDC.WRITE)
    sds_lst = sd.select(data_varname)
    sds_lst[:] = lst_fnl_a
    sd.end()
    
    ds.close()
    
def stack_MYDMOD11A2_mth_all():
    
    varnames = [('LST_Night_1km','QC_Night','Night_view_angl'),
                ('LST_Day_1km','QC_Day','Day_view_angl')]
    
    path = '/projects/daymet2/climate_office/modis/MYD11A2/'
    ncdf_path = "".join([path,'temp_lst.nc'])
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_mths_aqua_terra/'
    
    fnames = np.array(os.listdir(path))
    fnames = fnames[np.char.startswith(fnames,'MYD11A2.005.')]
    
    ydayPeriods = np.array([1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97,
                            105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
                            209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305,
                            313, 321, 329, 337, 345, 353, 361])
    
    mths = np.arange(1,13)
    
    for fname in fnames:
        
        fpath = "".join([path,fname,"/"])
        print "###########################################"
        print fname
        print "###########################################"
        
        for varname in varnames:
            
            for mth in mths:
                
                out_path_mth = "".join([out_path,"mth",'%02d' % mth,"/"])
                
                if not os.path.exists(out_path_mth):
                    os.mkdir(out_path_mth)
                
                out_fpath = "".join([out_path_mth,varname[0],"_",fname,".hdf"])

                stack_MYD11A2_hdf_mth(fpath, ncdf_path, out_fpath, mth ,varname[0], varname[1], varname[2])

def stack_MYD11A2_netcdf_all():
    
    varnames = [('LST_Night_1km','QC_Night','Night_view_angl'),
                ('LST_Day_1km','QC_Day','Day_view_angl')]
    
    path = '/projects/daymet2/climate_office/modis/MYD11A2/'
    ncdf_path = "".join([path,'temp_lst.nc'])
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_mths_aqua_terra/'
    
    fnames = np.array(os.listdir(path))
    fnames = fnames[np.char.startswith(fnames,'MYD11A2.005.')]
    
    ydayPeriods = np.array([1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97,
                            105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
                            209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305,
                            313, 321, 329, 337, 345, 353, 361])
    
    for fname in fnames:
        
        fpath = "".join([path,fname,"/"])
        print "###########################################"
        print fname
        print "###########################################"
        
        for varname in varnames:
            
            for ydayPeriod in ydayPeriods:
                
                out_path_yday = "".join([out_path,"yday",str(ydayPeriod),"/"])
                
                if not os.path.exists(out_path_yday):
                    os.mkdir(out_path_yday)
                
                out_fpath = "".join([out_path_yday,varname[0],"_",fname,".hdf"])

                stack_MYD11A2_hdf_8day(fpath, ncdf_path, out_fpath, ydayPeriod ,varname[0], varname[1], varname[2])


def stack_MYD11A2_to_nc(in_path,out_ncdf,data_varname='LST_Night_1km',qc_varname='QC_Night',va_varname='Night_view_angl'):
    
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
    fnames = fnames[np.char.startswith(fnames,"MYD")]
    fnamesYday = np.array([x[13:16] for x in fnames],dtype=np.int)
    fnamesYear = np.array([x[9:13] for x in fnames],dtype=np.int)
    
    yrs = np.arange(2002,2013)
    
    mask_fnlfnames = np.in1d(fnamesYear, yrs, False)
        
    fnames = fnames[mask_fnlfnames]
    fnamesYday = fnamesYday[mask_fnlfnames]
    fnamesYear = fnamesYear[mask_fnlfnames]
    
    sd = SD(fnames[0],SDC.READ)
    x,y = sd.datasets()[data_varname][1]
    fval = sd.select(data_varname).attributes()['_FillValue']
    
    xPts,yPts = getModisTileXy(sd)
    
    sd.end()
        
    ds = Dataset(out_ncdf,'w')
    
    ds.createDimension('time',yrs.size*MODIS_8DAY.size)
    ds.createVariable('year',np.int16,('time',))
    ds.createVariable('yday',np.int16,('time',))
    ds.createVariable('mth',np.int16,('time',))
    ds.variables['year'][:] = np.concatenate([[yr]*MODIS_8DAY.size for yr in yrs])
    ds.variables['yday'][:] = np.concatenate([MODIS_8DAY]*yrs.size)
    
    yrYdays = [str(yr)+str(yday) for yr,yday in zip(ds.variables['year'][:],ds.variables['yday'][:])]
    dates = np.array([datetime.datetime.strptime(aYrYday, '%Y%j') for aYrYday in yrYdays])
    days = utld.get_days_metadata_dates(dates)
    
    times = ds.createVariable('time','f8',('time',))
    times.long_name = "time"
    times.units = "".join(["days since ",str(np.min(days[YEAR])),"-1-1 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE],times.units)
    
    mths = np.arange(1,13)
    ydays = ds.variables['yday'][:]
    mthVarVal = np.zeros(ydays.size)
    for mth in mths:
        for yday in MYD11A2_MTH_DAYS[mth]:
            mthVarVal[ydays==yday] = mth
    ds.variables['mth'][:] = mthVarVal
        
    ds.createDimension('y',y)
    ds.createDimension('x',x)
    
    ds.createVariable(data_varname,np.int16,('time','y','x',),fill_value=fval)
    
    ds.createVariable('x',np.float,('x',))
    ds.createVariable('y',np.float,('y',))
    ds.variables['x'][:] = xPts
    ds.variables['y'][:] = yPts
    
    ds.sync()
    
    i = 0
    for yr in yrs:
        
        for yday in MODIS_8DAY:
            
            fnameYrYday = fnames[np.logical_and(fnamesYear==yr,fnamesYday==yday)]
            
            if fnameYrYday.size == 1:
                fnameYrYday = fnameYrYday[0]
                print yr,yday,fnameYrYday
                
                sd = SD(fnameYrYday,SDC.READ)
                sds_lst = sd.select(data_varname)
                lst = sds_lst[:]
                
                sd_dva = sd.select(va_varname)
                dva_scale = sd_dva.attributes()['scale_factor']
                dva_offset = sd_dva.attributes()['add_offset']
                dva_fill = sd_dva.attributes()['_FillValue']
                dva = sd_dva[:].astype(np.float64)
                dva[dva==dva_fill]  = np.nan
                dva = (dva*dva_scale) + dva_offset
                
                
                dva = np.abs(dva)
                plt.imshow(dva)
                plt.colorbar()
                plt.show()
                                
                dva_mask = dva > 40
                
                
                qc = sd.select(qc_varname)[:]
                qc1 = qc & qc_bitmask1
                qc3 = qc & qc_bitmask3
                qc4 = qc & qc_bitmask4
                        
                qc_mask1 =  np.logical_or(qc1==QC_BIT1_CLOUD,qc1==QC_BIT1_NA)
                
                qc_mask2 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_3)
                qc_mask3 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_4)
                
                qc_mask4 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_3)
                qc_mask5 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_4)
                
                #qc_mask6 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_2)
                #qc_mask7 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_2)
                
                #qc_fnl_mask = np.logical_or(np.logical_or(qc_mask1,qc_mask3),qc_mask5)
                
                qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5),dva_mask)
                #qc_fnl_mask = np.logical_or(qc_fnl_mask,np.logical_or(qc_mask6,qc_mask7))
                
                lst[qc_fnl_mask] = fval
                
                if np.ma.isMaskedArray(lst):
                    lst = lst.data
                
                ds.variables[data_varname][i,:,:] = lst
                ds.sync()
                sd.end()
            else:
                print yr,yday,"NO FILE"
                ds.variables[data_varname][i,:,:] = fval
            
            i+=1
    
    yrsNc = ds.variables['year'][:]
    #Remove pixels that are missing more than 75% of their obs
    lst = ds.variables[data_varname][:]
    lst03_12 = lst[np.logical_and(yrsNc >= 2003,yrsNc <= 2012),:,:] 
        
    mask = np.logical_not(lst03_12.mask)
    pct_obs = np.sum(mask,axis=0)/np.float(mask.shape[0])
    
    pct_obs_mask = pct_obs < 0.33
    
    if np.sum(pct_obs_mask) > 0:

        rows,cols = np.nonzero(pct_obs_mask)
        lst[:,rows,cols] = np.ma.masked
        ds.variables[data_varname][:] = np.ma.filled(lst, fval)

    ds.close()
    ds = None

def to_celsius_mthly_mosaics():
    
    varname = 'LST_Night_1km'
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/'
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/'
    
    for mth in np.arange(1,13):
        in_fpath = "".join([in_path,"MOSAIC.",varname,".%02d"%mth,".",varname,".tif",])
        out_fpath = "".join([out_path,"MOSAIC.",varname,".%02d.C.tif"%mth])
        output_celsius(in_fpath, out_fpath)
 
def resample_all_mthly_mosaics():
    
    varname = 'LST_Night_1km'
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/'
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/'

    for mth in np.arange(1,13):
            
        in_fpath = "".join([in_path,"MOSAIC.",varname,".%02d.hdf"%mth])    
        resample_to_geo(in_fpath, varname, out_path,"".join([in_path,"MOSAIC.",varname,".%02d.tif"%mth]))

def reset_fillval_all_mthly_mosaics():
    
    varname = 'LST_Night_1km'
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics'
    
    os.chdir(in_path)
    
    for mth in np.arange(1,13):
        reset_MYD11A2_fillval("".join(["MOSAIC.",varname,".%02d.hdf"%mth]), varname)

def mosaic_all_mthly():
    
    varname = 'LST_Night_1km'
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/'
    out_path = "".join([in_path,"mosaics/"])
    
    os.chdir(in_path)
    fnames = np.array(os.listdir(in_path))
    os.chdir(out_path)
    
    for mth in np.arange(1,13):
        
        fnamesMth = fnames[np.logical_and(np.char.startswith(fnames,varname),np.char.endswith(fnames, "%02d.hdf"%mth))]
        
        fout = open("".join([out_path,"mosaic_file_list.txt"]),'w')
        fname_lines = ["".join([in_path,x,"\n"]) for x in fnamesMth]        
        fout.writelines(fname_lines)
        fout.close()
        
        if varname == 'LST_Day_1km':
            sds = "'1 0 0 0 0 0 0 0 0 0 0 0'"
        elif varname == 'LST_Night_1km':
            sds = "'0 0 0 0 1 0 0 0 0 0 0 0'"
    
        mosaicCmd = "".join(["mrtmosaic -i mosaic_file_list.txt -o ","".join(["MOSAIC.",varname,".%02d.hdf"%mth]),' -s ',sds])
        print mosaicCmd
        os.system(mosaicCmd)

def mosaic_mod12q1():
    
    lst_tile_path = '/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/'
    in_path = '/projects/daymet2/climate_office/modis/MOD12Q1_Special/MOD12Q1.Special.004/'
    out_path = '/projects/daymet2/climate_office/modis/MOD12Q1_Special/'
    
    tileNamesLst = np.array(os.listdir(lst_tile_path))
    tileNamesLst = tileNamesLst[np.char.startswith(tileNamesLst,'LST')]
    tileNamesLst = np.unique(np.array([fname.split(".")[1] for fname in tileNamesLst]))
    
    os.chdir(in_path)
    fnames = np.array(os.listdir(in_path))
    tileNamesLc = np.array([fname.split(".")[2] for fname in fnames])
    fnamesFnl = fnames[np.in1d(tileNamesLc, tileNamesLst)]
    
    os.chdir(out_path)
                
    fout = open("".join([out_path,"mosaic_file_list.txt"]),'w')
    fname_lines = ["".join([in_path,x,"\n"]) for x in fnamesFnl]        
    fout.writelines(fname_lines)
    fout.close()
    
    #land cover type 2
    sds = "'0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'"

    mosaicCmd = "".join(["mrtmosaic -i mosaic_file_list.txt -o MOSAIC.MOD12Q1.Special.hdf",' -s ',sds])
    print mosaicCmd
    os.system(mosaicCmd)
     
def mosaic_all_8day():
    
    varnames = ('LST_Night_1km','LST_Day_1km')
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_8day_hdfs/'
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_8day_hdfs/mosaics/'
    
    ydayPeriods = np.array([1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97,
                            105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
                            209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305,
                            313, 321, 329, 337, 345, 353, 361])
    
        
    for ydayPeriod in ydayPeriods:
    
        in_path_yday = "".join([in_path,'yday',str(ydayPeriod),"/"])
        fnames = np.array(os.listdir(in_path_yday))
        
        out_path_yday = "".join([out_path,"yday",str(ydayPeriod),"/"])
        if not os.path.exists(out_path_yday):
            os.mkdir(out_path_yday)
        
        for aVar in varnames:
            
            fnames_aVar = fnames[np.logical_and(np.char.endswith(fnames, '.hdf'),
                                                np.char.startswith(fnames, aVar))]
            
            fout = open("".join([out_path_yday,"mosaic_file_list.txt"]),'w')
                
            fname_lines = ["".join([in_path_yday,x,"\n"]) for x in fnames_aVar]
                
            fout.writelines(fname_lines)
            fout.close()
            
            os.chdir(out_path_yday)
            
            if aVar == 'LST_Day_1km':
                sds = "'1 0 0 0 0 0 0 0 0 0 0 0'"
            elif aVar == 'LST_Night_1km':
                sds = "'0 0 0 0 1 0 0 0 0 0 0 0'"
            
            mosaicCmd = "".join(["mrtmosaic -i mosaic_file_list.txt -o ","".join(["MOSAIC.",aVar,".hdf"]),' -s ',sds])
            print mosaicCmd
            os.system(mosaicCmd)


def reset_MYD11A2_fillval(fpath,varname,new_fill=65535):
    #varname: LST_Night_1km or LST_Day_1km
    sd = SD(fpath,SDC.WRITE)
    sds_lst = sd.select(varname)
    cur_fill = sds_lst.attributes()['_FillValue']
    
    sds_lst.setfillvalue(new_fill)
    a = sds_lst[:]
    a[a==cur_fill] = new_fill
    sds_lst[:] = a
    sd.end()
    
def reset_fillval_all_mosaics():
    varnames = ('LST_Night_1km','LST_Day_1km')
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_8day_hdfs/mosaics/'
    
    ydayPeriods = np.array([1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97,
                            105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
                            209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305,
                            313, 321, 329, 337, 345, 353, 361])
    
    for ydayPeriod in ydayPeriods:
    
        in_path_yday = "".join([in_path,'yday',str(ydayPeriod),"/"])
        
        for aVar in varnames:
            
            in_fpath = "".join([in_path_yday,"MOSAIC.",aVar,".hdf"])
            reset_MYD11A2_fillval(in_fpath, aVar)


def resample_to_geo(in_fpath,varName,out_path,out_fname):
    
    os.chdir(out_path)
    
    fprm = open("InterpGridReproj.prm",'w')
    fprm.write("\n")
    fprm.write("".join(["INPUT_FILENAME = ",in_fpath,"\n"]))
    fprm.write("\n")
    fprm.write("SPECTRAL_SUBSET = ( 1 )\n")
    fprm.write("\n")
    fprm.write("SPATIAL_SUBSET_TYPE = INPUT_LAT_LONG\n")
    fprm.write("\n")
    fprm.write("SPATIAL_SUBSET_UL_CORNER = ( 53.004166448 -125.995833883 )\n")
    fprm.write("SPATIAL_SUBSET_LR_CORNER = ( 22.004166572 -63.995834131 )\n")
    fprm.write("\n")
    fprm.write("".join(["OUTPUT_FILENAME = ",out_fname,"\n"]))
    fprm.write("\n")
    #fprm.write("RESAMPLING_TYPE = BILINEAR\n")
    fprm.write("RESAMPLING_TYPE = CC\n")
    fprm.write("\n")
    fprm.write("OUTPUT_PROJECTION_TYPE = GEO\n")
    fprm.write("\n")
    fprm.write("OUTPUT_PROJECTION_PARAMETERS = ( \n")
    fprm.write(" 0.0 0.0 0.0\n")
    fprm.write(" 0.0 0.0 0.0\n")
    fprm.write(" 0.0 0.0 0.0\n")
    fprm.write(" 0.0 0.0 0.0\n")
    fprm.write(" 0.0 0.0 0.0 )\n")
    fprm.write("\n")
    fprm.write("DATUM = WGS84\n")
    fprm.write("\n")
    fprm.write("OUTPUT_PIXEL_SIZE = 0.0083333333\n")
    fprm.close()
    
    os.system("resample -p InterpGridReproj.prm")
    
'''
INPUT_FILENAME = /projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_day_mosaic/MOSAIC.LST_Day.hdf

SPECTRAL_SUBSET = ( 1 0 0 0 0 0 0 0 0 0 0 0 )

SPATIAL_SUBSET_TYPE = INPUT_LAT_LONG

SPATIAL_SUBSET_UL_CORNER = ( 53.004166448 -125.995833883 )
SPATIAL_SUBSET_LR_CORNER = ( 22.004166572 -63.9958341309 )

OUTPUT_FILENAME = /projects/daymet2/climate_office/modis/MYD11A2/mean_hdfs/lst_day_mosaic/MOSAIC.LST_Day.InterpGrid.tif

RESAMPLING_TYPE = BILINEAR

OUTPUT_PROJECTION_TYPE = GEO

OUTPUT_PROJECTION_PARAMETERS = (
 0.0 0.0 0.0
 0.0 0.0 0.0
 0.0 0.0 0.0
 0.0 0.0 0.0
 0.0 0.0 0.0 )

DATUM = WGS84

OUTPUT_PIXEL_SIZE = 0.0083333333
'''              

def resample_all_mosaics():
    
    varnames = ('LST_Night_1km','LST_Day_1km')
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_8day_hdfs/mosaics/'
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_8day_hdfs/mosaics/'
    
    ydayPeriods = np.array([1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97,
                            105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
                            209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305,
                            313, 321, 329, 337, 345, 353, 361])
    
        
    for ydayPeriod in ydayPeriods:
    
        in_path_yday = "".join([in_path,'yday',str(ydayPeriod),"/"])
        out_path_yday = "".join([out_path,"yday",str(ydayPeriod),"/"])
        
        for aVar in varnames:
            
            in_fpath = "".join([in_path_yday,"MOSAIC.",aVar,".hdf"])
            
            resample_to_geo(in_fpath, aVar, out_path_yday,"".join(["MOSAIC.",aVar,".tif"]))

def output_celsius(in_fpath,out_fpath,ndata=65535.0):
    
    in_ds = RasterDataset(in_fpath) 
    a = in_ds.gdalDs.GetRasterBand(1).ReadAsArray()
    a = np.ma.masked_equal(a,ndata)

    #To Celsius: multiply by scale factor (0.02) and subtract 273.15
    a = (a*0.02) - 273.15
            
    ds_out = gdal.GetDriverByName('GTiff').Create(  out_fpath, 
                                                    a.shape[1], 
                                                    a.shape[0], 1, gdalconst.GDT_Float32)
    
    band_out = ds_out.GetRasterBand(1)
    band_out.SetNoDataValue(ndata)
    
    #Manually set geoT. Gdal does not read bounds correctly from MRT GeoTiff output
    geoT = list(in_ds.geoT)
    geoT[0] = -125.995833883
    geoT[3] = 53.004166448
    geoT = tuple(geoT)
    
    ds_out.SetGeoTransform(geoT)
    ds_out.SetProjection(in_ds.projection)
    band_out.WriteArray(np.ma.filled(a, ndata))
    ds_out.FlushCache()

def output_all_celsius():
    
    varnames = ('LST_Night_1km','LST_Day_1km')
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_8day_hdfs/mosaics/'
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_8day_hdfs/mosaics/'
    
    ydayPeriods = np.array([1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97,
                            105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
                            209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305,
                            313, 321, 329, 337, 345, 353, 361])
    
        
    for ydayPeriod in ydayPeriods:
    
        in_path_yday = "".join([in_path,'yday',str(ydayPeriod),"/"])
        out_path_yday = "".join([out_path,"yday",str(ydayPeriod),"/"])
        
        for aVar in varnames:
            
            in_fpath = "".join([in_path_yday,"MOSAIC.",aVar,".",aVar,".tif"])
            out_fpath = "".join([out_path_yday,"MOSAIC.",aVar,".C.tif"])
            print out_fpath 
            output_celsius(in_fpath, out_fpath)

def output_mth_means():
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_8day_hdfs/mosaics/'
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/mean_8day_hdfs/mth_means/'
    
    mths = np.arange(1,13)
    varnames = ('LST_Night_1km','LST_Day_1km')
    
    for mth in mths:
        
        mth_ydays = MYD11A2_MTH_DAYS[mth]
        
        for aVar in varnames:
            
            lst_mth = None
            for ydayPeriod in mth_ydays:
                
                in_path_yday = "".join([in_path,'yday',str(ydayPeriod),"/"])
                in_fpath = "".join([in_path_yday,"MOSAIC.",aVar,".C.tif"])
                ds = RasterDataset(in_fpath)
                a = ds.readAsArray()
                a = np.ma.asarray(a, np.float64)
            
                if lst_mth is None:
                    lst_mth = a
                else:
                    lst_mth = np.ma.dstack((lst_mth,a)) 
            
            lst_mth = np.ma.mean(lst_mth,axis=2)
            
            out_fpath = "".join([out_path,aVar,".",str(mth),".tif"])
            ds_out = gdal.GetDriverByName('GTiff').Create(  out_fpath, 
                                                            a.shape[1], 
                                                            a.shape[0], 1, gdalconst.GDT_Float32)
            band_out = ds_out.GetRasterBand(1)
            ndata = ds.gdalDs.GetRasterBand(1).GetNoDataValue()
            band_out.SetNoDataValue(ndata)
            
            ds_out.SetGeoTransform(ds.geoT)
            ds_out.SetProjection(ds.projection)
            band_out.WriteArray(np.ma.filled(a, ndata))
            ds_out.FlushCache()

def mv_MOD11A2_to_MYD11A2():
    path_MYD11A2 = '/projects/daymet2/climate_office/modis/MYD11A2/'
    path_MOD11A2 = '/projects/daymet2/climate_office/modis/MOD11A2.005/'
    
    fnames = np.array(os.listdir(path_MYD11A2))
    tileDirs = fnames[np.char.startswith(fnames,'MYD11A2.005.')]
    
    dateDirs = np.array(os.listdir(path_MOD11A2))
    
    dateDirFiles = {}
    for aDateDir in dateDirs:
        dateDirFiles[aDateDir] = np.array(os.listdir("".join([path_MOD11A2,aDateDir])))
    
    for aTileDir in tileDirs:
        
        tileName = aTileDir.split('.')[-1]
        
        toPath = "".join([path_MYD11A2,aTileDir])
        
        for aDateDir in dateDirs:
            fromPath = "".join([path_MOD11A2,aDateDir,"/*",tileName,"*"])
            
            mvCmd = "".join(['mv ',fromPath,' ',toPath])
            print mvCmd
            os.system(mvCmd)

def stack_all_to_nc():
    
    varnames = [('LST_Night_1km','QC_Night','Night_view_angl'),
                ('LST_Day_1km','QC_Day','Day_view_angl')]
    
    path = '/projects/daymet2/climate_office/modis/MYD11A2/'
    out_path = '/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks_test/'
    
    fnames = np.array(os.listdir(path))
    fnames = fnames[np.char.startswith(fnames,'MYD11A2.005.')]
        
    for fname in fnames:
        
        fpath = "".join([path,fname,"/"])
        print "###########################################"
        print fname
        print "###########################################"
        
        for varname in varnames:
            
            out_ncpath = "".join([out_path,varname[0],".",fname.split('.')[-1],".nc"])
            print out_ncpath
            stack_MYD11A2_to_nc(fpath, out_ncpath, varname[0], varname[1], varname[2])

def linear_interp_missing(y,x):
    return np.interp(x, x[~y.mask], y[~y.mask])
    

def linear_interp_ncstack(ncpath,varname):
    ds = Dataset(ncpath,'r+')
    lst = ds.variables[varname][:]
    
    nmiss = np.sum(lst.mask,axis=0)
    rows,cols = np.nonzero(np.logical_and(nmiss < lst.shape[0], nmiss > 0))
    x = np.arange(lst.shape[0])
    statchk = status_check(rows.size,10000)
    
    for r,c in zip(rows,cols):
        
        rclst = lst[:,r,c]
        lst[:,r,c] = np.round(linear_interp_missing(rclst, x))
                
        statchk.increment()
    
    ds.variables[varname][:] = lst
    ds.sync()
    ds.close()
    
def impute_ncstack(ncpath,varname,stnda,impLst):
        
    ds = Dataset(ncpath)#,'r+')
    lst = ds.variables[varname][:]
    
    xPts = ds.variables['x'][:]
    yPts = ds.variables['y'][:]
    
    nmiss = np.sum(lst.mask,axis=0)
    rows,cols = np.nonzero(np.logical_and(nmiss < lst.shape[0], nmiss > 0))
    statchk = status_check(rows.size,1000)
    
#    rows = np.array([25,35])
#    cols = np.array([1111,1125])
    
    for r,c in zip(rows,cols):
        
        rclst = (lst[:,r,c]*0.02) - 273.15
        
        rclstImp = (impLst.imputeByMth(xPts[c], yPts[r], rclst) + 273.15)/0.02
        
        lst[:,r,c] = np.round(rclstImp)
                
        statchk.increment()
    
    ds.variables[varname][:] = lst
    ds.sync()
    ds.close()

def create8dayStnDb():
    dbin_path = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc'
    var_name = 'tmax'
    dbout_path = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax_8day.nc'
    
    dbin = station_data_infill(dbin_path, var_name)    
    mask_stns = np.isnan(dbin.stns[BAD])
    stns = dbin.stns[mask_stns] 

    #####################################################################
    yrs = np.arange(2002,2013)
    
    yr8Day = np.concatenate([[yr]*MODIS_8DAY.size for yr in yrs])
    yDay8day = np.concatenate([MODIS_8DAY]*yrs.size)
    
    yrYdays = [str(yr)+str(yday) for yr,yday in zip(yr8Day,yDay8day)]
    dates = np.array([datetime.datetime.strptime(aYrYday, '%Y%j') for aYrYday in yrYdays])
    days = utld.get_days_metadata_dates(dates)
    #####################################################################
    
    ds_out = dbDataset(dbout_path,'w')
    ds_out.db_create_time_dimvar(days)
    ds_out.db_create_stnid_dimvar(stns[STN_ID])
    ds_out.db_create_stn_vars(stns)
    
    if var_name == 'tmin':
        ds_out.db_create_tmin_var(chunk=(days.size,50))
    elif var_name == 'tmax':
        ds_out.db_create_tmax_var(chunk=(days.size,50))
        
    ds_out.sync()
    
    yrMask = np.logical_and(dbin.days[YEAR]>=np.min(yrs),dbin.days[YEAR]<=np.max(yrs))
    daysIn = dbin.days[yrMask]
    
    obs = dbin.ds.variables[var_name][yrMask,:]
    obs = obs[:,mask_stns]
    
    tagg = TairAggregate(daysIn)
    obs8Day = tagg.dailyToModis8Day(obs)
    print obs8Day.shape
    print ds_out.variables[var_name].shape
    ds_out.variables[var_name][:] = obs8Day
    ds_out.close()
    

def getModisTileXy(sd):

    metaLines = np.array(sd.attributes()['StructMetadata.0'].splitlines())

    ul_str = metaLines[np.char.find(metaLines,'UpperLeftPointMtrs') != -1][0][21:].strip("()")
    lr_str = metaLines[np.char.find(metaLines,'LowerRightMtrs') != -1][0][17:].strip("()")
    
    ul_xy = [np.float(x) for x in ul_str.split(',')]
    lr_xy = [np.float(x) for x in lr_str.split(',')]

    xdim = np.float(metaLines[np.char.find(metaLines,'XDim=') != -1][0].split('=')[-1])
    ydim = np.float(metaLines[np.char.find(metaLines,'YDim=') != -1][0].split('=')[-1])
    
    resX = np.abs((ul_xy[0]-lr_xy[0])/xdim)
    resY = np.abs((ul_xy[1]-lr_xy[1])/ydim)

    ul_pt = (ul_xy[0]+resX,ul_xy[1]-resY)
    lr_pt = (lr_xy[0]-resX,lr_xy[1]+resY)
    
    x = np.linspace(ul_pt[0], lr_pt[0], xdim)
    y = np.linspace(ul_pt[1], lr_pt[1], ydim)
    
    return x,y

class modis_sin_latlon_transform():

    def __init__(self):
        
        self.sr_sin = osr.SpatialReference()
        self.sr_sin.ImportFromProj4(PROJ4_MODIS)
        self.sr_wgs84 = osr.SpatialReference()
        self.sr_wgs84.ImportFromEPSG(EPSG_WGS84)
        self.sr_nad83 = osr.SpatialReference()
        self.sr_nad83.ImportFromEPSG(EPSG_NAD83)
        self.sr_lcc = osr.SpatialReference()
        self.sr_lcc.ImportFromProj4("+proj=lcc +datum=WGS84 +lat_1=25n +lat_2=60n +lat_0=42.5n +lon_0=100w")
        
        self.trans_sin_to_wgs84 = osr.CoordinateTransformation(self.sr_sin,self.sr_wgs84)
        self.trans_wgs84_to_sin = osr.CoordinateTransformation(self.sr_wgs84,self.sr_sin)
        self.trans_nad83_to_sin = osr.CoordinateTransformation(self.sr_nad83,self.sr_sin)
        self.trans_sin_to_lcc = osr.CoordinateTransformation(self.sr_sin,self.sr_lcc)
        self.trans_wgs84_to_lcc = osr.CoordinateTransformation(self.sr_wgs84,self.sr_lcc)

def calc_ioa(x, y):
    '''
    Calculate the index of agreement (Durre et al. 2010; Legates and McCabe 1999) between x and y
    '''
    
    y_mean = np.mean(y)
    d = np.sum(np.abs(x - y_mean) + np.abs(y - y_mean))
    
    if d == 0.0:
        print "|".join(["WARNING: calc_ioa: x, y identical"])
        #The x and y series are exactly the same
        #Return a perfect ioa
        return 1.0
    
    ioa = 1.0 - (np.sum(np.abs(y - x)) / d)
    
#    if ioa == 0:
#        print "|".join(["WARNING: calc_ioa: ioa == 0"])
#        #Means all ys are the same or only one observation.
#        #This could possibly happen with prcp in arid regions
#        #Add on an extra observation to the time series that has same difference as x[0] and y[0]
#        x_new = np.concatenate([x, np.array([x[0] + (x[0] * .1)])])
#        y_new = np.concatenate([y, np.array([y[0] + (x[0] * .1)])])
#        
#        y_mean = np.mean(y_new)
#        ioa = 1.0 - (np.sum(np.abs(y_new - x_new)) / np.sum(np.abs(x_new - y_mean) + np.abs(y_new - y_mean)))  
    
    return ioa

def calc_ioa2(x, y):
    '''
    Calculate the index of agreement (Durre et al. 2010; Legates and McCabe 1999) between x and y
    '''
    
    y_mean = np.mean(y)
    y.shape = (y.size,1)
    d = np.sum(np.abs(x - y_mean) + np.abs(y - y_mean),axis=0)
    
#    if d == 0.0:
#        print "|".join(["WARNING: calc_ioa: x, y identical"])
#        #The x and y series are exactly the same
#        #Return a perfect ioa
#        return 1.0
    
    ioa = 1.0 - (np.sum(np.abs(y - x),axis=0) / d)
    
#    if ioa == 0:
#        print "|".join(["WARNING: calc_ioa: ioa == 0"])
#        #Means all ys are the same or only one observation.
#        #This could possibly happen with prcp in arid regions
#        #Add on an extra observation to the time series that has same difference as x[0] and y[0]
#        x_new = np.concatenate([x, np.array([x[0] + (x[0] * .1)])])
#        y_new = np.concatenate([y, np.array([y[0] + (x[0] * .1)])])
#        
#        y_mean = np.mean(y_new)
#        ioa = 1.0 - (np.sum(np.abs(y_new - x_new)) / np.sum(np.abs(x_new - y_mean) + np.abs(y_new - y_mean)))  
    
    return ioa

class ImputeLST():
    
    def __init__(self,stnda,tairVar):
        
        self.stnTair = stnda.ds.variables[tairVar][:].astype(np.float)
        self.stnda = stnda
        self.stns = self.stnda.stns
        self.sinTrans = modis_sin_latlon_transform()
        
        self.mthMasks = []
        self.uYrs = np.unique(self.stnda.days[YEAR])
        
            
        for mth in np.arange(1,13):
            
            self.mthMasks.append(self.stnda.days[MONTH]==mth)
        
    def impute(self,x,y,lst,nstns=7):
        lon, lat, Z = self.sinTrans.trans_sin_to_wgs84.TransformPoint(x,y)
        
        dists = utlg.grt_circle_dist(lon, lat, self.stns[LON], self.stns[LAT])
        
        obsTair = self.stnTair[:,np.argsort(dists)[0:nstns]]
        
        lstFin = lst[~lst.mask]
        obsTairFin = obsTair[~lst.mask,:]
        
        idxStns = np.arange(obsTairFin.shape[1])
        
        #ioaStns = np.array([calc_ioa(lstFin,obsTairFin[:,y]) for y in np.arange(obsTairFin.shape[1])])
        
        ioaStns = calc_ioa2(obsTairFin, lstFin)
        lstFin = np.ravel(lstFin)
        
        wgtStns = ioaStns/np.sum(ioaStns)

        modStns = np.array([stats.linregress(obsTairFin[:,y], lstFin)[0:2] for y in idxStns])
        
        modLst = (obsTair*modStns[:,0]) + modStns[:,1]
        
        impLst = np.average(modLst, axis=1, weights=wgtStns)
        
        lstFnl = lst.data
        lstFnl[lst.mask] = impLst[lst.mask]
        
        return lstFnl
    
    def imputeByMth(self,x,y,lst,nstns=7):
        lon, lat, Z = self.sinTrans.trans_sin_to_wgs84.TransformPoint(x,y)
        
        dists = utlg.grt_circle_dist(lon, lat, self.stns[LON], self.stns[LAT])
        
        obsTair = self.stnTair[:,np.argsort(dists)[0:nstns]]
        
        maskLstFin = ~lst.mask
        idxStns = np.arange(obsTair.shape[1])
        
        lstFnl = np.copy(lst.data)
        
        for maskMth in self.mthMasks:
            
            maskLstFinMth = np.logical_and(maskLstFin,maskMth)
            
            lstFin = lst[maskLstFinMth]
            obsTairMth = obsTair[maskMth,:]
            obsTairFin = obsTair[maskLstFinMth,:]
        
            ioaStns = calc_ioa2(obsTairFin, lstFin)
            lstFin = np.ravel(lstFin)
            
            wgtStns = ioaStns/np.sum(ioaStns)
    
            modStns = np.array([stats.linregress(obsTairFin[:,y], lstFin)[0:2] for y in idxStns])
            
            modLst = (obsTairMth*modStns[:,0]) + modStns[:,1]
            
            impLst = np.average(modLst, axis=1, weights=wgtStns)

            #lstFnl[maskLstFinMth] = impLst[maskLstFinMth]
            lstFnl[maskMth] = impLst
        
#        plt.plot(lstFnl,color='r')
#        plt.plot(lst)
#        plt.show()
        
        return lstFnl
        

def lstMthMeans():
    ds = Dataset('/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks_test/LST_Night_1km.h11v05.nc')
    
    var_time = ds.variables['time']
            
    dates = num2date(var_time[:], var_time.units)
    days = utld.get_days_metadata_dates(dates)

    maskYr = np.logical_and(days[YEAR]>=2003,days[YEAR]<=2012)
    days = days[maskYr]
    
    lst = ds.variables['LST_Night_1km'][maskYr,:,:]*0.02
    
    tagg = TairAggregate(days)
    lst = tagg.dailyToMthly(lst, -1)[0]

    plt.imshow(np.mean(lst[tagg.yrMths[MONTH]==4,:,:],axis=0))
    plt.colorbar()
    plt.show()

def examineLst():
    xStart = 260
    xEnd = 1170
    yStart = 0
    yEnd = 50
    
    
    ds = Dataset('/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks/LST_Night_1km.h11v05.nc')
    mth = ds.variables['mth'][:]
    yr = ds.variables['year'][:]
    
    mask = np.logical_and(np.logical_and(yr>=2003,yr<=2012),mth==2)
    #lst = ds.variables['LST_Night_1km'][mask,yStart:yEnd,xStart:xEnd]*0.02
    lst = ds.variables['LST_Night_1km'][mask,:,:]*0.02
    plt.imshow(np.ma.mean(lst,axis=0))
    plt.show()
    
    lstMiss = lst.mask
    
    pctComp = 1.0-np.sum(lstMiss,axis=0)/np.float(lstMiss.shape[0])
    pctComp =np.ma.masked_less(pctComp, .95)
    
    print np.sum(~pctComp.mask)
    plt.imshow(pctComp)
    plt.colorbar()
    plt.show()
    
#    idx = np.nonzero(mask)[0]
#    
#    for x in idx:
#        plt.imshow(ds.variables['LST_Night_1km'][x,:,:]*0.02)
#        plt.colorbar()
#        plt.show()

def findRowsCols(x,y,nx,ny,rstep):
                
    lcol = x-rstep
    rcol = x+rstep
    trow = y-rstep
    brow = y+rstep
    
    set1c = np.arange(lcol,rcol)
    set1r = np.repeat(trow, set1c.size)
    
    set2r = np.arange(trow,brow)
    set2c = np.repeat(rcol, set2r.size)
    
    set3c = np.arange(rcol,lcol,step=-1)
    set3r = np.repeat(brow, set3c.size)
    
    set4r = np.arange(brow,trow,step=-1)
    set4c = np.repeat(lcol, set4r.size)
    
    rows = np.concatenate((set1r,set2r,set3r,set4r))
    cols = np.concatenate((set1c,set2c,set3c,set4c))
    
    maskOut = np.logical_or(np.logical_or(cols>=nx,rows>=ny),np.logical_or(cols<0,rows<0))
    maskOut = ~maskOut
    
    rows = rows[maskOut]
    cols = cols[maskOut]
    
    ridx = np.random.permutation(np.arange(rows.size))
    rows = rows[ridx]
    cols = cols[ridx]
    
    return rows,cols

class ImputeLstNorm(object):

    def __init__(self,lstData,fpathRfuncs='/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R'):
        
        self.lstData = lstData
        r.source(fpathRfuncs)
    
    def imputeNorm(self,x,y):

        ptLst = self.lstData.get_lst_data(x,y)
        return self.rImpute(ptLst)
    
    def rImpute(self,ptLst):
        impNorm = np.array(r.impute_norm_all(robjects.Matrix(ptLst))).ravel()
        return impNorm

def shrinkMatrix(aMatrix,idx,minObs=1):
    
    aMatrix = ~aMatrix
    
    validMask = aMatrix[:,0]
    validMask.shape = (validMask.size,1)
    
    nObs = np.sum(validMask,axis=1)
    maskBelowMin = nObs < minObs
    
    keepCol = np.ones(aMatrix.shape[1],dtype=np.bool)
    
    for i in np.arange(1,aMatrix.shape[1]):
        
        x = idx[i]
        
        aColValidMask = aMatrix[:,x]
        
        if np.count_nonzero(np.logical_and(maskBelowMin,aColValidMask)) > 0:
            
            aColValidMask.shape = (aColValidMask.size,1)
            validMask = np.hstack((validMask,aColValidMask))
            nObs = np.sum(validMask,axis=1)
            maskBelowMin = nObs < minObs
            
            if np.count_nonzero(maskBelowMin) == 0:
                keepCol[i+1:] = False
                break
        
        else:
            
            keepCol[i] = False
    
    return keepCol

class LstData(object):

    def __init__(self,pathNcStacks,tileName,lstVar,tairVar,stnda):
        
        tileCol = np.int(tileName[1:3]) #h
        tileRow = np.int(tileName[4:6]) #v
        
        ds = Dataset("".join([pathNcStacks,lstVar,'.h%02d' % tileCol,'v%02d' % tileRow,".nc"]))
        mth = ds.variables['mth'][:]
        yr = ds.variables['year'][:]
        
        self.maskTime = np.nonzero(np.logical_and(yr>=2003,yr<=2012))[0]
        self.idxTimeStart = self.maskTime[0]
        self.mth = mth[self.maskTime]
        self.yr = yr[self.maskTime]
        self.dsVarMain = ds.variables[lstVar]
        self.dsMain = ds
        
        self.xSin,self.ySin = np.meshgrid(self.dsMain.variables['x'][:],self.dsMain.variables['y'][:])
                
        maskTimeStns = np.nonzero(np.logical_and(stnda.days[YEAR]>=2003,stnda.days[YEAR]<=2012))[0]
        self.stnTair = stnda.ds.variables[tairVar][maskTimeStns,:].astype(np.float)+273.15
        self.stnda = stnda
        self.stns = self.stnda.stns
        self.sinTrans = modis_sin_latlon_transform()
        
        self.stnLccX = np.zeros(self.stns.size)
        self.stnLccY = np.zeros(self.stns.size)
        for i in np.arange(self.stns.size):
            x, y, z = self.sinTrans.trans_wgs84_to_lcc.TransformPoint(self.stns[LON][i],self.stns[LAT][i])
            self.stnLccX[i] = x
            self.stnLccY[i] = y
            
    def set_lst_chk(self,r_start,r_end,c_start,c_end):
        
        self.lstChk = self.dsVarMain[self.maskTime,r_start:r_end,c_start:c_end]
        
        if not np.ma.isMA(self.lstChk):
            self.lstChk = np.ma.masked_array(self.lstChk,self.lstChk==0)
        
        self.lstChkR = r_start
        self.lstChkC = c_start
        
    def get_lst_data(self,x,y):
        
        #Stn data
        #lon, lat, Z = self.sinTrans.trans_sin_to_wgs84.TransformPoint(self.xSin[y,x],self.ySin[y,x])
        lccX,lccY,lccZ = self.sinTrans.trans_sin_to_lcc.TransformPoint(self.xSin[y,x],self.ySin[y,x])
        
        #dists = utlg.grt_circle_dist(lon, lat, self.stns[LON], self.stns[LAT])
        dists = np.sqrt(np.square(self.stnLccY-lccY) + np.square(self.stnLccX-lccX))
        idxStn = np.argsort(dists)[0:3]
        obsTair = np.take(self.stnTair, idxStn, axis=1)
        
        x = x - self.lstChkC
        y = y - self.lstChkR
        
        ptLst = self.lstChk[:,y,x]*np.float(0.02)
        ptLst.shape = (ptLst.size,1)
        ptLst = np.ma.filled(ptLst, np.nan)
        chkForPt = np.hstack((ptLst,obsTair))
        
        return np.require(chkForPt,dtype=np.float,requirements=['C','A','W','O'])

def load_data_from_dsVar(dsVar,idxTimeStart,rows,cols,tileMask,rOffset,cOffset):
    return dsVar[idxTimeStart:,(np.min(rows[tileMask])-rOffset):(np.max(rows[tileMask])+1-rOffset),(np.min(cols[tileMask])-cOffset):(np.max(cols[tileMask])+1-cOffset)]

def testImputeLstNorm():
    ds = Dataset('/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks/LST_Night_1km.h11v05.nc')
    
    mth = ds.variables['mth'][:]
    yr = ds.variables['year'][:]
    
    mask = np.nonzero(np.logical_and(np.logical_and(yr>=2003,yr<=2012),mth==1))[0]
    #mask = np.nonzero(np.logical_and(yr>=2003,yr<=2012))[0]
    mth = mth[mask]
    janMask = mth==1
    
    #plt.imshow(np.mean(ds.variables['LST_Night_1km'][mask,0:60,1080:1150]*0.02,axis=0))
    plt.imshow(np.mean(ds.variables['LST_Night_1km'][mask,:,:]*0.02,axis=0))
    plt.colorbar()
    plt.show()
    
    lst = ds.variables['LST_Night_1km'][mask,:,:]#*0.02
    
    imp = ImputeLstNorm(lst)

    lstToTest = ds.variables['LST_Night_1km'][0,0:60,1080:1150]
    
    lstImp = np.zeros(lstToTest.shape)
    lstToTest = None
    
    print lstImp.shape
    print len(zip(np.arange(0,60),np.arange(lstImp.shape[0]))),len(zip(np.arange(1080,1150),np.arange(lstImp.shape[1])))
    
    impLst = imp.imputeNorm(1111,25)
    plt.plot(impLst,color='red')
    plt.plot(lst[:,25,1111]*0.02)
    print np.mean(impLst)
    plt.show()
    
#    schk = status_check(lstImp.size, 50)
#    for x,x1 in zip(np.arange(1080,1150),np.arange(lstImp.shape[1])):
#        
#        for y,y1 in zip(np.arange(0,60),np.arange(lstImp.shape[0])):
#            
#            #print x,y,np.mean(imp.imputeNorm(x, y)[janMask])
#            lstImp[y1,x1] = np.mean(imp.imputeNorm(x,y)[janMask])
#            schk.increment() 
#    
#    lstImp = np.ma.masked_equal(lstImp, 0)
#    plt.imshow(lstImp)
#    plt.colorbar()
#    plt.show()      

def testImputeLstNormFnl():
    xStart = 100
    xEnd = 200
    yStart = 900
    yEnd = 1000
    
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin_8day.nc', 'tmin',stn_dtype=DTYPE_STN_BASIC)
    lstData = LstData('/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks/', 'h10v04','LST_Night_1km','tmin',stnda)
    lstData.set_lst_chk(yStart, yEnd, xStart, xEnd)
    
#    lst1 = lstData.get_lst_data(500, 26)[:,0]
#    lst2 = lstData.dsVarMain[lstData.maskTime,26,500]*0.02
#    
#    plt.plot(lst1)
#    plt.plot(lst2)
#    plt.show()
    
    imp = ImputeLstNorm(lstData)
    
    lstImp = imp.imputeNorm(178,928)
    plt.plot(lstImp)
    plt.show()
    
    #imp = ImputeLstNorm('/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks/', 'h11v05','LST_Night_1km')
    yday = lstData.dsMain.variables['yday'][lstData.maskTime]
    
    janMask = np.in1d(yday, np.array(MYD11A2_MTH_DAYS8[1]), False)
    
    #janMask = lstData.mth == 1
    
    lstToTest = lstData.dsVarMain[0,yStart:yEnd,xStart:xEnd]
    lstImp = np.zeros(lstToTest.shape)
    lstToTest = None
    
    schk = status_check(lstImp.size, 100)
    for x,x1 in zip(np.arange(xStart,xEnd),np.arange(lstImp.shape[1])):
        
        for y,y1 in zip(np.arange(yStart,yEnd),np.arange(lstImp.shape[0])):
            #imp.imputeNorm(x,y)
            #print x,y
            lstImp[y1,x1] = np.mean(imp.imputeNorm(x,y)[janMask])
            schk.increment() 
    
#    lstImp = np.ma.masked_equal(lstImp, 0)
#    plt.imshow(lstImp)
#    plt.colorbar()
#    plt.show() 
    

def mosaic_ncstack():
    lstVar = 'LST_Night_1km'
    stackPath = '/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks/'
    os.chdir(stackPath)
    
    ncStackFiles = np.array(os.listdir(stackPath))
    ncStackFiles = ncStackFiles[np.char.startswith(ncStackFiles, lstVar)]
    
    tiles = np.array([x.split('.')[1] for x in ncStackFiles])
    cols = np.array([np.int(x[1:3]) for x in tiles])
    rows = np.array([np.int(x[4:6]) for x in tiles])
    
    sidx = np.lexsort((cols,rows))
    tiles = tiles[sidx]
    cols = cols[sidx]
    rows = rows[sidx]
    
    for r in np.unique(rows):
        print tiles[r==rows]

def check_mthly_rasters_ndata(pathMthly,fpathAnn,lstVar):
    
    dsAnn = RasterDataset(fpathAnn)
    ann = dsAnn.readAsArray()
    
    for mth in np.arange(1,13):
        fpath = "".join([pathMthly,'MOSAIC.',lstVar,".%02d.C.tif"%mth])
        ds = RasterDataset(fpath)
        mthVal = ds.readAsArray()
        
        nodataVals = np.logical_and(~ann.mask,mthVal.mask)
        plt.imshow(np.ma.masked_equal(nodataVals,0))
        plt.show()
        print mth,np.sum(np.logical_and(~ann.mask,mthVal.mask))

def check_mthly_rasters_ndata2(pathMthly,lstVar):
    
    for mth in np.arange(1,13):
        fpath = "".join([pathMthly,'MOSAIC.',lstVar,".%02d.C.tif"%mth])
        ds = RasterDataset(fpath)
        mthVal1 = ds.readAsArray()
        
        for mth2 in np.arange(mth+1,13):
            fpath2 = "".join([pathMthly,'MOSAIC.',lstVar,".%02d.C.tif"%mth2])
            ds2 = RasterDataset(fpath2)
            mthVal2 = ds2.readAsArray()
        
            print mth,mth2,np.sum(np.logical_or(np.logical_and(~mthVal1.mask,mthVal2.mask),
                                  np.logical_and(mthVal1.mask,~mthVal2.mask)))

def interpWaterLst():

    dsLc = gdal.Open('/projects/daymet2/climate_office/modis/MOD12Q1_Special/MOSAIC.MOD12Q1.Special.hdf')
    lc = dsLc.ReadAsArray()
    dsLc = None
    
#    sd = SD('/projects/daymet2/climate_office/modis/MOD12Q1_Special/MOSAIC.MOD12Q1.Special.hdf',SDC.READ)
#    lc = sd.select('Land_Cover_Type_2')[:]
#    sd.end()
    #sd = SD('/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/MOSAIC.LST_Night_1km.01.hdf',SDC.READ)
    dsLst = gdal.Open('/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/MOSAIC.LST_Night_1km.01.hdf')

    lst = dsLst.ReadAsArray()
    fval = dsLst.GetRasterBand(1).GetNoDataValue()
    #lst = sd.select('LST_Night_1km')[:]
    #fval = sd.select('LST_Night_1km').attributes()['_FillValue']
    lst = np.ma.masked_equal(lst, 0)*0.02
    
    maskFill = ~np.logical_and(lst != fval,lc==0)
    
    #Create the lst band in memory
    mem_drv = gdal.GetDriverByName( 'MEM' )
    dsLst= mem_drv.Create('',lc.shape[1],lc.shape[0],1,gdalconst.GDT_Float64)
    bndLst = dsLst.GetRasterBand(1)
    bndLst.WriteArray(np.ma.filled(lst, fval),0,0)
    
    #Create the mask band in memory
    mem_drv = gdal.GetDriverByName( 'MEM' )
    dsMask = mem_drv.Create('',lc.shape[1],lc.shape[0],1,gdalconst.GDT_Int16)
    bndMask = dsMask.GetRasterBand(1)
    bndMask.WriteArray(maskFill.astype(np.int16),0,0)
    #gdal.FillNodata(None,None,None,None)
    gdal.FillNodata(bndLst,bndMask,100,0)
    
    plt.imshow(bndLst.ReadAsArray())
    plt.show()
    
#    x,y = getModisTileXy(sd)
#    xGrid, yGrid = np.meshgrid(x, y)
#    
#    rlst = np.ravel(lst)
#    rXGrid = np.ravel(xGrid)
#    rYGrid = np.ravel(yGrid)
#    rlc = np.ravel(lc)
#    
#    lcMask = np.logical_and(rlc != 0,rlst != fval)
#    
#    rlst = rlst[lcMask]*0.02
#    rXGridLand = rXGrid[lcMask]
#    rYGridLand = rYGrid[lcMask]
#    
#    yI,xI = np.nonzero(np.logical_and(lc == 0,lst != fval))
#    
#    interpGrid = griddata(rXGridLand,rYGridLand,rlst,np.array([xGrid[yI[0],xI[0]]]),np.array([yGrid[yI[0],xI[0]]]))
#    print interpGrid
    #np.save('/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/MOSAIC.LST_Night_1km.01.npy', interpGrid)
    #plt.imshow(interpGrid)
    #plt.show()


def interpWaterLstGridData():

    dsLc = gdal.Open('/projects/daymet2/climate_office/modis/MOD12Q1_Special/MOSAIC.MOD12Q1.Special.hdf')
    lc = dsLc.ReadAsArray()
    dsLc = None
    
#    sd = SD('/projects/daymet2/climate_office/modis/MOD12Q1_Special/MOSAIC.MOD12Q1.Special.hdf',SDC.READ)
#    lc = sd.select('Land_Cover_Type_2')[:]
#    sd.end()
    sd = SD('/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/MOSAIC.LST_Night_1km.01.hdf',SDC.READ)
    #dsLst = gdal.Open('/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/MOSAIC.LST_Night_1km.01.hdf')

    #lst = dsLst.ReadAsArray()
    #fval = dsLst.GetRasterBand(1).GetNoDataValue()
    lst = sd.select('LST_Night_1km')[:]
    fval = sd.select('LST_Night_1km').attributes()['_FillValue']
    lst = np.ma.masked_equal(lst, 0)*0.02
    
    maskFill = ~np.logical_and(lst != fval,lc==0)

    
    x,y = getModisTileXy(sd)
    xGrid, yGrid = np.meshgrid(x, y)
    
    rlst = np.ravel(lst)
    rXGrid = np.ravel(xGrid)
    rYGrid = np.ravel(yGrid)
    rlc = np.ravel(lc)
    
    lcMask = np.logical_and(rlc != 0,rlst != fval)
    
    rlst = rlst[lcMask]*0.02
    rXGridLand = rXGrid[lcMask]
    rYGridLand = rYGrid[lcMask]
    
    yI,xI = np.nonzero(np.logical_and(lc == 0,lst != fval))
    
    y_pt = yGrid[yI[0],xI[0]]
    x_pt = xGrid[yI[0],xI[0]]
    print x_pt,y_pt
    interpGrid = griddata(rXGridLand,rYGridLand,rlst,np.array([x_pt,x_pt+1]),np.array([y_pt,y_pt+1]))
    print interpGrid
    #np.save('/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/MOSAIC.LST_Night_1km.01.npy', interpGrid)
    #plt.imshow(interpGrid)
    #plt.show()

def set_water_to_fval_all_mthly_mosaics():
    dsLc = gdal.Open('/projects/daymet2/climate_office/modis/MOD12Q1_Special/MOSAIC.MOD12Q1.Special.hdf')
    lc = dsLc.ReadAsArray()
    maskWater = lc==0
    
    varname = 'LST_Night_1km'
    in_path = '/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/'
    
    os.chdir(in_path)
    
    for mth in np.arange(1,13):
        set_to_fillval("".join(["MOSAIC.",varname,".%02d.hdf"%mth]), varname, maskWater)

def set_to_fillval(fpath,varname,mask):
    #varname: LST_Night_1km or LST_Day_1km
    sd = SD(fpath,SDC.WRITE)
    sds_lst = sd.select(varname)
    cur_fill = sds_lst.attributes()['_FillValue']
    
    a = sds_lst[:]
    a[mask] = cur_fill
    sds_lst[:] = a
    sd.end()


def reclassify_lc(in_fpath,out_fpath,ndata=255):
    
    in_ds = RasterDataset(in_fpath) 
    a = in_ds.gdalDs.GetRasterBand(1).ReadAsArray()
    
    a[a < 254] = 255
            
    ds_out = gdal.GetDriverByName('GTiff').Create(  out_fpath, 
                                                    a.shape[1], 
                                                    a.shape[0], 1, gdalconst.GDT_Int16)
    
    band_out = ds_out.GetRasterBand(1)
    band_out.SetNoDataValue(ndata)
    
    #Manually set geoT. Gdal does not read bounds correctly from MRT GeoTiff output
    geoT = list(in_ds.geoT)
    geoT[0] = -125.995833883
    geoT[3] = 53.004166448
    geoT = tuple(geoT)
    
    ds_out.SetGeoTransform(geoT)
    ds_out.SetProjection(in_ds.projection)
    band_out.WriteArray(np.ma.filled(a, ndata))
    ds_out.FlushCache()

def output_water_mask(in_fpath,out_fpath):
    
    in_ds = gdal.Open(in_fpath) 
    geoT = in_ds.GetGeoTransform()

    bnd = in_ds.GetRasterBand(1)
    a = bnd.ReadAsArray()
    maskWater = a==200
    maskElse = ~maskWater
    a[maskWater] = 0
    a[maskElse] = 1
            
    ds_out = gdal.GetDriverByName('GTiff').Create(  out_fpath, 
                                                    a.shape[1], 
                                                    a.shape[0], 1, gdalconst.GDT_Int16)
    
    band_out = ds_out.GetRasterBand(1)
        
    ds_out.SetGeoTransform(geoT)
    ds_out.SetProjection(in_ds.GetProjection())
    band_out.WriteArray(a)
    ds_out.FlushCache()


def createDtrMask():
    
    dsTmin = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Night_1km.02.C.tif') 
    dsTmax = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Day_1km.02.C.tif')
    
    stmin = dsTmin.readAsArray()
    stmax = dsTmax.readAsArray()
    
    sdtr = np.ma.abs(stmax - stmin)
    
    dtrMask = np.zeros(sdtr.shape,dtype=np.int16)
    dtrMask[sdtr <= 5.0] = 1
            
    ds_out = gdal.GetDriverByName('GTiff').Create(  '/projects/daymet2/dem/interp_grids/tifs/mthly_lst/dtrMask5w.tif', 
                                                    dtrMask.shape[1], 
                                                    dtrMask.shape[0], 1, gdalconst.GDT_Int16)
    
    band_out = ds_out.GetRasterBand(1)
    #band_out.SetNoDataValue(ndata)
        
    ds_out.SetGeoTransform(dsTmin.geoT)
    ds_out.SetProjection(dsTmin.projection)
    band_out.WriteArray(dtrMask)
    ds_out.FlushCache()

if __name__ == '__main__':

    createDtrMask()
    
   
#    output_water_mask('/projects/daymet2/climate_office/modis/MOD44B/mosaic_vcf_mask.tif','/projects/daymet2/climate_office/modis/MOD44B/mosaic_vcf_mask2.tif')

    #interpWaterLstGridData()
#    check_mthly_rasters_ndata2('/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Day/mosaics/','LST_Day_1km')
#    check_mthly_rasters_ndata('/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Night/mosaics/',
#                              '/projects/daymet2/dem/interp_grids/tifs/lst_tmin.tif', 'LST_Night_1km')

#    check_mthly_rasters_ndata('/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Day/mosaics/',
#                              '/projects/daymet2/dem/interp_grids/tifs/lst_tmax.tif', 'LST_Day_1km')


#    reclassify_lc('/projects/daymet2/climate_office/modis/MOD12Q1_Special/MOSAIC.MOD12Q1.Special.WaterOnly.Land_Cover_Type_2.tif',
#                  '/projects/daymet2/climate_office/modis/MOD12Q1_Special/MOSAIC.MOD12Q1.Special.WaterOnly.Fnl.tif')

    #interpWaterLst()    
    #mosaic_mod12q1()
    #to_celsius_mthly_mosaics()
    #resample_all_mthly_mosaics()
    #reset_fillval_all_mthly_mosaics()
    #set_water_to_fval_all_mthly_mosaics()
    #mosaic_all_mthly()
    
    
    #mosaic_ncstack()
    #testImputeLstNormFnl()
    #cProfile.run('testImputeLstNormFnl()')
    #testImputeLstNorm()
    #testImputeLstNormFnl()
    #testImputeLstNorm()
    #examineLst()
    #lstMthMeans()

    #stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin_8day.nc', 'tmin',stn_dtype=DTYPE_STN_BASIC)
    #impLst = ImputeLST(stnda, 'tmin')
    #impute_ncstack('/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks/LST_Night_1km.h11v05.nc', 'LST_Night_1km', stnda,impLst)

    #create8dayStnDb()
    #linear_interp_ncstack('/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks_test/LST_Night_1km.h11v05.nc', 'LST_Night_1km')
    #stack_all_to_nc()

    #mv_MOD11A2_to_MYD11A2()
    #output_mth_means()
    #output_all_celsius()
    #resample_all_mosaics()
    #reset_fillval_all_mosaics()
    #mosaic_all_8day()
    #stack_MYDMOD11A2_mth_all()

#    path = '/projects/daymet2/climate_office/modis/MYD11A2/'
#    ncdf_path = "".join([path,'temp_lst.nc'])
#    out_fpath = ''
#    ydayPeriod = 169
#    
#    stack_MYD11A2_hdf_8day('/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.h10v04/', ncdf_path,
#                           out_fpath, ydayPeriod)
