'''
Created on Feb 7, 2012

@author: jared.oyler
'''
from pyhdf.SD import SD, SDC
from to_ncdf import modis_et_dataset
import numpy as np
import utils.util_dates as utld
from utils.util_dates import YEAR
from datetime import datetime
import netCDF4
from et.to_ncdf import SCALE_FACTOR
import os
import matplotlib.pyplot as plt
import sys

TDAYCOEF = 0.45

OUT_PATH = '/projects/daymet2/modis/topomet_hdf4/ann_files/'

def test_outputs():
        
    sd = SD("".join([OUT_PATH,'rh2000.gz.hdf']), SDC.READ)
    sds = sd.select("rh")
    a = np.ma.masked_equal(sds[213,:,:],sds.getfillvalue())
    a = a*sds.scale_factor
    print np.min(a)
    print np.max(a)
    plt.imshow(a)
    plt.show()

def unzip():
    
    fnames = os.listdir(OUT_PATH)
    
    for fname in fnames:
        
        if ".hdf" in fname:
             
            sd = SD("".join([OUT_PATH,fname]), SDC.READ)
            
            ds_name = sd.datasets().keys()[0]
            ds_dtype = sd.datasets()[ds_name][2]
            ds_shape = sd.datasets()[ds_name][1]
            
            sds = sd.select(ds_name)
            a = sds[:]
            
            ds_fill = sds.getfillvalue()
            ds_scale = sds.attributes()['scale_factor']
            
            sds.endaccess()
            sd.end()
            sds = None
            sd = None
            
            new_fname = "".join([fname.split('.')[0],".hdf"])
            
            sd_out = SD("".join([OUT_PATH,'unzipped/',new_fname]), SDC.WRITE | SDC.CREATE)
            
            sds_out = sd_out.create(ds_name,ds_dtype,ds_shape)
    
            #Set dimension names
            dim0 = sds_out.dim(0)
            dim0.setname("time")
            dim1 = sds_out.dim(1)
            dim1.setname("row")
            dim2 = sds_out.dim(2)
            dim2.setname("col")
            
            sds_out.scale_factor = ds_scale
            sds_out.setfillvalue(ds_fill)
            sds_out[:] = a
            
            sds_out.endaccess()
            sd_out.end()
            
            sds_out = None
            sd_out = None
            a = None
            
            print fname
        
if __name__ == '__main__':
    
    unzip()
    #test_outputs()
    sys.exit()
    
    '''
    double* tmin; /*degrees C*/ *0.01 scale factor
    double* tmax; /*degrees C*/ *0.01 scale factor
    double* tday; /*degrees C*/ *0.01 scale factor
    double* prcp; /*cm*/
    double* srad; /*W m-2*/ *0.1 scale factor
    double* vpd; /*Pa*/ *0.1 (unsigned int)
    double* rh; /*rel. humid. as fraction*/ *0.001 scale factor
    '''
    
    SCALE_FACTOR = 0.001
    VAR_NAME = "rh"
    
#    sd1 = SD("/projects/daymet2/modis/topomet_hdf4/vpd1.hdf", SDC.READ)
#    sd2 = SD("/projects/daymet2/modis/topomet_hdf4/vpd2.hdf", SDC.READ)
#    sds1 = sd1.select(VAR_NAME)
#    sds2 = sd2.select(VAR_NAME)

    #TDAY
    ########################################################################
    sd1 = SD("/projects/daymet2/modis/topomet_hdf4/tmin1.hdf", SDC.READ)
    sd2 = SD("/projects/daymet2/modis/topomet_hdf4/tmin2.hdf", SDC.READ)
    sds1 = sd1.select("tmin")
    sds2 = sd2.select("tmin")
    tmin1 = sds1[:]
    tmin2 = sds2[:]
    tmin = np.vstack((tmin1,tmin2))
    tmin1 = None
    tmin2 = None
    
    sd1 = SD("/projects/daymet2/modis/topomet_hdf4/tmax1.hdf", SDC.READ)
    sd2 = SD("/projects/daymet2/modis/topomet_hdf4/tmax2.hdf", SDC.READ)
    sds1 = sd1.select("tmax")
    sds2 = sd2.select("tmax")
    tmax1 = sds1[:]
    tmax2 = sds2[:]
    tmax = np.vstack((tmax1,tmax2))
    tmax1 = None
    tmax2 = None
    
    tmean = (tmax + tmin)/2.0;
    tday = ((tmax - tmean)*TDAYCOEF) + tmean;
    tmax = None
    tmin = None
    tmean = None
    #a = tday
    
    sd1 = SD("/projects/daymet2/modis/topomet_hdf4/vpd1.hdf", SDC.READ)
    sd2 = SD("/projects/daymet2/modis/topomet_hdf4/vpd2.hdf", SDC.READ)
    sds1 = sd1.select("vpd")
    sds2 = sd2.select("vpd")
    vpd1 = sds1[:]
    vpd2 = sds2[:]
    vpd = np.vstack((vpd1,vpd2))
    vpd1 = None
    vpd2 = None
    
    tday = np.array(tday,dtype=np.float64)
    vpd = np.array(vpd,dtype=np.float64)
    
    #/*Get the saturated vapor pressure*/
    vps = 610.7 * np.exp(17.38 * tday/(239.0+tday));
    #/*Get the actual vapor pressure*/
    vpa = vps - vpd;
    #/*Get relative humidity*/
    rh = vpa/vps;
    a = rh
    
    print np.min(a)
    print np.max(a)
    ########################################################################
    
#    a1 = sds1[:]
#    a2 = sds2[:]
#    a = np.vstack((a1,a2))
#    a1 = None
#    a2 = None
#    
    ds_et = modis_et_dataset('/projects/daymet2/modis/MOD16A2.A2009177.h10v04.105.2010355183415.hdf')
    nrows = ds_et.et_a.shape[0]
    ncols = ds_et.et_a.shape[1]
    
    days = utld.get_days_metadata(datetime(2000, 1, 1), datetime(2009, 12, 31))
    
    for yr in np.unique(days[YEAR]):
        
        yr_mask = days[YEAR] == yr
        ndays = np.nonzero(yr_mask)[0].size
        a_out = np.array(np.ones((ndays,nrows,ncols))*netCDF4.default_fillvals["i2"],dtype=np.int16)
        
#        120 401 rows
#        125 775 cols
        a_out[:,120:402,125:776] = np.array(np.around(a[yr_mask,:,:]/SCALE_FACTOR),dtype=np.int16)
        
        sd_out = SD("".join([OUT_PATH,VAR_NAME,str(yr),".hdf"]), SDC.WRITE | SDC.CREATE)
        
        sds_out = sd_out.create(VAR_NAME, SDC.INT16,(ndays,nrows,ncols))

        #Set dimension names
        dim0 = sds_out.dim(0)
        dim0.setname("time")
        dim1 = sds_out.dim(1)
        dim1.setname("row")
        dim2 = sds_out.dim(2)
        dim2.setname("col")
        
        sds_out.scale_factor = SCALE_FACTOR
        sds_out.setfillvalue(netCDF4.default_fillvals["i2"])#netCDF4.default_fillvals["i2"]
        sds_out[:] = a_out
        
        sds_out.endaccess()
        sd_out.end()
        
        print yr
    
    os.chdir("/projects/daymet2/modis/topomet_hdf4/ann_files/")
    
    for yr in np.unique(days[YEAR]):
        
        cmd = "".join(["hrepack -v -i ","".join([VAR_NAME,str(yr),".hdf"])," -o ","".join([VAR_NAME,str(yr),".gz.hdf"])," -t '*:GZIP 8'"])
        print cmd
        os.system(cmd)
        
    
        