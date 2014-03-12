'''
Created on Oct 28, 2013

@author: jared.oyler
'''
from twx.utils.input_raster import RasterDataset
from netCDF4 import Dataset
import twx.utils.util_dates as utld
from datetime import datetime
from twx.utils.util_dates import YEAR, MONTH, DATE, YMD
from netcdftime import date2num
import numpy as np
import os
import matplotlib.pyplot as plt
from twx.utils.status_check import status_check

SCALE_FACTOR = np.float32(0.01)
CCE_BBOX = (-117.0,-109.0,45.0,50.0)

def create_prism_nc(aPrismRast,varname,fpathOut,startYr=1981,endYr=2010):
    
    lat,lon = aPrismRast.getCoordGrid1d()
#    lon = lon[np.logical_and(lon>=CCE_BBOX[0],lon<=CCE_BBOX[1])]
#    lat = lat[np.logical_and(lat>=CCE_BBOX[2],lat<=CCE_BBOX[3])]
    
    ndata = aPrismRast.gdalDs.GetRasterBand(1).GetNoDataValue()
    
    dsOut = Dataset(fpathOut,'w')
        
    dsOut.createDimension('lon',lon.size)
    dsOut.createDimension('lat',lat.size)
    
    print "Dataset is %d * %d"%(lat.size,lon.size)
        
    latitudes = dsOut.createVariable('lat','f8',('lat',))
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lat

    longitudes = dsOut.createVariable('lon','f8',('lon',))
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lon
    
    days = utld.get_days_metadata(datetime(startYr,1,1), datetime(endYr,12,31))
    dsOut.createDimension('time',days.size)
    times = dsOut.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.units = "".join(["days since ",str(days[0][YEAR]),"-",str(days[0][MONTH]),"-",str(days[0][DATE].day)," 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE], times.units)
    
    mainvar = dsOut.createVariable(varname,'i2',('time','lat','lon'),
                                   chunksizes=(365,50,50),#chunksizes=(days[DATE].size,50,50)
                                   fill_value=ndata,zlib=True)
    mainvar.scale_factor = SCALE_FACTOR
    
    dsOut.sync()

def addAnnZipToNc(yr,varname,zipPath,unzipPath,days,dsOut):
    
    os.system("".join(['unzip ',zipPath,'PRISM_',varname,"_stable_4kmD1_%d0101_%d1231_bil.zip > /dev/null"%(yr,yr)," -d ",unzipPath]))
    
    maskYr = days[YEAR]==yr
    ndays = np.sum(maskYr)
    daysYr = days[maskYr]
    maskYr = np.nonzero(maskYr)[0]
    
    lon = dsOut.variables['lon'][:]
    minLon,maxLon = np.min(lon),np.max(lon)
    lat = dsOut.variables['lat'][:]
    minLat,maxLat = np.min(lat),np.max(lat)
    
    a = np.zeros((ndays,lat.size,lon.size),dtype=np.int16)
    
    ymd = daysYr[YMD][0]
    dsDay = RasterDataset("".join([unzipPath,"PRISM_",varname,"_stable_4kmD1_%d_bil.bil"%(ymd,)]))
    latPrism,lonPrism = dsDay.getCoordGrid1d()
    
    colStart,colEnd = np.nonzero(np.logical_and(lonPrism>=minLon,lonPrism<=maxLon))[0][[0,-1]]
    colEnd = colEnd + 1
    rowStart,rowEnd = np.nonzero(np.logical_and(latPrism>=minLat,latPrism<=maxLat))[0][[0,-1]]
    rowEnd = rowEnd + 1
    
    varOut = dsOut.variables[varname]
    varOut.set_auto_maskandscale(False)
    
    for aday,x in zip(daysYr,np.arange(ndays)):
        dsDay = RasterDataset("".join([unzipPath,"PRISM_",varname,"_stable_4kmD1_%d_bil.bil"%(aday[YMD],)]))
        #print "".join([unzipPath,"PRISM_",varname,"_stable_4kmD1_%d_bil.bil"%(aday[YMD],)])
        valsDay = dsDay.gdalDs.GetRasterBand(1).ReadAsArray()[rowStart:rowEnd,colStart:colEnd]
        #print "Day Array Size %d * %d"%valsDay.shape
        valsDay = np.ma.masked_equal(valsDay,-9999)
        valsDay = np.round(valsDay,2)/SCALE_FACTOR
        valsDay = np.ma.filled(valsDay, -9999)
        valsDay = valsDay.astype(np.int16)
        a[x,:,:] = valsDay
    
    varOut[maskYr,:,:] = a
    dsOut.sync()
    
    os.system("rm "+unzipPath+"*")
    

if __name__ == '__main__':
    
    #TMIN
#    startYr=1981
#    endYr=2012
#    #ds = RasterDataset('/projects/daymet2/prism/4km_daily/annual_zips/PRISM_tmin_stable_4kmD1_19810831_bil.bil')
#    #createPrismNcDataset(ds,'tmin', '/projects/daymet2/prism/4km_daily/netcdf/prism4km_conus_tmin.nc',startYr,endYr)
#    
#
#    days = utld.get_days_metadata(datetime(startYr,1,1), datetime(endYr,12,31))
#    dsOut = Dataset('/projects/daymet2/prism/4km_daily/netcdf/prism4km_conus_tmin.nc','r+')
#    os.system("rm /projects/daymet2/prism/4km_daily/annual_zips/annual_unzip/*")
#    yrs = np.arange(startYr,endYr+1)
#    schk = status_check(yrs.size,1)
#    for yr in yrs:
#    
#        addAnnZipToNc(yr, 'tmin', '/projects/daymet2/prism/4km_daily/annual_zips/',
#                      '/projects/daymet2/prism/4km_daily/annual_zips/annual_unzip/', days, dsOut)
#        print yr
#        schk.increment()
    
    #TMAX
    startYr=1981
    endYr=2012
#    ds = RasterDataset('/projects/daymet2/prism/4km_daily/annual_zips/PRISM_tmin_stable_4kmD1_19810831_bil.bil')
#    createPrismNcDataset(ds,'tmax', '/projects/daymet2/prism/4km_daily/netcdf/prism4km_conus_tmax.nc',startYr,endYr)
#    

    days = utld.get_days_metadata(datetime(startYr,1,1), datetime(endYr,12,31))
    dsOut = Dataset('/projects/daymet2/prism/4km_daily/netcdf/prism4km_conus_tmax.nc','r+')
    os.system("rm /projects/daymet2/prism/4km_daily/annual_zips/annual_unzip/*")
    for yr in np.arange(startYr,endYr+1):
    
        addAnnZipToNc(yr, 'tmax', '/projects/daymet2/prism/4km_daily/annual_zips/tmax/',
                      '/projects/daymet2/prism/4km_daily/annual_zips/annual_unzip/', days, dsOut)
        print yr
