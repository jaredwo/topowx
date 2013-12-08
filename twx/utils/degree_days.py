'''
Created on Oct 11, 2012
Functions for calculating degree day metrics
@author: jared.oyler
'''
import numpy as np
from netCDF4 import Dataset
import utils.util_dates as utld
from utils.util_dates import YEAR
from netCDF4 import num2date
import matplotlib.pyplot as plt

WHEAT_GDD_BASE = 32
HAUN_STAGE2_GDD = 395
MAX_TMAX_BEFORE_HAUN2 = 70
MAX_TMAX_AFTER_HAUN2 = 95

HEAT_COOL_DD_BASE = 65

def c_to_f(tair):
    
    return (tair*1.8) + 32.0

def dd_heating(tmin,tmax):
    
    #convert to fahrenheit
    tmin = c_to_f(tmin)
    tmax = c_to_f(tmax)
    
    tavg = (tmin + tmax)/2.0
    
    hdd = HEAT_COOL_DD_BASE - tavg
    hdd[hdd < 0] = 0
    
    return hdd

def dd_cooling(tmin,tmax):
    
    #convert to fahrenheit
    tmin = c_to_f(tmin)
    tmax = c_to_f(tmax)
    
    tavg = (tmin + tmax)/2.0
    
    cdd = tavg - HEAT_COOL_DD_BASE 
    cdd[cdd < 0] = 0
    
    return cdd

def gdd_wheat_basic(tmin,tmax):
    #convert to fahrenheit
    tmin = c_to_f(tmin)
    tmax = c_to_f(tmax)
    tavg = (tmin + tmax)/2.0
    
    gdd = tavg - WHEAT_GDD_BASE
    gdd[gdd < 0] = 0
    
    return gdd

def gdd_wheat(tmin,tmax):
    '''
    Calculates cumulative wheat growing degree days in Fahrenheit for the provided time series of 
    daily tmin and tmax.  Calculation is based off: http://ndawn.ndsu.nodak.edu/help-wheat-growing-degree-days.html 
    
    :param tmin: numpy array of tmin in Celsius
    :param tmax: numpy array of tmax in Celsius
    '''
    
    #convert to fahrenheit
    tmin = c_to_f(tmin)
    tmax = c_to_f(tmax)
    
    #Any tmin/tmax below freezing, set to freezing
    tmin[tmin < 32] = 32
    tmax[tmax < 32] = 32
    
    #Calculate daily average temperature
    tavg = (tmin + tmax)/2.0
    
    #Calculate daily GDD and accumulated sum
    gdd = tavg - WHEAT_GDD_BASE
    gdd[gdd < 0] = 0
    gdd = np.cumsum(gdd,axis=0)
    
    #There are 2 constraints on Tmax depending on when haun stage 2 was reached
    mask_haun2 = gdd >= HAUN_STAGE2_GDD
    tmax[np.logical_and(np.logical_not(mask_haun2),tmax > MAX_TMAX_BEFORE_HAUN2)] = MAX_TMAX_BEFORE_HAUN2
    tmax[np.logical_and(mask_haun2,tmax > MAX_TMAX_AFTER_HAUN2)] = MAX_TMAX_AFTER_HAUN2
    
    #Recaulate GDD with the Tmax constraints now set to get final gdd
    tavg = (tmin + tmax)/2.0
    gdd = tavg - WHEAT_GDD_BASE
    gdd[gdd < 0] = 0
    #gdd = np.cumsum(gdd)
    
    return gdd

if __name__ == '__main__':
    
    stn_id = 'GHCN_USW00024143'#'GHCN_USC00241722'#'GHCN_USW00024143' #GREAT FALLS INTL AP HCN station
    ds_tmin = Dataset('/projects/daymet2/climate_office/impute_tmin_mt.nc')
    ds_tmax = Dataset('/projects/daymet2/climate_office/impute_tmax_mt.nc')
    
    stnids_tmin = ds_tmin.variables['stn_id'][:].astype("<S16")
    stnids_tmax = ds_tmax.variables['stn_id'][:].astype("<S16")
    
    tmin = ds_tmin.variables['tmin'][:,stnids_tmin==stn_id].ravel().astype(np.float64)
    tmax = ds_tmax.variables['tmax'][:,stnids_tmax==stn_id].ravel().astype(np.float64)
    
    var_time = ds_tmin.variables['time']
    start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
    days = utld.get_days_metadata(start, end)
    
    #Years in 1981 - 2010 normals
    norm_yrs = np.arange(1981,2011)
    
    ann_gdd = np.zeros(norm_yrs.size)
    
    for x in np.arange(norm_yrs.size):
        
        yr_mask = days[YEAR] == norm_yrs[x]
        
        ann_gdd[x] = np.sum(gdd_wheat(tmin[yr_mask], tmax[yr_mask]))
     
    norm_gdd = np.mean(ann_gdd)
    print ann_gdd
    plt.plot(norm_yrs,ann_gdd,"b")
    plt.xlim(norm_yrs[0],norm_yrs[-1])
    plt.hlines(norm_gdd,plt.xlim()[0],plt.xlim()[1],color="red")
    plt.ylabel("Wheat Growing Degree Days")
    plt.xlabel("Year")
    plt.legend(["Annual GDD","1981-2010 Normal"])
    plt.title("Great Falls Annual Wheat Growing Degree Days 1981 - 2010")
    plt.show()
    