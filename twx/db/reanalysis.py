'''
Created on Nov 6, 2012

@author: jared.oyler

0 UTC + 1 day (daily max temperature)
12 UTC (daily min temperature)

Bounds
60 lat
10 lat
-135 lon
-55 lon

Levels
850
700
600
500

850 mb | 1456.7 m
700 mb | 3010.9 m
600 mb | 4204.7 m
500 mb | 5572.1 m



'''

from netCDF4 import Dataset,num2date
import netCDF4
import numpy as np
from twx.utils.util_dates import A_DAY,YMD
import twx.utils.util_dates as utld
import twx.utils.util_geo as utlg
from twx.db.create_db_all_stations import dbDataset
from datetime import datetime

LEVELS_STD_ALTITUDE = {1000:110.8,925:761.7,850:1456.7,700:3010.9,600:4204.7,500:5572.1} #meters
UTC_TIMES = {0:0,6:6,12:12,18:18,24:0}
LAT_TOP,LAT_BOTTOM = 60,10
LON_LEFT,LON_RIGHT = -135,-55

class OutsideLevelException(Exception):
    pass

def linear_interp(x,x0,x1,y0,y1):
    
    return y0 + ((x-x0)*((y1-y0)/(x1-x0)))

def linear_extrap(x,x0,x1,y0,y1):
    
    return y0 + (((x - x0)/(x1 - x0))*(y1-y0))

class NNRNghData():
    
    NNR_VARS = np.array(['tair','hgt','thick','rhum','uwnd','vwnd','slp'])
    #NNR_VARS = np.array(['tair','thick','hgt','rhum','slp'])
    #NNR_VARS = np.array(['tair','thick'])
    NNR_TIMES = np.array(['24z','18z','12z'])
    TMIN = 'tmin'
    TMAX = 'tmax'
    UTC_OFFSET_TIMES = {TMIN:{-4:'12z',-5:'12z',-6:'12z',-7:'12z',-8:'12z'},
                        TMAX:{-4:'18z',-5:'18z',-6:'18z',-7:'24z',-8:'24z'}}

    def __init__(self, path_nnr,startend_ymd,nnr_vars=None):
        
        self.ds_nnr = {}
        
        if nnr_vars is None:
            self.nnr_vars = self.NNR_VARS
        else:
            self.nnr_vars = nnr_vars
        
        
        for nnr_var in self.nnr_vars:
            
            for nnr_time in self.NNR_TIMES:
                
                self.ds_nnr["".join([nnr_var,nnr_time])] = Dataset("".join([path_nnr,"nnr_",nnr_var,"_",nnr_time,".nc"]))
        
        eg_ds = self.ds_nnr.values()[0]
                
        var_time = eg_ds.variables['time']
        
        start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
        self.days = utld.get_days_metadata(start, end)
        
        self.day_mask = np.nonzero(np.logical_and(self.days[YMD] >= startend_ymd[0], self.days[YMD] <= startend_ymd[1]))[0]
        self.days = self.days[self.day_mask]
        
        self.nnr_lons = eg_ds.variables['lon'][:]
        self.nnr_lats = eg_ds.variables['lat'][:]
        llgrid = np.meshgrid(self.nnr_lons,self.nnr_lats)
        
        self.grid_lons = llgrid[0].ravel()
        self.grid_lats = llgrid[1].ravel()
        
    def get_nngh_matrix(self,lon,lat,tair_var,utc_offset=-7,nngh=4):
        
        dist_nnr = utlg.grt_circle_dist(lon, lat, self.grid_lons, self.grid_lats)
        sort_dist_nnr = np.argsort(dist_nnr)
        
        nnr_ngh_lons = self.grid_lons[sort_dist_nnr][0:nngh]
        nnr_ngh_lats = self.grid_lats[sort_dist_nnr][0:nngh]
        
        nnr_time = self.UTC_OFFSET_TIMES[tair_var][utc_offset]
        
        nnr_matrix = None
        
        for x in np.arange(nnr_ngh_lons.size):
            
            idx_lon = np.nonzero(self.nnr_lons == nnr_ngh_lons[x])[0][0]
            idx_lat = np.nonzero(self.nnr_lats == nnr_ngh_lats[x])[0][0]
            
            for nnr_var in self.nnr_vars:
            
                ds = self.ds_nnr["".join([nnr_var,nnr_time])]
                
                if "level" in ds.dimensions:
                    adata = ds.variables[nnr_var][self.day_mask,:,idx_lat,idx_lon]
                else:
                    adata = ds.variables[nnr_var][self.day_mask,idx_lat,idx_lon]
            
                if len(adata.shape) == 1:
                    adata.shape = (adata.size,1)
                    
#                if np.ma.isMA(adata):
#                    
#                    #print "|".join(['NNR NODATA',nnr_var,str(nnr_ngh_lons[x]),str(nnr_ngh_lats[x]),str(self.days[adata.mask.ravel()])])
#                    adata = np.ma.filled(adata, np.nan)
                if nnr_matrix is None:
                    nnr_matrix = adata
                else:
                    nnr_matrix = np.hstack((nnr_matrix,adata))
        

        
        return nnr_matrix

class NNRds(object):
    '''
    '''

    def __init__(self, path_nnr, var_name,startend_ymd):

        self.ds_nnr = Dataset(path_nnr)
        self.var_name = var_name
        
        var_time = self.ds_nnr.variables['time']
        
        start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
        self.days = utld.get_days_metadata(start, end)
        
        self.day_mask = np.nonzero(np.logical_and(self.days[YMD] >= startend_ymd[0], self.days[YMD] <= startend_ymd[1]))[0]
        self.days = self.days[self.day_mask]
        
        self.nnr_lons = self.ds_nnr.variables['lon'][:]
        self.nnr_lats = self.ds_nnr.variables['lat'][:]
        llgrid = np.meshgrid(self.nnr_lons,self.nnr_lats)
        
        self.grid_lons = llgrid[0].ravel()
        self.grid_lats = llgrid[1].ravel()
     
        self.levels = self.ds_nnr.variables['level'][:]
        self.elevs = np.array([LEVELS_STD_ALTITUDE[x] for x in self.levels])

    def interp_variable_mean(self,lon,lat,elev,nngh=4):
                        
        dist_nnr = utlg.grt_circle_dist(lon, lat, self.grid_lons, self.grid_lats)
        sort_dist_nnr = np.argsort(dist_nnr)
        
        nnr_ngh_dists = dist_nnr[sort_dist_nnr][0:nngh]
        nnr_ngh_lons = self.grid_lons[sort_dist_nnr][0:nngh]
        nnr_ngh_lats = self.grid_lats[sort_dist_nnr][0:nngh]
        
        ul_lon,ul_lat = np.min(nnr_ngh_lons),np.max(nnr_ngh_lats)
        lr_lon,lr_lat = np.max(nnr_ngh_lons),np.min(nnr_ngh_lats)
        
        dmax = utlg.grt_circle_dist(ul_lon, ul_lat, lr_lon, lr_lat)
        
        extrap_below = False
        extrap_above = False
        if elev < self.elevs[0]:
            
            #pt elev is below lowest level elev: need to extrapolate
            lwr_bnd = 0
            up_bnd = 1
            extrap_below = True
        
        elif elev > self.elevs[-1]:
            
            lwr_bnd = self.elevs.size - 2
            up_bnd = self.elev.size - 1
            extrap_above = True
        
        else:
    
            lwr_bnd = np.nonzero(self.elevs<=elev)[0][-1]
            up_bnd = np.nonzero(self.elevs>=elev)[0][0]
        
        lwr_elev = self.elevs[lwr_bnd]
        up_elev = self.elevs[up_bnd]
        levels = [lwr_bnd,up_bnd]
        
        mean_obs = np.zeros(nnr_ngh_lons.size)
        
        for x in np.arange(nnr_ngh_lons.size):
            
            idx_lon = np.nonzero(self.nnr_lons == nnr_ngh_lons[x])[0][0]
            idx_lat = np.nonzero(self.nnr_lats == nnr_ngh_lats[x])[0][0]
            
            #obs = self.ds_nnr.variables[self.var_name][self.day_mask,3,idx_lat,idx_lon]
            obs = self.ds_nnr.variables[self.var_name][self.day_mask,levels,idx_lat,idx_lon]
            lwr_obs = np.mean(obs[:,0],dtype=np.float64)
            up_obs = np.mean(obs[:,1],dtype=np.float64)
            
            if extrap_below:
                obs = linear_extrap(elev, up_elev, lwr_elev, up_obs, lwr_obs)
            elif extrap_above:
                obs = linear_extrap(elev, lwr_elev, up_bnd, lwr_obs, up_obs)
            else:
                obs = linear_interp(elev,lwr_elev,up_elev,lwr_obs,up_obs)

            mean_obs[x] = obs
        
        Di = np.cos((np.pi/2.0)*(nnr_ngh_dists/dmax))**4
        #Di = (1.0+np.cos(np.pi*(nnr_ngh_dists/dmax)))/2.0
        Wi = Di/np.sum(Di)
        
        interp_obs = np.average(mean_obs,weights=Wi)
        
        return interp_obs
 
    def interp_variable(self,lon,lat,elev,nngh=4):
                        
        dist_nnr = utlg.grt_circle_dist(lon, lat, self.grid_lons, self.grid_lats)
        sort_dist_nnr = np.argsort(dist_nnr)
        
        nnr_ngh_dists = dist_nnr[sort_dist_nnr][0:nngh]
        nnr_ngh_lons = self.grid_lons[sort_dist_nnr][0:nngh]
        nnr_ngh_lats = self.grid_lats[sort_dist_nnr][0:nngh]
        
        ul_lon,ul_lat = np.min(nnr_ngh_lons),np.max(nnr_ngh_lats)
        lr_lon,lr_lat = np.max(nnr_ngh_lons),np.min(nnr_ngh_lats)
        
        dmax = utlg.grt_circle_dist(ul_lon, ul_lat, lr_lon, lr_lat)
        
        extrap_below = False
        extrap_above = False
        if elev < self.elevs[0]:
            
            #pt elev is below lowest level elev: need to extrapolate
            lwr_bnd = 0
            up_bnd = 1
            extrap_below = True
        
        elif elev > self.elevs[-1]:
            
            lwr_bnd = self.elevs.size - 2
            up_bnd = self.elev.size - 1
            extrap_above = True
        
        else:
    
            lwr_bnd = np.nonzero(self.elevs<=elev)[0][-1]
            up_bnd = np.nonzero(self.elevs>=elev)[0][0]
        
        lwr_elev = self.elevs[lwr_bnd]
        up_elev = self.elevs[up_bnd]
        levels = [lwr_bnd,up_bnd]
        
        obs_matrix = None
        
        for x in np.arange(nnr_ngh_lons.size):
            
            idx_lon = np.nonzero(self.nnr_lons == nnr_ngh_lons[x])[0][0]
            idx_lat = np.nonzero(self.nnr_lats == nnr_ngh_lats[x])[0][0]
            
            #obs = self.ds_nnr.variables[self.var_name][self.day_mask,3,idx_lat,idx_lon]
            obs = self.ds_nnr.variables[self.var_name][self.day_mask,levels,idx_lat,idx_lon].astype(np.float64)
            
            if extrap_below:
                obs = linear_extrap(elev, up_elev, lwr_elev, obs[:,1], obs[:,0])
            elif extrap_above:
                obs = linear_extrap(elev, lwr_elev, up_bnd, obs[:,0], obs[:,1])
            else:
                obs = linear_interp(elev,lwr_elev,up_elev,obs[:,0],obs[:,1])

            obs.shape = (obs.size,1)
            
            if obs_matrix is None:
                obs_matrix = obs
            else:
                obs_matrix = np.hstack((obs_matrix,obs))
        
        Di = np.cos((np.pi/2.0)*(nnr_ngh_dists/dmax))**4
        #Di = (1.0+np.cos(np.pi*(nnr_ngh_dists/dmax)))/2.0
        Wi = Di/np.sum(Di)
        
        interp_obs = np.average(obs_matrix, axis=1, weights=Wi)
        
        return interp_obs
            
    def get_nngh(self,lon,lat,nngh=4):
            
        dist_nnr = utlg.grt_circle_dist(lon, lat, self.grid_lons, self.grid_lats)
        sort_dist_nnr = np.argsort(dist_nnr)
        
        nnr_ngh_dists = dist_nnr[sort_dist_nnr][0:nngh]
        nnr_ngh_lons = self.grid_lons[sort_dist_nnr][0:nngh]
        nnr_ngh_lats = self.grid_lats[sort_dist_nnr][0:nngh]
        
        nnr_matrix = None
        nnr_ngh_dists_all = np.zeros(4 * nnr_ngh_dists.size) #4 levels
        nnr_ngh_lons_all = np.zeros(4 * nnr_ngh_lons.size)
        nnr_ngh_lats_all = np.zeros(4 * nnr_ngh_lats.size)
        nnr_ngh_elev_all = np.zeros(4 * nnr_ngh_dists.size)
        
        i = 0
        for x in np.arange(nnr_ngh_lons.size):
            
            idx_lon = np.nonzero(self.nnr_lons == nnr_ngh_lons[x])[0][0]
            idx_lat = np.nonzero(self.nnr_lats == nnr_ngh_lats[x])[0][0]
            
            adata = self.ds_nnr.variables[self.var_name][self.day_mask,:,idx_lat,idx_lon]
            
            if nnr_matrix is None:
                nnr_matrix = adata
            else:
                nnr_matrix = np.hstack((nnr_matrix,adata))
                
            nnr_ngh_dists_all[i:i+4] = nnr_ngh_dists[x]
            nnr_ngh_lons_all[i:i+4] = nnr_ngh_lons[x]
            nnr_ngh_lats_all[i:i+4] = nnr_ngh_lats[x]
            nnr_ngh_elev_all[i:i+4] = self.elevs
            
            i+=4
            
        return nnr_matrix,nnr_ngh_dists_all,nnr_ngh_lons_all,nnr_ngh_lats_all,nnr_ngh_elev_all
    
    def get_nngh_matrix(self,lon,lat,nngh=4): 
        
        return self.get_nngh(lon, lat, nngh)[0]


def KtoC(k):
    return k - 273.15

def no_conv(x):
    return x

def create_tmin_subset():
    ds_out = dbDataset('/projects/daymet2/reanalysis_data/tair/nnr_tmin.nc','w')
    
    #LEVELS = np.array([850,700,600,500])
    #LEVELS = np.array([1000, 925, 850, 700, 600, 500])
    LEVELS = np.array([850])
    LAT_TOP,LAT_BOTTOM = 60,10
    LON_LEFT,LON_RIGHT = -135,-55
    yrs = np.arange(1948,2013)
    path = '/projects/daymet2/reanalysis_data/'

    ds = Dataset("".join([path,"air.",str(yrs[0]),".nc"]))
    levels = ds.variables['level'][:]
    lons = ds.variables['lon'][:]
    lons[lons>180] = lons[lons>180]-360.0
    lats = ds.variables['lat'][:]
    days = utld.get_days_metadata(datetime(1948,1,1), datetime(2012,12,31))
    
    mask_levels = np.in1d(levels, LEVELS, True)
    mask_lons = np.logical_and(lons >= LON_LEFT, lons <= LON_RIGHT)
    mask_lats = np.logical_and(lats >= LAT_BOTTOM, lats <= LAT_TOP)
    
    ds_out.db_create_global_attributes("NCEP/NCAR Daily Tmin Subset")
    ds_out.db_create_lonlat_dimvar(lons[mask_lons], lats[mask_lats])
    ds_out.db_create_time_dimvar(days)
    ds_out.db_create_level_dimvar(levels[mask_levels])
    
    tmin_var = ds_out.createVariable('tmin','f4',('time','level','lat','lon'),
                                     fill_value=netCDF4.default_fillvals['f4'],
                                     chunksizes=(days.size,levels[mask_levels].size,4,4))
    tmin_var.long_name = "minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "minimum_air_temperature"
    tmin_var.missing_value = netCDF4.default_fillvals['f4']
    
    for yr in yrs:
        
        ds = Dataset("".join([path,"air.",str(yr),".nc"]))
        
        start,end = ds.variables['time'][0],ds.variables['time'][-1]+6
        
        times_yr = num2date(np.arange(start,end,6),units=ds.variables['time'].units)
        hours_yr = np.array([x.hour for x in times_yr])
        mask_day = hours_yr == 12
        
        tair = KtoC(ds.variables['air'][mask_day,mask_levels,mask_lats,mask_lons]) 
        
        dates_yr = times_yr[mask_day]
        ymd_yr = utld.get_ymd_array(dates_yr)
        
        mask_ymd = np.logical_and(ymd_yr >= days[YMD][0],ymd_yr <= days[YMD][-1])
        ymd_yr = ymd_yr[mask_ymd]
        tair = tair[mask_ymd,:,:,:]
        
        fnl_day_mask = np.in1d(days[YMD], ymd_yr, True)
        
        ds_out.variables['tmin'][fnl_day_mask,:,:,:] = tair
        ds_out.sync()
        
        print yr

def create_tmax_subset():
    ds_out = dbDataset('/projects/daymet2/reanalysis_data/tair/nnr_tmax.nc','w')
    
    #LEVELS = np.array([850,700,600,500])
    #LEVELS = np.array([1000, 925, 850, 700, 600, 500])
    LEVELS = np.array([850])
    LAT_TOP,LAT_BOTTOM = 60,10
    LON_LEFT,LON_RIGHT = -135,-55
    yrs = np.arange(1948,2013)
    path = '/projects/daymet2/reanalysis_data/'

    ds = Dataset("".join([path,"air.",str(yrs[0]),".nc"]))
    levels = ds.variables['level'][:]
    lons = ds.variables['lon'][:]
    lons[lons>180] = lons[lons>180]-360.0
    lats = ds.variables['lat'][:]
    days = utld.get_days_metadata(datetime(1948,1,1), datetime(2012,12,31))
    
    mask_levels = np.in1d(levels, LEVELS, True)
    mask_lons = np.logical_and(lons >= LON_LEFT, lons <= LON_RIGHT)
    mask_lats = np.logical_and(lats >= LAT_BOTTOM, lats <= LAT_TOP)
    
    ds_out.db_create_global_attributes("NCEP/NCAR Daily Tmax Subset")
    ds_out.db_create_lonlat_dimvar(lons[mask_lons], lats[mask_lats])
    ds_out.db_create_time_dimvar(days)
    ds_out.db_create_level_dimvar(levels[mask_levels])
    
    tmax_var = ds_out.createVariable('tmax','f4',('time','level','lat','lon'),
                                     fill_value=netCDF4.default_fillvals['f4'],
                                     chunksizes=(days.size,levels[mask_levels].size,4,4))
    tmax_var.long_name = "maximum air temperature"
    tmax_var.units = "C"
    tmax_var.standard_name = "maximum_air_temperature"
    tmax_var.missing_value = netCDF4.default_fillvals['f4']
    
    for yr in yrs:
        
        ds = Dataset("".join([path,"air.",str(yr),".nc"]))
        
        start,end = ds.variables['time'][0],ds.variables['time'][-1]+6
        
        times_yr = num2date(np.arange(start,end,6),units=ds.variables['time'].units)
        hours_yr = np.array([x.hour for x in times_yr])
        mask_day = hours_yr == 0
        
        tair = KtoC(ds.variables['air'][mask_day,mask_levels,mask_lats,mask_lons]) 
        
        dates_yr = times_yr[mask_day]
        dates_yr = np.array([x - A_DAY for x in dates_yr])
        ymd_yr = utld.get_ymd_array(dates_yr)
        
        mask_ymd = np.logical_and(ymd_yr >= days[YMD][0],ymd_yr <= days[YMD][-1])
        ymd_yr = ymd_yr[mask_ymd]
        tair = tair[mask_ymd,:,:,:]
        
        fnl_day_mask = np.in1d(days[YMD], ymd_yr, True)
        
        ds_out.variables['tmax'][fnl_day_mask,:,:,:] = tair
        ds_out.sync()
        
        print yr

def create_nnr_subset(path_nnr,fpath_out,yrs,days,varname_in,varname_out,levels_subset,utc_time,conv_func):
    
    ds_out = dbDataset(fpath_out,'w')
    
    ds = Dataset("".join([path_nnr,varname_in,".",str(yrs[0]),".nc"]))
    levels = ds.variables['level'][:]
    lons = ds.variables['lon'][:]
    lons[lons>180] = lons[lons>180]-360.0
    lats = ds.variables['lat'][:]
    
    mask_levels = np.in1d(levels, levels_subset, True)
    mask_lons = np.logical_and(lons >= LON_LEFT, lons <= LON_RIGHT)
    mask_lats = np.logical_and(lats >= LAT_BOTTOM, lats <= LAT_TOP)
    
    ds_out.db_create_global_attributes("".join(["NCEP/NCAR Daily ",varname_out," ",str(utc_time),"Z Subset"]))
    ds_out.db_create_lonlat_dimvar(lons[mask_lons], lats[mask_lats])
    ds_out.db_create_time_dimvar(days)
    ds_out.db_create_level_dimvar(levels[mask_levels])
    
    ds_out.createVariable(varname_out,'f4',('time','level','lat','lon'),
                          fill_value=netCDF4.default_fillvals['f4'],
                          chunksizes=(days.size,levels[mask_levels].size,4,4))
    for yr in yrs:
        
        ds = Dataset("".join([path_nnr,varname_in,".",str(yr),".nc"]))
        
        start,end = ds.variables['time'][0],ds.variables['time'][-1]+6
        
        times_yr = num2date(np.arange(start,end,6),units=ds.variables['time'].units)
        hours_yr = np.array([x.hour for x in times_yr])
        mask_day = hours_yr == UTC_TIMES[utc_time]
        
        var_data = conv_func(ds.variables[varname_in][mask_day,mask_levels,mask_lats,mask_lons]) 
        
        dates_yr = times_yr[mask_day]
        if utc_time == 24:
            dates_yr = np.array([x - A_DAY for x in dates_yr])
            
        ymd_yr = utld.get_ymd_array(dates_yr)
        
        mask_ymd = np.logical_and(ymd_yr >= days[YMD][0],ymd_yr <= days[YMD][-1])
        ymd_yr = ymd_yr[mask_ymd]
        var_data = var_data[mask_ymd,:,:,:]
        
        fnl_day_mask = np.in1d(days[YMD], ymd_yr, True)
        
        ds_out.variables[varname_out][fnl_day_mask,:,:,:] = var_data
        ds_out.sync()
        
        print yr

def create_nnr_subset_nolevel(path_nnr,fpath_out,yrs,days,varname_in,varname_out,utc_time,conv_func,suffix=""):
    
    ds_out = dbDataset(fpath_out,'w')
    
    ds = Dataset("".join([path_nnr,varname_in,suffix,".",str(yrs[0]),".nc"]))
    lons = ds.variables['lon'][:]
    lons[lons>180] = lons[lons>180]-360.0
    lats = ds.variables['lat'][:]
    
    mask_lons = np.logical_and(lons >= LON_LEFT, lons <= LON_RIGHT)
    mask_lats = np.logical_and(lats >= LAT_BOTTOM, lats <= LAT_TOP)
    
    ds_out.db_create_global_attributes("".join(["NCEP/NCAR Daily ",varname_out," ",str(utc_time),"Z Subset"]))
    ds_out.db_create_lonlat_dimvar(lons[mask_lons], lats[mask_lats])
    ds_out.db_create_time_dimvar(days)
    
    ds_out.createVariable(varname_out,'f4',('time','lat','lon'),
                          fill_value=netCDF4.default_fillvals['f4'],
                          chunksizes=(days.size,4,4))
    for yr in yrs:
        
        ds = Dataset("".join([path_nnr,varname_in,suffix,".",str(yr),".nc"]))
        
        start,end = ds.variables['time'][0],ds.variables['time'][-1]+6
        
        times_yr = num2date(np.arange(start,end,6),units=ds.variables['time'].units)
        hours_yr = np.array([x.hour for x in times_yr])
        mask_day = hours_yr == UTC_TIMES[utc_time]
        
        var_data = conv_func(ds.variables[varname_in][mask_day,mask_lats,mask_lons]) 
        
        dates_yr = times_yr[mask_day]
        if utc_time == 24:
            dates_yr = np.array([x - A_DAY for x in dates_yr])
            
        ymd_yr = utld.get_ymd_array(dates_yr)
        
        mask_ymd = np.logical_and(ymd_yr >= days[YMD][0],ymd_yr <= days[YMD][-1])
        ymd_yr = ymd_yr[mask_ymd]
        var_data = var_data[mask_ymd,:,:]
        
        fnl_day_mask = np.in1d(days[YMD], ymd_yr, True)
        
        ds_out.variables[varname_out][fnl_day_mask,:,:] = var_data
        ds_out.sync()
        
        print yr
 
def create_thickness_nnr_subset(path_nnr,fpath_out,yrs,days,level_up,level_low,utc_time):
    
    ds_out = dbDataset(fpath_out,'w')
    
    ds = Dataset("".join([path_nnr,"hgt.",str(yrs[0]),".nc"]))
    levels = ds.variables['level'][:]
    lons = ds.variables['lon'][:]
    lons[lons>180] = lons[lons>180]-360.0
    lats = ds.variables['lat'][:]
    
    idx_levelup = np.nonzero(levels == level_up)[0][0]
    idx_levellow = np.nonzero(levels == level_low)[0][0]
    
    mask_lons = np.logical_and(lons >= LON_LEFT, lons <= LON_RIGHT)
    mask_lats = np.logical_and(lats >= LAT_BOTTOM, lats <= LAT_TOP)
    
    ds_out.db_create_global_attributes("".join(["NCEP/NCAR Daily ",str(level_up),"-",str(level_low)," thickness ",str(utc_time),"Z Subset"]))
    ds_out.db_create_lonlat_dimvar(lons[mask_lons], lats[mask_lats])
    ds_out.db_create_time_dimvar(days)
    
    ds_out.createVariable('thick','f4',('time','lat','lon'),
                          fill_value=netCDF4.default_fillvals['f4'],
                          chunksizes=(days.size,4,4))
    for yr in yrs:
        
        ds = Dataset("".join([path_nnr,"hgt.",str(yr),".nc"]))
        
        start,end = ds.variables['time'][0],ds.variables['time'][-1]+6
        
        times_yr = num2date(np.arange(start,end,6),units=ds.variables['time'].units)
        hours_yr = np.array([x.hour for x in times_yr])
        mask_day = hours_yr == UTC_TIMES[utc_time]
        
        data_levelup = ds.variables['hgt'][mask_day,idx_levelup,mask_lats,mask_lons]
        data_levellow = ds.variables['hgt'][mask_day,idx_levellow,mask_lats,mask_lons]
        data_thick =  data_levelup -  data_levellow
        
        dates_yr = times_yr[mask_day]
        if utc_time == 24:
            dates_yr = np.array([x - A_DAY for x in dates_yr])
            
        ymd_yr = utld.get_ymd_array(dates_yr)
        
        mask_ymd = np.logical_and(ymd_yr >= days[YMD][0],ymd_yr <= days[YMD][-1])
        ymd_yr = ymd_yr[mask_ymd]
        data_thick = data_thick[mask_ymd,:,:]
        
        fnl_day_mask = np.in1d(days[YMD], ymd_yr, True)
        
        ds_out.variables['thick'][fnl_day_mask,:,:] = data_thick
        ds_out.sync()
        
        print yr

def create_inversion_strength_nnr_subset(path_nnr,fpath_out,yrs,days,level_up,utc_time):
    
    ds_out = dbDataset(fpath_out,'w')
    
    ds = Dataset("".join([path_nnr,"air.",str(yrs[0]),".nc"]))
    levels = ds.variables['level'][:]
    lons = ds.variables['lon'][:]
    lons[lons>180] = lons[lons>180]-360.0
    lats = ds.variables['lat'][:]
    
    idx_levelup = np.nonzero(levels == level_up)[0][0]
    idx_1000mb = np.nonzero(levels == 1000)[0][0]
    
    mask_lons = np.logical_and(lons >= LON_LEFT, lons <= LON_RIGHT)
    mask_lats = np.logical_and(lats >= LAT_BOTTOM, lats <= LAT_TOP)
    
    ds_out.db_create_global_attributes("".join(["NCEP/NCAR Daily ",str(level_up),"-surface inversion strength thickness ",str(utc_time),"Z Subset"]))
    ds_out.db_create_lonlat_dimvar(lons[mask_lons], lats[mask_lats])
    ds_out.db_create_time_dimvar(days)
    
    ds_out.createVariable('inv_str','f4',('time','lat','lon'),
                          fill_value=netCDF4.default_fillvals['f4'],
                          chunksizes=(days.size,4,4))
    for yr in yrs:
        
        ds = Dataset("".join([path_nnr,"air.",str(yr),".nc"]))
        #ds_pot = Dataset("".join([path_nnr,"pottmp.sig995.",str(yr),".nc"]))
        
        start,end = ds.variables['time'][0],ds.variables['time'][-1]+6
        
        times_yr = num2date(np.arange(start,end,6),units=ds.variables['time'].units)
        hours_yr = np.array([x.hour for x in times_yr])
        mask_day = hours_yr == UTC_TIMES[utc_time]
        
        tair_levelup = ds.variables['air'][mask_day,idx_levelup,mask_lats,mask_lons]
        tair_1000mb = ds.variables['air'][mask_day,idx_1000mb,mask_lats,mask_lons]
        #calculate potential temperature
        ptair_levelup = tair_levelup*((1000.0/float(level_up))**0.286)
        #ptair_surface = ds_pot.variables['pottmp'][mask_day,mask_lats,mask_lons]
        #inv_str =  ptair_levelup - ptair_surface
        inv_str =  ptair_levelup - tair_1000mb
        
        dates_yr = times_yr[mask_day]
        if utc_time == 24:
            dates_yr = np.array([x - A_DAY for x in dates_yr])
            
        ymd_yr = utld.get_ymd_array(dates_yr)
        
        mask_ymd = np.logical_and(ymd_yr >= days[YMD][0],ymd_yr <= days[YMD][-1])
        ymd_yr = ymd_yr[mask_ymd]
        inv_str = inv_str[mask_ymd,:,:]
        
        fnl_day_mask = np.in1d(days[YMD], ymd_yr, True)
        
        ds_out.variables['inv_str'][fnl_day_mask,:,:] = inv_str
        ds_out.sync()
        
        print yr 

if __name__ == '__main__':
    
    path_nnr = '/projects/daymet2/reanalysis_data/'
    yrs = np.arange(1948,2014)
    days = utld.get_days_metadata(datetime(1948,1,1), datetime(2013,1,31))
    
############################################################
#Temperature
############################################################
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_tair_24z.nc', 
#                      yrs, days,'air','tair', np.array([850]), 24, KtoC)
#    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_tair_12z.nc', 
#                      yrs, days,'air','tair', np.array([850]), 12, KtoC)
#    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_tair_18z.nc', 
#                      yrs, days,'air','tair', np.array([850]), 18, KtoC)

############################################################
#500-1000 Thickness
############################################################
#    create_thickness_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_thick_24z.nc', 
#                                yrs, days, 500, 1000, 24)
#    
#    create_thickness_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_thick_12z.nc', 
#                                yrs, days, 500, 1000, 12)
#    
#    create_thickness_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_thick_18z.nc', 
#                                yrs, days, 500, 1000, 18)

############################################################
#Height
############################################################

#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_hgt_24z.nc', 
#                      yrs, days,'hgt','hgt', np.array([500,700]), 24, no_conv)
#    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_hgt_12z.nc', 
#                      yrs, days,'hgt','hgt', np.array([500,700]), 12, no_conv)
#    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_hgt_18z.nc', 
#                      yrs, days,'hgt','hgt', np.array([500,700]), 18, no_conv)
    
############################################################
#RH
############################################################
#
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_rhum_24z.nc', 
#                      yrs, days,'rhum','rhum', np.array([850]), 24, no_conv)
#    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_rhum_12z.nc', 
#                      yrs, days,'rhum','rhum', np.array([850]), 12, no_conv)
#    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_rhum_18z.nc', 
#                      yrs, days,'rhum','rhum', np.array([850]), 18, no_conv)

############################################################
#SLP
############################################################

#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_slp_24z.nc', 
#                              yrs, days, 'slp', 'slp', 24, no_conv)
#    
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_slp_12z.nc', 
#                              yrsnc, days, 'slp', 'slp', 12, no_conv)
#    
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_slp_18z.nc', 
#                              yrs, days, 'slp', 'slp', 18, no_conv)
    
############################################################
#Inversion Strength
############################################################
    
#    create_inversion_strength_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_inv_str_24z.nc', 
#                                         yrs, days, 700, 24)
#    create_inversion_strength_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_inv_str_12z.nc', 
#                                         yrs, days, 700, 12)
#    create_inversion_strength_nnr_subset(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_inv_str_18z.nc', 
#                                         yrs, days, 700, 18)

############################################################
#V-Wind Surface
############################################################

#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_24z.nc', 
#                              yrs, days, 'vwnd', 'vwnd', 24, no_conv,".sig995")
#    
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_12z.nc', 
#                              yrs, days, 'vwnd', 'vwnd', 12, no_conv,".sig995")
#    
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_18z.nc', 
#                              yrs, days, 'vwnd', 'vwnd', 18, no_conv,".sig995")
    
############################################################
#U-Wind Surface
############################################################

#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_24z.nc', 
#                              yrs, days, 'uwnd', 'uwnd', 24, no_conv,".sig995")
#    
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_12z.nc', 
#                              yrs, days, 'uwnd', 'uwnd', 12, no_conv,".sig995")
#    
#    create_nnr_subset_nolevel(path_nnr, '/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_18z.nc', 
#                              yrs, days, 'uwnd', 'uwnd', 18, no_conv,".sig995")



############################################################
#V-Wind 850mb
############################################################

#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_24z.nc', 
#                      yrs, days,'vwnd','vwnd', np.array([850]), 24, no_conv)
#    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_12z.nc', 
#                      yrs, days,'vwnd','vwnd', np.array([850]), 12, no_conv)
#    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_vwnd_18z.nc', 
#                      yrs, days,'vwnd','vwnd', np.array([850]), 18, no_conv)
    
############################################################
#U-Wind 850mb
############################################################

    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_24z.nc', 
                      yrs, days,'uwnd','uwnd', np.array([850]), 24, no_conv)
    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_12z.nc', 
#                      yrs, days,'uwnd','uwnd', np.array([850]), 12, no_conv)
#    
#    create_nnr_subset(path_nnr,'/projects/daymet2/reanalysis_data/conus_subset/nnr_uwnd_18z.nc', 
#                      yrs, days,'uwnd','uwnd', np.array([850]), 18, no_conv)

    