'''
Classes and utilities for accessing weather station data

@author: jared.oyler
'''
import numpy as np
import twx.utils.util_dates as utld
from twx.utils.ncdf_raster import ncdf_raster
from netCDF4 import Dataset, num2date
import netCDF4
import twx.utils.util_geo as utlg
from datetime import datetime
from copy import copy

TMIN = "TMIN"
TMAX = "TMAX"
PRCP = "PRCP"
SWE = "SWE"
TMIN_FLAG = "TMIN_FLAG"
TMAX_FLAG = "TMAX_FLAG"
PRCP_FLAG = "PRCP_FLAG"
DATE = "DATE"
YMD = "YMD"
YEAR = "YEAR"
MONTH = "MONTH"
DAY = "DAY"
YDAY = "YDAY"

LON = "lon"
LAT = "lat"
ELEV = "elev"
STN_ID = "stn_id"
STN_NAME = "name"
STATE = "state"
MEAN_OBS = "obs_mean"
TDI = "tdi" #Topographic Dissection
LST = "lst" #Land Surface Temperature
VCF = 'vcf' #vegetation continuous fields (% forest cover)
NEON = "neon" #NEON Ecoregion
LC = 'lc' #land cover
OPTIM_NNGH = 'optim_nnghs'
OPTIM_NNGH_ANOM = 'optim_nnghs_anom'
MASK = 'mask' #interpolation mask
VARIO_NUG = 'vario_nug'
VARIO_PSILL = 'vario_psill'
VARIO_RNG = 'vario_rng'
BAD = 'bad'
CLIMDIV = 'climdiv'
NORM = 'norm'

MEAN_TMIN = 'mean_tmin'
MEAN_TMAX = 'mean_tmax'
VAR_TMIN = 'var_tmin'
VAR_TMAX = 'var_tmax'
UTC_OFFSET = 'utc_offset'

NO_DATA = -9999
DTYPE_STN_BASIC = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), (LON, np.float64), (LAT, np.float64), (ELEV, np.float64)]

DTYPE_STN_UTC = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),(UTC_OFFSET, np.float64)]
DTYPE_STN_UTC_MEAN_VAR_TMIN_TMAX = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),(UTC_OFFSET, np.float64),
                                    (MEAN_TMIN, np.float64),(MEAN_TMAX, np.float64),(VAR_TMIN, np.float64),(VAR_TMAX, np.float64)]
DTYPE_STN_DFLT = DTYPE_STN_UTC_MEAN_VAR_TMIN_TMAX
DTYPE_STN_MEAN_LST_TDI = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), 
                          (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
                          (TDI, np.float64),(LST, np.float64),(NEON, np.float64),
                          (MEAN_OBS, np.float64),(MASK, np.float64),(BAD, np.float64)]#(OPTIM_NNGH, np.float64),(OPTIM_NNGH_ANOM, np.float64),,
#(VARIO_NUG, np.float64),(VARIO_PSILL, np.float64),(VARIO_RNG, np.float64)]

DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), 
                          (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
                          (TDI, np.float64),(LST, np.float64),(NEON, np.float64),
                          (MEAN_OBS, np.float64),(MASK, np.float64),(BAD, np.float64),(OPTIM_NNGH, np.float64)]#,(OPTIM_NNGH_ANOM, np.float64)]

DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), 
                          (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
                          (TDI, np.float64),(LST, np.float64),(NEON, np.float64),
                          (MEAN_OBS, np.float64),(MASK, np.float64),(BAD, np.float64),(OPTIM_NNGH, np.float64),
                          (VARIO_NUG, np.float64),(VARIO_PSILL, np.float64),(VARIO_RNG, np.float64)]

DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), 
                          (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
                          (TDI, np.float64),(LST, np.float64),(NEON, np.float64),
                          (MEAN_OBS, np.float64),(MASK, np.float64),(BAD, np.float64),(OPTIM_NNGH, np.float64),
                          (VARIO_NUG, np.float64),(VARIO_PSILL, np.float64),(VARIO_RNG, np.float64),
                          (OPTIM_NNGH_ANOM, np.float64)]#,(CLIMDIV,np.float64)]

def get_lst_varname(mth):
    if mth == None:
        return LST
    else:
        return "lst%02d"%mth

def get_norm_varname(mth):
    if mth == None:
        return MEAN_OBS
    else:
        return "norm%02d"%mth
    
def get_optim_varname(mth):
    if mth == None:
        return OPTIM_NNGH
    else:
        return "optim_nnghs%02d"%mth

def get_optim_anom_varname(mth):
    if mth == None:
        return OPTIM_NNGH_ANOM
    else:
        return "optim_nnghs_anom%02d"%mth

def get_krigparam_varname(mth,krigParam):
    if mth == None:
        return krigParam
    else:
        
        return "".join([krigParam,"%02d"%mth])
        
        return "optim_nnghs%02d"%mth


DTYPE_NORMS = [(get_norm_varname(mth),np.float64) for mth in np.arange(1,13)]
DTYPE_OPTIM = [(get_optim_varname(mth),np.float64) for mth in np.arange(1,13)] 
DTYPE_LST = [(get_lst_varname(mth),np.float64) for mth in np.arange(1,13)]
DTYPE_VARIO_NUG = [(get_krigparam_varname(mth,VARIO_NUG),np.float64) for mth in np.arange(1,13)]  
DTYPE_VARIO_PSILL = [(get_krigparam_varname(mth,VARIO_PSILL),np.float64) for mth in np.arange(1,13)]
DTYPE_VARIO_RNG = [(get_krigparam_varname(mth,VARIO_RNG),np.float64) for mth in np.arange(1,13)]
DTYPE_ANOM_OPTIM = [(get_optim_anom_varname(mth),np.float64) for mth in np.arange(1,13)]    
 
DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM.extend(DTYPE_NORMS)
DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM.extend(DTYPE_OPTIM)
DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM.extend(DTYPE_LST)
DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM.extend(DTYPE_VARIO_NUG)
DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM.extend(DTYPE_VARIO_PSILL)
DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM.extend(DTYPE_VARIO_RNG)
DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM.extend(DTYPE_ANOM_OPTIM)

DTYPE_STN_DFLT = DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO_OPTIMNNGHANOM


DTYPE_INTERP = copy(DTYPE_STN_BASIC)
DTYPE_INTERP.extend([(MASK, np.float64),(BAD, np.float64),(TDI, np.float64),(NEON, np.float64)])
DTYPE_INTERP.extend(DTYPE_NORMS)
DTYPE_INTERP.extend(DTYPE_LST)

DTYPE_INTERP_OPTIM = copy(DTYPE_INTERP)
DTYPE_INTERP_OPTIM.extend(DTYPE_OPTIM)

DTYPE_INTERP_OPTIM_ALL = copy(DTYPE_INTERP_OPTIM)
DTYPE_INTERP_OPTIM_ALL.extend(DTYPE_ANOM_OPTIM)
DTYPE_INTERP_OPTIM_ALL.extend(DTYPE_VARIO_NUG)
DTYPE_INTERP_OPTIM_ALL.extend(DTYPE_VARIO_PSILL)
DTYPE_INTERP_OPTIM_ALL.extend(DTYPE_VARIO_RNG)

#DTYPE_STN_MEAN_LST_TDI_VCF_LC = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), 
#                          (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
#                          (TDI, np.float64),(LST, np.float64),(VCF, np.float64),(LC, np.float64),(NEON, np.float64),(MEAN_OBS, np.float64)]
#
#DTYPE_STN_MEAN_LSTMTHS_TDI = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), 
#                          (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
#                          (TDI, np.float64),(LST, np.float64),(NEON, np.float64),(MEAN_OBS, np.float64),
#                          (LST+"1", np.float64),(LST+"2", np.float64),(LST+"3", np.float64),
#                          (LST+"4", np.float64),(LST+"5", np.float64),(LST+"6", np.float64),
#                          (LST+"7", np.float64),(LST+"8", np.float64),(LST+"9", np.float64),
#                          (LST+"10", np.float64),(LST+"11", np.float64),(LST+"12", np.float64)]



def build_stn_struct(ds,stn_dtype=DTYPE_STN_DFLT):
    
    stn_ids = ds.variables['stn_id'][:].astype("<S16")
    
    stns = np.empty(stn_ids.size, dtype=stn_dtype)
    
    dtype_names = [x[0] for x in stn_dtype]
    
    for dname in dtype_names:
        stn_var = ds.variables[dname][:]
        if np.ma.isMA(stn_var):
            mask = stn_var.mask
            stn_var = np.require(stn_var.data, np.float64)
            stn_var[mask] = np.nan
        stns[dname] = stn_var

    return stns

def get_neon_rgns(neon_path,stns):
    
    neon_ds = ncdf_raster(neon_path, 'neon')
    
    rgns = np.zeros(stns.size)
    for x in np.arange(rgns.size):
        try:
            rgns[x] = neon_ds.getDataValue(stns[LON][x],stns[LAT][x])
        except:
            rgns[x] = np.nan
    
    return rgns
                   
class station_data_ncdb(object):
    '''
    A station_data class for accessing stations and observations from a netcdf weather station database.
    '''
    
    def __init__(self, nc_path,startend_ymd=None,stnDtype=DTYPE_STN_UTC_MEAN_VAR_TMIN_TMAX,vcc_size=None,vcc_nelems=None,vcc_preemption=None,mode="r"):
        '''
        DTYPE_STN_UTC_MEAN_VAR_TMIN_TMAX
        Constructor
        @param nc_path: path to the netCDF4 dataset
        @param startend_ymd: a tuple of start/end ymds if want to load data for a specific time period only
        @param vcc_size: the netCDF4 var chunk cache size in bytes
        @param vcc_nelems:  the netCDF4 number of chunk slots in the raw data chunk cache hash table
        @param vcc_preemption:  the netCDF4 var chunk cache preemption value
        '''

        ds = Dataset(nc_path,mode=mode)

        var_time = ds.variables['time']
        start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
        self.days = utld.get_days_metadata(start, end)
        self.stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
        
        self.var_tmin = ds.variables['tmin']
        self.var_tmax = ds.variables['tmax']
        #self.var_prcp = ds.variables['prcp']
        self.var_ftmin = ds.variables['qflag_tmin']
        self.var_ftmax = ds.variables['qflag_tmax']
        #self.var_fprcp = ds.variables['qflag_prcp']
        #self.var_swe = ds.variables['swe']
        
        #don't do auto mask scale
        self.var_tmin.set_auto_maskandscale(False)
        self.var_tmax.set_auto_maskandscale(False)
        #self.var_prcp.set_auto_maskandscale(False)
        self.var_ftmin.set_auto_maskandscale(False)
        self.var_ftmax.set_auto_maskandscale(False)
        #self.var_fprcp.set_auto_maskandscale(False)
        #self.var_swe.set_auto_maskandscale(False)
        
        if vcc_size != None or vcc_nelems != None or vcc_preemption != None:
            
            chkc = list(self.var_tmin.get_var_chunk_cache())
            
            if vcc_size != None:
                chkc[0] = vcc_size
                
            if vcc_nelems != None:
                chkc[1] = vcc_nelems
                
            if vcc_preemption != None:
                chkc[2] = vcc_preemption
            
            self.var_tmin.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
            self.var_tmax.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
            #self.var_prcp.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
            self.var_ftmin.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
            self.var_ftmax.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
            #self.var_fprcp.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
            #self.var_swe.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        
        self.ds = ds
        
        self.nc_path = nc_path
        
        #self.stns = self.load_stns()
        self.stns = build_stn_struct(ds,stnDtype)
        
        self.day_mask = None
        if startend_ymd != None:
            
            if startend_ymd[0] != None and startend_ymd[1] != None:
            
                self.set_day_mask(startend_ymd[0], startend_ymd[1])
            
        self.stn_idxs = {}
        for x in np.arange(self.stn_ids.size):
            self.stn_idxs[self.stn_ids[x]] = x
    
    def set_day_mask(self, start_ymd, end_ymd):
        
        self.day_mask = np.nonzero(np.logical_and(self.days[YMD] >= start_ymd, self.days[YMD] <= end_ymd))[0]
        self.days = self.days[self.day_mask]
       
    def load_stns(self):
                
        return build_stn_struct(self.ds,DTYPE_STN_UTC_MEAN_VAR_TMIN_TMAX)
    
    def add_stn_variable(self,varname,long_name,units,dtype):
        
        if varname not in self.ds.variables.keys():
            
            newvar = self.ds.createVariable(varname,dtype,('stn_id',),
                                            fill_value=netCDF4.default_fillvals[dtype])
            newvar.long_name = long_name
            newvar.units = units
            
        else:
            
            newvar = self.ds.variables[varname]
            newvar[:] = netCDF4.default_fillvals[dtype]
            self.ds.sync()
        
        return newvar
        
    
    def load_all_stn_obs_var(self, stn_ids, var, set_flagged_nan=True):
        
        
        if isinstance(stn_ids, np.ndarray):
            
            num_stns = stn_ids.size 
            mask = np.nonzero(np.in1d(self.stn_ids, stn_ids, assume_unique=True))[0]
        
        else:
            
            num_stns = 1
            mask = np.array([self.stn_idxs[stn_ids]],dtype=np.int)
        
        
        if self.day_mask is not None:
            
            vals = self.ds.variables[var][self.day_mask, mask]
            
            try:
                flags = self.ds.variables[''.join(['qflag_', var])][self.day_mask, mask]
            except KeyError:
                #This variable does not have quality flags
                flags = np.zeros(vals.shape,dtype=np.str)
        
        else:
            
            vals = self.ds.variables[var][:, mask]
            
            try:
                flags = self.ds.variables[''.join(['qflag_', var])][:, mask]
            except KeyError:
                #This variable does not have quality flags
                flags = np.zeros(vals.shape,dtype=np.str)
        
        if set_flagged_nan:
            vals[np.logical_or(vals == NO_DATA, np.logical_not(flags == ""))] = np.nan
        else:
            vals[vals == NO_DATA] = np.nan

        if num_stns == 1:
            
            vals.shape = (vals.shape[0],)
            flags.shape = (flags.shape[0],)

        return vals, flags
    
    def get_stn_mean(self,tair_var,x=None):
        
        if x == None:
            return self.stns["_".join(["mean",tair_var])]
        else:
            return self.stns["_".join(["mean",tair_var])][x]
    
    def get_stn_std(self,tair_var,x=None):
        
        if x == None:   
            return np.sqrt(self.stns["_".join(["var",tair_var])])
        else:
            return np.sqrt(self.stns["_".join(["var",tair_var])][x])
    
    def load_all_stn_obs(self, stn_ids, set_flagged_nan=True):
        
        num_stns = stn_ids.size
        
        mask = np.nonzero(np.in1d(self.stn_ids, stn_ids, assume_unique=True))[0]
        
        obs = np.empty(self.days.size, dtype=[(TMAX, np.float32, num_stns), (TMIN, np.float32, num_stns),
                                              (TMAX_FLAG, "S1", num_stns), (TMIN_FLAG, "S1", num_stns)])#,
        
#        obs = np.empty(self.days.size, dtype=[(TMAX, np.float32, num_stns), (TMIN, np.float32, num_stns), (PRCP, np.float32, num_stns),
#                                     (TMAX_FLAG, "S1", num_stns), (TMIN_FLAG, "S1", num_stns), (PRCP_FLAG, "S1", num_stns)])#,
        #(SWE, np.float64, num_stns)])
        
        if self.day_mask is not None:
            
            tmin = self.var_tmin[self.day_mask, mask]
            tmax = self.var_tmax[self.day_mask, mask]
            #prcp = self.var_prcp[self.day_mask, mask]
            #swe = self.var_swe[self.day_mask,mask]
            
            flag_tmin = self.var_ftmin[self.day_mask, mask]
            flag_tmax = self.var_ftmax[self.day_mask, mask]
            #flag_prcp = self.var_fprcp[self.day_mask, mask]
            
        else:
            
            tmin = self.var_tmin[:, mask]
            tmax = self.var_tmax[:, mask]
            #prcp = self.var_prcp[:, mask]
            #swe = self.var_swe[:, mask]
            
            flag_tmin = self.var_ftmin[:, mask]
            flag_tmax = self.var_ftmax[:, mask]
            #flag_prcp = self.var_fprcp[:, mask]
        
        if set_flagged_nan:
            tmin[np.logical_or(tmin == NO_DATA, np.logical_not(flag_tmin == ""))] = np.nan
            tmax[np.logical_or(tmax == NO_DATA, np.logical_not(flag_tmax == ""))] = np.nan
            #prcp[np.logical_or(prcp == NO_DATA, np.logical_not(flag_prcp == ""))] = np.nan
            #swe[swe == NO_DATA] = np.nan
        else:
            tmin[tmin == NO_DATA] = np.nan
            tmax[tmax == NO_DATA] = np.nan
            #prcp[prcp == NO_DATA] = np.nan
            #swe[swe == NO_DATA] = np.nan
        
        if num_stns == 1:
            
            tmin.shape = (tmin.shape[0],)
            tmax.shape = (tmax.shape[0],)
            #prcp.shape = (prcp.shape[0],)
            #swe.shape = (swe.shape[0],)
            flag_tmin.shape = (flag_tmin.shape[0],)
            flag_tmax.shape = (flag_tmax.shape[0],)
            #flag_prcp.shape = (flag_prcp.shape[0],)

        obs[TMIN] = tmin
        tmin = None
        obs[TMAX] = tmax
        tmax = None
        #obs[PRCP] = prcp
        #prcp = None
        #obs[SWE] = swe
        #swe = None
        obs[TMIN_FLAG] = flag_tmin
        flag_tmin = None
        obs[TMAX_FLAG] = flag_tmax
        flag_tmax = None
        #obs[PRCP_FLAG] = flag_prcp
        #flag_prcp = None 

        return obs

class station_norms(object):
    
    def __init__(self,path_dsnorms):
        
        ds_norms = Dataset(path_dsnorms)
        norms = ds_norms.variables['norm'][:]
        
        norms_tmin = norms[0,:]
        norms_tmax = norms[1,:]
        
        norms_tmin = np.ma.filled(norms_tmin, np.nan)
        norms_tmax = np.ma.filled(norms_tmax, np.nan)
        
        ds_norms.close()
        
    
    

class station_data_combine(station_data_ncdb):
    '''
    A station_data class for accessing stations from both a netcdf database and also stations loaded from a separate
    db.all_create_db.insert object. This class has the same interface as station_data_ncdb so it hides the  separate 
    loading from the insert object
    '''
    
    def __init__(self, nc_path,ainsert,startend_ymd=None,vcc_size=None,vcc_nelems=None,vcc_preemption=None):
        '''
        Constructor
        @param nc_path: path to the netCDF4 dataset
        @param nc_path: a db.all_create_db.insert object
        @param startend_ymd: a tuple of start/end ymds if want to load data for a specific time period only
        @param vcc_size: the netCDF4 var chunk cache size in bytes
        @param vcc_nelems:  the netCDF4 number of chunk slots in the raw data chunk cache hash table
        @param vcc_preemption:  the netCDF4 var chunk cache preemption value
        '''
        
        station_data_ncdb.__init__(self, nc_path, startend_ymd, vcc_size, vcc_nelems, vcc_preemption)
  
        self.ainsert = ainsert
        
        self.combine_stns()
        self.preload_insert_obs()
        
        
    def load_all_stn_obs_var(self, stn_ids, var, set_flagged_nan=True):
        
        if isinstance(stn_ids, np.ndarray):
            
            mask_ids_insert = np.in1d(stn_ids,self.insert_ids,True)
            mask_ids_db = np.logical_not(mask_ids_insert)
            
            if np.sum(mask_ids_insert) == stn_ids.size:
                
                mask_stns = np.nonzero(np.in1d(self.insert_ids, stn_ids, assume_unique=True))[0]
                vals = self.iobs[var][:,mask_stns]
                flags = self.iobs[''.join(['qflag_', var])][:,mask_stns]
                
                if set_flagged_nan:
                    vals[np.logical_not(flags == "")] = np.nan
                
                if stn_ids.size == 1:
                    vals = vals.ravel()
                    flags = flags.ravel()
                
                return vals,flags
            
            elif np.sum(mask_ids_db) == stn_ids.size:
                
                return station_data_ncdb.load_all_stn_obs_var(self, stn_ids, var, set_flagged_nan)
            
            else:
                
                vals_db,flags_db = station_data_ncdb.load_all_stn_obs_var(self, stn_ids[mask_ids_db], var, set_flagged_nan)
                
                if len(vals_db.shape) == 1:
                    vals_db.shape = (vals_db.shape[0],1)
                    flags_db.shape = (flags_db.shape[0],1)
                    
                mask_stns = np.nonzero(np.in1d(self.insert_ids, stn_ids[mask_ids_insert], assume_unique=True))[0]
                
                vals_insert = self.iobs[var][:,mask_stns]
                flags_insert = self.iobs[''.join(['qflag_', var])][:,mask_stns]
                
                if len(vals_insert.shape) == 1:
                    vals_insert.shape = (vals_insert.shape[0],1)
                    flags_insert.shape = (flags_insert.shape[0],1)
            
                vals_all = np.zeros((vals_insert.shape[0],stn_ids.size))
                flags_all = np.zeros((vals_insert.shape[0],stn_ids.size),dtype="<S1")
                
                vals_all[:,mask_ids_db] = vals_db
                flags_all[:,mask_ids_db] = flags_db
                
                vals_all[:,mask_ids_insert] = vals_insert
                flags_all[:,mask_ids_insert] = flags_insert
                
                return vals_all,flags_all
        
        else:
            
            if stn_ids in self.insert_ids:
                
                idx = np.nonzero(self.insert_ids==stn_ids)[0][0]
                vals = self.iobs[var][:,idx]
                flags = self.iobs[''.join(['qflag_', var])][:,idx]
                
                if set_flagged_nan:
                    vals[np.logical_not(flags == "")] = np.nan
                
                return vals,flags
                
            else:
                
                return station_data_ncdb.load_all_stn_obs_var(self,stn_ids, var, set_flagged_nan)
    
    def load_all_stn_obs(self, stn_ids, set_flagged_nan=True):

        mask_ids_insert = np.in1d(stn_ids,self.insert_ids,True)
        mask_ids_db = np.logical_not(mask_ids_insert)
        num_stns = stn_ids.size
        
        if np.sum(mask_ids_insert) == stn_ids.size:
            
            mask_stns = np.nonzero(np.in1d(self.insert_ids, stn_ids, assume_unique=True))[0]
            
            obs = np.empty(self.days.size, dtype=[(TMAX, np.float64, num_stns), (TMIN, np.float64, num_stns), (PRCP, np.float64, num_stns),
                                         (TMAX_FLAG, "S1", num_stns), (TMIN_FLAG, "S1", num_stns), (PRCP_FLAG, "S1", num_stns),
                                         (SWE, np.float64, num_stns)])
            
            if stn_ids.size == 1:
                obs[TMIN] = self.iobs['tmin'][:,mask_stns].ravel()
                obs[TMAX] = self.iobs['tmax'][:,mask_stns].ravel()
                obs[PRCP] = self.iobs['prcp'][:,mask_stns].ravel()
                obs[TMIN_FLAG] = self.iobs['qflag_tmin'][:,mask_stns].ravel()
                obs[TMAX_FLAG] = self.iobs['qflag_tmax'][:,mask_stns].ravel()
                obs[PRCP_FLAG] = self.iobs['qflag_prcp'][:,mask_stns].ravel()
                obs[SWE] = self.iobs['swe'][:,mask_stns].ravel()
            else:
                obs[TMIN] = self.iobs['tmin'][:,mask_stns]
                obs[TMAX] = self.iobs['tmax'][:,mask_stns]
                obs[PRCP] = self.iobs['prcp'][:,mask_stns]
                obs[TMIN_FLAG] = self.iobs['qflag_tmin'][:,mask_stns]
                obs[TMAX_FLAG] = self.iobs['qflag_tmax'][:,mask_stns]
                obs[PRCP_FLAG] = self.iobs['qflag_prcp'][:,mask_stns]
                obs[SWE] = self.iobs['swe'][:,mask_stns]
            
            if set_flagged_nan:
                
                obs[TMIN][obs[TMIN_FLAG] != ""] = np.nan
                obs[TMAX][obs[TMAX_FLAG] != ""] = np.nan
                obs[PRCP][obs[PRCP_FLAG] != ""] = np.nan
            
            return obs
        
        elif np.sum(mask_ids_db) == stn_ids.size:
            
            return station_data_ncdb.load_all_stn_obs(self, stn_ids, set_flagged_nan)
        
        else:
            
            obs_db = station_data_ncdb.load_all_stn_obs(self, stn_ids[mask_ids_db], set_flagged_nan)
            
            mask_stns = np.nonzero(np.in1d(self.insert_ids, stn_ids[mask_ids_insert], assume_unique=True))[0]
            
            obs = np.empty(self.days.size, dtype=[(TMAX, np.float64, num_stns), (TMIN, np.float64, num_stns), (PRCP, np.float64, num_stns),
                                         (TMAX_FLAG, "S1", num_stns), (TMIN_FLAG, "S1", num_stns), (PRCP_FLAG, "S1", num_stns),
                                         (SWE, np.float64, num_stns)])
            
            if np.sum(mask_ids_db) == 1:
            
                obs[TMIN][:,mask_ids_db] = np.reshape(obs_db[TMIN],(obs_db[TMIN].shape[0],1))
                obs[TMAX][:,mask_ids_db] = np.reshape(obs_db[TMAX],(obs_db[TMAX].shape[0],1))
                obs[PRCP][:,mask_ids_db] = np.reshape(obs_db[PRCP],(obs_db[PRCP].shape[0],1))
                obs[TMIN_FLAG][:,mask_ids_db] = np.reshape(obs_db[TMIN_FLAG],(obs_db[TMIN_FLAG].shape[0],1))
                obs[TMAX_FLAG][:,mask_ids_db] = np.reshape(obs_db[TMAX_FLAG],(obs_db[TMAX_FLAG].shape[0],1))
                obs[PRCP_FLAG][:,mask_ids_db] = np.reshape(obs_db[PRCP_FLAG],(obs_db[PRCP_FLAG].shape[0],1))
                obs[SWE][:,mask_ids_db] = np.reshape(obs_db[SWE],(obs_db[SWE].shape[0],1))
            
            else:
                
                obs[TMIN][:,mask_ids_db] = obs_db[TMIN]
                obs[TMAX][:,mask_ids_db] = obs_db[TMAX]
                obs[PRCP][:,mask_ids_db] = obs_db[PRCP]
                obs[TMIN_FLAG][:,mask_ids_db] = obs_db[TMIN_FLAG]
                obs[TMAX_FLAG][:,mask_ids_db] = obs_db[TMAX_FLAG]
                obs[PRCP_FLAG][:,mask_ids_db] = obs_db[PRCP_FLAG]
                obs[SWE][:,mask_ids_db] = obs_db[SWE]
                

            obs[TMIN][:,mask_ids_insert] = self.iobs['tmin'][:,mask_stns]
            obs[TMAX][:,mask_ids_insert] = self.iobs['tmax'][:,mask_stns]
            obs[PRCP][:,mask_ids_insert] = self.iobs['prcp'][:,mask_stns]
            obs[TMIN_FLAG][:,mask_ids_insert] = self.iobs['qflag_tmin'][:,mask_stns]
            obs[TMAX_FLAG][:,mask_ids_insert] = self.iobs['qflag_tmax'][:,mask_stns]
            obs[PRCP_FLAG][:,mask_ids_insert] = self.iobs['qflag_prcp'][:,mask_stns]
            obs[SWE][:,mask_ids_insert] = self.iobs['swe'][:,mask_stns]
            
            if set_flagged_nan:
                obs[TMIN][obs[TMIN_FLAG] != ""] = np.nan
                obs[TMAX][obs[TMAX_FLAG] != ""] = np.nan
                obs[PRCP][obs[PRCP_FLAG] != ""] = np.nan
            
            return obs
    
    def preload_insert_obs(self):
        
        iobs = np.empty(self.days.size, dtype=[('tmin', np.float64, self.insert_ids.size), ('tmax', np.float64, self.insert_ids.size), ('prcp', np.float64, self.insert_ids.size),
                                     ('qflag_tmax', "S1", self.insert_ids.size), ('qflag_tmin', "S1", self.insert_ids.size), ('qflag_prcp', "S1", self.insert_ids.size),
                                     ('swe', np.float64, self.insert_ids.size)])
        
        for x in np.arange(self.insert_ids.size):
            
            a_iobs = self.ainsert.parse_stn_obs(self.insert_ids[x])
            
            mask_ymd = np.in1d(self.days[YMD], a_iobs['ymd'], assume_unique=True)
            mask_not_ymd = np.logical_not(mask_ymd)
            
            iobs['tmin'][mask_ymd,x] =  a_iobs['tmin']
            iobs['tmin'][mask_not_ymd,x] =  np.nan
            
            iobs['tmax'][mask_ymd,x] =  a_iobs['tmax']
            iobs['tmax'][mask_not_ymd,x] =  np.nan
            
            iobs['prcp'][mask_ymd,x] =  a_iobs['prcp']
            iobs['prcp'][mask_not_ymd,x] =  np.nan
            
            iobs['swe'][mask_ymd,x] =  a_iobs['swe']
            iobs['swe'][mask_not_ymd,x] =  np.nan
            
            iobs['qflag_tmin'][mask_ymd,x] =  a_iobs['qflag_tmin']
            iobs['qflag_tmin'][mask_not_ymd,x] =  ''
            
            iobs['qflag_tmax'][mask_ymd,x] =  a_iobs['qflag_tmax']
            iobs['qflag_tmax'][mask_not_ymd,x] =  ''
            
            iobs['qflag_prcp'][mask_ymd,x] =  a_iobs['qflag_prcp']
            iobs['qflag_prcp'][mask_not_ymd,x] =  ''
            
        iobs['tmin'][iobs['tmin']==NO_DATA] = np.nan
        iobs['tmax'][iobs['tmax']==NO_DATA] = np.nan    
        iobs['prcp'][iobs['prcp']==NO_DATA] = np.nan
        iobs['swe'][iobs['swe']==NO_DATA] = np.nan
        
        self.iobs = iobs 
    
    def combine_stns(self):
        
        #(stn_id,lat,lon,elev,state,name)
        stns_insert = self.ainsert.get_stns()
        ids_insert = np.array([x[0] for x in stns_insert], dtype="<S16")
        s_idxs = np.argsort(ids_insert)
        ids_insert = ids_insert[s_idxs]
        lat_insert = np.array([x[1] for x in stns_insert])[s_idxs]
        lon_insert = np.array([x[2] for x in stns_insert])[s_idxs]
        elev_insert = np.array([x[3] for x in stns_insert])[s_idxs]
        st_insert = np.array([x[4] for x in stns_insert],dtype="<S2")[s_idxs]
        name_insert = np.array([x[5] for x in stns_insert],dtype="<S30")[s_idxs]
        
        all_ids = np.concatenate([self.stn_ids,ids_insert])
        all_ids = np.sort(all_ids)
        
        mask_dbstns = np.in1d(all_ids, self.stn_ids, assume_unique=True)
        mask_istns = np.in1d(all_ids, ids_insert, assume_unique=True)
        
        all_stns = np.empty(self.stn_ids.size+ids_insert.size, dtype=[(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), (LON, np.float64), (LAT, np.float64), (ELEV, np.float64)])
        
        all_stns[STN_ID][mask_dbstns] = self.stn_ids
        all_stns[STN_ID][mask_istns] = ids_insert
        
        all_stns[STATE][mask_dbstns] = self.stns[STATE]
        all_stns[STATE][mask_istns] = st_insert
        
        all_stns[STN_NAME][mask_dbstns] = self.stns[STN_NAME]
        all_stns[STN_NAME][mask_istns] = name_insert
        
        all_stns[LON][mask_dbstns] = self.stns[LON]
        all_stns[LON][mask_istns] = lon_insert
        
        all_stns[LAT][mask_dbstns] = self.stns[LAT]
        all_stns[LAT][mask_istns] = lat_insert
        
        all_stns[ELEV][mask_dbstns] = self.stns[ELEV]
        all_stns[ELEV][mask_istns] = elev_insert
        
        self.stn_ids = all_ids
        self.stns = all_stns
        self.insert_ids = ids_insert

class station_data_infill(object):
    '''
    A station_data class for accessing stations and observations from a single variable infilled netcdf weather station database.
    '''
    #235280000 bytes
    def __init__(self, nc_path, var_name,vcc_size=470560000,vcc_nelems=None,vcc_preemption=0,stn_dtype=DTYPE_INTERP):
        '''
        Constructor
        
        @param nc_path: path to the netCDF4 dataset
        @param var_name: the variable name in the netCDF4 dataset
        @param vcc_size: the netCDF4 var chunk cache size in bytes
        @param vcc_nelems:  the netCDF4 number of chunk slots in the raw data chunk cache hash table
        @param vcc_preemption:  the netCDF4 var chunk cache preemption value
        '''
        ds = Dataset(nc_path)
        var_time = ds.variables['time']
        
#        start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
#        self.days = utld.get_days_metadata(start, end)
        
        dates = num2date(var_time[:], var_time.units)
        self.days = utld.get_days_metadata_dates(dates)  
        
        mthIdx = {}
        for mth in np.arange(1,13):
            mthIdx[mth] = np.nonzero(self.days[MONTH]==mth)[0]
        mthIdx[None] = np.arange(self.days.size)
        self.mthIdx = mthIdx
        
        self.stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
        self.var = ds.variables[var_name]
        self.var.set_auto_maskandscale(False) #no missing values, no need to mask
        
        if vcc_size != None or vcc_nelems != None or vcc_preemption != None:
            
            chkc = list(self.var.get_var_chunk_cache())
            
            if vcc_size != None:
                chkc[0] = vcc_size
                
            if vcc_nelems != None:
                chkc[1] = vcc_nelems
                
            if vcc_preemption != None:
                chkc[2] = vcc_preemption
            
            self.var.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        
        self.ds = ds
        
        self.stns = build_stn_struct(ds, stn_dtype)
        
        self.stn_idxs = {}
        for x in np.arange(self.stn_ids.size):
            self.stn_idxs[self.stn_ids[x]] = x
            
        self.last_stnids = np.array([])
        self.last_obs = None
               
    def load_obs_idxs(self,stn_idxs):
        obs = self.var[:,stn_idxs]
        return obs
    
    def load_obs(self, stn_ids,mth=None):
        
        if isinstance(stn_ids, np.ndarray):
        
            num_stns = stn_ids.size
            
            useLastCache = False
            if num_stns == self.last_stnids.size:
                if np.sum(stn_ids==self.last_stnids)==num_stns:
                    useLastCache = True
            
            if useLastCache:
                obs = self.last_obs
            else:
                mask = np.nonzero(np.in1d(self.stn_ids, stn_ids, assume_unique=True))[0]
                obs = self.var[:,mask]
                self.last_stnids = stn_ids
                self.last_obs = obs
    
        else:
            
            num_stns = 1
            obs = self.var[:,self.stn_idxs[stn_ids]]
        
        if mth != None:            
            obs = np.take(obs, self.mthIdx[mth], axis=0)

        if num_stns == 1:
            obs.shape = (obs.shape[0],)
        
        return obs
