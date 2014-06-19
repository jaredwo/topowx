'''
Classes and utilities for accessing weather station data
'''
import numpy as np
from twx.utils import get_days_metadata, get_days_metadata_dates
from netCDF4 import Dataset, num2date
import netCDF4

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

def _build_stn_struct(ds):
    
    varnames = ds.variables.keys()
    
    stn_var_data = []
    stn_var_dtype = []
    stn_var_name = []
    
    for var_name in varnames:
        
        if ds.variables[var_name].dimensions == ('stn_id',):
            
            var_data = ds.variables[var_name][:]
            var_dtype = var_data.dtype
            
            if np.ma.isMA(var_data):
                
                mask = var_data.mask
                var_data = np.require(var_data.data, np.float64)
                var_data[mask] = np.nan
                var_dtype = np.dtype(np.float64)
            
            elif var_data.dtype == np.object:
                
                var_data = var_data.astype(np.str)
                var_dtype = var_data.dtype
            
            stn_var_data.append(var_data)
            stn_var_dtype.append((str(var_name),var_dtype))
            stn_var_name.append(var_name)

    stns = np.empty(len(ds.dimensions['stn_id']), dtype=stn_var_dtype)
    
    for var_name,var_data in zip(stn_var_name,stn_var_data):
        stns[var_name] = var_data
    
    return stns

                   
class StationDataDb(object):
    '''
    A class for accessing stations and observations from
    a netCDF4 weather station database.
    '''
    
    def __init__(self, nc_path,startend_ymd=None,vcc_size=None,vcc_nelems=None,vcc_preemption=None,mode="r"):
        '''
        Parameters
        ----------
        nc_path : str
            File path to the netCDF4 dataset
        startend_ymd : tuple of 2 ints, optional
            A tuple of start/end ymds if want to load data
            for a specific time period only.
        vcc_size : int, optional
            The netCDF4 variable chunk cache size in bytes
        vcc_nelems : int, optional
            The netCDF4 number of chunk slots in the 
            raw data chunk cache hash table.
        vcc_preemption : int, optional
            The netCDF4 var chunk cache preemption value.
        mode : str, optional
            The dataset read mode (r or r+)
        '''
        
        self.nc_path = nc_path
        self.ds = Dataset(self.nc_path,mode=mode)
        
        var_time = self.ds.variables['time']
        start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
        self.days = get_days_metadata(start, end)
        self.stn_ids = self.ds.variables['stn_id'][:].astype(np.str)
        
        #don't do auto mask/scale on the main variables
        
        main_vars = []
        
        for a_varname in self.ds.variables.keys():
            
            if self.ds.variables[a_varname].dimensions == ('time', 'stn_id'):
                
                self.ds.variables[a_varname].set_auto_maskandscale(False)
                main_vars.append(self.ds.variables[a_varname])
                
        if vcc_size != None or vcc_nelems != None or vcc_preemption != None:
            
            chkc = list(self.var_tmin.get_var_chunk_cache())
            
            if vcc_size != None:
                chkc[0] = vcc_size
                
            if vcc_nelems != None:
                chkc[1] = vcc_nelems
                
            if vcc_preemption != None:
                chkc[2] = vcc_preemption
            
            for a_main_var in main_vars:
                
                a_main_var.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        
        self.stns = _build_stn_struct(self.ds)
        
        self.day_mask = None
        if startend_ymd != None:
            
            if startend_ymd[0] != None and startend_ymd[1] != None:
            
                self.__set_day_mask(startend_ymd[0], startend_ymd[1])
            
        self.stn_idxs = {}
        for x in np.arange(self.stn_ids.size):
            self.stn_idxs[self.stn_ids[x]] = x
    
    def __set_day_mask(self, start_ymd, end_ymd):
        
        self.day_mask = np.nonzero(np.logical_and(self.days[YMD] >= start_ymd, self.days[YMD] <= end_ymd))[0]
        self.days = self.days[self.day_mask]
           
    def add_stn_variable(self,varname,long_name,units,dtype):
        '''
        Add and initialize a station variable. If the variable
        already exists, it will be reset.
        
        Parameters
        ----------
        varname : str
            The name of the variable.
        long_name : str
            The long name of the variable.
        units : str
            The units of the variable.
        dtype : str
            The data type of the variable as a string.
            
        Returns
        -------
        newvar : netCDF4.Variable
            The new netCDF4 variable
        '''
        
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
        '''
        Load station observations for a specific variable
        
        Parameters
        ----------
        stn_ids : ndarray of str or str
            A numpy array of N station ids or a single station id
        var : str
            The name of the observation variable
        set_flagged_nan : bool
            If true, any QA-flagged observations will be set to nan
            
        Returns
        -------
        vals : ndarray
            The station observations of shape P*N where P is the
            number of days and N is the number of stations. If only
            1 station, returns a 1-D array.
        flags : ndarray
            The station observation flags of same shape as vals.
        '''
        
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
        '''
        Load Tmin and Tmax station observations.
        
        Parameters
        ----------
        stn_ids : ndarray of str or str
            A numpy array of N station ids.
        set_flagged_nan : bool
            If true, any QA-flagged observations will be set to nan
            
        Returns
        -------
        obs : ndarray
             A structured numpy array with field names: TMAX, TMIN,
             TMAX_FLAG, TMIN_FLAG. Each field is of shape P*N 
             where P is the number of days and N is the number of stations. 
        '''
        
        num_stns = stn_ids.size
        
        mask = np.nonzero(np.in1d(self.stn_ids, stn_ids, assume_unique=True))[0]
        
        obs = np.empty(self.days.size, dtype=[(TMAX, np.float32, num_stns), (TMIN, np.float32, num_stns),
                                              (TMAX_FLAG, "S1", num_stns), (TMIN_FLAG, "S1", num_stns)])
                
        if self.day_mask is not None:
            
            tmin = self.ds.variables['tmin'][self.day_mask, mask]
            tmax = self.ds.variables['tmax'][self.day_mask, mask]            
            flag_tmin = self.ds.variables['qflag_tmin'][self.day_mask, mask]
            flag_tmax = self.ds.variables['qflag_tmax'][self.day_mask, mask]
            
        else:
            
            tmin = self.ds.variables['tmin'][:, mask]
            tmax = self.ds.variables['tmax'][:, mask]            
            flag_tmin = self.ds.variables['qflag_tmin'][:, mask]
            flag_tmax = self.ds.variables['qflag_tmax'][:, mask]
        
        if set_flagged_nan:
            
            tmin[np.logical_or(tmin == NO_DATA, np.logical_not(flag_tmin == ""))] = np.nan
            tmax[np.logical_or(tmax == NO_DATA, np.logical_not(flag_tmax == ""))] = np.nan

        else:
            
            tmin[tmin == NO_DATA] = np.nan
            tmax[tmax == NO_DATA] = np.nan
        
        if num_stns == 1:
            
            tmin.shape = (tmin.shape[0],)
            tmax.shape = (tmax.shape[0],)
            flag_tmin.shape = (flag_tmin.shape[0],)
            flag_tmax.shape = (flag_tmax.shape[0],)

        obs[TMIN] = tmin
        tmin = None
        obs[TMAX] = tmax
        tmax = None
        obs[TMIN_FLAG] = flag_tmin
        flag_tmin = None
        obs[TMAX_FLAG] = flag_tmax
        flag_tmax = None

        return obs
        
class StationSerialDataDb(object):
    '''
    A class for accessing stations and observations from
    a serially complete netCDF4 weather station database.
    Each serially complete database only has one main variable.
    '''

    def __init__(self, nc_path, var_name,vcc_size=470560000,vcc_nelems=None,vcc_preemption=0):
        '''
        Parameters
        ----------
        nc_path : str
            File path to the netCDF4 dataset
        var_name : tuple of 2 ints, optional
            The name of main variable to be loaded.
        vcc_size : int, optional
            The netCDF4 variable chunk cache size in bytes
        vcc_nelems : int, optional
            The netCDF4 number of chunk slots in the 
            raw data chunk cache hash table.
        vcc_preemption : int, optional
            The netCDF4 var chunk cache preemption value.
        '''
                
        self.ds = Dataset(nc_path)
        var_time = self.ds.variables['time']        
        dates = num2date(var_time[:], var_time.units)
        self.days = get_days_metadata_dates(dates)  
        
        mthIdx = {}
        for mth in np.arange(1,13):
            mthIdx[mth] = np.nonzero(self.days[MONTH]==mth)[0]
        mthIdx[None] = np.arange(self.days.size)
        self.mthIdx = mthIdx
        
        self.stn_ids = np.array(self.ds.variables['stn_id'][:], dtype="<S16")
        self.var = self.ds.variables[var_name]
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
        
        self.stns = _build_stn_struct(self.ds)
        
        self.stn_idxs = {}
        for x in np.arange(self.stn_ids.size):
            self.stn_idxs[self.stn_ids[x]] = x
            
        self.last_stnids = np.array([])
        self.last_obs = None

        
    def load_obs(self, stn_ids,mth=None):
        '''
        Load station observations.
        
        Parameters
        ----------
        stn_ids : ndarray of str or str
            A numpy array of N station ids or a single station id
        mth : int, optional
            Only load observations for a specific month
        set_flagged_nan : bool
            If true, any QA-flagged observations will be set to nan
            
        Returns
        -------
        obs : ndarray
            The station observations of shape P*N where P is the
            number of days and N is the number of stations. If only
            1 station, returns a 1-D array.
        '''
        
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
