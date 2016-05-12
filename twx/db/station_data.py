'''
Classes and utilities for accessing weather station data.

Copyright 2014,2015, Jared Oyler.

This file is part of TopoWx.

TopoWx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TopoWx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.
'''

__all__ = ["TMAX", "TMIN", "TMIN_FLAG", "TMAX_FLAG", "LON", "LAT", "ELEV", "STN_ID",
           "STN_NAME", "STATE", "UTC_OFFSET", "StationDataDb", "StationSerialDataDb",
           "MEAN_TMAX", "MEAN_TMIN", "VAR_TMIN", "VAR_TMAX", "BAD", "CLIMDIV",
           "MASK", "TDI", "get_norm_varname", "get_optim_varname", "get_optim_anom_varname",
           "get_lst_varname", "get_krigparam_varname", "VARIO_NUG", "VARIO_PSILL", "VARIO_RNG",
           'LST']

import numpy as np
from twx.utils import get_days_metadata, get_days_metadata_dates, set_chunk_cache_params
from netCDF4 import Dataset, num2date, chartostring
import netCDF4
import xarray as xr

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

LON = "longitude"
LAT = "latitude"
ELEV = "elevation"
STN_ID = "station_id"
STN_NAME = "station_name"
STATE = "state"
NORM_OBS = "norm"
TDI = "tdi"  # Topographic Dissection
LST = "lst"  # Land Surface Temperature
VCF = 'vcf'  # vegetation continuous fields (% forest cover)
LC = 'lc'  # land cover
OPTIM_NNGH = 'optim_nnghs'
OPTIM_NNGH_ANOM = 'optim_nnghs_anom'
MASK = 'mask'  # interpolation mask
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

def get_mean_varname(varname, mth=None):
    
    if mth == None:
        return "mean_%s" % varname
    else:
        return "mean_%s%02d" % (varname, mth)
    
def get_variance_varname(varname, mth=None):
    
    if mth == None:
        return "vari_%s" % varname
    else:
        return "vari_%s%02d" % (varname, mth)

def get_lst_varname(mth):
    
    if mth == None:
        return LST
    else:
        return "lst%02d" % mth

def get_norm_varname(mth):
    
    if mth == None:
        return NORM_OBS
    else:
        return "norm%02d" % mth
    
def get_optim_varname(mth):
    
    if mth == None:
        return OPTIM_NNGH
    else:
        return "optim_nnghs%02d" % mth

def get_optim_anom_varname(mth):
    
    if mth == None:
        return OPTIM_NNGH_ANOM
    else:
        return "optim_nnghs_anom%02d" % mth

def get_krigparam_varname(mth, krigParam):
    
    if mth == None:
        return krigParam
    else:
        
        return "".join([krigParam, "%02d" % mth])
        
def _build_stn_struct(ds):
    
    varnames = ds.variables.keys()
    
    stn_var_data = []
    stn_var_dtype = []
    stn_var_name = []
    
    def is_chararray(a_var):
        
        # for now assume second dimension on char array
        # starts with string
        return (len(a_var.dimensions) == 2 and
                a_var.dimensions[0] == STN_ID and 
                a_var.dimensions[1].startswith('string'))
            
    for var_name in varnames:
        
        is_chara = is_chararray(ds.variables[var_name])
        
        if ds.variables[var_name].dimensions == (STN_ID,) or is_chara:
            
            prev_mask = ds.variables[var_name].mask
            prev_scale = ds.variables[var_name].scale
            ds.variables[var_name].set_auto_maskandscale(True)
            
            var_data = ds.variables[var_name][:]
            var_dtype = var_data.dtype
            
            if is_chara:
                var_data = chartostring(var_data)
                var_dtype = var_data.dtype
            
            elif np.ma.isMA(var_data):
                
                mask = var_data.mask
                var_data = np.require(var_data.data, np.float64)
                var_data[mask] = np.nan
                var_dtype = np.dtype(np.float64)
            
            elif var_data.dtype == np.object:
                
                var_data = var_data.astype(np.str)
                var_dtype = var_data.dtype
            
            stn_var_data.append(var_data)
            stn_var_dtype.append((str(var_name), var_dtype))
            stn_var_name.append(var_name)
            
            ds.variables[var_name].set_auto_mask(prev_mask)
            ds.variables[var_name].set_auto_scale(prev_scale)
            
    stns = np.empty(len(ds.dimensions[STN_ID]), dtype=stn_var_dtype)
    
    for var_name, var_data in zip(stn_var_name, stn_var_data):
        stns[var_name] = var_data
    
    return stns

def _parse_stn_ids(a_ds):
    
    var_stnid = a_ds.variables[STN_ID]
    
    if len(var_stnid.dimensions) == 2:
        stn_ids = chartostring(var_stnid[:])
    else:
        stn_ids = var_stnid[:].astype(np.str)
    
    return stn_ids
                   
class StationDataDb(object):
    '''
    A class for accessing stations and observations from
    a netCDF4 weather station database.
    '''
    
    def __init__(self, nc_path, startend_ymd=None, vcc_size=None,
                 vcc_nelems=None, vcc_preemption=None, mode="r"):
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
        
        store = xr.backends.NetCDF4DataStore(nc_path, mode=mode)
        self.xrds = xr.open_dataset(store)
        self.ds = store.ds #Dataset(self.nc_path, mode=mode)
        
        var_time = self.ds.variables['time']
        dates = num2date(var_time[:], var_time.units)  
        self.days = get_days_metadata_dates(dates)
                
        self.stn_ids = _parse_stn_ids(self.ds)
        
        # don't do auto mask/scale on the main variables
        main_vars = []
        
        for a_varname in self.ds.variables.keys():
            
            if self.ds.variables[a_varname].dimensions == ('time', 'station_id'):
                
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
                
                a_main_var.set_var_chunk_cache(chkc[0], chkc[1], chkc[2])
                
        else:
            
            # Set default cache size of 50MB
            for a_main_var in main_vars:
                set_chunk_cache_params(50000000, a_main_var)
        
        self.stns = _build_stn_struct(self.ds)
        
        self.day_mask = None
        if startend_ymd != None:
            
            if startend_ymd[0] != None and startend_ymd[1] != None:
            
                self.__set_day_mask(startend_ymd[0], startend_ymd[1])
            
        self.stn_idxs = {}
        for x in np.arange(self.stn_ids.size):
            self.stn_idxs[self.stn_ids[x]] = x
            
        #Add DataFrame version of stations
        self.stns_df = self.xrds[list(self.stns.dtype.names)].to_dataframe()
        self.stns_df['station_index'] = np.arange(len(self.stns_df))
        self.stns_df['station_id'] = self.stns_df.index
        
        self.stns_df.loc[:, self.stns_df.dtypes == object] = self.stns_df.loc[:, self.stns_df.dtypes == object].astype(np.str)
        self.stns_df.index = self.stns_df.index.astype(np.str)
        
    
    def __set_day_mask(self, start_ymd, end_ymd):
        
        self.day_mask = np.nonzero(np.logical_and(self.days[YMD] >= start_ymd, self.days[YMD] <= end_ymd))[0]
        self.days = self.days[self.day_mask]
           
    def add_stn_variable(self, varname, long_name, units, dtype, fill_value=None, reset=True):
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
        fill_value : int or float
            The fill or no data value for the variable.
            If None, the default netCDF4 fill value will be used
            
        Returns
        -------
        newvar : netCDF4.Variable
            The new netCDF4 variable
        '''
        
        fill_value = netCDF4.default_fillvals[dtype] if fill_value is None else fill_value
        
        if varname not in self.ds.variables.keys():
            
            newvar = self.ds.createVariable(varname, dtype, (STN_ID,),
                                            fill_value=fill_value)
            newvar.long_name = long_name
            newvar.units = units
            
        else:
            
            newvar = self.ds.variables[varname]
            
            if reset:
            
                newvar[:] = fill_value
        
        self.ds.sync()
        
        return newvar
    
    def add_obs_variable(self, varname, long_name, units, dtype,
                         fill_value=None, zlib=True, chunksizes=None,
                         reset=True):
        '''Add and initialize a 2D observation variable
        
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
        fill_value : int or float
            The fill or no data value for the variable.
            If None, the default netCDF4 fill value will be used
        zlib : boolean, optional
            Use zlib compression for the variable. Default: True
        chunksize: tuple of ints, optional
            Chunksize of the variable
        reset: boolean, optional
            Reset variable values if already exists. Default: True
            
        Returns
        -------
        newvar : netCDF4.Variable
            The new netCDF4 variable
        '''
        
        fill_value = netCDF4.default_fillvals[dtype] if fill_value is None else fill_value
        
        if varname not in self.ds.variables.keys():
            
            newvar = self.ds.createVariable(varname, dtype, ('time', STN_ID),
                                            fill_value=fill_value, zlib=zlib,
                                            chunksizes=chunksizes)
            newvar.long_name = long_name
            newvar.missing_value = fill_value
            newvar.units = units
                        
        else:
            
            newvar = self.ds.variables[varname]
            
            if reset:
            
                newvar[:] = fill_value
        
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
            mask = np.array([self.stn_idxs[stn_ids]], dtype=np.int)
        
        
        if self.day_mask is not None:
            
            vals = self.ds.variables[var][self.day_mask, mask]
            
            try:
                flags = self.ds.variables[''.join(['qflag_', var])][self.day_mask, mask]
            except KeyError:
                # This variable does not have quality flags
                flags = np.zeros(vals.shape, dtype=np.str)
        
        else:
            
            vals = self.ds.variables[var][:, mask]
            
            try:
                flags = self.ds.variables[''.join(['qflag_', var])][:, mask]
            except KeyError:
                # This variable does not have quality flags
                flags = np.zeros(vals.shape, dtype=np.str)
        
        if set_flagged_nan:
            vals[np.logical_or(vals == self.ds[var].missing_value,
                               np.logical_not(flags == ""))] = np.nan
        else:
            vals[vals == self.ds[var].missing_value] = np.nan

        if num_stns == 1:
            
            vals.shape = (vals.shape[0],)
            flags.shape = (flags.shape[0],)        
        
        return vals, flags
    
    def get_stn_mean(self, tair_var, x=None):
        
        if x is None:
            return self.stns["_".join(["mean", tair_var])]
        else:
            return self.stns["_".join(["mean", tair_var])][x]
    
    def get_stn_std(self, tair_var, x=None):
        
        if x is None:   
            return np.sqrt(self.stns["_".join(["var", tair_var])])
        else:
            return np.sqrt(self.stns["_".join(["var", tair_var])][x])
    
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
            
            tmin[np.logical_or(tmin == self.ds['tmin'].missing_value,
                               np.logical_not(flag_tmin == ""))] = np.nan
            tmax[np.logical_or(tmax == self.ds['tmax'].missing_value,
                               np.logical_not(flag_tmax == ""))] = np.nan

        else:
            
            tmin[tmin == self.ds['tmin'].missing_value] = np.nan
            tmax[tmax == self.ds['tmax'].missing_value] = np.nan
        
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

    def __init__(self, nc_path, var_name, vcc_size=470560000, vcc_nelems=None, vcc_preemption=0, mode="r"):
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
        mode : str, optional
            The dataset read mode (r or r+)
        '''
                
        self.ds = Dataset(nc_path, mode=mode)
        var_time = self.ds.variables['time']        
        dates = num2date(var_time[:], var_time.units)
        self.days = get_days_metadata_dates(dates)  
        
        mthIdx = {}
        for mth in np.arange(1, 13):
            mthIdx[mth] = np.nonzero(self.days[MONTH] == mth)[0]
        mthIdx[None] = np.arange(self.days.size)
        self.mth_idx = mthIdx
        
        self.stn_ids = _parse_stn_ids(self.ds)
        self.var = self.ds.variables[var_name]
        self.var.set_auto_maskandscale(False)  # no missing values, no need to mask
        self.var_name = var_name
        
        if vcc_size != None or vcc_nelems != None or vcc_preemption != None:
            
            chkc = list(self.var.get_var_chunk_cache())
            
            if vcc_size != None:
                chkc[0] = vcc_size
                
            if vcc_nelems != None:
                chkc[1] = vcc_nelems
                
            if vcc_preemption != None:
                chkc[2] = vcc_preemption
            
            self.var.set_var_chunk_cache(chkc[0], chkc[1], chkc[2])
            
        else:
            
            # Set default cache size of 50MB
            set_chunk_cache_params(50000000, self.var)
        
        self.stns = _build_stn_struct(self.ds)
        
        self.stn_idxs = {}
        for x in np.arange(self.stn_ids.size):
            self.stn_idxs[self.stn_ids[x]] = x
            
        self.last_stnids = np.array([])
        self.last_obs = None

        
    def load_obs(self, stn_ids, mth=None):
        '''
        Load station observations.
        
        Parameters
        ----------
        stn_ids : ndarray of str or str
            A numpy array of N station ids or a single station id
        mth : int, optional
            Only load observations for a specific month
            
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
                if np.sum(stn_ids == self.last_stnids) == num_stns:
                    useLastCache = True
            
            if useLastCache:
                obs = self.last_obs
            else:
                mask = np.nonzero(np.in1d(self.stn_ids, stn_ids, assume_unique=True))[0]
                obs = self.var[:, mask]
                self.last_stnids = stn_ids
                self.last_obs = obs
    
        else:
            
            num_stns = 1
            obs = self.var[:, self.stn_idxs[stn_ids]]
        
        if mth != None:            
            obs = np.take(obs, self.mth_idx[mth], axis=0)

        if num_stns == 1:
            obs.shape = (obs.shape[0],)
        
        return obs

    def add_stn_variable(self, varname, long_name, units, dtype, fill_value=None):
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
        fill_value : int or float
            The fill or no data value for the variable.
            If None, the default netCDF4 fill value will be used
            
        Returns
        -------
        newvar : netCDF4.Variable
            The new netCDF4 variable
        '''
        
        fill_value = netCDF4.default_fillvals[dtype] if fill_value is None else fill_value
        
        if varname not in self.ds.variables.keys():
            
            newvar = self.ds.createVariable(varname, dtype, (STN_ID,),
                                            fill_value=fill_value)
            newvar.long_name = long_name
            newvar.units = units
            
        else:
            
            newvar = self.ds.variables[varname]
            newvar[:] = fill_value
        
        self.ds.sync()
        
        return newvar

def create_stnobs_nc(fpath, start_date, end_date, ls_obsio):
    pass
    
    

