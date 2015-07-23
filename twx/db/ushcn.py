'''
Classes and utilities for working with USHCN data:
http://www.ncdc.noaa.gov/oa/climate/research/ushcn/

Copyright 2014, Jared Oyler.

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
import numpy as np
from twx.db.station_data import STN_ID,STATE,STN_NAME,LON,LAT,ELEV,YEAR,MONTH,DATE,YDAY,_build_stn_struct
import twx.utils.util_dates as utld
from netCDF4 import Dataset, date2num, num2date
from twx.utils.status_check import StatusCheck
import netCDF4
import os
from twx.db.create_db_all_stations import create_quick_db

DTYPE_STN_BASIC = [(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"), (LON, np.float64), (LAT, np.float64), (ELEV, np.float64)]
FNAME_USHCN_STNS = 'ushcn-v2.5-stations.txt'

USHCN_VARIABLES = [('tmax_raw', 'f4', netCDF4.default_fillvals['f4'], 'raw maximum air temperature', 'C'),
                   ('tmax_tob', 'f4', netCDF4.default_fillvals['f4'], 'time-of-observation corrected maximum air temperature', 'C'),
                   ('tmax_FLs.52i', 'f4', netCDF4.default_fillvals['f4'], 'final homogenized and infilled maximum air temperature', 'C'),
                   ('tmin_raw', 'f4', netCDF4.default_fillvals['f4'], 'raw minimum air temperature', 'C'),
                   ('tmin_tob', 'f4', netCDF4.default_fillvals['f4'], 'time-of-observation corrected minimum air temperature', 'C'),
                   ('tmin_FLs.52i', 'f4', netCDF4.default_fillvals['f4'], 'final homogenized and infilled minimum air temperature', 'C')]


class StationDataUSHCN(object):
    
    def __init__(self, nc_path):


        ds = Dataset(nc_path)

        var_time = ds.variables['time']
        start, end = num2date([var_time[0], var_time[-1]], var_time.units)
        self.mths = utld.get_mth_metadata(start.year, end.year)
          
        self.stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
        self.stns = _build_stn_struct(ds)          
        self.stn_idxs = {}
        for x in np.arange(self.stn_ids.size):
            self.stn_idxs[self.stn_ids[x]] = x
            
        self.ds = ds
        
        self.data = {}
        varNames = ['raw_tmin','raw_tmax','tob_tmin','tob_tmax','FLs.52i_tmin','FLs.52i_tmax']
        for aName in varNames:
            self.data[aName] = ds.variables[aName][:]
    
    def loadObs(self, stn_ids, var):
        
        
        if isinstance(stn_ids, np.ndarray):
            
            num_stns = stn_ids.size 
            mask = np.nonzero(np.in1d(self.stn_ids, stn_ids, assume_unique=True))[0]
        
        else:
            
            num_stns = 1
            mask = np.array([self.stn_idxs[stn_ids]],dtype=np.int)
        
                    
        vals = self.data[var][:, mask]
        
#        if np.ma.isMA(vals):
#            
#            valsData = vals.data
#            valsData[vals.mask] = np.nan
#            vals = valsData
            
        if num_stns == 1:
            vals.shape = (vals.shape[0],)

        return vals
    
class TairAggregate():
    
    MODIS_8DAY = np.array([1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97,
                            105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
                            209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305,
                            313, 321, 329, 337, 345, 353, 361])

    def __init__(self, days):
    
        uYrs = np.unique(days[YEAR])
        
        self.yrMthsMasks = []
        self.yrModis8dayMasks = []
        
        for aYr in uYrs:
            
            for aMth in np.arange(1,13):
                
                self.yrMthsMasks.append(np.nonzero(np.logical_and(days[YEAR]==aYr,days[MONTH]==aMth))[0])
                
            for x in np.arange(self.MODIS_8DAY.size):
                
                if self.MODIS_8DAY[x] == self.MODIS_8DAY[-1]:
                    self.yrModis8dayMasks.append(np.logical_and(days[YEAR]==aYr,days[YDAY]>=self.MODIS_8DAY[x]))
                else:
                    self.yrModis8dayMasks.append(np.logical_and(np.logical_and(days[YEAR]==aYr,days[YDAY]>=self.MODIS_8DAY[x]),days[YDAY]<self.MODIS_8DAY[x+1]))
        
        self.days = days
        self.yr_mths = utld.get_mth_metadata(uYrs[0],uYrs[-1])
        
        self.yrMasks = []
        
        for aYr in uYrs:
            
            self.yrMasks.append(np.nonzero(self.yr_mths[YEAR]==aYr)[0])
        
        self.u_yrs = uYrs
    
    def dailyToModis8Day(self,tair):
        
        if len(tair.shape) == 1:
            tair8day = np.ma.masked_array([np.ma.mean(tair[aMask],dtype=np.float) for aMask in self.yrModis8dayMasks])
        elif len(tair.shape) == 2:
            tair8day = np.ma.masked_array([np.ma.mean(tair[aMask,:],axis=0,dtype=np.float) for aMask in self.yrModis8dayMasks])
        elif len(tair.shape) == 3:
            tair8day = np.ma.masked_array([np.ma.mean(tair[aMask,:,:],axis=0,dtype=np.float) for aMask in self.yrModis8dayMasks])

        return tair8day
    
    def dailyToMthlyNorms(self,tair,startYear=1981,endYear=2010,yrMask=None):
        
        if yrMask == None:
            yrMask = np.nonzero(np.logical_and(self.yr_mths[YEAR] >= startYear,self.yr_mths[YEAR] <= endYear))[0]   
        tairMth = self.dailyToMthly(tair, -1)[0]
        
        if len(tairMth.shape) == 1:
            tairMth = np.take(tairMth, yrMask)
        elif len(tairMth.shape) == 2:
            tairMth = np.take(tairMth, yrMask,axis=0)
        elif len(tairMth.shape) == 3:
            tairMth = np.take(tairMth, yrMask,axis=0)
        
        mths = np.take(self.yr_mths[MONTH], yrMask)
        
        normShp = list(tairMth.shape)
        normShp[0] = 12
        normShp = tuple(normShp)
        
        tairMthNorms = np.ma.masked_array(np.zeros(normShp))
        
        for mth in np.arange(1,13):
            
            aMthMask = np.nonzero(mths==mth)[0]
            
            if len(tairMthNorms.shape) == 1:
                tairMthNorms[mth-1] = np.ma.mean(np.take(tairMth, aMthMask)) 
            elif len(tairMthNorms.shape) == 2:
                tairMthNorms[mth-1,:] = np.ma.mean(np.take(tairMth, aMthMask,axis=0),axis=0) 
            elif len(tairMthNorms.shape) == 3:
                tairMthNorms[mth-1,:,:] = np.ma.mean(np.take(tairMth, aMthMask,axis=0),axis=0)
    
        return tairMthNorms
    
    def mthlyToMthlyNorms(self,tairMth,startYear=1981,endYear=2010,yrMask=None):
        
        if yrMask == None:
            yrMask = np.nonzero(np.logical_and(self.yr_mths[YEAR] >= startYear,self.yr_mths[YEAR] <= endYear))[0]   
        
        if len(tairMth.shape) == 1:
            tairMth = np.take(tairMth, yrMask)
        elif len(tairMth.shape) == 2:
            tairMth = np.take(tairMth, yrMask,axis=0)
        elif len(tairMth.shape) == 3:
            tairMth = np.take(tairMth, yrMask,axis=0)
        
        mths = np.take(self.yr_mths[MONTH], yrMask)
        
        normShp = list(tairMth.shape)
        normShp[0] = 12
        normShp = tuple(normShp)
        
        tairMthNorms = np.ma.masked_array(np.zeros(normShp))
        
        for mth in np.arange(1,13):
            
            aMthMask = np.nonzero(mths==mth)[0]
            
            if len(tairMthNorms.shape) == 1:
                tairMthNorms[mth-1] = np.ma.mean(np.take(tairMth, aMthMask)) 
            elif len(tairMthNorms.shape) == 2:
                tairMthNorms[mth-1,:] = np.ma.mean(np.take(tairMth, aMthMask,axis=0),axis=0) 
            elif len(tairMthNorms.shape) == 3:
                tairMthNorms[mth-1,:,:] = np.ma.mean(np.take(tairMth, aMthMask,axis=0),axis=0)
    
        return tairMthNorms
    
    def dailyToMthly(self,tair,maxMiss=9):
        
        if len(tair.shape) == 1:
            tairM = np.ma.array([np.ma.mean(np.ma.take(tair, aMask),dtype=np.float) for aMask in self.yrMthsMasks])
        elif len(tair.shape) == 2:
            tairM = np.ma.array([np.ma.mean(np.ma.take(tair, aMask, axis=0),axis=0,dtype=np.float) for aMask in self.yrMthsMasks])
        elif len(tair.shape) == 3:
            tairM = np.ma.array([np.ma.mean(np.ma.take(tair, aMask, axis=0),axis=0,dtype=np.float) for aMask in self.yrMthsMasks])
        
        if maxMiss != -1:
            nMiss = np.array([np.sum(np.ma.take(tair.mask, aMask, axis=0),axis=0) for aMask in self.yrMthsMasks])     
            tairM[nMiss > maxMiss] = np.ma.masked
        else:
            nMiss = 0
        
#        if np.ma.isMA(tairM):
#            tairM[nMiss > maxMiss] = np.ma.masked
#        else:
#            tairM = np.ma.masked_array(tairM,mask=nMiss > maxMiss)
        return tairM,nMiss
    
    def dailyToAnn(self,tair):
        
        tairM,nMiss = self.dailyToMthly(tair,-1)
        tairA = self.mthlyToAnn(tairM)
        
        return tairA
    
    def mthlyToAnn(self,tairM,maxMiss=-1):
        
        if len(tairM.shape) == 1:
            tairA = np.ma.masked_array([np.ma.mean(tairM[aMask],dtype=np.float) for aMask in self.yrMasks])
        elif len(tairM.shape) == 2:
            tairA = np.ma.masked_array([np.ma.mean(tairM[aMask,:],axis=0,dtype=np.float) for aMask in self.yrMasks])
        elif len(tairM.shape) == 3:
            tairA = np.ma.masked_array([np.ma.mean(tairM[aMask,:,:],axis=0,dtype=np.float) for aMask in self.yrMasks])
            
        if maxMiss != -1 and np.ma.isMA(tairM):
            nMiss = np.array([np.sum(np.ma.take(tairM.mask, aMask, axis=0),axis=0) for aMask in self.yrMasks])     
            tairA[nMiss > maxMiss] = np.ma.masked
        else:
            nMiss = 0
            
        #tairA = np.ma.masked_array(tairA,np.isnan(tairA))
        
#        if np.sum(tairA.mask) > 0:
#            tairA[:] = np.nan
#            tairA = np.ma.masked_array(tairA,np.isnan(tairA))
        return tairA

def _parse_stns(fpath_ushcn_stns):
    
    afile = open(fpath_ushcn_stns)
    lines = afile.readlines()
    
    stns = np.empty(len(lines), dtype=DTYPE_STN_BASIC)
    stns[STN_ID] = np.array([aline[0:11] for aline in lines],np.str)
    stns[LAT] = np.array([np.float(aline[12:20]) for aline in lines])
    stns[LON] = np.array([np.float(aline[21:30]) for aline in lines])
    stns[ELEV] = np.array([np.float(aline[31:37]) for aline in lines])
    stns[STATE] = np.array([aline[38:40] for aline in lines],np.str)
    stns[STN_NAME] = np.array([aline[41:71].strip() for aline in lines],np.str)
    
    stns = stns[np.argsort(stns[STN_ID])]
    
    return stns

def _load_ushcn_stn_obs(stnid,path_obs_files,varname,min_yr,max_yr):
    
    #tair_var = tmax_raw,tmax_tob,tmax_FLs.52i,tmin_raw,tmin_tob,tmin_FLs.52i
    
    obs_varname,ushcn_type = varname.split('_')
    
    afile = open(os.path.join(path_obs_files, "%s.%s.%s"%(stnid,ushcn_type,obs_varname)))
    
    mth_vals = np.ones(12*((max_yr-min_yr)+1))*-9999.0
    
    for aline in afile.readlines():
        
        yr = np.int(aline[12:16])
        
        if yr >= min_yr and yr <= max_yr:
            
            idx = (yr-min_yr)*12
        
            yrmthvals = np.array([aline[17:17+5],aline[26:26+5],aline[35:35+5],aline[44:44+5],
                       aline[53:53+5],aline[62:62+5],aline[71:71+5],aline[80:80+5],
                       aline[89:89+5],aline[98:98+5],aline[107:107+5],aline[116:116+5]],dtype=np.float)
            
            mth_vals[idx:idx+12] = yrmthvals
    
    mth_vals[mth_vals==-9999] = np.nan
    mth_vals = mth_vals/100.0
    
    return mth_vals
    
def buildGhcnUShcnMask(stnids,fpathGhcnStns):
    
    #Build dict of HCN flags
    afile = open(fpathGhcnStns)
        
    lines = afile.readlines()
    
    ghcnIds = np.array(["".join(["GHCN_",aline[0:11].strip()]) for aline in lines],np.str)
    hcnMask = np.array([aline[76:79] == "HCN" for aline in lines],np.bool)
    hcnIds = ghcnIds[hcnMask]
    
    return np.array([stnid in hcnIds for stnid in stnids],dtype=np.bool)


def match_ghcn_to_ushcn(stns_ghcn,stns_ushcn,fpath_ghcn_stn_file):
    
    afile = open(fpath_ghcn_stn_file)
    lines = afile.readlines()
    
    ghcn_ids = np.array(["".join(["GHCN_",aline[0:11].strip()]) for aline in lines],np.str)
    mask_hcn = np.array([aline[76:79] == "HCN" for aline in lines],np.bool)
    ghcn_ids = np.sort(ghcn_ids[mask_hcn])
    
    stns_ghcn = stns_ghcn[np.in1d(stns_ghcn[STN_ID], ghcn_ids, True)]
    
    coop_ids_ushcn = np.array([a_id[-6:] for a_id in stns_ushcn[STN_ID]],dtype=np.str)
    rnd_lon_ushcn = np.round(stns_ushcn[LON],4)
    rnd_lat_ushcn = np.round(stns_ushcn[LAT],4)
    
    match_ushcn_ids = []
    
    for a_stn in stns_ghcn:
        
        #1. try to match by coop id
        match_ids = stns_ushcn[STN_ID][a_stn[STN_ID][-6:] == coop_ids_ushcn]
        
        if match_ids.size == 1:
            
            match_ushcn_ids.append(match_ids[0])
        
        else:

            #2. try to match by name and state
            match_ids = stns_ushcn[STN_ID][np.logical_and(a_stn[STN_NAME] == stns_ushcn[STN_NAME],
                                                          a_stn[STATE] == stns_ushcn[STATE])]
            
            if match_ids.size == 1:
            
                match_ushcn_ids.append(match_ids[0])
            
            else:
                
                #3. try to match by lon/lat
                match_ids = stns_ushcn[STN_ID][np.logical_and(np.round(a_stn[LON],4)==rnd_lon_ushcn,
                                                              np.round(a_stn[LAT],4)==rnd_lat_ushcn)]
                
                if match_ids.size == 1:
            
                    match_ushcn_ids.append(match_ids[0])
                
                else:
                    #Couldn't find an exact match
                    match_ushcn_ids.append('NONE')
        
    match_ushcn_ids = np.array(match_ushcn_ids)
    
    return stns_ghcn[STN_ID],match_ushcn_ids
    
    

def matchGhcnToUshcn(stnsG,stnsUS):
    
    matchUSIds = []
    
    coopIds = np.array([aId[-6:] for aId in stnsUS[STN_ID]],dtype=np.str)
    rndLon = np.round(stnsUS[LON],4)
    rndLat = np.round(stnsUS[LAT],4)
    
    for astn in stnsG:
        
        #1. try to match by coop id
        mId = stnsUS[STN_ID][astn[STN_ID][-6:] == coopIds]
        
        if mId.size == 1:
            
            matchUSIds.append(mId[0])
        
        else:

            #2. try to match by name and state
            mId = stnsUS[STN_ID][np.logical_and(astn[STN_NAME] == stnsUS[STN_NAME],astn[STATE] == stnsUS[STATE])]
            
            if mId.size == 1:
            
                matchUSIds.append(mId[0])
            
            else:
                
                #3. try to match by lon/lat
                mId = stnsUS[STN_ID][np.logical_and(np.round(astn[LON],4)==rndLon,np.round(astn[LAT],4)==rndLat)]
                
                if mId.size == 1:
            
                    matchUSIds.append(mId[0])
                
                else:
                
                    matchUSIds.append('NONE')
  
    return np.array(matchUSIds)

def create_ushcn_db(path_ushcn_data, fpath_out, min_yr, max_yr, ushcn_vars=USHCN_VARIABLES):
    
    fnames = np.array(os.listdir(path_ushcn_data))
    fnames = np.array([os.path.join(path_ushcn_data,a_name) for a_name in fnames])
    idx_dir = np.nonzero(np.array([os.path.isdir(a_name) for a_name in fnames]))[0]
    
    if idx_dir.size != 1:
        raise Exception('No directory or more than one directory in %s'%path_ushcn_data)
    
    path_obs_files = fnames[idx_dir[0]]
    
    stns = _parse_stns(os.path.join(path_ushcn_data,FNAME_USHCN_STNS))
    mths = utld.get_mth_metadata(min_yr,max_yr)
    
    create_quick_db(fpath_out, stns, mths, ushcn_vars)
    
    ushcn_variable_names = [a_var[0] for a_var in ushcn_vars]
    
    ds = Dataset(fpath_out,'r+')
    
    stchk = StatusCheck(stns.size,100)
    
    for x in np.arange(stns.size):
        
        for a_vname in ushcn_variable_names:
            
            obs = _load_ushcn_stn_obs(stns[STN_ID][x], path_obs_files, a_vname, min_yr, max_yr)
            
            obs[np.isnan(obs)] = netCDF4.default_fillvals['f4']
            
            ds.variables[a_vname][:,x] = obs
        
        stchk.increment()
        ds.sync()
            

def createUshcnDs(fpathStns,pathObs,fpathdsout,minYr=1948,maxYr=2012):
    
    stns = _parse_stns(fpathStns)
    stns = stns[np.argsort(stns[STN_ID])]
    
    mths = utld.get_mth_metadata(minYr,maxYr)
    minDate = mths[DATE][0]
    
    ds = Dataset(fpathdsout,'w')
    ds.createDimension('time',mths.size)
    ds.createDimension('stn_id',stns.size)
    
    times = ds.createVariable('time','f8',('time',),fill_value=False)
    times.units = "".join(["days since ",str(minDate.year),"-",str(minDate.month),"-",str(minDate.day)," 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(mths[DATE],times.units)
    
    ds.createVariable(STN_ID,'str',('stn_id',))[:] = np.array(stns[STN_ID],dtype=np.object)      
    ds.createVariable(STN_NAME,'str',('stn_id',))[:] = np.array(stns[STN_NAME],dtype=np.object)   
    ds.createVariable(STATE,'str',('stn_id',))[:] = np.array(stns[STATE],dtype=np.object)
    ds.createVariable(LAT,'f8',('stn_id',))[:] = stns[LAT]
    ds.createVariable(LON,'f8',('stn_id',))[:] = stns[LON]
    ds.createVariable(ELEV,'f8',('stn_id',))[:] = stns[ELEV]

    tairVarNames = ['raw_tmin','raw_tmax','tob_tmin','tob_tmax','FLs.52i_tmin','FLs.52i_tmax']
    
    stchk = StatusCheck(stns.size*len(tairVarNames),100)
    
    for aTairVar in tairVarNames:
        
        dsVar = ds.createVariable(aTairVar,'f4',('time','stn_id'),fill_value=netCDF4.default_fillvals['f4'])
        ds.sync()
        ushcnType,tairType = aTairVar.split('_')
        
        
        for x in np.arange(stns.size):
            
            obs = loadObs(stns[STN_ID][x], pathObs, tairType,ushcnType, minYr, maxYr)
            obs[np.isnan(obs)] = netCDF4.default_fillvals['f4']
            dsVar[:,x] = obs
            
            stchk.increment()
            
        ds.sync()
#    ds.variables[STN_ID][:] = np.array(stns[STN_ID],dtype=np.object)
#    ds.variables[STN_NAME][:] = np.array(stns[STN_NAME],dtype=np.object)
#    ds.variables[STATE][:] = np.array(stns[STATE],dtype=np.object)
#    ds.variables[LAT][:] = stns[LAT]
#    ds.variables[LON][:] = stns[LON]
#    ds.variables[ELEV][:] = stns[ELEV]

if __name__ == '__main__': 
    
#    createUshcnDs('/projects/daymet2/station_data/ushcn/ushcn-v2.5-stations.txt',
#                  '/projects/daymet2/station_data/ushcn/ushcn.v2.5.0.20130622/',
#                  '/projects/daymet2/station_data/ushcn/ushcn.nc')
    
#     createUshcnDs('/projects/daymet2/station_data/ushcn/ushcn-v2.5-stations.txt',
#               '/projects/daymet2/station_data/ushcn/ushcn.v2.5.0.20130622/',
#               '/projects/daymet2/station_data/ushcn/ushcn1895_2012.nc',minYr=1895, maxYr=2012)
    
    createUshcnDs('/projects/daymet2/station_data/ushcn/ushcn.v2.5.0.20140715/ushcn-v2.5-stations.txt',
                  '/projects/daymet2/station_data/ushcn/ushcn.v2.5.0.20140715/ushcn.v2.5.0.20140715/',
                  '/projects/daymet2/station_data/ushcn/ushcn.v2.5.0.20140715/ushcn.nc',minYr=1895, maxYr=2013)
    
    