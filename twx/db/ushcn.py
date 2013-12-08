'''
Created on Jun 21, 2013

@author: jared.oyler
'''
import numpy as np
from db.station_data import STN_ID,STATE,STN_NAME,LON,LAT,ELEV,DTYPE_STN_BASIC,YEAR,MONTH,DATE,YDAY,build_stn_struct
import utils.util_dates as utld
from netCDF4 import Dataset, date2num, num2date
from utils.status_check import status_check
import netCDF4
from idlelib.PyShell import ModifiedColorDelegator


class StationDataUSHCN(object):
    
    def __init__(self, nc_path):


        ds = Dataset(nc_path)

        var_time = ds.variables['time']
        start, end = num2date([var_time[0], var_time[-1]], var_time.units)
        self.mths = utld.get_mth_metadata(start.year, end.year)
          
        self.stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
        self.stns = build_stn_struct(ds,DTYPE_STN_BASIC)          
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
        self.yrMths = utld.get_mth_metadata(uYrs[0],uYrs[-1])
        
        self.yrMasks = []
        
        for aYr in uYrs:
            
            self.yrMasks.append(np.nonzero(self.yrMths[YEAR]==aYr)[0])
        
        self.uYrs = uYrs
    
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
            yrMask = np.nonzero(np.logical_and(self.yrMths[YEAR] >= startYear,self.yrMths[YEAR] <= endYear))[0]   
        tairMth = self.dailyToMthly(tair, -1)[0]
        
        if len(tairMth.shape) == 1:
            tairMth = np.take(tairMth, yrMask)
        elif len(tairMth.shape) == 2:
            tairMth = np.take(tairMth, yrMask,axis=0)
        elif len(tairMth.shape) == 3:
            tairMth = np.take(tairMth, yrMask,axis=0)
        
        mths = np.take(self.yrMths[MONTH], yrMask)
        
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
            yrMask = np.nonzero(np.logical_and(self.yrMths[YEAR] >= startYear,self.yrMths[YEAR] <= endYear))[0]   
        
        if len(tairMth.shape) == 1:
            tairMth = np.take(tairMth, yrMask)
        elif len(tairMth.shape) == 2:
            tairMth = np.take(tairMth, yrMask,axis=0)
        elif len(tairMth.shape) == 3:
            tairMth = np.take(tairMth, yrMask,axis=0)
        
        mths = np.take(self.yrMths[MONTH], yrMask)
        
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

def getStns(fpathStns):

    afile = open(fpathStns)
    lines = afile.readlines()
    
    stns = np.empty(len(lines), dtype=DTYPE_STN_BASIC)
    stns[STN_ID] = np.array([aline[0:11] for aline in lines],np.str)
    stns[LAT] = np.array([np.float(aline[12:20]) for aline in lines])
    stns[LON] = np.array([np.float(aline[21:30]) for aline in lines])
    stns[ELEV] = np.array([np.float(aline[31:37]) for aline in lines])
    stns[STATE] = np.array([aline[38:40] for aline in lines],np.str)
    stns[STN_NAME] = np.array([aline[41:71].strip() for aline in lines],np.str)
    
    return stns

def loadObs(stnid,pathObs,tairVar,ushcnType,minYr=1948,maxYr=2012):
    
    #ushcnType = raw,tob,FLs.52i
    afile = open("".join([pathObs,stnid,".",ushcnType,".",tairVar]))
    
    mthvals = np.ones(12*((maxYr-minYr)+1))*-9999.0
    
    for aline in afile.readlines():
        
        yr = np.int(aline[12:16])
        
        if yr >= minYr and yr <= maxYr:
            
            idx = (yr-minYr)*12
        
            yrmthvals = np.array([aline[17:17+5],aline[26:26+5],aline[35:35+5],aline[44:44+5],
                       aline[53:53+5],aline[62:62+5],aline[71:71+5],aline[80:80+5],
                       aline[89:89+5],aline[98:98+5],aline[107:107+5],aline[116:116+5]],dtype=np.float)
            
            #mthvals = np.concatenate((mthvals,yrmthvals))
            mthvals[idx:idx+12] = yrmthvals
    
    mthvals[mthvals==-9999] = np.nan
    mthvals = mthvals/100.0
    
#    if mthvals.size != (12*((maxYr-minYr)+1)):
#        raise Exception('Missing monthly values for '+"".join([pathObs,stnid,".raw.",tairVar]))
    
    return mthvals
    
def buildGhcnUShcnMask(stnids,fpathGhcnStns):
    
    #Build dict of HCN flags
    afile = open(fpathGhcnStns)
        
    lines = afile.readlines()
    
    ghcnIds = np.array(["".join(["GHCN_",aline[0:11].strip()]) for aline in lines],np.str)
    hcnMask = np.array([aline[76:79] == "HCN" for aline in lines],np.bool)
    hcnIds = ghcnIds[hcnMask]
    
    return np.array([stnid in hcnIds for stnid in stnids],dtype=np.bool)

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

def createUshcnDs(fpathStns,pathObs,fpathdsout,minYr=1948,maxYr=2012):
    
    stns = getStns(fpathStns)
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
    
    stchk = status_check(stns.size*len(tairVarNames),100)
    
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
    
    createUshcnDs('/projects/daymet2/station_data/ushcn/ushcn-v2.5-stations.txt',
              '/projects/daymet2/station_data/ushcn/ushcn.v2.5.0.20130622/',
              '/projects/daymet2/station_data/ushcn/ushcn1895_2012.nc',minYr=1895, maxYr=2012)
    