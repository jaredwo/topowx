'''
Created on May 15, 2013

@author: jared.oyler
'''
from twx.db.station_data import STN_ID,LON,LAT,BAD,StationSerialDataDb,STATE,STN_NAME,ELEV,\
    StationDataDb,YMD,DATE,DAY,DTYPE_STN_BASIC
import numpy as np
from twx.utils.util_dates import YEAR,MONTH
from twx.utils.status_check import status_check
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import obs_por as por
import twx.utils.util_dates as utld
from datetime import datetime
import twx.db.create_db_all_stations as createDB
import twx.db.ushcn as ushcn
from netCDF4 import date2num
import netCDF4


class insert_homog(createDB.insert):
    '''
    Class for inserting homogenized stations and observations
    '''
    
    def __init__(self,stnda,homog_dly_tmin,homog_dly_tmax,fpathPor,fpathUnusableTmin,fpathUnusableTmax ):
        
        createDB.insert.__init__(self,stnda.days[DATE][0],stnda.days[DATE][-1])
        
        self.homog_tmin = homog_dly_tmin
        self.homog_tmax = homog_dly_tmax
        self.stnda = stnda
        
        porData = por.load_por_csv(fpathPor)
        mask_por_tmin, mask_por_tmax, mask_por_prcp = por.build_valid_por_masks(porData)
        
        unuseTminIds = np.sort(np.loadtxt(fpathUnusableTmin, dtype=np.str,usecols=[0]))
        unuseTmaxIds = np.sort(np.loadtxt(fpathUnusableTmax, dtype=np.str,usecols=[0]))
        
        fmtIds = np.array([formatStnId(stnId) for stnId in stnda.stn_ids])
                
        self.stns_tmin = stnda.stns[np.logical_and(np.logical_and(mask_por_tmin,np.logical_not(np.in1d(fmtIds, unuseTminIds, True))),
                                   np.in1d(fmtIds,np.unique(homog_dly_tmin.pha_adjs['stn_id']),True))]
        self.stns_tmax = stnda.stns[np.logical_and(np.logical_and(mask_por_tmax,np.logical_not(np.in1d(fmtIds, unuseTmaxIds, True))),
                                   np.in1d(fmtIds,np.unique(homog_dly_tmax.pha_adjs['stn_id']),True))]
        
        uniqIds = np.unique(np.concatenate((self.stns_tmin[STN_ID],self.stns_tmax[STN_ID])))
        
        self.stns_all = stnda.stns[np.in1d(stnda.stn_ids, uniqIds, True)]
        
        self.stn_list = [(stn[STN_ID],stn[LAT],stn[LON],stn[ELEV],stn[STATE],stn[STN_NAME]) for stn in self.stns_all]
        
        self.empty_obs = np.ones(stnda.days.size)*createDB.MISSING
        self.empty_qa = np.zeros(stnda.days.size,dtype=np.str)
    
    def get_stns(self):
        return self.stn_list
                            
    def parse_stn_obs(self,stn_id):
        
        if stn_id in self.stns_tmin[STN_ID]:
            tminHomog = self.homog_tmin.homog_stn(stn_id)[2]
            tminHomog = np.ma.filled(tminHomog, createDB.MISSING)
        else:
            tminHomog = self.empty_obs
        
        if stn_id in self.stns_tmax[STN_ID]:
            tmaxHomog = self.homog_tmax.homog_stn(stn_id)[2]
            tmaxHomog = np.ma.filled(tmaxHomog, createDB.MISSING)
        else:
            tmaxHomog = self.empty_obs
                
        obs = np.empty(self.stnda.days.size, dtype=createDB.DTYPE_STNOBS)                
        obs['year'] = self.stnda.days[YEAR]
        obs['month'] = self.stnda.days[MONTH]
        obs['day'] = self.stnda.days[DAY]
        obs['ymd'] = self.stnda.days[YMD]
        obs['tmin'] = tminHomog
        obs['tmax'] = tmaxHomog
        obs['prcp'] = self.empty_obs
        obs['swe'] = self.empty_obs
        obs['qflag_tmin'] = self.empty_qa
        obs['qflag_tmax'] = self.empty_qa
        obs['qflag_prcp'] = self.empty_qa
                
        return obs

def tobsShiftTmax(tmax,tobs):
    
    #consider no tobs as 2400
    tobs = np.ma.filled(tobs, 2400)
    
    tobsMaskAM = tobs < 1100
            
    if np.sum(tobsMaskAM) > 0:
                        
        tobsMaskOk = np.logical_and(~tobsMaskAM,np.isfinite(tmax))
        idxShift = np.nonzero(tobsMaskAM)[0]
        idxShift = idxShift[idxShift > 0]
        idxShift = idxShift[~tobsMaskOk[idxShift-1]]
        
        if idxShift.size > 1:
        
            tmaxShift = np.ones(tmax.size)*np.nan
            tmaxShift[tobsMaskOk] = tmax[tobsMaskOk]
            tmaxShift[idxShift-1] = tmax[idxShift]
            tmax = tmaxShift
    
    return tmax
 
class insert_tobs(createDB.insert):
    '''
    Class for inserting stations observations that have been modified for time-of-observation
    '''
    
    def __init__(self,stnda,fpath_tobs_ds,fpathPor):
        
        createDB.insert.__init__(self,stnda.days[DATE][0],stnda.days[DATE][-1])
        self.stnda = stnda
        
        porData = por.load_por_csv(fpathPor)
        mask_por_tmin, mask_por_tmax, mask_por_prcp = por.build_valid_por_masks(porData)
                
        self.stns_tmin = stnda.stns[mask_por_tmin]
        self.stns_tmax = stnda.stns[mask_por_tmax]
        
        uniqIds = np.unique(np.concatenate((self.stns_tmin[STN_ID],self.stns_tmax[STN_ID])))
        
        self.stns_all = stnda.stns[np.in1d(stnda.stn_ids, uniqIds, True)]
        
        self.stn_list = [(stn[STN_ID],stn[LAT],stn[LON],stn[ELEV],stn[STATE],stn[STN_NAME]) for stn in self.stns_all]
        
        self.ds_tobs = Dataset(fpath_tobs_ds)
        self.ds_tobs_stnids = self.ds_tobs.variables['stn_id'][:].astype("<S16")
        
        self.empty_obs = np.ones(stnda.days.size)*createDB.MISSING
        self.empty_qa = np.zeros(stnda.days.size,dtype=np.str)
    
    def get_stns(self):
        return self.stn_list
                            
    def parse_stn_obs(self,stn_id):
        
        if stn_id in self.stns_tmin[STN_ID]:
            tmin = self.stnda.load_all_stn_obs_var(stn_id,'tmin')[0]
            tmin[np.isnan(tmin)] = createDB.MISSING
        else:
            tmin = self.empty_obs
        
        if stn_id in self.stns_tmax[STN_ID]:

            tmax = self.stnda.load_all_stn_obs_var(stn_id,'tmax')[0]
            tobs = self.ds_tobs.variables['tobs'][:,np.nonzero(self.ds_tobs_stnids==stn_id)[0][0]]
            tmax = tobsShiftTmax(tmax, tobs)
            tmax[np.isnan(tmax)] = createDB.MISSING
            
        else:
            
            tmax = self.empty_obs
                                
        obs = np.empty(self.stnda.days.size, dtype=createDB.DTYPE_STNOBS)                
        obs['year'] = self.stnda.days[YEAR]
        obs['month'] = self.stnda.days[MONTH]
        obs['day'] = self.stnda.days[DAY]
        obs['ymd'] = self.stnda.days[YMD]
        obs['tmin'] = tmin
        obs['tmax'] = tmax
        obs['prcp'] = self.empty_obs
        obs['swe'] = self.empty_obs
        obs['qflag_tmin'] = self.empty_qa
        obs['qflag_tmax'] = self.empty_qa
        obs['qflag_prcp'] = self.empty_qa
                
        return obs

def formatStnId(stnId):
    
    if stnId.startswith("GHCN_"):
        
        outId = stnId.split("_")[1]
    
    elif stnId.startswith("SNOTEL_"):
        
        outId = stnId.split("_")[1]
        outId = "".join(["SNT","{0:0>8}".format(outId)])
        
    elif stnId.startswith("RAWS_"):
        
        outId = stnId.split("_")[1]
        outId = "".join(["RAW","{0:0>8}".format(outId)])
    
    return outId

def toGhcnStnList(stns,fpathOut):

    fout = open(fpathOut,"w")
    
    for stn in stns:
        
        outId = formatStnId(stn[STN_ID])
                
        if stn[LAT] < 0 or stn[LON] >= 0:
            raise Exception("Only handles formating of positive Lats and negative Lons.")
        
        outLat = "{0:0<8.5F}".format(stn[LAT])
        
        if np.abs(stn[LON]) < 100:
            fmtLon = "{0:0<9.5F}"
        else:
            fmtLon = "{0:0<9.4F}"
        
        outLon = fmtLon.format(stn[LON])
    
        fout.write(" ".join([outId,outLat,outLon,"\n"]))

def toGHCNmDataFiles(stns,stn_da,varname,dirPathOut):
    
    yrs = np.unique(stn_da.days[YEAR])
    yrmthMasks = {}
    
    for yr in yrs:
        
        mthmasks = []
    
        for mth in np.arange(1,13):
            
            mthmasks.append(np.logical_and(stn_da.days[YEAR]==yr,stn_da.days[MONTH]==mth))
        
        yrmthMasks[yr] = mthmasks
    
    
    schk = status_check(stns.size,50)
    for stn in stns:
        
        outId = formatStnId(stn[STN_ID])
        
        fout = open("".join([dirPathOut,outId,".raw.",varname]),'w')
        
        tair = stn_da.load_obs(stn[STN_ID])
        
        for yr in yrs:
            
            outLine = " ".join([outId,str(yr)])
            
            for yrmthmask in yrmthMasks[yr]:
                
                mthTair = np.mean(tair[yrmthmask],dtype=np.float64)*100.0
                outLine = "".join([outLine," {0:>5.0f}".format(mthTair),"   "])
            outLine = "".join([outLine,"\n"])
            fout.write(outLine)
        schk.increment()

def mthlyToGHCNmDataFiles(stns,data,yrs,varname,dirPathOut):
        
    schk = status_check(stns.size,50)
    for stn,x in zip(stns,np.arange(stns.size)):
        
        outId = formatStnId(stn[STN_ID])
        
        fout = open("".join([dirPathOut,outId,".raw.",varname]),'w')
        
        tair = data[:,x]
        
        for yr,i in zip(yrs,np.arange(0,yrs.size*12,12)):
            
            outLine = " ".join([outId,str(yr)])
            tairYr = tair[i:i+12]
            
            if np.ma.isMA(tairYr):
                ttairYr = tairYr.data
                validMask = np.logical_not(tairYr.mask)
                ttairYr[validMask] = ttairYr[validMask]*100.0
                ttairYr[tairYr.mask] = -9999
                tairYr = ttairYr
            else:
                tairYr = tairYr*100.0
            
            for aVal in tairYr:
                outLine = "".join([outLine," {0:>5.0f}".format(aVal),"   "])
            outLine = "".join([outLine,"\n"])
            fout.write(outLine)
        schk.increment()

class HomogRawDaily():
    
    def __init__(self,stnda,homogDfilesPath,adjLogPath,varname):
        
        self.stnda = stnda
        self.mthly_data = self.stnda.ds.variables['_'.join([varname,'mth'])][:]
        self.miss_data = self.stnda.ds.variables['_'.join([varname,'mthmiss'])][:]
        
        self.pha_adjs = parsePhaAdj(adjLogPath)
        
        self.path_FLs_data = homogDfilesPath
        self.varname = varname
        self.mths = utld.get_mth_metadata(self.stnda.days[YEAR][0], self.stnda.days[YEAR][-1])
        
        self.dly_yrmth_masks = []
        yrs = np.unique(self.stnda.days[YEAR])
        
        for yr in yrs:
        
            for mth in np.arange(1,13):
                
                self.dly_yrmth_masks.append(np.logical_and(stnda.days[YEAR]==yr,stnda.days[MONTH]==mth))
        
        self.ndays_per_mth = np.zeros(len(self.dly_yrmth_masks))
        
        for x in np.arange(self.ndays_per_mth.size):
            self.ndays_per_mth[x] = np.sum(self.dly_yrmth_masks[x])
        
        self.mthly_yr_masks = {}
        for yr in yrs:
            self.mthly_yr_masks[yr] = self.mths[YEAR] == yr
    
    def getFLs(self,stnId):
        
        fstnId = formatStnId(stnId)
        fileHomogMth = open("".join([self.path_FLs_data,fstnId,'.FLs.r00.',self.varname]))        
        mthvalsHomog = np.ones(self.mths.size)*-9999
        
        for aline in fileHomogMth.readlines():
            
            yr = int(aline[12:17])
            yrmthvals = np.array([aline[17:17+5],aline[26:26+5],aline[35:35+5],aline[44:44+5],
                       aline[53:53+5],aline[62:62+5],aline[71:71+5],aline[80:80+5],
                       aline[89:89+5],aline[98:98+5],aline[107:107+5],aline[116:116+5]],dtype=np.float)
            
            mthvalsHomog[self.mthly_yr_masks[yr]] = yrmthvals
        
        if np.sum(mthvalsHomog == -9999) > 0:
            print 'Warning: incomplete station record: '+"".join([self.path_FLs_data,fstnId,'.FLs.r00.',self.varname])
        mthvalsHomog = np.ma.masked_array(mthvalsHomog,mask=mthvalsHomog == -9999)
        mthvalsHomog = mthvalsHomog/100.0
        return mthvalsHomog
       
    def homog_stn(self,stn_id):
        
        fstnId = formatStnId(stn_id)
        fileHomogMth = open("".join([self.path_FLs_data,fstnId,'.FLs.r00.',self.varname]))
        
        #print "".join([self.path_FLs_data,fstnId,'.FLs.r00.',self.varname])
        
        mthvalsHomog = np.ones(self.mths.size,dtype=np.float)*-9999
        
        for aline in fileHomogMth.readlines():
            
            yr = int(aline[12:17])
            yrmthvals = np.array([aline[17:17+5],aline[26:26+5],aline[35:35+5],aline[44:44+5],
                       aline[53:53+5],aline[62:62+5],aline[71:71+5],aline[80:80+5],
                       aline[89:89+5],aline[98:98+5],aline[107:107+5],aline[116:116+5]],dtype=np.float)
            
            mthvalsHomog[self.mthly_yr_masks[yr]] = yrmthvals
        
        mthvalsHomog = np.ma.masked_array(mthvalsHomog,mthvalsHomog==-9999)
        mthvalsHomog = mthvalsHomog/100.0
        mthvalsHomog = np.round(mthvalsHomog,2)

        dlyVals = self.stnda.load_all_stn_obs_var(stn_id,self.varname)[0].astype(np.float64)
        dlyVals = np.ma.masked_array(dlyVals,np.isnan(dlyVals))        
        mthVals = np.ma.round(self.mthly_data[:,self.stnda.stn_idxs[stn_id]].astype(np.float64),2)
        missCnts = self.miss_data[:,self.stnda.stn_idxs[stn_id]]
        dlyValsHomog = np.copy(dlyVals)
        
        stnChgPt = self.pha_adjs[self.pha_adjs['stn_id']==fstnId]
        stnChgPt = stnChgPt[np.argsort(stnChgPt['ymdStr'])]
        
        difCnt = 0
        
        for x in np.arange(mthVals.size):
            
            if not np.ma.is_masked(mthVals[x]) and not np.ma.is_masked(mthvalsHomog[x]):
            
                if mthVals[x] != mthvalsHomog[x]:

                    delta = mthvalsHomog[x] - mthVals[x]                    
                    #print self.mths[YMD][x],delta
                    
                    #dlyValsHomog[self.dly_yrmth_masks[x]] = (dlyValsHomog[self.dly_yrmth_masks[x]]-np.round(mthVals[x],2))+mthvalsHomog[x]
                    dlyValsHomog[self.dly_yrmth_masks[x]] = dlyValsHomog[self.dly_yrmth_masks[x]] + delta
                    difCnt+=1
            
            elif missCnts[x] < self.ndays_per_mth[x] and not np.ma.is_masked(mthvalsHomog[x]):
                
                ymd = self.mths[YMD][x]
                
                if ymd < stnChgPt['ymdStr'][0]:
                    #before all change points. assume it falls under the earliest change point
                    delta = -stnChgPt['adj'][0]
                else:
                    maskChgPt = np.logical_and(stnChgPt['ymdStr'] <= ymd, stnChgPt['ymdEnd'] >= ymd)
                    sumMask = np.sum(maskChgPt)
                    
                    if sumMask == 0:
                        #don't do anything. past the last change point which is theoretically 0
                        delta = 0
                    elif sumMask == 1:
                        
                        delta = -stnChgPt[maskChgPt]['adj'][0]
                    else:
                        raise Exception("Falls within more than one change point")
                        
                dlyValsHomog[self.dly_yrmth_masks[x]] = dlyValsHomog[self.dly_yrmth_masks[x]] + np.round(delta,2)
                            
        return mthvalsHomog,mthVals,dlyValsHomog,dlyVals,difCnt
        
class HomogDaily():
    
    def __init__(self,stnda,homogDfilesPath,varname):
        
        self.stnda = stnda
        self.path_FLs_data = homogDfilesPath
        self.varname = varname
        
        self.yrmthMasks = []
        yrs = np.unique(self.stnda.days[YEAR])
        
        for yr in yrs:
        
            for mth in np.arange(1,13):
                
                self.yrmthMasks.append(np.logical_and(stnda.days[YEAR]==yr,stnda.days[MONTH]==mth))
        
    def homog_stn(self,stn_id):
        
        fstnId = formatStnId(stn_id)
        fileHomogMth = open("".join([self.path_FLs_data,fstnId,'.FLs.r00.',self.varname]))
        #print fileHomogMth
        mthvalsHomog = np.array([])
        
        for aline in fileHomogMth.readlines():
            
            yrmthvals = np.array([aline[17:17+5],aline[26:26+5],aline[35:35+5],aline[44:44+5],
                       aline[53:53+5],aline[62:62+5],aline[71:71+5],aline[80:80+5],
                       aline[89:89+5],aline[98:98+5],aline[107:107+5],aline[116:116+5]],dtype=np.float)/100.0
            
            mthvalsHomog = np.concatenate((mthvalsHomog,yrmthvals))
        
        dlyVals = self.stnda.load_obs(stn_id).astype(np.float64)
        mthVals = self.getMthVals(dlyVals)
        dlyValsHomog = np.copy(dlyVals)
        
        
        difCnt = 0
        
        for x in np.arange(mthVals.size):
            
            if np.round(mthVals[x],2) != np.round(mthvalsHomog[x],2):
                
                dlyValsHomog[self.yrmthMasks[x]] = (dlyValsHomog[self.yrmthMasks[x]]-mthVals[x])+mthvalsHomog[x]
                difCnt+=1
        
        return dlyValsHomog,difCnt
        
    def getMthVals(self,dlyVals):
        
        mthVals = np.zeros(len(self.yrmthMasks))
        
        for x in np.arange(len(self.yrmthMasks)):
            
            mthVals[x] = np.round(np.mean(dlyVals[self.yrmthMasks[x]],dtype=np.float64),2)
        
        return mthVals

def updateHomogStns(stns,homogDly,dsout):
    
    stnids = dsout.variables['stn_id'][:].astype('<S16')
    
    if 'nmths_adj' not in dsout.variables.keys():
            
        avar = dsout.createVariable('nmths_adj','f8',('stn_id',))
        avar.long_name = 'Number of months adjusted for homogenization'
        avar.units = 'NA'
        dsout.sync()
    
    dsout.variables['nmths_adj'][:] = 0
    dsout.sync()
    
    schk = status_check(stns.size,100)
    nUnuse = 0
    for stn in stns:
        
        try:
            hTair,difCnt = homogDly.homog_stn(stn[STN_ID])
        except IOError:
            difCnt = 0
            idx = np.nonzero(stnids==stn[STN_ID])[0][0]
            dsout.variables[BAD][idx] = 1
            dsout.sync()
            print "".join(['STN Considered Unusable: ',stn[STN_ID]])
            nUnuse+=1
            
        if difCnt > 0:
            
            #New mean
            mtair = np.mean(hTair,dtype=np.float64)
            #Centered tair
            ctair = hTair.astype(np.float64) - mtair
            
            idx = np.nonzero(stnids==stn[STN_ID])[0][0]
            
            dsout.variables[homogDly.varname][:,idx] = hTair
            dsout.variables["".join([homogDly.varname,"_center"])][:,idx] = ctair
            dsout.variables['obs_mean'][idx] = mtair
            dsout.variables['nmths_adj'][idx] = difCnt
            dsout.sync()
        
        schk.increment()
    
    print "Total # of unusable stations: "+str(nUnuse)

def getAnnVals(yrs,dlyVals):
    
    uyrs = np.unique(yrs)
    
    annVals = []
    for yr in uyrs:
        annVals.append(np.mean(dlyVals[yrs==yr],dtype=np.float))
    return np.array(annVals)

def parsePhaAdj(adjLogPath):
    f = open(adjLogPath)
    
    #aDtype = [('stn_id', "<S16"), ('yrStr',np.int), ('mthStr', np.int), ('yrEnd',np.int), ('mthEnd', np.int), ('adj', np.float64)]
    aDtype = [('stn_id', "<S16"), ('ymdStr',np.int),('ymdEnd',np.int),('adj', np.float64)]
    
    valsAdj = []
    
    for aline in f.readlines():
        
        stnid = aline[10:21]
        yrmthStr = aline[25:31]
        yrmthEnd = aline[45:51]
        valAdj = np.float(aline[75:81])
        
        dateStr = datetime(np.int(yrmthStr[0:4]),np.int(yrmthStr[-2:]),1)
        ymdStr = np.int(datetime.strftime(dateStr,"%Y%m%d"))
        
        dateEnd = datetime(np.int(yrmthEnd[0:4]),np.int(yrmthEnd[-2:]),1)
        ymdEnd = np.int(datetime.strftime(dateEnd,"%Y%m%d"))
         
        #if valAdj != 0.0:

        valsAdj.append((stnid,ymdStr,ymdEnd,valAdj))
    
    valsAdj = np.array(valsAdj,dtype=aDtype)
    
#    print valsAdj[valsAdj['adj'] > 4]
#    print np.mean(valsAdj['adj']),np.max(valsAdj['adj']),np.min(valsAdj['adj'])
#    plt.hist(valsAdj['adj'],200)
#    plt.show()
#    plt.boxplot(valsAdj['adj'])
#    plt.show()

    return valsAdj

def addMthlyMeansTobsDs(dsPathRaw,dsPathTobs,varName):
    
    ds = Dataset(dsPathTobs,'r+')
    stnids = ds.variables['stn_id'][:].astype("<S16")
    
    stnda = StationDataDb(dsPathRaw)
    tagg = ushcn.TairAggregate(stnda.days)
    minDate = stnda.days[DATE][0]
    stns = stnda.stns[np.in1d(stnda.stn_ids, stnids, True)]
    stnda.ds.close()
    stnda = None
    
    if 'time_mth' not in ds.variables.keys():
        
        ds.createDimension('time_mth',tagg.yr_mths.size)
        times = ds.createVariable('time_mth','f8',('time_mth',),fill_value=False)
        times.units = "".join(["days since ",str(minDate.year),"-",str(minDate.month),"-",str(minDate.day)," 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times[:] = date2num(tagg.yr_mths[DATE],times.units)
    
    varMthlyName = "_".join([varName,"mth"])
    if varMthlyName not in ds.variables.keys(): 
        varMthly = ds.createVariable(varMthlyName,'f4',('time_mth','stn_id'),fill_value=netCDF4.default_fillvals['f4'])
    else:
        varMthly = ds.variables[varMthlyName]
        
    varMissName = "_".join([varName,"mthmiss"])
    if varMissName not in ds.variables.keys(): 
        varMiss = ds.createVariable(varMissName,'i2',('time_mth','stn_id'),fill_value=netCDF4.default_fillvals['i2'])
    else:
        varMiss = ds.variables[varMissName]
        
    varDly = ds.variables[varName]
    chkSize = 50
    
    stchk = status_check(np.int(np.round(stns.size/np.float(chkSize))), 10)
    for i in np.arange(0,stns.size,chkSize):
        
        if i + chkSize < stns.size:
            nStns = chkSize
        else:
            nStns = stns.size - i
        
        dlyVals = varDly[:,i:i+nStns]
                        
        mthVals,nMiss = tagg.dailyToMthly(dlyVals,maxMiss=9)
        
        if np.ma.isMA(mthVals):
            tmthVals = mthVals.data
            tmthVals[mthVals.mask] = varMthly._FillValue
            mthVals = tmthVals
            
        varMthly[:,i:i+nStns] = mthVals
        varMiss[:,i:i+nStns] = nMiss
        ds.sync()
        stchk.increment()

if __name__ == '__main__':
    
    
    ###############################
    #Analyze rate of change points
    ###############################
#    valsAdj = parsePhaAdj('/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/output/PhaAdjTmin.log')
#    stnYrs = np.unique(valsAdj['stn_id']).size*65
#    nChgPts = np.sum(valsAdj['adj'] != 0)
#    
#    print nChgPts,stnYrs
    
    ###############################
#    valsAdj = parsePhaAdj('/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/output/PhaAdjTmax.log')
##    stn_da = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130518/serial_tmin.nc', 'tmin',
##                                 stn_dtype=[(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"),
##                                            (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),(BAD, np.float64)])
##    
#    print valsAdj['stn_id'][valsAdj['adj'] == np.min(valsAdj['adj'])]
#    
#    valsAdjNz = valsAdj[valsAdj['adj'] != 0]
#    print np.mean(np.abs(valsAdjNz['adj'])),np.mean(valsAdjNz['adj']),np.max(valsAdjNz['adj']),np.min(valsAdjNz['adj'])
#    plt.hist(valsAdjNz['adj'],bins=50)
#    plt.show()
#    sys.exit()
#    
#    maskBad = np.isnan(stn_da.stns[BAD])
#    stns = stn_da.stns[maskBad]
#    
#    print np.sum(np.logical_and(np.logical_and(valsAdj['yrStr']==1948,valsAdj['mthStr']==1),
#                 np.logical_and(valsAdj['yrEnd']==2012,valsAdj['mthEnd']==12)))
#    
##    
#    adjIds = np.unique(valsAdj[STN_ID])
#    
#    #The # of chg points in the # of adjustments - 1
#    cnt1 = 0
#    for aId in adjIds:
#        
#        idMask = valsAdj[STN_ID] == aId
#        if np.sum(idMask)==1:
#            print aId
#            cnt1+=1
#    print cnt1
#    
#    fmtIds = np.array([formatStnId(stnId) for stnId in stns[STN_ID]])
#    
#    print stns[STN_ID][np.logical_not(np.in1d(fmtIds, adjIds, True))].size
#    print fmtIds[np.logical_not(np.in1d(fmtIds, adjIds, True))]
    
#############################################################
#Update daily values to match homogenized monthly values 
#############################################################
#    stn_da = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130518/serial_tmax.nc', 'tmax',
#                                 stn_dtype=[(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"),
#                                            (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),(BAD, np.float64)])
#    homog = HomogDaily(stn_da,'/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/monthly/FLs.r00/', 'tmax')
#    maskBad = np.isnan(stn_da.stns[BAD])
#    stns = stn_da.stns[maskBad]
#    dsout = Dataset('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc','r+')
#    
#    updateHomogStns(stns, homog, dsout)

#############################################################
#Look at homogenization updates to a single station
#############################################################
#    stn_da = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130518/serial_tmax.nc', 'tmax',
#                                 stn_dtype=[(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"),
#                                            (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),(BAD, np.float64)])
#    homog = HomogDaily(stn_da,'/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/monthly/FLs.r00/', 'tmax')
#    stnId = 'GHCN_USC00013160'#'SNOTEL_13C01S'#SNOTEL_19K08S SNOTEL_07K13S
#    print stn_da.stns[stn_da.stn_ids==stnId][0]
#    hTair,difCnt = homog.homog_stn(stnId)
#    print "".join([stnId,": # of adjusted months: ",str(difCnt)])
#    tair = stn_da.load_obs(stnId)
#    
#    plt.plot(tair)
#    plt.show()
#    plt.clf()
#    
#    hTairAnn = getAnnVals(stn_da.days[YEAR], hTair)
#    tairAnn = getAnnVals(stn_da.days[YEAR], tair)
#    
#    print np.mean(hTair),np.mean(tair)
#    
#    plt.subplot(211)
#    plt.plot(hTairAnn)#-np.mean(hTairAnn))
#    plt.plot(tairAnn)#-np.mean(tairAnn))
##    ylim=plt.ylim()
##    plt.subplot(312)
##    plt.plot(tair-np.mean(tair))
##    plt.ylim(ylim)
#    plt.subplot(212)
#    plt.plot((hTair-tair))
#    plt.show()

#############################################################
#Output station data to Ghcn format for input to the PHA Fortran program
#############################################################
#    stn_da = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130518/serial_tmax.nc', 'tmax',
#                                 stn_dtype=[(STN_ID, "<S16"), (STATE, "<S2"), (STN_NAME, "<S30"),
#                                            (LON, np.float64), (LAT, np.float64), (ELEV, np.float64),(BAD, np.float64)])
#    maskBad = np.isnan(stn_da.stns[BAD])
##    stnids_aoi = np.loadtxt('/projects/daymet2/station_data/infill/infill_fnl/montana_aoi_stns.csv',np.str, delimiter=",",usecols=[0])
##    maskMt = np.in1d(stn_da.stn_ids,stnids_aoi,True)
##    stns = stn_da.stns[np.logical_and(maskMt,maskBad)]
#    stns = stn_da.stns[maskBad]
##    print stns.size
#    toGhcnStnList(stns,'/projects/daymet2/inhomo_software/mydata/tmax/world1_stnlist.tmax')
#    toGHCNmDataFiles(stns, stn_da,'tmax', '/projects/daymet2/inhomo_software/mydata/tmax/raw/')

#############################################################
#Output RAW station data to Ghcn format for input to the PHA Fortran program
#############################################################
#    stn_da = StationDataDb('/projects/daymet2/station_data/all/all_1948_2012.nc')
#    ds_tobs = Dataset('/projects/daymet2/station_data/all/tairTobs_1948_2012.nc')
#    
#    porData = por.load_por_csv('/projects/daymet2/station_data/all/all_por_1948_2012.csv')
#    mask_por_tmin, mask_por_tmax, mask_por_prcp = por.build_valid_por_masks(porData)
#    
#    stnids = ds_tobs.variables['stn_id'][:].astype("<S16")
#    maskIds = np.in1d(stnids,stn_da.stn_ids[mask_por_tmin])
#    stnids = stnids[maskIds]
#    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stnids, True)]
#    
#    tairMth = ds_tobs.variables['tmin_mth'][:,maskIds]
#
#    print stnids.size,stns.size,tairMth.shape
#    print stnids[6000],stns[6000]
#
#    toGhcnStnList(stns,'/projects/daymet2/inhomo_software/mydata/tmin/world1_stnlist.tmin')
#    mthlyToGHCNmDataFiles(stns, tairMth, np.unique(stn_da.days[YEAR]), 'tmin', 
#                          '/projects/daymet2/inhomo_software/mydata/tmin/raw/')
    
#############################################################
#RAW HOMOG: Look at homogenization updates to a single station
#############################################################

    
#    stn_da = StationDataDb('/projects/daymet2/station_data/all/tairTobs_1948_2012.nc',stnDtype=DTYPE_STN_BASIC)
#    homog = HomogRawDaily(stn_da,'/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/monthly/FLs.r00/', 
#                          '/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/output/PhaAdjTmin.log','tmin')
    
    stn_da = StationDataDb('/projects/daymet2/station_data/all/tairTobs_1948_2012.nc',stnDtype=DTYPE_STN_BASIC)
    homog = HomogRawDaily(stn_da,'/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/monthly/FLs.r00/', 
                          '/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/output/PhaAdjTmin.log','tmin')
                  
    stnId = 'GHCN_USC00244328'#'SNOTEL_13C01S'#SNOTEL_19K08S SNOTEL_07K13S
    print stn_da.stns[stn_da.stn_ids==stnId][0]
    mthvalsHomog,mthVals,dlyValsHomog,dlyVals,difCnt = homog.homog_stn(stnId)
    print "".join([stnId,": # of adjusted months: ",str(difCnt)])
    tair = stn_da.load_all_stn_obs_var(stnId,'tmin')[0].astype(np.float64)
    
    plt.plot(tair)
    plt.show()
    plt.clf()
    
    #hTairAnn = getAnnVals(stn_da.days[YEAR], hTair)
    #tairAnn = getAnnVals(stn_da.days[YEAR], tair)
    
    #print np.mean(hTair),np.mean(tair)
    
    #plt.subplot(211)
    #plt.plot(hTairAnn)#-np.mean(hTairAnn))
    #plt.plot(tairAnn)#-np.mean(tairAnn))
#    ylim=plt.ylim()
#    plt.subplot(312)
#    plt.plot(tair-np.mean(tair))
#    plt.ylim(ylim)
    #plt.subplot(212)
    #plt.plot((np.round(mthvalsHomog,2)-(np.round(mthVals,2))))
    plt.plot(dlyValsHomog-dlyVals)
    plt.show()
    plt.plot(mthvalsHomog)
    plt.plot(mthVals)
    plt.show()

#############################################################
#Create modified time-of-obs database
#############################################################
#    stnda = StationDataDb('/projects/daymet2/station_data/all/all_1948_2012.nc')
#    iTobs = insert_tobs(stnda,'/projects/daymet2/station_data/ghcn/ghcn_yrly/AllTobsTmax.nc',
#                        '/projects/daymet2/station_data/all/all_por_1948_2012.csv')
#    
#    createDB.create_db_ncdf('/projects/daymet2/station_data/all/tairTobs_1948_2012.nc', stnda.days[DATE][0], stnda.days[DATE][-1], (iTobs,))
#    createDB.insert_data_ncdf('/projects/daymet2/station_data/all/tairTobs_1948_2012.nc', (iTobs,))
#    
#    addMthlyMeansTobsDs('/projects/daymet2/station_data/all/all_1948_2012.nc',
#                        '/projects/daymet2/station_data/all/tairTobs_1948_2012.nc', 'tmin')
#    addMthlyMeansTobsDs('/projects/daymet2/station_data/all/all_1948_2012.nc',
#                        '/projects/daymet2/station_data/all/tairTobs_1948_2012.nc', 'tmax')
    
#############################################################
#Create homog database
#############################################################
#    stnda = StationDataDb('/projects/daymet2/station_data/all/tairTobs_1948_2012.nc',stnDtype=DTYPE_STN_BASIC)
#    homog_tmin = HomogRawDaily(stnda,'/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/monthly/FLs.r00/', 
#                              '/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/output/PhaAdjTmin.log','tmin')
#    homog_tmax = HomogRawDaily(stnda,'/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/monthly/FLs.r00/', 
#                              '/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/output/PhaAdjTmax.log','tmax')
#    fpathPor = '/projects/daymet2/station_data/all/tairTobs_por_1948_2012.csv'
#    fpathUnusableTmin = '/projects/daymet2/inhomo_software/pha_v52i_tmin/data/benchmark/world1/corr/meta.world1.tmin.r00.1307231630.1.input_not_stnlist'
#    fpathUnusableTmax = '/projects/daymet2/inhomo_software/pha_v52i_tmax/data/benchmark/world1/corr/meta.world1.tmax.r00.1307231609.1.input_not_stnlist'
#    
#    iHomog = insert_homog(stnda, homog_tmin, homog_tmax, fpathPor, fpathUnusableTmin, fpathUnusableTmax)
#    createDB.create_db_ncdf('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc', stnda.days[DATE][0], stnda.days[DATE][-1], (iHomog,))
#    createDB.insert_data_ncdf('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc', (iHomog,))