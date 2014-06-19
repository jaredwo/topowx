'''
MPI driver for running quality assurance procedures.

@author: jared.oyler
'''

from mpi4py import MPI
import qa_prcp
import qa_temp
import numpy as np
from twx.db.station_data import StationDataDb, STN_ID, TMIN, TMAX, PRCP, YMD, TMIN_FLAG, TMAX_FLAG, PRCP_FLAG,STATE
from netCDF4 import Dataset, date2num
import twx.utils.util_dates as utld
from twx.utils.util_misc import Unbuffered
import sys
from twx.utils.status_check import status_check
from qa_spatial import run_qa_spatial

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_SPATIAL_QA = 'P_SPATIAL_QA'
P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_STN_MASK = 'P_STN_MASK'

sys.stdout=Unbuffered(sys.stdout)

NA_STATE = np.array(["NA"], dtype="<U2")[0]

GHCN_FLAGS_MAP = {qa_temp.QA_OK:"",
                  qa_temp.QA_MISSING:"",
                  qa_temp.DUP:"D",
                  qa_temp.QA_DUP_YEAR:"D",
                  qa_temp.QA_DUP_MONTH:"D",
                  qa_temp.QA_DUP_YEAR_MONTH:"D",
                  qa_temp.QA_DUP_WITHIN_MONTH:"D",
                  qa_temp.QA_GAP:"G",
                  qa_temp.QA_INTERNAL_INCONSIST:"I",
                  qa_temp.QA_STREAK:"K",
                  qa_prcp.QA_FREQUENT:"K",
                  qa_temp.QA_MEGA_INCONSIST:"M",
                  qa_temp.QA_NAUGHT:"N",
                  qa_temp.QA_CLIM_OUTLIER:"O",
                  qa_temp.QA_LAGRANGE_INCONSIST:"R",
                  qa_temp.QA_SPATIAL_REGRESS:"S",
                  qa_temp.QA_SPATIAL_CORROB:"S",
                  qa_temp.QA_SPIKE_DIP:"T",
                  qa_temp.QA_IMPOSS_VALUE:"X", }


TOPOMET_FLAGS_MAP = {"":qa_temp.QA_OK,
                     "D":qa_temp.DUP,
                     "G":qa_temp.QA_GAP,
                     "I":qa_temp.QA_INTERNAL_INCONSIST,
                     "K":qa_temp.QA_STREAK,
                     "M":qa_temp.QA_MEGA_INCONSIST,
                     "N":qa_temp.QA_NAUGHT,
                     "O":qa_temp.QA_CLIM_OUTLIER,
                     "R":qa_temp.QA_LAGRANGE_INCONSIST,
                     "S":qa_temp.QA_SPATIAL_REGRESS,
                     "T":qa_temp.QA_SPIKE_DIP,
                     "X":qa_temp.QA_IMPOSS_VALUE}

class iter_flag_update:
       
    def __init__(self, stn_id, tmin_flags, tmax_flags, prcp_flags, ymd):
        self.stn_id = stn_id
        self.tmin_flags = tmin_flags
        self.tmax_flags = tmax_flags
        self.prcp_flags = prcp_flags
        self.ymd = ymd
        self.x = -1
        self.max = self.ymd.size - 1
        self.flag_map = GHCN_FLAGS_MAP

    def __iter__(self):
        return self

    def update_flags(self, curs):
        raise Exception('update_flags called on iter_flag_update')

    def next(self):
        self.x += 1
        if self.x > self.max:
            raise StopIteration
        else:
            return (self.flag_map[self.tmin_flags[self.x]], self.flag_map[self.tmax_flags[self.x]], self.flag_map[self.prcp_flags[self.x]], self.stn_id, long(self.ymd[self.x]))

class iter_multi_flag_update:
    
    def __init__(self):
        self.iters = []
        self.x = -1
    
    def __iter__(self):
        return self
    
    def add_iter(self, a_iter):
        self.iters.append(a_iter)
    
    def update_flags(self, db_path):
        self.max = len(self.iters) - 1
        self.x = 0
        self.cur_iter = self.iters[self.x]
        
        ds = Dataset(db_path, 'r+')
        stn_ids = ds.variables['stn_id'][:]
        time_units = ds.variables['time'].units
        
        print self.cur_iter.stn_id
        
        stn_id = ""
        stn_idx = 0
        while 1:
            
            try:
                #QFLAG_TMIN,QFLAG_TMAX,QFLAG_PRCP,STN_ID,YMD
                flag_row = self.next()
                
                if stn_id != flag_row[3]:
                    
                    stn_id = flag_row[3]
                    stn_idx = np.nonzero(stn_ids == stn_id)[0][0]
                 
                time_idx = int(date2num(utld.ymdL_to_date(flag_row[4]), units=time_units))
                
                ds.variables['qflag_tmin'][time_idx, stn_idx] = flag_row[0]
                ds.variables['qflag_tmax'][time_idx, stn_idx] = flag_row[1]
                ds.variables['qflag_prcp'][time_idx, stn_idx] = flag_row[2]
                
            except StopIteration:
                break
        
        ds.close()

    def next(self):
        try:
            return self.cur_iter.next()
        except StopIteration:
            self.x += 1
            if self.x > self.max:
                raise StopIteration
            else:
                while True:
                    try:
                        self.cur_iter = self.iters[self.x]
                        print self.cur_iter.stn_id
                        return self.cur_iter.next()
                    except StopIteration:
                        self.x += 1
                        if self.x > self.max:
                            raise StopIteration
     
def proc_work(params,rank):
    
    status = MPI.Status()
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    while 1:
        
        try:
            
            stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
            
            if status.tag == TAG_STOPWORK:
                MPI.COMM_WORLD.send(None, dest=RANK_WRITE, tag=TAG_STOPWORK)
                print "".join(["Worker ",str(rank),": Finished"]) 
                return 0
            
            obs = stn_da.load_all_stn_obs(np.array([stn_id]), set_flagged_nan=True)
            stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
            
            if params[P_SPATIAL_QA]:
                
                flags_tmin, flags_tmax, flags_prcp = run_qa_spatial(stn, stn_da, obs[TMIN], obs[TMAX], obs[PRCP], stn_da.days)
            
            else:
                
                flags_tmin, flags_tmax = qa_temp.run_qa_non_spatial(obs[TMIN], obs[TMAX], stn_da.days)
                flags_prcp = qa_prcp.run_qa_non_spatial(obs[PRCP], (obs[TMIN] + obs[TMAX]) / 2.0, stn_da.days) 
    
            a_iter = create_update_iter(stn, flags_tmin, flags_tmax, flags_prcp, stn_da.days,
                                             obs[TMIN_FLAG], obs[TMAX_FLAG], obs[PRCP_FLAG])
            
            MPI.COMM_WORLD.send(a_iter, dest=RANK_WRITE, tag=TAG_DOWORK)
        
        except Exception,e:
            
            print "".join(["Error in QA of ",stn_id,":",str(e),"\n"])
        
        MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
        
        
def create_update_iter(stn, flags_tmin, flags_tmax, flags_prcp, days, prev_flgs_tmin=np.array([]), prev_flgs_tmax=np.array([]), prev_flgs_prcp=np.array([])):
    
    mask_tmin = np.logical_not(np.logical_or(flags_tmin == qa_temp.QA_OK, flags_tmin == qa_temp.QA_MISSING))
    mask_tmax = np.logical_not(np.logical_or(flags_tmax == qa_temp.QA_OK, flags_tmax == qa_temp.QA_MISSING))
    mask_prcp = np.logical_not(np.logical_or(flags_prcp == qa_prcp.QA_OK, flags_prcp == qa_prcp.QA_MISSING))
    
    nobs_tmin = np.sum(flags_tmin != qa_temp.QA_MISSING)
    nobs_tmax = np.sum(flags_tmax != qa_temp.QA_MISSING)
    nobs_prcp = np.sum(flags_prcp != qa_temp.QA_MISSING)
    
    nflag_tmin = np.sum(mask_tmin)
    nflag_tmax = np.sum(mask_tmax)
    nflag_prcp = np.sum(mask_prcp)
    
    if np.sum(np.logical_or(flags_prcp==9,flags_prcp==19)):
        print "".join([stn[STN_ID]," has prcp freq/strk: ",str(np.sum(np.logical_or(flags_prcp==9,flags_prcp==19)))])
    
    if nobs_tmin == 0:
        pctflg_tmin = np.nan
    else:
        pctflg_tmin = (nflag_tmin/float(nobs_tmin))*100.0
        
    if nobs_tmax == 0:
        pctflg_tmax = np.nan
    else:
        pctflg_tmax = (nflag_tmax/float(nobs_tmax))*100.0
        
    if nobs_prcp == 0:
        pctflg_prcp = np.nan
    else:
        pctflg_prcp = (nflag_prcp/float(nobs_prcp))*100.0
    
    print "|".join([stn[STN_ID],"%.2f|%.2f|%.2f"%(pctflg_tmin,pctflg_tmax,pctflg_prcp)])
    
    mask_all = np.logical_or(np.logical_or(mask_tmin, mask_tmax), mask_prcp)
    #make sure previous flags are not overwritten in the update
    flags_tmin = set_prev_flags(flags_tmin, prev_flgs_tmin)
    flags_tmax = set_prev_flags(flags_tmax, prev_flgs_tmax)
    flags_prcp = set_prev_flags(flags_prcp, prev_flgs_prcp)
    
    return iter_flag_update(stn[STN_ID], flags_tmin[mask_all], flags_tmax[mask_all], flags_prcp[mask_all], days[YMD][mask_all])

def set_prev_flags(flags, prev_flgs):
    
    if prev_flgs.size > 0:
        
        uniq_flgs = np.unique(prev_flgs)
        
        for flag in uniq_flgs:
            
            if flag != "":
            
                mask_flg = prev_flgs == flag
                
                flags[np.logical_and(mask_flg, np.logical_or(flags == qa_temp.QA_OK, flags == qa_temp.QA_MISSING))] = TOPOMET_FLAGS_MAP[flag]
    
    return flags
     
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    iter_all = iter_multi_flag_update()

    nstns = np.nonzero(params[P_STN_MASK])[0].size
    stat_chk = status_check(nstns,10)
    
    while 1:
       
        a_iter = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                iter_all.update_flags(params[P_PATH_DB])
                print "Writer: Finished"
                return 0
        else:
            
            iter_all.add_iter(a_iter)
            stat_chk.increment()

def proc_coord(params,nwrkers):
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    stns = stn_da.stns[params[P_STN_MASK]]
    
    cnt = 0
    nrec = 0
    
    for stn in stns:
                
        if cnt < nwrkers:
            dest = cnt+N_NON_WRKRS
        else:
            dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            nrec+=1

        MPI.COMM_WORLD.send(stn[STN_ID], dest=dest, tag=TAG_DOWORK)
        cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send(stn[STN_ID], dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
 
if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    np.seterr(invalid='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    #params[P_PATH_DB] = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    params[P_PATH_DB] = '/projects/daymet2/station_data/china/chinaStns.nc'
    params[P_SPATIAL_QA] = True
    params[P_START_YMD] = None
    params[P_END_YMD] = None
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    params[P_STN_MASK] = np.ones(stn_da.stn_ids.size, dtype=np.bool)
    
#    sntl_mask = np.logical_and(np.char.startswith(stn_da.stn_ids,"SNOTEL"),stn_da.stns[STATE] != "AK")#np.char.startswith(stn_da.stn_ids,"SNOTEL")
#    glac_mask = np.char.startswith(stn_da.stn_ids,"GLAC")
#    raws_mask = np.char.startswith(stn_da.stn_ids,"RAWS")
#    camx_mask = np.logical_or(np.char.startswith(stn_da.stn_ids,"MX"),np.char.startswith(stn_da.stn_ids,"CA"))
#    #params[P_STN_MASK] = np.logical_or(camx_mask,np.logical_or(sntl_mask,glac_mask))
#    params[P_STN_MASK] = np.logical_or(raws_mask,sntl_mask)
    
    stn_da.ds.close()
    stn_da = None
    
    if rank == RANK_COORD:   
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)
