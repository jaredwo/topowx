'''
MPI script for estimating/infilling period-of-record means
and variances of incomplete Tmin/Tmax station time series.

Must be run using mpiexec or mpirun.
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db import StationDataDb,STN_ID,MEAN_TMIN,MEAN_TMAX,\
VAR_TMIN,VAR_TMAX,load_por_csv,build_valid_por_masks
from twx.utils import StatusCheck, Unbuffered
import netCDF4
from twx.infill import infill_mean_variance
from twx.db import NNRNghData
import os

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_POR = 'P_PATH_POR'
P_PATH_NNR = 'P_PATH_NNR'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_MIN_POR = 'P_MIN_POR'
P_MIN_NNGH_DAILY = 'P_MIN_NNGH_DAILY'

sys.stdout = Unbuffered(sys.stdout)

def proc_work(params,rank):
    
    status = MPI.Status()
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    #mask_por_tmin,mask_por_tmax = bcast_msg
    
    por = load_por_csv(params[P_PATH_POR])
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por,params[P_MIN_POR])
        
    stn_masks = {'tmin':mask_por_tmin,'tmax':mask_por_tmax}
    
    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD],params[P_END_YMD]))
                
    print "".join(["WORKER ",str(rank),": Received broadcast msg"])
        
    while 1:
    
        stn_id,tair_var = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)

        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                stn_mean,stn_vari = infill_mean_variance(stn_id, stn_da, stn_masks[tair_var],
                                                         tair_var, ds_nnr, nnghs=params[P_MIN_NNGH_DAILY])
            
            except Exception as e:
            
                print "".join(["ERROR: WORKER ",str(rank),": could not infill ",
                               tair_var," for ",stn_id,str(e)])
                
                stn_mean = netCDF4.default_fillvals['f8']
                stn_vari = netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id,tair_var,stn_mean,stn_vari), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]),mode="r+")
    var_meantmin = stn_da.add_stn_variable(MEAN_TMIN,MEAN_TMIN,"C",'f8')
    var_meantmax = stn_da.add_stn_variable(MEAN_TMAX,MEAN_TMAX,"C",'f8')
    var_vartmin = stn_da.add_stn_variable(VAR_TMIN,VAR_TMIN,"C**2",'f8')
    var_vartmax = stn_da.add_stn_variable(VAR_TMAX,VAR_TMAX,"C**2",'f8')
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    mask_por_tmin,mask_por_tmax = bcast_msg
    stn_ids_tmin,stn_ids_tmax = stn_da.stn_ids[mask_por_tmin],stn_da.stn_ids[mask_por_tmax]
    print "Writer: Received broadcast msg"
    stn_ids_uniq = np.unique(np.concatenate([stn_ids_tmin,stn_ids_tmax]))
    
    stn_idxs = {}
    for x in np.arange(stn_da.stn_ids.size):
        if stn_da.stn_ids[x] in stn_ids_uniq:
            stn_idxs[stn_da.stn_ids[x]] = x
        
    tair_varmeans = {'tmin':var_meantmin,'tmax':var_meantmax}
    tair_varvary = {'tmin':var_vartmin,'tmax':var_vartmax}
    
    ttl_infills = stn_ids_tmin.size + stn_ids_tmax.size
    
    stat_chk = StatusCheck(ttl_infills,30)
    
    while 1:
       
        stn_id,tair_var,stn_mean,stn_vari = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "WRITER: Finished"
                return 0
        else:
            
            stnid_dim = stn_idxs[stn_id]
                        
            tair_varmeans[tair_var][stnid_dim] = stn_mean
            tair_varvary[tair_var][stnid_dim] = stn_vari
            stn_da.ds.sync()
                        
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    #Load the period-of-record datafile
    por = load_por_csv(params[P_PATH_POR])
    
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por,params[P_MIN_POR])
        
    #Extract stn_ids that have min # of observations
    stn_ids_tmin = por[STN_ID][mask_por_tmin]
    stn_ids_tmax = por[STN_ID][mask_por_tmax]
        
    #Send stn masks to all processes
    MPI.COMM_WORLD.bcast((mask_por_tmin,mask_por_tmax), root=RANK_COORD)
    
    print "COORD: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in np.unique(np.concatenate([stn_ids_tmin,stn_ids_tmax])):
        
        tair_vars = []
        if stn_id in stn_ids_tmin:
            tair_vars.append('tmin')
        if stn_id in stn_ids_tmax:
            tair_vars.append('tmax')
            
        for tair_var in tair_vars:
            
            if cnt < nwrkers:
                dest = cnt+N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                nrec+=1

            MPI.COMM_WORLD.send((stn_id,tair_var), dest=dest, tag=TAG_DOWORK)
            cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None,None), dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "COORD: done"

if __name__ == '__main__':
    
    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = os.path.join(FPATH_STNDATA, 'all', 'tair_homog_1948_2012.nc')
    params[P_PATH_POR] = os.path.join(FPATH_STNDATA, 'all', 'homog_por_1948_2012.csv')
    params[P_PATH_NNR] = os.path.join(PROJECT_ROOT, 'reanalysis_data', 'conus_subset')
    params[P_MIN_POR] = 5  #minimum period-of-record in years
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    params[P_MIN_NNGH_DAILY] = 3 #minimum number of neighboring observations per day

    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()