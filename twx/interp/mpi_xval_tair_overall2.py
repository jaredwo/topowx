'''
A MPI driver for performing "leave one out" cross-validation of tair interpolation in interp_tair

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import station_data_infill,STN_ID,MASK,BAD,\
    get_norm_varname
from twx.utils.status_check import status_check
import netCDF4
from netCDF4 import Dataset
import rpy2.robjects as robjects
from twx.interp.optimize import XvalTairOverall
r = robjects.r

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_WRITEDB = 'P_PATH_WRITEDB'

P_PATH_OUT = 'P_PATH_OUT'
P_PATH_DB_XVAL = 'P_PATH_DB_XVAL'
P_PATH_RMSTNS = 'P_PATH_RMSTNS'
P_PATH_PARAMS_MEAN = 'P_PATH_PARAMS_MEAN'

P_MAX_NNGH_DELTA = 'P_MAX_NNGH_DELTA'
P_NGH_RNG = 'P_NGH_RNG'
P_NGH_RNG_STEP = 'P_NGH_RNG_STEP'
P_VARNAME = 'P_VARNAME'
P_VARNAME_XVAL = 'P_VARNAME_XVAL'
P_NEON_ECORGN = 'P_NEON_ECORGN'
P_GWP_RNG = 'P_GWP_RNG'
P_RESTART_NNGH = 'P_RESTART_NNGH'


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

def proc_work(params,rank):
    
    status = MPI.Status()
    
    xval = XvalTairOverall(params[P_PATH_DB], params[P_VARNAME])
    ndays = xval.stn_da.days.size
        
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*3, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                tair_daily,tair_norms,tair_se = xval.run_interp(stn_id)
                                            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not xval ",stn_id,str(e)])
                
                tair_daily = np.ones(ndays)*netCDF4.default_fillvals['f8']
                tair_norms = np.ones(12)*netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id,tair_daily,tair_norms), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    stn_ids = stn_da.stn_ids
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD]))    
    stns = stn_da.stns[stn_mask]
    stn_da.ds.close()
    stn_da = None
    ds = Dataset(params[P_PATH_WRITEDB],'r+')
    
    mths = np.arange(12)
    
    mthNames = []
    for mth in mths:
        mthNames.append(get_norm_varname(mth+1))
    
    stat_chk = status_check(stns.size,250)
    
    while 1:
       
        stn_id,tair_daily,tair_norms = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            x = np.nonzero(stn_ids==stn_id)[0][0]
            ds.variables[params[P_VARNAME]][:,x] = tair_daily
            
            for i in mths:
                ds.variables[mthNames[i]] = tair_norms[i]
            
            ds.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD]))    
    stns = stn_da.stns[stn_mask]
            
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
                
    for stn_id in stns[STN_ID]:
            
        if cnt < nwrkers:
            dest = cnt+N_NON_WRKRS
        else:
            dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            nrec+=1

        MPI.COMM_WORLD.send(stn_id, dest=dest, tag=TAG_DOWORK)
        cnt+=1
                        
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send(None, dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc"
    params[P_PATH_WRITEDB] = "/projects/daymet2/station_data/infill/serial_fnl/xval_tmax.nc"
    params[P_VARNAME] = 'tmax'
        
    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
