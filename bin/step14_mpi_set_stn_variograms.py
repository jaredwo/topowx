'''
MPI script for setting moving window regression kriging variogram
parameters at each station location based on the U.S. climate
division optimal station bandwidths from step13_mpi_optim_nstns_norms.py.
Adds an exponential variogram nugget, partial sill,
and range station attribute for each month to the serially-complete
station database.

Must be run using mpiexec or mpirun.
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db import StationSerialDataDb,\
STN_ID,MASK,BAD,get_krigparam_varname, VARIO_NUG, VARIO_PSILL, VARIO_RNG
from twx.utils import StatusCheck, Unbuffered
import netCDF4
from twx.interp import OptimKrigParams
import os

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_VARNAME = 'P_VARNAME'

sys.stdout=Unbuffered(sys.stdout)

def proc_work(params,rank):
    
    status = MPI.Status()
    
    kparams = OptimKrigParams(params[P_PATH_DB], params[P_VARNAME])
                
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            MPI.COMM_WORLD.send([None]*4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        
        else:
            
            try:
                
                nug,psill,rng = kparams.get_krig_params(stn_id)
                                
            except Exception as e:
            
                print "".join(["ERROR: WORKER ",str(rank),": could not get krig params for ",stn_id,str(e)])
                nug = np.ones(12)*netCDF4.default_fillvals['f8']
                psill = np.ones(12)*netCDF4.default_fillvals['f8']
                rng = np.ones(12)*netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id,nug,psill,rng), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME],mode="r+")
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD])) 
    nstns = np.sum(mask_stns)

    dsvars = {}
    for mth in np.arange(1,13):
                
        vname_nug = get_krigparam_varname(mth, VARIO_NUG)
        vname_psill = get_krigparam_varname(mth, VARIO_PSILL)
        vname_rng = get_krigparam_varname(mth, VARIO_RNG)
        
        dsvars[vname_nug] = stn_da.add_stn_variable(vname_nug, vname_nug, "C**2", 'f8')
        dsvars[vname_psill] = stn_da.add_stn_variable(vname_psill, vname_nug, "C**2", 'f8')
        dsvars[vname_rng] = stn_da.add_stn_variable(vname_rng, vname_nug, "km", 'f8')
        
    stat_chk = StatusCheck(nstns, 250)    
    
    while 1:
       
        stn_id,nug,psill,rng = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "WRITER: Finished"
                return 0
        else:
            
            x = stn_da.stn_idxs[stn_id]
            
            for mth in np.arange(1,13):
                
                dsvars[get_krigparam_varname(mth, VARIO_NUG)][x] = nug[mth-1]
                dsvars[get_krigparam_varname(mth, VARIO_PSILL)][x] = psill[mth-1]
                dsvars[get_krigparam_varname(mth, VARIO_RNG)][x] = rng[mth-1]
            
            stn_da.ds.sync()
                        
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    #Only set kriging params for stations within mask and that are not marked as bad
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD])) 
    stns = stn_da.stns[mask_stns]
                    
    print "COORD: Done initialization. Starting to send work."
    
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
        
    print "COORD: done"

if __name__ == '__main__':
    
    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = os.path.join(FPATH_STNDATA, 'infill', 'serial_tmin.nc')
    params[P_VARNAME] = 'tmin'

    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params, nsize-N_NON_WRKRS)
    else:
        proc_work(params, rank)

    MPI.COMM_WORLD.Barrier()
