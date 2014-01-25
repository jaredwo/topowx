'''
A MPI driver for performing "leave one out" cross-validation of tair interpolation in interp_tair

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import station_data_infill,STN_ID,MASK,BAD,get_krigparam_varname, VARIO_NUG, VARIO_PSILL, VARIO_RNG
from twx.utils.status_check import status_check
import netCDF4
from netCDF4 import Dataset
import rpy2.robjects as robjects
from twx.interp.optimize import OptimKrigParams
r = robjects.r

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_VARNAME = 'P_VARNAME'

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
    
    optim = OptimKrigParams(params[P_PATH_DB], params[P_VARNAME])
                
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            MPI.COMM_WORLD.send([None]*4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        
        else:
            
            try:
                
                nug,psill,rng = optim.getKrigParams(stn_id)
                                
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not optim ",stn_id,str(e)])
                nug = np.ones(12)*netCDF4.default_fillvals['f8']
                psill = np.ones(12)*netCDF4.default_fillvals['f8']
                rng = np.ones(12)*netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id,nug,psill,rng), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD])) 
    nstns = np.sum(mask_stns)
    stn_ids = stn_da.stn_ids
    stn_da.ds.close()
    stn_da = None
        
    ds = Dataset(params[P_PATH_DB],'r+')
    
    dsvars = {}
    for mth in np.arange(1,13):
        dsvars[get_krigparam_varname(mth, VARIO_NUG)] = None
        dsvars[get_krigparam_varname(mth, VARIO_PSILL)] = None
        dsvars[get_krigparam_varname(mth, VARIO_RNG)] = None
    
    for avarname in dsvars.keys():
    
        if avarname not in ds.variables.keys():
                
            avar = ds.createVariable(avarname,'f8',('stn_id',),fill_value=netCDF4.default_fillvals['f8'])
            avar.long_name = " ".join(['variogram',avarname])
            avar.units = 'NA'
                
        else:
                
            avar = ds.variables[avarname]
            avar[:] = avar._FillValue
            ds.sync()
        
        dsvars[avarname] = avar
    
    stat_chk = status_check(nstns, 250)    
    while 1:
       
        stn_id,nug,psill,rng = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            x = np.nonzero(stn_ids==stn_id)[0][0]
            
            for mth in np.arange(1,13):
                
                dsvars[get_krigparam_varname(mth, VARIO_NUG)][x] = nug[mth-1]
                dsvars[get_krigparam_varname(mth, VARIO_PSILL)][x] = psill[mth-1]
                dsvars[get_krigparam_varname(mth, VARIO_RNG)][x] = rng[mth-1]

            ds.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD])) 
    stns = stn_da.stns[mask_stns]
                    
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

def build_min_ngh_windows(rng_min,rng_max,pct_step):
    
    min_nghs = []
    n = rng_min
    
    while n <= rng_max:
        min_nghs.append(n)
        n = n + np.round(pct_step*n)
    
    return np.array(min_nghs)

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/serial_nolst/serial_tmin.nc"
    params[P_VARNAME] = 'tmin'

    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params, nsize-N_NON_WRKRS)
    else:
        proc_work(params, rank)

    MPI.COMM_WORLD.Barrier()
