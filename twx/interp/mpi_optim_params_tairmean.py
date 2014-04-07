'''
A MPI driver for performing "leave one out" cross-validation of monthly Tair normals to optimize the
number of stations to use in each climate division.

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import station_data_infill,STN_ID,NEON,MASK,BAD,\
    DTYPE_INTERP
from twx.utils.status_check import status_check
from twx.db.all_create_db import dbDataset
from netCDF4 import Dataset
from twx.interp.optimize import OptimKrigBwNstns, setOptimTairParams,\
    OptimGwrNormBwNstns

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_RLIB = 'P_PATH_RLIB'

P_MAX_NNGH_DELTA = 'P_MAX_NNGH_DELTA'
P_NGH_RNG = 'P_NGH_RNG'
P_NGH_RNG_STEP = 'P_NGH_RNG_STEP'
P_VARNAME = 'P_VARNAME'
P_NEON_ECORGN = 'P_NEON_ECORGN'

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
    
    optim = OptimKrigBwNstns(params[P_PATH_DB], params[P_VARNAME])
    #optim = OptimGwrNormBwNstns(params[P_PATH_DB], params[P_VARNAME])
    
    bcast_msg = None
    MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
            
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*2, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
      
                err = optim.runXval(stn_id, params[P_NGH_RNG])
                                            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not xval ",stn_id,str(e)])
                            
            MPI.COMM_WORLD.send((stn_id,err), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0

    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids = bcast_msg
    print "Writer: Received broadcast msg"
    
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME],stn_dtype=DTYPE_INTERP)
    stn_mask = np.in1d(stn_da.stn_ids,stn_ids,True)
    stns = stn_da.stns[stn_mask]
    
    neon_ds = {}
    ttl_xval_stns = 0
    for neon in params[P_NEON_ECORGN]:
        
        stnids_neon = stns[STN_ID][stns[NEON]==neon]
        neon_ds[neon] = create_ncdf(params,stnids_neon,neon),stnids_neon
        ttl_xval_stns+=stnids_neon.size
    
    print "Writer: Output NCDF file created"
    
    stn_idxs = {}
    for x in np.arange(stns.size):
        stn_idxs[stns[STN_ID][x]] = x
            
    ttl_xvals = ttl_xval_stns# * min_ngh_wins.size
    
    stat_chk = status_check(ttl_xvals,10)
    
    while 1:
       
        stn_id,err = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                
                ######################################################
                print "Writer: Setting the optim # of nghs..."
                
                stn_da.ds.close()
                stn_da.ds = None
                stn_da = None
                
                setOptimTairParams(params[P_PATH_DB], params[P_PATH_OUT])
            
                ######################################################
                
                print "Writer: Finished"
                return 0
        else:
                        
            stn = stns[stn_idxs[stn_id]]
            ds,stnids_neon = neon_ds[stn[NEON]]
            dim2 = np.nonzero(stnids_neon==stn_id)[0][0]
                        
            ds.variables['mae'][:,:,dim2] = np.abs(err)

            ds.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
        
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME],stn_dtype=DTYPE_INTERP)
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD])) 
    stns = stn_da.stns[mask_stns]
        
    #Send stn ids to all processes
    MPI.COMM_WORLD.bcast(stns[STN_ID], root=RANK_COORD)
        
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
                    
    for neon in params[P_NEON_ECORGN]:
    
        mask_stns_xval = np.logical_and(stn_da.stns[NEON]==neon,mask_stns)
        stn_ids_xval = stn_da.stn_ids[mask_stns_xval]
    
        for stn_id in stn_ids_xval:
                
            if cnt < nwrkers:
                dest = cnt+N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                nrec+=1

            MPI.COMM_WORLD.send(stn_id, dest=dest, tag=TAG_DOWORK)
            cnt+=1
    
        print "".join(["Coord: Finished xval of neon: ",str(neon)])
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send([None]*3, dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

def create_ncdf(params,stn_ids,neon):
    
    path = params[P_PATH_OUT]
    fpath = "".join([path,"_",str(neon),".nc"])
        
    ds = dbDataset(fpath,'w')
    ds.db_create_global_attributes("Cross Validation "+params[P_VARNAME])
        
    ds.createDimension('min_nghs',params[P_NGH_RNG].size)
    ds.db_create_stnid_dimvar(stn_ids)
    
    nghs = ds.createVariable('min_nghs','i4',('min_nghs',),fill_value=False)
    nghs.long_name = "min_nghs"
    nghs.standard_name = "min_nghs"
    nghs[:] = params[P_NGH_RNG]
    
    ds.createDimension('mth',12)
    mthVar = ds.createVariable('mth','i4',('mth',),fill_value=False)
    mthVar[:] = np.arange(1,13)
    
    ds.db_create_mae_var(('mth','min_nghs','stn_id'))
                
    ds.sync()
        
    return ds

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
    
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmax.nc"
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/infill_nonhomog_20140329/xval/optimTairMean/tmax/xval_tmax_mean'   
    params[P_NGH_RNG] = build_min_ngh_windows(35, 150, 0.10)
    params[P_NGH_RNG_STEP] = .10 #in pct
    params[P_VARNAME] = 'tmax'
    
    ds = Dataset(params[P_PATH_DB])
    divs = ds.variables['neon'][:]
    params[P_NEON_ECORGN] = np.unique(divs.data[np.logical_not(divs.mask)])#np.array([2401])
    ds.close()
    ds = None
    
    if rank == RANK_COORD:        
        proc_coord(params,nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
