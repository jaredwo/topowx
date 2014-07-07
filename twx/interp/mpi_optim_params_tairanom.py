'''
A MPI driver for performing "leave one out" cross-validation of tair interpolation in interp_tair

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db import StationSerialDataDb,STN_ID,CLIMDIV,MASK,BAD,dbDataset
from twx.utils import StatusCheck, Unbuffered
import netCDF4
from netCDF4 import Dataset
from twx.interp.optimize import XvalTairAnom, setOptimTairAnomParams,build_nstn_bandwidths
import os

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'

P_NGH_RNG = 'P_NGH_RNG'
P_VARNAME = 'P_VARNAME'
P_CLIMDIVS = 'P_CLIMDIVS'

sys.stdout=Unbuffered(sys.stdout)

def proc_work(params,rank):
    
    status = MPI.Status()
    
    optim = XvalTairAnom(params[P_PATH_DB], params[P_VARNAME])
        
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                bias,mae,r2 = optim.run_xval(stn_id, params[P_NGH_RNG])
                                            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not xval ",stn_id,"...",str(e)])
                
                mae = np.ones((params[P_NGH_RNG].size,12))*netCDF4.default_fillvals['f8']
                bias = np.ones((params[P_NGH_RNG].size,12))*netCDF4.default_fillvals['f8']
                r2 = np.ones((params[P_NGH_RNG].size,12))*netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id,mae,bias,r2), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids = bcast_msg
    print "Writer: Received broadcast msg"
    
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
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
    
    min_ngh_wins = params[P_NGH_RNG]
    ngh_idxs = {}
    for x in np.arange(min_ngh_wins.size):
        ngh_idxs[min_ngh_wins[x]] = x
            
    ttl_xvals = ttl_xval_stns# * min_ngh_wins.size
    
    stat_chk = StatusCheck(ttl_xvals,250)
    
    while 1:
       
        stn_id,mae,bias,r2 = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                #######################################################
                print "Writer: Setting the optim # of nghs..."
                
                stn_da.ds.close()
                stn_da.ds = None
                stn_da = None
                
                setOptimTairAnomParams(params[P_PATH_DB], params[P_PATH_OUT])
            
                #######################################################
                
                
                print "Writer: Finished"
                return 0
        else:
                        
            stn = stns[stn_idxs[stn_id]]
            ds,stnids_neon = neon_ds[stn[NEON]]
            x = np.nonzero(stnids_neon==stn_id)[0][0]
                        
            ds.variables['mae'][:,:,x] = mae
            ds.variables['bias'][:,:,x] = bias
            ds.variables['r2'][:,:,x] = r2
            ds.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    #Only run xval optimization for stations within mask and that are not marked as bad
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD])) 
    stns = stn_da.stns[mask_stns]
        
    #Send stn ids to all processes
    MPI.COMM_WORLD.bcast(stns[STN_ID], root=RANK_COORD)
        
    print "COORD: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
                    
    for climdiv in params[P_CLIMDIVS]:
    
        mask_stns_xval = np.logical_and(stn_da.stns[CLIMDIV]==climdiv,mask_stns)
        stn_ids_xval = stn_da.stn_ids[mask_stns_xval]
    
        for stn_id in stn_ids_xval:
                
            if cnt < nwrkers:
                dest = cnt+N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                nrec+=1

            MPI.COMM_WORLD.send(stn_id, dest=dest, tag=TAG_DOWORK)
            cnt+=1
    
        print "".join(["COORD: Finished xval of climate division: ",str(climdiv)])
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send(None, dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "COORD: done"

def create_ncdf(params,stn_ids,neon):
    
    
    path = params[P_PATH_OUT]
    fpath = "".join([path,"_",str(neon),".nc"])
        
    ds = dbDataset(fpath,'w')
    ds.db_create_global_attributes("Cross Validation "+params[P_VARNAME])
    
    min_ngh_wins = params[P_NGH_RNG]
    
    ds.createDimension('min_nghs',min_ngh_wins.size)
    ds.db_create_stnid_dimvar(stn_ids)
    
    nghs = ds.createVariable('min_nghs','i4',('min_nghs',),fill_value=False)
    nghs.long_name = "min_nghs"
    nghs.standard_name = "min_nghs"
    nghs[:] = min_ngh_wins
    
    ds.createDimension('mth',12)
    mthVar = ds.createVariable('mth','i4',('mth',),fill_value=False)
    mthVar[:] = np.arange(1,13)
    
    ds.db_create_mae_var(('min_nghs','mth','stn_id'))
    ds.db_create_bias_var(('min_nghs','mth','stn_id'))
        
    r2 = ds.createVariable('r2','f8',('min_nghs','mth','stn_id'))
    r2.missing_value = netCDF4.default_fillvals['f8']
    r2.long_name = "r-squared"
            
    ds.sync()
        
    return ds


if __name__ == '__main__':

    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')

    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    
    params[P_PATH_DB] = os.path.join(FPATH_STNDATA, 'infill', 'serial_tmin.nc')
    params[P_PATH_OUT] = os.path.join(FPATH_STNDATA, 'infill', 'optim_anoms')  #xval_tmin_anom' 
    params[P_NGH_RNG] = build_nstn_bandwidths(35, 150, 0.10)
    params[P_VARNAME] = 'tmin'
    
    #Run for all climate divisions
    ds = Dataset(params[P_PATH_DB])
    divs = ds.variables[CLIMDIV][:]
    params[P_CLIMDIVS] = np.unique(divs.data[np.logical_not(divs.mask)])
    ds.close()
    ds = None
    
    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
