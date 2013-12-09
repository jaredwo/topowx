'''
A MPI driver for validating the prcp infilling methods of infill_normal. A random set of
HCN stations spread across different ecoregions have all but a specified # 
of years of data artificially set to missing. Infilled values are then compared to observed values. 

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import station_data_ncdb
from twx.utils.status_check import status_check
from netCDF4 import Dataset
import netCDF4
from twx.infill.infill_normals import build_mth_masks,infill_prcp_norm,MTH_BUFFER
from twx.db.all_create_db import dbDataset
from random_xval_stations import build_xval_masks,get_random_xval_stns

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_POR = 'P_PATH_POR'
P_PATH_NEON = 'P_PATH_NEON'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'

P_MIN_POR_PCT = 'P_MIN_POR_PCT'
P_STNS_PER_RGN = 'P_STNS_PER_RGN'
P_NYRS_MOD = 'P_NYRS_MOD'
P_EXCLUDE_STNIDS = 'P_EXCLUDE_STNIDS'
P_INCLUDE_STNIDS = 'P_INCLUDE_STNIDS'
P_PATH_GHCN_STNS = 'P_PATH_GHCN_STNS'

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
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids,xval_masks_prcp = bcast_msg
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:

            try:
                
                x = np.nonzero(stn_ids==stn_id)[0][0]
                xval_mask = xval_masks_prcp[x]
                
                obs_prcp = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]),'prcp')[0],dtype=np.float64)
                
                prcp_norm,obs_norm = infill_prcp_norm(stn_id, stn_da, days, mth_masks, mthbuf_masks, use_prcp_only=False, prcp_mask=xval_mask)
                
                obs_xvalmean = np.mean(obs_prcp[xval_mask],dtype=np.float64)
                fit_xvalmean = np.mean(prcp_norm[xval_mask],dtype=np.float64)
                
                perr = (fit_xvalmean - obs_xvalmean)/obs_xvalmean * 100
                    
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not infill prcp for ",stn_id," ",str(e)])
                MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                continue
            
            MPI.COMM_WORLD.send((stn_id,fit_xvalmean,obs_xvalmean,perr), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids,xval_masks_prcp = bcast_msg
    print "Writer: Received broadcast msg"
    
    ds = create_ncdf(params,stn_ids)
    print "Writer: Output NCDF file created"
    
    stn_idxs = {}
    for x in np.arange(stn_ids.size):
        stn_idxs[stn_ids[x]] = x
    
    ttl_infills = stn_ids.size

    stat_chk = status_check(ttl_infills,10)
        
    while 1:
       
        stn_id,fit_xvalmean,obs_xvalmean,perr = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                print "Writer: Finished"
                return 0
        else:
            
            dim1 = stn_idxs[stn_id]
            
            ds.variables['fit_mean'][dim1] = fit_xvalmean
            ds.variables['obs_mean'][dim1] = obs_xvalmean
            ds.variables['perr'][dim1] = perr
            ds.sync()
            
            print "|".join(["WRITER",stn_id,'%.4f'%(perr,),str(fit_xvalmean),str(obs_xvalmean)])
            stat_chk.increment()
            
                
def proc_coord(params,nwrkers):
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    if params[P_INCLUDE_STNIDS] is None:
        
        fnl_stn_ids = get_random_xval_stns(stn_da, params[P_MIN_POR_PCT], params[P_STNS_PER_RGN], params[P_PATH_POR], params[P_PATH_NEON],
                             params[P_PATH_GHCN_STNS], params[P_EXCLUDE_STNIDS])
    
    else:
        
        fnl_stn_ids = params[P_INCLUDE_STNIDS]
    
    
    xval_masks_prcp = build_xval_masks(fnl_stn_ids, params[P_NYRS_MOD], stn_da)
    
    #Send stn ids and masks to all processes
    MPI.COMM_WORLD.bcast((fnl_stn_ids,xval_masks_prcp), root=RANK_COORD)
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in fnl_stn_ids:
                
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

def create_ncdf(params,stn_ids):
    
    path = params[P_PATH_OUT]
    
    ds = dbDataset(path,'w')
            
    ds.db_create_global_attributes("PRCP NORM IMPUTE MEAN TESTING")
    
    ds.db_create_stnid_dimvar(stn_ids)
    
    perr = ds.createVariable('perr','f4',('stn_id',))
    perr.missing_value = netCDF4.default_fillvals['f4']
    
    perr = ds.createVariable('obs_mean','f4',('stn_id',))
    perr.missing_value = netCDF4.default_fillvals['f4']
    
    perr = ds.createVariable('fit_mean','f4',('stn_id',))
    perr.missing_value = netCDF4.default_fillvals['f4']
    
    ds.sync()
    
    return ds

if __name__ == '__main__':
        
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/xval_infill_prcp_norm4.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por.csv'
    params[P_PATH_NEON] = '/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.nc'
    params[P_PATH_GHCN_STNS] = '/projects/daymet2/station_data/ghcn/ghcnd-stations.txt'
    params[P_MIN_POR_PCT] = 0.90
    params[P_STNS_PER_RGN] = 10
    params[P_NYRS_MOD] = 5
    params[P_START_YMD] = None #19480101
    params[P_END_YMD] = None #20111231
    
    if rank == RANK_COORD:
        
        params[P_EXCLUDE_STNIDS] = np.array([])
        ds = Dataset('/projects/daymet2/station_data/infill/xval_infill_po.nc')
        params[P_INCLUDE_STNIDS] = np.array(ds.variables['stn_id'][:],dtype="<S16")
        ds.close()
        
        proc_coord(params, nsize-N_NON_WRKRS)
        
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
