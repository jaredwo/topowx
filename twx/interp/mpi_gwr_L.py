'''
A MPI driver for calculating the geographically weighted regression L matrix as described in Equation 16:

Leung, Y., Mei, C.-L., & Zhang, W.-X. (2000). 
Statistical tests for spatial nonstationarity based on the geographically weighted regression model. 
Environment and Planning A, 32(1), 9–32. doi:10.1068/a3162

This is used for estimating prediction confidence intervals.

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from db.station_data import station_data_infill,STN_ID,LON,LAT,ELEV
from interp.station_select import station_select
from utils.status_check import status_check

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'

P_NNGH = 'P_NNGH'
P_VARNAME = 'P_VARNAME'

RM_STN_IDS = np.array(['RAWS_NLIM','RAWS_OKEE','RAWS_NHCA'])

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
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    min_ngh = params[P_NNGH]
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids = bcast_msg
    
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stn_ids,True)]
    
    stn_idxs = {}
    for x in np.arange(stn_ids.size):
        stn_idxs[stn_ids[x]] = x
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*3, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            stn_slct = station_select(stns,min_ngh,min_ngh+10)
            stn = stns[stn_idxs[stn_id]]
        
            ngh_stns,wgts,rad = stn_slct.get_interp_stns(stn[LAT], stn[LON])
            stn_mask = np.in1d(stns[STN_ID],ngh_stns[STN_ID],assume_unique=True)
            
            #Calculate row of L
            x = np.array([1.,stn[LON],stn[LAT],stn[ELEV]])
            x.shape = (x.shape[0],1)
            X = np.column_stack((np.ones(ngh_stns.size),ngh_stns[LON],ngh_stns[LAT],ngh_stns[ELEV]))
            W = np.diag(wgts)
            Y = ngh_stns['MEAN_OBS']
            Y.shape = (Y.shape[0],1)
            Lr = np.dot(np.dot(np.dot(np.transpose(x),np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))),np.transpose(X)),W)
            
            MPI.COMM_WORLD.send((stn_id,stn_mask,Lr), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids = bcast_msg
    
    stn_idxs = {}
    for x in np.arange(stn_ids.size):
        stn_idxs[stn_ids[x]] = x
    
    stat_chk = status_check(stn_da.stns.size,100)
    
    L = np.zeros((stn_ids.size,stn_ids.size))
    Y = stn_da.stns['MEAN_OBS'][np.in1d(stn_da.stn_ids, stn_ids,True)]
    Y.shape = (Y.shape[0],1)

    while 1:
       
        stn_id,stn_mask,Lr = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                np.save("".join([params[P_PATH_OUT],"L","_",params[P_VARNAME]]), L)
                np.save("".join([params[P_PATH_OUT],"Y","_",params[P_VARNAME]]), Y)
                print "Writer: Finished"
                return 0
        else:
            L[stn_idxs[stn_id],stn_mask] = np.ravel(Lr)
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    
    #U.S. only mask
    mask_us = np.logical_and(np.char.find(stn_da.stns[STN_ID],'GHCN_CA')==-1,np.char.find(stn_da.stns[STN_ID],'GHCN_MX')==-1)
    mask_mt = np.logical_and(np.logical_and(stn_da.stns[LON] >= -119,stn_da.stns[LON] <= -101),np.logical_and(stn_da.stns[LAT] >= 42,stn_da.stns[LAT] <= 52))
    mask_rm_stns = np.logical_not(np.in1d(stn_da.stns[STN_ID],RM_STN_IDS,assume_unique=True))

    stn_ids = stn_da.stns[STN_ID][np.logical_and(mask_rm_stns,mask_mt)]
    MPI.COMM_WORLD.bcast(stn_ids, root=RANK_COORD)
    
    cnt = 0
    nrec = 0
    for stn_id in stn_ids:
                
        if cnt < nwrkers:
            dest = cnt+N_NON_WRKRS
        else:
            dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            nrec+=1

        MPI.COMM_WORLD.send(stn_id, dest=dest, tag=TAG_DOWORK)
        cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send(stn_id, dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/infill/infill_tair/infill_tmax.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/gwr_L/'
    params[P_NNGH] = 53
    params[P_VARNAME] = 'tmax' 
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
