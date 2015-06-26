'''
MPI script for running a leave-one-out cross validation
of monthly temperature normals interpolated using 
moving window regression kriging. Once cross validation
is complete, the script uses the mean absolute error of
the cross validation results to set the optimal bandwidth
for the local number of station to be used for point interpolation
in each U.S. climate division.

Must be run using mpiexec or mpirun.

Copyright 2014,2015, Jared Oyler.

This file is part of TopoWx.

TopoWx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TopoWx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.
'''

import readline
import numpy as np
from mpi4py import MPI
import sys
from twx.db import StationSerialDataDb,STN_ID,MASK,BAD,CLIMDIV
from twx.utils import StatusCheck, Unbuffered
from netCDF4 import Dataset
from twx.interp import XvalTairNorm, set_optim_nstns_tair_norm,\
build_nstn_bandwidths,create_climdiv_optim_nstns_db
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
    
    optim = XvalTairNorm(params[P_PATH_DB], params[P_VARNAME])
    
    bcast_msg = None
    MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
            
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*2, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
      
                err = optim.run_xval(stn_id, params[P_NGH_RNG])
                MPI.COMM_WORLD.send((stn_id,err), dest=RANK_WRITE, tag=TAG_DOWORK)
                                            
            except Exception as e:
            
                print "".join(["ERROR: WORKER ",str(rank),": could not xval ",stn_id,str(e)])
                            
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0

    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids = bcast_msg
    print "WRITER: Received broadcast msg"
    
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME],mode="r+")
    stn_mask = np.in1d(stn_da.stn_ids,stn_ids,True)
    stns = stn_da.stns[stn_mask]
    
    climdiv_ds = {}
    ttl_xval_stns = 0
    for climdiv in params[P_CLIMDIVS]:
        
        stnids_climdiv = stns[STN_ID][stns[CLIMDIV]==climdiv]
        
        a_ds =  create_climdiv_optim_nstns_db(params[P_PATH_OUT], params[P_VARNAME],
                                              stnids_climdiv, params[P_NGH_RNG], climdiv)
        climdiv_ds[climdiv] = a_ds,stnids_climdiv
        
        ttl_xval_stns+=stnids_climdiv.size
    
    print "WRITER: Output NCDF files created"
    
    stn_idxs = {}
    for x in np.arange(stns.size):
        stn_idxs[stns[STN_ID][x]] = x
            
    ttl_xvals = ttl_xval_stns
    
    stat_chk = StatusCheck(ttl_xvals,10)
    
    while 1:
       
        stn_id,err = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                
                ######################################################
                print "WRITER: Setting the optim # of nghs..."
                                
                set_optim_nstns_tair_norm(stn_da, params[P_PATH_OUT])
            
                ######################################################
                
                print "WRITER: Finished"
                return 0
        else:
                        
            stn = stns[stn_idxs[stn_id]]
            ds,stnids_climdiv = climdiv_ds[stn[CLIMDIV]]
            dim2 = np.nonzero(stnids_climdiv==stn_id)[0][0]         
            ds.variables['mae'][:,:,dim2] = np.abs(err)
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

if __name__ == '__main__':
    
    PROJECT_ROOT = os.getenv('TOPOWX_DATA')
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    #Run for Tmin or Tmax
    params[P_VARNAME] = 'tmax'
    params[P_PATH_DB] = os.path.join(FPATH_STNDATA, 'infill', 'serial_%s.nc'%params[P_VARNAME])
    params[P_PATH_OUT] = os.path.join(FPATH_STNDATA, 'infill', 'optim')  
    params[P_NGH_RNG] = build_nstn_bandwidths(35, 150, 0.10)
    
    #Run for all climate divisions
    ds = Dataset(params[P_PATH_DB])
    divs = ds.variables[CLIMDIV][:]
    params[P_CLIMDIVS] = np.unique(divs.data[np.logical_not(divs.mask)])
    ds.close()
    ds = None
    
    if rank == RANK_COORD:        
        proc_coord(params,nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
