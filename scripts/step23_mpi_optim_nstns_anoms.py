'''
MPI script for running a leave-one-out cross validation
of daily temperature anomalies interpolated using 
geographically weighted regression. Once cross validation
is complete, the script uses the mean absolute error of
the cross validation results to set the optimal bandwidth
for the local number of station to be used for point interpolation
in each U.S. climate division.

Must be run using mpiexec or mpirun.
'''

from mpi4py import MPI
from netCDF4 import Dataset
from twx.db import StationSerialDataDb, STN_ID, CLIMDIV, MASK, BAD
from twx.interp import XvalTairAnom, build_nstn_bandwidths, \
    create_climdiv_optim_nstns_db, set_optim_nstns_tair_anom
from twx.utils import StatusCheck, Unbuffered, TwxConfig
import argparse
import netCDF4
import numpy as np
import os
import sys

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

sys.stdout = Unbuffered(sys.stdout)

def proc_work(fpath_stndb, elem, ngh_rng, rank):
    
    status = MPI.Status()
    
    optim = XvalTairAnom(fpath_stndb, elem)
        
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    print "".join(["WORKER ", str(rank), ": Received broadcast msg"])
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD, tag=MPI.ANY_TAG,
                                     status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None] * 4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ", str(rank), ": Finished"]) 
            return 0
        else:
            
            try:
                
                bias, mae, r2 = optim.run_xval(stn_id, ngh_rng)
                                            
            except Exception as e:
            
                print "".join(["ERROR: WORKER ", str(rank), ": could not xval ",
                               stn_id, "...", str(e)])
                
                mae = np.ones((ngh_rng.size, 12)) * netCDF4.default_fillvals['f8']
                bias = np.ones((ngh_rng.size, 12)) * netCDF4.default_fillvals['f8']
                r2 = np.ones((ngh_rng.size, 12)) * netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id, mae, bias, r2), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(fpath_stndb, elem, climdivs, ngh_rng, path_out_optim, nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids = bcast_msg
    print "WRITER: Received broadcast msg"
    
    stn_da = StationSerialDataDb(fpath_stndb, elem, mode="r+")
    stn_mask = np.in1d(stn_da.stn_ids, stn_ids, True)
    stns = stn_da.stns[stn_mask]
            
    climdiv_ds = {}
    ttl_xval_stns = 0
    for climdiv in climdivs:
        
        stnids_climdiv = stns[STN_ID][stns[CLIMDIV] == climdiv]
        
        a_ds = create_climdiv_optim_nstns_db(path_out_optim, elem,
                                             stnids_climdiv, ngh_rng, climdiv)
        climdiv_ds[climdiv] = a_ds, stnids_climdiv
        
        ttl_xval_stns += stnids_climdiv.size
    
    print "WRITER: Output NCDF files created"
        
    stn_idxs = {}
    for x in np.arange(stns.size):
        stn_idxs[stns[STN_ID][x]] = x
    
    min_ngh_wins = ngh_rng
    ngh_idxs = {}
    for x in np.arange(min_ngh_wins.size):
        ngh_idxs[min_ngh_wins[x]] = x
            
    ttl_xvals = ttl_xval_stns
    
    stat_chk = StatusCheck(ttl_xvals, 250)
    
    while 1:
       
        stn_id, mae, bias, r2 = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,
                                                    tag=MPI.ANY_TAG, status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:
                
                #######################################################
                print "WRITER: Setting the optim # of nghs..."
                                
                set_optim_nstns_tair_anom(stn_da, path_out_optim)
            
                ######################################################
                
                print "WRITER: Finished"
                return 0
                
        else:
            
            stn = stns[stn_idxs[stn_id]]
            ds, stnids_climdiv = climdiv_ds[stn[CLIMDIV]]
            dim2 = np.nonzero(stnids_climdiv == stn_id)[0][0]         
            ds.variables['mae'][:, :, dim2] = mae
            ds.sync()
            
            stat_chk.increment()
                
def proc_coord(fpath_stndb, elem, climdivs, nwrkers):
    
    stn_da = StationSerialDataDb(fpath_stndb, elem)
    # Only run xval optimization for stations within mask and that are not marked as bad
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[MASK]),
                               np.isnan(stn_da.stns[BAD])) 
    stns = stn_da.stns[mask_stns]
        
    # Send stn ids to all processes
    MPI.COMM_WORLD.bcast(stns[STN_ID], root=RANK_COORD)
        
    print "COORD: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
                    
    for climdiv in climdivs:
    
        mask_stns_xval = np.logical_and(stn_da.stns[CLIMDIV] == climdiv, mask_stns)
        stn_ids_xval = stn_da.stn_ids[mask_stns_xval]
    
        for stn_id in stn_ids_xval:
                
            if cnt < nwrkers:
                dest = cnt + N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                nrec += 1

            MPI.COMM_WORLD.send(stn_id, dest=dest, tag=TAG_DOWORK)
            cnt += 1
    
        print "".join(["COORD: Finished xval of climate division: ", str(climdiv)])
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send(None, dest=w + N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "COORD: done"


if __name__ == '__main__':
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    # Run for Tmin or Tmax
    parser = argparse.ArgumentParser()
    parser.add_argument("elem",
                        help="name of observation element (e.g.-tmin)")
    args = parser.parse_args()
    elem = args.elem
    
    if elem == 'tmin':
        fpath_stndb = twx_cfg.fpath_stndata_nc_serial_tmin
    elif elem == 'tmax':
        fpath_stndb = twx_cfg.fpath_stndata_nc_serial_tmax
    else:
        raise ValueError("Unrecognized element: " + elem)
        
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()
    
    print "Process %d of %d: element is %s" % (rank, nsize, elem)
    
    # out path for optimization files
    path_out_optim = twx_cfg.path_interp_optim_anoms
    
    # Neighbor bandwidths over which to run cross validation
    ngh_rng = build_nstn_bandwidths(35, 150, 0.10)
    
    # Run for all climate divisions
    ds = Dataset(fpath_stndb)
    divs = ds.variables[CLIMDIV][:]
    climdivs = np.unique(divs.data[np.logical_not(divs.mask)])
    ds.close()
    ds = None
    
    if rank == RANK_COORD:        
        proc_coord(fpath_stndb, elem, climdivs, nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(fpath_stndb, elem, climdivs, ngh_rng, path_out_optim,
                   nsize - N_NON_WRKRS)
    else:
        proc_work(fpath_stndb, elem, ngh_rng, rank)

    MPI.COMM_WORLD.Barrier()
