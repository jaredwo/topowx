'''
MPI script for running a leave-one-out cross validation
of interpolated temperature normals and daily temperatures
using the optimal variogram and station bandwidth parameters
set in steps 21-23.

Must be run using mpiexec or mpirun.
'''

from mpi4py import MPI
from twx.db import StationSerialDataDb, STN_ID, MASK, BAD, \
    get_norm_varname, create_quick_db
from twx.interp import XvalTairOverall
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

DB_VARIABLES = {'tmin':[('tmin', 'f4', netCDF4.default_fillvals['f4'],
                         'minimum air temperature', 'C')],
                'tmax':[('tmax', 'f4', netCDF4.default_fillvals['f4'],
                         'maximum air temperature', 'C')]}

sys.stdout = Unbuffered(sys.stdout)

def proc_work(fpath_stndb, elem, rank):
    
    status = MPI.Status()
    
    xval = XvalTairOverall(fpath_stndb, elem)
    ndays = xval.stn_da.days.size
        
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD, tag=MPI.ANY_TAG, status=status)
        
        if status.tag == TAG_STOPWORK:
            
            MPI.COMM_WORLD.send([None] * 3, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ", str(rank), ": Finished"]) 
            return 0
        
        else:
            
            try:
                
                tair_daily, tair_norms, tair_se = xval.run_interp(stn_id)
                                            
            except Exception as e:
            
                print "".join(["ERROR: Worker ", str(rank),
                               ": could not xval ", stn_id, str(e)])
                
                tair_daily = np.ones(ndays) * netCDF4.default_fillvals['f4']
                tair_norms = np.ones(12) * netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id, tair_daily, tair_norms),
                                dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(fpath_stndb, elem, fpath_out, nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    stn_da = StationSerialDataDb(fpath_stndb, elem)
    stn_ids = stn_da.stn_ids
    stns = stn_da.stns
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),
                              np.isnan(stn_da.stns[BAD]))    
    days = stn_da.days
    stn_da.ds.close()
    stn_da = None
    
    print "WRITER: Creating output station netCDF database..."
    
    create_quick_db(fpath_out, stns, days, DB_VARIABLES[elem])
    stnda_out = StationSerialDataDb(fpath_out, elem, mode='r+')
    
    mth_names = []
    for mth in np.arange(1, 13):
        
        norm_var_name = get_norm_varname(mth)
        stnda_out.add_stn_variable(norm_var_name, '', units='C',
                                   dtype='f8', fill_value=netCDF4.default_fillvals['f8'])
        mth_names.append(norm_var_name)
    
    stnda_out.ds.sync()
    
    print "WRITER: Output station netCDF database created."
    
    mths = np.arange(12)
        
    stat_chk = StatusCheck(np.sum(stn_mask), 50)
    
    while 1:
       
        stn_id, tair_daily, tair_norms = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,
                                                             tag=MPI.ANY_TAG,
                                                             status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:
                print "WRITER: Finished"
                return 0
        else:
            
            x = np.nonzero(stn_ids == stn_id)[0][0]
            stnda_out.ds.variables[elem][:, x] = tair_daily
            
            for i in mths:
                stnda_out.ds.variables[mth_names[i]][x] = tair_norms[i]
            
            stnda_out.ds.sync()
            
            stat_chk.increment()
                
def proc_coord(fpath_stndb, elem, nwrkers):
    
    stn_da = StationSerialDataDb(fpath_stndb, elem)
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),
                              np.isnan(stn_da.stns[BAD]))    
    stns = stn_da.stns[stn_mask]
            
    print "COORD: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
                
    for stn_id in stns[STN_ID]:
            
        if cnt < nwrkers:
            dest = cnt + N_NON_WRKRS
        else:
            dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            nrec += 1

        MPI.COMM_WORLD.send(stn_id, dest=dest, tag=TAG_DOWORK)
        cnt += 1
                        
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
        fpath_out = twx_cfg.fpath_xval_interp_nc_tmin
    elif elem == 'tmax':
        fpath_stndb = twx_cfg.fpath_stndata_nc_serial_tmax
        fpath_out = twx_cfg.fpath_xval_interp_nc_tmax
    else:
        raise ValueError("Unrecognized element: " + elem)
        
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()
    
    print "Process %d of %d: element is %s" % (rank, nsize, elem)
            
    if rank == RANK_COORD:        
        proc_coord(fpath_stndb, elem, nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(fpath_stndb, elem, fpath_out, nsize - N_NON_WRKRS)
    else:
        proc_work(fpath_stndb, elem, rank)

    MPI.COMM_WORLD.Barrier()
