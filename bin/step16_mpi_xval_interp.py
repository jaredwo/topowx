'''
MPI script for running a leave-one-out cross validation
of interpolated temperature normals and daily temperatures
using the optimal variogram and station bandwidth parameters
set in steps 13-15.

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
from twx.db import StationSerialDataDb, STN_ID, MASK, BAD, \
    get_norm_varname
from twx.utils import StatusCheck, Unbuffered
import netCDF4
from twx.interp import XvalTairOverall
import os
from twx.db import create_quick_db

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_VARNAME = 'P_VARNAME'

DB_VARIABLES = {'tmin':[('tmin', 'f4', netCDF4.default_fillvals['f4'], 'minimum air temperature', 'C')],
                'tmax':[('tmax', 'f4', netCDF4.default_fillvals['f4'], 'maximum air temperature', 'C')]}

sys.stdout = Unbuffered(sys.stdout)

def proc_work(params, rank):
    
    status = MPI.Status()
    
    xval = XvalTairOverall(params[P_PATH_DB], params[P_VARNAME])
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
            
                print "".join(["ERROR: Worker ", str(rank), ": could not xval ", stn_id, str(e)])
                
                tair_daily = np.ones(ndays) * netCDF4.default_fillvals['f8']
                tair_norms = np.ones(12) * netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id, tair_daily, tair_norms), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params, nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    stn_ids = stn_da.stn_ids
    stns = stn_da.stns
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]), np.isnan(stn_da.stns[BAD]))    
    days = stn_da.days
    stn_da.ds.close()
    stn_da = None
    
    print "WRITER: Creating output station netCDF database..."
    
    create_quick_db(params[P_PATH_OUT], stns, days, DB_VARIABLES[params[P_VARNAME]])
    stnda_out = StationSerialDataDb(params[P_PATH_OUT], params[P_VARNAME], mode='r+')
    
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
       
        stn_id, tair_daily, tair_norms = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:
                print "WRITER: Finished"
                return 0
        else:
            
            x = np.nonzero(stn_ids == stn_id)[0][0]
            stnda_out.ds.variables[params[P_VARNAME]][:, x] = tair_daily
            
            for i in mths:
                stnda_out.ds.variables[mth_names[i]][x] = tair_norms[i]
            
            stnda_out.ds.sync()
            
            stat_chk.increment()
                
def proc_coord(params, nwrkers):
    
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]), np.isnan(stn_da.stns[BAD]))    
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
    #Path to database file where interpolated normals and daily values will
    #be output.
    params[P_PATH_OUT] = os.path.join(FPATH_STNDATA, 'infill', 'xval_interp_%s.nc'%params[P_VARNAME])

        
    if rank == RANK_COORD:        
        proc_coord(params, nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params, nsize - N_NON_WRKRS)
    else:
        proc_work(params, rank)

    MPI.COMM_WORLD.Barrier()
