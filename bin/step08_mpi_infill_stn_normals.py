'''
MPI script for estimating/infilling period-of-record means
and variances of incomplete Tmin/Tmax station time series.

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
from twx.db import StationDataDb, STN_ID, load_por_csv, build_valid_por_masks
from twx.utils import StatusCheck, Unbuffered, ymdL
import netCDF4
from twx.infill import infill_mean_variance
from twx.db import NNRNghData
import os
from twx.utils.util_dates import MONTH
from twx.db.station_data import get_mean_varname, get_variance_varname
from datetime import datetime

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_POR = 'P_PATH_POR'
P_PATH_NNR = 'P_PATH_NNR'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_MIN_POR = 'P_MIN_POR'
P_MIN_NNGH_DAILY = 'P_MIN_NNGH_DAILY'

sys.stdout = Unbuffered(sys.stdout)

def proc_work(params, rank):
    
    status = MPI.Status()
    stn_da = StationDataDb(params[P_PATH_DB], (params[P_START_YMD], params[P_END_YMD]))
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    # mask_por_tmin,mask_por_tmax = bcast_msg
    
    por = load_por_csv(params[P_PATH_POR])
    mask_por_tmin, mask_por_tmax = build_valid_por_masks(por, params[P_MIN_POR])
        
    stn_masks = {'tmin':mask_por_tmin, 'tmax':mask_por_tmax}
    
    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD], params[P_END_YMD]))
    
    mth_masks = [stn_da.days[MONTH] == mth for mth in np.arange(1, 13)]
                    
    print "".join(["WORKER ", str(rank), ": Received broadcast msg"])
        
    while 1:
    
        stn_id, tair_var = MPI.COMM_WORLD.recv(source=RANK_COORD, tag=MPI.ANY_TAG, status=status)

        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None] * 4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ", str(rank), ": Finished"]) 
            return 0
        else:
            
            try:
                
                stn_mean, stn_vari = infill_mean_variance(stn_id, stn_da, stn_masks[tair_var],
                                                         tair_var, ds_nnr, nnghs=params[P_MIN_NNGH_DAILY],
                                                         day_masks=mth_masks)
            
            except Exception as e:
            
                print "".join(["ERROR: WORKER ", str(rank), ": could not infill ",
                               tair_var, " for ", stn_id, str(e)])
                
                empty = np.empty(12)
                empty.fill(netCDF4.default_fillvals['f8'])
                
                stn_mean = empty
                stn_vari = empty
            
            MPI.COMM_WORLD.send((stn_id, tair_var, stn_mean, stn_vari), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params, nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    stn_da = StationDataDb(params[P_PATH_DB], (params[P_START_YMD], params[P_END_YMD]), mode="r+")
    
    mths = np.arange(1, 13)
    
    for mth in mths:
        
        for varname in ['tmin', 'tmax']:
                    
            varname_mean = get_mean_varname(varname, mth)
            varname_vari = get_variance_varname(varname, mth)
        
            stn_da.add_stn_variable(varname_mean, varname_mean, "C", 'f8')
            stn_da.add_stn_variable(varname_vari, varname_vari, "C**2", 'f8')
    
    stn_da.ds.sync()
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    mask_por_tmin, mask_por_tmax = bcast_msg
    stn_ids_tmin, stn_ids_tmax = stn_da.stn_ids[mask_por_tmin], stn_da.stn_ids[mask_por_tmax]
    print "WRITER: Received broadcast msg"
    stn_ids_uniq = np.unique(np.concatenate([stn_ids_tmin, stn_ids_tmax]))
    
    stn_idxs = {}
    for x in np.arange(stn_da.stn_ids.size):
        if stn_da.stn_ids[x] in stn_ids_uniq:
            stn_idxs[stn_da.stn_ids[x]] = x
            
    ttl_infills = stn_ids_tmin.size + stn_ids_tmax.size
    
    stat_chk = StatusCheck(ttl_infills, 30)
    
    while 1:
       
        stn_id, tair_var, stn_mean, stn_vari = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:
                print "WRITER: Finished"
                return 0
        else:
            
            stnid_dim = stn_idxs[stn_id]
            
            for mth in mths:
                
                stn_da.ds.variables[get_mean_varname(tair_var, mth)][stnid_dim] = stn_mean[mth - 1]
                stn_da.ds.variables[get_variance_varname(tair_var, mth)][stnid_dim] = stn_vari[mth - 1]       

            stn_da.ds.sync()
                        
            stat_chk.increment()
                
def proc_coord(params, nwrkers):
    
    # Load the period-of-record datafile
    por = load_por_csv(params[P_PATH_POR])
    
    mask_por_tmin, mask_por_tmax = build_valid_por_masks(por, params[P_MIN_POR])
        
    # Extract stn_ids that have min # of observations
    stn_ids_tmin = por[STN_ID][mask_por_tmin]
    stn_ids_tmax = por[STN_ID][mask_por_tmax]
        
    # Send stn masks to all processes
    MPI.COMM_WORLD.bcast((mask_por_tmin, mask_por_tmax), root=RANK_COORD)
    
    print "COORD: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in np.unique(np.concatenate([stn_ids_tmin, stn_ids_tmax])):
        
        tair_vars = []
        if stn_id in stn_ids_tmin:
            tair_vars.append('tmin')
        if stn_id in stn_ids_tmax:
            tair_vars.append('tmax')
            
        for tair_var in tair_vars:
            
            if cnt < nwrkers:
                dest = cnt + N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                nrec += 1

            MPI.COMM_WORLD.send((stn_id, tair_var), dest=dest, tag=TAG_DOWORK)
            cnt += 1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None, None), dest=w + N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "COORD: done"

if __name__ == '__main__':
    
    PROJECT_ROOT = os.getenv('TOPOWX_DATA')
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    START_YEAR_STNDB = 1895
    END_YEAR_STNDB = 2015
    START_YEAR_POR = 1948
    END_YEAR_POR = 2014
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = os.path.join(FPATH_STNDATA, 'all', 'tair_homog_%s_%s.nc' % (START_YEAR_STNDB, END_YEAR_STNDB))
    params[P_PATH_POR] = os.path.join(FPATH_STNDATA, 'all', 'homog_por_%s_%s.csv' % (START_YEAR_POR, END_YEAR_POR))
    params[P_PATH_NNR] = os.path.join(PROJECT_ROOT, 'reanalysis_data', 'n_america_subset')
    params[P_MIN_POR] = 5  # minimum period-of-record in years
    params[P_START_YMD] = ymdL(datetime(START_YEAR_POR, 1, 1))
    params[P_END_YMD] = ymdL(datetime(END_YEAR_POR, 12, 31))
    params[P_MIN_NNGH_DAILY] = 3  # minimum number of neighboring observations per day

    
    if rank == RANK_COORD:
        proc_coord(params, nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params, nsize - N_NON_WRKRS)
    else:
        proc_work(params, rank)

    MPI.COMM_WORLD.Barrier()
