'''
MPI script for running a cross validation 
of the infill model for infilling 
missing daily Tmin/Tmax.

Must be run using mpiexec or mpirun.

Copyright 2014, Jared Oyler.

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

import numpy as np
from mpi4py import MPI
import sys
from twx.db import StationDataDb,NNRNghData,create_quick_db
from twx.utils import StatusCheck, Unbuffered
from netCDF4 import Dataset
import netCDF4
from twx.infill import XvalInfillParams,XvalInfill,load_default_xval_stnids
import os

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_NNR = 'P_PATH_NNR'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'

P_NYRS_TRAIN = 'P_NYRS_TRAIN'
P_NGH_RNG = 'P_NGH_RNG'
P_XVAL_STNIDS = 'P_XVAL_STNIDS'
P_SNOTEL_RAWS_XVAL = 'P_SNOTEL_RAWS_XVAL'

P_MIN_NNGH_DAILY = 'P_MIN_NNGH_DAILY'
P_NNGH_NNR = 'P_NNGH_NNR'
P_NNR_VARYEXPLAIN = 'P_NNR_VARYEXPLAIN'
P_FRACOBS_INIT_PCS = 'P_FRACOBS_INIT_PCS'
P_PPCA_VARYEXPLAIN = 'P_PPCA_VARYEXPLAIN'
P_CHCK_IMP_PERF = 'P_CHCK_IMP_PERF'
P_NPCS_PPCA = 'P_NPCS_PPCA'
P_VERBOSE = 'P_VERBOSE'

NETCDF_OUT_VARIABLES = [('obs_tmin', 'f4', netCDF4.default_fillvals['f4'], 'observed minimum air temperature', 'C'),
                        ('obs_tmax', 'f4', netCDF4.default_fillvals['f4'], 'observed maximum air temperature', 'C'),
                        ('infilled_tmin', 'f4', netCDF4.default_fillvals['f4'], 'infilled minimum air temperature', 'C'),
                        ('infilled_tmax', 'f4', netCDF4.default_fillvals['f4'], 'infilled maximum air temperature', 'C')]

sys.stdout=Unbuffered(sys.stdout)

def proc_work(params,rank):
    
    status = MPI.Status()
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))    
    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD],params[P_END_YMD]))
    
    infill_params = XvalInfillParams(nnr_ds=ds_nnr,
                                     min_daily_nnghs=params[P_MIN_NNGH_DAILY], 
                                     nnghs_nnr=params[P_NNGH_NNR], 
                                     max_nnr_var=params[P_NNR_VARYEXPLAIN],
                                     chk_perf=params[P_CHCK_IMP_PERF],
                                     npcs=params[P_NPCS_PPCA], 
                                     frac_obs_initnpcs=params[P_FRACOBS_INIT_PCS],
                                     ppca_varyexplain=params[P_PPCA_VARYEXPLAIN], 
                                     verbose=params[P_VERBOSE])
    
    xval_infills = {}
    xval_infills['tmin'] = XvalInfill(stn_da, 'tmin', infill_params,
                                      xval_stnids=params[P_XVAL_STNIDS],
                                      ntrain_yrs=params[P_NYRS_TRAIN])
    xval_infills['tmax'] = XvalInfill(stn_da, 'tmax', infill_params,
                                      xval_stnids=params[P_XVAL_STNIDS],
                                      ntrain_yrs=params[P_NYRS_TRAIN])
        
    print "".join(["WORKER ",str(rank),": ready to receive work."])
    
    empty = np.ones(stn_da.days.size)*np.nan
        
    while 1:
    
        stn_id,tair_var = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            MPI.COMM_WORLD.send([None]*4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ",str(rank),": Finished"]) 
            return 0
       
        else:
                
            try:
            
                obs_tair,infill_tair = xval_infills[tair_var].run_xval(stn_id)
            
            except Exception as e:
            
                print "".join(["ERROR: WORKER ",str(rank),": could not infill ",
                               tair_var," for ",stn_id,str(e)])
                
                infill_tair = empty
                obs_tair = empty
            
            MPI.COMM_WORLD.send((stn_id,tair_var,infill_tair,obs_tair), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    if params[P_XVAL_STNIDS] is None:
        xval_stnids = load_default_xval_stnids(stn_da.stn_ids)
    else:
        xval_stnids = params[P_XVAL_STNIDS]
    
    ttl_infills = xval_stnids.size * 2
    
    xval_stns = stn_da.stns[np.in1d(stn_da.stn_ids, xval_stnids, True)]
    
    create_quick_db(params[P_PATH_OUT], xval_stns, stn_da.days, NETCDF_OUT_VARIABLES)
    ds_out = Dataset(params[P_PATH_OUT],'r+')
    
    stn_idxs = {}
    for x in np.arange(xval_stnids.size):
        stn_idxs[xval_stnids[x]] = x
                    
    stat_chk = StatusCheck(ttl_infills,10)
    
    while 1:

        stn_id,tair_var,infill_tair,obs_tair = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "WRITER: Finished"
                return 0
        else:
            
            infill_tair = np.ma.masked_array(infill_tair,np.isnan(infill_tair))
            obs_tair = np.ma.masked_array(obs_tair,np.isnan(obs_tair))
            
            i = stn_idxs[stn_id]
            
            difs = infill_tair - obs_tair
            bias = np.ma.mean(difs)
            mae = np.ma.mean(np.ma.abs(difs))
            
            print "|".join(["WRITER",stn_id,tair_var,"MAE: %.2f"%(mae,),"BIAS: %.2f"%(bias,)])
            ds_out.variables["obs_%s"%(tair_var,)][:,i] = np.ma.filled(obs_tair, netCDF4.default_fillvals['f4'])
            ds_out.variables["infilled_%s"%(tair_var,)][:,i] = np.ma.filled(infill_tair, netCDF4.default_fillvals['f4'])
            ds_out.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    if params[P_XVAL_STNIDS] is None:
        xval_stnids = load_default_xval_stnids(stn_da.stn_ids)
    else:
        xval_stnids = params[P_XVAL_STNIDS]
    
    print "COORD: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in xval_stnids:
                    
        for tair_var in ['tmin','tmax']:
            
            if cnt < nwrkers:
                dest = cnt+N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                nrec+=1

            MPI.COMM_WORLD.send((stn_id,tair_var), dest=dest, tag=TAG_DOWORK)
            cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None,None), dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "COORD: done"

if __name__ == '__main__':
    
    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = os.path.join(FPATH_STNDATA, 'all', 'tair_homog_1948_2012.nc')
    params[P_PATH_OUT] = os.path.join(FPATH_STNDATA, 'infill', "xval_infill_tair.nc")
    params[P_PATH_NNR] = os.path.join(PROJECT_ROOT, 'reanalysis_data', 'conus_subset')
    
    #The number of years on which that infill model
    #should be trained and built
    params[P_NYRS_TRAIN] = 5
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    #If None, a default set of cross validation stations will be used
    params[P_XVAL_STNIDS] = None
    
    # PPCA parameters for infilling
    params[P_MIN_NNGH_DAILY] = 3
    params[P_NNGH_NNR] = 4
    params[P_NNR_VARYEXPLAIN] = 0.99
    params[P_FRACOBS_INIT_PCS] = 0.5
    params[P_PPCA_VARYEXPLAIN] = 0.99
    params[P_CHCK_IMP_PERF] = True
    params[P_NPCS_PPCA] = 0
    params[P_VERBOSE] = False
    
    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
