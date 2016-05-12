'''
MPI script for estimating/infilling period-of-record means
and variances of incomplete Tmin/Tmax station time series.

Must be run using mpiexec or mpirun.
'''
from mpi4py import MPI
from twx.db import StationDataDb, NNRNghData, build_por_mask
from twx.db.station_data import get_mean_varname, get_variance_varname
from twx.infill import infill_mean_variance
from twx.utils import StatusCheck, Unbuffered, ymdL, MONTH, TwxConfig
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

def proc_work(twx_cfg, start_ymd, end_ymd, min_nngh_daily, rank):
    
    status = MPI.Status()
    stn_da = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                           (start_ymd, end_ymd))
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    mask_por_tmin, mask_por_tmax = bcast_msg
    print "".join(["WORKER ", str(rank), ": Received broadcast msg"])
            
    stn_masks = {'tmin':mask_por_tmin, 'tmax':mask_por_tmax}
    
    ds_nnr = NNRNghData(twx_cfg.path_reanalysis_namerica, (start_ymd, end_ymd))
    
    mth_masks = [stn_da.days[MONTH] == mth for mth in np.arange(1, 13)]
                        
    while 1:
    
        stn_id, tair_var = MPI.COMM_WORLD.recv(source=RANK_COORD,
                                               tag=MPI.ANY_TAG, status=status)

        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None] * 4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ", str(rank), ": Finished"]) 
            return 0
        else:
            
            try:
                
                stn_mean, stn_vari = infill_mean_variance(stn_id, stn_da,
                                                          stn_masks[tair_var],
                                                          tair_var, ds_nnr,
                                                          nnghs=min_nngh_daily,
                                                          day_masks=mth_masks)
            
            except Exception as e:
            
                print "".join(["ERROR: WORKER ", str(rank), ": could not infill ",
                               tair_var, " for ", stn_id, str(e)])
                
                empty = np.empty(12)
                empty.fill(netCDF4.default_fillvals['f8'])
                
                stn_mean = empty
                stn_vari = empty
            
            MPI.COMM_WORLD.send((stn_id, tair_var, stn_mean, stn_vari),
                                dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(twx_cfg, start_ymd, end_ymd, nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    stn_da = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                           (start_ymd, end_ymd), mode="r+")
    
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
    stn_ids_tmin, stn_ids_tmax = (stn_da.stn_ids[mask_por_tmin],
                                  stn_da.stn_ids[mask_por_tmax])
    print "WRITER: Received broadcast msg"
    stn_ids_uniq = np.unique(np.concatenate([stn_ids_tmin, stn_ids_tmax]))
    
    stn_idxs = {}
    for x in np.arange(stn_da.stn_ids.size):
        if stn_da.stn_ids[x] in stn_ids_uniq:
            stn_idxs[stn_da.stn_ids[x]] = x
            
    ttl_infills = stn_ids_tmin.size + stn_ids_tmax.size
    
    stat_chk = StatusCheck(ttl_infills, 30)
    
    while 1:
       
        stn_id, tair_var, stn_mean, stn_vari = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, 
                                                                   tag=MPI.ANY_TAG,
                                                                   status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:
                print "WRITER: Finished"
                return 0
        else:
            
            stnid_dim = stn_idxs[stn_id]
            
            for mth in mths:
                
                vname_mean = get_mean_varname(tair_var, mth)
                stn_da.ds.variables[vname_mean][stnid_dim] = stn_mean[mth - 1]
                
                vname_vary = get_variance_varname(tair_var, mth)
                stn_da.ds.variables[vname_vary][stnid_dim] = stn_vari[mth - 1]       

            stn_da.ds.sync()
                        
            stat_chk.increment()
                
def proc_coord(twx_cfg, min_por, start_ymd, end_ymd, nwrkers):
    
    
    stndb = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                          (start_ymd, end_ymd))
    
    mask_por_tmin = build_por_mask(stndb.ds, ['tmin'], 
                                   twx_cfg.interp_start_date,
                                   twx_cfg.interp_end_date,
                                   min_por_yrs=min_por)
    
    mask_por_tmax = build_por_mask(stndb.ds, ['tmax'], 
                                   twx_cfg.interp_start_date,
                                   twx_cfg.interp_end_date,
                                   min_por_yrs=min_por)
    stndb.stn_ids[mask_por_tmin]
    
    # Extract stn_ids that have min # of observations
    stn_ids_tmin = stndb.stn_ids[mask_por_tmin]
    stn_ids_tmax = stndb.stn_ids[mask_por_tmax]
    
    stndb.ds.close()
    del stndb
        
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
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()
    
    # Start and end YMD for infilling
    start_ymd = ymdL(twx_cfg.interp_start_date)
    end_ymd = ymdL(twx_cfg.interp_end_date)
    
    # Minimum period-of-record in years required for station to be infilled
    min_por = 5
    
    # Minimum number of neighboring observations per day
    min_nngh_daily = 3
    
    if rank == RANK_COORD:
        proc_coord(twx_cfg, min_por, start_ymd, end_ymd,  nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(twx_cfg, start_ymd, end_ymd, nsize - N_NON_WRKRS)
    else:
        proc_work(twx_cfg, start_ymd, end_ymd, min_nngh_daily, rank)

    MPI.COMM_WORLD.Barrier()
