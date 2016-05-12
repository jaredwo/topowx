'''
MPI script for running a cross validation of the infill model for infilling 
missing daily Tmin/Tmax.

Must be run using mpiexec or mpirun.
'''
from mpi4py import MPI
from netCDF4 import Dataset
from twx.db import StationDataDb, NNRNghData, create_quick_db
from twx.infill import XvalInfillParams, XvalInfill, load_default_xval_stnids
from twx.utils import StatusCheck, Unbuffered, ymdL, TwxConfig
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

NETCDF_OUT_VARIABLES = [('obs_tmin', 'f4', netCDF4.default_fillvals['f4'],
                         'observed minimum air temperature', 'C'),
                        ('obs_tmax', 'f4', netCDF4.default_fillvals['f4'],
                         'observed maximum air temperature', 'C'),
                        ('infilled_tmin', 'f4', netCDF4.default_fillvals['f4'],
                         'infilled minimum air temperature', 'C'),
                        ('infilled_tmax', 'f4', netCDF4.default_fillvals['f4'],
                         'infilled maximum air temperature', 'C')]

sys.stdout = Unbuffered(sys.stdout)

def proc_work(twx_cfg, xval_stnids, nyrs_train, start_ymd, end_ymd, params_ppca, rank):
    
    status = MPI.Status()
    
    stn_da = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                           (start_ymd, end_ymd))    
    ds_nnr = NNRNghData(twx_cfg.path_reanalysis_namerica,
                        (start_ymd, end_ymd))
    
    infill_params = XvalInfillParams(nnr_ds=ds_nnr, **params_ppca)
    
    xval_infills = {}
    xval_infills['tmin'] = XvalInfill(stn_da, 'tmin', infill_params,
                                      xval_stnids=xval_stnids,
                                      ntrain_yrs=nyrs_train)
    xval_infills['tmax'] = XvalInfill(stn_da, 'tmax', infill_params,
                                      xval_stnids=xval_stnids,
                                      ntrain_yrs=nyrs_train)
        
    print "".join(["WORKER ", str(rank), ": ready to receive work."])
    
    empty = np.ones(stn_da.days.size) * np.nan
        
    while 1:
    
        stn_id, tair_var = MPI.COMM_WORLD.recv(source=RANK_COORD,
                                               tag=MPI.ANY_TAG, status=status)
        
        if status.tag == TAG_STOPWORK:
            
            MPI.COMM_WORLD.send([None] * 4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ", str(rank), ": Finished"]) 
            return 0
       
        else:
                
            try:
            
                obs_tair, infill_tair = xval_infills[tair_var].run_xval(stn_id)
            
            except Exception as e:
            
                print "".join(["ERROR: WORKER ", str(rank), ": could not infill ",
                               tair_var, " for ", stn_id, str(e)])
                
                infill_tair = empty
                obs_tair = empty
            
            MPI.COMM_WORLD.send((stn_id, tair_var, infill_tair, obs_tair),
                                dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(twx_cfg, xval_stnids, start_ymd, end_ymd, nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    stn_da = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                           (start_ymd, end_ymd))
    
    if xval_stnids is None:
        xval_stnids = load_default_xval_stnids(stn_da.stn_ids)
    
    ttl_infills = xval_stnids.size * 2
    
    xval_stns = stn_da.stns[np.in1d(stn_da.stn_ids, xval_stnids, True)]
    
    create_quick_db(twx_cfg.fpath_xval_infill_nc, xval_stns, stn_da.days,
                    NETCDF_OUT_VARIABLES)
    ds_out = Dataset(twx_cfg.fpath_xval_infill_nc, 'r+')
    
    stn_idxs = {}
    for x in np.arange(xval_stnids.size):
        stn_idxs[xval_stnids[x]] = x
                    
    stat_chk = StatusCheck(ttl_infills, 10)
    
    while 1:

        stn_id, tair_var, infill_tair, obs_tair = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,
                                                                      tag=MPI.ANY_TAG,
                                                                      status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:
                print "WRITER: Finished"
                return 0
        else:
            
            infill_tair = np.ma.masked_array(infill_tair, np.isnan(infill_tair))
            obs_tair = np.ma.masked_array(obs_tair, np.isnan(obs_tair))
            
            i = stn_idxs[stn_id]
            
            difs = infill_tair - obs_tair
            bias = np.ma.mean(difs)
            mae = np.ma.mean(np.ma.abs(difs))
            
            print "|".join(["WRITER", stn_id, tair_var,
                            "MAE: %.2f" % (mae,), "BIAS: %.2f" % (bias,)])
            
            obs_tair = np.ma.filled(obs_tair, netCDF4.default_fillvals['f4'])
            ds_out.variables["obs_%s" % (tair_var,)][:, i] = obs_tair
            
            infill_tair = np.ma.filled(infill_tair, netCDF4.default_fillvals['f4'])
            ds_out.variables["infilled_%s" % (tair_var,)][:, i] = infill_tair
            
            ds_out.sync()
            
            stat_chk.increment()
                
def proc_coord(twx_cfg, xval_stnids, start_ymd, end_ymd, nwrkers):
    
    stn_da = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                           (start_ymd, end_ymd))
    
    if xval_stnids is None:
        xval_stnids = load_default_xval_stnids(stn_da.stn_ids)
    
    print "COORD: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in xval_stnids:
                    
        for tair_var in ['tmin', 'tmax']:
            
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
    
    print "Started process %d of %d"%(rank,nsize)
    
    # The number of years on which that infill model
    # should be trained and built
    nyrs_train = 5
    # Start and end YMD for infilling
    start_ymd = ymdL(twx_cfg.interp_start_date)
    end_ymd = ymdL(twx_cfg.interp_end_date)
    
    # Station IDs for which to run cross validation. If None, a default set of
    # cross validation stations will be used
    xval_stnids = None
        
    # PPCA parameters for infilling
    params_ppca = {}
    params_ppca['min_daily_nnghs'] = 3
    params_ppca['nnghs_nnr'] = 4
    params_ppca['max_nnr_var'] = 0.99
    params_ppca['frac_obs_initnpcs'] = 0.5
    params_ppca['ppca_varyexplain'] = 0.99
    params_ppca['chk_perf'] = True
    params_ppca['npcs'] = 0
    params_ppca['verbose'] = False
    
    if rank == RANK_COORD:        
        proc_coord(twx_cfg, xval_stnids, start_ymd, end_ymd, nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(twx_cfg, xval_stnids, start_ymd, end_ymd, nsize - N_NON_WRKRS)
    else:
        proc_work(twx_cfg, xval_stnids, nyrs_train, start_ymd, end_ymd,
                  params_ppca, rank)

    MPI.COMM_WORLD.Barrier()
