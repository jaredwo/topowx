'''
MPI script for infilling missing values in incomplete Tmin/Tmax station time series.

Must be run using mpiexec or mpirun.
'''
from mpi4py import MPI
from netCDF4 import Dataset
from twx.db import StationDataDb, STN_ID, NNRNghData, create_quick_db
from twx.db.station_data import get_mean_varname, get_variance_varname
from twx.infill.infill_daily import infill_daily_obs
from twx.infill.post_infill import get_bad_infill_stnids
from twx.utils import StatusCheck, Unbuffered, ymdL, MONTH, TwxConfig
import netCDF4
import numpy as np
import os
import sys
import argparse

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

LAST_VAR_WRITTEN = 'mae'

sys.stdout = Unbuffered(sys.stdout)

def proc_work(twx_cfg, start_ymd, end_ymd, params_ppca, rank):

    status = MPI.Status()

    stn_da = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                           (start_ymd, end_ymd))
    days = stn_da.days
    ndays = float(days.size)

    empty_fill = np.ones(ndays, dtype=np.float32) * netCDF4.default_fillvals['f4']
    empty_flags = np.ones(ndays, dtype=np.int8) * netCDF4.default_fillvals['i1']
    empty_bias = netCDF4.default_fillvals['f4']
    empty_mae = netCDF4.default_fillvals['f4']

    ds_nnr = NNRNghData(twx_cfg.path_reanalysis_namerica,
                        (start_ymd, end_ymd))
    
    mths = np.arange(1, 13)
    mth_masks = [stn_da.days[MONTH] == mth for mth in mths]
    vnames_mean_tmin = [get_mean_varname('tmin', mth) for mth in mths]
    vnames_vari_tmin = [get_variance_varname('tmin', mth) for mth in mths]
    vnames_mean_tmax = [get_mean_varname('tmax', mth) for mth in mths]
    vnames_vari_tmax = [get_variance_varname('tmax', mth) for mth in mths]
    

    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stnids_tmin, stnids_tmax = bcast_msg
    print "".join(["WORKER ", str(rank), ": Received broadcast msg"])
    print "".join(["WORKER ", str(rank),
                   ": Minimum number of station neighbors for infilling: ",
                   str(params_ppca['min_daily_nnghs'])])

    while 1:

        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD, tag=MPI.ANY_TAG, status=status)

        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None] * 7, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ", str(rank), ": Finished"])
            return 0
        else:

            try:
                
                run_infill_tmin = stn_id in stnids_tmin
                run_infill_tmax = stn_id in stnids_tmax

                if run_infill_tmin:
                    
                    results = infill_tair(stn_id, stn_da, 'tmin', ds_nnr,
                                          vnames_mean_tmin, vnames_vari_tmin,
                                          mth_masks, params_ppca)
                    fnl_tmin, fill_mask_tmin, infill_tmin, mae_tmin, bias_tmin = results
                    
                if run_infill_tmax:
     
                    results = infill_tair(stn_id, stn_da,'tmax', ds_nnr,
                                          vnames_mean_tmax, vnames_vari_tmax,
                                          mth_masks, params_ppca)
                    fnl_tmax, fill_mask_tmax, infill_tmax, mae_tmax, bias_tmax = results 
            
            except Exception as e:

                print "".join(["ERROR: Could not infill ", stn_id, "|", str(e)])
                if run_infill_tmin:
                    
                    results = empty_fill, empty_flags, empty_fill, empty_mae, empty_bias
                    fnl_tmin, fill_mask_tmin, infill_tmin, mae_tmin, bias_tmin = results
                    
                if run_infill_tmax:
                    
                    results = empty_fill, empty_flags, empty_fill, empty_mae, empty_bias
                    fnl_tmax, fill_mask_tmax, infill_tmax, mae_tmax, bias_tmax = results

            if run_infill_tmin:
                MPI.COMM_WORLD.send((stn_id, 'tmin', fnl_tmin, fill_mask_tmin,
                                     infill_tmin, mae_tmin, bias_tmin),
                                    dest=RANK_WRITE, tag=TAG_DOWORK)
            if run_infill_tmax:
                MPI.COMM_WORLD.send((stn_id, 'tmax', fnl_tmax, fill_mask_tmax,
                                     infill_tmax, mae_tmax, bias_tmax),
                                    dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)

def proc_write(twx_cfg, ncdf_mode, start_ymd, end_ymd, nwrkers):

    status = MPI.Status()
    stn_da = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                           (start_ymd, end_ymd))
    days = stn_da.days
    nwrkrs_done = 0

    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stnids_tmin, stnids_tmax = bcast_msg
    print "WRITER: Received broadcast msg"

    if ncdf_mode == 'r+':

        ds_tmin = Dataset(twx_cfg.fpath_stndata_nc_infill_tmin, 'r+')
        ds_tmax = Dataset(twx_cfg.fpath_stndata_nc_infill_tmax, 'r+')
        ttl_infills = stnids_tmin.size + stnids_tmax.size
        stnids_tmin = ds_tmin.variables[STN_ID][:].astype(np.str)
        stnids_tmax = ds_tmax.variables[STN_ID][:].astype(np.str)

    else:

        stns_tmin = stn_da.stns[np.in1d(stn_da.stns[STN_ID], stnids_tmin,
                                        assume_unique=True)]
        variables_tmin = [('tmin', 'f4', netCDF4.default_fillvals['f4'],
                           'minimum air temperature', 'C'),
                          ('flag_infilled', 'i1', netCDF4.default_fillvals['i1'],
                           'infilled flag', ''),
                          ('tmin_infilled', 'f4', netCDF4.default_fillvals['f4'],
                           'infilled minimum air temperature', 'C')]
        create_quick_db(twx_cfg.fpath_stndata_nc_infill_tmin, stns_tmin, days,
                        variables_tmin)
        stnda_out_tmin = StationDataDb(twx_cfg.fpath_stndata_nc_infill_tmin,
                                       mode="r+")
        stnda_out_tmin.add_stn_variable('mae', 'mean absolute error', 'C', "f8")
        stnda_out_tmin.add_stn_variable('bias', 'bias', 'C', "f8")
        ds_tmin = stnda_out_tmin.ds

        stns_tmax = stn_da.stns[np.in1d(stn_da.stns[STN_ID], stnids_tmax,
                                        assume_unique=True)]
        variables_tmax = [('tmax', 'f4', netCDF4.default_fillvals['f4'],
                           'maximum air temperature', 'C'),
                          ('flag_infilled', 'i1', netCDF4.default_fillvals['i1'],
                           'infilled flag', ''),
                          ('tmax_infilled', 'f4', netCDF4.default_fillvals['f4'],
                           'infilled maximum air temperature', 'C')]
        create_quick_db(twx_cfg.fpath_stndata_nc_infill_tmax, stns_tmax, days,
                        variables_tmax)
        stnda_out_tmax = StationDataDb(twx_cfg.fpath_stndata_nc_infill_tmax,
                                       mode="r+")
        stnda_out_tmax.add_stn_variable('mae', 'mean absolute error', 'C', "f8")
        stnda_out_tmax.add_stn_variable('bias', 'bias', 'C', "f8")
        ds_tmax = stnda_out_tmax.ds

        ttl_infills = stnids_tmin.size + stnids_tmax.size

    print "WRITER: Infilling a total of %d station time series " % (ttl_infills,)
    print "WRITER: Output NCDF files ready"

    stat_chk = StatusCheck(ttl_infills, 10)

    while 1:

        result = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                     status=status)
        stn_id, tair_var, tair, fill_mask, tair_infill, mae, bias = result

        if status.tag == TAG_STOPWORK:

            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:

                print "Writer: Finished"
                return 0
        else:

            if tair_var == 'tmin':
                stn_idx = np.nonzero(stnids_tmin == stn_id)[0][0]
                ds = ds_tmin
            else:
                stn_idx = np.nonzero(stnids_tmax == stn_id)[0][0]
                ds = ds_tmax

            ds.variables[tair_var][:, stn_idx] = tair
            ds.variables["".join([tair_var, "_infilled"])][:, stn_idx] = tair_infill
            ds.variables['flag_infilled'][:, stn_idx] = fill_mask
            ds.variables['bias'][stn_idx] = bias
            ds.variables[LAST_VAR_WRITTEN][stn_idx] = mae

            ds.sync()

            print "|".join(["WRITER", stn_id, tair_var, "%.4f" % (mae,),
                            "%.4f" % (bias,)])

            stat_chk.increment()

def proc_coord(twx_cfg, ncdf_mode, stnids_to_infill_tmin, stnids_to_infill_tmax,
               start_ymd, end_ymd, nwrkers):

    stn_da = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                           (start_ymd, end_ymd))

    mask_tmin = np.isfinite(stn_da.stns[get_mean_varname('tmin', 1)])
    mask_tmax = np.isfinite(stn_da.stns[get_mean_varname('tmax', 1)])

    stnids_tmin = stn_da.stn_ids[mask_tmin]
    stnids_tmax = stn_da.stn_ids[mask_tmax]

    # Check if we're restarting a run
    if ncdf_mode == 'r+':

        # If rerunning remove stn ids that have already been completed
        try:

            if stnids_to_infill_tmin is None:
                
                ds_tmin = Dataset(twx_cfg.fpath_stndata_nc_infill_tmin)
                mask_incplt = ds_tmin.variables[LAST_VAR_WRITTEN][:].mask
                stnids_tmin = stnids_tmin[mask_incplt]

            else:

                stnids_tmin = stnids_to_infill_tmin

        except AttributeError:
            # no mask: infill complete
            stnids_tmin = np.array([], dtype="<S16")

        try:

            if stnids_to_infill_tmax is None:

                ds_tmax = Dataset(twx_cfg.fpath_stndata_nc_infill_tmax)
                mask_incplt = ds_tmax.variables[LAST_VAR_WRITTEN][:].mask
                stnids_tmax = stnids_tmax[mask_incplt]

            else:

                stnids_tmax = stnids_to_infill_tmax

        except AttributeError:
            # no mask: infill complete
            stnids_tmax = np.array([], dtype="<S16")

    stnids_all = np.unique(np.concatenate((stnids_tmin, stnids_tmax)))

    # Send stn ids to all processes
    MPI.COMM_WORLD.bcast((stnids_tmin, stnids_tmax), root=RANK_COORD)

    print "COORD: Done initialization. Starting to send work."

    cnt = 0
    nrec = 0

    for stn_id in stnids_all:

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

def infill_tair(stn_id, stn_da, tair_var, nnr_ds, vname_means, vname_varis,
                day_masks, params_ppca):
    
    fnl_tair, mask_infill, infill_tair = infill_daily_obs(stn_id=stn_id,
                                                          stn_da=stn_da,
                                                          tair_var=tair_var,
                                                          nnr_ds=nnr_ds,
                                                          vname_mean=vname_means,
                                                          vname_vari=vname_varis,
                                                          day_masks=day_masks,
                                                          add_bestngh=True,
                                                          **params_ppca)

    # Calculate MAE/bias on days with both observed and infilled values
    obs_mask = np.logical_not(mask_infill)
    difs = infill_tair[obs_mask] - fnl_tair[obs_mask]
    mae = np.mean(np.abs(difs))
    bias = np.mean(difs)

    return fnl_tair, mask_infill, infill_tair, mae, bias


if __name__ == '__main__':

    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    # Need to run this mpi script twice. First run, min_daily_nnghs is set to 
    # 3 stations. Second run, path to log file from previous run is passed as
    # and argument and infilling is executed again for those stations
    # with suspect infilling models, but with min_daily_nnghs set to 7.
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", help="path to logfile from previous run")
    parser.add_argument("--nc_mode", help="netcdf access mode. 'w' or 'r+'",
                        default='w')
    args = parser.parse_args()
    fpath_log = args.logfile  
    ncdf_mode = args.nc_mode

    np.seterr(all='raise')
    np.seterr(under='ignore')

    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    print "NetCDF access mode is %s. Process %d of %d."%(ncdf_mode, rank, nsize)
    
    # Start and end YMD for infilling
    start_ymd = ymdL(twx_cfg.interp_start_date)
    end_ymd = ymdL(twx_cfg.interp_end_date)
    
    # PPCA parameters for infilling
    # PPCA parameters for infilling
    params_ppca = {}
    # Min daily station neighbors is 3 for initial run and 7 for re-run
    params_ppca['min_daily_nnghs'] = 3 if fpath_log is None else 7
    params_ppca['nnghs_nnr'] = 4
    params_ppca['max_nnr_var'] = 0.99
    params_ppca['frac_obs_initnpcs'] = 0.5
    params_ppca['ppca_varyexplain'] = 0.99
    params_ppca['chk_perf'] = True
    params_ppca['npcs'] = 0
    params_ppca['verbose'] = False
    
    if rank == RANK_COORD:
        
        # If previous log file specified, run infilling
        # only for stations that were suspect.
        if fpath_log is not None:
            
            print ("Initializing rerun of previous infill run. "
                   "Process %d of %d"%(rank, nsize))
                                
            stn_da = StationDataDb(twx_cfg.fpath_stndata_nc_tair_homog,
                                   (start_ymd, end_ymd))
         
            mask_tmin = np.isfinite(stn_da.stns[get_mean_varname('tmin', 1)])
            mask_tmax = np.isfinite(stn_da.stns[get_mean_varname('tmax', 1)])
         
            stnids_tmin = stn_da.stn_ids[mask_tmin]
            stnids_tmax = stn_da.stn_ids[mask_tmax]
             
            stnids_bad = get_bad_infill_stnids(fpath_log)
                          
            stnids_to_infill_tmin = stnids_tmin[np.in1d(stnids_tmin, stnids_bad, True)]
            stnids_to_infill_tmax = stnids_tmax[np.in1d(stnids_tmax, stnids_bad, True)]
             
            stn_da.ds.close()
            stn_da = None
            mask_tmin = None
            mask_tmax = None
            stnids_tmin = None
            stnids_tmax = None
            stnids_bad = None
            
        else:
            
            stnids_to_infill_tmin = None
            stnids_to_infill_tmax = None
        
        proc_coord(twx_cfg, ncdf_mode, stnids_to_infill_tmin,
                   stnids_to_infill_tmax, start_ymd, end_ymd, nsize - N_NON_WRKRS)
        
    elif rank == RANK_WRITE:
        proc_write(twx_cfg, ncdf_mode, start_ymd, end_ymd, nsize - N_NON_WRKRS)
    else:
        proc_work(twx_cfg, start_ymd, end_ymd, params_ppca, rank)

    MPI.COMM_WORLD.Barrier()
