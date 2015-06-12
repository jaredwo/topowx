'''
MPI script for infilling missing values in 
incomplete Tmin/Tmax station time series.

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
from twx.db import StationDataDb, STN_ID, NNRNghData, create_quick_db
from twx.utils import StatusCheck, Unbuffered, ymdL
from netCDF4 import Dataset
import netCDF4
import os
from twx.db.station_data import get_mean_varname, get_variance_varname
from twx.infill.infill_daily import infill_daily_obs
from twx.utils.util_dates import MONTH
from twx.infill.post_infill import get_bad_infill_stnids
from datetime import datetime

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
P_NCDF_MODE = 'P_NCDF_MODE'
P_STNIDS_TMIN = 'P_STNIDS_TMIN'
P_STNIDS_TMAX = 'P_STNIDS_TMAX'

P_MIN_NNGH_DAILY = 'P_MIN_NNGH_DAILY'
P_NNGH_NNR = 'P_NNGH_NNR'
P_NNR_VARYEXPLAIN = 'P_NNR_VARYEXPLAIN'
P_FRACOBS_INIT_PCS = 'P_FRACOBS_INIT_PCS'
P_PPCA_VARYEXPLAIN = 'P_PPCA_VARYEXPLAIN'
P_CHCK_IMP_PERF = 'P_CHCK_IMP_PERF'
P_NPCS_PPCA = 'P_NPCS_PPCA'
P_VERBOSE = 'P_VERBOSE'
P_FPATH_LOG = 'P_FPATH_LOG'

LAST_VAR_WRITTEN = 'mae'

sys.stdout = Unbuffered(sys.stdout)

def proc_work(params, rank):

    status = MPI.Status()

    stn_da = StationDataDb(params[P_PATH_DB], (params[P_START_YMD], params[P_END_YMD]))
    days = stn_da.days
    ndays = float(days.size)

    empty_fill = np.ones(ndays, dtype=np.float32) * netCDF4.default_fillvals['f4']
    empty_flags = np.ones(ndays, dtype=np.int8) * netCDF4.default_fillvals['i1']
    empty_bias = netCDF4.default_fillvals['f4']
    empty_mae = netCDF4.default_fillvals['f4']

    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD], params[P_END_YMD]))
    
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
                    
                    fnl_tmin, fill_mask_tmin, infill_tmin, mae_tmin, bias_tmin = infill_tair(stn_id, stn_da,
                                                                                             'tmin', ds_nnr,
                                                                                             vnames_mean_tmin,
                                                                                             vnames_vari_tmin,
                                                                                             mth_masks, params)
                    
                if run_infill_tmax:
     
                    fnl_tmax, fill_mask_tmax, infill_tmax, mae_tmax, bias_tmax = infill_tair(stn_id, stn_da,
                                                                                             'tmax', ds_nnr,
                                                                                             vnames_mean_tmax,
                                                                                             vnames_vari_tmax,
                                                                                             mth_masks, params)
            
            except Exception as e:

                print "".join(["ERROR: Could not infill ", stn_id, "|", str(e)])
                if run_infill_tmin:
                    fnl_tmin, fill_mask_tmin, infill_tmin, mae_tmin, bias_tmin = empty_fill, empty_flags, empty_fill, empty_mae, empty_bias
                if run_infill_tmax:
                    fnl_tmax, fill_mask_tmax, infill_tmax, mae_tmax, bias_tmax = empty_fill, empty_flags, empty_fill, empty_mae, empty_bias

            if run_infill_tmin:
                MPI.COMM_WORLD.send((stn_id, 'tmin', fnl_tmin, fill_mask_tmin, infill_tmin, mae_tmin, bias_tmin), dest=RANK_WRITE, tag=TAG_DOWORK)
            if run_infill_tmax:
                MPI.COMM_WORLD.send((stn_id, 'tmax', fnl_tmax, fill_mask_tmax, infill_tmax, mae_tmax, bias_tmax), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)

def proc_write(params, nwrkers):

    status = MPI.Status()
    stn_da = StationDataDb(params[P_PATH_DB], (params[P_START_YMD], params[P_END_YMD]))
    days = stn_da.days
    nwrkrs_done = 0

    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stnids_tmin, stnids_tmax = bcast_msg
    print "WRITER: Received broadcast msg"

    path_out_tmin = os.path.join(params[P_PATH_OUT], 'infill_tmin.nc')
    path_out_tmax = os.path.join(params[P_PATH_OUT], 'infill_tmax.nc')

    if params[P_NCDF_MODE] == 'r+':

        ds_tmin = Dataset(path_out_tmin, 'r+')
        ds_tmax = Dataset(path_out_tmax, 'r+')
        ttl_infills = stnids_tmin.size + stnids_tmax.size
        stnids_tmin = np.array(ds_tmin.variables['stn_id'][:], dtype="<S16")
        stnids_tmax = np.array(ds_tmax.variables['stn_id'][:], dtype="<S16")

    else:

        stns_tmin = stn_da.stns[np.in1d(stn_da.stns[STN_ID], stnids_tmin, assume_unique=True)]
        variables_tmin = [('tmin', 'f4', netCDF4.default_fillvals['f4'], 'minimum air temperature', 'C'),
                          ('flag_infilled', 'i1', netCDF4.default_fillvals['i1'], 'infilled flag', ''),
                          ('tmin_infilled', 'f4', netCDF4.default_fillvals['f4'], 'infilled minimum air temperature', 'C')]
        create_quick_db(path_out_tmin, stns_tmin, days, variables_tmin)
        stnda_out_tmin = StationDataDb(path_out_tmin, mode="r+")
        stnda_out_tmin.add_stn_variable('mae', 'mean absolute error', 'C', "f8")
        stnda_out_tmin.add_stn_variable('bias', 'bias', 'C', "f8")
        ds_tmin = stnda_out_tmin.ds

        stns_tmax = stn_da.stns[np.in1d(stn_da.stns[STN_ID], stnids_tmax, assume_unique=True)]
        variables_tmax = [('tmax', 'f4', netCDF4.default_fillvals['f4'], 'maximum air temperature', 'C'),
                          ('flag_infilled', 'i1', netCDF4.default_fillvals['i1'], 'infilled flag', ''),
                          ('tmax_infilled', 'f4', netCDF4.default_fillvals['f4'], 'infilled maximum air temperature', 'C')]
        create_quick_db(path_out_tmax, stns_tmax, days, variables_tmax)
        stnda_out_tmax = StationDataDb(path_out_tmax, mode="r+")
        stnda_out_tmax.add_stn_variable('mae', 'mean absolute error', 'C', "f8")
        stnda_out_tmax.add_stn_variable('bias', 'bias', 'C', "f8")
        ds_tmax = stnda_out_tmax.ds

        ttl_infills = stnids_tmin.size + stnids_tmax.size

    print "WRITER: Infilling a total of %d station time series " % (ttl_infills,)
    print "WRITER: Output NCDF files ready"

    stat_chk = StatusCheck(ttl_infills, 10)

    while 1:

        stn_id, tair_var, tair, fill_mask, tair_infill, mae, bias = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

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

            print "|".join(["WRITER", stn_id, tair_var, "%.4f" % (mae,), "%.4f" % (bias,)])

            stat_chk.increment()

def proc_coord(params, nwrkers):

    stn_da = StationDataDb(params[P_PATH_DB], (params[P_START_YMD], params[P_END_YMD]))

    mask_tmin = np.isfinite(stn_da.stns[get_mean_varname('tmin', 1)])
    mask_tmax = np.isfinite(stn_da.stns[get_mean_varname('tmax', 1)])

    stnids_tmin = stn_da.stn_ids[mask_tmin]
    stnids_tmax = stn_da.stn_ids[mask_tmax]

    # Check if we're restarting a run
    if params[P_NCDF_MODE] == 'r+':

        # If rerunning remove stn ids that have already been completed
        try:

            if params[P_STNIDS_TMIN] is None:

                ds_tmin = Dataset(os.path.join(params[P_PATH_OUT], 'infill_tmin.nc'))
                mask_incplt = ds_tmin.variables[LAST_VAR_WRITTEN][:].mask
                stnids_tmin = stnids_tmin[mask_incplt]

            else:

                stnids_tmin = params[P_STNIDS_TMIN]

        except AttributeError:
            # no mask: infill complete
            stnids_tmin = np.array([], dtype="<S16")

        try:

            if params[P_STNIDS_TMAX] is None:

                ds_tmax = Dataset(os.path.join(params[P_PATH_OUT], 'infill_tmax.nc'))
                mask_incplt = ds_tmax.variables[LAST_VAR_WRITTEN][:].mask
                stnids_tmax = stnids_tmax[mask_incplt]

            else:

                stnids_tmax = params[P_STNIDS_TMAX]

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

def infill_tair(stn_id, stn_da, tair_var, nnr_ds, vname_means, vname_varis, day_masks, params):
    
    
    fnl_tair, mask_infill, infill_tair = infill_daily_obs(stn_id=stn_id,
                                                            stn_da=stn_da,
                                                            tair_var=tair_var,
                                                            nnr_ds=nnr_ds,
                                                            vname_mean=vname_means,
                                                            vname_vari=vname_varis,
                                                            day_masks=day_masks,
                                                            add_bestngh=True,
                                                            min_daily_nnghs=params[P_MIN_NNGH_DAILY],
                                                            nnghs_nnr=params[P_NNGH_NNR],
                                                            max_nnr_var=params[P_NNR_VARYEXPLAIN],
                                                            chk_perf=params[P_CHCK_IMP_PERF],
                                                            npcs=params[P_NPCS_PPCA],
                                                            frac_obs_initnpcs=params[P_FRACOBS_INIT_PCS],
                                                            ppca_varyexplain=params[P_PPCA_VARYEXPLAIN],
                                                            verbose=params[P_VERBOSE])

    # Calculate MAE/bias on days with both observed and infilled values
    obs_mask = np.logical_not(mask_infill)
    difs = infill_tair[obs_mask] - fnl_tair[obs_mask]
    mae = np.mean(np.abs(difs))
    bias = np.mean(difs)

    return fnl_tair, mask_infill, infill_tair, mae, bias


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
    params[P_PATH_OUT] = os.path.join(FPATH_STNDATA, 'infill')
    params[P_PATH_NNR] = os.path.join(PROJECT_ROOT, 'reanalysis_data', 'n_america_subset')
    params[P_NCDF_MODE] = 'w'  # w or r+
    params[P_START_YMD] = ymdL(datetime(START_YEAR_POR, 1, 1))
    params[P_END_YMD] = ymdL(datetime(END_YEAR_POR, 12, 31))

    # PPCA parameters for infilling
    params[P_MIN_NNGH_DAILY] = 3
    params[P_NNGH_NNR] = 4
    params[P_NNR_VARYEXPLAIN] = 0.99
    params[P_FRACOBS_INIT_PCS] = 0.5
    params[P_PPCA_VARYEXPLAIN] = 0.99
    params[P_CHCK_IMP_PERF] = True
    params[P_NPCS_PPCA] = 0
    params[P_VERBOSE] = False

    params[P_FPATH_LOG] = None  # '/projects/topowx/mpi_runs/infill/infill_run_20150224.log'

    if rank == RANK_COORD:
        
        if params[P_FPATH_LOG] is not None:
        
            stn_da = StationDataDb(params[P_PATH_DB], (params[P_START_YMD], params[P_END_YMD]))
         
            mask_tmin = np.isfinite(stn_da.stns[get_mean_varname('tmin', 1)])
            mask_tmax = np.isfinite(stn_da.stns[get_mean_varname('tmax', 1)])
         
            stnids_tmin = stn_da.stn_ids[mask_tmin]
            stnids_tmax = stn_da.stn_ids[mask_tmax]
             
            stnids_bad = get_bad_infill_stnids(params[P_FPATH_LOG])
             
            params[P_STNIDS_TMIN] = stnids_tmin[np.in1d(stnids_tmin, stnids_bad, True)]
            params[P_STNIDS_TMAX] = stnids_tmax[np.in1d(stnids_tmax, stnids_bad, True)]
             
            stn_da.ds.close()
            stn_da = None
            mask_tmin = None
            mask_tmax = None
            stnids_tmin = None
            stnids_tmax = None
            stnids_bad = None
        
        proc_coord(params, nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params, nsize - N_NON_WRKRS)
    else:
        proc_work(params, rank)

    MPI.COMM_WORLD.Barrier()
