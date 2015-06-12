'''
A MPI script for running Tmin/Tmax quality assurance procedures.
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

from mpi4py import MPI
import numpy as np
from twx.db import StationDataDb, STN_ID, TMIN, TMAX, TMIN_FLAG, TMAX_FLAG, STATE
import twx.qa.qa_temp as qa_temp
from netCDF4 import Dataset, date2num
from twx.utils import YMD, StatusCheck, ymdL_to_date, Unbuffered
import sys
import os
import twx
import traceback
from twx.utils.util_dates import YEAR

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_POR_OUT = 'P_PATH_POR_OUT'
P_STN_MASK = 'P_STN_MASK'
P_QA_SPATIAL = 'P_QA_SPATIAL'

sys.stdout = Unbuffered(sys.stdout)

NA_STATE = np.array(["NA"], dtype="<U2")[0]

class IterFlagUpdate:

    def __init__(self, stn_id, tmin_flags, tmax_flags, ymd):
        self.stn_id = stn_id
        self.tmin_flags = tmin_flags
        self.tmax_flags = tmax_flags
        self.ymd = ymd
        self.x = -1
        self.max = self.ymd.size - 1
        self.flag_map = qa_temp.TWX_TO_GHCN_FLAGS_MAP

    def __iter__(self):
        return self

    def update_flags(self, curs):
        raise Exception('update_flags called on IterFlagUpdate')

    def next(self):
        self.x += 1
        if self.x > self.max:
            raise StopIteration
        else:
            return (self.flag_map[self.tmin_flags[self.x]], self.flag_map[self.tmax_flags[self.x]], self.stn_id, long(self.ymd[self.x]))

class IterMultiFlagUpdate:

    def __init__(self):
        self.iters = []
        self.x = -1

    def __iter__(self):
        return self

    def add_iter(self, a_iter):
        self.iters.append(a_iter)

    def update_flags(self, db_path):
        self.max = len(self.iters) - 1
        self.x = 0
        self.cur_iter = self.iters[self.x]

        ds = Dataset(db_path, 'r+')
        stn_ids = ds.variables['stn_id'][:]
        time_units = ds.variables['time'].units

        print self.cur_iter.stn_id

        stn_id = ""
        stn_idx = 0
        while 1:

            try:
                # QFLAG_TMIN,QFLAG_TMAX,STN_ID,YMD
                flag_row = self.next()

                if stn_id != flag_row[2]:

                    stn_id = flag_row[2]
                    stn_idx = np.nonzero(stn_ids == stn_id)[0][0]

                time_idx = int(date2num(ymdL_to_date(flag_row[3]), units=time_units))

                ds.variables['qflag_tmin'][time_idx, stn_idx] = flag_row[0]
                ds.variables['qflag_tmax'][time_idx, stn_idx] = flag_row[1]

            except StopIteration:
                break

        ds.close()

    def next(self):
        try:
            return self.cur_iter.next()
        except StopIteration:
            self.x += 1
            if self.x > self.max:
                raise StopIteration
            else:
                while True:
                    try:
                        self.cur_iter = self.iters[self.x]
                        print self.cur_iter.stn_id
                        return self.cur_iter.next()
                    except StopIteration:
                        self.x += 1
                        if self.x > self.max:
                            raise StopIteration

def proc_work(params, rank):

    status = MPI.Status()
    stn_da = StationDataDb(params[P_PATH_DB])

    while 1:

        try:

            stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD, tag=MPI.ANY_TAG, status=status)

            if status.tag == TAG_STOPWORK:
                MPI.COMM_WORLD.send(None, dest=RANK_WRITE, tag=TAG_STOPWORK)
                print "".join(["Worker ", str(rank), ": Finished"])
                return 0

            obs = stn_da.load_all_stn_obs(np.array([stn_id]))
            stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
            
            if params[P_QA_SPATIAL]:
                
                flags_tmin, flags_tmax = qa_temp.run_qa_spatial_only(stn, stn_da, obs[TMIN], obs[TMAX], stn_da.days)
            
            else:
                
                flags_tmin, flags_tmax = qa_temp.run_qa_non_spatial(obs[TMIN], obs[TMAX], stn_da.days)
            

            a_iter = create_update_iter(stn, flags_tmin, flags_tmax, stn_da.days,
                                             obs[TMIN_FLAG], obs[TMAX_FLAG])

            MPI.COMM_WORLD.send(a_iter, dest=RANK_WRITE, tag=TAG_DOWORK)

        except Exception, e:
            print traceback.format_exc()
            print "".join(["Error in QA of ", stn_id, ":", str(e), "\n"])

        MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)


def create_update_iter(stn, flags_tmin, flags_tmax, days, prev_flgs_tmin=np.array([]), prev_flgs_tmax=np.array([])):

    mask_tmin = np.logical_not(np.logical_or(flags_tmin == qa_temp.QA_OK, flags_tmin == qa_temp.QA_MISSING))
    mask_tmax = np.logical_not(np.logical_or(flags_tmax == qa_temp.QA_OK, flags_tmax == qa_temp.QA_MISSING))

    nobs_tmin = np.sum(flags_tmin != qa_temp.QA_MISSING)
    nobs_tmax = np.sum(flags_tmax != qa_temp.QA_MISSING)

    nflag_tmin = np.sum(mask_tmin)
    nflag_tmax = np.sum(mask_tmax)

    if nobs_tmin == 0:
        pctflg_tmin = np.nan
    else:
        pctflg_tmin = (nflag_tmin / float(nobs_tmin)) * 100.0

    if nobs_tmax == 0:
        pctflg_tmax = np.nan
    else:
        pctflg_tmax = (nflag_tmax / float(nobs_tmax)) * 100.0

    print "%s: Pct. of Tmin|Tmax observations flagged: %.2f|%.2f" % (stn[STN_ID], pctflg_tmin, pctflg_tmax)

    mask_all = np.logical_or(mask_tmin, mask_tmax)
    # make sure previous flags are not overwritten in the update
    flags_tmin = set_prev_flags(flags_tmin, prev_flgs_tmin)
    flags_tmax = set_prev_flags(flags_tmax, prev_flgs_tmax)

    return IterFlagUpdate(stn[STN_ID], flags_tmin[mask_all], flags_tmax[mask_all], days[YMD][mask_all])

def set_prev_flags(flags, prev_flgs):

    if prev_flgs.size > 0:

        uniq_flgs = np.unique(prev_flgs)

        for flag in uniq_flgs:

            if flag != "":

                mask_flg = prev_flgs == flag

                flags[np.logical_and(mask_flg, np.logical_or(flags == qa_temp.QA_OK, flags == qa_temp.QA_MISSING))] = qa_temp.GHCN_TO_TWX_FLAGS_MAP[flag]

    return flags

def proc_write(params, nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0

    iter_all = IterMultiFlagUpdate()

    nstns = np.nonzero(params[P_STN_MASK])[0].size
    stat_chk = StatusCheck(nstns, 10)

    while 1:

        a_iter = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        if status.tag == TAG_STOPWORK:

            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:
                
                print "Writer: updating QA flags in database..."
                
                iter_all.update_flags(params[P_PATH_DB])
                
                print "Writer: Creating new period-of-record csv file..."
                # Create new period-of-record csv file
                # since some stations might now fall below the minimum
                # por requirement after QA is run
                stn_da = StationDataDb(params[P_PATH_DB])
                stns = stn_da.stns
                
                fpath_out = os.path.join(params[P_PATH_POR_OUT],
                                         'all_por_%d_%d.csv' % (stn_da.days[YEAR][0], stn_da.days[YEAR][-1]))
                
                twx.db.output_por_csv(stn_da, stns, fpath_out)
                
                
                print "Writer: Finished"
                return 0
        else:

            iter_all.add_iter(a_iter)
            stat_chk.increment()

def proc_coord(params, nwrkers):

    stn_da = StationDataDb(params[P_PATH_DB])
    stns = stn_da.stns[params[P_STN_MASK]]

    cnt = 0
    nrec = 0

    for stn in stns:

        if cnt < nwrkers:
            dest = cnt + N_NON_WRKRS
        else:
            dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            nrec += 1

        MPI.COMM_WORLD.send(stn[STN_ID], dest=dest, tag=TAG_DOWORK)
        cnt += 1

    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send(stn[STN_ID], dest=w + N_NON_WRKRS, tag=TAG_STOPWORK)

if __name__ == '__main__':

    PROJECT_ROOT = os.getenv('TOPOWX_DATA')
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    START_YEAR = 1895
    END_YEAR = 2015

    np.seterr(all='raise')
    np.seterr(under='ignore')
    np.seterr(invalid='ignore')

    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = os.path.join(FPATH_STNDATA, 'all', 'all_%s_%s.nc' % (START_YEAR, END_YEAR))
    params[P_PATH_POR_OUT] = os.path.join(FPATH_STNDATA, 'all')
    
    # Need to run this QA script twice
    # First run, only apply non-spatial QA checks (P_QA_SPATIAL = False)
    # Second run, only apply spatial QA checks (P_QA_SPATIAL = True)
    params[P_QA_SPATIAL] = False
    
    # Only run QA for SNOTEL and RAWS stations since GHCH-D observations
    # are already flagged by the Durre et al. 2010 procedures
    stn_da = StationDataDb(params[P_PATH_DB])
    mask_sntl = np.logical_and(np.char.startswith(stn_da.stn_ids, "SNOTEL"), stn_da.stns[STATE] != "AK")
    mask_raws = np.char.startswith(stn_da.stn_ids, "RAWS")
    params[P_STN_MASK] = np.logical_or(mask_raws, mask_sntl)
    stn_da.ds.close()
    stn_da = None

    if rank == RANK_COORD:
        proc_coord(params, nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params, nsize - N_NON_WRKRS)
    else:
        proc_work(params, rank)
