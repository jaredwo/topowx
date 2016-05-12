from mpi4py import MPI
from netCDF4 import Dataset, date2index
from twx.db import StationDataDb, STN_ID, TMIN, TMAX, TMIN_FLAG, TMAX_FLAG, add_obs_cnt
from twx.utils import YMD, StatusCheck, ymdL_to_date, Unbuffered, TwxConfig, ymdL
import argparse
import numpy as np
import os
import traceback
import twx.qa.qa_temp as qa_temp
import sys

sys.stdout = Unbuffered(sys.stdout)

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

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
        stn_ids = ds.variables[STN_ID][:]
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

                time_idx = int(date2index(ymdL_to_date(flag_row[3]), ds.variables['time']))
                
                if flag_row[0] != '':
                    print 'qflag_tmin',stn_id, time_idx, stn_idx,flag_row[0]
                if flag_row[1] != '':
                    print 'qflag_tmax',stn_id, time_idx, stn_idx,flag_row[1]
                ds.variables['qflag_tmin'][time_idx, stn_idx] = flag_row[0]
                ds.variables['qflag_tmax'][time_idx, stn_idx] = flag_row[1]

            except StopIteration:
                break
        
        ds.sync()
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

def proc_work(twx_cfg, qa_spatial, rank):

    status = MPI.Status()
    stndb = StationDataDb(twx_cfg.fpath_stndata_nc_all)

    while 1:

        try:

            stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD, tag=MPI.ANY_TAG,
                                         status=status)

            if status.tag == TAG_STOPWORK:
                MPI.COMM_WORLD.send(None, dest=RANK_WRITE, tag=TAG_STOPWORK)
                print "".join(["Worker ", str(rank), ": Finished"])
                return 0

            obs = stndb.load_all_stn_obs(np.array([stn_id]))
            stn = stndb.stns[stndb.stn_ids == stn_id][0]
            
            if qa_spatial:
                
                flags_tmin, flags_tmax = qa_temp.run_qa_spatial_only(stn, stndb,
                                                                     obs[TMIN],
                                                                     obs[TMAX],
                                                                     stndb.days)
            
            else:
                
                flags_tmin, flags_tmax = qa_temp.run_qa_non_spatial(obs[TMIN],
                                                                    obs[TMAX],
                                                                    stndb.days)
            

            a_iter = create_update_iter(stn, flags_tmin, flags_tmax, stndb.days,
                                             obs[TMIN_FLAG], obs[TMAX_FLAG])

            MPI.COMM_WORLD.send(a_iter, dest=RANK_WRITE, tag=TAG_DOWORK)

        except Exception, e:
            print traceback.format_exc()
            print "".join(["Error in QA of ", stn_id, ":", str(e), "\n"])

        MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)

def create_update_iter(stn, flags_tmin, flags_tmax, days,
                       prev_flgs_tmin=np.array([]),
                       prev_flgs_tmax=np.array([])):

    mask_tmin = np.logical_not(np.logical_or(flags_tmin == qa_temp.QA_OK,
                                             flags_tmin == qa_temp.QA_MISSING))
    mask_tmax = np.logical_not(np.logical_or(flags_tmax == qa_temp.QA_OK,
                                             flags_tmax == qa_temp.QA_MISSING))

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

    print ("%s: Pct. of Tmin|Tmax observations flagged: %.2f (%d total)|%.2f (%d total)" %
           (stn[STN_ID], pctflg_tmin, nflag_tmin, pctflg_tmax, nflag_tmax))

    mask_all = np.logical_or(mask_tmin, mask_tmax)
    # make sure previous flags are not overwritten in the update
    flags_tmin = set_prev_flags(flags_tmin, prev_flgs_tmin)
    flags_tmax = set_prev_flags(flags_tmax, prev_flgs_tmax)

    return IterFlagUpdate(stn[STN_ID], flags_tmin[mask_all],
                          flags_tmax[mask_all], days[YMD][mask_all])

def set_prev_flags(flags, prev_flgs):

    if prev_flgs.size > 0:

        uniq_flgs = np.unique(prev_flgs)

        for flag in uniq_flgs:

            if flag != "":

                mask_flg = prev_flgs == flag
                mask_set = np.logical_and(mask_flg,
                                          np.logical_or(flags == qa_temp.QA_OK,
                                                        flags == qa_temp.QA_MISSING))

                flags[mask_set] = qa_temp.GHCN_TO_TWX_FLAGS_MAP[flag]

    return flags

def proc_write(twx_cfg, mask_stns, nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0

    iter_all = IterMultiFlagUpdate()

    nstns = np.nonzero(mask_stns)[0].size
    stat_chk = StatusCheck(nstns, 10)

    while 1:

        a_iter = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        if status.tag == TAG_STOPWORK:

            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:
                
                print "Writer: updating QA flags in database..."
                
                iter_all.update_flags(twx_cfg.fpath_stndata_nc_all)
                
                # Recalculate period-of-record for Tmin and Tmax
                # since some stations might now fall below the min
                # por requirement after QA is run
                
                for elem in ['tmin', 'tmax']:
                
                    print ("Updating monthly observation counts for %s from %d to %d... " % 
                           (elem, ymdL(twx_cfg.obs_start_date),
                            ymdL(twx_cfg.obs_end_date)))

                    add_obs_cnt(twx_cfg.fpath_stndata_nc_all, elem,
                                twx_cfg.obs_start_date, twx_cfg.obs_end_date,
                                twx_cfg.stn_agg_chunk)
                    
                    print ("Updating monthly observation counts for %s from %d to %d... " % 
                           (elem, ymdL(twx_cfg.interp_start_date), ymdL(twx_cfg.interp_end_date)))
                    
                    add_obs_cnt(twx_cfg.fpath_stndata_nc_all, elem,
                                twx_cfg.interp_start_date, twx_cfg.interp_end_date,
                                twx_cfg.stn_agg_chunk)
                
                                
                print "Writer: Finished"
                return 0
        else:

            iter_all.add_iter(a_iter)
            stat_chk.increment()

def proc_coord(twx_cfg, mask_stns, nwrkers):

    stndb = StationDataDb(twx_cfg.fpath_stndata_nc_all)
    stns = stndb.stns[mask_stns]

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
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--spatial", help="run spatial QA checks",
                        action="store_true")

    args = parser.parse_args()

    # Need to run this QA script twice
    # First run, only apply non-spatial QA checks (qa_spatial = False)
    # Second run, only apply spatial QA checks (qa_spatial = True)
    qa_spatial = args.spatial
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    np.seterr(invalid='ignore')

    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()
    
    print "Process %d of %d. Is spatial QA: %s"%(rank,nsize,qa_spatial)
    
    # Only run QA for SNOTEL and RAWS stations since GHCH-D observations
    # are already flagged by the Durre et al. 2010 procedures
    stndb = StationDataDb(twx_cfg.fpath_stndata_nc_all)
    mask_stns = stndb.stns_df.provider.isin(['NRCS','WRCC']).values
    stndb.ds.close()
    stndb = None
    
    #mask_stns[0:-100] = False

    if rank == RANK_COORD:
        proc_coord(twx_cfg, mask_stns, nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(twx_cfg, mask_stns, nsize - N_NON_WRKRS)
    else:
        proc_work(twx_cfg, qa_spatial, rank)
