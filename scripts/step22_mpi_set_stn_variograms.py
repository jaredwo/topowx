'''
MPI script for setting moving window regression kriging variogram
parameters at each station location based on the U.S. climate
division optimal station bandwidths from step21_mpi_optim_nstns_norms.py.
Adds an exponential variogram nugget, partial sill,
and range station attribute for each month to the serially-complete
station database.

Must be run using mpiexec or mpirun.
'''

from mpi4py import MPI
from twx.db import StationSerialDataDb, \
STN_ID, MASK, BAD, get_krigparam_varname, VARIO_NUG, VARIO_PSILL, VARIO_RNG
from twx.interp import StationKrigParams
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

sys.stdout = Unbuffered(sys.stdout)

def proc_work(fpath_stndb, elem, rank):

    status = MPI.Status()

    kparams = StationKrigParams(fpath_stndb, elem)

    while 1:

        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD, tag=MPI.ANY_TAG,
                                     status=status)

        if status.tag == TAG_STOPWORK:

            MPI.COMM_WORLD.send([None] * 4, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["WORKER ", str(rank), ": Finished"])
            return 0

        else:

            try:

                nug, psill, rng = kparams.get_krig_params(stn_id)

            except Exception as e:

                print "".join(["ERROR: WORKER ", str(rank),
                               ": could not get krig params for ", stn_id, str(e)])
                nug = np.ones(12) * netCDF4.default_fillvals['f8']
                psill = np.ones(12) * netCDF4.default_fillvals['f8']
                rng = np.ones(12) * netCDF4.default_fillvals['f8']

            MPI.COMM_WORLD.send((stn_id, nug, psill, rng), dest=RANK_WRITE,
                                tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)

def proc_write(fpath_stndb, elem, nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0

    stn_da = StationSerialDataDb(fpath_stndb, elem, mode="r+")
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[MASK]),
                               np.isnan(stn_da.stns[BAD]))
    nstns = np.sum(mask_stns)

    dsvars = {}
    for mth in np.arange(1, 13):

        vname_nug = get_krigparam_varname(mth, VARIO_NUG)
        vname_psill = get_krigparam_varname(mth, VARIO_PSILL)
        vname_rng = get_krigparam_varname(mth, VARIO_RNG)

        dsvars[vname_nug] = stn_da.add_stn_variable(vname_nug, vname_nug, "C**2", 'f8')
        dsvars[vname_psill] = stn_da.add_stn_variable(vname_psill, vname_nug, "C**2", 'f8')
        dsvars[vname_rng] = stn_da.add_stn_variable(vname_rng, vname_nug, "km", 'f8')

    stat_chk = StatusCheck(nstns, 250)

    while 1:

        stn_id, nug, psill, rng = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,
                                                      tag=MPI.ANY_TAG, status=status)

        if status.tag == TAG_STOPWORK:

            nwrkrs_done += 1
            if nwrkrs_done == nwrkers:
                print "WRITER: Finished"
                return 0
        else:

            x = stn_da.stn_idxs[stn_id]

            for mth in np.arange(1, 13):

                dsvars[get_krigparam_varname(mth, VARIO_NUG)][x] = nug[mth - 1]
                dsvars[get_krigparam_varname(mth, VARIO_PSILL)][x] = psill[mth - 1]
                dsvars[get_krigparam_varname(mth, VARIO_RNG)][x] = rng[mth - 1]

            stn_da.ds.sync()

            stat_chk.increment()

def proc_coord(fpath_stndb, elem, nwrkers):

    stn_da = StationSerialDataDb(fpath_stndb, elem)
    # Only set kriging params for stations within mask and that are not marked as bad
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[MASK]), np.isnan(stn_da.stns[BAD]))
    stns = stn_da.stns[mask_stns]

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
    elif elem == 'tmax':
        fpath_stndb = twx_cfg.fpath_stndata_nc_serial_tmax
    else:
        raise ValueError("Unrecognized element: " + elem)

    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()
    
    print "Process %d of %d: element is %s" % (rank, nsize, elem)
    
    if rank == RANK_COORD:
        proc_coord(fpath_stndb, elem, nsize - N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(fpath_stndb, elem, nsize - N_NON_WRKRS)
    else:
        proc_work(fpath_stndb, elem, rank)

    MPI.COMM_WORLD.Barrier()
