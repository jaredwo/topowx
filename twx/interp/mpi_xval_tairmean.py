'''
A MPI driver for performing "leave one out" cross-validation of tair interpolation in interp_tair

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import StationSerialDataDb,STN_ID,MEAN_OBS,MASK,BAD,DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO
from twx.interp.station_select import station_select
from twx.utils.status_check import StatusCheck
import twx.interp.interp_tair as it
import netCDF4
from netCDF4 import Dataset
import rpy2.robjects as robjects
from twx.interp.optimize import XvalTairMean
r = robjects.r

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_DB_XVAL = 'P_PATH_DB_XVAL'
P_PATH_RLIB = 'P_PATH_RLIB'
P_PATH_RMSTNS = 'P_PATH_RMSTNS'
P_PATH_PARAMS_MEAN = 'P_PATH_PARAMS_MEAN'

P_MAX_NNGH_DELTA = 'P_MAX_NNGH_DELTA'
P_NGH_RNG = 'P_NGH_RNG'
P_NGH_RNG_STEP = 'P_NGH_RNG_STEP'
P_VARNAME = 'P_VARNAME'
P_VARNAME_XVAL = 'P_VARNAME_XVAL'
P_NEON_ECORGN = 'P_NEON_ECORGN'
P_GWP_RNG = 'P_GWP_RNG'
P_RESTART_NNGH = 'P_RESTART_NNGH'

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

def proc_work(params,rank):
    
    status = MPI.Status()
    
    xval = XvalTairMean(params[P_PATH_DB], params[P_PATH_RLIB], params[P_VARNAME])
        
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*3, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                err,std_err = xval.runXval(stn_id)
                #in_ci = True if  xval_stn[MEAN_OBS] >= ci[0] and  xval_stn[MEAN_OBS] <= ci[1] else False
                            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not xval ",stn_id,str(e)])
                
                err = np.ones(13)*netCDF4.default_fillvals['f8']
                std_err = np.ones(13)*netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id,err,std_err), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME],stn_dtype=DTYPE_STN_MEAN_LST_TDI_OPTIMNNGH_VARIO)
    stn_ids = stn_da.stn_ids
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD]))    
    stns = stn_da.stns[stn_mask]
    stn_da.ds.close()
    stn_da = None
        
    ds = Dataset(params[P_PATH_DB],'r+')
    dsvars = ('xval_err_mthly','xval_stderr_mthly')
    
    if 'time_mthly_err' not in ds.dimensions.keys():
        ds.createDimension('time_mthly_err',13)
        ds.sync()
    
    for avarname in dsvars:
    
        if avarname not in ds.variables.keys():
                
            avar = ds.createVariable(avarname,'f8',('time_mthly_err','stn_id'),fill_value=netCDF4.default_fillvals['f8'])
            avar.long_name = avarname
            avar.units = 'NA'
            ds.sync()
        
        else:
            
            avar = ds.variables[avarname]
    
        avar[:] = netCDF4.default_fillvals['f8']
        ds.sync()
    
    stat_chk = StatusCheck(stns.size,250)
    
    while 1:
       
        stn_id,err,std_err = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            x = np.nonzero(stn_ids==stn_id)[0][0]
            ds.variables['xval_err_mthly'][:,x] = err
            ds.variables['xval_stderr_mthly'][:,x] = std_err
            ds.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD]))    
    stns = stn_da.stns[stn_mask]
            
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
                
    for stn_id in stns[STN_ID]:
            
        if cnt < nwrkers:
            dest = cnt+N_NON_WRKRS
        else:
            dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            nrec+=1

        MPI.COMM_WORLD.send(stn_id, dest=dest, tag=TAG_DOWORK)
        cnt+=1
                        
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send(None, dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc"
    params[P_VARNAME] = 'tmax'
    params[P_PATH_RLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R'
        
    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
