'''
A MPI driver for performing "leave one out" cross-validation of tair interpolation in interp_tair

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import station_data_infill,STN_ID,MASK,BAD
from twx.utils.status_check import status_check
import netCDF4
from netCDF4 import Dataset
import rpy2.robjects as robjects
from twx.interp.optimize import XvalTairOverall
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
    
    xval = XvalTairOverall(params[P_PATH_DB], params[P_VARNAME])
        
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*7, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                biasNorm,maeNorm,maeDly,biasDly,r2Dly,seNorm = xval.runXval(stn_id)
                                            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not xval ",stn_id,str(e)])
                
                ndata = np.ones(13)*netCDF4.default_fillvals['f8']
                biasNorm = ndata
                maeNorm = ndata
                maeDly = ndata
                biasDly = ndata
                r2Dly = ndata
                seNorm = ndata[0:12]
            
            MPI.COMM_WORLD.send((stn_id,biasNorm,maeNorm,maeDly,biasDly,r2Dly,seNorm), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    stn_ids = stn_da.stn_ids
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD]))    
    stns = stn_da.stns[stn_mask]
    stn_da.ds.close()
    stn_da = None
    ds = Dataset(params[P_PATH_DB],'r+')
    dsvars = ('xvalfnl_bias_norm','xvalfnl_mae_norm','xvalfnl_bias_dly','xvalfnl_mae_dly','xvalfnl_r2_dly','xvalfnl_se_norm')
    
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
    
    stat_chk = status_check(stns.size,250)
    
    while 1:
       
        stn_id,biasNorm,maeNorm,maeDly,biasDly,r2Dly,seNorm = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            x = np.nonzero(stn_ids==stn_id)[0][0]
            ds.variables['xvalfnl_bias_norm'][:,x] = biasNorm
            ds.variables['xvalfnl_mae_norm'][:,x] = maeNorm
            ds.variables['xvalfnl_bias_dly'][:,x] = biasDly
            ds.variables['xvalfnl_mae_dly'][:,x] = maeDly
            ds.variables['xvalfnl_r2_dly'][:,x] = r2Dly
            ds.variables['xvalfnl_se_norm'][0:12,x] = seNorm
            ds.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
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
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc"
    params[P_VARNAME] = 'tmax'
        
    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
