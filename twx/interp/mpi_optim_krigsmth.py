'''
A MPI driver for performing "leave one out" cross-validation of tair interpolation in interp_tair

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import StationSerialDataDb,STN_ID,MEAN_OBS,MASK,BAD
from twx.interp.station_select import station_select
from twx.utils.status_check import StatusCheck
import twx.interp.interp_tair as it
import netCDF4
from netCDF4 import Dataset
import rpy2.robjects as robjects
r = robjects.r

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_RLIB = 'P_PATH_RLIB'
P_VARNAME = 'P_VARNAME'
P_GWP_RNG = 'P_GWP_RNG'

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

    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    mask_stns = np.isnan(stn_da.stns[BAD])         
    stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
          
    it.init_interp_R_env(params[P_PATH_RLIB])
        
    krig = it.KrigTair2(stn_slct)
        
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*2, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:

                xval_stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
                
                incis = np.zeros(params[P_GWP_RNG].size)
                x=0
                for gw_p in params[P_GWP_RNG]:
                
                    tair_mean,tair_var = krig.krig(xval_stn, np.array([xval_stn[STN_ID]]),smth_nnghs=False,gw_p=gw_p)
                    std_err,ci = krig.std_err_ci(tair_mean, tair_var)
                
                    in_ci = True if  xval_stn[MEAN_OBS] >= ci[0] and  xval_stn[MEAN_OBS] <= ci[1] else False
                    incis[x] = in_ci
                    x+=1
                            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not xval ",stn_id,str(e)])
                incis = np.ones(params[P_GWP_RNG].size)*netCDF4.default_fillvals['f8']
                            
            MPI.COMM_WORLD.send((stn_id,incis), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    stn_ids = stn_da.stn_ids
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD]))    
    stns = stn_da.stns[stn_mask]
    stn_da.ds.close()
    stn_da = None
        
    ds = Dataset(params[P_PATH_DB],'r+')
    avarname = 'inci'
    
    if avarname not in ds.variables.keys():
            
        ds.createDimension('gw_p',params[P_GWP_RNG].size)
        avar = ds.createVariable(avarname,'f8',('gw_p','stn_id'),fill_value=netCDF4.default_fillvals['f8'])
        avar.long_name = avarname
        avar.units = 'NA'
        ds.sync()
        
    else:
            
        avar = ds.variables[avarname]
    
    avar[:] = netCDF4.default_fillvals['f8']
    ds.sync()
    
    stat_chk = StatusCheck(stns.size,250)
    
    while 1:
       
        stn_id,incis = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            x = np.nonzero(stn_ids==stn_id)[0][0]
            avar[:,x] = incis
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
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc"
    params[P_VARNAME] = 'tmax'
    params[P_PATH_RLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R'
    params[P_GWP_RNG] = np.array([1,2,5,10])
        
    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
