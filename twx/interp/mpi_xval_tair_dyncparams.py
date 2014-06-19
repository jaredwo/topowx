'''
A MPI driver for performing "leave one out" cross-validation of tair interpolation using optimized parameters.

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import StationSerialDataDb,STN_ID,MEAN_OBS,NEON
from twx.interp.station_select import station_select
from twx.utils.status_check import status_check
import twx.interp.interp_tair as it
import netCDF4
from twx.interp_constants import *
from twx.db.create_db_all_stations import dbDataset
import rpy2.robjects as robjects
import time
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
P_PATH_CLIB = 'P_PATH_CLIB'
P_PATH_RMSTNS = 'P_PATH_RMSTNS'
P_PATH_GLOBAL_VARIO = 'P_PATH_GLOBAL_VARIO'
P_PATH_GLOBAL_VARIO_BINMASKS = 'P_PATH_GLOBAL_VARIO_BINMASKS'
P_PATH_PARAMS_MEAN = 'P_PATH_PARAMS_MEAN'
P_PATH_PARAMS_ANOM = 'P_PATH_PARAMS_ANOM'

P_VARNAME = 'P_VARNAME'
P_VARNAME_XVAL = 'P_VARNAME_XVAL'

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
    stn_da_xval = StationSerialDataDb(params[P_PATH_DB_XVAL], params[P_VARNAME_XVAL])
    mask_stns = it.build_stn_mask(stn_da.stn_ids, params[P_PATH_RMSTNS])    
    stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
    
    p_mean = it.load_neon_params_mean(params[P_PATH_PARAMS_MEAN])
    p_anom = it.load_neon_params_anom(params[P_PATH_PARAMS_ANOM])
       
    it.init_krig_R_env(params[P_PATH_RLIB])
    
#    krig_params = it.KrigParamsDynamic(stn_slct, p_mean, NEON)
#    krig = it.KrigTair(stn_slct, krig_params)
    
    krig_params = it.KrigParamsDynamic2(stn_slct, p_mean, NEON)
    krig = it.KrigTair2(stn_slct, krig_params)
    
    gwrpca = it.GwrPcaTairDynamic(stn_slct, params[P_PATH_CLIB], p_anom, NEON, set_pt=False)
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*10, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                #start = time.time()
                xval_stn = stn_da_xval.stns[stn_da.stn_idxs[stn_id]]
                xval_mean = xval_stn[MEAN_OBS]
                
                xval_obs = stn_da_xval.load_obs(xval_stn[STN_ID])
                xval_anom = xval_obs - xval_stn[MEAN_OBS]
                
                rm_stnid = np.array([xval_stn[STN_ID]])
                
                tair_mean,tair_var = krig.krig(xval_stn, np.array([xval_stn[STN_ID]]))
                std_err,ci = krig.std_err_ci(tair_mean, tair_var)
                
                mean_bias = tair_mean - xval_mean
                mean_mae = np.abs(mean_bias)
                cir = ci[1] - ci[0]
                in_ci = True if  xval_stn[MEAN_OBS] >= ci[0] and  xval_stn[MEAN_OBS] <= ci[1] else False
        
                gwrpca.setup_for_pt(xval_stn,rm_stnid)
                tair_daily = gwrpca.gwr_pca()
                interp_anom = tair_daily - xval_stn[MEAN_OBS]                
                difs = interp_anom - xval_anom
                anom_bias = np.mean(difs)
                anom_mae = np.mean(np.abs(difs))
                
                xval_stn[MEAN_OBS] = tair_mean
                tair_daily = gwrpca.gwr_pca()
                difs = tair_daily - xval_obs
                overall_bias = np.mean(difs)
                overall_mae = np.mean(np.abs(difs))
            
            except Exception as e:
                
                print "ERROR WITH STN: "+xval_stn[STN_ID]
                raise(e)
                
                
            
            #end = time.time()
            
            #print "".join(["Worker ",str(rank),": ",str(end-start)])
            
            MPI.COMM_WORLD.send((stn_id,mean_mae,mean_bias,anom_mae,anom_bias,overall_mae,overall_bias,std_err,cir,in_ci), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids = bcast_msg
    print "Writer: Received broadcast msg"
    
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    stn_mask = np.in1d(stn_da.stn_ids,stn_ids,True)
    stns = stn_da.stns[stn_mask]
    
    ds = create_ncdf(params, stns)
        
    print "Writer: Output NCDF file created"
    
    stn_idxs = {}
    for x in np.arange(stns.size):
        stn_idxs[stns[STN_ID][x]] = x
            
    stat_chk = status_check(stn_ids.size,250)
    
    while 1:
       
        stn_id,mean_mae,mean_bias,anom_mae,anom_bias,overall_mae,overall_bias,std_err,cir,in_ci = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            dim1 = stn_idxs[stn_id]
            stn = stns[stn_idxs[stn_id]]
            
            ds.variables['mean_mae'][dim1] = mean_mae
            ds.variables['mean_bias'][dim1] = mean_bias
            ds.variables['anom_mae'][dim1] = anom_mae
            ds.variables['anom_bias'][dim1] = anom_bias
            ds.variables['overall_mae'][dim1] = overall_mae
            ds.variables['overall_bias'][dim1] = overall_bias
            ds.variables['std_err'][dim1] = std_err
            ds.variables['in_ci'][dim1] = in_ci
            ds.variables['neon'][dim1] = stn[NEON]
            ds.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    mask_stns = it.build_stn_mask(stn_da.stn_ids,params[P_PATH_RMSTNS])
    stns = stn_da.stns[np.logical_and(mask_stns,np.isfinite(stn_da.stns[NEON]))]
        
    #Send stn ids to all processes
    MPI.COMM_WORLD.bcast(stns[STN_ID], root=RANK_COORD)
    
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

def create_err_var(varname,ds):
    
    mae_var = ds.createVariable(varname,'f8',('stn_id',))
    mae_var.long_name = varname
    mae_var.units = "C"
    mae_var.standard_name = varname
    mae_var.missing_value = netCDF4.default_fillvals['f8']

def create_ncdf(params,stns):
    
    fpath = params[P_PATH_OUT]
    
    ds = dbDataset(fpath,'w')
    ds.db_create_global_attributes("Cross Validation "+params[P_VARNAME])
    
    ds.db_create_stnid_dimvar(stns[STN_ID])

    create_err_var("mean_mae", ds)
    create_err_var("mean_bias", ds)
    create_err_var("anom_mae", ds)
    create_err_var("anom_bias", ds)
    create_err_var("overall_mae", ds)
    create_err_var("overall_bias", ds)
        
    in_ci = ds.createVariable('in_ci','i1',('stn_id',))
    in_ci.long_name = "in prediction interval"
    in_ci.missing_value = netCDF4.default_fillvals['i1']
    
    std_err = ds.createVariable('std_err','f8',('stn_id',))
    std_err.long_name = "standard air"
    std_err.missing_value = netCDF4.default_fillvals['f8']
    
    neon = ds.createVariable('neon','f8',('stn_id',))
    neon.long_name = "neon"
    neon.missing_value = netCDF4.default_fillvals['f8']
    
    ds.sync()
    
    return ds

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc"
    params[P_PATH_DB_XVAL] = '/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/infill_fnl/xval/xval_tmin_overall.nc'
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_PATH_RMSTNS] = "/projects/daymet2/station_data/infill/infill_fnl/rm_stns_all.csv" 
    params[P_PATH_RLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R'
    params[P_PATH_PARAMS_MEAN] = '/projects/daymet2/station_data/infill/infill_fnl/param_files/neon_mean_tmin_params.csv'
    params[P_PATH_PARAMS_ANOM] = '/projects/daymet2/station_data/infill/impute_tair/param_files/neon_anom_tmin_params.csv'
    params[P_VARNAME] = 'tmin'
    params[P_VARNAME_XVAL] = 'tmin'
    
    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
