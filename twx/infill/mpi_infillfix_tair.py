'''
A MPI driver for trying different nnghs on stations that had an Tair infilling MAE greater than a specified threshold. 

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from db.station_data import station_data_ncdb
from utils.status_check import status_check
from netCDF4 import Dataset
import netCDF4
from infill.infill_daily import pca_matrix,tmin_tmax_fixer
import os

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_POR = 'P_PATH_POR'
P_PATH_NORMALS = 'P_PATH_NORMALS'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_NNGH_TMIN = 'P_NNGH_TMIN'
P_NNGH_TMAX = 'P_NNGH_TMAX'
P_NNGH_TMIN_NEW = 'P_NNGH_TMIN_NEW'
P_NNGH_TMAX_NEW = 'P_NNGH_TMAX_NEW'
P_NCDF_MODE = 'P_NCDF_MODE'
P_SORT_AND_CENTER = 'P_SORT_AND_CENTER'
P_MAE_RERUN_TMIN = 'P_MAE_RERUN_TMIN'
P_MAE_RERUN_TMAX = 'P_MAE_RERUN_TMAX'

NCDF_CHK_COLS = 50

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

def load_all_infilled_data(ds,stn_ids,var,stn_id):
    
    idx = np.nonzero(stn_ids==stn_id)[0][0]
    
    vals = ds.variables[var][:,idx].ravel()
    flag_fill = ds.variables['flag_fill'][:,idx].ravel()
    mae = ds.variables['mae'][idx]
    bias = ds.variables['bias'][idx]
    npcs = ds.variables['npcs'][idx]
    nnghs = ds.variables['nnghs'][idx]
    max_dist = ds.variables['max_dist'][idx]
    
    return vals,flag_fill,npcs,nnghs,max_dist,mae,bias 

def proc_work(params,rank):
    
    status = MPI.Status()
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    ndays = float(days.size)
    
    empty_fill = np.ones(ndays,dtype=np.float32)*netCDF4.default_fillvals['f4']
    empty_flags = np.ones(ndays,dtype=np.int8)*netCDF4.default_fillvals['i1']
    empty_ngh_radius = netCDF4.default_fillvals['f4']
    empty_npcs = netCDF4.default_fillvals['i2']
    empty_nngh = netCDF4.default_fillvals['i2']
    empty_bias = netCDF4.default_fillvals['f4']
    empty_mae = netCDF4.default_fillvals['f4']
    
    ds_norms = Dataset(params[P_PATH_NORMALS])
    norms = np.array(ds_norms.variables['norm'][:],dtype=np.float)
    norms_tmin = norms[0,:]
    norms_tmax = norms[1,:]
    ds_norms.close()
    
    ds_tmin = Dataset("".join([params[P_PATH_OUT],'infill_tmin.nc']))
    ds_tmax = Dataset("".join([params[P_PATH_OUT],'infill_tmax.nc']))
    
    mae_tmins = ds_tmin.variables['mae'][:]
    mae_tmaxs = ds_tmax.variables['mae'][:]
    
    stn_ids_tmin = ds_tmin.variables['stn_id'][:].astype("<S16")
    stn_ids_tmax = ds_tmax.variables['stn_id'][:].astype("<S16")
    
    while 1:
    
        stn_id,fix_tmin,fix_tmax,tmin_stn,tmax_stn = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)

        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*9, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                tmin_fixed = False
                tmax_fixed = False
                
                if fix_tmin:
                    
                    prev_mae_tmin = mae_tmins[stn_ids_tmin==stn_id][0]
                    
                    a_pca_matrix = pca_matrix(stn_id, stn_da,'tmin', norms_tmin)
                    fnl_tmin,fill_mask_tmin,npcs_tmin,fnl_nnghs_tmin,max_dist_tmin,mae_tmin,bias_tmin = infill_tair(a_pca_matrix, params[P_NNGH_TMIN_NEW])
                    
                    if np.round(mae_tmin,4) < np.round(prev_mae_tmin,4):
                        print "|".join(["WORKER",stn_id,"TMIN MAE-IMPROVED","%.4f"%(prev_mae_tmin,),"%.4f"%(mae_tmin,),"%.4f"%(mae_tmin-prev_mae_tmin,)])
                        tmin_fixed = True
                    else:
                        print "|".join(["WORKER",stn_id,"TMIN MAE-NOT-IMPROVED","%.4f"%(prev_mae_tmin,),"%.4f"%(mae_tmin,),"%.4f"%(mae_tmin-prev_mae_tmin,)])
                
                ##########################################################
                if fix_tmax:
                    
                    prev_mae_tmax = mae_tmaxs[stn_ids_tmax==stn_id][0]
                    
                    a_pca_matrix = pca_matrix(stn_id, stn_da,'tmax', norms_tmax)
                    fnl_tmax,fill_mask_tmax,npcs_tmax,fnl_nnghs_tmax,max_dist_tmax,mae_tmax,bias_tmax = infill_tair(a_pca_matrix, params[P_NNGH_TMAX_NEW])
                
                    if np.round(mae_tmax,4) < np.round(prev_mae_tmax,4):
                        print "|".join(["WORKER",stn_id,"TMAX MAE-IMPROVED","%.4f"%(prev_mae_tmax,),"%.4f"%(mae_tmax,),"%.4f"%(mae_tmax-prev_mae_tmax,)])
                        tmax_fixed = True
                    else:
                        print "|".join(["WORKER",stn_id,"TMAX MAE-NOT-IMPROVED","%.4f"%(prev_mae_tmax,),"%.4f"%(mae_tmax,),"%.4f"%(mae_tmax-prev_mae_tmax,)])
                
                ########################################################
                if not tmin_fixed and tmin_stn and tmax_fixed:
                    a_pca_matrix = pca_matrix(stn_id, stn_da,'tmin', norms_tmin)
                    fnl_tmin,fill_mask_tmin,npcs_tmin,fnl_nnghs_tmin,max_dist_tmin,mae_tmin,bias_tmin = infill_tair(a_pca_matrix, params[P_NNGH_TMIN])
                elif not tmin_fixed and tmin_stn and not tmax_fixed:
                    fnl_tmin,fill_mask_tmin,npcs_tmin,fnl_nnghs_tmin,max_dist_tmin,mae_tmin,bias_tmin = load_all_infilled_data(ds_tmin, stn_ids_tmin, "tmin", stn_id)
                            
                if not tmax_fixed and tmax_stn and tmin_fixed:
                    a_pca_matrix = pca_matrix(stn_id, stn_da,'tmax', norms_tmax)
                    fnl_tmax,fill_mask_tmax,npcs_tmax,fnl_nnghs_tmax,max_dist_tmax,mae_tmax,bias_tmax = infill_tair(a_pca_matrix, params[P_NNGH_TMAX])
                elif not tmax_fixed and tmax_stn and not tmin_fixed:
                    fnl_tmax,fill_mask_tmax,npcs_tmax,fnl_nnghs_tmax,max_dist_tmax,mae_tmax,bias_tmax = load_all_infilled_data(ds_tmax, stn_ids_tmax, "tmax", stn_id)
                
                if (fix_tmin or tmin_stn)  and (fix_tmax or tmax_stn):
                    fnl_tmin,fnl_tmax,ninvalid = tmin_tmax_fixer(fnl_tmin, fnl_tmax)
                    
                    if ninvalid > 0:
                        print "".join([stn_id,": percent of obs tmin >= tmax: %.4f"%(ninvalid/ndays*100.,)])
            
            except Exception as e:
                
                print "".join(["ERROR: Could not infill ",stn_id,"|",str(e)])
                MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                continue
                
#                if fix_tmin or tmin_stn:
#                    fnl_tmin,fill_mask_tmin,npcs_tmin,fnl_nnghs_tmin,max_dist_tmin,mae_tmin,bias_tmin = empty_fill,empty_flags,empty_npcs,empty_nngh,empty_ngh_radius,empty_mae,empty_bias
#                if fix_tmax or tmax_stn:
#                    fnl_tmax,fill_mask_tmax,npcs_tmax,fnl_nnghs_tmax,max_dist_tmax,mae_tmax,bias_tmax = empty_fill,empty_flags,empty_npcs,empty_nngh,empty_ngh_radius,empty_mae,empty_bias
            
            if fix_tmin or tmin_stn:
                MPI.COMM_WORLD.send((stn_id,'tmin',fnl_tmin,fill_mask_tmin,npcs_tmin,fnl_nnghs_tmin,max_dist_tmin,mae_tmin,bias_tmin), dest=RANK_WRITE, tag=TAG_DOWORK)
            if fix_tmax or tmax_stn:
                MPI.COMM_WORLD.send((stn_id,'tmax',fnl_tmax,fill_mask_tmax,npcs_tmax,fnl_nnghs_tmax,max_dist_tmax,mae_tmax,bias_tmax), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    ds_tmin = Dataset("".join([params[P_PATH_OUT],'infill_tmin.nc']),"r+")
    ds_tmax = Dataset("".join([params[P_PATH_OUT],'infill_tmax.nc']),"r+")
        
    mae_tmin = ds_tmin.variables['mae'][:]
    mae_tmax = ds_tmax.variables['mae'][:]
    
    stn_ids_tmin = ds_tmin.variables['stn_id'][:].astype("<S16")
    stn_ids_tmax = ds_tmax.variables['stn_id'][:].astype("<S16")
    
    stn_ids_rerun_tmin =  stn_ids_tmin[mae_tmin >= params[P_MAE_RERUN_TMIN]]
    stn_ids_rerun_tmax =  stn_ids_tmax[mae_tmax >= params[P_MAE_RERUN_TMAX]]
    stn_ids_all = np.unique(np.concatenate([stn_ids_rerun_tmin,stn_ids_rerun_tmax]))
    
    if params[P_NCDF_MODE] == "r+":
        
        ds_rst = Dataset("".join([params[P_PATH_OUT],'restart_fixtair.nc']),"r+")
        stn_ids_all = ds_rst.variables['stn_id'][:].astype("<S16")
    
    else:
        
        ds_rst = create_restart_file("".join([params[P_PATH_OUT],'restart_fixtair.nc']), stn_ids_all)
    
    a = np.sum(np.logical_and(np.in1d(stn_ids_rerun_tmin, stn_ids_tmax, assume_unique=True),np.logical_not(np.in1d(stn_ids_rerun_tmin, stn_ids_rerun_tmax, assume_unique=True))))
    b = np.sum(np.logical_and(np.in1d(stn_ids_rerun_tmax, stn_ids_tmin, assume_unique=True),np.logical_not(np.in1d(stn_ids_rerun_tmax, stn_ids_rerun_tmin, assume_unique=True))))
    
    ttl_infills = stn_ids_rerun_tmin.size + stn_ids_rerun_tmax.size + a + b
    
    stat_chk = status_check(ttl_infills,10)
    
    ttl_infills = ttl_infills - np.sum(ds_rst.variables['tmin'][:]) - np.sum(ds_rst.variables['tmax'][:]) 
    
    while 1:

        stn_id,tair_var,tair,fill_mask,npcs,nnghs,max_dist,mae,bias = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                if params[P_SORT_AND_CENTER]:
                
                    print "Writer: Sorting netcdf output by stn_id..."
                    sort_tair_ds(ds_tmin, "tmin")
                    sort_tair_ds(ds_tmax, "tmax")
                    ds_tmin.close()
                    ds_tmax.close()
                    
                    print "Writer: Creating centered versions of output datasets..."
                    
                    path_tmin = "".join([params[P_PATH_OUT],'infill_tmin.nc'])
                    path_tmax = "".join([params[P_PATH_OUT],'infill_tmax.nc'])
                    
                    path_tmin_center = "".join([params[P_PATH_OUT],'infill_tmin_center.nc'])
                    path_tmax_center = "".join([params[P_PATH_OUT],'infill_tmax_center.nc'])
                    
                    os.system("cp "+path_tmin+" "+path_tmin_center)
                    os.system("cp "+path_tmax+" "+path_tmax_center)
                    
                    ds_tmin_center = Dataset(path_tmin_center,'r+')
                    ds_tmax_center = Dataset(path_tmax_center,'r+')
                    
                    center_dataset(ds_tmin_center, 'tmin')
                    center_dataset(ds_tmax_center, 'tmax')
                
                print "Writer: Finished"
                return 0
        else:
            
            print "|".join(["WRITER",stn_id,tair_var,str(npcs),str(nnghs),"%.2f"%(max_dist,),"%.4f"%(mae,),"%.4f"%(bias,)])
            
            if tair_var == 'tmin':
                stn_idx = np.nonzero(stn_ids_tmin == stn_id)[0][0]
                ds = ds_tmin
            else:
                stn_idx = np.nonzero(stn_ids_tmax == stn_id)[0][0]
                ds = ds_tmax
            
            do_write = True
            
            if params[P_NCDF_MODE] == "r+":
            
                cur_mae = ds.variables['mae'][stn_idx]
            
            ds.variables['npcs'][stn_idx] = npcs
            ds.variables['mae'][stn_idx] = mae
            ds.variables['bias'][stn_idx] = bias
            ds.variables['max_dist'][stn_idx] = max_dist
            ds.variables[tair_var][:,stn_idx] = tair
            ds.variables["".join([tair_var,"_mean"])][stn_idx] = np.mean(tair,dtype=np.float64)
            ds.variables['flag_fill'][:,stn_idx] = fill_mask
            ds.variables['nnghs'][stn_idx] = nnghs
            ds.sync()
            
            stn_idx_rst = np.nonzero(stn_ids_all == stn_id)[0][0]
            ds_rst.variables[tair_var][stn_idx_rst] = 1
            ds_rst.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
        
    ds_tmin = Dataset("".join([params[P_PATH_OUT],'infill_tmin.nc']))
    ds_tmax = Dataset("".join([params[P_PATH_OUT],'infill_tmax.nc']))
    
    mae_tmin = ds_tmin.variables['mae'][:]
    mae_tmax = ds_tmax.variables['mae'][:]
    
    stn_ids_tmin = ds_tmin.variables['stn_id'][:].astype("<S16")
    stn_ids_tmax = ds_tmax.variables['stn_id'][:].astype("<S16")
    
    stn_ids_rerun_tmin =  stn_ids_tmin[mae_tmin >= params[P_MAE_RERUN_TMIN]]
    stn_ids_rerun_tmax =  stn_ids_tmax[mae_tmax >= params[P_MAE_RERUN_TMAX]]
    stn_ids_all = np.unique(np.concatenate([stn_ids_rerun_tmin,stn_ids_rerun_tmax]))
    
    ds_tmin.close()
    ds_tmax.close()
    
    if params[P_NCDF_MODE] == "r+":
        
        ds_rst = Dataset("".join([params[P_PATH_OUT],'restart_fixtair.nc']))
        stn_ids_all = ds_rst.variables['stn_id'][:].astype("<S16")
    
    else:
        
        ds_rst = None
        
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    stn_sent = True
    for stn_id in stn_ids_all:
        
        if stn_sent:
            if cnt < nwrkers:
                dest = cnt+N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                nrec+=1
        
        fix_tmin = stn_id in stn_ids_rerun_tmin
        fix_tmax = stn_id in stn_ids_rerun_tmax
        tmin_stn = stn_id in stn_ids_tmin
        tmax_stn = stn_id in stn_ids_tmax
        
        send_stn = False
        
        if ds_rst is not None:
            
            stn_idx = np.nonzero(stn_ids_all == stn_id)[0][0]
            
            tmin_flg = np.bool(ds_rst.variables['tmin'][stn_idx])
            tmax_flg = np.bool(ds_rst.variables['tmax'][stn_idx])
            
            if (not tmin_flg and tmin_stn) or (not tmax_flg and tmax_stn):
                
                send_stn = True
        
        else:
            
            send_stn = True
        
        if send_stn:
            MPI.COMM_WORLD.send((stn_id,fix_tmin,fix_tmax,tmin_stn,tmax_stn), dest=dest, tag=TAG_DOWORK)
            cnt+=1
            stn_sent = True
        else:
            #print "Did not send ",stn_id
            stn_sent = False
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send([None]*5, dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

def create_restart_file(path_out,stn_ids):
    
    ds = Dataset(path_out,'w')
    ds.createDimension('stn_id',stn_ids.size)
    
    stations = ds.createVariable('stn_id','str',('stn_id',))
    stations[:] = np.array(stn_ids,dtype=np.object)
    
    ds.createVariable('tmin','i1',('stn_id',),fill_value=0)
    ds.createVariable('tmax','i1',('stn_id',),fill_value=0)
    ds.sync()
    
    return ds

def infill_tair(a_pca_matrix,nngh):

    fit_tair, obs_tair, npcs, fnl_nnghs, max_dist = a_pca_matrix.infill(nngh)

    fnl_tair = np.copy(obs_tair)
    fill_mask = np.isnan(fnl_tair)
    fnl_tair[fill_mask] = fit_tair[fill_mask]
    
    fin_mask = np.logical_not(fill_mask)
    difs = fit_tair[fin_mask] - obs_tair[fin_mask]
    mae = np.mean(np.abs(difs))
    bias = np.mean(difs)
    
    return fnl_tair,fill_mask,npcs,fnl_nnghs,max_dist,mae,bias

def center_dataset(ds,var_name):
    
    vals = np.require(ds.variables[var_name][:], dtype=np.float64)
    mean_vals = np.require(ds.variables["".join([var_name,"_mean"])][:], dtype=np.float64)
    vals_cent = vals - mean_vals
    vals = None
    
    ds.variables[var_name][:] = vals_cent
    ds.sync()

def sort_tair_ds(ds,tair_var):
    
    stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
    idx_s = np.argsort(stn_ids)

    ds.variables['stn_id'][:] = np.array(stn_ids[idx_s],dtype=np.object)
    ds.sync()
    
    name = ds.variables['name'][:]
    ds.variables['name'][:] = np.array(name[idx_s],dtype=np.object)
    ds.sync()
    
    state = ds.variables['state'][:]
    ds.variables['state'][:] = np.array(state[idx_s],dtype=np.object)
    ds.sync()
    
    lat = ds.variables['lat'][:]
    ds.variables['lat'][:] = lat[idx_s]
    ds.sync()
    
    lon = ds.variables['lon'][:]
    ds.variables['lon'][:] = lon[idx_s]
    ds.sync()
    
    elev = ds.variables['elev'][:]
    ds.variables['elev'][:] = elev[idx_s]
    ds.sync()

    mae = ds.variables['mae'][:]
    ds.variables['mae'][:] = mae[idx_s]
    ds.sync()
    
    bias = ds.variables['bias'][:]
    ds.variables['bias'][:] = bias[idx_s]
    ds.sync()

    npcs = ds.variables['npcs'][:]
    ds.variables['npcs'][:] = npcs[idx_s]
    ds.sync()

    nnghs = ds.variables['nnghs'][:]
    ds.variables['nnghs'][:] = nnghs[idx_s]
    ds.sync()
    
    max_dist = ds.variables['max_dist'][:]
    ds.variables['max_dist'][:] = max_dist[idx_s]
    ds.sync()
    
    tair_mean = ds.variables["".join([tair_var,"_mean"])][:]
    ds.variables["".join([tair_var,"_mean"])][:] = tair_mean[idx_s]
    ds.sync()
    
    tair = ds.variables[tair_var][:]
    ds.variables[tair_var][:] = tair[:,idx_s]
    ds.sync()
    tair = None

    flag = ds.variables['flag_fill'][:]
    ds.variables['flag_fill'][:] = flag[:,idx_s]
    ds.sync()

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por.csv'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/infill_tair/'
    params[P_PATH_NORMALS] = '/projects/daymet2/station_data/infill/normals_tair.nc'
    params[P_NCDF_MODE] = 'r+'
    params[P_NNGH_TMIN] = 20
    params[P_NNGH_TMAX] = 22
    params[P_NNGH_TMIN_NEW] = params[P_NNGH_TMIN]+10
    params[P_NNGH_TMAX_NEW] = params[P_NNGH_TMAX]+10
    params[P_SORT_AND_CENTER] = True
    params[P_START_YMD] = None #19480101
    params[P_END_YMD] = None #20111231
    params[P_MAE_RERUN_TMIN] = 1.4776 #90th percentile
    params[P_MAE_RERUN_TMAX] = 1.3383 #90th percentile
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
