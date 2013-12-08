'''
A MPI driver for infilling/extending station observation prcp records using the methods of obs_infill_daily. 

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from db.station_data import station_data_ncdb,STN_ID,YEAR,DATE,STN_NAME,ELEV,LON,LAT,STATE
from infill.obs_por import load_por_csv,build_valid_por_masks
from utils.status_check import status_check
from netCDF4 import Dataset,date2num
import netCDF4
import datetime
from infill.infill_daily import infill_prcp,prcp_infill_results,\
    build_yr_mth_masks
from infill.infill_normals import build_mth_masks,MTH_BUFFER
import os
from db.all_create_db import dbDataset

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

LAST_VAR_WRITTEN = 'mth_cor'

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_POR = 'P_PATH_POR'
P_PATH_NORMALS = 'P_PATH_NORMALS'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_NNGH = 'P_NNGH'
P_NCDF_MODE = 'P_NCDF_MODE'
P_PATH_NORMALS_PO = 'P_PATH_NORMALS_PO'

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

def center_dataset(ds,var_name):
    
    vals = np.require(ds.variables[var_name][:], dtype=np.float64)
    mean_vals = np.require(ds.variables["".join([var_name,"_mean"])][:], dtype=np.float64)
    vals_cent = vals - mean_vals
    vals = None
    
    ds.variables[var_name][:] = vals_cent
    ds.sync()

def proc_work(params,rank):
    
    status = MPI.Status()
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    yr_mth_masks = build_yr_mth_masks()
    
    ds_norms = Dataset(params[P_PATH_NORMALS])
        
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            MPI.COMM_WORLD.send(None, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        
        else:
            
            try:
                
                ntries = 0
                nngh_prcp = params[P_NNGH]
                
                while 1:
                    
                    fill_rslts = infill_prcp(stn_id, stn_da, ds_norms, days, mth_masks, mthbuf_masks, nngh_prcp)
                    fill_rslts.calc_obs_vs_fit_stats(yr_mth_masks)
                    ntries+=1
                    
                    if ntries < 2 and (fill_rslts.hss <= 0.0 or 
                                       np.abs(fill_rslts.perr_freq) >= 100 or 
                                       np.abs(fill_rslts.perr_intsy) >= 100 or 
                                       np.abs(fill_rslts.perr_ttlamt) >= 100):
                        
                        nngh_prcp+=1
                        print "".join(["WARNING: ",stn_id," had large error when infilling prcp. Trying again with more nnghs."])
                    
                    else:
                        break
                
                MPI.COMM_WORLD.send(fill_rslts, dest=RANK_WRITE, tag=TAG_DOWORK)
                MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                            
            except Exception as e:
                
                print "".join(["ERROR: Could not infill ",stn_id,"|",str(e)])
                MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stnids_prcp = bcast_msg
    print "Writer: Received broadcast msg"
    
    if params[P_NCDF_MODE] == 'r+':
        
        ds_prcp = Dataset("".join([params[P_PATH_OUT],'infill_prcp.nc']),'r+')
        ttl_infills = stnids_prcp.size
        stnids_prcp = np.array(ds_prcp.variables['stn_id'][:], dtype="<S16")
        
    else:
        
        ds_prcp = create_ncdf("".join([params[P_PATH_OUT],'infill_prcp.nc']), stnids_prcp, stn_da.stns, days)
        ttl_infills = stnids_prcp.size
    
    print "Writer: Output NCDF files ready"
    
    stat_chk = status_check(ttl_infills,10)
    
    while 1:

        infill_rslts = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                ds_prcp.close()
                
                print "Writer: Creating centered versions of output dataset..."
                
                path = "".join([params[P_PATH_OUT],'infill_prcp.nc'])
                
                path_center = "".join([params[P_PATH_OUT],'infill_prcp_center.nc'])
                
                os.system("cp "+path+" "+path_center)
                
                ds_center = Dataset(path_center,'r+')
                
                center_dataset(ds_center, 'prcp')
                
                print "Writer: Finished"
                return 0
        else:
            
            stn_idx = np.nonzero(stnids_prcp == infill_rslts.stn_id)[0][0]
            
            ds_prcp.variables['prcp'][:,stn_idx] = infill_rslts.prcp
            ds_prcp.variables['prcp_mod'][:,stn_idx] = infill_rslts.prcp_fit
            ds_prcp.variables['prcp_mean'][stn_idx] = np.mean(infill_rslts.prcp,dtype=np.float64)
            ds_prcp.variables['flag_fill'][:,stn_idx] = infill_rslts.fill_mask
            ds_prcp.variables['npcs_po'][stn_idx] = infill_rslts.npcs_po
            ds_prcp.variables['nnghs_po'][stn_idx] = infill_rslts.nnghs_po
            ds_prcp.variables['max_dist_po'][stn_idx] = infill_rslts.maxdist_po
            ds_prcp.variables['hss'][stn_idx] = infill_rslts.hss
            ds_prcp.variables['err_amt'][stn_idx] = infill_rslts.perr_ttlamt
            ds_prcp.variables['err_intsy'][stn_idx] = infill_rslts.perr_intsy
            ds_prcp.variables['err_freq'][stn_idx] = infill_rslts.perr_freq
            ds_prcp.variables['var_obs'][stn_idx] = infill_rslts.var_obs
            ds_prcp.variables['var_mod'][stn_idx] = infill_rslts.var_fit
            ds_prcp.variables[LAST_VAR_WRITTEN][stn_idx] = infill_rslts.mth_cor
            ds_prcp.sync()

            print "|".join(["WRITER",infill_rslts.stn_id,"%.4f"%(infill_rslts.hss,),"%.2f"%(infill_rslts.perr_ttlamt,),
                "%.2f"%(infill_rslts.perr_intsy,),"%.2f"%(infill_rslts.perr_freq,)])
            
            stat_chk.increment()


def proc_coord(params,nwrkers):
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    #Load the period-of-record datafile
    por = load_por_csv(params[P_PATH_POR])
    
    mask_por_prcp = build_valid_por_masks(por)[2]
    
    stnids_prcp = stn_da.stn_ids[mask_por_prcp]
    
    #Check if we're restarting a run
    if params[P_NCDF_MODE] == 'r+':
        
        #If rerunning remove stn ids that have already been completed
        ds_prcp = Dataset("".join([params[P_PATH_OUT],'infill_prcp.nc']))
        #LAST_VAR_WRITTEN is the last item to be written, so use its nodata mask
        mask_incplt = ds_prcp.variables[LAST_VAR_WRITTEN][:].mask
        stnids_prcp = stnids_prcp[mask_incplt]
    
    #Send stn ids to all processes
    MPI.COMM_WORLD.bcast(stnids_prcp, root=RANK_COORD)
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in stnids_prcp:
                
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

def create_ncdf(path_out,stnids_prcp,all_stns,days):
    
    stns = all_stns[np.in1d(all_stns[STN_ID], stnids_prcp, assume_unique=True)]
    
    ds = dbDataset(path_out,'w')
    
    ds.db_create_global_attributes('Infilled prcp weather station data')
    ds.db_create_time_dimvar(days)
    ds.db_create_stnid_dimvar(stns[STN_ID])    
    ds.db_create_stn_vars(stns)
    ds.db_create_prcp_var((days.size,NCDF_CHK_COLS))
    
    ncdf_var = ds.createVariable('prcp_mod','f4',('time','stn_id'),chunksizes=(days.size,NCDF_CHK_COLS))
    ncdf_var.long_name = "precipitation amount modeled"
    ncdf_var.units = "cm"
    ncdf_var.standard_name = "precipitation_amount_modeled"
    ncdf_var.missing_value = netCDF4.default_fillvals['f4']
    
    prcp_var = ds.createVariable('prcp_mean','f8',('stn_id',))
    prcp_var.long_name = "mean precipitation"
    prcp_var.units = "cm"
    prcp_var.standard_name = "mean_precipitation"
    prcp_var.missing_value = netCDF4.default_fillvals['f8']
    
    ds.db_create_binflag_var('flag_fill', "infilled flag", "infilled_flag", (days.size,NCDF_CHK_COLS))
      
    npcs_var = ds.createVariable('npcs_po','i2',('stn_id',))
    npcs_var.long_name = "number of principal components precipitation occurrence"
    npcs_var.standard_name = "number_of_principal_components_precipitation_occurrence"
    npcs_var.missing_value = netCDF4.default_fillvals['i2']
        
    nnghs_var = ds.createVariable('nnghs_po','i2',('stn_id',))
    nnghs_var.long_name = "number of neighbor stations precipitation occurrence"
    nnghs_var.standard_name = "number_of_neighbors_precipitation_occurrence"
    nnghs_var.missing_value = netCDF4.default_fillvals['i2']
        
    max_dist_var = ds.createVariable('max_dist_po','f4',('stn_id',))
    max_dist_var.long_name = "max neighbor radius distance precipitation occurrence"
    max_dist_var.units = "km"
    max_dist_var.standard_name = "max neighbor radius distance precipitation occurrence"
    max_dist_var.missing_value = netCDF4.default_fillvals['f4']
    
    ######################################################### 
    err_var = ds.createVariable('hss','f4',('stn_id',))
    err_var.long_name = "heidke skill score"
    err_var.units = "C"
    err_var.standard_name = "heidke_skill_score"
    err_var.missing_value = netCDF4.default_fillvals['f4']
    
    err_var = ds.createVariable('err_amt','f4',('stn_id',))
    err_var.long_name = "percentage error amount"
    err_var.units = "percentage"
    err_var.standard_name = "percentage_error_amount"
    err_var.missing_value = netCDF4.default_fillvals['f4']
    
    err_var = ds.createVariable('err_intsy','f4',('stn_id',))
    err_var.long_name = "percentage error intensity"
    err_var.units = "percentage"
    err_var.standard_name = "percentage_error_intensity"
    err_var.missing_value = netCDF4.default_fillvals['f4']
    
    err_var = ds.createVariable('err_freq','f4',('stn_id',))
    err_var.long_name = "percentage error frequency"
    err_var.units = "percentage"
    err_var.standard_name = "percentage_error_frequency"
    err_var.missing_value = netCDF4.default_fillvals['f4']
    
    err_var = ds.createVariable('var_obs','f4',('stn_id',))
    err_var.long_name = "variance observed"
    err_var.units = "cm"
    err_var.standard_name = "variance observed"
    err_var.missing_value = netCDF4.default_fillvals['f4']
    
    err_var = ds.createVariable('var_mod','f4',('stn_id',))
    err_var.long_name = "variance modeled"
    err_var.units = "cm"
    err_var.standard_name = "variance modeled"
    err_var.missing_value = netCDF4.default_fillvals['f4']
    
    err_var = ds.createVariable('mth_cor','f4',('stn_id',))
    err_var.long_name = "monthly correlation coefficient"
    err_var.standard_name = "monthly correlation coefficient"
    err_var.missing_value = netCDF4.default_fillvals['f4']
    
    ds.sync()
    
    return ds

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por.csv'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/'
    params[P_PATH_NORMALS_PO] = '/projects/daymet2/station_data/infill/normals_po.nc'
    params[P_PATH_NORMALS] = '/projects/daymet2/station_data/infill/normals_prcp.nc'
    params[P_NCDF_MODE] = 'r+' #w or r+
    params[P_NNGH] = 33 
    params[P_START_YMD] = None #19480101
    params[P_END_YMD] = None #20111231
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
