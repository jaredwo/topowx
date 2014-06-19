'''
A MPI driver for infilling/extending station observation records using the methods of obs_infill_daily. 

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.create_db_all_stations import dbDataset
from twx.db.station_data import StationDataDb,STN_ID,DATE,MEAN_TMIN,MEAN_TMAX
from twx.utils.status_check import status_check
from netCDF4 import Dataset
import netCDF4
from twx.infill.infill_daily import ImputeMatrixPCA,source_r
from twx.db.reanalysis import NNRNghData
from scipy import stats
from twx.interp.clibs import clib_wxTopo

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_NNR_TMIN = 'P_PATH_NNR_TMIN'
P_PATH_NNR_TMAX = 'P_PATH_NNR_TMAX'
P_PATH_NNR = 'P_PATH_NNR'
P_PATH_R_FUNCS = 'P_PATH_R_FUNCS'
P_PATH_CLIB = 'P_PATH_CLIB'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_NCDF_MODE = 'P_NCDF_MODE'
P_CHK_MAE_IMPROVE = 'P_CHK_MAE_IMPROVE'
P_STNIDS_TMIN = 'P_STNIDS_TMIN'
P_STNIDS_TMAX = 'P_STNIDS_TMAX'

P_MIN_NNGH_DAILY = 'P_MIN_NNGH_DAILY'
P_NNGH_NNR = 'P_NNGH_NNR'
P_NNR_VARYEXPLAIN = 'P_NNR_VARYEXPLAIN'
P_FRACOBS_INIT_PCS = 'P_FRACOBS_INIT_PCS'
P_PPCA_VARYEXPLAIN = 'P_PPCA_VARYEXPLAIN'
P_CHCK_IMP_PERF = 'P_CHCK_IMP_PERF'
P_NPCS_PPCA = 'P_NPCS_PPCA'

NCDF_CHK_COLS = 50
LAST_VAR_WRITTEN = 'nnghs'

#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri
r = robjects.r

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
    
    source_r(params[P_PATH_R_FUNCS])
    
    status = MPI.Status()
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    ndays = float(days.size)
    
    empty_fill = np.ones(ndays,dtype=np.float32)*netCDF4.default_fillvals['f4']
    empty_flags = np.ones(ndays,dtype=np.int8)*netCDF4.default_fillvals['i1']
    empty_ngh_radius = netCDF4.default_fillvals['f4']
    empty_npcs = netCDF4.default_fillvals['i2']
    empty_nngh = netCDF4.default_fillvals['i2']
    empty_bias = netCDF4.default_fillvals['f4']
    empty_mae = netCDF4.default_fillvals['f4']
    empty_varpct = netCDF4.default_fillvals['f4']
    
    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD],params[P_END_YMD]))
    aclib = clib_wxTopo()
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stnids_tmin,stnids_tmax = bcast_msg
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*11, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                infill_tmin = stn_id in stnids_tmin
                infill_tmax = stn_id in stnids_tmax
                
                if infill_tmin:
                    a_pca_matrix = ImputeMatrixPCA(stn_id, stn_da, 'tmin',ds_nnr,aclib)
                    fnl_tmin,fill_mask_tmin,imp_tmin,npcs_tmin,fnl_nnghs_tmin,max_dist_tmin,mae_tmin,bias_tmin,vary_pct_tmin = infill_tair(a_pca_matrix,params)
                    
                if infill_tmax:
                    a_pca_matrix = ImputeMatrixPCA(stn_id, stn_da, 'tmax',ds_nnr,aclib)
                    fnl_tmax,fill_mask_tmax,imp_tmax,npcs_tmax,fnl_nnghs_tmax,max_dist_tmax,mae_tmax,bias_tmax,vary_pct_tmax = infill_tair(a_pca_matrix,params)
                            
            except Exception as e:
                
                print "".join(["ERROR: Could not infill ",stn_id,"|",str(e)])
                if infill_tmin:
                    fnl_tmin,fill_mask_tmin,imp_tmin,npcs_tmin,fnl_nnghs_tmin,max_dist_tmin,mae_tmin,bias_tmin,vary_pct_tmin = empty_fill,empty_flags,empty_fill,empty_npcs,empty_nngh,empty_ngh_radius,empty_mae,empty_bias,empty_varpct
                if infill_tmax:
                    fnl_tmax,fill_mask_tmax,imp_tmax,npcs_tmax,fnl_nnghs_tmax,max_dist_tmax,mae_tmax,bias_tmax,vary_pct_tmax = empty_fill,empty_flags,empty_fill,empty_npcs,empty_nngh,empty_ngh_radius,empty_mae,empty_bias,empty_varpct
            
            if infill_tmin:
                MPI.COMM_WORLD.send((stn_id,'tmin',fnl_tmin,fill_mask_tmin,imp_tmin,npcs_tmin,fnl_nnghs_tmin,max_dist_tmin,mae_tmin,bias_tmin,vary_pct_tmin), dest=RANK_WRITE, tag=TAG_DOWORK)
            if infill_tmax:
                MPI.COMM_WORLD.send((stn_id,'tmax',fnl_tmax,fill_mask_tmax,imp_tmax,npcs_tmax,fnl_nnghs_tmax,max_dist_tmax,mae_tmax,bias_tmax,vary_pct_tmax), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stnids_tmin,stnids_tmax = bcast_msg
    print "Writer: Received broadcast msg"
    
    
    if params[P_NCDF_MODE] == 'r+':
        
        ds_tmin = Dataset("".join([params[P_PATH_OUT],'infill_tmin.nc']),'r+')
        ds_tmax = Dataset("".join([params[P_PATH_OUT],'infill_tmax.nc']),'r+')
        ttl_infills = stnids_tmin.size + stnids_tmax.size
        stnids_tmin = np.array(ds_tmin.variables['stn_id'][:], dtype="<S16")
        stnids_tmax = np.array(ds_tmax.variables['stn_id'][:], dtype="<S16")
        
        print "Writer: Infilling a total of %d station time series "%(ttl_infills,)
        
    else:
        
        ds_tmin,ds_tmax = create_ncdf(params[P_PATH_OUT], stnids_tmin, stnids_tmax, stn_da.stns, days)
        ttl_infills = stnids_tmin.size + stnids_tmax.size
    
    print "Writer: Output NCDF files ready"
    
    stat_chk = status_check(ttl_infills,10)
    
    while 1:

        stn_id,tair_var,tair,fill_mask,tair_imp,npcs,nnghs,max_dist,mae,bias,vary = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
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
                
            do_write = True
            
            if params[P_NCDF_MODE] == "r+" and params[P_CHK_MAE_IMPROVE]:
            
                cur_mae = ds.variables['mae'][stn_idx]
                
                if cur_mae != netCDF4.default_fillvals['f8'] and cur_mae < mae:
                    
                    print "|".join(["WRITER","MAE NOT IMPROVED",stn_id,tair_var,"%.4f"%(mae,),"%.4f"%(cur_mae,)])
                    
                    do_write = False
#                    ds.variables[LAST_VAR_WRITTEN][stn_idx] = nnghs
#                    ds.sync()
            
            if do_write:
                         
                print "|".join(["WRITER",stn_id,tair_var,"%.4f"%(mae,),"%.4f"%(bias,),"%.4f"%(vary,)])
                
                ds.variables['npcs'][stn_idx] = npcs
                ds.variables['mae'][stn_idx] = mae
                ds.variables['bias'][stn_idx] = bias
                ds.variables['r2'][stn_idx] = vary
                ds.variables['max_dist'][stn_idx] = max_dist
                ds.variables[tair_var][:,stn_idx] = tair
                ds.variables["".join([tair_var,"_imp"])][:,stn_idx] = tair_imp
                ds.variables["".join([tair_var,"_mean"])][stn_idx] = np.mean(tair,dtype=np.float64)
                ds.variables['flag_impute'][:,stn_idx] = fill_mask
                ds.variables[LAST_VAR_WRITTEN][stn_idx] = nnghs
                
                ds.sync()
                
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
        
    mask_tmin = np.isfinite(stn_da.stns[MEAN_TMIN])
    mask_tmax = np.isfinite(stn_da.stns[MEAN_TMAX])
    
#    mask_ca = np.char.startswith(stn_da.stns[STN_ID], "GHCN_CA")
#    mask_tmin = np.logical_and(mask_tmin,mask_ca)
#    mask_tmax = np.logical_and(mask_tmax,mask_ca)
        
    stnids_tmin = stn_da.stn_ids[mask_tmin]
    stnids_tmax = stn_da.stn_ids[mask_tmax]
    
    #Check if we're restarting a run
    if params[P_NCDF_MODE] == 'r+':
        
        #If rerunning remove stn ids that have already been completed
        try:
            
            if params[P_STNIDS_TMIN] == None:
            
                ds_tmin = Dataset("".join([params[P_PATH_OUT],'infill_tmin.nc']))
                mask_incplt = ds_tmin.variables[LAST_VAR_WRITTEN][:].mask
                stnids_tmin = stnids_tmin[mask_incplt]
                
            else:
                
                stnids_tmin = params[P_STNIDS_TMIN]
                
        except AttributeError:
            #no mask: infill complete
            stnids_tmin = np.array([],dtype="<S16")
        
        try:
            
            if params[P_STNIDS_TMAX] == None:
        
                ds_tmax = Dataset("".join([params[P_PATH_OUT],'infill_tmax.nc']))
                mask_incplt = ds_tmax.variables[LAST_VAR_WRITTEN][:].mask
                stnids_tmax = stnids_tmax[mask_incplt]
                
            else:
                
                stnids_tmax = params[P_STNIDS_TMAX]
        
        except AttributeError:
            #no mask: infill complete
            stnids_tmax = np.array([],dtype="<S16")
            
        
    stnids_all = np.unique(np.concatenate((stnids_tmin,stnids_tmax)))
    
    #Send stn ids to all processes
    MPI.COMM_WORLD.bcast((stnids_tmin,stnids_tmax), root=RANK_COORD)
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in stnids_all:
                
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

def create_tair_ncdf(fpath,stns,days,tair_var):
        
    ds = dbDataset(fpath,'w')
    
    ds.db_create_global_attributes("".join(['Imputed ',tair_var,' Weather Stations']))
    
    ds.db_create_time_dimvar(days)
    ds.db_create_stnid_dimvar(stns[STN_ID])
    
    ds.db_create_stn_vars(stns)

    ds.db_create_binflag_var('flag_impute', "imputed flag", "imputed_flag", chunk=(days[DATE].size,NCDF_CHK_COLS))
    
    ds.db_create_mae_var()
    ds.db_create_bias_var()

    npcs_var = ds.createVariable('npcs','i2',('stn_id',))
    npcs_var.long_name = "number of principal components"
    npcs_var.standard_name = "number_of_principal_components"
    npcs_var.missing_value = netCDF4.default_fillvals['i2']
    
    nnghs_var = ds.createVariable('nnghs','i2',('stn_id',))
    nnghs_var.long_name = "number of neighbor stations"
    nnghs_var.standard_name = "number_of_neighbors"
    nnghs_var.missing_value = netCDF4.default_fillvals['i2']
    
    max_dist_var = ds.createVariable('max_dist','f4',('stn_id',))
    max_dist_var.long_name = "max neighbor radius distance"
    max_dist_var.units = "km"
    max_dist_var.standard_name = "max neighbor radius distance"
    max_dist_var.missing_value = netCDF4.default_fillvals['f4']
    
    max_dist_var = ds.createVariable('r2','f4',('stn_id',))
    max_dist_var.long_name = "R-squared obs vs. imputed"
    max_dist_var.missing_value = netCDF4.default_fillvals['f4']
    
    ds.sync()
    return ds

def create_ncdf(path_out,stnids_tmin,stnids_tmax,all_stns,days):
    
    stns_tmin = all_stns[np.in1d(all_stns[STN_ID], stnids_tmin, assume_unique=True)]
    stns_tmax = all_stns[np.in1d(all_stns[STN_ID], stnids_tmax, assume_unique=True)]
    
    ds_tmin = create_tair_ncdf("".join([path_out,'infill_tmin.nc']), stns_tmin, days,'tmin')
    ds_tmax = create_tair_ncdf("".join([path_out,'infill_tmax.nc']), stns_tmax, days,'tmax')
    
    ds_tmin.db_create_tmin_var(chunk=(days[DATE].size,NCDF_CHK_COLS))
    ds_tmax.db_create_tmax_var(chunk=(days[DATE].size,NCDF_CHK_COLS))
    
    tmin_var = ds_tmin.createVariable('tmin_imp','f4',('time','stn_id'),chunksizes=(days[DATE].size,NCDF_CHK_COLS))
    tmin_var.long_name = "imputed minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "imputed_minimum_air_temperature"
    tmin_var.missing_value = netCDF4.default_fillvals['f4']
    
    tmin_var = ds_tmin.createVariable('tmin_mean','f8',('stn_id',))
    tmin_var.long_name = "mean minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "mean_minimum_air_temperature"
    tmin_var.missing_value = netCDF4.default_fillvals['f8']
    
    ds_tmin.sync()
    
    tmax_var = ds_tmax.createVariable('tmax_imp','f4',('time','stn_id'),chunksizes=(days[DATE].size,NCDF_CHK_COLS))
    tmax_var.long_name = "imputed maximum air temperature"
    tmax_var.units = "C"
    tmax_var.standard_name = "imputed_maximum_air_temperature"
    tmax_var.missing_value = netCDF4.default_fillvals['f4']
    
    tmax_var = ds_tmax.createVariable('tmax_mean','f8',('stn_id',))
    tmax_var.long_name = "mean maximum air temperature"
    tmax_var.units = "C"
    tmax_var.standard_name = "mean_maximum_air_temperature"
    tmax_var.missing_value = netCDF4.default_fillvals['f8']
    
    ds_tmax.sync()
    
    return ds_tmin,ds_tmax

def infill_tair(a_pca_matrix,params):
    
    
    fit_tair, obs_tair, npcs, fnl_nnghs, max_dist = a_pca_matrix.impute(min_daily_nnghs=params[P_MIN_NNGH_DAILY],
                                                                        nnghs_nnr=params[P_NNGH_NNR],
                                                                        max_nnr_var=params[P_NNR_VARYEXPLAIN],
                                                                        chk_perf=params[P_CHCK_IMP_PERF],
                                                                        npcs=params[P_NPCS_PPCA],
                                                                        frac_obs_initnpcs=params[P_FRACOBS_INIT_PCS],
                                                                        ppca_varyexplain=params[P_PPCA_VARYEXPLAIN])

    #Check for extreme values to see if the imputation converged to a reasonable solution
    if np.sum(fit_tair > 57.7) > 0 or np.sum(fit_tair < -89.4) > 0:
        print "".join(["WARNING|",a_pca_matrix.stn_id,
                       " appears to have bad imputation convergence for ",a_pca_matrix.tair_var])

    fnl_tair = np.copy(obs_tair)
    fill_mask = np.isnan(fnl_tair)
    fnl_tair[fill_mask] = fit_tair[fill_mask]
    
    fin_mask = np.logical_not(fill_mask)
    difs = fit_tair[fin_mask] - obs_tair[fin_mask]
    mae = np.mean(np.abs(difs))
    bias = np.mean(difs)
    
    r_value = stats.linregress(fit_tair[fin_mask], obs_tair[fin_mask])[2]
    vary_pct = r_value**2 #r-squared value; variance explained
    
    return fnl_tair,fill_mask,fit_tair,npcs,fnl_nnghs,max_dist,mae,bias,vary_pct

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
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/infill_nonhomog_20140329/' 
    params[P_PATH_NNR] = '/projects/daymet2/reanalysis_data/conus_subset/'
    params[P_PATH_R_FUNCS] = '/home/jared.oyler/repos/twx/twx/lib/rpy/pca_infill.R'
    params[P_NCDF_MODE] = 'r+' #w or r+
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    params[P_CHK_MAE_IMPROVE] = False
    params[P_MIN_NNGH_DAILY] = 3
    params[P_NNGH_NNR] = 4
    params[P_NNR_VARYEXPLAIN] = 0.99
    params[P_FRACOBS_INIT_PCS] = 0.5
    params[P_PPCA_VARYEXPLAIN] = 0.99
    params[P_CHCK_IMP_PERF] = True
    params[P_NPCS_PPCA] = 0
    params[P_STNIDS_TMIN] = None
    params[P_STNIDS_TMAX] = None
    
#    bad_infill = np.loadtxt('/projects/daymet2/station_data/infill/bad_infill.csv', dtype=[('stn_id_ca_tmin', "<S16"),('stn_id_ca_tmax', "<S16"),('stn_id_tmin', "<S16"),('stn_id_tmax', "<S16")],delimiter=',',skiprows=1)
#    stnids_tmin = bad_infill['stn_id_tmin']
#    stnids_tmax = bad_infill['stn_id_tmax']
#    params[P_STNIDS_TMIN] = stnids_tmin[stnids_tmin != '']
#    params[P_STNIDS_TMAX] = stnids_tmax[stnids_tmax != '']
    
    #These stations need to have min nnghs set to 10
#    params[P_STNIDS_TMAX] = np.array(['GHCN_USC00053553','GHCN_USC00165146','GHCN_USC00238754','GHCN_USC00300331',
#                                      'GHCN_USC00411033','GHCN_USW00023156'])
#    params[P_STNIDS_TMIN] = np.array(['GHCN_USC00010260','GHCN_USC00127362','GHCN_USC00466467','GHCN_USC00481175'])
    
    #Just rerun
#    params[P_STNIDS_TMAX] = np.array(['GHCN_CA003022772'])
#    params[P_STNIDS_TMIN] = np.array(['GHCN_CA006130409'])
    
    #These stations need to have min nnghs set to 15
    #params[P_STNIDS_TMAX] = np.array(['GHCN_USC00344861'])
    #params[P_STNIDS_TMIN] = np.array(['GHCN_USC00109493'])    
    
#    ds_tmin = Dataset('/projects/daymet2/station_data/infill/infill_tmin.nc')
#    ds_tmax = Dataset('/projects/daymet2/station_data/infill/infill_tmax.nc')
#    
#    stn_ids = ds_tmin.variables['stn_id'][:].astype("<S16")
#    npcs = ds_tmin.variables['npcs'][:]
#    params[P_STNIDS_TMIN] = stn_ids[npcs <= 5]
#    
#    stn_ids = ds_tmax.variables['stn_id'][:].astype("<S16")
#    npcs = ds_tmax.variables['npcs'][:]
#    params[P_STNIDS_TMAX] = stn_ids[npcs <= 5]
#    
#    ds_tmin.close()
#    ds_tmax.close()
#    
#    ds_tmin = None
#    ds_tmax = None
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
