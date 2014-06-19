'''
Created on Apr 1, 2013

@author: jared.oyler
'''
from twx.db.station_data import StationDataDb,STN_ID
from twx.db.reanalysis import NNRNghData
from twx.interp.clibs import clib_wxTopo
from twx.infill.infill_daily import ImputeMatrixPCA,source_r
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF4
from twx.db.create_db_all_stations import dbDataset
from twx.utils.status_check import status_check

NCDF_CHK_COLS = 50

def update_single_infill(stn_id,var,npcs,path_out):
    
    source_r('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R')
    
    stn_da = StationDataDb('/projects/daymet2/station_data/all/all_1948_2012.nc',
                               (19480101,20121231))
        
    ds_nnr = NNRNghData('/projects/daymet2/reanalysis_data/conus_subset/', (19480101,20121231))
    aclib = clib_wxTopo('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C')
    
    a_pca_matrix = ImputeMatrixPCA(stn_id, stn_da, var,ds_nnr,aclib)
    
    fit_tair, obs_tair, npcs, fnl_nnghs, max_dist = a_pca_matrix.impute(npcs=npcs)

    fnl_tair = np.copy(obs_tair)
    fill_mask = np.isnan(fnl_tair)
    fnl_tair[fill_mask] = fit_tair[fill_mask]
    
    fin_mask = np.logical_not(fill_mask)
    difs = fit_tair[fin_mask] - obs_tair[fin_mask]
    mae = np.mean(np.abs(difs))
    bias = np.mean(difs)
    
    r_value = stats.linregress(fit_tair[fin_mask], obs_tair[fin_mask])[2]
    vary_pct = r_value**2 #r-squared value; variance explained
    
    print stn_id,mae,bias,vary_pct
    #return fnl_tair,fill_mask,fit_tair,npcs,fnl_nnghs,max_dist,mae,bias,vary_pct

    ds = Dataset("".join([path_out,'infill_',var,'.nc']),'r+')
    stn_ids = ds.variables['stn_id'][:].astype("<S16")
    stn_idx = np.nonzero(stn_ids==stn_id)[0][0]
    
    ds.variables['npcs'][stn_idx] = npcs
    ds.variables['mae'][stn_idx] = mae
    ds.variables['bias'][stn_idx] = bias
    ds.variables['r2'][stn_idx] = vary_pct
    ds.variables['max_dist'][stn_idx] = max_dist
    ds.variables[var][:,stn_idx] = fnl_tair
    ds.variables["".join([var,"_imp"])][:,stn_idx] = fit_tair
    ds.variables["".join([var,"_mean"])][stn_idx] = np.mean(fnl_tair,dtype=np.float64)
    ds.variables['flag_impute'][:,stn_idx] = fill_mask
    ds.variables['nnghs'][stn_idx] = fnl_nnghs


def create_ncdf(path_out,stns_tmin,stns_tmax,days):
        
    ds_tmin = create_tair_ncdf("".join([path_out,'infill_tmin.nc']), stns_tmin, days,'tmin')
    ds_tmax = create_tair_ncdf("".join([path_out,'infill_tmax.nc']), stns_tmax, days,'tmax')
    
    ds_tmin.db_create_tmin_var(chunk=(days.size,NCDF_CHK_COLS))
    ds_tmax.db_create_tmax_var(chunk=(days.size,NCDF_CHK_COLS))
    
    tmin_var = ds_tmin.createVariable('tmin_imp','f4',('time','stn_id'),chunksizes=(days.size,NCDF_CHK_COLS))
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
    
    tmax_var = ds_tmax.createVariable('tmax_imp','f4',('time','stn_id'),chunksizes=(days.size,NCDF_CHK_COLS))
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

def create_tair_ncdf(fpath,stns,days,tair_var):
        
    ds = dbDataset(fpath,'w')
    
    ds.db_create_global_attributes("".join(['Imputed ',tair_var,' Weather Stations']))
    
    ds.db_create_time_dimvar(days)
    ds.db_create_stnid_dimvar(stns[STN_ID])
    
    ds.db_create_stn_vars(stns)

    ds.db_create_binflag_var('flag_impute', "imputed flag", "imputed_flag", chunk=(days.size,NCDF_CHK_COLS))
    
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

def create_new_ds():
    
    ds_tmin = Dataset('/projects/daymet2/station_data/infill/infill_tmin.nc')
    ds_tmax = Dataset('/projects/daymet2/station_data/infill/infill_tmax.nc')
    ds_catmin = Dataset('/projects/daymet2/station_data/infill_ca/infill_tmin.nc')
    ds_catmax = Dataset('/projects/daymet2/station_data/infill_ca/infill_tmax.nc')

    stnids_tmin = ds_tmin.variables['stn_id'][:].astype("<S16")
    stnids_tmax = ds_tmax.variables['stn_id'][:].astype("<S16")
    stnids_catmin = ds_catmin.variables['stn_id'][:].astype("<S16")
    stnids_catmax = ds_catmax.variables['stn_id'][:].astype("<S16")
    
    stnids_tmin = np.unique(np.concatenate((stnids_tmin,stnids_catmin)))
    stnids_tmax = np.unique(np.concatenate((stnids_tmax,stnids_catmax)))
    
    stn_da = StationDataDb('/projects/daymet2/station_data/all/all_1948_2012.nc',(19480101,20121231))

    stns_tmin = stn_da.stns[np.in1d(stn_da.stns[STN_ID], stnids_tmin, assume_unique=True)]
    stns_tmax = stn_da.stns[np.in1d(stn_da.stns[STN_ID], stnids_tmax, assume_unique=True)]
    
    create_ncdf('/projects/daymet2/station_data/infill/infill_fnl/', stns_tmin, stns_tmax, stn_da.days)
    

def insert_to_new_ds(ds_old,ds_new,var):
    
    stnids_old = ds_old.variables['stn_id'][:].astype("<S16")
    stnids_new = ds_new.variables['stn_id'][:].astype("<S16")
    
    schk = status_check(stnids_old.size,500)
    
    for x in np.arange(stnids_old.size):
        
        i = np.nonzero(stnids_new==stnids_old[x])[0][0]
        
        ds_new.variables['npcs'][i] = ds_old.variables['npcs'][x]
        ds_new.variables['mae'][i] = ds_old.variables['mae'][x]
        ds_new.variables['bias'][i] = ds_old.variables['bias'][x]
        ds_new.variables['r2'][i] = ds_old.variables['r2'][x]
        ds_new.variables['max_dist'][i] = ds_old.variables['max_dist'][x]
        ds_new.variables[var][:,i] = ds_old.variables[var][:,x]
        ds_new.variables["".join([var,"_imp"])][:,i] = ds_old.variables["".join([var,"_imp"])][:,x]
        ds_new.variables["".join([var,"_mean"])][i] = ds_old.variables["".join([var,"_mean"])][x]
        ds_new.variables['flag_impute'][:,i] = ds_old.variables['flag_impute'][:,x]
        ds_new.variables['nnghs'][i] = ds_old.variables['nnghs'][x]
        
        schk.increment()
    
    ds_new.sync()

if __name__ == '__main__':
    
#    insert_to_new_ds(Dataset('/projects/daymet2/station_data/infill/infill_tmin.nc'), 
#                     Dataset('/projects/daymet2/station_data/infill/infill_fnl/infill_tmin.nc','r+'),'tmin')
    
#    insert_to_new_ds(Dataset('/projects/daymet2/station_data/infill/infill_tmax.nc'), 
#                     Dataset('/projects/daymet2/station_data/infill/infill_fnl/infill_tmax.nc','r+'),'tmax')
#    
#    insert_to_new_ds(Dataset('/projects/daymet2/station_data/infill_ca/infill_tmin.nc'), 
#                     Dataset('/projects/daymet2/station_data/infill/infill_fnl/infill_tmin.nc','r+'),'tmin')
#    
    insert_to_new_ds(Dataset('/projects/daymet2/station_data/infill_ca/infill_tmax.nc'), 
                     Dataset('/projects/daymet2/station_data/infill/infill_fnl/infill_tmax.nc','r+'),'tmax')
    
    #create_new_ds()
    
    ####################################################################
#    update_single_infill('GHCN_CA007027259','tmin', 15, '/projects/daymet2/station_data/infill_ca/')
#    update_single_infill('GHCN_CA007063647','tmin', 12, '/projects/daymet2/station_data/infill_ca/')
#    
#    update_single_infill('GHCN_USC00086633','tmax', 10, '/projects/daymet2/station_data/infill/')
#    update_single_infill('GHCN_CA007025267','tmax', 9, '/projects/daymet2/station_data/infill_ca/')
#    update_single_infill('GHCN_CA00704CBGH','tmax', 5, '/projects/daymet2/station_data/infill_ca/')