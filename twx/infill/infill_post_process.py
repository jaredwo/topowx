'''
Methods to create final serially complete station time series datasets for input to the interpolation procedures

@author: jared.oyler
'''
from twx.db.all_create_db import dbDataset
from twx.db.station_data import build_stn_struct,STN_ID,DATE,LON,LAT,ELEV,station_data_infill,DTYPE_STN_DFLT,DTYPE_STN_BASIC,station_data_ncdb,\
    NEON, MASK, BAD, OPTIM_NNGH, OPTIM_NNGH_ANOM, MEAN_TMIN,MEAN_TMAX
from twx.db.reanalysis import NNRds
from netCDF4 import num2date,Dataset,date2num
import netCDF4
import twx.utils.util_dates as utld
import numpy as np
from twx.utils.status_check import status_check
from twx.utils.input_raster import input_raster, OutsideExtent
from twx.modis.montana_ndvi import modis_sin_rast
import twx.utils.util_geo as utlg
from twx.infill.infill_daily import ImputeMatrixPCA,source_r
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from twx.db.reanalysis import NNRNghData
from scipy import stats
from twx.interp.clibs import clib_wxTopo
import twx.db.ushcn as ushcn
from datetime import datetime
import twx.utils as utils

#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri
r = robjects.r

NCDF_CHK_COLS = 50
USE_ALL_IMP_THRESHOLD = np.round(365.25 * 5.0)

RM_STN_FLAG = "RM_STN_FLAG"
RM_STN_DUP = 1
RM_STN_BAD_DATA = 2
RM_STN_NO_TDI = 3

DTYPE_RM_STN = [(STN_ID,"<S16"),(RM_STN_FLAG,np.int)]

MTH_ABBRV = {1:'JAN',
            2:'FEB',
            3:'MAR',
            4:'APR',
            5:'MAY',
            6:'JUN',
            7:'JUL',
            8:'AUG',
            9:'SEP',
            10:'OCT',
            11:'NOV',
            12:'DEC'}

def create_serial_complete_nnrdif_ds(fpath,tair_var,ds_in):
    
    ds = dbDataset(fpath,'w')
    
    var_time = ds_in.variables['time']
    start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
    days = utld.get_days_metadata(start, end)
    
    stns = build_stn_struct(ds_in)
    
    ds.db_create_global_attributes("".join(['Serially complete ',tair_var,'  differences with NCEP/NCAR free air']))
    
    ds.db_create_time_dimvar(days)
    ds.db_create_stnid_dimvar(stns[STN_ID])
    
    ds.db_create_stn_vars(stns)

    ds.db_create_binflag_var('flag_impute', "imputed flag", "imputed_flag", chunk=(days[DATE].size,NCDF_CHK_COLS))

    if tair_var == 'tmin':

        ds.db_create_tmin_var(chunk=(days[DATE].size,NCDF_CHK_COLS))
    
        tmin_var = ds.createVariable('obs_mean','f8',('stn_id',))
        tmin_var.long_name = "mean minimum air temperature"
        tmin_var.units = "C"
        tmin_var.standard_name = "mean_minimum_air_temperature"
        tmin_var.missing_value = netCDF4.default_fillvals['f8']
        
        tmin_var = ds.createVariable('tmin_center','f4',('time','stn_id'),fill_value=netCDF4.default_fillvals['f4'],chunksizes=(days[DATE].size,NCDF_CHK_COLS))
        tmin_var.long_name = "mean centered minimum air temperature"
        tmin_var.units = "C"
        tmin_var.standard_name = "mean_centered_minimum_air_temperature"
        tmin_var.missing_value = netCDF4.default_fillvals['f4']
    
    elif tair_var == 'tmax':
        
        ds.db_create_tmax_var(chunk=(days[DATE].size,NCDF_CHK_COLS))
        
        tmax_var = ds.createVariable('obs_mean','f8',('stn_id',))
        tmax_var.long_name = "mean maximum air temperature"
        tmax_var.units = "C"
        tmax_var.standard_name = "mean_maximum_air_temperature"
        tmax_var.missing_value = netCDF4.default_fillvals['f8']
        
        tmax_var = ds.createVariable('tmax_center','f4',('time','stn_id'),fill_value=netCDF4.default_fillvals['f4'],chunksizes=(days[DATE].size,NCDF_CHK_COLS))
        tmax_var.long_name = "mean centered maximum air temperature"
        tmax_var.units = "C"
        tmax_var.standard_name = "mean_centered_maximum_air_temperature"
        tmax_var.missing_value = netCDF4.default_fillvals['f4']
        
    ds.sync()
    return ds

def has_ndata(rasts,ndata_vals,lon,lat):
    
    for x in np.arange(len(rasts)):
        
        val = rasts[x].getDataValue(lon,lat)
        
        for i in np.arange(len(ndata_vals[x])):
            
            if val == ndata_vals[x][i]:
                return True
            
    return False
    

def update_raster_stn_locs(rasts,ndata_vals,ds_path,varname,log_path,rm_stns=[]):
    
    fout = open(log_path,"w")
    fout.write(",".join(["STN_ID","LON", "LAT","LON_NEW","LAT_NEW","DIST\n"]))
        
    stn_da = station_data_infill(ds_path, varname,stn_dtype=DTYPE_STN_BASIC)
    stns = stn_da.stns
    stn_da.ds.close()
    stn_da = None
    
    ds = Dataset(ds_path,'r+')
    
    a_rast = rasts[0]
    
    cnt=0
    for stn_idx in np.arange(stns.size):
        
        stn = stns[stn_idx]
        try:
            if has_ndata(rasts, ndata_vals, stn[LON],stn[LAT]) and stn[STN_ID] not in rm_stns:
                
                try:
                    x,y = a_rast.getGridCellOffset(stn[LON],stn[LAT])
                    
                    r = 1
                    nn = []
                    
                    while len(nn) == 0:
                        
                        lcol = x-r
                        rcol = x+r
                        trow = y-r
                        brow = y+r
                        
                        #top ring
                        if trow > 0 and trow < a_rast.rows:
                            
                            for i in np.arange(lcol,rcol+1):
                                
                                if i > 0 and i < a_rast.cols:
                                    
                                    lat, lon = a_rast.getLatLon(i,trow,transform=False)
                                    if not has_ndata(rasts, ndata_vals, lon, lat):                                
                                        nn.append((trow,i))
                        
                        #left ring
                        if lcol > 0 and lcol < a_rast.cols:
                            
                            for i in np.arange(trow,brow+1):
                                
                                if i > 0 and i < a_rast.rows:
                                    
                                    lat, lon = a_rast.getLatLon(lcol,i,transform=False)
                                    if not has_ndata(rasts, ndata_vals, lon, lat):                                                                
                                        nn.append((i,lcol))
                                    
                        #bottom ring
                        if brow > 0 and brow < a_rast.rows:
                            
                            for i in np.arange(rcol,lcol,-1):
                                
                                if i > 0 and i < a_rast.cols:
                                    
                                    lat, lon = a_rast.getLatLon(i,brow,transform=False)
                                    if not has_ndata(rasts, ndata_vals, lon, lat):                                
                                        nn.append((brow,i))
                                        
                        #right ring
                        if rcol > 0 and rcol < a_rast.cols:
                            
                            for i in np.arange(brow,trow,-1):
                                
                                if i > 0 and i < a_rast.rows:
                                    
                                    lat, lon = a_rast.getLatLon(rcol,i,transform=False)
                                    if not has_ndata(rasts, ndata_vals, lon, lat):   
                                        nn.append((i,rcol))
                        
                        r+=1
                    
                    nn = np.array(nn)
                    
                    lats,lons = a_rast.getLatLon(nn[:,1], nn[:,0], transform=False)
                    d = utlg.grt_circle_dist(stn[LON],stn[LAT], lons, lats)
                    j = np.argsort(d)[0]
                    nlat,nlon = lats[j],lons[j]
                    
                    fout.write(",".join([stn[STN_ID],str(stn[LON]),str(stn[LAT]),
                                         str(nlon),str(nlat),str(d[j])+"\n"]))
                    
                    ds.variables[LON][stn_idx] = nlon
                    ds.variables[LAT][stn_idx] = nlat
                    ds.sync()
                                    
                    cnt+=1
                
                except utils.input_raster.OutsideExtent:
                    print stn
        except utils.input_raster.OutsideExtent:
            print stn
    ds.close()
    fout.close()
    

def update_waterlc_stn_locs(lc_fpath,ds_path,varname,log_path):
    
    #fout = open(log_path,"w")
    #fout.write(",".join(["STN_ID","LON", "LAT","LON_NEW","LAT_NEW","DIST","LC\n"]))
        
    stn_da = station_data_infill(ds_path, varname,stn_dtype=DTYPE_STN_BASIC)
    stns = stn_da.stns
    stn_da.ds.close()
    stn_da = None
    
    #ds = Dataset(ds_path,'r+')
    
    a_rast = input_raster(lc_fpath)
    a_lc = a_rast.readEntireRaster()
    
    cnt=0
    for stn_idx in np.arange(stns.size):
        
        stn = stns[stn_idx]
        
        if a_rast.getDataValue(stn[LON],stn[LAT]) == 0:
            
            x,y = a_rast.getGridCellOffset(stn[LON],stn[LAT])
            
            r = 1
            nn = []
            
            while len(nn) == 0:
                
                lcol = x-r
                rcol = x+r
                trow = y-r
                brow = y+r
                
                #top ring
                if trow > 0 and trow < a_rast.rows:
                    
                    for i in np.arange(lcol,rcol+1):
                        
                        if i > 0 and i < a_rast.cols:
                            
                            if a_lc[trow,i] != 0 and a_lc[trow,i] != 254 and a_lc[trow,i] != a_rast.ndata:
                                
                                nn.append((trow,i))
                
                #left ring
                if lcol > 0 and lcol < a_rast.cols:
                    
                    for i in np.arange(trow,brow+1):
                        
                        if i > 0 and i < a_rast.rows:

                            if a_lc[i,lcol] != 0 and a_lc[i,lcol] != 254 and a_lc[i,lcol] != a_rast.ndata:
                                
                                nn.append((i,lcol))
                            
                #bottom ring
                if brow > 0 and brow < a_rast.rows:
                    
                    for i in np.arange(rcol,lcol,-1):
                        
                        if i > 0 and i < a_rast.cols:
                            
                            if a_lc[brow,i] != 0 and a_lc[brow,i] != 254 and a_lc[brow,i] != a_rast.ndata:
                                
                                nn.append((brow,i))
                                
            
                #right ring
                if rcol > 0 and rcol < a_rast.cols:
                    
                    for i in np.arange(brow,trow,-1):
                        
                        if i > 0 and i < a_rast.rows:

                            if a_lc[i,rcol] != 0 and a_lc[i,rcol] != 254 and a_lc[i,rcol] != a_rast.ndata:
                                
                                nn.append((i,rcol))
                
                r+=1
            
            nn = np.array(nn)
            lats,lons = a_rast.getLatLon(nn[:,1], nn[:,0], transform=False)
            d = utlg.grt_circle_dist(stn[LON],stn[LAT], lons, lats)
            j = np.argsort(d)[0]
            nlat,nlon = lats[j],lons[j]
            nlc = a_rast.getDataValue(nlon, nlat)
            
            #fout.write(",".join([stn[STN_ID],str(stn[LON]),str(stn[LAT]),
                                 #str(nlon),str(nlat),str(d[j]),str(nlc)+"\n"]))
            print stn
            #ds.variables[LON][stn_idx] = nlon
            #ds.variables[LAT][stn_idx] = nlat
            #ds.sync()
                            
            cnt+=1
    
    #ds.close()
    #fout.close()


def reset_stn_locs(fpath_db,fpath_ds):
    
    stn_da = station_data_ncdb(fpath_db)

    ds = Dataset(fpath_ds,'r+')
    stn_ids = ds.variables['stn_id'][:].astype("<S16")
    
    for x in np.arange(stn_ids.size):
        
        i = np.nonzero(stn_da.stn_ids == stn_ids[x])[0][0]
        ds.variables[LON][x] = stn_da.stns[LON][i]
        ds.variables[LAT][x] = stn_da.stns[LAT][i]
    
    ds.sync()
    ds.close()
       

def create_serial_complete_ds(fpath,tair_var,ds_in):
    
    ds = dbDataset(fpath,'w')
    
    var_time = ds_in.variables['time']
    start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
    days = utld.get_days_metadata(start, end)
    
    stns = build_stn_struct(ds_in,DTYPE_STN_BASIC)
    
    ds.db_create_global_attributes("".join(['Serially complete ',tair_var,' time series.']))
    
    ds.db_create_time_dimvar(days)
    ds.db_create_stnid_dimvar(stns[STN_ID])
    
    ds.db_create_stn_vars(stns)

    ds.db_create_binflag_var('flag_impute', "imputed flag", "imputed_flag", chunk=(days[DATE].size,NCDF_CHK_COLS))

    if tair_var == 'tmin':

        ds.db_create_tmin_var(chunk=(days[DATE].size,NCDF_CHK_COLS))
    
        tmin_var = ds.createVariable('obs_mean','f8',('stn_id',))
        tmin_var.long_name = "mean minimum air temperature"
        tmin_var.units = "C"
        tmin_var.standard_name = "mean_minimum_air_temperature"
        tmin_var.missing_value = netCDF4.default_fillvals['f8']
        
#        tmin_var = ds.createVariable('tmin_center','f4',('time','stn_id'),fill_value=netCDF4.default_fillvals['f4'],chunksizes=(days[DATE].size,NCDF_CHK_COLS))
#        tmin_var.long_name = "mean centered minimum air temperature"
#        tmin_var.units = "C"
#        tmin_var.standard_name = "mean_centered_minimum_air_temperature"
#        tmin_var.missing_value = netCDF4.default_fillvals['f4']
    
    elif tair_var == 'tmax':
        
        ds.db_create_tmax_var(chunk=(days[DATE].size,NCDF_CHK_COLS))
        
        tmax_var = ds.createVariable('obs_mean','f8',('stn_id',))
        tmax_var.long_name = "mean maximum air temperature"
        tmax_var.units = "C"
        tmax_var.standard_name = "mean_maximum_air_temperature"
        tmax_var.missing_value = netCDF4.default_fillvals['f8']
        
#        tmax_var = ds.createVariable('tmax_center','f4',('time','stn_id'),fill_value=netCDF4.default_fillvals['f4'],chunksizes=(days[DATE].size,NCDF_CHK_COLS))
#        tmax_var.long_name = "mean centered maximum air temperature"
#        tmax_var.units = "C"
#        tmax_var.standard_name = "mean_centered_maximum_air_temperature"
#        tmax_var.missing_value = netCDF4.default_fillvals['f4']
        
    ds.sync()
    return ds


def build_serial_complete_nnrdif_ds(ds_in,ds_nnr,tair_var,out_path):

    ds_out = create_serial_complete_nnrdif_ds(out_path, tair_var, ds_in)
    
    n_stns = len(ds_in.dimensions['stn_id'])
    stns = build_stn_struct(ds_out)
    
    stat_chk = status_check(n_stns,100)
    for x in np.arange(n_stns):
        
        nnr_obs = ds_nnr.interp_variable(stns[LON][x],stns[LAT][x],stns[ELEV][x])
        stn_obs = ds_in.variables[tair_var][:,x].astype(np.float64)
        dif_obs = stn_obs - nnr_obs
        mdif = np.mean(dif_obs)
        cdif = dif_obs - mdif
        flag_impute = ds_in.variables['flag_impute'][:,x]
            
        ds_out.variables[tair_var][:,x] = dif_obs
        ds_out.variables["".join([tair_var,"_center"])][:,x] = cdif
        ds_out.variables['flag_impute'][:,x] = flag_impute
        ds_out.variables['obs_mean'][x] = mdif 
        ds_out.sync()
        
        stat_chk.increment()

def runs_of_ones_array(bits):
    #http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts

def update_serial_complete_ds(fpath_infill,fpath_serial,tair_var,stn_ids):

    ds_in = Dataset(fpath_infill)
    ds_serial = Dataset(fpath_serial,'r+')
    ds_stnids = ds_in.variables['stn_id'][:].astype("<S16")
    
    n_stns = stn_ids.size
    n_days = len(ds_in.dimensions['time'])
    all_imp_flags = np.ones(n_days,dtype=np.bool)
    all_imp_stns = np.zeros(n_stns,dtype=np.bool)
    
    stat_chk = status_check(n_stns,100)
    for stn_id in stn_ids:
        
        x = np.nonzero(ds_stnids==stn_id)[0][0]
        
        imp_mask = ds_in.variables['flag_impute'][:,x].astype(np.bool)
        
        imp_runs = runs_of_ones_array(imp_mask)
        
        if imp_runs.size > 0:
            max_imp = np.max(imp_runs)
        else:
            max_imp = 0
        
        if max_imp >= USE_ALL_IMP_THRESHOLD:
            
            #Use all imputed variables for this station to avoid discontinuties between imputed and observed portions of time series
            tair_stn = ds_in.variables["".join([tair_var,"_imp"])][:,x]
            flag_stn = all_imp_flags
            mtair_stn = np.mean(tair_stn,dtype=np.float64)
            
            all_imp_stns[np.nonzero(stn_ids==stn_id)[0][0]] = True
        
        else:
            
            tair_stn = ds_in.variables[tair_var][:,x]
            flag_stn = imp_mask
            mtair_stn = ds_in.variables["".join([tair_var,"_mean"])][x]
            
        ctair_stn = tair_stn.astype(np.float64) - mtair_stn
            
        ds_serial.variables[tair_var][:,x] = tair_stn
        ds_serial.variables["".join([tair_var,"_center"])][:,x] = ctair_stn
        ds_serial.variables['flag_impute'][:,x] = flag_stn
        ds_serial.variables['obs_mean'][x] = mtair_stn 
        ds_serial.sync()
        
        stat_chk.increment()
    
    print "% of stns with all imputed values: "+str((np.sum(all_imp_stns)/np.float(all_imp_stns.size))*100.)
    

def build_serial_complete_ds(ds_in,tair_var,out_path):

    ds_out = create_serial_complete_ds(out_path, tair_var, ds_in)
    
    n_stns = len(ds_in.dimensions['stn_id'])
    #n_days = len(ds_in.dimensions['time'])
    #all_imp_flags = np.ones(n_days,dtype=np.bool)
    #all_imp_stns = np.zeros(n_stns,dtype=np.bool)
    
    stat_chk = status_check(n_stns,100)
    for x in np.arange(n_stns):
        
        imp_mask = ds_in.variables['flag_impute'][:,x].astype(np.bool)
        
#        imp_runs = runs_of_ones_array(imp_mask)
#        
#        if imp_runs.size > 0:
#            max_imp = np.max(imp_runs)
#        else:
#            max_imp = 0
#        
#        if max_imp >= USE_ALL_IMP_THRESHOLD:
#            
#            #Use all imputed variables for this station to avoid discontinuties between imputed and observed portions of time series
#            tair_stn = ds_in.variables["".join([tair_var,"_imp"])][:,x]
#            flag_stn = all_imp_flags
#            mtair_stn = np.mean(tair_stn,dtype=np.float64)
#            
#            all_imp_stns[x] = True
#        
#        else:
            
        tair_stn = ds_in.variables[tair_var][:,x]
        flag_stn = imp_mask
        mtair_stn = ds_in.variables["".join([tair_var,"_mean"])][x]
            
        #ctair_stn = tair_stn.astype(np.float64) - mtair_stn
            
        ds_out.variables[tair_var][:,x] = tair_stn
        #ds_out.variables["".join([tair_var,"_center"])][:,x] = ctair_stn
        ds_out.variables['flag_impute'][:,x] = flag_stn
        ds_out.variables['obs_mean'][x] = mtair_stn 
        ds_out.sync()
        
        stat_chk.increment()
    
    #print "% of stns with all imputed values: "+str((np.sum(all_imp_stns)/np.float(all_imp_stns.size))*100.)

def add_stn_raster_values(ds_path,var_name,name,units,a_rast,handle_ndata=False, skip_stnids=None):
    
    ds = Dataset(ds_path,'r+')
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    stn_id = ds.variables['stn_id'][:].astype("<S16")
    a_data = None
    
    if var_name not in ds.variables.keys():
        newvar = ds.createVariable(var_name,'f8',('stn_id',),fill_value=a_rast.ndata)
    else:
        newvar = ds.variables[var_name]
    
    #newvar.setncattr('_FillValue',a_rast.ndata)
    #newvar._FillValue = a_rast.ndata
    newvar.long_name = name
    newvar.units = units
    newvar.standard_name = name
    newvar.missing_value = a_rast.ndata
    newvar[:] = a_rast.ndata
    ds.sync()
    
    for x in np.arange(lon.size):
        
        if skip_stnids is not None:
            
            if stn_id[x] in skip_stnids:
                newvar[x] = a_rast.ndata
                continue
        try:
            a = a_rast.getDataValue(lon[x], lat[x])
            newvar[x] = a
            if a == a_rast.ndata:
                raise Exception('No data raster value')
            
        except Exception:
            print "No data value for stn: "+stn_id[x]
            
            if handle_ndata:
                if a_data == None:
                    a_data = a_rast.readEntireRaster()
                
                x_grid,y_grid = a_rast.getGridCellOffset(lon[x],lat[x])
                newvar[x] = find_nn_data(a_data, a_rast, x_grid, y_grid)
                print "NN value for "+stn_id[x]+" is "+str(newvar[x])
    
    ds.sync()


def find_nn_data(a_data,a_rast,x,y):
                    
    r = 1
    nn = []
    nn_vals = []
    
    while len(nn) == 0:
        
        lcol = x-r
        rcol = x+r
        trow = y-r
        brow = y+r
        
        #top ring
        if trow > 0 and trow < a_rast.rows:
            
            for i in np.arange(lcol,rcol+1):
                
                if i > 0 and i < a_rast.cols:
                                           
                    if a_data[trow,i] != a_rast.ndata:                                
                        nn.append((trow,i))
                        nn_vals.append(a_data[trow,i])
        
        #left ring
        if lcol > 0 and lcol < a_rast.cols:
            
            for i in np.arange(trow,brow+1):
                
                if i > 0 and i < a_rast.rows:
                    
                    if a_data[i,lcol] != a_rast.ndata:                                                              
                        nn.append((i,lcol))
                        nn_vals.append(a_data[i,lcol])
                    
        #bottom ring
        if brow > 0 and brow < a_rast.rows:
            
            for i in np.arange(rcol,lcol,-1):
                
                if i > 0 and i < a_rast.cols:
                    
                    if a_data[brow,i] != a_rast.ndata:                                                                                          
                        nn.append((brow,i))
                        nn_vals.append(a_data[brow,i])
                        
        #right ring
        if rcol > 0 and rcol < a_rast.cols:
            
            for i in np.arange(brow,trow,-1):
                
                if i > 0 and i < a_rast.rows:
                    
                    if a_data[i,rcol] != a_rast.ndata:   
                        nn.append((i,rcol))
                        nn_vals.append(a_data[i,rcol])
        
        r+=1
    

    nn = np.array(nn)
    nn_vals = np.array(nn_vals)
    lats,lons = a_rast.getLatLon(nn[:,1], nn[:,0], transform=False)
    pt_lat,pt_lon = a_rast.getLatLon(x, y, transform=False)
    d = utlg.grt_circle_dist(pt_lon,pt_lat, lons, lats)
    j = np.argsort(d)[0]
    nval = nn_vals[j]
    return nval
    

def output_dup_stns(fpath_infilldb,tair_var,fpath_out,mode="w"):
    
    stn_da = station_data_infill(fpath_infilldb,tair_var,stn_dtype=DTYPE_STN_BASIC)
    dup_stnids = []
    fout = open(fpath_out,mode)
    
    if mode == "w":
        fout.write(",".join([STN_ID,RM_STN_FLAG+"\n"]))
    
    stat_chk = status_check(stn_da.stns.size,100)
    for stn in stn_da.stns:
        
        if stn[STN_ID] not in dup_stnids:
        
            ngh_stns = stn_da.stns[stn_da.stn_ids != stn[STN_ID]]
            dists = utlg.grt_circle_dist(stn[LON],stn[LAT],ngh_stns[LON],ngh_stns[LAT])
            
            dup_nghs = ngh_stns[dists==0]
            
            if dup_nghs.size > 0:
                #print dup_nghs
                dup_stnids.extend(dup_nghs[STN_ID])
    
                stn_ids_load = np.sort(np.concatenate([np.array([stn[STN_ID]]).ravel(),np.array([dup_nghs[STN_ID]]).ravel()]))
                print stn_ids_load
                stn_idxs = np.nonzero(np.in1d(stn_da.stn_ids, stn_ids_load, True))[0]
                imp_flgs = stn_da.ds.variables['flag_impute'][:,stn_idxs]
                imp_flg_sum = np.sum(imp_flgs, axis=0)
                
                stn_ids_rm = stn_ids_load[imp_flg_sum != np.min(imp_flg_sum)]
                
                for stn_id in stn_ids_rm:
                    fout.write(",".join([stn_id,str(RM_STN_DUP)+"\n"]))
        stat_chk.increment()
    fout.close()
    
def output_no_tdi_stns(fpath_db,fpath_out,mode="w"):
    
    ds = Dataset(fpath_db)
    tdi = ds.variables['tdi'][:]
    stn_ids = ds.variables['stn_id'][:].astype("<S16")
    
    if np.ma.isMA(tdi):
    
        ids_no_tdi = stn_ids[tdi.mask]
        
        fout = open(fpath_out,mode)
    
        if mode == "w":
            fout.write(",".join([STN_ID,RM_STN_FLAG+"\n"]))
        
        for stn_id in ids_no_tdi:
            fout.write(",".join([stn_id,str(RM_STN_NO_TDI)+"\n"]))

        fout.close()

def set_rm_bad_stations(bad_ids,fpath_ds):
    
    ds = Dataset(fpath_ds,'r+')

    stn_id = ds.variables['stn_id'][:].astype("<S16")

    if 'bad' not in ds.variables.keys():
    
        avar = ds.createVariable('bad','f8',('stn_id',),fill_value=0)
        avar.long_name = "bad station flag"
        avar.units = "NA"
        avar.missing_value = 0
        avar = ds.variables['bad']
        ds.sync()
    else:
        avar = ds.variables['bad']
        
    avar[:] = 0
    
    for aid in bad_ids:
        
        try:
            x = np.nonzero(stn_id==aid)[0][0]
        except IndexError:
            pass
        
        avar[x] = 1
    ds.sync()
    
def set_optim_nnghs(dspath,varname_tair,min_nghs,varname=OPTIM_NNGH,longname=None):
        
    stn_da = station_data_infill(dspath,varname_tair)
    
    rgns = stn_da.ds.variables['neon'][:]
    rgns = np.unique(rgns.data[np.logical_not(rgns.mask)])
    
    stn_rgn = stn_da.stns[NEON]
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD]))
    stn_da.ds.close()
    stn_da = None
    
    ds = Dataset(dspath,'r+')
    
    if varname not in ds.variables.keys():
            
        avar = ds.createVariable(varname,'f8',('stn_id',),fill_value=netCDF4.default_fillvals['f8'])
        if longname is not None:
            avar.long_name = longname
        avar.units = 'NA'
            
    else:
            
        avar = ds.variables[varname]
    
    avar[:] = avar._FillValue
    ds.sync()
    
    for x in np.arange(rgns.size):
        lcc_mask = np.logical_and(stn_rgn ==rgns[x],stn_mask)
        avar[lcc_mask] = min_nghs[x]
    ds.sync()
    
def updateImputeDaily():
    
    P_PATH_DB = 'P_PATH_DB'
    P_PATH_OUT = 'P_PATH_OUT'
    P_PATH_NNR = 'P_PATH_NNR'
    P_PATH_R_FUNCS = 'P_PATH_R_FUNCS'
    P_PATH_CLIB = 'P_PATH_CLIB'
    
    P_START_YMD = 'P_START_YMD'
    P_END_YMD = 'P_END_YMD'
    
    P_MIN_NNGH_DAILY = 'P_MIN_NNGH_DAILY'
    P_NNGH_NNR = 'P_NNGH_NNR'
    P_NNR_VARYEXPLAIN = 'P_NNR_VARYEXPLAIN'
    P_FRACOBS_INIT_PCS = 'P_FRACOBS_INIT_PCS'
    P_PPCA_VARYEXPLAIN = 'P_PPCA_VARYEXPLAIN'
    P_CHCK_IMP_PERF = 'P_CHCK_IMP_PERF'
    P_NPCS_PPCA = 'P_NPCS_PPCA'
    LAST_VAR_WRITTEN = 'nnghs'
    
    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/tairHomog_1948_2012.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/infill_20130725/' 
    params[P_PATH_NNR] = '/projects/daymet2/reanalysis_data/conus_subset/'
    params[P_PATH_R_FUNCS] = ['/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R',
                              '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R']
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    params[P_MIN_NNGH_DAILY] = 3
    params[P_NNGH_NNR] = 4
    params[P_NNR_VARYEXPLAIN] = 0.99
    params[P_FRACOBS_INIT_PCS] = 0.75
    params[P_PPCA_VARYEXPLAIN] = 0.99
    params[P_CHCK_IMP_PERF] = False
    params[P_NPCS_PPCA] = 12
    
    for path in params[P_PATH_R_FUNCS]:    
        source_r(path)
    
    stn_id = 'GHCN_CA004026480'
    tair_var = 'tmax'
    ppcaConThres=1e-8
    runUpdate = True
    tair_mask = None
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    stn_masks = {}
    stn_masks['tmin'] = np.isfinite(stn_da.stns[MEAN_TMIN])
    stn_masks['tmax'] = np.isfinite(stn_da.stns[MEAN_TMAX])
    
    print stn_da.stns[stn_da.stn_ids==stn_id][0]
    
    obs = stn_da.load_all_stn_obs_var(stn_id, tair_var)[0]
#    tair_mask = np.arange(obs.size) > 21767
#    obs[tair_mask] = np.nan
    
    plt.plot(np.arange(stn_da.days.size),obs)
    plt.show()
    
    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD],params[P_END_YMD]))
    aclib = clib_wxTopo(params[P_PATH_CLIB])
    
    
    a_pca_matrix = ImputeMatrixPCA(stn_id, stn_da, tair_var,ds_nnr,aclib,tair_mask=tair_mask)
    
    fit_tair, obs_tair, npcs, fnl_nnghs, max_dist = a_pca_matrix.impute(min_daily_nnghs=params[P_MIN_NNGH_DAILY],
                                                                        nnghs_nnr=params[P_NNGH_NNR],
                                                                        max_nnr_var=params[P_NNR_VARYEXPLAIN],
                                                                        chk_perf=params[P_CHCK_IMP_PERF],
                                                                        npcs=params[P_NPCS_PPCA],
                                                                        frac_obs_initnpcs=params[P_FRACOBS_INIT_PCS],
                                                                        ppca_varyexplain=params[P_PPCA_VARYEXPLAIN],
                                                                        ppcaConThres=ppcaConThres)
    
    print np.array(r.getVarChgPt(robjects.FloatVector(fit_tair)))
    fnl_tair = np.copy(obs_tair)
    fill_mask = np.isnan(fnl_tair)
    fnl_tair[fill_mask] = fit_tair[fill_mask]
    
    xval_mask = np.logical_not(fill_mask)
    xval_fit = fit_tair[xval_mask]
    xval_obs = obs[xval_mask]
    
    mae = np.mean(np.abs(xval_fit-xval_obs))
    bias = np.mean(xval_fit-xval_obs)
    r_value = stats.linregress(xval_fit, xval_obs)[2]
    vary_pct = r_value**2 #r-squared value; variance explained
        
    print mae,bias,vary_pct
    plt.plot(fit_tair)
    plt.show()
    
    #return fnl_tair,fill_mask,fit_tair,npcs,fnl_nnghs,max_dist,mae,bias,vary_pct
    if runUpdate:
        ds_out = Dataset("".join([params[P_PATH_OUT],'infill_',tair_var,'.nc']),'r+')
        stnids_out = np.array(ds_out.variables['stn_id'][:], dtype="<S16")
        stn_idx = np.nonzero(stnids_out == stn_id)[0][0]
        
        ds_out.variables['npcs'][stn_idx] = npcs
        ds_out.variables['mae'][stn_idx] = mae
        ds_out.variables['bias'][stn_idx] = bias
        ds_out.variables['r2'][stn_idx] = vary_pct
        ds_out.variables['max_dist'][stn_idx] = max_dist
        ds_out.variables[tair_var][:,stn_idx] = fnl_tair
        ds_out.variables["".join([tair_var,"_imp"])][:,stn_idx] = fit_tair
        ds_out.variables["".join([tair_var,"_mean"])][stn_idx] = np.mean(fnl_tair,dtype=np.float64)
        ds_out.variables['flag_impute'][:,stn_idx] = fill_mask
        ds_out.variables[LAST_VAR_WRITTEN][stn_idx] = fnl_nnghs
        print "UPDATED!"

def analyzeBadImpsVarChgPt():
    
    source_r('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R')
    
    f = open('/projects/daymet2/station_data/infill/infill_20130725/impute.log')
    lines = np.array(f.readlines())
    lmask = np.logical_and(np.logical_and(np.char.find(lines, 'ERROR|') != -1,
                                          np.char.find(lines, 'variance') != -1),
                           np.char.find(lines, 'tmax') != -1)
    print np.sum(lmask)
    
    stnids = np.unique(np.array([aline.split('|')[1].split()[0] for aline in lines[lmask]]))
    
    for stnid in stnids:
        print stnid
    
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc','tmax_imp',stn_dtype=DTYPE_STN_BASIC)
    stnda2 = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc','tmax',stn_dtype=DTYPE_STN_BASIC)
    
    
    stnsBadImp = stnda.stns[np.in1d(stnda.stn_ids, stnids, True)] 
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(stnsBadImp[LAT])-1,urcrnrlat=np.max(stnsBadImp[LAT])+1,\
                llcrnrlon=np.min(stnsBadImp[LON])-1,urcrnrlon=np.max(stnsBadImp[LON])+1,resolution='c')
    
    m.scatter(stnsBadImp[LON],stnsBadImp[LAT])
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    
    plt.show()
    
    
    #x = np.nonzero(stnids=='GHCN_USC00047888')[0][0]
    
    for stnid in stnids:#stnids[x:]:
        
        plt.clf()
        plt.subplot(211)
        print "#############################################"
        tair = stnda.load_obs(stnid)
        try:
            chgPt = np.array(r.getVarChgPt(robjects.FloatVector(tair)))[0]
        except IndexError:
            chgPt = 0
            
        
        plt.plot(tair)
        ylim = plt.ylim()
        plt.vlines(chgPt,ylim[0],ylim[1],color='red')
        plt.title(stnid)
        xlim = plt.xlim()
        
        idx = np.nonzero(stnda.stn_ids==stnid)[0][0]
        
        
        print stnda.stns[idx]
        
        flg = stnda.ds.variables['flag_impute'][:,idx].astype(np.bool)
        tair = stnda2.load_obs(stnid)
        tair[flg] = np.nan
        plt.subplot(212)
        plt.plot(tair)
        plt.xlim(xlim)
        
        print "".join(["% Imputed: ",str(np.sum(flg)/np.float(flg.size)*100.)])
        
        plt.show()
        
def analyzeBadImpsHighMAE():
    
    source_r('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R')
    
    f = open('/projects/daymet2/station_data/infill/infill_20130725/impute.log')
    lines = np.array(f.readlines())
    lmask = np.logical_and(np.logical_and(np.char.find(lines, 'ERROR|') != -1,
                                          np.char.find(lines, 'low impute performance') != -1),
                                          np.char.find(lines, 'tmax') != -1)
    
    lmask = np.logical_and(lmask,np.char.find(lines, 'variance') == -1)
    
    chkLineMask = np.zeros(np.sum(lmask),dtype=np.bool)
    x = 0
    
    for aline in lines[lmask]:
        #print aline.split()[-1].strip(")(").split(",")
        mae,r2 = [float(i) for i in aline.split()[-1].strip(")(").split(",")]
        if mae >= 3.0 or r2 <= .80:
            chkLineMask[x] = True
        x+=1
    
    fnlLines = lines[lmask][chkLineMask]
    
    stnids = np.unique(np.array([aline.split('|')[1].split()[0] for aline in fnlLines]))
    
    for stnid in stnids:
        print stnid
    print stnids.size
    
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc','tmax_imp',stn_dtype=DTYPE_STN_BASIC)
    stnda2 = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc','tmax',stn_dtype=DTYPE_STN_BASIC)
    
    
    stnsBadImp = stnda.stns[np.in1d(stnda.stn_ids, stnids, True)] 
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(stnsBadImp[LAT])-1,urcrnrlat=np.max(stnsBadImp[LAT])+1,\
                llcrnrlon=np.min(stnsBadImp[LON])-1,urcrnrlon=np.max(stnsBadImp[LON])+1,resolution='c')
    
    m.scatter(stnsBadImp[LON],stnsBadImp[LAT])
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    
    plt.show()
    
    
    #x = np.nonzero(stnids=='GHCN_USC00047888')[0][0]
    
    for stnid in stnids:#stnids[x:]:
        
        plt.clf()
        plt.subplot(211)
        
        tair = stnda.load_obs(stnid)
        try:
            chgPt = np.array(r.getVarChgPt(robjects.FloatVector(tair)))[0]
        except IndexError:
            chgPt = 0
            
        
        plt.plot(tair)
        ylim = plt.ylim()
        plt.vlines(chgPt,ylim[0],ylim[1],color='red')
        plt.title(stnid)
        xlim = plt.xlim()
        
        idx = np.nonzero(stnda.stn_ids==stnid)[0][0]
        
        print "#############################################"
        print stnda.stns[idx]
        
        flg = stnda.ds.variables['flag_impute'][:,idx].astype(np.bool)
        tair = stnda2.load_obs(stnid)
        tair[flg] = np.nan
        plt.subplot(212)
        plt.plot(tair)
        plt.xlim(xlim)
        
        print "".join(["% Imputed: ",str(np.sum(flg)/np.float(flg.size)*100.)])
        
        plt.show()

def add_monthly_means(dsPath,varName):
    
    stnda = station_data_infill(dsPath, varName)
    tagg = ushcn.TairAggregate(stnda.days)
    minDate = stnda.days[DATE][0]
    stns = stnda.stns
    stnda.ds.close()
    stnda = None
    ds = Dataset(dsPath,'r+')
    
    if 'time_mth' not in ds.variables.keys():
        
        ds.createDimension('time_mth',tagg.yrMths.size)
        times = ds.createVariable('time_mth','f8',('time_mth',),fill_value=False)
        times.units = "".join(["days since ",str(minDate.year),"-",str(minDate.month),"-",str(minDate.day)," 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times[:] = date2num(tagg.yrMths[DATE],times.units)
        
        varMthly = ds.createVariable("_".join([varName,"mth"]),'f4',('time_mth','stn_id'),fill_value=netCDF4.default_fillvals['f4'])
    
    else:
        
        varMthly = ds.variables["_".join([varName,"mth"])]
        
    varDly = ds.variables[varName]
    chkSize = 50
    
    stchk = status_check(np.int(np.round(stns.size/np.float(chkSize))), 10)
    for i in np.arange(0,stns.size,chkSize):
        
        if i + chkSize < stns.size:
            nStns = chkSize
        else:
            nStns = stns.size - i
        
        dlyVals = varDly[:,i:i+nStns]
        mthVals = tagg.dailyToMthly(dlyVals,maxMiss=-1)[0]
        varMthly[:,i:i+nStns] = mthVals
        ds.sync()
        stchk.increment()

def add_monthly_norms(dsPath,varName,startYr,endYr):
    
    stnda = station_data_infill(dsPath, varName)
    tagg = ushcn.TairAggregate(stnda.days)

    stns = stnda.stns
    stnda.ds.close()
    stnda = None
    ds = Dataset(dsPath,'r+')
    
    varNorms = {}
    for mth in np.arange(1,13):
        
        varNameNorm = 'norm%02d'%mth
        if varNameNorm not in ds.variables.keys():
            varNorm = ds.createVariable(varNameNorm,'f8',('stn_id',),fill_value=netCDF4.default_fillvals['f8'])
        else:
            varNorm = ds.variables[varNameNorm]
        varNorm.long_name = "%d - %d Monthly Normal"%(startYr,endYr)
        varNorms[mth] = varNorm
            
    varDly = ds.variables[varName]
    chkSize = 50
    
    stchk = status_check(np.int(np.round(stns.size/np.float(chkSize))), 10)
    for i in np.arange(0,stns.size,chkSize):
        
        if i + chkSize < stns.size:
            nStns = chkSize
        else:
            nStns = stns.size - i
        
        dlyVals = varDly[:,i:i+nStns]
        
        normVals = tagg.dailyToMthlyNorms(dlyVals, startYr, endYr)
        
        for mth in np.arange(1,13):
            varNorms[mth][i:i+nStns] = normVals[mth-1,:]
        
        ds.sync()
        stchk.increment()

def add_ann_means(dsPath,varName):
    
    stnda = station_data_infill(dsPath, varName)
    tagg = ushcn.TairAggregate(stnda.days)
    minDate = stnda.days[DATE][0]
    stns = stnda.stns
    stnda.ds.close()
    stnda = None
    ds = Dataset(dsPath,'r+')
    
    if 'time_ann' not in ds.variables.keys():
        
        ds.createDimension('time_ann',tagg.uYrs.size)
        times = ds.createVariable('time_ann','f8',('time_ann',),fill_value=False)
        times.units = "".join(["days since ",str(minDate.year),"-",str(minDate.month),"-",str(minDate.day)," 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times[:] = date2num([datetime(yr,1,1) for yr in tagg.uYrs],times.units)
        
        varAnn = ds.createVariable("_".join([varName,"ann"]),'f4',('time_ann','stn_id'),fill_value=netCDF4.default_fillvals['f4'])
    
    else:
        
        varAnn = ds.variables["_".join([varName,"ann"])]
        
    varMthly = ds.variables["_".join([varName,"mth"])]
    chkSize = 50
    
    stchk = status_check(np.int(np.round(stns.size/np.float(chkSize))), 10)
    for i in np.arange(0,stns.size,chkSize):
        
        if i + chkSize < stns.size:
            nStns = chkSize
        else:
            nStns = stns.size - i
        
        mthVals = varMthly[:,i:i+nStns]
        annVals = tagg.mthlyToAnn(mthVals)
        varAnn[:,i:i+nStns] = annVals
        ds.sync()
        stchk.increment()

if __name__ == '__main__':

#Pre-STEP: Analyze flagged imputations and fix if necessary
    #analyzeBadImpsVarChgPt()
    #analyzeBadImpsHighMAE()
    #updateImputeDaily()

    
#################################################################################################################    
# 1.) STEP 1: Create the serially-complete datasets    
#################################################################################################################
#CREATE A FINAL SERIALLY COMPLETE TMIN DATASET FROM IMPUTED/INFILLED STATIONS
#################################################################################################################   
#    ds_tmin = Dataset('/projects/daymet2/station_data/infill/infill_20130725/infill_tmin.nc')
#    build_serial_complete_ds(ds_tmin, 'tmin', '/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc')
#################################################################################################################   
#################################################################################################################
#CREATE A FINAL SERIALLY COMPLETE TMAX DATASET FROM IMPUTED/INFILLED STATIONS
#################################################################################################################  
    ds_tmax = Dataset('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc')
    build_serial_complete_ds(ds_tmax, 'tmax', '/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc')
#################################################################################################################
#Perform any one off updates
#    update_serial_complete_ds('/projects/daymet2/station_data/infill/infill_fnl/infill_tmin.nc',
#                              '/projects/daymet2/station_data/infill/infill_fnl/serial_tmin.nc','tmin',
#                              np.array(['GHCN_USC00010260','GHCN_USC00127362','GHCN_USC00466467','GHCN_USC00481175',
#                                        'GHCN_CA006130409','GHCN_USC00109493'])) 
#    update_serial_complete_ds('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc',
#                              '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax',
#                              np.array(['GHCN_CA004026480']))
#################################################################################################################


#################################################################################################################
# 2.) STEP 2: Mark any stations that should not be used
#################################################################################################################
#    ds_path_tmin = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc'
#    ds_path_tmax = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc' 
#    rm_stns = np.loadtxt('/projects/daymet2/station_data/infill/infill_20130725/BadStns.csv',delimiter=",",dtype=np.str,skiprows=1,usecols=(0,))
#    rm_ids = np.unique(rm_stns)
#    print rm_stns.size,rm_ids.size
#    set_rm_bad_stations(rm_ids, ds_path_tmin)
#    set_rm_bad_stations(rm_ids, ds_path_tmax)

#Fallback if issues: Reset all stations to okay
#    set_rm_bad_stations([],'/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc')
#    set_rm_bad_stations([],'/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
#    set_rm_bad_stations([],'/projects/daymet2/station_data/infill/infill_20130725/infill_tmin.nc')
#    set_rm_bad_stations([],'/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc')  

################################################################################################################# 
    
#################################################################################################################
# 3.) STEP 3: Check for stns that have no data values for some of the predictor grids and move to closest valid pixel
#################################################################################################################
#    rasts = []
#    rasts.append(input_raster('/projects/daymet2/climate_office/modis/MCD12Q1.051/MOSAIC_MCD12Q1.Land_Cover_Type_2.null.tif'))
#    rasts.append(input_raster('/projects/daymet2/dem/interp_grids/tifs/tdi.tif'))
#    rasts.append(input_raster('/projects/daymet2/dem/interp_grids/tifs/elev.tif'))
##    rasts.append(input_raster('/projects/daymet2/dem/interp_grids/tifs/lst_tmax.tif'))
##    rasts.append(input_raster('/projects/daymet2/dem/interp_grids/tifs/lst_tmin.tif'))
#    rasts.append(input_raster('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Day_1km.01.C.tif'))
#    rasts.append(input_raster('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Night_1km.01.C.tif'))
#  
#    ndata_vals = []
#    ndata_vals.append((256,0)) #lc
#    ndata_vals.append((-999,)) #tdi
#    ndata_vals.append((-999,)) #elev
##    ndata_vals.append((np.float32(-3.40282346639e+038),)) #lst_tmin
##    ndata_vals.append((np.float32(-3.40282346639e+038),)) #lst_tmax
#    ndata_vals.append((65535,)) #lst_tmax
#    ndata_vals.append((65535,)) #lst_tmin
#
#    ds_path_infill = '/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc'
#    ds_path_serial = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc'
#
#    ds = Dataset(ds_path_serial)
#    stn_ids = ds.variables['stn_id'][:].astype("<S16")
#    rm_stns = stn_ids[ds.variables['bad'][:]==1]
#    ds.close()
#    ds = None
#    
#    update_raster_stn_locs(rasts, ndata_vals,ds_path_serial, 'tmax',
#                           '/projects/daymet2/station_data/infill/infill_20130725/moved_tmax_stns.csv',rm_stns)
#    
#    update_raster_stn_locs(rasts, ndata_vals,ds_path_infill, 'tmax',
#                           '/projects/daymet2/station_data/infill/infill_20130725/moved_tmax_stns.csv',rm_stns)

#Fallback if issues: Reset all stations to original locations  
#    reset_stn_locs('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc', 
#                   '/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc')
#    reset_stn_locs('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc', 
#                   '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
#    reset_stn_locs('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc', 
#                   '/projects/daymet2/station_data/infill/infill_20130725/infill_tmin.nc')
#    reset_stn_locs('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc', 
#                   '/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc')
################################################################################################################# 

#################################################################################################################
# 4.) STEP 4: Check for dups and then set the Bad flag on them
#################################################################################################################
#    output_dup_stns('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc', 'tmax',
#                    '/projects/daymet2/station_data/infill/infill_20130725/dupstns_tmax.csv')
#    
#    output_dup_stns('/projects/daymet2/station_data/infill/infill_20130725/infill_tmin.nc', 'tmin',
#                    '/projects/daymet2/station_data/infill/infill_20130725/dupstns_tmin.csv')
    ###################################################################
#    rmIds = np.loadtxt('/projects/daymet2/station_data/infill/infill_20130725/BadStns.csv',delimiter=",",dtype=np.str,skiprows=1,usecols=(0,))
#    dupIdsTmin = np.loadtxt('/projects/daymet2/station_data/infill/infill_20130725/dupstns_tmin.csv',delimiter=",",dtype=np.str,skiprows=1,usecols=(0,))
#    dupIdsTmax = np.loadtxt('/projects/daymet2/station_data/infill/infill_20130725/dupstns_tmax.csv',delimiter=",",dtype=np.str,skiprows=1,usecols=(0,))
#    allIds = np.unique(np.concatenate([rmIds,dupIdsTmin,dupIdsTmax]))
#    
#    set_rm_bad_stations(allIds, '/projects/daymet2/station_data/infill/infill_20130725/infill_tmin.nc')
#    set_rm_bad_stations(allIds, '/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc')
#    
#    set_rm_bad_stations(allIds, '/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc')
#    set_rm_bad_stations(allIds, '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')

#################################################################################################################
# 5.) STEP 5: Add auxilary predictor values from rasters
#################################################################################################################
#    ds_path_tmin = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc'
#    ds_path_tmax = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc'
#
##TMIN LST
##    name = 'land surface temperature'
##    units = "C"
##    for mth in np.arange(1,13):
##        var_name = 'lst%02d'%mth
##        a_rast = input_raster('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Night_1km.%02d.C.tif'%mth)
##        add_stn_raster_values(ds_path_tmin, var_name, name, units, a_rast)
##
###TMAX LST
##    name = 'land surface temperature'
##    units = "C"
##    for mth in np.arange(1,13):
##        var_name = 'lst%02d'%mth
##        a_rast = input_raster('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Day_1km.%02d.C.tif'%mth)
##        add_stn_raster_values(ds_path_tmax, var_name, name, units, a_rast)
#    
##TMIN TDI
#    var_name = 'tdi'
#    name = 'topographic dissection index'
#    units = "[3,6,9,12,15] km radius"
#    a_rast = input_raster('/projects/daymet2/dem/interp_grids/tifs/tdi.tif')
#    add_stn_raster_values(ds_path_tmin, var_name, name, units, a_rast)
#    
##TMAX TDI
#    var_name = 'tdi'
#    name = 'topographic dissection index'
#    units = "[3,6,9,12,15] km radius"
#    a_rast = input_raster('/projects/daymet2/dem/interp_grids/tifs/tdi.tif')
#    add_stn_raster_values(ds_path_tmax, var_name, name, units, a_rast)
#
##TMAX Interp Mask
#    var_name = 'mask'
#    name = 'Interpolation Mask'
#    units = "0 or 1"
#    a_rast = input_raster('/projects/daymet2/dem/interp_grids/conus/tifs/mask_all_nd.tif')
#    a_rast.ndata = 0
#    add_stn_raster_values(ds_path_tmax, var_name, name, units, a_rast)
#
##TMIN Interp Mask
#    var_name = 'mask'
#    name = 'Interpolation Mask'
#    units = "0 or 1"
#    a_rast = input_raster('/projects/daymet2/dem/interp_grids/conus/tifs/mask_all_nd.tif')
#    a_rast.ndata = 0
#    add_stn_raster_values(ds_path_tmin, var_name, name, units, a_rast)
##
###U.S. Climate Divisions
#    var_name = "neon"
#    name = "U.S. Climate Division"
#    units = "NA"
#    a_rast = input_raster('/projects/daymet2/dem/interp_grids/tifs/climdivLccMerge.tif')
#    
#    skipids = None
#    add_stn_raster_values(ds_path_tmin, var_name, name, units, a_rast, handle_ndata=False, skip_stnids=skipids)
#    add_stn_raster_values(ds_path_tmax, var_name, name, units, a_rast, handle_ndata=False, skip_stnids=skipids)
    
#################################################################################################################

#################################################################################################################
# 6.) STEP 6: Optim # of neighbors for Climate Divisions
#################################################################################################################

#    ds_path_tmin = '/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmin.nc'
#    ds_path_tmax = '/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc'
##
##    optim_nghs_tmin = np.array([92, 111, 29, 32, 147, 20, 111, 147, 47, 111, 134, 147, 147, 147, 52, 147, 147, 147, 35, 26, 101, 76, 57, 111, 47, 84, 101, 147, 101, 39, 57, 52, 43, 147, 39, 147, 35, 52, 147, 39, 20, 20, 147, 35, 122, 101, 147, 22, 76, 24, 147, 147, 147, 147, 147, 22, 147, 76, 147, 47, 32, 84, 35, 57, 24, 92, 52, 84, 20, 147, 43, 147, 63, 47, 43, 147, 47, 147, 147, 29, 134, 39, 147, 20, 43, 147, 92, 24, 32, 122, 101, 20, 29, 147, 122, 147, 84, 29, 47, 147, 29, 147, 147, 122, 122, 69, 84, 147, 111, 20, 101, 147, 43, 147, 26, 147, 122, 35, 76, 39, 63, 26, 84, 147, 52, 57, 147, 26, 76, 147, 69, 76, 32, 147, 24, 24, 22, 147, 92, 147, 47, 76, 92, 147, 101, 147, 84, 63, 147, 20, 69, 134, 57, 147, 20, 147, 147, 147, 147, 84, 39, 147, 147, 147, 63, 69, 52, 76, 147, 52, 147, 147, 52, 147, 43, 147, 147, 24, 84, 147, 111, 122, 43, 92, 147, 76, 20, 57, 20, 147, 22, 147, 39, 122, 84, 101, 63, 52, 20, 47, 147, 147, 47, 134, 147, 147, 39, 47, 101, 147, 122, 101, 147, 76, 35, 147, 92, 29, 147, 147, 52, 147, 20, 147, 76, 101, 24, 147, 35, 147, 26, 20, 20, 84, 101, 134, 147, 111, 147, 35, 147, 63, 35, 47, 134, 147, 24, 47, 147, 147, 69, 101, 147, 147, 147, 39, 76, 147, 147, 52, 147, 147, 76, 147, 147, 26, 20, 101, 32, 43, 147, 20, 147, 76, 147, 22, 134, 47, 147, 147, 134, 134, 147, 69, 69, 134, 63, 101, 32, 147, 147, 69, 101, 57, 20, 147, 69, 69, 26, 69, 35, 147, 52, 29, 147, 111, 57, 22, 147, 22, 20, 111, 147, 57, 92, 134, 32, 20, 63, 147, 147, 134, 43, 147, 147, 147, 122, 57, 63, 92, 147, 147, 22, 122, 147, 20, 43, 43, 84, 39, 147, 147, 147, 22, 29, 111])
##    set_optim_nnghs(ds_path_tmin,'tmin', optim_nghs_tmin,OPTIM_NNGH,"optimal number of neighbors for interpolation")
###    
##    optim_nghs_tmax = np.array([47, 63, 111, 92, 26, 147, 134, 47, 57, 147, 57, 52, 76, 147, 32, 122, 101, 20, 39, 101, 147, 43, 147, 134, 122, 47, 26, 22, 84, 35, 92, 39, 57, 43, 147, 69, 47, 92, 69, 20, 69, 20, 147, 24, 43, 52, 63, 32, 43, 84, 147, 57, 84, 147, 122, 111, 147, 134, 29, 39, 92, 35, 147, 76, 63, 47, 84, 147, 111, 52, 84, 39, 43, 122, 147, 84, 92, 147, 147, 147, 63, 69, 76, 147, 147, 147, 147, 147, 69, 43, 92, 29, 101, 147, 147, 20, 22, 147, 147, 92, 47, 92, 101, 20, 57, 147, 147, 134, 122, 43, 101, 147, 26, 76, 47, 84, 22, 39, 76, 47, 92, 147, 92, 147, 57, 20, 26, 47, 32, 76, 63, 32, 92, 84, 122, 26, 26, 147, 101, 84, 47, 57, 147, 147, 92, 57, 22, 147, 63, 147, 35, 147, 147, 22, 20, 39, 147, 147, 111, 29, 57, 134, 147, 43, 39, 29, 111, 52, 20, 134, 63, 57, 147, 101, 147, 147, 39, 26, 76, 147, 147, 147, 111, 147, 47, 147, 147, 147, 147, 134, 35, 134, 147, 92, 92, 147, 39, 147, 92, 122, 26, 147, 92, 52, 32, 92, 147, 147, 101, 101, 147, 92, 63, 147, 92, 20, 134, 147, 147, 122, 52, 29, 101, 111, 147, 122, 39, 29, 147, 147, 69, 122, 76, 122, 147, 147, 29, 134, 29, 69, 57, 43, 122, 35, 147, 39, 39, 63, 101, 47, 147, 52, 92, 147, 147, 92, 134, 147, 92, 22, 122, 147, 101, 147, 63, 147, 147, 92, 20, 39, 147, 147, 20, 20, 111, 20, 57, 147, 63, 134, 92, 147, 122, 147, 134, 147, 147, 147, 92, 39, 43, 39, 47, 39, 111, 147, 39, 147, 20, 147, 47, 147, 147, 147, 147, 43, 147, 147, 76, 147, 147, 76, 147, 52, 29, 76, 29, 147, 63, 147, 47, 147, 147, 147, 147, 147, 20, 147, 24, 39, 147, 47, 76, 147, 92, 35, 52, 147, 147, 76, 52, 147, 39, 147, 32, 35])
##    set_optim_nnghs(ds_path_tmax,'tmax', optim_nghs_tmax,OPTIM_NNGH,"optimal number of neighbors for interpolation")
#    
#    optim_nghs_tmin = np.array([52, 63, 111, 92, 69, 22, 111, 69, 101, 111, 63, 63, 84, 122, 69, 57, 111, 69, 47, 101, 69, 63, 92, 57, 76, 76, 76, 63, 84, 52, 57, 47, 63, 76, 57, 84, 63, 63, 111, 63, 147, 20, 147, 92, 84, 84, 84, 76, 76, 63, 39, 92, 134, 122, 111, 92, 111, 84, 84, 63, 35, 134, 69, 63, 57, 43, 52, 35, 52, 84, 39, 111, 76, 84, 76, 63, 92, 69, 84, 134, 111, 29, 122, 111, 57, 111, 111, 76, 76, 84, 84, 92, 84, 76, 63, 122, 63, 92, 76, 52, 76, 76, 63, 63, 92, 76, 69, 84, 76, 147, 101, 76, 69, 84, 84, 92, 101, 69, 57, 69, 101, 101, 84, 147, 39, 52, 47, 39, 39, 147, 84, 84, 52, 63, 57, 69, 52, 92, 84, 111, 69, 76, 69, 101, 76, 84, 84, 76, 101, 111, 84, 76, 57, 84, 111, 111, 147, 147, 76, 134, 84, 92, 111, 101, 122, 76, 111, 69, 63, 63, 43, 47, 63, 84, 69, 84, 92, 57, 84, 84, 63, 92, 57, 101, 84, 57, 76, 76, 43, 52, 84, 52, 76, 69, 101, 101, 69, 57, 43, 69, 122, 122, 52, 101, 84, 92, 76, 76, 111, 111, 69, 84, 84, 147, 92, 92, 92, 69, 122, 69, 76, 57, 84, 69, 76, 84, 84, 92, 76, 147, 92, 134, 39, 111, 69, 111, 63, 84, 111, 69, 63, 101, 134, 69, 111, 76, 57, 63, 101, 111, 52, 92, 92, 63, 101, 134, 111, 111, 92, 63, 84, 111, 76, 122, 69, 101, 52, 84, 92, 92, 76, 57, 69, 69, 43, 111, 76, 57, 76, 63, 84, 134, 111, 92, 76, 57, 76, 92, 101, 76, 76, 84, 84, 76, 92, 69, 147, 84, 43, 63, 32, 101, 39, 101, 52, 52, 92, 76, 122, 76, 69, 69, 52, 63, 63, 47, 63, 57, 57, 84, 39, 76, 69, 92, 101, 76, 84, 52, 76, 69, 111, 111, 63, 92, 69, 101, 47, 69, 92, 47, 52, 57, 57, 69, 47, 76])
#    set_optim_nnghs(ds_path_tmin,'tmin', optim_nghs_tmin,OPTIM_NNGH_ANOM,"optimal number of neighbors for anomaly interpolation")
#        
#    optim_nghs_tmax = np.array([39, 47, 147, 111, 134, 39, 111, 43, 92, 101, 84, 76, 92, 52, 84, 76, 122, 29, 35, 134, 69, 92, 122, 111, 63, 101, 43, 69, 92, 35, 43, 39, 63, 63, 69, 122, 92, 69, 134, 84, 101, 24, 92, 84, 63, 84, 92, 76, 63, 76, 43, 57, 92, 122, 69, 92, 101, 101, 101, 63, 63, 84, 84, 52, 47, 47, 101, 57, 92, 92, 24, 92, 84, 76, 24, 76, 111, 92, 147, 147, 147, 52, 147, 147, 29, 147, 147, 92, 69, 76, 92, 134, 122, 69, 84, 122, 111, 76, 92, 52, 76, 111, 92, 76, 69, 76, 92, 84, 84, 147, 147, 76, 69, 147, 147, 47, 147, 84, 43, 32, 92, 57, 92, 92, 43, 39, 52, 122, 35, 22, 63, 84, 69, 84, 69, 101, 29, 101, 57, 101, 84, 76, 92, 111, 52, 84, 63, 69, 111, 92, 147, 76, 43, 69, 92, 111, 43, 147, 84, 147, 69, 111, 134, 92, 101, 122, 63, 57, 57, 63, 63, 69, 84, 76, 111, 122, 92, 47, 101, 76, 63, 111, 111, 101, 57, 57, 63, 76, 43, 47, 147, 63, 92, 57, 147, 92, 63, 63, 69, 147, 134, 147, 52, 147, 92, 52, 101, 63, 111, 122, 92, 111, 134, 147, 147, 147, 57, 52, 84, 69, 57, 101, 101, 63, 122, 84, 122, 69, 47, 147, 134, 63, 22, 134, 147, 63, 57, 122, 84, 47, 111, 47, 111, 84, 111, 47, 35, 43, 76, 32, 63, 92, 39, 69, 147, 147, 122, 20, 76, 63, 63, 122, 84, 111, 47, 76, 63, 76, 76, 134, 134, 84, 39, 57, 63, 43, 92, 111, 63, 147, 84, 122, 147, 101, 63, 43, 63, 92, 101, 84, 101, 111, 57, 57, 111, 52, 84, 92, 63, 57, 32, 84, 43, 47, 92, 39, 92, 84, 147, 122, 63, 24, 63, 57, 47, 47, 52, 57, 76, 76, 26, 69, 43, 111, 147, 84, 84, 69, 111, 63, 147, 111, 101, 101, 69, 122, 52, 84, 63, 57, 63, 101, 147, 47, 63, 92])
#    set_optim_nnghs(ds_path_tmax,'tmax', optim_nghs_tmax,OPTIM_NNGH_ANOM,"optimal number of neighbors for anomaly interpolation")
    

#################################################################################################################
# 7.) STEP 7: Add aggregated values
#################################################################################################################

#    add_monthly_means('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
#    add_monthly_means('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
#    
#    add_ann_means('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
#    add_ann_means('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')

#    add_monthly_norms('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin', 1981, 2010)
#    add_monthly_norms('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax', 1981, 2010)
