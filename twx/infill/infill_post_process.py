'''
Methods to create final serially complete station time series datasets for input to the interpolation procedures

@author: jared.oyler
'''
from twx.db.create_db_all_stations import dbDataset
from twx.db.station_data import build_stn_struct,STN_ID,DATE,LON,LAT,ELEV,StationSerialDataDb,DTYPE_STN_DFLT,DTYPE_STN_BASIC,StationDataDb,\
    NEON, MASK, BAD, OPTIM_NNGH, OPTIM_NNGH_ANOM, MEAN_TMIN,MEAN_TMAX
from twx.db.reanalysis import NNRds
from netCDF4 import num2date,Dataset,date2num
import netCDF4
import twx.utils.util_dates as utld
import numpy as np
from twx.utils.status_check import status_check
from twx.utils.input_raster import input_raster, OutsideExtent, RasterDataset
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
import mpl_toolkits.basemap as bm

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
        
    stn_da = StationSerialDataDb(ds_path, varname,stn_dtype=DTYPE_STN_BASIC)
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
        
    stn_da = StationSerialDataDb(ds_path, varname,stn_dtype=DTYPE_STN_BASIC)
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
    
    stn_da = StationDataDb(fpath_db)

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
    n_days = len(ds_in.dimensions['time'])
    all_imp_flags = np.ones(n_days,dtype=np.bool)
    all_imp_stns = np.zeros(n_stns,dtype=np.bool)
    
    stat_chk = status_check(n_stns,100)
    for x in np.arange(n_stns):
        
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
            
            all_imp_stns[x] = True
        
        else:
            
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
    
    print "% of stns with all imputed values: "+str((np.sum(all_imp_stns)/np.float(all_imp_stns.size))*100.)

#def add_stn_raster_values(ds_path,var_name,name,units,a_rast,handle_ndata=False, skip_stnids=None):
#    
#    ds = Dataset(ds_path,'r+')
#    lon = ds.variables['lon'][:]
#    lat = ds.variables['lat'][:]
#    stn_id = ds.variables['stn_id'][:].astype("<S16")
#    a_data = None
#    
#    if var_name not in ds.variables.keys():
#        newvar = ds.createVariable(var_name,'f8',('stn_id',),fill_value=a_rast.ndata)
#    else:
#        newvar = ds.variables[var_name]
#    
#    #newvar.setncattr('_FillValue',a_rast.ndata)
#    #newvar._FillValue = a_rast.ndata
#    newvar.long_name = name
#    newvar.units = units
#    newvar.standard_name = name
#    newvar.missing_value = a_rast.ndata
#    newvar[:] = a_rast.ndata
#    ds.sync()
#    
#    for x in np.arange(lon.size):
#        
#        if skip_stnids is not None:
#            
#            if stn_id[x] in skip_stnids:
#                newvar[x] = a_rast.ndata
#                continue
#        try:
#            a = a_rast.getDataValue(lon[x], lat[x])
#            newvar[x] = a
#            if a == a_rast.ndata:
#                raise Exception('No data raster value')
#            
#        except Exception:
#            print "No data value for stn: "+stn_id[x]
#            
#            if handle_ndata:
#                if a_data == None:
#                    a_data = a_rast.readEntireRaster()
#                
#                x_grid,y_grid = a_rast.getGridCellOffset(lon[x],lat[x])
#                newvar[x] = find_nn_data(a_data, a_rast, x_grid, y_grid)
#                print "NN value for "+stn_id[x]+" is "+str(newvar[x])
#    
#    ds.sync()

def add_stn_raster_values(ds_path,var_name,name,units,a_rast,handle_ndata=True,nn=False):
    
    ds = Dataset(ds_path,'r+')
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    stn_id = ds.variables['stn_id'][:].astype("<S16")
    
    if var_name not in ds.variables.keys():
        newvar = ds.createVariable(var_name,'f8',('stn_id',),fill_value=a_rast.ndata)
    else:
        newvar = ds.variables[var_name]
    
    newvar.long_name = name
    newvar.units = units
    newvar.standard_name = name
    newvar.missing_value = a_rast.ndata
    newvar[:] = a_rast.ndata
    ds.sync()
    
    ###################################
    a = a_rast.readAsArray()
    aflip = np.flipud(a)
    aflip = aflip.astype(np.float)
    a = a.data
    
    yGrid,xGrid = a_rast.getCoordGrid1d()
    yGrid = np.sort(yGrid)
    
    interpOrder = 0 if nn else 1
    
    rvals = np.zeros(len(newvar[:]))
    
    schk = status_check(lon.size, 1000) 
    for x in np.arange(lon.size):
        
        rval = bm.interp(aflip, xGrid, yGrid, np.array(lon[x]), np.array(lat[x]), checkbounds=False, masked=True, order=interpOrder)
        
        if np.ma.is_masked(rval):
            
            if handle_ndata:
            
                rval = bm.interp(aflip, xGrid, yGrid, np.array(lon[x]), np.array(lat[x]), checkbounds=False, masked=True, order=0)
                
                if np.ma.is_masked(rval):
                    row,col = a_rast.getRowCol(lon[x], lat[x])
                    rval = find_nn_data(a, a_rast, col, row)
            
            else:
                
                rval = a_rast.ndata
        
        rvals[x] = rval
        schk.increment()
    
    newvar[:] = rvals
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
    lats,lons = a_rast.getCoord(nn[:,0], nn[:,1])#getLatLon(nn[:,1], nn[:,0], transform=False)
    pt_lat,pt_lon = a_rast.getCoord(y,x)
    d = utlg.grt_circle_dist(pt_lon,pt_lat, lons, lats)
    j = np.argsort(d)[0]
    nval = nn_vals[j]
    return nval
    

def output_dup_stns(fpath_infilldb,tair_var,fpath_out,mode="w"):
    
    stn_da = StationSerialDataDb(fpath_infilldb,tair_var,stn_dtype=DTYPE_STN_BASIC)
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
        
    stn_da = StationSerialDataDb(dspath,varname_tair)
    
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
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/infill_nonhomog_20140329/' 
    params[P_PATH_NNR] = '/projects/daymet2/reanalysis_data/conus_subset/'
    params[P_PATH_R_FUNCS] = '/home/jared.oyler/repos/twx/twx/lib/rpy/pca_infill.R'
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    params[P_MIN_NNGH_DAILY] = 7
    params[P_NNGH_NNR] = 4
    params[P_NNR_VARYEXPLAIN] = 0.99
    params[P_FRACOBS_INIT_PCS] = 0.5
    params[P_PPCA_VARYEXPLAIN] = 0.99
    params[P_CHCK_IMP_PERF] = True
    params[P_NPCS_PPCA] = 0
    

    source_r(params[P_PATH_R_FUNCS])
    
    stn_id = 'GHCN_USW00012924'
    tair_var = 'tmax'
    ppcaConThres=1e-5
    runUpdate = True
    tair_mask = None
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
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
    aclib = clib_wxTopo()
    
    
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
    
    stnda = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc','tmax_imp',stn_dtype=DTYPE_STN_BASIC)
    stnda2 = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc','tmax',stn_dtype=DTYPE_STN_BASIC)
    
    
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
    
    stnda = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc','tmax_imp',stn_dtype=DTYPE_STN_BASIC)
    stnda2 = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc','tmax',stn_dtype=DTYPE_STN_BASIC)
    
    
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

def add_monthly_means(ds_path,var_name):
    
    stnda = StationSerialDataDb(ds_path, var_name)
    tagg = ushcn.TairAggregate(stnda.days)
    minDate = stnda.days[DATE][0]
    stns = stnda.stns
    stnda.ds.close()
    stnda = None
    ds = Dataset(ds_path,'r+')
    
    if 'time_mth' not in ds.variables.keys():
        
        ds.createDimension('time_mth',tagg.yr_mths.size)
        times = ds.createVariable('time_mth','f8',('time_mth',),fill_value=False)
        times.units = "".join(["days since ",str(minDate.year),"-",str(minDate.month),"-",str(minDate.day)," 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times[:] = date2num(tagg.yr_mths[DATE],times.units)
        
        varMthly = ds.createVariable("_".join([var_name,"mth"]),'f4',('time_mth','stn_id'),fill_value=netCDF4.default_fillvals['f4'])
    
    else:
        
        varMthly = ds.variables["_".join([var_name,"mth"])]
        
    varDly = ds.variables[var_name]
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
    
    stnda = StationSerialDataDb(dsPath, varName,stn_dtype=DTYPE_STN_BASIC)
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
    
    stnda = StationSerialDataDb(dsPath, varName)
    tagg = ushcn.TairAggregate(stnda.days)
    minDate = stnda.days[DATE][0]
    stns = stnda.stns
    stnda.ds.close()
    stnda = None
    ds = Dataset(dsPath,'r+')
    
    if 'time_ann' not in ds.variables.keys():
        
        ds.createDimension('time_ann',tagg.u_yrs.size)
        times = ds.createVariable('time_ann','f8',('time_ann',),fill_value=False)
        times.units = "".join(["days since ",str(minDate.year),"-",str(minDate.month),"-",str(minDate.day)," 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times[:] = date2num([datetime(yr,1,1) for yr in tagg.u_yrs],times.units)
        
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
#    ds_tmin = Dataset('/projects/daymet2/station_data/infill/infill_nonhomog_20140329/infill_tmin.nc')
#    build_serial_complete_ds(ds_tmin, 'tmin', '/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmin.nc')
#################################################################################################################   
#################################################################################################################
#CREATE A FINAL SERIALLY COMPLETE TMAX DATASET FROM IMPUTED/INFILLED STATIONS
#################################################################################################################  
#    ds_tmax = Dataset('/projects/daymet2/station_data/infill/infill_nonhomog_20140329/infill_tmax.nc')
#    build_serial_complete_ds(ds_tmax, 'tmax', '/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmax.nc')
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

#    ds_path_tmin = '/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmin.nc'
#    ds_path_tmax = '/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmax.nc'
#    
#    stnda_tmin = StationSerialDataDb('/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc', 'tmin')
#    stnda_tmax = StationSerialDataDb('/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc', 'tmax')
#    
#    rm_ids = np.concatenate((stnda_tmin.stn_ids[np.isfinite(stnda_tmin.stns[BAD])],stnda_tmax.stn_ids[np.isfinite(stnda_tmax.stns[BAD])]))
#    rm_ids = np.unique(rm_ids)
#    
#    stnda_tmin = None
#    stnda_tmax = None
#    
##    ds_path_tmin = '/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc'
##    ds_path_tmax = '/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc'
##    rm_stns = np.loadtxt('/projects/daymet2/station_data/infill/infill_20130725/BadStns.csv',delimiter=",",dtype=np.str,skiprows=1,usecols=(0,))
##    rm_ids = np.unique(rm_stns)
#    print rm_ids.size
#    set_rm_bad_stations(rm_ids, ds_path_tmin)
#    set_rm_bad_stations(rm_ids, ds_path_tmax)

#Fallback if issues: Reset all stations to okay
#    set_rm_bad_stations([],'/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc')
#    set_rm_bad_stations([],'/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
#    set_rm_bad_stations([],'/projects/daymet2/station_data/infill/infill_20130725/infill_tmin.nc')
#    set_rm_bad_stations([],'/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc')  

################################################################################################################# 
    
#################################################################################################################
# 4.) STEP 4: Check for dups and then set the Bad flag on them and other bad stations
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
##    set_rm_bad_stations(allIds, '/projects/daymet2/station_data/infill/infill_20130725/infill_tmin.nc')
##    set_rm_bad_stations(allIds, '/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc')
##    
#    set_rm_bad_stations(allIds, '/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc')
#    set_rm_bad_stations(allIds, '/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc')

#################################################################################################################
# 5.) STEP 5: Add auxilary predictor values from rasters
#################################################################################################################
#    ds_path_tmin = '/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc'
#    ds_path_tmax = '/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc'
#    ds_path_tmin = '/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmin.nc'
#    ds_path_tmax = '/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmax.nc'


#TMIN LST
#    name = 'land surface temperature'
#    units = "C"
#    for mth in np.arange(1,13):
#        var_name = 'lst%02d'%mth
#        print var_name
#        a_rast = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Night_1km.%02d.C.tif'%mth)
#        add_stn_raster_values(ds_path_tmin, var_name, name, units, a_rast)

###TMAX LST
#    name = 'land surface temperature'
#    units = "C"
#    for mth in np.arange(1,13):
#        var_name = 'lst%02d'%mth
#        print var_name
#        a_rast = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/mthly_lst/MOSAIC.LST_Day_1km.%02d.C.tif'%mth)
#        add_stn_raster_values(ds_path_tmax, var_name, name, units, a_rast)
#    
##TMIN TDI
#    var_name = 'tdi'
#    name = 'topographic dissection index'
#    units = "[3,6,9,12,15] km radius"
#    a_rast = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/tdi.tif')
#    add_stn_raster_values(ds_path_tmin, var_name, name, units, a_rast)
#    
##TMAX TDI
#    var_name = 'tdi'
#    name = 'topographic dissection index'
#    units = "[3,6,9,12,15] km radius"
#    a_rast = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/tdi.tif')
#    add_stn_raster_values(ds_path_tmax, var_name, name, units, a_rast)
#
##TMAX Interp Mask
#    var_name = 'mask'
#    name = 'Interpolation Mask'
#    units = "0 or 1"
#    a_rast = RasterDataset('/projects/daymet2/dem/interp_grids/conus/tifs/mask_all_nd.tif')
#    a_rast.ndata = 0
#    add_stn_raster_values(ds_path_tmax, var_name, name, units, a_rast,handle_ndata=False,nn=True)
#
##TMIN Interp Mask
#    var_name = 'mask'
#    name = 'Interpolation Mask'
#    units = "0 or 1"
#    a_rast = RasterDataset('/projects/daymet2/dem/interp_grids/conus/tifs/mask_all_nd.tif')
#    a_rast.ndata = 0
#    add_stn_raster_values(ds_path_tmin, var_name, name, units, a_rast,handle_ndata=False,nn=True)
##
###U.S. Climate Divisions
#    var_name = "neon"
#    name = "U.S. Climate Division"
#    units = "NA"
#    a_rast = RasterDataset('/projects/daymet2/dem/interp_grids/tifs/climdivLccMerge.tif')
#    add_stn_raster_values(ds_path_tmin, var_name, name, units, a_rast,handle_ndata=False,nn=True)
#    add_stn_raster_values(ds_path_tmax, var_name, name, units, a_rast,handle_ndata=False,nn=True)
    
#################################################################################################################

#################################################################################################################
# 7.) STEP 7: Add aggregated values
#################################################################################################################

#    add_monthly_means('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
#    add_monthly_means('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
#    
#    add_ann_means('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc', 'tmin')
#    add_ann_means('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')

    add_monthly_norms('/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmin.nc','tmin', 1981, 2010)
    add_monthly_norms('/projects/daymet2/station_data/infill/infill_nonhomog_20140329/serial_tmax.nc','tmax', 1981, 2010)
