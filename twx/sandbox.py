#import matplotlib
#matplotlib.use('TkAgg')

#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
from twx.infill.infill_post_process import RM_STN_FLAG, RM_STN_DUP
from test.test_robotparser import bad
robjects.conversion.py2ri = numpy2ri
r = robjects.r

from copy import copy
from twx.db.snotel_clean import parse_hist_stns,filterToLowResSnotel
from twx.utils.input_raster import input_raster,RasterDataset
import twx.utils.util_dates as utld
from twx.utils.util_dates import YMD
import netCDF4
from netCDF4 import Dataset
import numpy as np
from datetime import datetime
from datetime import date
from netCDF4 import date2num,num2date
from twx.db.station_data import STN_ID,STATE,LON,TDI,LST,NEON,LAT,ELEV,STN_NAME,TMIN,TMAX,PRCP,TMIN_FLAG,TMAX_FLAG,PRCP_FLAG,MONTH,station_data_ncdb,station_data_infill,YEAR,SWE,get_neon_rgns,station_data_combine,MEAN_OBS,DTYPE_STN_DFLT,DTYPE_STN_MEAN_LST_TDI,MEAN_TMIN,MEAN_TMAX,VAR_TMIN,VAR_TMAX,DTYPE_STN_BASIC,VCF,OPTIM_NNGH,\
    MASK,BAD, OPTIM_NNGH_ANOM, get_lst_varname
import sys
import matplotlib.pyplot as plt
from twx.db.all_create_db import create_db_ncdf, MISSING,copy_db_ncdf_nometa
from twx.db.all_create_db import insert_glac,insert_data_ncdf,insert_usfs
import sqlite3 as sql
from twx.utils.status_check import status_check
from twx.utils.util_ncdf import ncdf_raster,to_geotiff,to_ncdf,expand_grid
from qa.qa_location import load_locs_fixed
import twx.utils.util_geo as utlg
from twx.infill.infill_daily import gwrpca_matrix,source_r,pca_matrix,calc_hss,calc_forecast_scores,pca_matrix_prcp,infill_prcp,PO_THRESHS,prcp_infill_results,build_yr_mth_masks,tmin_tmax_fixer,ioapca_matrix,nnrpca_matrix,ImputeMatrixPCA,MIN_NNR_VAR
from qa.qa_location import get_elev_usgs

from twx.infill.infill_normals import infill_tair,build_mth_masks,MTH_BUFFER,infill_prcp_norm,ImputeMatrix,impute_tair_norm
from twx.infill.obs_por import load_por_csv,POR_DTYPE,build_valid_por_masks
from twx.interp.station_select import station_select
from twx.interp.interp_tair import KrigTair, LST_TMAX, LST_TMIN, StationDataWrkChk
from twx.interp.clibs import clib_wxTopo
from twx.utils.status_check import timer
import twx.interp.interp_prcp as ip
import twx.interp.interp_tair as it
import twx.utils.ncdf_raster as nr
import os
import osgeo.gdalconst as gdalconst
import osgeo.gdal as gdal
import scipy.stats as stats
import infill.obs_por as obs_por
from qa.qa_spatial import run_qa_spatial
import twx.utils.decimaldegrees as dd
from twx.qa.qa_location import get_elev_geonames,get_elev_usgs
import ogr
import infill.random_xval_stations as xval
from twx.db.reanalysis import NNRds,NNRNghData
import twx.interp.tiling as ti
from twx.interp.interp_constants import RM_STN_IDS_TAIR,AREA_MONTANA_BUFFER
from twx.infill.infill_post_process import output_no_tdi_stns
#import osgeo.osr as osr
#from modis.montana_ndvi import modis_sin_rast
import time
from twx.interp.station_select import station_select_neon

from twx.db import station_data, ushcn
from twx.utils.output_raster import output_raster
from qa import qa_temp
from qa.qa_temp import run_qa_spatial_only
from twx.interp.topo_disect import TopoDisectDEM
from twx.interp.station_select import NEON_AREAS
from mpl_toolkits.basemap import Basemap
import twx.interp.tiling as tl
import qa.qa_change_point as qa_cp
import matplotlib.cm as cm
import matplotlib
from qa.qa_location import get_elev
from modis.montana_ndvi import EOSGridSD,KtoC
from modis.montana_ndvi import modis_sin_rast
from osgeo import osr
from twx.infill.infill_post_process import runs_of_ones_array
from twx.utils.util_ncdf import toGTiff,GeoNc
import cProfile


def load_stns():
    
    ncdf_data = Dataset('/projects/daymet2/station_data/all_infill/decade_split/test_stns_infill.nc')
    ncdf_stns = ncdf_data.variables['stns'][:]
    print ncdf_stns[STN_ID].shape
    stns =  np.recarray(ncdf_stns[STN_ID].shape[0],dtype=[(STN_ID,"<U16"),(STATE,"<U2"),(LON,np.float64),(LAT,np.float64),(ELEV,np.float64),
                                      (INFILLED_TMIN,np.bool),(INFILLED_TMAX,np.bool),(INFILLED_PRCP,np.bool),
                                      (POPCRIT_JAN,np.float64),(POPCRIT_FEB,np.float64),(POPCRIT_MAR,np.float64),
                                      (POPCRIT_APR,np.float64),(POPCRIT_MAY,np.float64),(POPCRIT_JUN,np.float64),
                                      (POPCRIT_JUL,np.float64),(POPCRIT_AUG,np.float64),(POPCRIT_SEP,np.float64),
                                      (POPCRIT_OCT,np.float64),(POPCRIT_NOV,np.float64),(POPCRIT_DEC,np.float64)])
    for x in np.arange(ncdf_stns[STN_ID].shape[0]):
        stns[STN_ID][x] = ncdf_stns[STN_ID][x].tostring()
        stns[STATE][x] = ncdf_stns[STATE][x].tostring()
    print stns[STN_ID][x] == stns[STN_ID][x] 
    print stns[STATE][x]
    stns[INFILLED_TMIN] = ncdf_stns[INFILLED_TMIN]


def create_topomet_ncdf_file(path,var,lons,lats,days,srt_yr,end_yr):
    
    ncdf_file = Dataset("".join([path,"topomet_montana_",var,".nc"]),'w')
    #Set global attributes
    title = "".join(["Daily Interpolated CONUS Meteorological Data ",str(srt_yr),"-",str(end_yr)])
    ncdf_file.title = title
    ncdf_file.institution = " | ".join(["University of Montana Numerical Terradynamics Simulation Group",
                                "NASA Ames Ecological Forecasting Lab"])
    ncdf_file.source = "TopoMet v0.0.1"
    ncdf_file.history = "".join(["Created on: ",datetime.strftime(date.today(),"%Y-%m-%d")]) 
    ncdf_file.references = "http://www.ntsg.umt.edu/project/topomet"
    ncdf_file.comment = "30-arcsec spatial resolution, daily timestep"
    
    #Get year mask to set number of days for interpolation
    yr_mask = np.logical_and(days[YEAR] >= srt_yr,days[YEAR] <= end_yr)
    
    #Create 3-dimensions
    dim_time = ncdf_file.createDimension('time',np.nonzero(yr_mask)[0].size)
    dim_lat = ncdf_file.createDimension('lat',lats.size)
    dim_lon = ncdf_file.createDimension('lon',lons.size)
    
    #Create dimension variables and fill with values
    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.units = "days since 1950-1-1 0:0:0"
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE][yr_mask],times.units)
    
    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    return ncdf_file


def init_ncdf(path,lons,lats,days,srt_yr,end_yr,row_chksize,col_chksize,verb=1):
    
    #TMIN
#    ncdf_tmin = create_topomet_ncdf_file(path,'tmin',lons, lats, days, srt_yr, end_yr)
#    chnk_size = (len(ncdf_tmin.dimensions['time']),row_chksize,col_chksize)
#    if verb >= 1: print "netcdf chunk size:  "+str(chnk_size)
#    ncdf_var = ncdf_tmin.createVariable('tmin','f4',('time','lat','lon',),chunksizes=chnk_size,zlib=True)
#    ncdf_var.long_name = "minimum air temperature"
#    ncdf_var.units = "C"
#    ncdf_var.standard_name = "minimum_air_temperature"
#    ncdf_var.missing_value = netCDF4.default_fillvals['f4']
    
    #TMAX
    ncdf_tmax = create_topomet_ncdf_file(path,'tmax',lons, lats, days, srt_yr, end_yr)
    chnk_size = (len(ncdf_tmax.dimensions['time']),row_chksize,col_chksize)
    ncdf_var = ncdf_tmax.createVariable('tmax','f4',('time','lat','lon',),chunksizes=chnk_size,zlib=True)
    ncdf_var.long_name = "maximum air temperature"
    ncdf_var.units = "C"
    ncdf_var.standard_name = "maximum_air_temperature"
    ncdf_var.missing_value = netCDF4.default_fillvals['f4']
    
    #PRCP
    ncdf_prcp = create_topomet_ncdf_file(path,'prcp',lons, lats, days, srt_yr, end_yr)
    ncdf_var = ncdf_prcp.createVariable('prcp','f4',('time','lat','lon',),chunksizes=chnk_size,zlib=True)
    ncdf_var.long_name = "precipitation amount"
    ncdf_var.units = "cm"
    ncdf_var.standard_name = "precipitation_amount"
    ncdf_var.missing_value = netCDF4.default_fillvals['f4']
    
    #VPD
    ncdf_vpd = create_topomet_ncdf_file(path,'vpd',lons, lats, days, srt_yr, end_yr)
    ncdf_var = ncdf_vpd.createVariable('vpd','f4',('time','lat','lon',),chunksizes=chnk_size,zlib=True)
    ncdf_var.long_name = "water vapor saturation deficit"
    ncdf_var.units = "Pa"
    ncdf_var.standard_name = "water_vapor_saturation_deficit"
    ncdf_var.missing_value = netCDF4.default_fillvals['f4']
    
    #SRAD
    ncdf_srad = create_topomet_ncdf_file(path,'srad',lons, lats, days, srt_yr, end_yr)
    ncdf_var = ncdf_srad.createVariable('srad','f4',('time','lat','lon',),chunksizes=chnk_size,zlib=True)
    ncdf_var.long_name = "surface net downward shortwave flux"
    ncdf_var.units = "W m-2"
    ncdf_var.standard_name = "surface_net_downward_shortwave_flux"
    ncdf_var.missing_value = netCDF4.default_fillvals['f4']
    
    #ncdf_tmin.close()
    ncdf_tmax.close()
    ncdf_prcp.close()
    ncdf_vpd.close()
    ncdf_srad.close()


def write_subset(var_out,var_in,var_name,mask_time,row_str,nrows,col_str,ncols,row_chksize,col_chksize):
    
    x = 0
    for i in np.arange(row_str,row_str+nrows,row_chksize):
        
        if i + row_chksize < row_str+nrows:
            numRows = row_chksize
        else:
            numRows = row_str+nrows - i
          
        for j in np.arange(col_str,col_str+ncols,col_chksize):
            
            if j + col_chksize < col_str+ncols:
                numCols = col_chksize
            else:
                numCols = col_str+ncols - j
            
            x+=1
            var_out[:,i-row_str:i-row_str+numRows,j-col_str:j-col_str+numCols] = var_in[mask_time,i:i+numRows,j:j+numCols]
            print "".join(["wrote chk #",str(x)," for ",var_name])

def broot_subset():
    
    #The coordinates of our domain are:
    #Upper left corner (NW), lat = 49.217, lon = -117.8986
    #Lower right corner (SE), lat = 43.783, lon = -110.1014
    
    ds = Dataset('/projects/daymet2/nasa_debug/topomet_avg_tmin.nc')
    days = utld.get_days_metadata()
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    mask_lon = np.logical_and(lon>=-117.8986,lon<=-110.1014)
    mask_lat = np.logical_and(lat>=43.783,lat<=49.217)
    mask_time = np.logical_and(days[YEAR]>=2000,days[YEAR]<=2009)
    
    lons = lon[mask_lon]
    lats = lat[mask_lat]
    
    str_row = np.nonzero(mask_lat)[0][0]
    nrow = lats.size
    str_col = np.nonzero(mask_lon)[0][0]
    ncol = lons.size
    
#    ds_tmin = Dataset('/projects/daymet2/interp_output/fullrun20110916/topomet_tmin.nc')
#    ds_tmax = Dataset('/projects/daymet2/interp_output/fullrun20110916/topomet_tmax.nc')
#    ds_prcp = Dataset('/projects/daymet2/interp_output/fullrun20110916/topomet_prcp.nc')
#    ds_srad = Dataset('/projects/daymet2/interp_output/fullrun20110916/topomet_srad.nc')
    ds_vpd = Dataset('/projects/daymet2/interp_output/fullrun20110916/topomet_vpd.nc')
    
#    init_ncdf('/projects/daymet2/interp_output/fullrun20110916/subset_mt/',lons, lats, days,2000,2009,50,50)
#    
##    out_tmin = Dataset('/projects/daymet2/interp_output/fullrun20110916/subset_mt/topomet_montana_tmin.nc',"r+")
##    write_subset(out_tmin.variables['tmin'], ds_tmin.variables['tmin'],'tmin',mask_time, str_row, nrow, str_col, ncol, 50,50)
##    out_tmin.close()
#    
#    out_tmax = Dataset('/projects/daymet2/interp_output/fullrun20110916/subset_mt/topomet_montana_tmax.nc',"r+")
#    write_subset(out_tmax.variables['tmax'], ds_tmax.variables['tmax'],'tmax',mask_time, str_row, nrow, str_col, ncol, 50,50)
#    out_tmax.close()
#    
#    out_prcp = Dataset('/projects/daymet2/interp_output/fullrun20110916/subset_mt/topomet_montana_prcp.nc',"r+")
#    write_subset(out_prcp.variables['prcp'], ds_prcp.variables['prcp'],'prcp',mask_time, str_row, nrow, str_col, ncol, 50,50)
#    out_prcp.close()
#    
#    out_srad = Dataset('/projects/daymet2/interp_output/fullrun20110916/subset_mt/topomet_montana_srad.nc',"r+")
#    write_subset(out_srad.variables['srad'], ds_srad.variables['srad'],'srad',mask_time, str_row, nrow, str_col, ncol, 50,50)
#    out_srad.close()
    
    out_vpd = Dataset('/projects/daymet2/interp_output/fullrun20110916/subset_mt/topomet_montana_vpd.nc',"r+")
    write_subset(out_vpd.variables['vpd'], ds_vpd.variables['vpd'],'vpd',mask_time, str_row, nrow, str_col, ncol, 50,50)
    out_vpd.close()
    
    
#    out_tmin.variables['tmin'][:,:,:] = ds_tmin.variables['tmin'][mask_time,mask_lat,mask_lon]
#    out_tmin.close()
#    print "tmin written"
#    
#    out_tmax = Dataset('/projects/daymet2/interp_output/fullrun20110916/subset_mt/topomet_montana_tmax.nc',"r+")
#    out_tmax.variables['tmax'][:,:,:] = ds_tmax.variables['tmax'][mask_time,mask_lat,mask_lon]
#    out_tmax.close()
#    print "tmax written"
#    
#    out_prcp = Dataset('/projects/daymet2/interp_output/fullrun20110916/subset_mt/topomet_montana_prcp.nc',"r+")
#    out_prcp.variables['prcp'][:,:,:] = ds_prcp.variables['prcp'][mask_time,mask_lat,mask_lon]
#    out_prcp.close()
#    print "prcp written"
#    
#    out_srad = Dataset('/projects/daymet2/interp_output/fullrun20110916/subset_mt/topomet_montana_srad.nc',"r+")
#    out_srad.variables['srad'][:,:,:] = ds_srad.variables['srad'][mask_time,mask_lat,mask_lon]
#    out_srad.close()
#    print "srad written"
#    
#    out_vpd = Dataset('/projects/daymet2/interp_output/fullrun20110916/subset_mt/topomet_montana_vpd.nc',"r+")
#    out_vpd.variables['vpd'][:,:,:] = ds_vpd.variables['vpd'][mask_time,mask_lat,mask_lon]
#    out_vpd.close()
#    print "vpd written"

def check_smry_file():
    ds = Dataset('/projects/daymet2/interp_output/bootstrap/topomet_stderr_tmin.nc')
    tmin = ds.variables['tmin'][:,:]
    plt.imshow(tmin)
    plt.show()


def check_station_netcdf():
    
    stn_db = station_data_ncdb("/projects/daymet2/station_data/crown_stns.nc")
    stns = stn_db.load_stns()
    

    obs = stn_db.load_all_stn_obs(np.array([stns[STN_ID][0]]), set_flagged_nan=True)
    plt.plot(obs[PRCP])
    plt.show()
    print "done"


def check_qaflags():
    stn_db = station_data_ncdb("/projects/daymet2/station_data/crown_stns.nc")
    stns = stn_db.load_stns()
    stn_mask = np.char.find(stns[STN_ID],"GLAC") != -1
    
    stns_glac = stns[stn_mask]
    #print stns_glac[STN_ID]
    #print stns_glac[STN_NAME]
    
#    for x in np.arange(stns_glac.size):
#        print "".join([stns_glac[STN_ID][x],": ",stns_glac[STN_NAME][x]])
    
    obs_glac = stn_db.load_all_stn_obs(stns_glac[STN_ID], set_flagged_nan=False)
    
    #TMIN % Flagged 
    
    print "TMIN QA Flagged"
    for stn_id in stns_glac[STN_ID]:
        
        stn_idx = np.nonzero(stns_glac[STN_ID] == stn_id)[0][0]
        
        tmin = obs_glac[TMIN][:,stn_idx]
        tmin_flags = obs_glac[TMIN_FLAG][:,stn_idx]
        
        obs_mask = np.isfinite(tmin)
        
        nflg = tmin[np.logical_and(obs_mask,tmin_flags != "")].size
        if nflg > 0:
            pct_flg = float(nflg)/float(tmin[obs_mask].size)*100.0
        else:
            pct_flg = 0
        
        print "".join([stns_glac[STN_NAME][stn_idx],": ","%.3f"%(pct_flg),"%"," (",str(nflg),")"])
    
    print "##################################################"
    print "TMAX QA Flagged"
    for stn_id in stns_glac[STN_ID]:
        
        stn_idx = np.nonzero(stns_glac[STN_ID] == stn_id)[0][0]
        
        tmax = obs_glac[TMAX][:,stn_idx]
        tmax_flags = obs_glac[TMAX_FLAG][:,stn_idx]
        
        obs_mask = np.isfinite(tmax)
        
        nflg = tmax[np.logical_and(obs_mask,tmax_flags != "")].size
        if nflg > 0:
            pct_flg = float(nflg)/float(tmax[obs_mask].size)*100.0
        else:
            pct_flg = 0
        
        print "".join([stns_glac[STN_NAME][stn_idx],": ","%.3f"%(pct_flg),"%"," (",str(nflg),")"])
    
    
    for stn_id in stns_glac[STN_ID]:
        
        stn_idx = np.nonzero(stns_glac[STN_ID] == stn_id)[0][0]
        
        tmin = obs_glac[TMIN][:,stn_idx]
        tmax = obs_glac[TMAX][:,stn_idx]
        tmin_flags = obs_glac[TMIN_FLAG][:,stn_idx]
        tmax_flags = obs_glac[TMAX_FLAG][:,stn_idx]
        
        obs_mask = np.logical_or(np.isfinite(tmin),np.isfinite(tmax))
        
        ymd = stn_db.days[YMD][obs_mask]
        tmin = tmin[obs_mask]
        tmax = tmax[obs_mask]
        tmin_flags = tmin_flags[obs_mask]
        tmax_flags = tmax_flags[obs_mask]
        
        name = stns_glac[STN_NAME][stn_idx]
        name = name.replace(" ","_")
        name = name.replace("(","[")
        name = name.replace(")","]")
        
        afile = open("".join(["/projects/daymet2/station_data/glac/qa/",name,".csv"]),"w")
        afile.write(",".join(["YMD","TMIN","TMAX","TMIN_QFLG","TMAX_QFLG"])+"\n")
        
        for x in np.arange(ymd.size):
            
            afile.write("".join([",".join([str(ymd[x]),"%.3f"%(tmin[x]),"%.3f"%(tmax[x]),tmin_flags[x],tmax_flags[x]]),"\n"]))
        afile.close()
        
        
#        if stn_id == "GLAC_7":
#            stn_idx = np.nonzero(stns_glac[STN_ID] == stn_id)[0][0]
#        
#            tmin_flags = obs_glac[TMIN_FLAG][:,stn_idx]
#            tmax_flags = obs_glac[TMAX_FLAG][:,stn_idx]
#        
#            print stn_id
#            print TMIN
#            for flag in np.unique(tmin_flags):
#                if flag != "":
#                    print "".join([flag,": ",str(tmin_flags[tmin_flags==flag].size)])
#                    print stn_db.days[YMD][tmin_flags==flag]
#            print TMAX
#            for flag in np.unique(tmax_flags):
#                if flag != "":
#                    print "".join([flag,": ",str(tmax_flags[tmax_flags==flag].size)])
#                    print stn_db.days[YMD][tmax_flags==flag]
    
    
#    plt.plot(obs_glac[TMIN][:,stn_idx])
#    plt.plot(obs_glac[TMAX][:,stn_idx])
#    plt.show()

#    ds = Dataset("/projects/daymet2/station_data/crown_stns.nc",'r')
#    stn_idx = np.nonzero(ds.variables['stn_id'][:]=='SNOTEL_13C01S')[0][0]
#    flags =  ds.variables['qflag_prcp'][:,stn_idx]
#    print flags[~flags.mask]
#    #plt.plot(ds.variables['tmin'][:,stn_idx])
#    #plt.show()
#    
#    CONN = sql.connect('/projects/daymet2/station_data/all/all.db')
#    CURS = CONN.cursor()
#    
#    stn_da = station_data(CURS,srtDate=datetime(1990,1,1),endDate=datetime(2011,12,31))
#    obs = stn_da.load_all_stn_obs(np.array(['SNOTEL_13C01S']),100, cache=False, set_flagged_nan=False, load_swe=False)
#
#    plt.plot(obs[PRCP]-ds.variables['prcp'][:,stn_idx].data)
#    print obs[PRCP_FLAG][obs[PRCP_FLAG] != '']
#    plt.show()
#def station_netcdf_insert():
#    
#    ds = Dataset("/projects/daymet2/station_data/crown_stns_final.nc",'r+')
#    x = len(ds.dimensions['stn_id'])
#    
#    CONN = sql.connect('/projects/daymet2/station_data/all/all.db')
#    CURS = CONN.cursor()
#
#    stn_da = station_data(CURS,srtDate=datetime(1990,1,1),endDate=datetime(2011,12,31))
#    stns = stn_da.load_stns()
#    
#    lon_mask = np.logical_and(stns[LON]>=-122.0,stns[LON]<=-105.0)
#    lat_mask = np.logical_and(stns[LAT]>=42.0,stns[LAT]<=51.0)
#    bnds_mask = np.logical_and(lon_mask,lat_mask)
#    
#    stns = stns[bnds_mask]
#    
#    for stn in stns:
#        ds.variables['stn_id'][x] = stn[STN_ID]
#        ds.variables['lat'][x] = stn[LAT]
#        ds.variables['lon'][x] = stn[LON]
#        ds.variables['elev'][x] = stn[ELEV]
#        ds.variables['state'][x] = stn[STATE]
#        ds.variables['name'][x] = stn[STN_NAME]
#        x+=1
#    ds.sync()
#
#    stn_ids = ds.variables['stn_id'][:]    
#    obs = stn_da.load_all_stn_obs(stns[STN_ID],100, cache=False, set_flagged_nan=False, load_swe=False)
#    obs[TMIN][np.isnan(obs[TMIN])] = MISSING
#    obs[TMAX][np.isnan(obs[TMAX])] = MISSING
#    obs[PRCP][np.isnan(obs[PRCP])] = MISSING
#    
#    stat_chk = status_check(stns.size,50)
#    for i in np.arange(stns[STN_ID].size):
#        
#        stn_idx = np.nonzero(stn_ids==stns[STN_ID][i])[0][0]
#     
#        ds.variables['tmin'][:,stn_idx] = obs[TMIN][:,i]
#        ds.variables['tmax'][:,stn_idx] = obs[TMAX][:,i]
#        ds.variables['prcp'][:,stn_idx] = obs[PRCP][:,i]
#        ds.variables['qflag_tmin'][:,stn_idx] = obs[TMIN_FLAG][:,i]
#        ds.variables['qflag_tmax'][:,stn_idx] = obs[TMAX_FLAG][:,i]
#        ds.variables['qflag_prcp'][:,stn_idx] = obs[PRCP_FLAG][:,i]
#        stat_chk.increment()
#    ds.close()

def reset_glac_flags():
    
    db = station_data_ncdb("/projects/daymet2/station_data/crown_stns_final.nc")
    
    obs = db.load_all_stn_obs(np.array(['GLAC_7']), set_flagged_nan=False)
    print db.days[YMD][obs[TMIN_FLAG]!= ""]
    print np.unique(obs[TMIN_FLAG])
    print np.unique(obs[TMAX_FLAG])
    print np.unique(obs[PRCP_FLAG])
    
    

def station_netcdf():
    
#    ds = Dataset("/projects/daymet2/station_data/ncdf_stn_test.nc")
#    print ds.variables['stn_id'][18]
#    tmax = ds.variables['tmax'][:,18]
#    print tmax[~tmax.mask]
#    plt.plot(tmax)
#    plt.show()
#    sys.exit()
#    ncdf_file = Dataset("/projects/daymet2/station_data/ncdf_stn_test.nc",'r')
#    ncdf_file.variables['station'][3] = 1.1
#    sys.exit()
#
#    ncdf_file = Dataset("/projects/daymet2/station_data/ncdf_stn_test.nc",'w')
#    
#    days = utld.get_days_metadata()
#    
#    dim_time = ncdf_file.createDimension('time',size=days[DATE].size)
#    dim_station = ncdf_file.createDimension('station',size=5)
#    
#    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
#    times.long_name = "time"
#    times.units = "days since 1948-1-1 0:0:0"
#    times.standard_name = "time"
#    times.calendar = "standard"
#    times[:] = date2num(days[DATE],times.units)
#    
#    stations = ncdf_file.createVariable('station','f8',('station',))
#    stations.long_name = "station"
#    stations.standard_name = "station"
#    
#    ncdf_file.close()

    GLAC_DATA_PATH = '/projects/daymet2/station_data/glac/ftp_site/ftpext.usgs.gov/pub/cr/mt/west.glacier/lbengtson/Oyler/Daily_GNP_vs_topomet/csv/'

    create_db_ncdf("/projects/daymet2/station_data/crown_stns_final.nc",1990,2011)
    
    glac = insert_glac('/projects/daymet2/station_data/glac/gnp_alpine_lonlat.csv',
                       GLAC_DATA_PATH,1990,2011)
    
    insert_data_ncdf("/projects/daymet2/station_data/crown_stns_final.nc", [glac])
    sys.exit()
    
#    ncdf_file = Dataset("/projects/daymet2/station_data/ncdf_stn_test.nc",'r')
#    print ncdf_file.variables['lon'][5]
#    ncdf_file.variables['lon'][7] = 8.9
#    sys.exit()
#    
#    da = station_data_ncdf("/projects/daymet2/station_data/all_infill/nc_files/stn_infill_prcp.nc")
#    stns = da.load_stns()
#    ids = stns[STN_ID]
#    
#    ncdf_file = Dataset("/projects/daymet2/station_data/ncdf_stn_test.nc",'w')
#    #Set global attributes
##    title = "CONUS Statoin observations"
##    ncdf_file.title = title
##    ncdf_file.institution = " | ".join(["University of Montana Numerical Terradynamics Simulation Group",
##                                "NASA Ames Ecological Forecasting Lab"])
##    ncdf_file.source = "TopoMet v0.0.1"
##    ncdf_file.history = "".join(["Created on: ",datetime.strftime(date.today(),"%Y-%m-%d")]) 
##    ncdf_file.references = "http://www.ntsg.umt.edu/project/topomet"
##    ncdf_file.comment = "30-arcsec spatial resolution, daily timestep"
#    
#    #Create 2-dimensions
#    
#    days = utld.get_days_metadata()
#    
#    dim_time = ncdf_file.createDimension('time',size=days[DATE].size)
#    dim_station = ncdf_file.createDimension('station')
#    
#    
#    
#    #Create dimension variables and fill with values
#    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
#    times.long_name = "time"
#    times.units = "days since 1948-1-1 0:0:0"
#    times.standard_name = "time"
#    times.calendar = "standard"
#    times[:] = date2num(days[DATE],times.units)
#    
#    stations = ncdf_file.createVariable('station','str',('station',),fill_value=False)
#    stations.long_name = "station"
#    stations.standard_name = "station"
#    
#    for x in np.arange(ids.size):
#        
#        stations[x] = ids[x] 
#    
#    stations[45] = 'blah'
#    
#    latitudes = ncdf_file.createVariable('lat','f8',('station',),fill_value=False)
#    latitudes.long_name = "latitude"
#    latitudes.units = "degrees_north"
#    latitudes.standard_name = "latitude"
#    latitudes[:] = stns[LAT]
#    
#    longitudes = ncdf_file.createVariable('lon','f8',('station',),fill_value=False)
#    longitudes.long_name = "longitude"
#    longitudes.units = "degrees_east"
#    longitudes.standard_name = "longitude"
#    longitudes[:] = stns[LON]
    
     
    
#    latitudes[:] = lats
    
#    stations = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
#    times.long_name = "time"
#    times.units = "days since 1948-1-1 0:0:0"
#    times.standard_name = "time"
#    times.calendar = "standard"
#    times[:] = date2num(days[DATE],times.units)
#    
#    latitudes = ncdf_file.createVariable('lat','f8',('lat',),fill_value=False)
#    latitudes.long_name = "latitude"
#    latitudes.units = "degrees_north"
#    latitudes.standard_name = "latitude"
#    latitudes[:] = lats
#
#    longitudes = ncdf_file.createVariable('lon','f8',('lon',),fill_value=False)
#    longitudes.long_name = "longitude"
#    longitudes.units = "degrees_east"
#    longitudes.standard_name = "longitude"
#    longitudes[:] = lons
#    
#    print ncdf_file.variables['station'][0:5]
#    ncdf_file.createDimension('station', size=None)
#    names = ncdf_file.createVariable('station','str',('station',),fill_value=False)
#  
#    for x in np.arange(ids.size):
#        
#        names[x] = ids[x] 
    
#    ncdf_file.createDimension('station', size=None)
#    names = ncdf_file.createVariable('station','str',('station',),fill_value=False)
#    names[0] = 'blah'
#    names[1] = 'cookie'

def get_date_indices():
    
    ns_rast = ncdf_raster("/projects/daymet2/interp_output/fullrun20110916/topomet_tmin.nc",'time')
    col,row = ns_rast.getGridCellOffset(-113.304,48.395)
    ds_tmin = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_tmin.nc")
    ds_tmax = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_tmax.nc")
    ds_prcp = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_prcp.nc")
    ds_srad = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_srad.nc")
    ds_vpd = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_vpd.nc")
    
    tmin = ds_tmin.variables['tmin'][18262:,row,col]
    tmax = ds_tmax.variables['tmax'][18262:,row,col]
    prcp = ds_prcp.variables['prcp'][18262:,row,col]
    srad = ds_srad.variables['srad'][18262:,row,col]
    vpd = ds_vpd.variables['vpd'][18262:,row,col]
    
    tmean = (tmax + tmin)/2.0;
    tday = ((tmax - tmean)*0.45) + tmean;
    vps = 610.7 * np.exp(17.38 * tday/(239.0+tday));
    vpa = vps - vpd;
    rh = vpa/vps;
    
    print rh
    
    plt.plot(rh)
    plt.show()

    
#    ds = Dataset("/projects/daymet2/interp_output/fullrun20110916/topomet_tmin.nc",'r')
#    #print ds.variables['time'].units
#    dates = num2date(ds.variables['time'][:], units=ds.variables['time'].units, calendar='standard')
#    dates = utld.get_days_metadata(dates[0], dates[-1])
#    #print date2num(utld.ymdL_to_date(19500101), units=ds.variables['time'].units, calendar='standard')
#    idx_start = date2num(utld.ymdL_to_date(20000101), units=ds.variables['time'].units, calendar='standard')
#    idx_end = date2num(utld.ymdL_to_date(20091231), units=ds.variables['time'].units, calendar='standard')
#    print idx_end - idx_start + 1
#    print dates[YMD][dates[YMD] >= 20000101].size

def check_stn_locs():
    stn_da = station_data_ncdb('/projects/daymet2/station_data/crown_stns_final.nc')
    
    locs_fixed = load_locs_fixed("/projects/daymet2/station_data/all/qa_elev_fixed.csv")
    
    for stn in stn_da.stns:
        
        if locs_fixed.has_key(stn[STN_ID]):
            
            lon,lat,elev = locs_fixed[stn[STN_ID]]
            
            if np.round(stn[LON],4) != np.round(lon,4) or np.round(stn[LAT],4) != np.round(lat,4) or np.round(stn[ELEV],0) != np.round(elev,0):
                print stn

def glac_lapse_rate_stns():
    
    #w. glac, st. mary, babb1, babb2, 
    stns_lapse = np.array(['GHCN_USC00248809','GHCN_USW00004130','GHCN_USC00240389','GHCN_USC00240392',
                           'GLAC_1','GLAC_3','GLAC_4','GLAC_7','GLAC_21','GLAC_15','GLAC_5','GLAC_6'])
    
    stn_da = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    
    stns_lapse = stn_da.stns[np.in1d(stn_da.stn_ids, stns_lapse, assume_unique=True)]
    
    obs = stn_da.load_all_stn_obs(stns_lapse[STN_ID])
    
    print stns_lapse 
    
    day_mask = np.logical_and(stn_da.days[YMD] >= 19931001,stn_da.days[YMD] <= 20110930)
    
    tmin = obs[TMIN][day_mask,:]
    tmax = obs[TMAX][day_mask,:]
    
    ndays = float(np.nonzero(day_mask)[0].size)
    print ndays
    sys.exit()
    
    fout = open('/projects/daymet2/station_data/glac/stns_for_lapse.csv','w')
    fout.write(",".join(['NAME','ID','LAT','LON','ELEV','START_YMD','END_YMD','PCT_POR\n']))
    
    for x in np.arange(stns_lapse.size):
        
        valid_tair_mask = np.logical_and(np.isfinite(obs[TMIN][:,x]),np.isfinite(obs[TMAX][:,x]))    
        start_tair,end_tair = np.nonzero(valid_tair_mask)[0][[0,-1]]
        start_ymd,end_ymd = stn_da.days[YMD][start_tair],stn_da.days[YMD][end_tair]
        
        valid_tair_mask =np.logical_and(np.isfinite(tmin[:,x]),np.isfinite(tmax[:,x]))
        
        plt.clf()
        plt.plot(tmin[:,x])
        plt.plot(tmax[:,x])
        plt.title(stns_lapse[x][STN_NAME])
        plt.show()
        
        nobs = float(np.nonzero(valid_tair_mask)[0].size)
    
        fout.write(",".join([stns_lapse[x][STN_NAME],
                             stns_lapse[x][STN_ID],
                             "%.4f"%(stns_lapse[x][LAT],),
                             "%.4f"%(stns_lapse[x][LON],),
                             "%d"%(np.round(stns_lapse[x][ELEV]),),
                             str(start_ymd),
                             str(end_ymd),
                             "".join(["%.1f"%(nobs/ndays*100,),'%\n'])]))
    
    fout.close()  

def hare_snotels():
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    obs = db.load_all_stn_obs(np.array(['SNOTEL_10D12S']))
    
    plt.plot(obs[TMIN])
    plt.plot(obs[TMAX])
    plt.show()
    
    
    
def hare_closest_sites():
    
    lonlat_seeley = (-113.430344,47.230488,1405)
    lonlat_gardiner = (-110.587747,45.096985,2389)
    lonlat = lonlat_seeley
#    print get_elev_usgs(lonlat_seeley[0], lonlat_seeley[1])
#    print get_elev_usgs(lonlat_gardiner[0], lonlat_gardiner[1])
#    sys.exit()
    
    stn_da = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    dists = utlg.grt_circle_dist(lonlat[0], lonlat[1], stn_da.stns[LON], stn_da.stns[LAT])
    idx_sort = np.argsort(dists)
    
    for x in np.arange(20):
        print "|".join([str(dists[idx_sort[x]]),str(stn_da.stns[idx_sort[x]][ELEV]-lonlat[2]),str(stn_da.stns[idx_sort[x]])])
    

def test_ncdf_db():
    stn_da1 = station_data_ncdb('/projects/daymet2/station_data/[bak]crown_stns_final.nc')
    #stn_da2 = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    
    obs1 = stn_da1.load_all_stn_obs(np.array(['SNOTEL_06G02S']))
    
    print np.nonzero(obs1[TMIN_FLAG] != "")[0].size
    print np.nonzero(obs1[TMAX_FLAG] != "")[0].size
    print np.nonzero(obs1[PRCP_FLAG] != "")[0].size

#def get_ann_por_stns(path_csv_por,min_por_pct,days):
#    
#    #ttl_mth_days = np.zeros(12)
#    
#    min_obs = np.round(MIN_POR_PCT*days.size)
#    
##    x = 0
##    for mth in MONTHS:
##        ttl_mth_days[x] = np.nonzero(days[MONTH] == mth)[0].size
##        x+=1
##    print np.sum(ttl_mth_days)
##    print days[YMD].size
#    
#    cols = np.array([x[0] for x in POR_DTYPE])
#    cols_tmin = cols[6:18]
#    cols_tmax = cols[18:30]
#    cols_prcp = cols[30:42]  
#
#    por = load_por_csv(path_csv_por)
#
#    por_tmin = por[cols_tmin].view(np.int32).reshape((por.size,cols_tmin.size))
#    por_tmax = por[cols_tmax].view(np.int32).reshape((por.size,cols_tmax.size))
#    por_prcp = por[cols_prcp].view(np.int32).reshape((por.size,cols_prcp.size))
#    
#    mask_tmin = np.sum(por_tmin,axis=1)>=min_obs
#    mask_tmax = np.sum(por_tmax,axis=1)>=min_obs
#    mask_prcp = np.sum(por_prcp,axis=1)>=min_obs
#    
#    dem = input_raster('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif')
#    vals = dem.readEntireRaster()
#    vals = np.array(vals,dtype=np.float32)
#    vals[vals==255] = np.nan
#
#    uniq_rgns = np.unique(vals[np.isfinite(vals)])
#
#    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
#            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')
#    m.imshow(vals,origin='upper')
#    
#    lons = por[LON][np.logical_and(mask_tmax,mask_tmin)]
#    lats = por[LAT][np.logical_and(mask_tmax,mask_tmin)]
#    
#    rgns = np.zeros(lons.size)
#    for x in np.arange(lons.size):
#        try:
#            rgns[x] = dem.getDataValue(lons[x],lats[x])
#        except:
#            rgns[x] = np.nan
#    
#    for rgn in uniq_rgns:
#        print "".join([str(rgn),": ",str(rgns[rgns==rgn].size)])
#    
#    m.scatter(lons,lats)
#    plt.show()

def hcn_map():
     
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_da.set_day_mask(19480101,20111231)
    days = stn_da.days[stn_da.day_mask]
    
    #The min number of observations at station must have to be considered
    #for extrapolation testing
    min_obs = np.round(0.90*days.size)
    
    #Load the period-of-record datafile
    por = load_por_csv('/projects/daymet2/station_data/all/all_por.csv')
    
    #Number of obs in period-of-record for each month and variable
    por_cols = np.array([x[0] for x in POR_DTYPE])
    cols_tmin = por_cols[6:18]
    cols_tmax = por_cols[18:30]
    por_tmin = por[cols_tmin].view(np.int32).reshape((por.size,cols_tmin.size))
    por_tmax = por[cols_tmax].view(np.int32).reshape((por.size,cols_tmax.size))
    
    #Mask stations that have the min number of observations for both tmin and tmax
    mask_tmin = np.sum(por_tmin,axis=1)>=min_obs
    mask_tmax = np.sum(por_tmax,axis=1)>=min_obs
    mask_all = np.logical_and(mask_tmax,mask_tmin)
    
    #Load the neon ecoregion raster into memory
    neon_rast = input_raster('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif')
    neon = neon_rast.readEntireRaster()
    neon = np.array(neon,dtype=np.float32)
    neon[neon==255] = np.nan
    uniq_rgns = np.unique(neon[np.isfinite(neon)])

    #Extract lons, lats, and stn_ids that have min # of observations
    lons = por[LON][mask_all]
    lats = por[LAT][mask_all]
    stn_ids = por[STN_ID][mask_all]
    
    hcn_mask = np.zeros(stn_ids.size,dtype=np.bool)
    hcn_dict = get_hcn_dict()
    
    for x in np.arange(stn_ids.size):
    
        if hcn_dict[stn_ids[x]] == "HCN":
            hcn_mask[x] = True
    
    lons = lons[hcn_mask]
    lats = lats[hcn_mask]
    stn_ids = stn_ids[hcn_mask]
    
    #Determine neon region for each station
    rgns = np.zeros(lons.size)
    for x in np.arange(lons.size):
        try:
            rgns[x] = neon_rast.getDataValue(lons[x],lats[x])
        except:
            rgns[x] = np.nan
            
    for rgn in uniq_rgns:
        
        print "|".join([str(rgn),str(stn_ids[rgns==rgn].size)])
        
    m = Basemap(projection='cyl',llcrnrlat=neon_rast.min_y,urcrnrlat=neon_rast.max_y,\
                llcrnrlon=neon_rast.min_x,urcrnrlon=neon_rast.max_x,resolution='c')
    m.imshow(neon,origin='upper')
    
    m.scatter(lons,lats)
    
    plt.show()

def extrapolate_map_po():
    
    #ds = Dataset('/projects/daymet2/station_data/all/extrap_stats_wgtavg_po.nc')
    ds = Dataset('/projects/daymet2/station_data/all/extrap_stats_po.nc')
    stn_ids = np.array(ds.variables['stn_id'][:],dtype="S16")
    #ds.close()
    
    threshs = ds.variables['po_thresh']
    hss = ds.variables['hss'][:]
#    hss_mean = np.mean(hss,axis=1)
#    hss_max = np.max(hss_mean)
#    max_idx = np.nonzero(hss_mean==hss_max)[0][0]
    #hss_fnl = hss[max_idx,:]
    #print threshs[max_idx],max_idx
    hss_fnl = hss[20,:]
    #hss_fnl = hss
    print ds.variables['nnghs'][20]
    print hss_fnl[np.argsort(hss_fnl)][0:3]
    print stn_ids[np.argsort(hss_fnl)][0:3]
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stn_ids, assume_unique=True)]
    
    dem = input_raster('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif')
    vals = dem.readEntireRaster()
    vals = np.array(vals,dtype=np.float32)
    vals[vals==255] = np.nan

    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')
    m.imshow(vals,origin='upper')
    
    colors = hss_fnl
    
    sizes = hss_fnl/np.max(hss_fnl)
    
    m.scatter(stns[LON],stns[LAT],c=colors,s=sizes*100)
    m.colorbar()
    
    ds.close()
    
    plt.show()

def xval_prcp_map():
    
    ds = Dataset('/projects/daymet2/station_data/infill/xval_prcp.nc')
    stn_ids = np.array(ds.variables['stn_id'][:],dtype="S16")
    
    err = np.abs(ds.variables['err_freq'][0,:])
    #err = 1.0 - ds.variables['hss'][0,:]
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_prcp.nc','prcp')
    
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stn_ids, assume_unique=True)]
    
    dem = input_raster('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif')
    vals = dem.readEntireRaster()
    vals = np.array(vals,dtype=np.float32)
    vals[vals==255] = np.nan

    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')
    m.imshow(vals,origin='upper')
    
    colors = err
    
    sizes = err/np.max(err)
    
    m.scatter(stns[LON],stns[LAT],c=colors,s=sizes*100)
    m.colorbar()
    
    ds.close()
    
    plt.show()

def error_map():
    
    ds = Dataset('/projects/daymet2/station_data/infill/xval_tmin.nc')
    stn_ids = np.array(ds.variables['stn_id'][:],dtype="S16")
    
    err = np.abs(ds.variables['bias'][:])
    err_mean = np.mean(err,axis=1)
    min_idx = np.nonzero(np.min(err_mean)==err_mean)[0][0]
    err_fnl = err[min_idx,:]
    
    print ds.variables['min_nghs'][min_idx]
    print err_fnl[np.argsort(err_fnl)][-3:]
    print stn_ids[np.argsort(err_fnl)][-3:]
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_tmin.nc','tmin')
    
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stn_ids, assume_unique=True)]
    
    dem = input_raster('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif')
    vals = dem.readEntireRaster()
    vals = np.array(vals,dtype=np.float32)
    vals[vals==255] = np.nan

    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')
    m.imshow(vals,origin='upper')
    
    colors = err_fnl
    
    sizes = err_fnl/np.max(err_fnl)
    
    m.scatter(stns[LON],stns[LAT],c=colors,s=sizes*100)
    m.colorbar()
    
    ds.close()
    
    plt.show()
    
    
def interp_tair_map():
    
    ds = Dataset('/projects/daymet2/station_data/infill/xval_tmin.nc')
    stn_ids = np.array(ds.variables['stn_id'][:],dtype="S16")
    #ds.close()
    
    err_var = ds.variables['mae']
    min_nghs = ds.variables['min_nghs'][:]
    err_idx = 27
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stn_ids, assume_unique=True)]
    
    dem = input_raster('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif')
    vals = dem.readEntireRaster()
    vals = np.array(vals,dtype=np.float32)
    vals[vals==255] = np.nan

    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')
    m.imshow(vals,origin='upper')
    
    colors = np.abs(err_var[err_idx,:])
    
    sizes = np.abs(err_var[err_idx,:])/np.max(np.abs(err_var[err_idx,:]))
    
    m.scatter(stns[LON],stns[LAT],c=colors,s=sizes*100)
    m.colorbar()
    
    ds.close()
    
    plt.show()


def interp_tile_map():
    
    ds = Dataset('/projects/daymet2/station_data/infill/xval_tmin.nc')
    stn_ids = np.array(ds.variables['stn_id'][:],dtype="S16")
    
    err = np.abs(ds.variables['mae'][:])
    err_mean = np.mean(err,axis=1)
    min_idx = np.nonzero(np.min(err_mean)==err_mean)[0][0]
    err_fnl = err[min_idx,:]
    
    
    ds = Dataset('/projects/daymet2/interp_output/wxTopo_tests/h05v00_v1/h05v00_tmin.nc')
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_tmin.nc','tmin')
    ds_rast = nr.ncdf_raster('/projects/daymet2/interp_output/wxTopo_tests/h05v00_v1/h05v00_tmin.nc', 'tmin_mean')
    
    m = Basemap(projection='cyl',llcrnrlat=ds_rast.min_y,urcrnrlat=ds_rast.max_y,\
            llcrnrlon=ds_rast.min_x,urcrnrlon=ds_rast.max_x,resolution='c')
    m.imshow(ds.variables['tmin_cir'][:],origin='upper')
    m.colorbar()
    
    stns = stn_da.stns
    stns = stns[np.in1d(stns[STN_ID], stn_ids, assume_unique=True)]
    
    stn_mask = np.logical_and(np.logical_and(stns[LON]>=ds_rast.min_x,stns[LON]<=ds_rast.max_x),np.logical_and(stns[LAT]>=ds_rast.min_y,stns[LAT]<=ds_rast.max_y))
    
    print stn_mask.shape
    print err_fnl.shape
    
    stns = stns[stn_mask]
    err_fnl = err_fnl[stn_mask]
    
    colors = err_fnl
    sizes = err_fnl/np.max(err_fnl)
    m.scatter(stns[LON],stns[LAT],c=colors,s=sizes*100)
    
    #m.scatter(stn_da.stns[LON],stn_da.stns[LAT],color='yellow')
    plt.show()

def extrapolate_map():
    
    ds = Dataset('/projects/daymet2/station_data/all/extrap_stats_tair.nc')
    stn_ids = np.array(ds.variables['stn_id'][:],dtype="S16")
    #ds.close()
    
    err_var = ds.variables['bias']
    min_nghs = ds.variables['min_nghs'][:]
    tair_var = 1
    err_idx = 20
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stn_ids, assume_unique=True)]
    
    dem = input_raster('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif')
    vals = dem.readEntireRaster()
    vals = np.array(vals,dtype=np.float32)
    vals[vals==255] = np.nan

    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')
    m.imshow(vals,origin='upper')
    
    colors = np.abs(err_var[tair_var,err_idx,:])
    
    print stn_ids[np.abs(err_var[tair_var,err_idx,:])==np.max(np.abs(err_var[tair_var,err_idx,:]))]
    
    sizes = np.abs(err_var[tair_var,err_idx,:])/np.max(np.abs(err_var[tair_var,err_idx,:]))
    
    m.scatter(stns[LON],stns[LAT],c=colors,s=sizes*100)
    m.colorbar()
    
    ds.close()
    
    plt.show()

def infill_error_map():
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_tair/infill_tmin.nc')
    mae = ds.variables['bias'][:]
    err_mask = np.logical_or(mae > 0.4,mae < -0.4)#mae > 2.0
    stn_ids = ds.variables['stn_id'][:]
    lons = ds.variables['lon'][err_mask]
    lats = ds.variables['lat'][err_mask]
    print stn_ids[err_mask]
    print mae[err_mask]
    
    domain = input_raster('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif')
    
    m = Basemap(projection='cyl',llcrnrlat=domain.min_y,urcrnrlat=domain.max_y,\
            llcrnrlon=domain.min_x,urcrnrlon=domain.max_x,resolution='c')
    m.drawcountries()
    m.drawstates()
    m.drawcoastlines()
    
    m.scatter(lons,lats)
    
    plt.show()

def test_infill_outputs():
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_tmin.nc')
    nnghs = ds.variables['nnghs'][:]
    
    mask_cmplt = np.nonzero(np.logical_not(nnghs.mask))[0]
    
    flag_var = ds.variables['flag_fill']
    tmin_var = ds.variables['tmin']
    
    tmin = tmin_var[:,mask_cmplt]
    flag = np.array(flag_var[:,mask_cmplt],dtype=np.bool)
    stn_ids = ds.variables['stn_id'][mask_cmplt]
    nnghs = ds.variables['nnghs'][mask_cmplt]
    mae = ds.variables['mae'][mask_cmplt]
    bias = ds.variables['bias'][mask_cmplt]
    npcs = ds.variables['npcs'][mask_cmplt]
    max_dist = ds.variables['max_dist'][mask_cmplt]
    
    print np.ma.isMA(tmin),np.ma.isMA(flag_var[:,mask_cmplt]),np.ma.isMA(stn_ids),np.ma.isMA(nnghs),np.ma.isMA(mae),np.ma.isMA(bias),np.ma.isMA(npcs),np.ma.isMA(max_dist)
    
    print stn_ids[0],npcs[0],nnghs[0],mae[0],bias[0],max_dist[0]
    
    tmin1 = tmin[:,0]
    flag1 = flag[:,0]
    
    tmin_obs = np.zeros(tmin1.size)*np.nan
    tmin_obs[np.logical_not(flag1)] = tmin1[np.logical_not(flag1)]
    
    plt.plot(tmin1)
    plt.plot(tmin_obs)
    plt.show()
    
    

def test_infill():
    
    tair_var = 'tmax'
    stn_id = 'GHCN_CA001010720'
    nnghs = 16
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_da.set_day_mask(19480101,20111231)
    days = stn_da.days[stn_da.day_mask]
    
    ds_norms = Dataset('/projects/daymet2/station_data/all/normals_tair.nc')
    norms = ds_norms.variables['norm'][:]
#    norms_na_mask = norms.mask
#    norms = norms.data
#    norms[norms_na_mask] = np.nan
    
    norms_tmin = norms[0,:]
    norms_tmax = norms[1,:]
    ds_norms.close()
    
    if tair_var == 'tmin':
        norms = norms_tmin
    else:
        norms = norms_tmax
    
    pca = pca_matrix(stn_id, stn_da, tair_var, norms)
    fit_tair, obs_tair, npcs, nnghs,max_dist = pca.infill(nnghs)
    
    print npcs, nnghs,max_dist
    fin_mask = np.isfinite(obs_tair)
    difs = fit_tair[fin_mask] - obs_tair[fin_mask]
    mae = np.mean(np.abs(difs))
    bias = np.mean(difs)
    print mae,bias
    
    plt.plot(fit_tair)
    plt.plot(obs_tair)
    plt.show()
    
def test_infill_extrapolate_po():
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_da.set_day_mask(19480101,20111231)
    days = stn_da.days[stn_da.day_mask]
    
    n_yrs_mod = 5
    nmask = int(np.round(n_yrs_mod*365.25))
    idxs = np.arange(days.size)
    nnghs = 19
    stn_id = 'GHCN_USC00057337'#'GHCN_USC00422996'#'GHCN_USC00293265'#'GHCN_USC00248809'#'GHCN_USC00244241'#'GHCN_USC00211630'#'GHCN_USC00240392'#BABB    #'GHCN_USC00248809'#W.GLAC 
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    po_thres = 0.5
    
    ds_norms = Dataset('/projects/daymet2/station_data/infill/normals_po.nc')
    norms = ds_norms.variables['norm'][:]
    norms_na_mask = norms.mask
    norms = norms.data
    norms[norms_na_mask] = np.nan

    
    #Mask out data for validation
    obs_prcp = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), 'prcp')[0],dtype=np.float64)
    xval_obs_prcp = np.copy(obs_prcp)
    xval_obs_prcp[xval_obs_prcp > 0] = 1
#    
    fin_prcp = np.isfinite(obs_prcp)
    last_idxs = np.nonzero(fin_prcp)[0][-nmask:]
    xval_mask_prcp = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_prcp)
    
    #Estimate normal
#    fit_po,obs_po = infill_po(stn_id, stn_da, days, mth_masks, mthbuf_masks)
#    fin_mask = np.isfinite(obs_po)
#    print np.mean(fit_po[fin_mask]),np.mean(obs_po[fin_mask])
    
    a_pca_matrix = pca_matrix_po_norm(stn_id, stn_da, norms, prcp_mask=xval_mask_prcp)
    #a_pca_matrix = pca_matrix_po(stn_id, stn_da,prcp_mask=xval_mask_prcp)
    fit_po, obs_po, npcs, nnghs, max_dist = a_pca_matrix.infill(nnghs)
    
    
    fin_mask = np.isfinite(obs_po)
    po_mask = np.zeros(fit_po.size,dtype=np.bool)

    max_hss = 0
    max_thres = 0
    for thres in PO_THRESHS:
        
        thres_mask = np.array(fit_po >= thres,dtype=np.int)
        hss = calc_hss(obs_po, np.array(thres_mask,dtype=np.int))
    
        if hss > max_hss:
            max_hss = hss
            max_thres = thres
            po_mask = thres_mask
    
    po_mask = np.array(po_mask,dtype=np.bool)
        

#    for mth_mask in mth_masks:
#        
#        fin_mth_mask = np.logical_and(mth_mask,fin_mask)
#        obs_po_mth = obs_po[fin_mth_mask]
#        fit_po_mth = fit_po[fin_mth_mask]
#        
#        max_hss = 0
#        max_thres = 0
#        for thres in PO_THRESHS:
#            
#            thres_mask = fit_po_mth >= thres
#            po_mth = np.array(thres_mask,dtype=np.int)
#            
#            hss = calc_hss(obs_po_mth, po_mth)
#            if hss > max_hss:
#                max_hss = hss
#                max_thres = thres
#        
#        print "MONTH HSS/THRES",max_hss,max_thres
#        thres_mask = np.logical_and(mth_mask,fit_po >= max_thres)
#        po_mask[thres_mask] = True
        
    po_mask[fin_mask] = np.array(obs_po[fin_mask],dtype=np.bool)
    
    #print calc_hss(xval_obs_prcp[xval_mask_prcp], np.array(po_mask,dtype=np.int)[xval_mask_prcp])
    print calc_forecast_scores(xval_obs_prcp[xval_mask_prcp], np.array(po_mask,dtype=np.int)[xval_mask_prcp])
#    
#    po = np.zeros(fit_po.size)
#    for po_thres in np.arange(0.1,0.71,.01):
#        
#        thres_mask = fit_po >= po_thres
#        po[thres_mask] = 1
#        po[np.logical_not(thres_mask)] = 0
#        print po_thres,calc_forecast_scores(xval_obs_prcp[last_idxs],po[last_idxs])
        #print po_thres,calc_forecast_scores(xval_obs_prcp[xval_mask_prcp],po[xval_mask_prcp])


def add_fill_means():
    
    ds_tmin = Dataset('/projects/daymet2/station_data/infill/infill_tmin.nc','r+')
    ds_tmax = Dataset('/projects/daymet2/station_data/infill/infill_tmax.nc','r+')
    
    tmin = ds_tmin.variables['tmin'][:]
    tmin_mean = np.mean(tmin,axis=0,dtype=np.float64)
    tmin = None
    ds_tmin
    tmin_var = ds_tmin.createVariable('tmin_mean','f8',('stn_id',))
    tmin_var.long_name = "mean minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "mean_minimum_air_temperature"
    tmin_var.missing_value = netCDF4.default_fillvals['f4']
    tmin_var[:] = tmin_mean
    ds_tmin.close()
    
    tmax = ds_tmax.variables['tmax'][:]
    tmax_mean = np.mean(tmax,axis=0,dtype=np.float64)
    tmax = None
    
    tmax_var = ds_tmax.createVariable('tmax_mean','f8',('stn_id',))
    tmax_var.long_name = "mean maximum air temperature"
    tmax_var.units = "C"
    tmax_var.standard_name = "mean_maximum_air_temperature"
    tmax_var.missing_value = netCDF4.default_fillvals['f4']
    tmax_var[:] = tmax_mean
    ds_tmax.close()
    
    
def sort_tair_ds(ds,tair_var):

    stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
    idx_s = np.argsort(stn_ids)

    ds.variables['stn_id'][:] = stn_ids[idx_s]
    ds.sync()
    
    name = ds.variables['name'][:]
    ds.variables['name'][:] = name[idx_s]
    ds.sync()
    
    state = ds.variables['state'][:]
    ds.variables['state'][:] = state[idx_s]
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

def test_moments_extrapolate_prcp():
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_da.set_day_mask(19480101,20111231)
    days = stn_da.days[stn_da.day_mask]
    
    n_yrs_mod = 5
    nmask = int(np.round(n_yrs_mod*365.25))
    idxs = np.arange(days.size)
    nnghs = 27   #GHCN_USC00044259
    stn_id = 'GHCN_USC00248809'#'GHCN_USC00422996'#'GHCN_USC00293265'#'GHCN_USC00248809'#'GHCN_USC00244241'#'GHCN_USC00211630'#'GHCN_USC00240392'#BABB    #'GHCN_USC00248809'#W.GLAC 
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    
    ds = Dataset('/projects/daymet2/station_data/all/extrap_stats_wgtavg_po.nc')
    stn_ids = np.array(ds.variables['stn_id'][:],dtype="<S16")
    ds.close()
    
    po_mask = np.ones(days.size,dtype=np.bool)
    
    fit_errs = []
    train_errs = []
    for stn_id in stn_ids:
    
        obs_prcp = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), 'prcp')[0],dtype=np.float64)
        
        #Mask out data for validation
        fin_prcp = np.isfinite(obs_prcp)
        train_idxs = np.nonzero(fin_prcp)[0][-nmask:]
        validate_mask = np.logical_and(np.logical_not(np.in1d(idxs,train_idxs,assume_unique=True)),fin_prcp)
    
        fit_prcp = infill_prcp_norm(stn_id, stn_da, days, mth_masks, mthbuf_masks, prcp_mask=validate_mask)[0]
        
        mean_fit = np.mean(fit_prcp[validate_mask])
        mean_obs = np.mean(obs_prcp[validate_mask])
        mean_train = np.mean(obs_prcp[train_idxs])
        
        prr_fit = ((mean_fit-mean_obs)/mean_obs)*100.
        prr_train = ((mean_fit-mean_train)/mean_obs)*100.
        
        fit_errs.append(prr_fit)
        train_errs.append(prr_train)
        
        print stn_id,"%.2f"%(prr_fit,),"%.2f"%(prr_train,)
        #print stn_id,np.mean(fit_prcp[validate_mask]),np.mean(obs_prcp[validate_mask]),np.mean(obs_prcp[train_idxs])
    
    print "FIT MAE",np.mean(np.abs(fit_errs))
    print "TRAIN MAE",np.mean(np.abs(train_errs))
    print "FIT BIAS",np.mean(fit_errs)
    print "TRAIN BIAS",np.mean(train_errs)

def test_infill_prcp_with_stats():
    
    stn_id = 'GHCN_CA006151042' #GHCN_USC00248809
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc",(None,None))
    days = stn_da.days
    
    nngh_prcp = 33 
    ndays = float(days.size)
    mth_masks = build_mth_masks(days)
    
    empty_rslts = prcp_infill_results(None)
    empty_rslts.init_empty(ndays)
    
    ds_norms_po = Dataset('/projects/daymet2/station_data/infill/normals_po.nc')
    ds_norms = Dataset('/projects/daymet2/station_data/infill/normals_prcp.nc')
    
    norms_po = ds_norms_po.variables['norm'][:]
    norms_na_mask = norms_po.mask
    norms_po = norms_po.data
    norms_po[norms_na_mask] = np.nan
    
    obs_prcp = stn_da.load_all_stn_obs_var(stn_id, 'prcp')[0]
    fin_mask = np.isfinite(obs_prcp)
    
    atimer = timer()
    atimer.start()
    #infill_rslts = infill_prcp_with_stats(stn_id, stn_da, norms_po, ds_norms, days, mth_masks, nngh_prcp)
    infill_rslts = infill_prcp_with_stats(stn_id, stn_da,ds_norms, days, mth_masks, nngh_prcp)
    atimer.stop("INFILL TOOK")
    
    print np.sum(infill_rslts.prcp[fin_mask]-obs_prcp[fin_mask])
    print "|".join(["WRITER",infill_rslts.stn_id,"%.4f"%(infill_rslts.hss,),"%.2f"%(infill_rslts.perr_ttlamt,),
        "%.2f"%(infill_rslts.perr_intsy,),"%.2f"%(infill_rslts.perr_freq,)])

    plt.plot(infill_rslts.prcp)
    plt.plot(obs_prcp)
    plt.show()
    

def test_infill_extrapolate_prcp():
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    days = stn_da.days
    
    n_yrs_mod = 5
    nmask = int(np.round(n_yrs_mod*365.25))
    idxs = np.arange(days.size)
    nnghs = 34
    stn_id = 'GHCN_USC00027281'
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,15)
    
    ds_norms = Dataset('/projects/daymet2/station_data/infill/normals_prcp.nc')
    
    #Observed values
    obs_prcp = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), 'prcp')[0],dtype=np.float64)
    obs_po = np.copy(obs_prcp)
    obs_po[obs_po > 0] = 1
    
    fin_prcp = np.isfinite(obs_prcp)
    train_idxs = np.nonzero(fin_prcp)[0][-nmask:]
    validate_mask = np.logical_and(np.logical_not(np.in1d(idxs,train_idxs,assume_unique=True)),fin_prcp)
    
    prcp_norm = infill_prcp_norm(stn_id, stn_da, days, mth_masks, mthbuf_masks, use_prcp_only=False,prcp_mask=validate_mask)[0]
    
    #validate_mask = np.logical_and(validate_mask,obs_prcp > 0)
    
    print "Norm Percent Error: ",((np.sum(prcp_norm[validate_mask])-np.sum(obs_prcp[validate_mask]))/np.sum(obs_prcp[validate_mask]))*100.
    
    prcp_norm[train_idxs] = obs_prcp[train_idxs]
    
    fit_prcp = infill_prcp(stn_id, stn_da, ds_norms, days, mth_masks, nnghs, validate_mask, prcp_norm)
    fit_po = np.copy(fit_prcp)
    fit_po[fit_po > 0] = 1
    
    obs_prcp_validate = obs_prcp[validate_mask]
    fit_prcp_validate = fit_prcp[validate_mask]
    obs_po_validate = obs_po[validate_mask]
    fit_po_validate = fit_po[validate_mask]
    
    #HSS
    hss = calc_forecast_scores(obs_po_validate, fit_po_validate)[5]
    
    #Total amount cm
    perr_ttlamt = ((np.sum(fit_prcp_validate)-np.sum(obs_prcp_validate))/np.sum(obs_prcp_validate))*100.
    
    #Frequency wd d-1
    freq_obs = np.nonzero(obs_prcp_validate > 0)[0].size/np.float(obs_prcp_validate.size)
    freq_fit = np.nonzero(fit_prcp_validate > 0)[0].size/np.float(fit_prcp_validate.size)
    perr_freq = ((freq_fit-freq_obs)/freq_obs)*100.

    #Intensity cm wd-1
    intsy_obs = np.sum(obs_prcp_validate[obs_prcp_validate  > 0])/obs_prcp_validate[obs_prcp_validate  > 0].size
    intsy_fit = np.sum(fit_prcp_validate[fit_prcp_validate  > 0])/fit_prcp_validate[fit_prcp_validate  > 0].size
    perr_intsy = ((intsy_fit-intsy_obs)/intsy_obs)*100.
    
    #Amount cm d-1
    amt_obs = np.sum(obs_prcp_validate[obs_prcp_validate  > 0])/np.float(obs_prcp_validate.size)
    amt_fit = np.sum(fit_prcp_validate[fit_prcp_validate  > 0])/np.float(fit_prcp_validate.size)
    perr_amt = ((amt_fit-amt_obs)/amt_obs)*100.
    
    print "|".join(["WRITER",stn_id,'%.4f'%(hss,),str(perr_ttlamt),'%.2f'%(perr_freq,),'%.2f'%(perr_intsy,),'%.2f'%(perr_amt,)])
    sys.exit()
#    fit_prcp = infill_prcp2(stn_id, stn_da,ds_norms, days,mth_masks, nnghs, prcp_mask=validate_mask,prcp_norm=prcp_norm)
#    #fit_prcp = infill_prcp_by_mth(stn_id, stn_da, days, mth_masks, mthbuf_masks, nnghs, prcp_mask=validate_mask)
#    fit_po = np.copy(fit_prcp)
#    fit_po[fit_po > 0] = 1
#    
#    #validate_mask = np.in1d(idxs,train_idxs,assume_unique=True)
#    
#    print calc_forecast_scores(obs_po[validate_mask], fit_po[validate_mask])
#    
#    
#    
##    a_pca_matrix = pca_matrix_po(stn_id, stn_da,prcp_mask=validate_mask)
##    fit_po = a_pca_matrix.infill(nnghs)[0]
###    
##    fit_po_fnl = np.zeros(fit_po.size)
##    
##    max_hss = 0
##    max_thres = 0
##    
##    for thres in np.arange(0.1,0.71,.01):
##        
##        thres_mask = fit_po >= thres
##        fit_po_fnl[thres_mask] = 1
##        fit_po_fnl[np.logical_not(thres_mask)] = 0
##        
##        hss = calc_hss(obs_po[train_idxs], fit_po_fnl[train_idxs])
##        if hss > max_hss:
##            max_hss = hss
##            max_thres = thres
##    
##    print max_hss,max_thres
##    thres_mask = fit_po >= max_thres
##    fit_po_fnl[thres_mask] = 1
##    fit_po_fnl[np.logical_not(thres_mask)] = 0
##    fit_po_fnl[train_idxs] = obs_po[train_idxs]
##    
##    print calc_forecast_scores(obs_po[validate_mask], fit_po_fnl[validate_mask])
##    fit_po_bool = np.array(fit_po_fnl,dtype=np.bool)
#    #Estimate normal
#    #fit_po,obs_po = infill_po(stn_id, stn_da, days, mth_masks, mthbuf_masks, prcp_mask=xval_mask_prcp)
#    
##    a_pca_matrix = pca_matrix_po(stn_id, stn_da,prcp_mask=xval_mask_prcp)
##    fit_po = a_pca_matrix.infill(nnghs)[0]
###    
##
##    po = np.zeros(fit_po.size)
##    
##    max_hss = 0
##    max_thres = 0
##    
##    for thres in np.arange(0.1,0.71,.01):
##        
##        thres_mask = fit_po >= thres
##        po[thres_mask] = 1
##        po[np.logical_not(thres_mask)] = 0
##        
##        hss = calc_hss(obs_po[last_idxs], po[last_idxs])
##        if hss > max_hss:
##            max_hss = hss
##            max_thres = thres
##    
##    print max_hss,max_thres
##    thres_mask = fit_po >= max_thres
##    po[thres_mask] = 1
##    po[np.logical_not(thres_mask)] = 0
##    po[last_idxs] = obs_po[last_idxs]
##    
##    print calc_forecast_scores(obs_po[xval_mask_prcp], po[xval_mask_prcp])
##    po = np.array(po,dtype=np.bool)
##    a_pca_matrix = pca_matrix_prcp(stn_id, stn_da, fit_po_bool,prcp_mask=validate_mask)
##    fit_prcp = a_pca_matrix.infill(nnghs)[0]
##    fit_prcp[fit_prcp < 0] = 0
##    
##    fnl_fit_prcp = np.zeros(fit_po_bool.size)
##    fnl_fit_prcp[fit_po_bool] = fit_prcp
##    
##    print "# of prcp occur days  = 0 prcp amt", np.nonzero(np.logical_and(fit_po_bool,fnl_fit_prcp==0))[0].size
#    #print "# of days no prcp amount, but prcp occur ",np.nonzero(np.logical_and(fit_prcp <= 0,po))[0].size
#    
#    #fit_prcp[np.logical_not(po)] = 0
#    
#    # print "# of days negative prcp amount, but prcp occur ",np.nonzero(fit_prcp < 0)[0].size
#    
#    # fit_prcp[fit_prcp < 0] = 0
#    
#    print "min/max: ",np.min(fit_prcp),np.max(fit_prcp)
#    
#    print "total obs prcp: ",np.sum(obs_prcp[validate_mask])
#    print "total mod prcp: ",np.sum(fit_prcp[validate_mask])
#    print np.mean(fit_prcp[validate_mask]-obs_prcp[validate_mask])
    print "Overall % error: ",((np.sum(fit_prcp[validate_mask])-np.sum(obs_prcp[validate_mask]))/np.sum(obs_prcp[validate_mask]))*100.
#    
    ls_freq_mth_obs = []
    ls_freq_mth_fit = []
    
    ls_intsy_mth_obs = []
    ls_intsy_mth_fit = []
    
    ls_amt_mth_obs = []
    ls_amt_mth_fit = []
    
    mths = np.arange(1,13)
    for mth_mask in mth_masks:
        
        mth_validate_mask = np.logical_and(validate_mask,mth_mask)
        mth_prcp_obs = obs_prcp[mth_validate_mask]
        mth_prcp_fit = fit_prcp[mth_validate_mask]
        
        #Frequency wd d-1
        freq_mth_obs = np.nonzero(mth_prcp_obs > 0)[0].size/np.float(mth_prcp_obs.size)
        freq_mth_fit = np.nonzero(mth_prcp_fit > 0)[0].size/np.float(mth_prcp_fit.size)
        ls_freq_mth_obs.append(freq_mth_obs)
        ls_freq_mth_fit.append(freq_mth_fit)
        
        #Intensity cm wd-1
        intsy_mth_obs = np.sum(mth_prcp_obs[mth_prcp_obs  > 0])/mth_prcp_obs[mth_prcp_obs  > 0].size
        intsy_mth_fit = np.sum(mth_prcp_fit[mth_prcp_fit  > 0])/mth_prcp_fit[mth_prcp_fit  > 0].size
        ls_intsy_mth_obs.append(intsy_mth_obs)
        ls_intsy_mth_fit.append(intsy_mth_fit)
        
        #Amount cm d-1
        amt_mth_obs = np.sum(mth_prcp_obs[mth_prcp_obs  > 0])/mth_prcp_obs.size
        amt_mth_fit = np.sum(mth_prcp_fit[mth_prcp_fit  > 0])/mth_prcp_fit.size
        ls_amt_mth_obs.append(amt_mth_obs)
        ls_amt_mth_fit.append(amt_mth_fit)
    
    mthyr_obs_ls = []
    mthyr_fit_ls = []
    for yr in np.unique(days[YEAR]):
        
        #for mth_mask in mth_masks:
        
            #mthyr_validate_mask = np.logical_and(np.logical_and(validate_mask,mth_mask),days[YEAR]==yr)
        mthyr_validate_mask = np.logical_and(validate_mask,days[YEAR]==yr)
        mthyr_prcp_obs = np.sum(obs_prcp[mthyr_validate_mask])
        mthyr_prcp_fit = np.sum(fit_prcp[mthyr_validate_mask])
        mthyr_obs_ls.append(mthyr_prcp_obs)
        mthyr_fit_ls.append(mthyr_prcp_fit)
            
    plt.subplot(221)
    plt.title('Freq')
    plt.plot(mths,np.array(ls_freq_mth_obs))
    plt.plot(mths,ls_freq_mth_fit)
    
    plt.subplot(222)
    plt.title('Intensity')
    plt.plot(mths,ls_intsy_mth_obs)
    plt.plot(mths,ls_intsy_mth_fit)
    
    plt.subplot(223)
    plt.title('Amt')
    plt.plot(mths,ls_amt_mth_obs)
    plt.plot(mths,ls_amt_mth_fit)
    
    plt.subplot(224)
    plt.title('Mth Time Series')
    #plt.plot(np.array(mthyr_fit_ls)-np.array(mthyr_obs_ls))
    plt.plot(mthyr_obs_ls)
    plt.plot(mthyr_fit_ls)
    
    plt.show()

def test_infill_extrapolate_new():
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_da.set_day_mask(19480101,20111231)
    days = stn_da.days[stn_da.day_mask]
    
    ds_norms = Dataset('/projects/daymet2/station_data/all/normals_tair.nc')
    norms = ds_norms.variables['norm'][:]
    norms_na_mask = norms.mask
    norms = norms.data
    norms[norms_na_mask] = np.nan
    
    norms_tmin = norms[0,:]
    norms_tmax = norms[1,:]
    ds_norms.close()
    
    tair_var = 'tmin'
    if tair_var == 'tmin':
        norms = norms_tmin
    else:
        norms = norms_tmax
    n_yrs_mod = 5
    nmask = int(np.round(n_yrs_mod*365.25))
    idxs = np.arange(days.size)
    nnghs = 20
    stn_id = 'GHCN_CA007090960'#'GHCN_USC00293265'#'GHCN_USC00248809'#'GHCN_USC00244241'#'GHCN_USC00211630'#'GHCN_USC00240392'#BABB    #'GHCN_USC00248809'#W.GLAC 
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    
    #Mask out data for validation
    obs_tair = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0],dtype=np.float64)
    fin_tair = np.isfinite(obs_tair)
    last_idxs = np.nonzero(fin_tair)[0][-nmask:]
    xval_mask_tair = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_tair)
    
    plt.plot(obs_tair)
    temp_tair = np.copy(obs_tair)
    temp_tair[xval_mask_tair] = np.nan
    plt.plot(temp_tair)
    plt.show()
    
    #Estimate normal
    fit_tair,masked_tair = infill_tair(stn_id, stn_da, days, tair_var, mth_masks, mthbuf_masks,xval_mask_tair)
    mask_nan = np.isnan(masked_tair)
    fnl_tair = np.copy(masked_tair)
    fnl_tair[mask_nan] = fit_tair[mask_nan]
    norm_est = np.mean(fnl_tair)
    print "|".join([str(norms[stn_da.stn_ids==stn_id]),str(norm_est)])
    norms[stn_da.stn_ids==stn_id] = norm_est
    
    
    pca = pca_matrix(stn_id, stn_da, tair_var, norms,tair_mask=xval_mask_tair)
    
    fit_tair = pca.infill(nnghs)[0]
    
    plt.plot(fit_tair)
    plt.show()
    
    xval_fit = fit_tair[xval_mask_tair]
    xval_obs = obs_tair[xval_mask_tair]
    
    mae = np.mean(np.abs(xval_fit-xval_obs))
    bias = np.mean(xval_fit-xval_obs)
    press = np.sum(np.square(xval_fit-xval_obs))
    
    print "|".join([str(mae),str(bias),str(press)])

def test_infill_extrapolate():
    
    r.source("/home/jared.oyler/ecl_helios_workspace/daymet2/pca_infill.R")
    stn_id = 'GHCN_USC00211630'#'GHCN_USC00240392'#BABB    #'GHCN_USC00248809'#W.GLAC                 
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_da.set_day_mask(19480101,20111231)
    days = stn_da.days[stn_da.day_mask]
    
    pct_nan = 0.90
    idx_nan = np.arange(int(np.round(days.size*pct_nan)))
    idx_all = np.arange(days.size)
    
#    nran_nan = int(np.round(0.30*days.size))
#    ran_idx = np.random.randint(0,days.size,nran_nan)
    
    obs_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]),'tmin')[0]
    #obs_tair[ran_idx] = np.nan
    
    day_nan_mask = np.logical_and(np.isfinite(obs_tair),np.in1d(idx_all, idx_nan, assume_unique=True))
    
    pca_tair,valid_pca_tair_mask,ngh_dists = build_pca_matrices_tair(stn_id,stn_da,'tmin',day_nan_mask)
    
    min_nghs = np.arange(15,16)
    
    for min_ngh in min_nghs:
        
        print "########################################"
        print "Min Ngh: "+str(min_ngh)
        
        pcat_tair = trim_pca_matrices(min_ngh, pca_tair, valid_pca_tair_mask)
    
        nstns = pcat_tair.shape[1]
        print "|".join(["min/max ngh dists",str(np.min(ngh_dists[1:nstns])),str(np.max(ngh_dists[0:nstns]))])
    
        ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(pcat_tair))
    
        fit_tair = np.array(ppca_rslt.rx('ppca_fit'))
        fit_tair.shape = (fit_tair.shape[1],)
        
        difs = fit_tair[day_nan_mask] - obs_tair[day_nan_mask]
        difs = difs[np.isfinite(difs)]
        print np.mean(np.abs(difs)),np.mean(difs)
        
        plt.plot(fit_tair-obs_tair)
        #plt.plot(obs_tair[0:500])
        #plt.plot(fit_tair[0:500])
        plt.show()
        print "########################################"


def test_po_infill():
    
        r.source("/home/jared.oyler/ecl_helios_workspace/daymet2/pca_infill.R")
        stn_id = 'GHCN_USC00248809'                 
    
        stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
        stn_da.set_day_mask(19480101,20111231)
        
        days = stn_da.days[stn_da.day_mask]
        
        nan_mask = days[YMD] <= 20050101
        
        print "loading data..."
        pca_prcp,valid_pca_prcp_mask,ngh_dists = build_pca_matrices_tair(stn_id, stn_da,'prcp',nan_mask)
        
        pca_po = np.copy(pca_prcp)
        pca_po[np.logical_and(valid_pca_prcp_mask,pca_po > 0)] = 1
        pca_po[np.logical_and(valid_pca_prcp_mask,pca_po == 0)] = 0
        
        pcat_po = trim_pca_matrices(15,pca_po,valid_pca_prcp_mask)
        
        print "infilling..."
        ppca_rslt = r.ppca_tair_no_xval(robjects.Matrix(pcat_po))
    
        fit_po = np.array(ppca_rslt.rx('ppca_fit'))
        fit_po.shape = (fit_po.shape[1],)
        
        for x in np.arange(0.4,0.61,.01):
        
            fnl_po = np.zeros(fit_po.size)
            fnl_po[fit_po >= x] = 1
            
            obs_po = stn_da.load_all_stn_obs_var(np.array([stn_id]),'prcp')[0]
            obs_po[obs_po > 0] = 1
            obs_po[obs_po == 0] = 0
            
            mask = np.logical_and(nan_mask,np.isfinite(obs_po))
            
            print str(x)+"|"+str(calc_hss(obs_po[mask], fnl_po[mask]))

def analyze_infill_po():
    ds = Dataset('/projects/daymet2/station_data/all/extrap_stats_pca_po.nc')
    #('nnghs','stn_id')
    nnghs = ds.variables['nnghs'][:] 
    po_threshs = ds.variables['po_thresh'][:]
    stat = ds.variables['hss'][:]
    
    max_score = 0
    max_nnghs = None
    max_thresh = None
    
    for x in np.arange(nnghs.size):
        
        for i in np.arange(po_threshs.size):
            
            score = np.mean(stat[x,i,:])
            #print nnghs[x],po_threshs[i],score
            if score > max_score:
                max_score = score
                max_nnghs = nnghs[x],x
                max_thresh = po_threshs[i],i
    
    print max_nnghs,max_thresh,max_score
    
            
            

def analyze_infill_error():
    
    ds = Dataset('/projects/daymet2/station_data/all/extrap_stats4.nc')
    min_nghs = ds.variables['min_nghs'][:]
    
    mae_var = ds.variables['mae']
    bias_var = ds.variables['bias']
    
    mae_tmin = mae_var[0,:,:]
    mae_tmax = mae_var[1,:,:]
    
    bias_tmin = bias_var[0,:,:]
    bias_tmax = bias_var[1,:,:]
    
#    for x in np.arange(min_nghs.size):
#        
#        print "|".join([str(min_nghs[x]),str(mae_tmin[x])])
#        
#    sys.exit()
    print "###########################################"
    print "TMIN"
    print "###########################################"
    for x in np.arange(min_nghs.size):
        
        mae_tmin_min_ngh = mae_tmin[x,:]
        bias_tmin_min_ngh = bias_tmin[x,:]
        
        try:
            nruns = np.nonzero(np.logical_not(mae_tmin_min_ngh.mask))[0].size
        except AttributeError:
            nruns = mae_tmin_min_ngh.size
        
        print "|".join([str(min_nghs[x]),str(np.mean(mae_tmin_min_ngh)),str(np.mean(bias_tmin_min_ngh)),str(np.max(mae_tmin_min_ngh)),str(np.max(np.abs(bias_tmin_min_ngh))),str(nruns)])
        
    print "###########################################"
    print "TMAX"
    print "###########################################"
    for x in np.arange(min_nghs.size):
        
        mae_tmax_min_ngh = mae_tmax[x,:]
        bias_tmax_min_ngh = bias_tmax[x,:]
        
        try:
            nruns = np.nonzero(np.logical_not(mae_tmax_min_ngh.mask))[0].size
        except AttributeError:
            nruns = mae_tmax_min_ngh.size
        
        print "|".join([str(min_nghs[x]),str(np.mean(mae_tmax_min_ngh)),str(np.mean(bias_tmax_min_ngh)),str(np.max(mae_tmax_min_ngh)),str(np.max(np.abs(bias_tmax_min_ngh))),str(nruns)])
        
    sys.exit()
    
    for x in np.arange(mae_tmin.shape[1]):
        
        try:
            mae_stn = mae_tmin[:,x]
            nruns = np.nonzero(np.logical_not(mae_stn.mask))[0].size
            min_mae = np.min(mae_tmin[:,x])
            min_mae_ngh = min_nghs[min_mae==mae_tmin[:,x]][0]
            print "|".join([str(min_mae_ngh),str(min_mae),str(nruns)])
        except:
            continue

def center_dataset(ds,var_name):
    
    vals = np.require(ds.variables[var_name][:], dtype=np.float64)
    mean_vals = np.require(ds.variables["".join([var_name,"_mean"])][:], dtype=np.float64)
    vals_cent = vals - mean_vals
    vals = None
    
    ds.variables[var_name][:] = vals_cent
    ds.sync()

def test_sngl_tair_interp():
    stn_db = station_data_infill('/projects/daymet2/station_data/infill/infill_tmin_center.nc','tmin')
    stn_db_xval = station_data_infill('/projects/daymet2/station_data/infill/infill_tmin.nc','tmin')
    
    
    interper = interp_tair()
    
    stn = stn_db.stns[stn_db.stn_idxs['GHCN_USC00435416']]
    lat =  stn[LAT]
    lon = stn[LON] 
    elev =  stn[ELEV]
    
    pt = {LAT:lat,LON:lon,ELEV:elev}
    
    stn_slct = station_select(stn_db.stns,30,40)
    ngh_stns, wgts, radius = stn_slct.get_interp_stns(lat, lon,np.array([stn[STN_ID]]))
    ngh_obs = stn_db.load_obs(ngh_stns[STN_ID])

    obs_vals = stn_db_xval.load_obs(stn[STN_ID])

    interp_vals, se, ci = interper.model_tair(ngh_stns, wgts, ngh_obs,pt,True)
    print se,ci
    print np.mean(np.abs(interp_vals-obs_vals))
    print np.mean(interp_vals),np.mean(obs_vals)
    
    #plt.plot(interp_vals)
    #plt.show()

def find_optim_po_nghs():
    ds = Dataset('/projects/daymet2/station_data/infill/xval_po.nc')
    hss = ds.variables['hss'][:]
    hssm = np.mean(hss,axis=1)
    min_nghs = ds.variables['min_nghs'][:]
    
    for x in np.arange(hssm.size):
        
        pct_chg = ((hssm[x+10]-hssm[x])/hssm[x])*100
        if pct_chg < 0.5:
            print pct_chg
            break
    
    
    print hssm[x],min_nghs[x]
    
def hare_site_interp():
    
    #Seeley Lake Hare Site
#    lon = -113.430344
#    lat = 47.230488
#    elev = 1405
    
    #Gardiner Hare Site
    lon=-110.587747
    lat= 45.096985
    elev= 2389
    
    aoutput = ip.quick_interp_prcp(lon, lat, elev)
    plt.plot(aoutput.prcp)
    plt.show()
    
    aoutput = quick_interp_tair(lon, lat, elev)
    plt.plot(aoutput.tmin)
    plt.plot(aoutput.tmax)
    plt.show()

def test_tair_interp_xval():
    
    modR = modeler_R()
    modC = modeler_clib()
    
    interptR = interp_tair(modR)
    interptC = interp_tair(modC)
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_tmin_center.nc','tmin')
    stn_da_xval = station_data_infill('/projects/daymet2/station_data/infill/infill_tmin.nc','tmin')
    
    stn_id = 'GHCN_USC00101180'
    min_nghs=37

    stn_slct = station_select(stn_da.stns,min_nghs,min_nghs+10)

    xval_stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
    xval_obs = stn_da_xval.load_obs(stn_id)

    dist = utlg.grt_circle_dist(xval_stn[LON], xval_stn[LAT], stn_da.stns[LON], stn_da.stns[LAT])
    rm_stn_ids = np.unique(np.concatenate((np.array([stn_id]),stn_da.stn_ids[dist==0])))

    stns,wgts,rad = stn_slct.get_interp_stns(xval_stn[LAT], xval_stn[LON], rm_stn_ids)
    
    stn_obs = stn_da.load_obs(stns[STN_ID])
    
    ##C implementation
    interp_vals, se, mean_val, ci = interptC.model_tair(stns, wgts, stn_obs, xval_stn)
    err = interp_vals - xval_obs
    mae = np.mean(np.abs(np.abs(err)))
    bias = np.mean(err)
    print mae,bias,ci,np.mean(interp_vals),np.mean(xval_obs)
    
    plt.plot(xval_obs)
    plt.plot(interp_vals)
    plt.show()
    
    ##R implementation
    interp_vals, se, mean_val, ci = interptR.model_tair(stns, wgts, stn_obs, xval_stn)
    err = interp_vals - xval_obs
    mae = np.mean(np.abs(np.abs(err)))
    bias = np.mean(err)
    print mae,bias,ci,np.mean(interp_vals),np.mean(xval_obs)


def test_po_mth_thres_interp():
    
    cmod = ip.modeler_po_mth_thres_clib()
    po_interper = ip.interp_po_mth_thres(cmod)
    
    cmod1 = ip.modeler_clib()
    po_interper1 = ip.interp_po(cmod1)
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_prcp.nc','prcp')
    mths = stn_da.days[MONTH]
    
    stn_id = 'GHCN_USC00247448'
    min_nghs = 51
    thres = 0.5
    
    ds_thres = Dataset('/projects/daymet2/station_data/infill/po_thres.nc')
    stn_thres = ds_thres.variables['po_thres'][:]
    
    stn_slct = station_select(stn_da.stns,min_nghs,min_nghs+10)

    xval_stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
    xval_obs = stn_da.load_obs(stn_id)
    xval_obs[xval_obs > 0] = 1

    dist = utlg.grt_circle_dist(xval_stn[LON], xval_stn[LAT], stn_da.stns[LON], stn_da.stns[LAT])
    rm_stn_ids = np.unique(np.concatenate((np.array([stn_id]),stn_da.stn_ids[dist==0])))

    stns,wgts,rad = stn_slct.get_interp_stns(xval_stn[LAT], xval_stn[LON], rm_stn_ids)
    
    stn_obs = stn_da.load_obs(stns[STN_ID])
    stn_obs[stn_obs > 0] = 1
    
    ngh_thres = stn_thres[:,np.in1d(stn_da.stn_ids, stns[STN_ID], assume_unique=True)]
    
    po = po_interper.model_po(stns, wgts, stn_obs, ngh_thres, mths, xval_stn)
    po1 = po_interper1.model_po(stns, wgts, stn_obs, xval_stn, thres)
    
    hss = calc_hss(xval_obs, po)
    hss1 = calc_hss(xval_obs, po1)
    print hss,hss1,hss-hss1

def test_prcp_interp():
    
    ainput = ip.prcp_input()
    aoutput = ip.prcp_output()
    cmod = ip.modeler_clib_prcp()
    interper = ip.interp_prcp(cmod)
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_prcp.nc','prcp')
    
    stn_id = 'GHCN_US1MOFSA207'
    min_nghs = 51
    thres = 0.5

    stn_slct = station_select(stn_da.stns,min_nghs,min_nghs+10)

    xval_stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
    xval_obs = stn_da.load_obs(stn_id)
    xval_obs_po = np.copy(xval_obs)
    xval_obs_po[xval_obs_po > 0] = 1

    dist = utlg.grt_circle_dist(xval_stn[LON], xval_stn[LAT], stn_da.stns[LON], stn_da.stns[LAT])
    rm_stn_ids = np.unique(np.concatenate((np.array([stn_id]),stn_da.stn_ids[dist==0])))

    stns,wgts,rad = stn_slct.get_interp_stns(xval_stn[LAT], xval_stn[LON], rm_stn_ids)
    
    stn_obs = stn_da.load_obs(stns[STN_ID])
    atimer = timer()
    atimer.start()
    
    ainput.stns = stns
    ainput.stn_wgts = wgts
    ainput.stn_obs = stn_obs
    ainput.set_pt(xval_stn[LON], xval_stn[LAT], xval_stn[ELEV])
    
    interper.model_prcp(ainput, aoutput)
    atimer.stop("")
    ci_l,ci_u = aoutput.ci
    po = np.copy(aoutput.prcp)
    po[po > 0] = 1
    prcp = aoutput.prcp
    print aoutput.to_string()
    print np.mean(xval_obs)
    
    #Calculate hss comparing to observed
    hss = calc_hss(xval_obs_po,po)
    
    #Frequency wd d-1
    freq_obs = np.nonzero(xval_obs_po > 0)[0].size / np.float(xval_obs_po.size)
    freq_fit = np.nonzero(po > 0)[0].size / np.float(po.size)
    perr_freq = ((freq_fit - freq_obs) / freq_obs) * 100.
    
    perr_ttlamt = ((np.sum(prcp) - np.sum(xval_obs)) / np.sum(xval_obs)) * 100.

    #Intensity cm wd-1
    intsy_obs = np.sum(xval_obs[xval_obs > 0]) / xval_obs[xval_obs > 0].size
    intsy_fit = np.sum(prcp[prcp > 0]) / prcp[prcp > 0].size
    perr_intsy = ((intsy_fit - intsy_obs) / intsy_obs) * 100.
    
    print hss,perr_ttlamt,perr_freq,perr_intsy
    
    plt.plot(prcp)
    plt.show()

def test_po_interp():
    
    rmod = ip.modeler_R()
    cmod = ip.modeler_clib()
    po_interperR = ip.interp_po(rmod)
    po_interperC = ip.interp_po(cmod)
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_prcp.nc','prcp')
    
    stn_id = 'GHCN_US1COSU0004'
    min_nghs = 30
    thres = 0.5

    stn_slct = station_select(stn_da.stns,min_nghs,min_nghs+10)

    xval_stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
    xval_obs = stn_da.load_obs(stn_id)
    xval_obs[xval_obs > 0] = 1

    dist = utlg.grt_circle_dist(xval_stn[LON], xval_stn[LAT], stn_da.stns[LON], stn_da.stns[LAT])
    rm_stn_ids = np.unique(np.concatenate((np.array([stn_id]),stn_da.stn_ids[dist==0])))

    stns,wgts,rad = stn_slct.get_interp_stns(xval_stn[LAT], xval_stn[LON], rm_stn_ids)
    
    stn_obs = stn_da.load_obs(stns[STN_ID])
    stn_obs[stn_obs > 0] = 1
    
    poR = po_interperR.model_po(stns, wgts, stn_obs, xval_stn,thres)
    poC = po_interperC.model_po(stns, wgts, stn_obs, xval_stn,thres)
    
    hssR = calc_hss(xval_obs, poR)
    hssC = calc_hss(xval_obs, poC)
    
    print hssR,hssC
    print np.sum(poR-poC)
   
def xval_cir_mae():
    
    ds_xval = Dataset('/projects/daymet2/station_data/infill/xval_tmax_cir.nc')
    db = station_data_infill('/projects/daymet2/station_data/infill/infill_tmax_center.nc','tmax')
    
    stn_ids = np.array(ds_xval.variables['stn_id'][:],dtype="<S16")
    stns = db.stns[np.in1d(db.stns[STN_ID], stn_ids, assume_unique=True)]
    
    rgns = get_neon_rgns('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.nc', stns)
    mae = ds_xval.variables['mae'][:].flatten()
    cir = ds_xval.variables['cir'][:].flatten()
    
    print rgns.shape
    print mae.shape
    print cir.shape
    
    amask = rgns == 12
    
    mae = mae[amask]
    cir = cir[amask]
    
    plt.plot(mae,cir,'.')
    plt.show()
    
 
def test_tair_interp():
    
    
    
    a_clib = clib_wxTopo('/home/jared.oyler/ecl_helios_workspace/wxTopo/Release/libwxTopo')
    
    db_tmin = station_data_infill('/projects/daymet2/station_data/infill/infill_tmin_center.nc','tmin')
    db_tmax = station_data_infill('/projects/daymet2/station_data/infill/infill_tmax_center.nc','tmax')
    
    #My house
#    lat =  46.837142  
#    lon = -114.013169 
#    elev = 977
    
    #Seeley Lake Hare Site
#    lon = -113.430344
#    lat = 47.230488
#    elev = 1405
    
    #Gardiner Hare Site
    lon=-110.587747
    lat= 45.096985
    elev= 2389


    stn_slct_tmin = station_select(db_tmin.stns,30,40)
    stn_slct_tmax = station_select(db_tmax.stns,30,40)

    
    atimer = timer()
    atimer.start()

    stns_tmin,wgts_tmin,rad_tmin = stn_slct_tmin.get_interp_stns(lat, lon)
    stns_tmax,wgts_tmax,rad_tmax = stn_slct_tmax.get_interp_stns(lat, lon)
    
    stn_obs_tmin = db_tmin.load_obs(stns_tmin[STN_ID])
    stn_obs_tmax = db_tmax.load_obs(stns_tmax[STN_ID])
    
    ainput = tair_input()
    aoutput = tair_output()
    
    ainput.set_pt(lon, lat, elev)
    ainput.stns_tmin = stns_tmin
    ainput.stn_obs_tmin = stn_obs_tmin
    ainput.stn_wgts_tmin = wgts_tmin
    ainput.stns_tmax = stns_tmax
    ainput.stn_obs_tmax = stn_obs_tmax
    ainput.stn_wgts_tmax = wgts_tmax
    

#    boot_rslts_tmax = a_clib.bootstrap(stns_tmax[LON], stns_tmax[LAT], stns_tmax[ELEV], wgts_tmax, stns_tmax["MEAN_OBS"], lon, lat, elev)
#    boot_rslts_tmin = a_clib.bootstrap(stns_tmin[LON], stns_tmin[LAT], stns_tmin[ELEV], wgts_tmin, stns_tmin["MEAN_OBS"], lon, lat, elev)
#    
#    vals_tmax = a_clib.prcomp_interp(stns_tmax[LON], stns_tmax[LAT], stns_tmax[ELEV], wgts_tmax, stn_obs_tmax, lon,lat,elev, boot_rslts_tmax[0])
#    vals_tmin = a_clib.prcomp_interp(stns_tmin[LON], stns_tmin[LAT], stns_tmin[ELEV], wgts_tmin, stn_obs_tmin, lon,lat,elev, boot_rslts_tmin[0])
#    
#    print vals_tmax
#    print vals_tmin
    
    #plt.plot(vals)
    #plt.show()
    #print a_clib.bootstrap(stns_tmin[LON], stns_tmin[LAT], stns_tmin[ELEV], wgts_tmin, stns_tmin["MEAN_OBS"], lon, lat, elev)
    #print a_clib.regress(stns_tmax[LON], stns_tmax[LAT], stns_tmax[ELEV], wgts_tmax, stns_tmax["MEAN_OBS"], lon, lat, elev)
    #print a_clib.regress(stns_tmin[LON], stns_tmin[LAT], stns_tmin[ELEV], wgts_tmin, stns_tmin["MEAN_OBS"], lon, lat, elev)
    modR = modeler_R()
    modC = modeler_clib()
    
    interptR = interp_tair(modR)
    interptC = interp_tair(modC)
    print "start"
    interptC.model_tmin_tmax(ainput, aoutput)
    print aoutput.tmax
    print aoutput.tmin
#    aoutput.to_csv(db_tmin.days, "gardiner_tair.csv")
#    print aoutput.to_str("GARDINER")
    
    #print aoutput.mean_tmax,aoutput.se_tmax,aoutput.ci_tmax[0],aoutput.ci_tmax[1]
    #print aoutput.mean_tmin,aoutput.se_tmin,aoutput.ci_tmin[0],aoutput.ci_tmin[1]
    #print aoutput.mean_tmax
    #print aoutput.mean_tmin
    interptR.model_tmin_tmax(ainput, aoutput)
    print aoutput.tmax
    print aoutput.tmin
#    print aoutput.ci_tmin
#    print aoutput.ci_tmax
#    plt.plot(aoutput.tmin)
#    plt.plot(aoutput.tmax)
#    plt.show()
    
def get_hcn_dict():
    
    afile = open('/projects/daymet2/station_data/ghcn/ghcnd-stations.txt')
    hcn_dict = {}
    
    for line in afile.readlines():
        
        stn_id = "".join(["GHCN_",line[0:11].strip()])
        hcn_flg = line[76:79]
        hcn_dict[stn_id] = hcn_flg
        
    return hcn_dict

def make_tiles_ncdf():
    
    ds_mask = Dataset('/projects/daymet2/dem/smoothed/ncdf/tiles.nc','r+')
    mask = np.array(ds_mask.variables['mask'][:],dtype=np.bool)
    ds_mask.variables['mask'].missing_value =  netCDF4.default_fillvals['i2']
    tiles = np.ones(mask.shape,dtype=np.int32)
    tiles.fill(netCDF4.default_fillvals['i2'])
    
    lons = ds_mask.variables['lon'][:]
    lats = ds_mask.variables['lat'][:]
    nrows = lats.size
    ncols = lons.size
    
    x = 0
    for i in np.arange(0,nrows,250):
        
        for j in np.arange(0,ncols,250):
            
            msk_tile = mask[i:i+250,j:j+250]
            
            if msk_tile[msk_tile].size > 0:
                
                ds_mask.variables['mask'][i:i+250,j:j+250] = x
                x+=1
            
            else:
                
                ds_mask.variables['mask'][i:i+250,j:j+250] = netCDF4.default_fillvals['i2']
                
                
    ds_mask.close()

    to_geotiff(Dataset('/projects/daymet2/dem/smoothed/ncdf/tiles.nc'), 'mask', '/projects/daymet2/dem/tiles.tif')

def create_neon_nc():
    to_ncdf('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif', 'neon', '/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.nc', np.int16, 255)

def create_nc_grids():
    #to_ncdf('/projects/daymet2/dem/interp_grids/tifs/elev.tif', 'elev', '/projects/daymet2/dem/interp_grids/ncdf/elev.nc', np.float32, -9999.)
    #to_ncdf('/projects/daymet2/dem/interp_grids/tifs/interp_mask_conus.tif', 'mask', '/projects/daymet2/dem/interp_grids/ncdf/interp_mask_conus.nc', np.int8)
    #to_ncdf('/projects/daymet2/dem/interp_grids/tifs/lst_tmin.tif', 'lst_tmin', '/projects/daymet2/dem/interp_grids/ncdf/lst_tmin.nc', np.float32, -9999.)
    #to_ncdf('/projects/daymet2/dem/interp_grids/tifs/lst_tmax.tif', 'lst_tmax', '/projects/daymet2/dem/interp_grids/ncdf/lst_tmax.nc', np.float32, -9999.)
    #to_ncdf('/projects/daymet2/dem/interp_grids/tifs/tdi.tif', 'tdi', '/projects/daymet2/dem/interp_grids/ncdf/tdi.nc', np.float32, -9999.)
    #to_ncdf('/Users/jaredwo/Downloads/wxtopo_data/interp_grids/neon.tif', 'neon', '/Users/jaredwo/Downloads/wxtopo_data/interp_grids/neon.nc', np.float32, -9999.)
    
    #CCE
    #to_ncdf('/projects/daymet2/dem/cce/interp_grids/cce_elev.tif', 'elev', '/projects/daymet2/dem/cce/interp_grids/ncdf/elev.nc', np.float32, -9999.)
    #to_ncdf('/projects/daymet2/dem/cce/interp_grids/cce_mask.tif', 'mask', '/projects/daymet2/dem/cce/interp_grids/ncdf/mask.nc', np.int8)
    #to_ncdf('/projects/daymet2/dem/cce/interp_grids/cce_lst_tmin.tif', 'lst_tmin', '/projects/daymet2/dem/cce/interp_grids/ncdf/lst_tmin.nc', np.float32, -9999.)
    #to_ncdf('/projects/daymet2/dem/cce/interp_grids/cce_lst_tmax.tif', 'lst_tmax', '/projects/daymet2/dem/cce/interp_grids/ncdf/lst_tmax.nc', np.float32, -9999.)
    #to_ncdf('/projects/daymet2/dem/cce/interp_grids/cce_tdi.tif', 'tdi', '/projects/daymet2/dem/cce/interp_grids/ncdf/tdi.nc', np.float32, -9999.)
    
    #CCE Expand
    new_dims = (500,500)
    
    ds_mask = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/mask.nc')
    mask = ds_mask.variables['mask'][:]
    ds_elev = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/elev.nc')
    ds_lst_tmin = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/lst_tmin.nc')
    ds_lst_tmax = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/lst_tmax.nc')
    ds_tdi = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/tdi.nc')
    
    expand_grid(ds_mask,'mask',new_dims, '/projects/daymet2/dem/cce/interp_grids/ncdf/500_mask.nc',0,mask)
    expand_grid(ds_elev,'elev',new_dims, '/projects/daymet2/dem/cce/interp_grids/ncdf/500_elev.nc',-9999.,mask)
    expand_grid(ds_lst_tmin,'lst_tmin',new_dims, '/projects/daymet2/dem/cce/interp_grids/ncdf/500_lst_tmin.nc',-9999.,mask)
    expand_grid(ds_lst_tmax,'lst_tmax',new_dims, '/projects/daymet2/dem/cce/interp_grids/ncdf/500_lst_tmax.nc',-9999.,mask)
    expand_grid(ds_tdi,'tdi',new_dims, '/projects/daymet2/dem/cce/interp_grids/ncdf/500_tdi.nc',-9999.,mask)

def create_cce_blank_neon_nc():
    
    ds_mask = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_mask.nc')
    lons = ds_mask.variables['lon'][:]
    lats = ds_mask.variables['lat'][:]
    
    ds_neon = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_neon.nc','w')
        
    ds_neon.createDimension('lat',lats.size)
    ds_neon.createDimension('lon',lons.size)
    
    latitudes = ds_neon.createVariable('lat','f8',('lat',),fill_value=False)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lats

    longitudes = ds_neon.createVariable('lon','f8',('lon',),fill_value=False)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lons
    
    neon = ds_neon.createVariable('neon','f4',('lat','lon'),fill_value=-9999.)
    neon[:] = 2
    
    ds_neon.sync()
    
    

def create_center_prcp_var():
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_prcp.nc','r+')
    
    prcp_var = ds.createVariable('prcp_mean','f8',('stn_id',))
    prcp_var.long_name = "mean precipitation"
    prcp_var.units = "cm"
    prcp_var.standard_name = "mean_precipitation"
    prcp_var.missing_value = netCDF4.default_fillvals['f8']
    
    prcp = ds.variables['prcp'][:]
    
    prcp_mean = np.mean(prcp, axis=0, dtype=np.float64)
    
    ds.variables['prcp_mean'][:] = prcp_mean
    ds.close()
    
def create_centered_prcp_ds():
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_prcp_center.nc','r+')
    
    vals = np.require(ds.variables['prcp'][:], dtype=np.float64)
    mean_vals = np.require(ds.variables['prcp_mean'][:], dtype=np.float64)
    vals_cent = vals - mean_vals
    vals = None
    
    ds.variables['prcp'][:] = vals_cent
    ds.close()

def test_trends():
    stn_db = station_data.station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    obs = stn_db.load_all_stn_obs(np.array(["GHCN_USC00244558"]))
    
    yrs = np.arange(1950,2012)
    
    plt.plot(yrs,[np.mean(obs[TMIN][np.logical_and(np.isfinite(obs[TMIN]),stn_db.days[YEAR]==yr)]) for yr in yrs])
    #plt.plot(yrs,[np.mean(obs[TMAX][stn_db.days[YEAR]==yr]) for yr in yrs])
    plt.show()

def test_quick_interps():

    #Kalispell
#    lon=-114.312760
#    lat=48.204611
#    elev= 900
    
    #Big Salmon Lake
    lon=-113.364377
    lat=47.616419
    elev= 1315
    
    path_tmin = '/projects/daymet2/station_data/infill/infill_tmin_center.nc'
    path_tmax = '/projects/daymet2/station_data/infill/infill_tmax_center_f4.nc'
    path_clib = '/home/jared.oyler/ecl_helios_workspace/wxTopo/Release/libwxTopo'
    path_rlib = '/home/jared.oyler/ecl_helios_workspace/wxTopo_R/topomet.R'
    
    db_tmin = station_data_infill('/projects/daymet2/station_data/infill/infill_tmin_center.nc', 'tmin')
    
    aoutput = quick_interp_tair(lon,lat,elev,ngh_tmin=30,ngh_tmax=36,
                      path_tmin=path_tmin,
                      path_tmax=path_tmax,
                      path_clib=path_clib,
                      path_rlib=path_rlib,
                      use_R=True)
    print aoutput.to_str("")
    
    aoutput = quick_interp_tair(lon,lat,elev,ngh_tmin=30,ngh_tmax=36,
                  path_tmin=path_tmin,
                  path_tmax=path_tmax,
                  path_clib=path_clib,
                  path_rlib=path_rlib,
                  use_R=False)
    print aoutput.to_str("")
    
    
    #aoutput = ip.quick_interp_prcp(lon, lat, elev, ngh=38, path_prcp='/projects/daymet2/station_data/infill/infill_prcp.nc')
    
#    yrs = np.arange(1950,2012)
#       
#
#    #plt.plot(yrs,[np.sum(aoutput.prcp[np.logical_and(np.isfinite(aoutput.prcp),db_tmin.days[YEAR]==yr)]) for yr in yrs] )
#    plt.plot(aoutput.tmax)
#    plt.show()

def mean_val_to_geotiff():
    
    path = '/projects/daymet2/interp_output/wxTopo_tests/montana/'
    out_path = '/projects/daymet2/interp_output/wxTopo_tests/mt_cir/'
    fnames = ["_prcp","_tmax","_tmin"]
    
    dirs = os.listdir(path)
    
    for dir in dirs:
        
        for name in fnames:
        
            fname = "".join([path,dir,"/",dir,name,".nc"])
            print fname
            to_geotiff(Dataset(fname), "".join([name[1:],"_cir"]), 
                       "".join([out_path,dir,name,"_cir.tif"]),
                       None,gdalconst.GDT_Float32, netCDF4.default_fillvals['f4'])
            
def gwr_sigma(lon=-114.013563,lat=46.836744,elev=976):
    
    path_tmax='/projects/daymet2/station_data/infill/infill_tmax_center.nc'
    db_tmax = station_data_infill(path_tmax,'tmax')
    ngh_tmax=36
    stn_slct_tmax = station_select(db_tmax.stns,ngh_tmax,ngh_tmax+10)
    ngh_stns,wgts,rad = stn_slct_tmax.get_interp_stns(lat, lon)
    #print wgts

    
    L = np.zeros((ngh_stns.size,ngh_stns.size))
    Y = ngh_stns['MEAN_OBS']
    Y.shape = (Y.shape[0],1)
    X = np.column_stack((np.ones(ngh_stns.size),ngh_stns[LON],ngh_stns[LAT],ngh_stns[ELEV]))
    I = np.identity(ngh_stns.size)

    for i in np.arange(ngh_stns.size):
        
        stn = ngh_stns[i]
        #Calculate row of L
        x = np.array([1.,stn[LON],stn[LAT],stn[ELEV]])
        x.shape = (x.shape[0],1)
        
        wgts_ngh = stn_slct_tmax.get_interp_wgts(stn[LAT], stn[LON], ngh_stns)
        
        W = np.diag(wgts_ngh)
        Lr = np.dot(np.dot(np.dot(np.transpose(x),np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))),np.transpose(X)),W)
        L[i,:] = np.ravel(Lr)
    
    RSSg = np.dot(np.dot(np.dot(np.transpose(Y),np.transpose(I-L)),(I-L)),Y)
    d1 = np.float(np.trace(np.dot(np.transpose(I-L),(I-L))))
    d2 = np.float(np.trace(np.square(np.dot(np.transpose(I-L),(I-L)))))
    df = (d1**2)/d2
    sigma2 = RSSg/d1
    sigma=np.sqrt(sigma2)
    
    ##############################
    x = np.array([1.,lon,lat,elev])
    x.shape = (x.shape[0],1)
    W = np.diag(wgts)
    S = np.float(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.transpose(x),np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))),np.transpose(X)),np.square(W)),X),np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))),x))
    
    return (sigma*((1.0+S)**.5))*stats.t.ppf(0.025,df)
    
                   
def gwr_uncertainty(lon=-114.013563,lat=46.836744,elev=976,sigma=0.63900541):
    
    path_tmax='/projects/daymet2/station_data/infill/infill_tmax_center.nc'
    db_tmax = station_data_infill(path_tmax,'tmax')
    stns = db_tmax.stns
    ngh_tmax=36
    stn_slct_tmax = station_select(db_tmax.stns,ngh_tmax,ngh_tmax+10)
    ngh_stns,wgts,rad = stn_slct_tmax.get_interp_stns(lat, lon)
    #print wgts
    x = np.array([1.,lon,lat,elev])
    x.shape = (x.shape[0],1)
    
#    X = np.column_stack((np.ones(stns.size),stns[LON],stns[LAT],stns[ELEV]))
#    w = np.zeros(stns.size)
#    w[np.in1d(stns[STN_ID], ngh_stns[STN_ID], assume_unique=True)] = wgts
#    W = np.diag(w)
#    
#    Y = stns['MEAN_OBS']
#    Y.shape = (Y.shape[0],1)

    X = np.column_stack((np.ones(ngh_stns.size),ngh_stns[LON],ngh_stns[LAT],ngh_stns[ELEV]))
    W = np.diag(wgts)
    Y = ngh_stns['MEAN_OBS']
    Y.shape = (Y.shape[0],1)
    
    inverse = np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))
    print "new"
    #S = np.float(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.transpose(x),np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))),np.transpose(X)),np.square(W)),X),np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))),x))
    S = np.float(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.transpose(x),inverse),np.transpose(X)),np.square(W)),X),inverse),x))
    return (sigma*((1.0+S)**.5))*1.96    

def gwr_calc_sigma_df(path_L,path_Y):
    L = np.load(path_L)
    Y = np.load(path_Y)
    I = np.identity(Y.size)
    RSSg = np.dot(np.dot(np.dot(np.transpose(Y),np.transpose(I-L)),(I-L)),Y)
    
    dot_prod = np.dot(np.transpose(I-L),(I-L))
    d1 = np.float(np.trace(dot_prod))
    d2 = np.float(np.trace(np.square(dot_prod)))
    df = (d1**2)/d2
    sigma2 = RSSg/d1
    sigma=np.sqrt(sigma2)
    return sigma,df

def OUTPUT_STN_CSV():
    db_tmin = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc',"tmin")
    
    fout = open("/projects/daymet2/station_data/infill/impute_tair/tmin_stn_list.csv",'w')
    fout.write(",".join([STN_ID,STN_NAME,LON,LAT,ELEV+"\n"]))

    for stn in db_tmin.stns:
        stn_name = stn[STN_NAME].replace(","," ")
        fout.write(",".join([stn[STN_ID],stn_name,str(stn[LON]),str(stn[LAT]),str(stn[ELEV])+"\n"]))
    fout.close()

def output_stn_csv():
    
    db_tmin = station_data_infill("/projects/daymet2/station_data/infill/infill_tmin.nc","tmin")
    db_tmax = station_data_infill("/projects/daymet2/station_data/infill/infill_tmax.nc","tmax")
    db_prcp = station_data_infill("/projects/daymet2/station_data/infill/infill_prcp.nc","prcp")
    
    stn_ids = np.unique(np.concatenate((db_tmin.stn_ids,db_tmax.stn_ids,db_prcp.stn_ids)))
    
    fout = open("/projects/daymet2/station_data/infill/stn_list.csv",'w')
    fout.write(",".join([STN_ID,STN_NAME,LON,LAT,ELEV,TMIN,TMAX,PRCP,"\n"]))
    
    stn = None
    
    for stn_id in stn_ids:
        
        tmin = 0
        tmax = 0
        prcp = 0
        stn = None
        
        if stn_id in db_tmin.stn_ids:
            tmin = 1
            if stn is None:
                stn = db_tmin.stns[db_tmin.stn_ids==stn_id][0]

        if stn_id in db_tmax.stn_ids:
            tmax = 1
            if stn is None:
                stn = db_tmax.stns[db_tmax.stn_ids==stn_id][0]
                
        if stn_id in db_prcp.stn_ids:
            prcp = 1
            if stn is None:
                stn = db_prcp.stns[db_prcp.stn_ids==stn_id][0]
            
        fout.write(",".join([stn[STN_ID],stn[STN_NAME],str(stn[LON]),str(stn[LAT]),str(stn[ELEV]),str(tmin),str(tmax),str(prcp),"\n"]))
    
    fout.close()

def stn_csv_from_por():
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    stns = db.stns
    por = obs_por.load_por_csv('/projects/daymet2/station_data/all/all_por.csv')
    mask_por_tmin,mask_por_tmax,mask_por_prcp = obs_por.build_valid_por_masks(por)
    
    fout = open("/projects/daymet2/station_data/all/stn_list.csv",'w')
    fout.write(",".join([STN_ID,STN_NAME,LON,LAT,ELEV,TMIN,TMAX,PRCP,"\n"]))
    for x in np.arange(stns.size):
        
        if mask_por_tmin[x] or mask_por_tmax[x] or mask_por_prcp[x]:
            
            stn = stns[x]
            
            fout.write(",".join([stn[STN_ID],stn[STN_NAME],str(stn[LON]),str(stn[LAT]),str(stn[ELEV]),str(int(mask_por_tmin[x])),str(int(mask_por_tmax[x])),str(int(mask_por_prcp[x])),"\n"]))
     
    fout.close()       
    
def fix_ca_locs():
    
    ds = Dataset('/projects/daymet2/station_data/all/all.nc',"r+")
    stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
    ca_mask = np.char.startswith(stn_ids, prefix="CA_")
    lons = ds.variables['lon'][ca_mask]
    lats = ds.variables['lat'][ca_mask]
    ds.variables['lon'][ca_mask] = lats
    ds.variables['lat'][ca_mask] = lons
    ds.sync()
    ds.close()

def gtopo_prism_merge():
    
    dem_orig = input_raster('/projects/daymet2/dem/dem_orig.tif')
    envi_stack = input_raster('/projects/daymet2/dem/gtopo/srtm30/envi_stack_srtm.tif')
    
    band1 = envi_stack.readEntireRasterBand(1)
    band2 = envi_stack.readEntireRasterBand(2)
    band2[band2==32767.0] = np.nan
    band1[band1==-999.] = np.nan
    
    lons,lats = envi_stack.x_y_arrays()
    
    lat_mask = np.logical_or(lats>dem_orig.max_y,lats<dem_orig.min_y)
    band1[lat_mask] = band2[lat_mask]
    
    lon_mask = np.logical_or(lons>dem_orig.max_x,lons<dem_orig.min_x)
    band1[lon_mask] = band2[lon_mask]
    
    band1[np.isnan(band1)] = -32767.
    outr = output_raster("/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge.tif",envi_stack)
    outr.writeDataArray(band1,0,0)
    
def gtopo_prism_merge2():
    to_geotiff(Dataset('/projects/daymet2/dem/gtopo/srtm30/srtm30_mosaic_resample.nc'), "elev","/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge_fnl.tif",nodata_val=-999)
    
    #to_ncdf('/projects/daymet2/dem/dem_orig.tif', "elev", '/projects/daymet2/dem/gtopo/srtm30/dem_orig.nc', np.float32, -999)
    #to_ncdf('/projects/daymet2/dem/gtopo/srtm30/srtm30_mosaic_resample.tif', "elev", '/projects/daymet2/dem/gtopo/srtm30/srtm30_mosaic_resample.nc', np.float32, 32767)
#    ds_orig = Dataset('/projects/daymet2/dem/gtopo/srtm30/dem_orig.nc')
#    ds_srtm = Dataset('/projects/daymet2/dem/gtopo/srtm30/srtm30_mosaic_resample.nc','r+')
#    
#    a_orig = ds_orig.variables['elev'][:]
#    lon_orig = ds_orig.variables['lon'][:]
#    lat_orig = ds_orig.variables['lat'][:]
#    max_lon = np.max(lon_orig)+0.000001
#    min_lon = np.min(lon_orig)-0.000001
#    max_lat = np.max(lat_orig)+0.000001
#    min_lat = np.min(lat_orig)-0.000001
#    
#    lon_srtm = ds_srtm.variables['lon'][:]
#    lat_srtm = ds_srtm.variables['lat'][:]
#    lon_mask = np.logical_and(lon_srtm <= max_lon,lon_srtm >= min_lon)
#    lat_mask = np.logical_and(lat_srtm <= max_lat,lat_srtm >= min_lat)
#    
#    print a_orig.shape
#    print ds_srtm.variables['elev'][lat_mask,lon_mask].shape
#    ds_srtm.variables['elev'][lat_mask,lon_mask] = a_orig
#    ds_srtm.sync()
#    elev = ds_srtm.variables['elev'][:].data
#    elev[elev == 32767] = -999
#    ds_srtm.variables['elev'][:] = elev
#    
#    ds_srtm.close()
    
def gtopo_prism_mask_merge():
    
    to_geotiff(Dataset('/projects/daymet2/dem/gtopo/srtm30/world_shp/interp_mask_na.nc'), "mask","/projects/daymet2/dem/gtopo/srtm30/world_shp/interp_mask_merge.tif",nodata_val=-999)
    sys.exit()
    to_ncdf('/projects/daymet2/dem/interp_mask.tif', "mask", '/projects/daymet2/dem/gtopo/srtm30/world_shp/interp_mask_orig.nc', np.float32,-32767)
    to_ncdf('/projects/daymet2/dem/gtopo/srtm30/world_shp/na_mask.tif', "mask", '/projects/daymet2/dem/gtopo/srtm30/world_shp/interp_mask_na.nc', np.float32)
    #sys.exit()
    ds_orig = Dataset('/projects/daymet2/dem/gtopo/srtm30/world_shp/interp_mask_orig.nc')
    ds_srtm = Dataset('/projects/daymet2/dem/gtopo/srtm30/world_shp/interp_mask_na.nc','r+')
    
    a_orig = ds_orig.variables['mask'][:]
    lon_orig = ds_orig.variables['lon'][:]
    lat_orig = ds_orig.variables['lat'][:]
    max_lon = np.max(lon_orig)+0.0001
    min_lon = np.min(lon_orig)-0.0001
    max_lat = np.max(lat_orig)+0.0001
    min_lat = np.min(lat_orig)-0.0001
    
    lon_srtm = ds_srtm.variables['lon'][:]
    lat_srtm = ds_srtm.variables['lat'][:]
    lon_mask = np.logical_and(lon_srtm <= max_lon,lon_srtm >= min_lon)
    lat_mask = np.logical_and(lat_srtm <= max_lat,lat_srtm >= min_lat)
    
    a_new = ds_srtm.variables['mask'][lat_mask,lon_mask]
    
    print a_orig.shape
    print a_new.shape
    
    a_new[a_new != 1] = a_orig[a_new != 1]
    
    ds_srtm.variables['mask'][lat_mask,lon_mask] = a_new
    ds_srtm.sync()
    ds_srtm.close()


def srtm_prism_clip():
    srtm = input_raster("/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge.tif")
    lons,lats = srtm.x_y_arrays()
    clip_mask = np.logical_or(np.logical_or(lats>62,lats<14),np.logical_or(lons<-150,lons>-50))
    
    
    
    a = srtm.readEntireRaster()
    a[clip_mask] = -32767.
    outr = output_raster("/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge_clip2.tif",srtm)
    outr.writeDataArray(a,0,0)
    
def co_site():
    
    db_tmin = station_data_infill('/projects/daymet2/station_data/infill/infill_tmin_center.nc','tmin')
    days = db_tmin.days
    ymd = days[YMD]
    
    lon = -102.1
    lat = 38.1
    elev = 1069
    
    aoutput = ip.quick_interp_prcp(lon, lat, elev)
    #print aoutput.to_str(site_name)
    prcp = aoutput.prcp
    print np.sum(prcp[days[YEAR]==2009])
    
    aoutput = it.quick_interp_tair(lon, lat, elev)
    #print aoutput.to_str(site_name)
    tmin = aoutput.tmin
    tmax = aoutput.tmax
    
    fmet = open('/projects/daymet2/co_site.csv','w')
    fmet.write(",".join([YMD,TMIN,TMAX,PRCP,"\n"]))
    
    for x in np.arange(tmin.size):
        
        fmet.write(",".join([str(ymd[x]),str(tmin[x]),str(tmax[x]),str(prcp[x]),"\n"]))
    
    fmet.close()

def stn_names_location_qa():
    
    path_out = "/projects/daymet2/station_data/all/qa_elev_20120906_names.csv"
    path_in = "/projects/daymet2/station_data/all/qa_elev_20120906.csv"
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    stns = db.stns
    stn_ids = db.stn_ids
    
    fout = open(path_out,"w")
    fout.write(",".join(["STN_ID", "NAME","ST", "LON", "LAT", "ELEV", "DEM", "DIF","LON_NEW","LAT_NEW","ELEV_NEW","\n"]))
    
    fin = open(path_in)
    fin.readline() #skip header
    
    for line in fin.readlines():
        
        vals = line.split(",")
        name = stns["NAME"][stn_ids==vals[0]][0]
        name = name.replace(","," ")
        vals.insert(1,name)
        fout.write(",".join(vals))
    
    fout.close()

def fix_raws_prcp():
    
    ds = Dataset('/projects/daymet2/station_data/all/all.nc',"r+")
    
    stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
    raws_mask = np.nonzero(np.char.startswith(stn_ids, prefix="RAWS_"))[0]
    print "reading...."
    prcp = ds.variables['prcp'][:,raws_mask] 
    prcp[np.round(prcp)==-1000.0] = -9999.
    print "writing..."
    ds.variables['prcp'][:,raws_mask] = prcp
    ds.close()

def srtm_to_ncdf():
    
    ds = Dataset("/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge_clip.nc")
    plt.imshow(ds.variables['elev'][:])
    plt.show()
    
    to_geotiff(ds, "elev","/projects/daymet2/dem/gtopo/srtm30/srtm_prism_fnl.tif")
    sys.exit()
    
    #to_ncdf("/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge_clip2.tif","elev", "/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge.nc",np.float32,-32767)
    ds = Dataset('/projects/daymet2/dem/gtopo/srtm30/srtm30_mosaic_resample.nc')
    elev = np.array(ds.variables['elev'][:])
    lons = ds.variables['lon'][:]
    lats = ds.variables['lat'][:]
    
    elev[np.logical_or(lats>62,lats<14),:] = -999
    elev[:,np.logical_or(lons<-150,lons>-50)] = -999
    
    mask = elev != -999
    
    nonzero_rows,nonzero_cols = np.nonzero(mask)
    nonzero_rows = np.unique(nonzero_rows)
    nonzero_cols = np.unique(nonzero_cols)
    
    nonzero_lons = lons[nonzero_cols]
    nonzero_lats = lats[nonzero_rows]
    
    nonzero_elev = elev[nonzero_rows,:]
    nonzero_elev = nonzero_elev[:,nonzero_cols]
    elev = nonzero_elev
    
    ncdf_file = Dataset("/projects/daymet2/dem/gtopo/srtm30/srtm_prism_merge_clip.nc",'w')

    #Create 2-dimensions
    ncdf_file.createDimension('lat',nonzero_lats.size)
    ncdf_file.createDimension('lon',nonzero_lons.size)

    latitudes = ncdf_file.createVariable('lat','f8',('lat',))
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = nonzero_lats

    longitudes = ncdf_file.createVariable('lon','f8',('lon',))
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = nonzero_lons
    
    ncdf_var = ncdf_file.createVariable("elev",np.float32,('lat','lon',),fill_value=False)
    ncdf_var.missing_value = -999
    ncdf_var[:,:] = elev
    ncdf_file.close()

def fix_raws_loc_qa():
    
    path_out = "/projects/daymet2/station_data/all/qa_elev_20120906_rawsfix.csv"
    path_in = "/projects/daymet2/station_data/all/qa_elev_20120906.csv"
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    stns = db.stns
    stn_ids = db.stn_ids
    
    por = obs_por.load_por_csv('/projects/daymet2/station_data/all/all_por.csv')
    mask_por_tmin,mask_por_tmax,mask_por_prcp = obs_por.build_valid_por_masks(por)
    mask = np.logical_or(np.logical_or(mask_por_tmin,mask_por_tmax),mask_por_prcp)
    mask = np.logical_and(mask,np.char.startswith(db.stn_ids, prefix="RAWS"))
    raw_ids = stn_ids[mask]
    
    fout = open(path_out,"w")
    fout.write(",".join(["STN_ID","NAME","ST", "LON", "LAT", "ELEV", "DEM", "DIF","LON_NEW","LAT_NEW","ELEV_NEW","\n"]))
    
    fin = open(path_in)
    fin.readline() #skip header
    
    for line in fin.readlines():
        
        vals = line.split(",")
        
        if vals[0][0:4] == "RAWS" and vals[0] not in raw_ids:
            print vals[0]
        else:
            name = stns["NAME"][stn_ids==vals[0]][0]
            name = name.replace(","," ")
            vals.insert(1,name)
            fout.write(",".join(vals))
        
    
    fout.close()

def stn_map():
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    stns = db.stns
    por = obs_por.load_por_csv('/projects/daymet2/station_data/all/all_por.csv')
    mask_por_tmin,mask_por_tmax,mask_por_prcp = obs_por.build_valid_por_masks(por)
    
    ds = Dataset('/projects/daymet2/station_data/infill/xval_infill_tair.nc')
    stn_ids = np.array(ds.variables['stn_id'][:],dtype="<S16")
    ds.close()
    
    
    
    stns = stns[np.in1d(db.stn_ids, stn_ids, assume_unique=True)]
    
    #stns = stns[np.logical_and(stns[LON]>-360,stns[LON]<-48)]
    print stns.size
    m = Basemap(projection='cyl',llcrnrlat=np.min(stns[LAT]),urcrnrlat=np.max(stns[LAT]),\
            llcrnrlon=np.min(stns[LON]),urcrnrlon=np.max(stns[LON]),resolution='c')
    m.drawcountries()
    m.drawcoastlines()
    m.scatter(stns[LON],stns[LAT])
    
    plt.show()

def debug_qa():
    np.seterr(all='raise')
    np.seterr(under='ignore')
    np.seterr(invalid='ignore')
    stn_id = 'SNOTEL_05N23S'
    db = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    stn = db.stns[db.stn_ids==stn_id][0]
    print stn
    obs = db.load_all_stn_obs(np.array([stn_id]))
    
    plt.plot(obs[TMIN])
    #ftmin,ftmax,fprcp = run_qa_spatial(stn, db, obs[TMIN],obs[TMAX],obs[PRCP],db.days)
    ftmin,ftmax = qa_temp.run_qa_spatial_only(stn, db, obs[TMIN],obs[TMAX],db.days)
    plt.plot(obs[TMIN])
    plt.show()
    
    print np.unique(ftmin)
    print np.unique(ftmax)
    obs = db.load_all_stn_obs(np.array([stn_id]))
    
    mask_tmin = np.logical_not(np.logical_or(ftmin == qa_temp.QA_OK, ftmin == qa_temp.QA_MISSING))
    mask_tmax = np.logical_not(np.logical_or(ftmax == qa_temp.QA_OK, ftmax == qa_temp.QA_MISSING))
    
    nobs_tmin = np.sum(ftmin != qa_temp.QA_MISSING)
    nobs_tmax = np.sum(ftmax != qa_temp.QA_MISSING)
    
    nflag_tmin = np.sum(mask_tmin)
    nflag_tmax = np.sum(mask_tmax)
    
    if nobs_tmin == 0:
        pctflg_tmin = np.nan
    else:
        pctflg_tmin = (nflag_tmin/float(nobs_tmin))*100.0
        
    if nobs_tmax == 0:
        pctflg_tmax = np.nan
    else:
        pctflg_tmax = (nflag_tmax/float(nobs_tmax))*100.0
    
    print "|".join([stn_id,"%.2f|%.2f"%(pctflg_tmin,pctflg_tmax)])

def reset_raw_flags():
    
    ds = Dataset('/projects/daymet2/station_data/all/all.nc','r+')
    stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
    raws_mask = np.nonzero(np.char.startswith(stn_ids,"RAWS"))[0]
    print raws_mask.size
    
    ds.variables['qflag_tmin'][:,raws_mask] = ''
    ds.variables['qflag_tmax'][:,raws_mask] = ''
    ds.variables['qflag_prcp'][:,raws_mask] = ''
    
    ds.close()

def qa_stats():
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    raws_mask = np.char.startswith(db.stn_ids,"SNOTEL")
    raws_ids = db.stn_ids[raws_mask]
    
    obs = db.load_all_stn_obs(raws_ids, set_flagged_nan=False)
    
    ntmin = np.sum(np.isfinite(obs[TMIN]))
    ntmax = np.sum(np.isfinite(obs[TMAX]))
    nprcp = np.sum(np.isfinite(obs[PRCP]))
    
    nftmin = np.sum(obs[TMIN_FLAG] != "")
    nftmax = np.sum(obs[TMAX_FLAG] != "")
    nfprcp = np.sum(obs[PRCP_FLAG] != "")

    print ntmin,ntmax,nprcp
    print nftmin,nftmax,nfprcp
    print float(nftmin)/ntmin*100.,float(nftmax)/ntmax*100,float(nfprcp)/nprcp*100

def infill_micromet():
    
    min_date = datetime(1948,1,1)
    max_date = datetime(2012,6,30)
    stn_id = 'USFS_L9900300842'#'SNOTEL_15F01S'#'USFS_L9900300825'#"USFS_L9900300842"
    var = 'tmax'
    nnghs = 18
    
    usfs = insert_usfs("/projects/daymet2/station_data/usfs_micro/", min_date, max_date)
    db = station_data_combine('/projects/daymet2/station_data/all/all.nc', usfs)
    #db = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    
    ds_norms = Dataset('/projects/daymet2/station_data/infill/normals_tair.nc')
    norms = ds_norms.variables['norm'][:]
    norms_tmin = norms[0,:]
    norms_tmax = norms[1,:]
    norms_stnid = np.array(ds_norms.variables['stn_id'][:], dtype="<S16")
    norms = {}
    norms['tmin'] = norms_tmin
    norms['tmax'] = norms_tmax
    ds_norms.close()
    
    mth_masks = build_mth_masks(db.days)
    mthbuf_masks = build_mth_masks(db.days,MTH_BUFFER)
    
    print "Calculating norm..."
    fit,obs = infill_tair(stn_id, db, db.days, var, mth_masks, mthbuf_masks)
    
    ###############
    #STATS on infill
    nan_mask = np.isnan(obs)
    fin_mask = np.logical_not(nan_mask)
    val_fit = fit[fin_mask]
    val_obs = obs[fin_mask]
    
    mae = np.mean(np.abs(val_fit-val_obs))
    bias = np.mean(val_fit-val_obs)
    print mae,bias
    
    plt.plot(fit)
    plt.plot(obs)
    plt.show()
    
    plt.subplot(211)
    plt.plot(obs[fin_mask],fit[fin_mask],".")
    plt.xlabel("Observed")
    plt.ylabel("Modeled")
    plt.xlim((-30,30))
    plt.ylim((-30,30))
    aline = np.arange(-30,31)
    plt.plot(aline,aline,"r")
    plt.title("Regression-Based Model")
    ###############
    
    mask_nan = np.isnan(obs)
    fnl_tair = np.copy(obs)
    fnl_tair[mask_nan] = fit[mask_nan]
    norm_est = np.mean(fnl_tair)
    norms_pca = np.zeros(db.stn_ids.size)
    
    norms_pca[np.in1d(db.stn_ids, norms_stnid,True)] = norms[var]
    norms_pca[db.stn_ids==stn_id] = norm_est
    #norms_pca = norms[var]
    
    print "Building PCA matrix..."
    a_pca_matrix = pca_matrix(stn_id, db,var, norms_pca)
    
    print "Performing final infill..."
    fit_tair,obs_tair,npcs = a_pca_matrix.infill(nnghs)[0:3]
    
    print "# of PCS: "+str(npcs)
    
    nan_mask = np.isnan(obs_tair)
    fin_mask = np.logical_not(nan_mask)
    val_fit = fit_tair[fin_mask]
    val_obs = obs_tair[fin_mask]
    
    mae = np.mean(np.abs(val_fit-val_obs))
    bias = np.mean(val_fit-val_obs)
    print mae,bias
    
    plt.subplot(212)
    plt.plot(obs_tair[fin_mask],fit_tair[fin_mask],".")
    plt.xlabel("Observed")
    plt.ylabel("Modeled")
    plt.xlim((-30,30))
    plt.ylim((-30,30))
    aline = np.arange(-30,31)
    plt.plot(aline,aline,"r")
    plt.title("PPCA-Based Model")
    
    plt.show()
    
    plt.plot(fit_tair)
    plt.plot(obs_tair)
    plt.show()

def pnw_raws():
    
    out_dir = "/projects/daymet2/station_data/raws/raws_pnw/"
    obs_dir = "/projects/daymet2/station_data/raws/raws_data/"
    fmeta = open("/projects/daymet2/station_data/raws/raws_meta.txt")
    PNW_STATES = ["Oregon","Washington","Idaho","Montana"]
    pnw_lines = []
    
    for line in fmeta.readlines():
        vals = line.split()
        
        if vals[-1].strip() in PNW_STATES:
            
            pnw_lines.append(line)
            fobs = "".join([obs_dir,vals[0],".txt "])
            os.system("".join(["cp ",fobs,out_dir]))
    
    fmeta.close()
    
    fometa = open("".join([out_dir,"stn_meta.txt"]),"w")
    fometa.writelines(pnw_lines)
    fometa.close()

def stn_id_list():
    
    ds = Dataset('/projects/daymet2/station_data/all/all.nc')
    stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
    names = [x.encode('ascii','replace') for x in ds.variables['name'][:]]

    fout = open("/projects/daymet2/station_data/stn_list.csv","w")
    for x in np.arange(stn_ids.size):
    
        fout.write("".join([stn_ids[x],",",names[x],"\n"]))
    fout.close()

def create_nometa_db():
    copy_db_ncdf_nometa('/projects/daymet2/station_data/all/all.nc', 
                        '/projects/daymet2/station_data/all/all_nometa.nc', 
                        '/projects/daymet2/station_data/all/all_stnmeta.csv')

def sites_dm_dd():

    fout = open("/projects/daymet2/cory_lab/sites.csv","w")
    fout.write(",".join(["NAME","LAT","LON","ELEV\n"]))
    
    lons = []
    lats = []
    names = []
    
    name = "Easton Glacier WA:"
    lat,lon = dd.dm2decimal(48,44.879),dd.dm2decimal(-121,50.272)
    elev = get_elev_geonames(lon, lat)
    fout.write(",".join([name,str(lat),str(lon),str(elev)+"\n"]))
    lons.append(lon)
    lats.append(lat)
    names.append(name)

    #Morgan sites
    name = "Sulfur Cinquefoil Alley:"
    lat,lon = dd.dms2decimal(46,41,26.03),dd.dms2decimal(-113,59,1.71)
    elev = get_elev_geonames(lon, lat)
    fout.write(",".join([name,str(lat),str(lon),str(elev)+"\n"]))
    lons.append(lon)
    lats.append(lat)
    names.append(name)
    
    name = "Sheep Camp:"
    lat,lon = dd.dms2decimal(46,42,4.07),dd.dms2decimal(-114,1,17.21)
    elev = get_elev_geonames(lon, lat)
    fout.write(",".join([name,str(lat),str(lon),str(elev)+"\n"]))
    lons.append(lon)
    lats.append(lat)
    names.append(name)
    
    name = "Jens Site:"
    lat,lon = dd.dms2decimal(46,41,41.99),dd.dms2decimal(-113,59,49.81)
    elev = get_elev_geonames(lon, lat)
    fout.write(",".join([name,str(lat),str(lon),str(elev)+"\n"]))
    lons.append(lon)
    lats.append(lat)
    names.append(name)

    name = "Orange Street:"
    lat,lon = dd.dms2decimal(46,52,55.51),dd.dms2decimal(-113,59,24.54)
    elev = get_elev_geonames(lon, lat)
    fout.write(",".join([name,str(lat),str(lon),str(elev)+"\n"]))
    lons.append(lon)
    lats.append(lat)
    names.append(name)

    name = "Grant Creek:"
    lat,lon = dd.dms2decimal(46,55,23.66),dd.dms2decimal(-114,1,28.90)
    elev = get_elev_geonames(lon, lat)
    fout.write(",".join([name,str(lat),str(lon),str(elev)+"\n"]))
    lons.append(lon)
    lats.append(lat)
    names.append(name)

    name = "Cox Property:"
    lat,lon = dd.dms2decimal(46,50,20.26),dd.dms2decimal(-113,58,15.00)
    elev = get_elev_geonames(lon, lat)
    fout.write(",".join([name,str(lat),str(lon),str(elev)+"\n"]))
    lons.append(lon)
    lats.append(lat)
    names.append(name)
    
    print lons,lats,names
    create_kml_pts("/projects/daymet2/cory_lab/sites.kml", np.array(lons), np.array(lats), np.array(names))


def create_kml_pts(out_path,lons,lats,field,field_name="Name",field_width=30):
    
    drv = ogr.GetDriverByName("KML")
    ds = drv.CreateDataSource(out_path)
    lyr = ds.CreateLayer("point_out",None,ogr.wkbPoint)
    
    field_defn = ogr.FieldDefn(field_name, ogr.OFTString )
    field_defn.SetWidth(field_width)
    lyr.CreateField (field_defn)
    
    for x in np.arange(lons.size):
    
        feat = ogr.Feature( lyr.GetLayerDefn())
        feat.SetField( "Name", field[x])
    
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.SetPoint_2D(0,lons[x],lats[x])
        feat.SetGeometry(pt)
    
        lyr.CreateFeature(feat)
        feat.Destroy()

    ds = None

def quick_interp_cory_tair():
    tair_out = tair_output()
    tair_in = tair_input()
    
    #Missoula
    tair_in,tair_out = quick_interp_tair(-113.99015,46.8820861111,1069.18)
    #Baker
    #tair_in,tair_out = quick_interp_tair(-121.837866667,48.7479833333,2043.88)

    ngh_ids = np.unique(np.concatenate((tair_in.stns_tmin[STN_ID],tair_in.stns_tmax[STN_ID])))
    db = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    ngh_stns = db.stns[np.in1d(db.stn_ids, ngh_ids, assume_unique=True)]
    print ngh_stns
    
    #create_kml_pts("/projects/daymet2/cory_lab/baker_tair_stns.kml",ngh_stns[LON],ngh_stns[LAT],ngh_stns[STN_NAME])
    
    print tair_out.ci_tmax
    print tair_out.ci_tmin
    print tair_out.mean_tmax,np.mean(tair_out.tmax)
    print tair_out.mean_tmin,np.mean(tair_out.tmin)
    
    plt.plot(tair_out.tmin)
    plt.plot(tair_out.tmax)
    plt.show()
    
def quick_interp_cory_prcp():
    
    prcp_out = ip.prcp_output()
    prcp_in = ip.prcp_input()
    
    #Missoula
    #prcp_in,prcp_out = ip.quick_interp_prcp(-113.99015,46.8820861111,1069.18)
    #Baker
    prcp_in,prcp_out = ip.quick_interp_prcp(-121.837866667,48.7479833333,2043.88)
    
    db = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    ngh_stns = db.stns[np.in1d(db.stn_ids, prcp_in.stns[STN_ID], assume_unique=True)]
    print ngh_stns
       
    #create_kml_pts("/projects/daymet2/cory_lab/missoula_prcp_stns.kml",ngh_stns[LON],ngh_stns[LAT],ngh_stns[STN_NAME])

    print prcp_out.ci
    print prcp_out.mean,np.mean(prcp_out.prcp)
    
    print np.sum(prcp_out.prcp > 0)/float(prcp_out.prcp.size)*100.
    
    plt.plot(prcp_out.prcp)
    plt.show()
    

def po_analysis():
    db = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    prcp = db.load_all_stn_obs_var("GHCN_USW00024153","tmax")[0]
    plt.plot(prcp)
    plt.show()
    sys.exit()
    prcp = prcp[np.isfinite(prcp)]
    print np.sum(prcp > 0)/float(prcp.size)*100.

def montana_stns_kml():
    
    db = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stns = db.stns
    
    stns = stns[np.logical_or(np.char.upper(stns[STATE]) == "MT",np.char.find(stns[STN_NAME],"Montana") != -1)]
    create_kml_pts("/projects/daymet2/cory_lab/mt_stns.kml",stns[LON],stns[LAT],stns[STN_NAME])

def error_stats_tair():
    
    ds_tmin = Dataset('/projects/daymet2/station_data/infill/infill_tair/infill_tmin.nc')
    ds_tmax = Dataset('/projects/daymet2/station_data/infill/infill_tair/infill_tmax.nc')
    
    ids_tmin = np.array(ds_tmin.variables['stn_id'][:],dtype="<S16")
    ids_tmax = np.array(ds_tmax.variables['stn_id'][:],dtype="<S16")
    
    mae_tmin = ds_tmin.variables['mae'][:]
    mae_tmax = ds_tmax.variables['mae'][:]
    
    bias_tmin = ds_tmin.variables['bias'][:]
    bias_tmax = ds_tmax.variables['bias'][:]
    
    mask_ghcn_tmin = np.char.startswith(ids_tmin, "GHCN")
    mask_ghcn_tmax = np.char.startswith(ids_tmax, "GHCN")
    
    mask_snotel_tmin = np.char.startswith(ids_tmin, "SNOTEL")
    mask_snotel_tmax = np.char.startswith(ids_tmax, "SNOTEL")

    mask_raws_tmin = np.char.startswith(ids_tmin, "RAWS")
    mask_raws_tmax = np.char.startswith(ids_tmax, "RAWS")
    
    print "TMIN OVERALL MAE: ",str(np.mean(mae_tmin))
    print "TMIN GHCN MAE: ",str(np.mean(mae_tmin[mask_ghcn_tmin]))
    print "TMIN SNOTEL MAE: ",str(np.mean(mae_tmin[mask_snotel_tmin]))
    print "TMIN RAWS MAE: ",str(np.mean(mae_tmin[mask_raws_tmin]))
    print ""
    print "TMIN OVERALL BIAS: ",str(np.mean(bias_tmin))
    print "TMIN GHCN BIAS: ",str(np.mean(bias_tmin[mask_ghcn_tmin]))
    print "TMIN SNOTEL BIAS: ",str(np.mean(bias_tmin[mask_snotel_tmin]))
    print "TMIN RAWS BIAS: ",str(np.mean(bias_tmin[mask_raws_tmin]))
    print ""
    print "TMAX OVERALL MAE: ",str(np.mean(mae_tmax))
    print "TMAX GHCN MAE: ",str(np.mean(mae_tmax[mask_ghcn_tmax]))
    print "TMAX SNOTEL MAE: ",str(np.mean(mae_tmax[mask_snotel_tmax]))
    print "TMAX RAWS MAE: ",str(np.mean(mae_tmax[mask_raws_tmax]))
    print ""
    print "TMAX OVERALL BIAS: ",str(np.mean(bias_tmax))
    print "TMAX GHCN BIAS: ",str(np.mean(bias_tmax[mask_ghcn_tmax]))
    print "TMAX SNOTEL BIAS: ",str(np.mean(bias_tmax[mask_snotel_tmax]))
    print "TMAX RAWS BIAS: ",str(np.mean(bias_tmax[mask_raws_tmax]))         
    

    
def test_infill_po_norms():
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    days = stn_da.days
    
    stn_id = 'GHCN_USC00027281'
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    
    fit_po,obs_po = infill_po(stn_id, stn_da, days, mth_masks, mthbuf_masks)
    print fit_po
    plt.plot(fit_po,".")
    plt.show()

def test_infill_prcp_by_mth():
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_id = 'SNOTEL_20G12S'#'GHCN_USC00207690'#'SNOTEL_13C01S'#'SNOTEL_20G12S'
    nnghs = 34
    mth_masks = build_mth_masks(stn_da.days)
    
    ds_po = Dataset('/projects/daymet2/station_data/infill/normals_po.nc')
    ds_prcp = Dataset('/projects/daymet2/station_data/infill/normals_prcp.nc')
    
    yr_mth_masks = build_yr_mth_masks(stn_da.days)
    
    apca_matrix = pca_matrix_po2_mth(stn_id, stn_da, ds_po, ds_prcp, yr_mth_masks)
    
    fit_prcp_dly = infill_po_only(stn_id, stn_da, nnghs, mth_masks,ds_prcp)
    fit_po_dly = np.array(fit_prcp_dly > 0,dtype=np.int)
    
    fit_po, obs_po, npcs, nnghs, max_dist = apca_matrix.infill_po(nnghs)
    
    fit_po_dly_mth = apca_matrix.calc_prcp_mth_totals(fit_po_dly, yr_mth_masks)
    fit_po_mth = apca_matrix.calc_prcp_mth_totals(fit_po, yr_mth_masks)
    obs_po_mth = apca_matrix.calc_prcp_mth_totals(obs_po, yr_mth_masks)
    
    plt.plot(fit_po_dly_mth)
    plt.plot(fit_po_mth)
    plt.plot(obs_po_mth)
    plt.show()
    
    print npcs,nnghs,max_dist
    
    fin_mask = np.isfinite(obs_po)
    
    print calc_hss(obs_po[fin_mask], fit_po[fin_mask])
    print calc_hss(obs_po[fin_mask], fit_po_dly[fin_mask])
 
    fit_po[fin_mask] = obs_po[fin_mask]
    
    fit_po = np.array(fit_po,dtype=np.bool)
    
    fit_prcp, obs_prcp, npcs, nnghs, max_dist = apca_matrix.infill_prcp(nnghs, fit_po)
    
    plt.plot(fit_prcp_dly)
    #plt.plot(fit_prcp)
    plt.plot(obs_prcp)
    plt.show()
 
    print npcs,nnghs,max_dist
    
    perr_ttlamt = ((np.sum(fit_prcp_dly[fin_mask]) - np.sum(obs_prcp[fin_mask])) / np.sum(obs_prcp[fin_mask])) * 100.
    print perr_ttlamt

def TEST_NNRPCA_INFILL_TAIR():
    
    stn_da = station_data_ncdb("/Users/jaredwo/Downloads/wxtopo_data/all.nc")
    stn_id = 'GHCN_USC00247286'
    print stn_da.stns[stn_da.stn_ids==stn_id]
    tair_var = 'tmin'
    ntrain_yrs = 5
    xval_mask = xval.build_xval_masks(np.array([stn_id]), ntrain_yrs, stn_da,tair_var)[0]
    tair = stn_da.load_all_stn_obs_var(stn_id,tair_var)[0]
    
    source_r('/Users/jaredwo/Documents/workspace/wxTopo_R/imputation.R')
    
    nnr_path = "/Users/jaredwo/Downloads/wxtopo_data/ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/"
    min_date = datetime(1948,1,1)
    max_date = datetime(2012,6,30)
    n_ngh = 9
    nnr_vars = ["air",'shum','hgt','uwnd','vwnd','omega']
    levels = [3,4,5]
    nnr_levels = (levels,levels,levels,levels,levels,levels)    
    nnr_ds = nnr_pca_ds(nnr_path, min_date, max_date, n_ngh, nnr_vars, nnr_levels,min_pcvar=0.99)
    
    a_mat = nnrpca_matrix(stn_id, stn_da, tair_var, nnr_ds, n_ngh, xval_mask)
    fit_tair, obs_tair = a_mat.infill()
    
    difs = fit_tair[xval_mask] - tair[xval_mask]
    mae_tmin = np.mean(np.abs(difs))
    bias_tmin = np.mean(difs)
    var_tmin_obs = np.var(tair[xval_mask],ddof=1)
    var_tmin_fit = np.var(fit_tair[xval_mask],ddof=1)
    
    print "|".join(["TMIN MAE/BIAS",str(mae_tmin),str(bias_tmin)])
    print "|".join(["TMIN OBS/FIT VAR",str(var_tmin_obs),str(var_tmin_fit),str(var_tmin_fit/var_tmin_obs)])
    
    plt.boxplot((tair[xval_mask],fit_tair[xval_mask]))
    plt.show()
    
    plt.subplot(211)
    plt.plot(tair[xval_mask])
    ylim = plt.ylim()
    plt.subplot(212)
    plt.plot(fit_tair[xval_mask])
    plt.ylim(ylim)
    plt.show()
    
    #############################

    ####################
    plt.plot(tair)
    plt.plot(fit_tair)
    plt.show()

def load_test_pca():
    stn_da = station_data_infill("/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc","tmin")
    stn_id = 'SNOTEL_13A19S'
    tdi_rast = input_raster('/projects/daymet2/dem/topo_disect_msd.tif')
    #modeler = modeler_clib(stn_da.days)
    modeler = modeler_tair_mean(stn_da.days)
    #interper = interp_tair(modeler)
    
    #lst_rast = input_raster('/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs/mosaic_mean.tif')
    lst_rast = input_raster('/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs/mosaic_mean_gdal.tif')
    sr_sin = osr.SpatialReference()
    sr_sin.ImportFromProj4(PROJ4_MODIS)
    sr_wgs84 = osr.SpatialReference()
    sr_wgs84.ImportFromEPSG(EPSG_WGS84)
    trans_wgs84_to_sin = osr.CoordinateTransformation(sr_wgs84,sr_sin)
    lst_rast.coordTrans_wgs84_to_src = trans_wgs84_to_sin
    
    MIN_STNS = 28
    
    a_pt = stn_da.stns[stn_da.stn_ids==stn_id]
    print a_pt
    pt_struct = np.empty(1,dtype=[(LON, np.float64), (LAT, np.float64), (ELEV, np.float64),(TDI, np.float64),(LST, np.float64),(MEAN_OBS,np.float64)])
    pt_struct[LON] = a_pt[LON]
    pt_struct[LAT] = a_pt[LAT]
    pt_struct[ELEV] = a_pt[ELEV]
    pt_struct[TDI] = tdi_rast.getDataValue(a_pt[LON][0], a_pt[LAT][0])
    pt_struct[LST] = lst_rast.getDataValue(a_pt[LON][0], a_pt[LAT][0])
    pt_struct[MEAN_OBS] = a_pt[MEAN_OBS]
    
    a_pt = pt_struct[0]
    print a_pt
    obs = stn_da.load_obs(stn_id)
    
    stn_slct = station_select(stn_da.stns,MIN_STNS , MIN_STNS+10, AREA_MONTANA_BUFFER)
    interp_stns, wgt, rad = stn_slct.get_interp_stns(a_pt[LAT], a_pt[LON],np.concatenate((RM_STN_IDS_TAIR,np.array([stn_id]))))
    
    tdi = []
    lst = []
    for x in np.arange(interp_stns.size):
        tdi.append(tdi_rast.getDataValue(interp_stns[LON][x], interp_stns[LAT][x]))
        lst.append(lst_rast.getDataValue(interp_stns[LON][x], interp_stns[LAT][x]))
    tdi = np.array(tdi)
    lst = np.array(lst)
    
    interp_stns2 = np.empty(interp_stns.size,dtype=[(STN_ID,"<S16"),(LON, np.float64), (LAT, np.float64), (ELEV, np.float64),(TDI, np.float64),(LST, np.float64),(MEAN_OBS,np.float64)])
    interp_stns2[STN_ID] = interp_stns[STN_ID]
    interp_stns2[LON] = interp_stns[LON]
    interp_stns2[LAT] = interp_stns[LAT]
    interp_stns2[ELEV] = interp_stns[ELEV]
    interp_stns2[TDI] = tdi
    interp_stns2[LST] = lst
    interp_stns2[MEAN_OBS] = interp_stns[MEAN_OBS]
    interp_stns = interp_stns2
    
    X = np.column_stack((interp_stns2[ELEV],interp_stns2[TDI],interp_stns2[LST]))
    
#    robjects.globalenv["X"] = X
#    robjects.globalenv["wgt"] = wgt
#    r("save.image('/projects/daymet2/rdata/SNOTEL_13A19S.Rdata')")
    
    return X,wgt

def TEST_GWPCA():
    
    A = np.genfromtxt('/Users/jaredwo/Downloads/wxtopo_data/X.csv',delimiter=",",skip_header=1,usecols=(1,2,3))
    wgt = np.genfromtxt('/Users/jaredwo/Downloads/wxtopo_data/wgt.csv',delimiter=",",skip_header=1,usecols=(1))
    
    aclib = clib_wxTopo('/Users/jaredwo/Documents/workspace/wxTopo_C/Release/libwxTopo_C')
   
    pc_loads, pc_scores, var_explain, error = aclib.pca_gwpca(A, wgt)
    print pc_loads
    print np.cumsum(var_explain)
    print pc_scores

def TEST_INTERP_TMAX():
    
    stn_da = station_data_infill("/projects/daymet2/station_data/infill/impute_tair/serial_tmax.nc","tmax")
    
    dup_stn_ids = np.loadtxt("/projects/daymet2/station_data/infill/impute_tair/dup_tmax_stns_rm.txt",dtype="<S16")
    rm_stnids = np.concatenate([dup_stn_ids,RM_STN_IDS_TAIR])
    
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[TDI]),np.logical_not(np.in1d(stn_da.stns[STN_ID], rm_stnids, True)))
    stns = stn_da.stns[stn_mask]
    
    
    modeler = modeler_Rkrig('/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C')
    #modeler = modeler_tair_mean(stn_da.days,'/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C')
    #modeler = modeler_clib(stn_da.days,'/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C')
    nnghs = 37
    stn_slct = station_select(stns, stn_da, nnghs, nnghs,rm_zero_dist_stns=True)
    
    stn_id = 'GHCN_USC00489459'#GHCN_USC00040693'
    a_pt = stn_da.stns[stn_da.stn_ids==stn_id]
#    a_pt[ELEV] = 3050.
#    a_pt[TDI] = 5.0
#    a_pt[LST] = -8
    print a_pt
    
    obs_tair = stn_da.load_obs(stn_id)
    
    interp_stns, stn_obs, wgt = stn_slct.get_interp_stns(a_pt[LAT], a_pt[LON], np.array([stn_id]))
    
    tair = modeler.model(interp_stns, stn_obs, a_pt)[0]
    plt.plot(tair)
    plt.show()
    #tair = modeler.model(interp_stns, wgt, stn_obs, a_pt)[0]
    bias = np.mean(tair - obs_tair)
    mae = np.mean(np.abs(tair - obs_tair))
    
    plt.plot(obs_tair,tair,'.')
    
    xlim = plt.xlim()
    aline = np.arange(xlim[0],xlim[1]+.1,.1)
    plt.plot(aline,aline)
    plt.show()
    
    print mae,bias

def TEST_INTERP_TMINTMAX():
    nnghs =35
    #stn_id = 'SNOTEL_13C01S'       
    #pt_lon,pt_lat, = -112.632906,48.492764   
    pt_lon,pt_lat, = -112.940651,48.500218
    
    ds_elev = ncdf_raster('/projects/daymet2/dem/interp_grids/ncdf/elev.nc','elev')
    ds_tdi = ncdf_raster('/projects/daymet2/dem/interp_grids/ncdf/tdi.nc','tdi')
    ds_lsttmin = ncdf_raster('/projects/daymet2/dem/interp_grids/ncdf/lst_tmin.nc','lst_tmin')
    ds_lsttmax = ncdf_raster('/projects/daymet2/dem/interp_grids/ncdf/lst_tmax.nc','lst_tmax')
    
    x,y = ds_elev.getGridCellOffset(pt_lon, pt_lat)
    pt_lat,pt_lon = ds_elev.lats[y],ds_elev.lons[x]
    pt_elev = ds_elev.getDataValue(pt_lon, pt_lat)
    pt_tdi = ds_tdi.getDataValue(pt_lon, pt_lat)
    pt_lsttmin = ds_lsttmin.getDataValue(pt_lon, pt_lat)
    pt_lsttmax = ds_lsttmax.getDataValue(pt_lon, pt_lat)
    
    stn_da_tmin = station_data_infill("/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc","tmin")
    stn_da_tmax = station_data_infill("/projects/daymet2/station_data/infill/impute_tair/serial_tmax.nc","tmax")
    
    dups_tmin = np.loadtxt("/projects/daymet2/station_data/infill/impute_tair/dup_tmin_stns_rm.txt",dtype="<S16")
    dups_tmax = np.loadtxt("/projects/daymet2/station_data/infill/impute_tair/dup_tmin_stns_rm.txt",dtype="<S16")
    
    rm_stnids_tmin = np.concatenate([dups_tmin,RM_STN_IDS_TAIR])
    rm_stnids_tmax = np.concatenate([dups_tmax,RM_STN_IDS_TAIR])
   
    mask_stns_tmin = np.logical_not(np.in1d(stn_da_tmin.stns[STN_ID],rm_stnids_tmin,assume_unique=True))
    mask_stns_tmax = np.logical_not(np.in1d(stn_da_tmax.stns[STN_ID],rm_stnids_tmax,assume_unique=True))
    stn_slct_tmin = station_select(stn_da_tmin.stns[mask_stns_tmin], stn_da_tmin, nnghs, nnghs,rm_zero_dist_stns=True)
    stn_slct_tmax = station_select(stn_da_tmax.stns[mask_stns_tmax], stn_da_tmax, nnghs, nnghs,rm_zero_dist_stns=True)
    #sys.exit()
#    stns_tmin = stn_da_tmin.stns[mask_stns_tmin]
#    stns_df = {LON:stns_tmin[LON],LAT:stns_tmin[LAT],ELEV:stns_tmin[ELEV],TDI:stns_tmin[TDI],LST:stns_tmin[LST],NEON:stns_tmin[NEON],'tair':stns_tmin[MEAN_OBS]}
#    stns_df = robjects.DataFrame(stns_df)
#    robjects.globalenv["stns_tmin"] = stns_df
#    r("save.image('/projects/daymet2/rdata/all_stns_tmin.Rdata')")
#    sys.exit()
    
#    modC = modeler_Rkrig(stn_da_tmin.days, '/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C',
#                         '/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/gwrk.R')
    
    modC = modeler_RkrigMean(stn_da_tmin.days, '/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C',
                         '/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R')   
        
    interper = interp_tair(modC)
    ainput = tair_input()
    aoutput = tair_output()
    
#    a_pt_tmin = stn_da_tmin.stns[stn_da_tmin.stn_ids==stn_id]
#    a_pt_tmax = stn_da_tmax.stns[stn_da_tmax.stn_ids==stn_id]
#    
#    pt_lat,pt_lon = a_pt_tmin[LAT],a_pt_tmin[LON]
#    pt_elev = a_pt_tmin[ELEV]
#    pt_tdi = a_pt_tmin[TDI]
#    pt_lsttmin = a_pt_tmin[LST]
#    pt_lsttmax = a_pt_tmax[LST]
    
#    obs_tmin = stn_da_tmin.load_obs(stn_id)
#    obs_tmax = stn_da_tmax.load_obs(stn_id)
    
    stns_tmin,stn_obs_tmin,stn_dists_tmin = stn_slct_tmin.get_interp_stns(pt_lat, pt_lon)#np.array([stn_id]))
    stns_tmax,stn_obs_tmax,stn_dists_tmax = stn_slct_tmax.get_interp_stns(pt_lat, pt_lon)#np.array([stn_id]))
    
    ainput.set_pt(pt_lon, pt_lat, pt_elev, pt_tdi, pt_lsttmin, pt_lsttmax)
    ainput.stns_tmin = stns_tmin
    ainput.stn_obs_tmin = stn_obs_tmin
    ainput.stn_dists_tmin = stn_dists_tmin
    ainput.stns_tmax = stns_tmax
    ainput.stn_obs_tmax = stn_obs_tmax
    ainput.stn_dists_tmax = stn_dists_tmax

    interper.model_tmin_tmax(ainput, aoutput)
    
#    bias_tmin = np.mean(aoutput.tmin - obs_tmin)
#    mae_tmin = np.mean(np.abs(aoutput.tmin - obs_tmin))
#    
#    bias_tmax = np.mean(aoutput.tmax - obs_tmax)
#    mae_tmax = np.mean(np.abs(aoutput.tmax - obs_tmax))
#
    mae_tmin,bias_tmin,mae_tmax,bias_tmax = ['NA']*4
    print np.mean(aoutput.tmin),np.mean(aoutput.tmax)
    print "|".join(["TMIN MAE|BIAS|STDERR|CI: ",str(mae_tmin),str(bias_tmin),str(aoutput.se_tmin),str(aoutput.ci_tmin)])
    print "|".join(["TMAX MAE|BIAS|STDERR|CI: ",str(mae_tmax),str(bias_tmax),str(aoutput.se_tmax),str(aoutput.ci_tmax)])
    print "n days TMIN >= TMAX: "+str(aoutput.ninvalid)
    
    plt.plot(aoutput.tmin)
    plt.plot(aoutput.tmax)
    plt.show()

def TEST_INTERP_TAIR():
    
    stn_da = station_data_infill("/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc","tmin",stn_dtype=DTYPE_STN_MEAN_LSTMTHS_TDI)
    
    dup_stn_ids = np.loadtxt("/projects/daymet2/station_data/infill/impute_tair/dup_tmin_stns_rm.txt",dtype="<S16")
    rm_stnids = np.concatenate([dup_stn_ids,RM_STN_IDS_TAIR])
    
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[TDI]),np.logical_not(np.in1d(stn_da.stns[STN_ID], rm_stnids, True)))
    stns = stn_da.stns[stn_mask]
#    stns_neon = stns[np.isfinite(stns[NEON])]
#    stns_df = {LON:stns_neon[LON],LAT:stns_neon[LAT],ELEV:stns_neon[ELEV],TDI:stns_neon[TDI],LST:stns_neon[LST],NEON:stns_neon[NEON],'tair':stns_neon[MEAN_OBS]}
#    stns_df = robjects.DataFrame(stns_df)
    
    min_nghs = 50
    max_nghs = 60
    
    stn_slct = station_select(stns, stn_da, min_nghs, max_nghs,rm_zero_dist_stns=True)
    modeler = modeler_RkrigMean('/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C',
                                '/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R',1.4,0,stn_slct)

    #neon_nums,nugs,psills,rngs,kappa
    #neon_varios = np.array(r.build_neon_varios(max_nghs,stns_df))
    #print neon_varios
    
    stn_id = 'SNOTEL_13C01S'#'GHCN_USC00140201'#GHCN_USC00040693'
    a_pt = stn_da.stns[np.nonzero(stn_da.stn_ids==stn_id)[0][0]]
    print a_pt
    
    #interp_stns, stn_obs, stn_dists = stn_slct.get_interp_stns(a_pt[LAT], a_pt[LON], np.array([stn_id]),False)
    #vario_stns = stn_slct_vario.get_interp_stns(a_pt[LAT], a_pt[LON], np.array([stn_id]),False)[0]
    
    #tair = modeler.model_bymth(interp_stns, stn_obs, a_pt)[0]
    #tair,std_err,ci = modeler.model(interp_stns, stn_dists, a_pt, neon_varios)
    
    tair,std_err,ci = modeler.model_local(a_pt,np.array([a_pt[STN_ID]]))
    
    #tair,std_err,ci = modeler.model_local(vario_stns, interp_stns, stn_dists, a_pt)
    
    bias = tair - a_pt[MEAN_OBS]
    mae = np.abs(bias)
    
    print tair,std_err,ci
    print mae,bias

def TEST_INTERP_TAIR_NNR():
    stn_da = station_data_infill("/projects/daymet2/station_data/infill/impute_tair/serial_nnrdif_tmax.nc","tmax")
    stn_da_xval = station_data_infill("/projects/daymet2/station_data/infill/impute_tair/serial_tmax.nc","tmax")
    
    dup_stn_ids = np.loadtxt("/projects/daymet2/station_data/infill/impute_tair/dup_tmax_stns_rm.txt",dtype="<S16")
    rm_stnids = np.concatenate([dup_stn_ids,RM_STN_IDS_TAIR])
    
    stn_mask = np.logical_and(stn_da.stns[TDI] != np.nan,np.logical_not(np.in1d(stn_da.stns[STN_ID], rm_stnids, True)))
    stns = stn_da.stns[stn_mask]
    
    ds_nnr = NNRds('/projects/daymet2/reanalysis_data/tair/nnr_interp_tmax.nc','tmax',(19480101,20111231))
    #modeler = modeler_tair_mean(stn_da.days,'/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C')
    #modeler = modeler_clib(stn_da.days,'/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C')
    modeler = modeler_Rkrig('/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C')
    
    nnghs = 37
    stn_slct = station_select(stns, stn_da, nnghs, nnghs,rm_zero_dist_stns=True)
    stn_id = 'GHCN_USC00489459'#'SNOTEL_10D12S'#'SNOTEL_10D12S''SNOTEL_20G12S'#'GHCN_USC00207690'#'SNOTEL_13C01S'#'SNOTEL_20G12S'
    a_pt = stn_da.stns[stn_da.stn_ids==stn_id]
    print a_pt
    
    obs_tair = stn_da_xval.load_obs(stn_id)
    pt_fair = ds_nnr.interp_variable(a_pt[LON], a_pt[LAT], a_pt[ELEV])
    #pt_mfair = np.mean(pt_fair)
    
    interp_stns, stn_obs, wgt = stn_slct.get_interp_stns(a_pt[LAT], a_pt[LON], np.array([stn_id]))
    
    fair_difs = modeler.model(interp_stns, stn_obs, a_pt)[0]
    tair = pt_fair + fair_difs
    #tair_mean = pt_mfair + fair_dif
    
    bias = np.mean(tair - obs_tair)
    mae = np.mean(np.abs(tair - obs_tair))
    
    print mae,bias
    

def TEST_PCACOR_IMPUTE_TAIR():
    startend_ymd=(19480101,20111231)
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc",startend_ymd=startend_ymd)
    stn_id = 'GLAC_7'#'SNOTEL_13A19S'#'RAWS_MHHS'#'RAWS_IPOW'#GHCN_USC00244558'#GHCN_USC00244558'
    ntrain_yrs = 1
    nnghs = 3
    print stn_da.stns[stn_da.stn_ids==stn_id]
    
    
    source_r('/nfshome/jared.oyler/ecl_helios_workspace/wxTopo_R/pca_infill.R')
    source_r('/nfshome/jared.oyler/ecl_helios_workspace/wxTopo_R/imputation.R')
    tair_var = 'tmin'
    
    por = load_por_csv("/projects/daymet2/station_data/all/all_por_1948_2011.csv")
    mask_por_tmin,mask_por_tmax,mask_por_prcp = build_valid_por_masks(por)
    stn_masks = {'tmin':mask_por_tmin,'tmax':mask_por_tmax,'prcp':mask_por_prcp}
    
    ds_norms = Dataset('/projects/daymet2/station_data/infill/normals_tair_1948_2011.nc')
    norms_tmin = ds_norms.variables['norm'][0,:].data
    norms_tmax = ds_norms.variables['norm'][1,:].data
    norms = {'tmin':norms_tmin,'tmax':norms_tmax}
    va_tmin = ds_norms.variables['var'][0,:].data
    va_tmax = ds_norms.variables['var'][1,:].data
    va = {'tmin':va_tmin,'tmax':va_tmax}
    ds_nnr_tmin = NNRds('/projects/daymet2/reanalysis_data/tair/nnr_tmin.nc', 'tmin',startend_ymd)
    ds_nnr_tmax = NNRds('/projects/daymet2/reanalysis_data/tair/nnr_tmax.nc','tmax',startend_ymd)
    ds_nnr = {'tmin':ds_nnr_tmin,'tmax':ds_nnr_tmax}
    
    #xval_mask = xval.build_xval_masks(np.array([stn_id]), ntrain_yrs, stn_da,tair_var)[0]
    xval_mask = None
    
    tair = stn_da.load_all_stn_obs_var(stn_id,tair_var)[0]
    
    if xval_mask is None:
        fin_mask = np.isfinite(tair)
    else:
        fin_mask = np.logical_and(np.logical_not(xval_mask),np.isfinite(tair))
        
        
    fit_norm,masked_norm = impute_tair_norm(stn_id, stn_da, stn_masks[tair_var], tair_var,ds_nnr[tair_var],tair_mask=xval_mask)
    #fit_tair,masked_tair = infill_tair(stn_id, stn_da, days, tair_var, mth_masks, mthbuf_masks, tair_mask)
    mask_nan = np.isnan(masked_norm)
    fnl_norm = np.copy(masked_norm)
    fnl_norm[mask_nan] = fit_norm[mask_nan]
    
    norm_est = np.mean(fnl_norm)
    va_est = np.var(fnl_norm,ddof=1)
    
    norms_pca = np.copy(norms[tair_var])
    norms_pca[stn_da.stn_ids==stn_id] = norm_est
    
    va_pca = np.copy(va[tair_var])
    va_pca[stn_da.stn_ids==stn_id] = va_est
    
    a_matrix = ImputeMatrixPCA(stn_id, stn_da, tair_var, norms_pca,ds_nnr[tair_var],tair_mask=xval_mask)
    
#    for x in np.arange(a_matrix.pca_tair.shape[1]):
#        plt.plot(a_matrix.pca_tair[:,x])
#        plt.show()
        
    
    #a_matrix = impute_matrix(stn_id, stn_da, stn_masks[tair_var], tair_var,tair_mask=xval_mask)
    fit_tair, obs_tair, npcs, fnl_nnghs, max_dist = a_matrix.impute(nnghs=nnghs)
    
    if xval_mask is not None:
        difs = fit_tair[xval_mask] - tair[xval_mask]
        mae_tmin = np.mean(np.abs(difs))
        bias_tmin = np.mean(difs)
        var_tmin_obs = np.var(tair[xval_mask],ddof=1)
        var_tmin_fit = np.var(fit_tair[xval_mask],ddof=1)
        
        print "|".join(["TMIN MAE/BIAS",str(mae_tmin),str(bias_tmin)])
        print "|".join(["TMIN OBS/FIT VAR",str(var_tmin_obs),str(var_tmin_fit),str(var_tmin_fit/var_tmin_obs)])
        
        plt.boxplot((tair[xval_mask],fit_tair[xval_mask]))
        plt.show()
        
        plt.subplot(211)
        plt.plot(tair[xval_mask])
        ylim = plt.ylim()
        plt.subplot(212)
        plt.plot(fit_tair[xval_mask])
        plt.ylim(ylim)
        plt.show()
        
        plt.plot(tair[xval_mask],fit_tair[xval_mask],".")
        xlim = plt.xlim()
        aline = np.arange(xlim[0],xlim[1]+.1,.1)
        plt.plot(aline,aline)
        plt.show()
    
    #############################

    difs = fit_tair[fin_mask] - tair[fin_mask]
    mae_tair = np.mean(np.abs(difs))
    bias_tair = np.mean(difs)
    var_tair_obs = np.var(tair[fin_mask],ddof=1)
    var_tair_fit = np.var(fit_tair[fin_mask],ddof=1)
    
    fnl_tair = np.copy(obs_tair)
    fnl_tair[np.isnan(obs_tair)] = fit_tair[np.isnan(obs_tair)]
    
    print np.mean(fnl_tair),np.var(fnl_tair,ddof=1)
    print "|".join(["TMIN MAE/BIAS",str(mae_tair),str(bias_tair)])
    print "|".join(["TMIN OBS/FIT VAR",str(var_tair_obs),str(var_tair_fit),str(var_tair_fit/var_tair_obs)])
    
    plt.boxplot((tair[fin_mask],fit_tair[fin_mask]))
    plt.show()
    
    plt.subplot(211)
    plt.plot(tair[fin_mask])
    ylim = plt.ylim()
    plt.subplot(212)
    plt.plot(fit_tair[fin_mask])
    plt.ylim(ylim)
    plt.show()

    ####################
    plt.plot(tair)
    plt.plot(fit_tair)
    plt.show()

def TEST_MICROMET_IMPUTE_TAIR():
    startend_ymd=(19480101,20111231)
    #stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc",startend_ymd=startend_ymd)
    stn_id = 'USFS_L9900300825'#USFS_L9900300825
    ntrain_yrs = 1
    
    
    usfs = insert_usfs("/projects/daymet2/station_data/usfs_micro/", startend_ymd[0], startend_ymd[1])
    stn_da = station_data_combine('/projects/daymet2/station_data/all/all.nc', usfs,startend_ymd=startend_ymd)
    print stn_da.stns[stn_da.stn_ids==stn_id]
    
    source_r('/nfshome/jared.oyler/ecl_helios_workspace/wxTopo_R/pca_infill.R')
    source_r('/nfshome/jared.oyler/ecl_helios_workspace/wxTopo_R/imputation.R')
    tair_var = 'tmin'
    

    
    ds_norms = Dataset('/projects/daymet2/station_data/infill/normals_tair_1948_2011.nc')
    norms_tmin = ds_norms.variables['norm'][0,:].data
    norms_tmax = ds_norms.variables['norm'][1,:].data
    norms_stnid = np.array(ds_norms.variables['stn_id'][:], dtype="<S16")
    norms = {'tmin':norms_tmin,'tmax':norms_tmax}
    
    va_tmin = ds_norms.variables['var'][0,:].data
    va_tmax = ds_norms.variables['var'][1,:].data
    va = {'tmin':va_tmin,'tmax':va_tmax}
    
    
    por = load_por_csv("/projects/daymet2/station_data/all/all_por_1948_2011.csv")
    mask_por_tmin,mask_por_tmax,mask_por_prcp = build_valid_por_masks(por,ntrain_yrs)
    
    mask_tmin = np.ones(stn_da.stn_ids.size,dtype=np.bool)
    mask_tmax = np.ones(stn_da.stn_ids.size,dtype=np.bool)
    mask_prcp = np.ones(stn_da.stn_ids.size,dtype=np.bool)
    
    mask_tmin[np.in1d(stn_da.stn_ids, norms_stnid,True)] = mask_por_tmin
    mask_tmax[np.in1d(stn_da.stn_ids, norms_stnid,True)] = mask_por_tmax
    mask_prcp[np.in1d(stn_da.stn_ids, norms_stnid,True)] = mask_por_prcp
    
    stn_masks = {'tmin':mask_tmin,'tmax':mask_tmax,'prcp':mask_prcp}

    ds_nnr_tmin = NNRds('/projects/daymet2/reanalysis_data/tair/nnr_tmin.nc', 'tmin',startend_ymd)
    ds_nnr_tmax = NNRds('/projects/daymet2/reanalysis_data/tair/nnr_tmax.nc','tmax',startend_ymd)
    ds_nnr = {'tmin':ds_nnr_tmin,'tmax':ds_nnr_tmax}
    
    #xval_mask = xval.build_xval_masks(np.array([stn_id]), ntrain_yrs, stn_da,tair_var)[0]
    xval_mask = None
    
    tair = stn_da.load_all_stn_obs_var(stn_id,tair_var)[0]
    
    if xval_mask is None:
        fin_mask = np.isfinite(tair)
    else:
        fin_mask = np.logical_and(np.logical_not(xval_mask),np.isfinite(tair))
        
        
    fit_norm,masked_norm = impute_tair_norm(stn_id, stn_da, stn_masks[tair_var], tair_var,ds_nnr[tair_var],tair_mask=xval_mask)

    mask_nan = np.isnan(masked_norm)
    fnl_norm = np.copy(masked_norm)
    fnl_norm[mask_nan] = fit_norm[mask_nan]
    
    norm_est = np.mean(fnl_norm)
    va_est = np.var(fnl_norm,ddof=1)
    
    norms_pca = np.zeros(stn_da.stn_ids.size)
    norms_pca[np.in1d(stn_da.stn_ids, norms_stnid,True)] = norms[tair_var]
    norms_pca[stn_da.stn_ids==stn_id] = norm_est
    
    va_pca = np.zeros(stn_da.stn_ids.size)
    va_pca[np.in1d(stn_da.stn_ids, norms_stnid,True)] = va[tair_var]
    va_pca[stn_da.stn_ids==stn_id] = va_est
    
    a_matrix = ImputeMatrixPCA(stn_id, stn_da, tair_var, norms_pca, va_pca,ds_nnr[tair_var],tair_mask=xval_mask)
    #a_matrix = impute_matrix(stn_id, stn_da, stn_masks[tair_var], tair_var,tair_mask=xval_mask)
    fit_tair, obs_tair, npcs, fnl_nnghs, max_dist = a_matrix.impute()
    
    if xval_mask is not None:
        difs = fit_tair[xval_mask] - tair[xval_mask]
        mae_tmin = np.mean(np.abs(difs))
        bias_tmin = np.mean(difs)
        var_tmin_obs = np.var(tair[xval_mask],ddof=1)
        var_tmin_fit = np.var(fit_tair[xval_mask],ddof=1)
        
        print "|".join(["TMIN MAE/BIAS",str(mae_tmin),str(bias_tmin)])
        print "|".join(["TMIN OBS/FIT VAR",str(var_tmin_obs),str(var_tmin_fit),str(var_tmin_fit/var_tmin_obs)])
        
        plt.boxplot((tair[xval_mask],fit_tair[xval_mask]))
        plt.show()
        
        plt.subplot(211)
        plt.plot(tair[xval_mask])
        ylim = plt.ylim()
        plt.subplot(212)
        plt.plot(fit_tair[xval_mask])
        plt.ylim(ylim)
        plt.show()
        
        plt.plot(tair[xval_mask],fit_tair[xval_mask],".")
        xlim = plt.xlim()
        aline = np.arange(xlim[0],xlim[1]+.1,.1)
        plt.plot(aline,aline)
        plt.show()
    
    #############################

    difs = fit_tair[fin_mask] - tair[fin_mask]
    mae_tair = np.mean(np.abs(difs))
    bias_tair = np.mean(difs)
    var_tair_obs = np.var(tair[fin_mask],ddof=1)
    var_tair_fit = np.var(fit_tair[fin_mask],ddof=1)
    
    fnl_tair = np.copy(obs_tair)
    fnl_tair[np.isnan(obs_tair)] = fit_tair[np.isnan(obs_tair)]
    
    print np.mean(fnl_tair),np.var(fnl_tair,ddof=1)
    print "|".join(["TMIN MAE/BIAS",str(mae_tair),str(bias_tair)])
    print "|".join(["TMIN OBS/FIT VAR",str(var_tair_obs),str(var_tair_fit),str(var_tair_fit/var_tair_obs)])
    
    plt.boxplot((tair[fin_mask],fit_tair[fin_mask]))
    plt.show()
    
    plt.subplot(211)
    plt.plot(tair[fin_mask])
    ylim = plt.ylim()
    plt.subplot(212)
    plt.plot(fit_tair[fin_mask])
    plt.ylim(ylim)
    plt.show()

    ####################
    plt.plot(tair)
    plt.plot(fit_tair)
    plt.show()

def TEST_GWRPCA_INFILL_TAIR():
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    days = stn_da.days
    stn_id = 'GHCN_USC00262780'#GHCN_USC00244558'#GHCN_USC00244558'
    ntrain_yrs = 5
    nngh = 3
    print stn_da.stns[stn_da.stn_ids==stn_id]

    mth_masks = build_mth_masks(stn_da.days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    
    ds_norms = Dataset('/projects/daymet2/station_data/infill/normals_tair.nc')
    norms = np.array(ds_norms.variables['norm'][:],dtype=np.float)
    norms_tmin = norms[0,:]
    norms_tmax = norms[1,:]
    ds_norms.close()
    source_r('/nfshome/jared.oyler/ecl_helios_workspace/wxTopo_R/pca_infill.R')
    #source_r('/Users/jaredwo/Documents/workspace/wxTopo_R/imputation.R')
    norms_tair = norms_tmin
    tair_var = 'tmin'
    
    
    xval_mask = xval.build_xval_masks(np.array([stn_id]), ntrain_yrs, stn_da,tair_var)[0]
    
    tmin = stn_da.load_all_stn_obs_var(stn_id,tair_var)[0]
    
    fit_norm,obs_norm = infill_tair(stn_id, stn_da, days,tair_var, mth_masks, mthbuf_masks, xval_mask) 
    
    fin_mask = np.isfinite(obs_norm)
    
    fnl_norm = np.copy(obs_norm)
    fnl_norm[np.isnan(obs_norm)] = fit_norm[np.isnan(obs_norm)]
    
    norm_est = np.mean(fnl_norm)
    norms_pca = np.copy(norms_tair)
    norms_pca[stn_da.stn_ids==stn_id] = norm_est
    
    a_pca_matrix = gwrpca_matrix(stn_id, stn_da,tair_var, norms_pca, tair_mask=xval_mask)
    #a_pca_matrix = pca_matrix(stn_id, stn_da,tair_var, norms_pca, tair_mask=xval_mask)
    #a_pca_matrix = ioapca_matrix(stn_id, stn_da, tair_var, norms_pca, tair_mask=xval_mask)
    fit_tmin, obs_tmin, npcs_tmin, fnl_nnghs_tmin, max_dist_tmin = a_pca_matrix.infill(nngh)
    
    difs = fit_tmin[xval_mask] - tmin[xval_mask]
    mae_tmin = np.mean(np.abs(difs))
    bias_tmin = np.mean(difs)
    var_tmin_obs = np.var(tmin[xval_mask],ddof=1)
    var_tmin_fit = np.var(fit_tmin[xval_mask],ddof=1)
    
    print "|".join(["TMIN MAE/BIAS",str(mae_tmin),str(bias_tmin)])
    print "|".join(["TMIN OBS/FIT VAR",str(var_tmin_obs),str(var_tmin_fit),str(var_tmin_fit/var_tmin_obs)])
    
    plt.boxplot((tmin[xval_mask],fit_tmin[xval_mask]))
    plt.show()
    
    plt.subplot(211)
    plt.plot(tmin[xval_mask])
    ylim = plt.ylim()
    plt.subplot(212)
    plt.plot(fit_tmin[xval_mask])
    plt.ylim(ylim)
    plt.show()
    
    #############################

    difs = fit_tmin[fin_mask] - tmin[fin_mask]
    mae_tmin = np.mean(np.abs(difs))
    bias_tmin = np.mean(difs)
    var_tmin_obs = np.var(tmin[fin_mask],ddof=1)
    var_tmin_fit = np.var(fit_tmin[fin_mask],ddof=1)
    
    print "|".join(["TMIN MAE/BIAS",str(mae_tmin),str(bias_tmin)])
    print "|".join(["TMIN OBS/FIT VAR",str(var_tmin_obs),str(var_tmin_fit),str(var_tmin_fit/var_tmin_obs)])
    
    plt.boxplot((tmin[fin_mask],fit_tmin[fin_mask]))
    plt.show()
    
    plt.subplot(211)
    plt.plot(tmin[fin_mask])
    ylim = plt.ylim()
    plt.subplot(212)
    plt.plot(fit_tmin[fin_mask])
    plt.ylim(ylim)
    plt.show()

    ####################
    plt.plot(tmin)
    plt.plot(fit_tmin)
    plt.show()


def TEST_INFILL_TAIR():
    #Fix this station: GHCN_USC00244193
    stn_da = station_data_ncdb("/Users/jaredwo/Downloads/wxtopo_data/all.nc")
    stn_id = 'GLAC_7'#'GHCN_USC00244558'#"GLAC_7"RAWS_MMIS
    print stn_da.stns[stn_da.stn_ids==stn_id]
    nngh_tmin = 20
    nngh_tmax = 22
    ds_norms = Dataset('/Users/jaredwo/Downloads/wxtopo_data/normals_tair.nc')
    norms = np.array(ds_norms.variables['norm'][:],dtype=np.float)
    norms_tmin = norms[0,:]
    norms_tmax = norms[1,:]
    ds_norms.close()
    source_r('/Users/jaredwo/Documents/workspace/wxTopo_R/pca_infill.R')

    #####################################
    #TMIN
    #####################################
    a_pca_matrix = gwrpca_matrix(stn_id, stn_da,'tmin', norms_tmin)
    fit_tmin, obs_tmin, npcs_tmin, fnl_nnghs_tmin, max_dist_tmin = a_pca_matrix.infill(nngh_tmin)

    fnl_tmin = np.copy(obs_tmin)
    fill_mask = np.isnan(fnl_tmin)
    fnl_tmin[fill_mask] = fit_tmin[fill_mask]

    fin_mask = np.logical_not(fill_mask)
    difs = fit_tmin[fin_mask] - obs_tmin[fin_mask]
    mae_tmin = np.mean(np.abs(difs))
    bias_tmin = np.mean(difs)
    var_tmin_obs = np.var(obs_tmin[fin_mask],ddof=1)
    var_tmin_fit = np.var(fit_tmin[fin_mask],ddof=1)
    
    print "|".join(["TMIN MAE/BIAS",str(mae_tmin),str(bias_tmin)])
    print "|".join(["TMIN OBS/FIT VAR",str(var_tmin_obs),str(var_tmin_fit),str(var_tmin_fit/var_tmin_obs)])
    
    plt.boxplot((obs_tmin[fin_mask],fit_tmin[fin_mask]))
    plt.show()
    
    plt.subplot(121)
    plt.plot(obs_tmin[fin_mask])
    ylim = plt.ylim()
    plt.subplot(122)
    plt.plot(fit_tmin[fin_mask])
    plt.ylim(ylim)
    plt.show()
    
    plt.plot(fnl_tmin)
    plt.show()
    
    #####################################
    sys.exit()
    #####################################
    #TMAX
    #####################################
    a_pca_matrix = pca_matrix(stn_id, stn_da,'tmax', norms_tmax)
    fit_tmax, obs_tmax, npcs_tmax, fnl_nnghs_tmax, max_dist_tmax = a_pca_matrix.infill(nngh_tmax)

    fnl_tmax = np.copy(obs_tmax)
    fill_mask = np.isnan(fnl_tmax)
    fnl_tmax[fill_mask] = fit_tmax[fill_mask]

    fin_mask = np.logical_not(fill_mask)
    difs = fit_tmax[fin_mask] - obs_tmax[fin_mask]
    mae_tmax = np.mean(np.abs(difs))
    bias_tmax = np.mean(difs)
    var_tmax_obs = np.var(obs_tmax[fin_mask],ddof=1)
    var_tmax_fit = np.var(fit_tmax[fin_mask],ddof=1)
    #####################################

    print "|".join(["TMIN MAE/BIAS",str(mae_tmin),str(bias_tmin)])
    print "|".join(["TMIN OBS/FIT VAR",str(var_tmin_obs),str(var_tmin_fit),str(var_tmin_fit/var_tmin_obs)])
    print "|".join(["TMAX MAE/BIAS",str(mae_tmax),str(bias_tmax)])
    print "|".join(["TMAX OBS/FIT VAR",str(var_tmax_obs),str(var_tmax_fit),str(var_tmax_fit/var_tmax_obs)])

    fnl_tmin,fnl_tmax,ninvalid = tmin_tmax_fixer(fit_tmin, fit_tmax)
    print "".join([stn_id,": percent of obs tmin >= tmax: %.4f"%(ninvalid/float(stn_da.days.size)*100.,)])


def xval_stats_by_neon():
    
    neon_names = {0:'northeast',
                  2:'mid atlantic',
                  3:'southeast + atlantic neotropcial',
                  5:'great lakes',
                  6:'prairie peninsula',
                  7:'appalachians / cumberland plateau',
                  8:'ozarks complex',
                  9:'northern plains',
                  10:'central plains',
                  11:'southern plains',
                  12:'northern rockies',
                  13:'southern rockies / colorado plateau',
                  14:'desert southwest',
                  15:'great basin',
                  16:'pacific northwest',
                  17:'pacific southwest'}
    
    ds = Dataset('/projects/daymet2/station_data/infill/xval_infill_po_log10.nc')
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")    
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, ds.variables['stn_id'][:].astype("<S16"),True)]
    neon_rast = ncdf_raster('/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.nc', 'neon')

    nngh = ds.variables['nnghs'][:]
    hss = ds.variables['hss'][:].astype(np.float64)
    freq = ds.variables['freq_err'][:].astype(np.float64)
    amt = ds.variables['amt_err2'][:].astype(np.float64)
    intsy = ds.variables['intsy_err2'][:].astype(np.float64)
    
    mhss = np.mean(hss,axis=1)
    maxidx = np.nonzero(mhss == np.max(mhss))[0][0]
    

    neon = np.zeros(stns.size,dtype=np.int)
    for x in np.arange(stns.size):
        neon[x] = neon_rast.getDataValue(stns[LON][x],stns[LAT][x])
    
    uneon = np.unique(neon)
    
    for nrng in uneon:
        
        neon_mask = neon == nrng
        
        if nrng == 12:
            print stns[neon_mask]
        
        nhss = np.mean(hss[:,neon_mask],axis=1)
        
        nfreq = np.mean(freq[:,neon_mask],axis=1)
        nafreq = np.mean(np.abs(freq[:,neon_mask]),axis=1)
        
        namt = np.mean(amt[:,neon_mask],axis=1)
        naamt = np.mean(np.abs(amt[:,neon_mask]),axis=1)
        
        nintsy = np.mean(intsy[:,neon_mask],axis=1)
        naintsy= np.mean(np.abs(intsy[:,neon_mask]),axis=1)
        
        #maxidx = np.nonzero(nhss == np.max(nhss))[0][0]
        
        print "|".join([neon_names[nrng],str(nngh[maxidx]),str(nhss[maxidx]),str(nafreq[maxidx]),str(nfreq[maxidx]),
                        str(naamt[maxidx]),str(namt[maxidx]),str(naintsy[maxidx]),str(nintsy[maxidx])])
        
        
        
    
    #neon_rast.getDataValue(lons[x],lats[x])

def TEST_INFILL_PRCP():
    
    np.seterr(all='raise')
    np.seterr(under='ignore')

    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    ds_prcp = Dataset('/projects/daymet2/station_data/infill/normals_prcp_log10.nc')
    days = stn_da.days
    stn_id = "SNOTEL_13C01S"
    nnghs = 31
    norms_stnids = np.array(ds_prcp.variables['stn_id'][:], dtype="<S16")
    
    prcp = stn_da.load_all_stn_obs_var(stn_id, 'prcp')[0]
    obs_po = np.copy(prcp)
    obs_po[obs_po > 0] = 1

    a_pca_matrix = pca_matrix_prcp_transform(stn_id, stn_da, ds_prcp)
    
    fnl_norm = ds_prcp.variables['prcp'][:, np.nonzero(norms_stnids == stn_id)[0][0]]
    fnl_norm = backtransform_prcp(fnl_norm)
    
    fit_prcp, obs_prcp, npcs, actual_nnghs, max_dist = a_pca_matrix.infill(nnghs)
    
    fin_mask = np.isfinite(obs_prcp)
    nan_mask = np.logical_not(fin_mask)
    fin_fit_prcp = fit_prcp[fin_mask]
    fin_obs_po = obs_prcp[fin_mask] > 0
    
    max_hss = 0
    max_thres = 0
    for thres in PO_THRESHS:
        
        thres_mask = fin_fit_prcp >= thres
        
        hss = calc_hss(fin_obs_po, thres_mask)
        if hss > max_hss:
            max_hss = hss
            max_thres = thres
            print max_hss
    
    print max_hss,max_thres
    fit_po = fit_prcp >= max_thres
    
    fit_prcp1 = np.zeros(fit_po.size)
    fit_prcp2 = np.zeros(fit_po.size)
    #Get prcp amounts method 1: use amounts form prcp norm
    #######################################################
    norm_on_po = fnl_norm[fit_po]
    fit_prcp1[fit_po] = norm_on_po
    ######################################################
    #Get prcp amounts method 2: calc amounts using ppca
    fit_prcp_on_po = a_pca_matrix.infill(nnghs, fit_po)[0]
    fit_prcp_on_po[fit_prcp_on_po < 0.01] = 0.01 
    fit_prcp2[fit_po] = fit_prcp_on_po
    fit_prcp2[np.isfinite(obs_prcp)] = obs_prcp[np.isfinite(obs_prcp)]
    ######################################################
    plt.plot(fit_prcp2[fit_po]-fit_prcp1[fit_po])
    plt.show()
    
    print (np.sum(fit_prcp2[nan_mask]) - np.sum(fit_prcp1[nan_mask]))/np.sum(fit_prcp1[nan_mask])
    
    #Scale amts to the mean
    prcp_norm_mean = np.mean(fnl_norm,dtype=np.float64)
    
    #Only scale the daily amounts that were predicted
    scaler1 = ((prcp_norm_mean*fnl_norm.size) - np.sum(fit_prcp1[fin_mask]))/np.sum(fit_prcp1[nan_mask])
    scaler2 = ((prcp_norm_mean*fnl_norm.size) - np.sum(fit_prcp2[fin_mask]))/np.sum(fit_prcp2[nan_mask])
    fit_prcp1[nan_mask] = scaler1 * fit_prcp1[nan_mask]
    fit_prcp2[nan_mask] = scaler2 * fit_prcp2[nan_mask]
    
    print "TOTAL PRCP ",np.sum(fit_prcp1[nan_mask]),np.sum(fit_prcp2[nan_mask]) 
    
    ymax = np.max(np.concatenate([fit_prcp1[nan_mask],fit_prcp2[nan_mask]]))
    
    plt.subplot(211)
    plt.plot(fit_prcp1[nan_mask])
    plt.ylim((0,ymax))
    plt.subplot(212)
    plt.plot(fit_prcp2[nan_mask])
    plt.ylim((0,ymax))
    plt.show()
    
    yrs = np.unique(days[YEAR])
    ann_prcp_fit1 = np.zeros(yrs.size)
    ann_prcp_fit2 = np.zeros(yrs.size)
    
    for x in np.arange(yrs.size):
        
        yr_mask = days[YEAR] == yrs[x]
        ann_prcp_fit1[x] = np.sum(fit_prcp1[np.logical_and(nan_mask,yr_mask)])
        ann_prcp_fit2[x] = np.sum(fit_prcp2[np.logical_and(nan_mask,yr_mask)])
    
    plt.clf()
    plt.plot(ann_prcp_fit1)
    plt.plot(ann_prcp_fit2)
    plt.show()
    
    print np.var(fit_prcp1[nan_mask],ddof=1),np.var(fit_prcp2[nan_mask],ddof=1)
    
def TEST_INFILL_PRCP_XVAL():
    
    np.seterr(all='raise')
    np.seterr(under='ignore')

    stn_da = station_data_ncdb("/Users/jaredwo/Downloads/wxtopo_data/all.nc")
    ds_prcp = Dataset('/Users/jaredwo/Downloads/wxtopo_data/normals_prcp.nc')
    days = stn_da.days
    stn_id = "SNOTEL_13C01S"
    nmask = int(np.round(5*365.25))
    idxs = np.arange(stn_da.days.size)
    nnghs = 15
    mth_masks = build_mth_masks(stn_da.days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    norms_stnids = np.array(ds_prcp.variables['stn_id'][:], dtype="<S16")
    trans_func = lambda x: np.sqrt(x) 
    btrans_func = lambda x: np.square(x)
    
    #trans_func = lambda x: np.log10(x+0.01) 
    #btrans_func = lambda x: 10**(x)-0.01
    
    #trans_func = lambda x: x
    #btrans_func = lambda x: x
    
    #trans_func = lambda x: boxcox(x, .25)
    #btrans_func = lambda x: boxcox_inverse(x, .25)
    
    
    prcp = stn_da.load_all_stn_obs_var(stn_id, 'prcp')[0]
    m = np.mean(prcp[np.isfinite(prcp)])
    plt.subplot(211)
    plt.hist(np.log10(prcp[np.logical_and(np.isfinite(prcp),prcp > 0)]),bins=50)
    plt.subplot(212)
    plt.hist(np.sqrt(prcp[np.isfinite(prcp)]),bins=50)
    plt.show()
    
    obs_po = np.copy(prcp)
    obs_po[obs_po > 0] = 1
    #obs_po = obs_po > 0
    fin_mask = np.isfinite(prcp)
    last_idxs = np.nonzero(fin_mask)[0][-nmask:]
    xval_mask_prcp = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_mask)
    print np.sum(xval_mask_prcp)
    
    prcp_norm,obs_norm = infill_prcp_norm(stn_id, stn_da, days, mth_masks, mthbuf_masks, prcp_mask=xval_mask_prcp)
    fnl_norm = np.copy(obs_norm)
    fnl_norm[np.isnan(obs_norm)] = prcp_norm[np.isnan(obs_norm)]
    
    #fnl_norm = ds_prcp.variables['prcp'][:, np.nonzero(norms_stnids == stn_id)[0][0]]

    a_pca_matrix = pca_matrix_prcp_final(stn_id, stn_da, ds_prcp,prcp_norm=fnl_norm, prcp_mask=xval_mask_prcp,prcp_trans_funcs=(trans_func,btrans_func))
    
    fit_prcp, obs_prcp, npcs, actual_nnghs, max_dist = a_pca_matrix.infill(nnghs)
    
    fin_mask = np.isfinite(obs_prcp)
    fin_fit_prcp = fit_prcp[fin_mask]
    fin_obs_po = obs_prcp[fin_mask] > 0
    
    max_hss = 0
    max_thres = 0
    for thres in PO_THRESHS:
        
        thres_mask = fin_fit_prcp >= thres
        
        hss = calc_hss(fin_obs_po, thres_mask)
        if hss > max_hss:
            max_hss = hss
            max_thres = thres
            print max_hss
    
    print max_hss,max_thres
    fit_po = fit_prcp >= max_thres
    
    fit_prcp1 = np.zeros(fit_po.size)
    fit_prcp2 = np.zeros(fit_po.size)
    #Get prcp amounts method 1: use amounts form prcp norm
    #######################################################
    norm_on_po = fnl_norm[fit_po]
    norm_on_po[norm_on_po < 0.01] = 0.01 
    fit_prcp1[fit_po] = norm_on_po
    ######################################################
    #Get prcp amounts method 2: calc amounts using ppca
    fit_prcp_on_po = a_pca_matrix.infill(nnghs, fit_po)[0]
    fit_prcp_on_po[fit_prcp_on_po < 0.01] = 0.01 
    fit_prcp2[fit_po] = fit_prcp_on_po
    fit_prcp2[np.isfinite(obs_norm)] = obs_norm[np.isfinite(obs_norm)]
    ######################################################
    plt.plot(fit_prcp2[fit_po]-fit_prcp1[fit_po])
    #plt.show()
    
    print (np.sum(fit_prcp2) - np.sum(fit_prcp1))/np.sum(fit_prcp1)
    print (np.sum(fit_prcp1[xval_mask_prcp]) - np.sum(prcp[xval_mask_prcp]))/np.sum(prcp[xval_mask_prcp])
    print (np.sum(fit_prcp2[xval_mask_prcp]) - np.sum(prcp[xval_mask_prcp]))/np.sum(prcp[xval_mask_prcp])
    
    
    #Scale amts to the mean
    prcp_norm_mean = np.mean(fnl_norm,dtype=np.float64)
    fin_mask = np.isfinite(obs_prcp)
    nan_mask = np.logical_not(fin_mask)
    
    #Only scale the daily amounts that were predicted
    scaler1 = ((prcp_norm_mean*fnl_norm.size) - np.sum(fit_prcp1[fin_mask]))/np.sum(fit_prcp1[nan_mask])
    scaler2 = ((prcp_norm_mean*fnl_norm.size) - np.sum(fit_prcp2[fin_mask]))/np.sum(fit_prcp2[nan_mask])
    fit_prcp1[nan_mask] = scaler1 * fit_prcp1[nan_mask]
    fit_prcp2[nan_mask] = scaler2 * fit_prcp2[nan_mask]
    
    print "TOTAL PRCP ",np.sum(prcp[xval_mask_prcp]),np.sum(fit_prcp1[xval_mask_prcp]),np.sum(fit_prcp2[xval_mask_prcp]) 
    
    ymax = np.max(np.concatenate([prcp[xval_mask_prcp],fit_prcp1[xval_mask_prcp],fit_prcp2[xval_mask_prcp]]))
    
    plt.subplot(311)
    plt.plot(prcp[xval_mask_prcp])
    plt.ylim((0,ymax))
    plt.subplot(312)
    plt.plot(fit_prcp1[xval_mask_prcp])
    plt.ylim((0,ymax))
    plt.subplot(313)
    plt.plot(fit_prcp2[xval_mask_prcp])
    plt.ylim((0,ymax))
    #plt.show()
    
    yrs = np.unique(days[YEAR])
    ann_prcp_obs = np.zeros(yrs.size)
    ann_prcp_fit1 = np.zeros(yrs.size)
    ann_prcp_fit2 = np.zeros(yrs.size)
    
    for x in np.arange(yrs.size):
        
        yr_mask = days[YEAR] == yrs[x]
        ann_prcp_obs[x] = np.sum(prcp[np.logical_and(xval_mask_prcp,yr_mask)])
        ann_prcp_fit1[x] = np.sum(fit_prcp1[np.logical_and(xval_mask_prcp,yr_mask)])
        ann_prcp_fit2[x] = np.sum(fit_prcp2[np.logical_and(xval_mask_prcp,yr_mask)])
    
    plt.clf()
    plt.plot(ann_prcp_obs)
    plt.plot(ann_prcp_fit1)
    plt.plot(ann_prcp_fit2)
    #plt.show()
    
    print np.var(prcp[xval_mask_prcp],ddof=1),np.var(fit_prcp1[xval_mask_prcp],ddof=1),np.var(fit_prcp2[xval_mask_prcp],ddof=1)
    
    
    print calc_hss(obs_po[xval_mask_prcp], fit_po[xval_mask_prcp])
    
    freq_obs = np.sum(obs_po[xval_mask_prcp]) / np.float(obs_po[xval_mask_prcp].size)
    freq_fit = np.sum(fit_po[xval_mask_prcp]) / np.float(fit_po[xval_mask_prcp].size)
    
    npo_obs = np.sum(obs_po[xval_mask_prcp])
    npo_fit = np.sum(fit_po[xval_mask_prcp])
    
    perr_freq = ((freq_fit - freq_obs) / freq_obs) * 100.
    perr_freq2 = ((npo_fit - npo_obs) / float(npo_obs)) * 100.
    #print freq_obs,freq_fit
    print npo_obs,npo_fit
    print perr_freq,perr_freq2


def test_infill_prcp_transform2():

    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    nmask = int(np.round(5*365.25))
    idxs = np.arange(stn_da.days.size)
    nnghs = 3
    mth_masks = build_mth_masks(stn_da.days)
    
    ds_xval = Dataset('/projects/daymet2/station_data/infill/xval_infill_po.nc')
    stn_ids = np.array(ds_xval.variables['stn_id'][:],dtype="<S16")
    #stn_ids = ['GHCN_USC00247286']
    for stn_id in stn_ids:
    
        #stn_id = 'GHCN_USC00047306'#GHCN_USC00027435GHCN_USC00247286
        
        prcp = stn_da.load_all_stn_obs_var(stn_id, 'prcp')[0]
        fin_mask = np.isfinite(prcp)
    
        
        last_idxs = np.nonzero(fin_mask)[0][-nmask:]
        xval_mask_prcp = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_mask)
        #xval_mask_prcp = fin_mask
        
        ds_prcp = Dataset('/projects/daymet2/station_data/infill/normals_prcp_sqrt.nc')
        
        po_mask = prcp > 0
        #po_mask = np.ones(prcp.size,dtype=np.bool)
        
        matpca = pca_matrix_prcp(stn_id, stn_da, ds_prcp, po_mask,prcp_mask=xval_mask_prcp, prcp_norm=None)
        fit_prcp = matpca.infill(nnghs)[0]
        
    #    robjects.globalenv["prcp.obs"] = prcp
    #    robjects.globalenv["po_mask"] = po_mask
    #    robjects.globalenv["xval_mask"] = xval_mask_prcp
    #    r("save.image('/projects/daymet2/rdata/redlands_prcp_all.Rdata')")
        
        fnl_prcp = np.zeros(prcp.size)
        fnl_prcp[po_mask] = fit_prcp
        fit_prcp = fnl_prcp
        #fit_prcp = infill_prcp(stn_id, stn_da, ds_prcp, stn_da.days, mth_masks, nnghs, prcp_mask=xval_mask_prcp, prcp_norm=None,po_mask=po_mask)
    
        #print calc_hss(prcp[xval_mask_prcp]>0,fit_prcp[xval_mask_prcp]>0)
        perr_ttlamt = ((np.sum(fit_prcp[xval_mask_prcp]) - np.sum(prcp[xval_mask_prcp])) / np.sum(prcp[xval_mask_prcp])) * 100.
        print stn_id,perr_ttlamt
        
        prcp[xval_mask_prcp] = np.nan
        plt.plot(fnl_prcp)
        plt.plot(prcp)
        plt.show()
        
    #    plt.plot(fit_prcp[xval_mask_prcp]-prcp[xval_mask_prcp])
    #    plt.show()
    

def test_infill_prcp_transform():
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_id = 'GHCN_USC00027435'#'GHCN_USC00207690'#'SNOTEL_13C01S'#'SNOTEL_20G12S'SNOTEL_13A19S
    nnghs = 34
    days = stn_da.days
    nmask = int(np.round(5*365.25))
    idxs = np.arange(days.size)
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    
    prcp = stn_da.load_all_stn_obs_var(stn_id, 'prcp')[0]
    fin_prcp = np.isfinite(prcp)
    po = prcp > 0
    
    
    last_idxs = np.nonzero(fin_prcp)[0][-nmask:]
    xval_mask_prcp = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_prcp)
    xval_mask_prcp = np.ones(days.size,dtype=np.bool)
    
    prcp_norm,obs_norm = infill_prcp_norm(stn_id, stn_da, days, mth_masks, mthbuf_masks, prcp_mask=xval_mask_prcp)

    fnl_norm = np.copy(obs_norm)
    fnl_norm[np.isnan(obs_norm)] = prcp_norm[np.isnan(obs_norm)]
    fnl_norm = transform_prcp(fnl_norm)
    
    ds_prcp = Dataset('/projects/daymet2/station_data/infill/normals_prcp_log10.nc')
    
    fit_prcp = infill_prcp(stn_id, stn_da, ds_prcp, days, mth_masks, nnghs, prcp_mask=xval_mask_prcp, prcp_norm=fnl_norm)

    plt.plot(prcp)
    plt.plot(fit_prcp)
    plt.show()
    fit_prcp = fit_prcp > 0
    #print calc_hss(po[xval_mask_prcp],fit_prcp[xval_mask_prcp])
    sys.exit()
    
    #apca = pca_matrix_prcp_transform(stn_id, stn_da, ds_prcp,prcp_norm=fnl_norm,prcp_mask=xval_mask_prcp)
    #fit_prcp,obs_prcp = apca.infill(nnghs)[0:2]
    fit_prcp2 = fit_prcp
    PO_THRESHS = np.arange(0.0001, 0.9999, .0001)
    
    fin_mask = np.isfinite(obs_prcp)
    fit_prcp = fit_prcp[fin_mask]
    obs_prcp = obs_prcp[fin_mask]
    obs_po = obs_prcp > 0
    print np.sum(fin_mask)
    
    max_hss = 0
    max_thres = 0
    for thres in PO_THRESHS:
        
        thres_mask = fit_prcp >= thres
        
        hss = calc_hss(obs_po, thres_mask)
        if hss > max_hss:
            max_hss = hss
            max_thres = thres
            print max_hss
    
    print max_hss,max_thres
    fit_prcp2 = fit_prcp2 >= max_thres
    
    print calc_hss(po[xval_mask_prcp],fit_prcp2[xval_mask_prcp])
    
    sys.exit()
    
    fit_prcp[fit_prcp <= 0.01] = 0
    print np.sum(fit_prcp == 0)
    
    fin_mask = np.isfinite(obs_prcp)
    perr_ttlamt = ((np.sum(fit_prcp[fin_mask]) - np.sum(obs_prcp[fin_mask])) / np.sum(obs_prcp[fin_mask])) * 100.
    print perr_ttlamt
    
    plt.boxplot(fit_prcp)
    plt.show()
    
    plt.plot(obs_prcp)
    plt.plot(fit_prcp)
    plt.show()
    
def test_dist():
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    prcp = stn_da.load_all_stn_obs_var("GHCN_USC00207690", "prcp")[0]
    
    plt.hist(prcp[np.logical_and(np.isfinite(prcp),prcp > 0)],bins=50)
    plt.show()

def test_infills_prcp_norms_error():  
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    stn_da = station_data_ncdb("/projects/daymet2/station_data/all/all.nc")
    stn_id = 'GHCN_USC00041048'#'GHCN_USC00207690'#'SNOTEL_13C01S'#'SNOTEL_20G12S'SNOTEL_13A19S
    days = stn_da.days
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    nmask = int(np.round(5*365.25))
    idxs = np.arange(days.size)
    
    prcp = stn_da.load_all_stn_obs_var("GHCN_USC00041048", 'prcp')[0]
    fin_prcp = np.isfinite(prcp)
    
    last_idxs = np.nonzero(fin_prcp)[0][-nmask:]
    xval_mask_prcp = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_prcp)
    
    prcp_norm,obs_norm = infill_prcp_norm(stn_id, stn_da, days, mth_masks, mthbuf_masks,prcp_mask=xval_mask_prcp)

def transform_prcp_norms(path_ds):
    
    ds = Dataset(path_ds,'r+')
    vprcp = ds.variables['prcp']
    vprcp.set_auto_maskandscale(False)
    prcp = vprcp[:]
    
    #log10
    prcp[prcp < 0.01] = 0.01
    prcp = np.log10(prcp)
    
    #square root
    #prcp = np.sqrt(prcp)
    
    #5/12 root
    #prcp = prcp**(5.0/12.0)
    
    vprcp[:] = prcp
    
    ds.sync()
    ds.close()

def extract_tiles():
    aextract = ti.tile_extract('/projects/daymet2/climate_office/interp_grid/',
                    '/projects/daymet2/climate_office/interp_grid_geotifs/wheat_mature/','gdd_len', 'gdd_anom_2011')
    
    aextract.extract_all()
    aextract.mosaic()
    

def reset_imputes():
    
    stn_ids_rst = np.array(['RAWS_DBLB','GHCN_USC00104318'])
    
    #ds_tmin = Dataset('/projects/daymet2/station_data/infill/impute_tair/infill_tmin.nc')
    ds_tmax = Dataset('/projects/daymet2/station_data/infill/impute_tair/infill_tmax.nc','r+')
    
    #stnids_tmin = ds_tmin.variables['stn_id'][:].astype("<S16")
    stnids_tmax = ds_tmax.variables['stn_id'][:].astype("<S16")
    
    ds_tmax.variables['nnghs'][np.in1d(stnids_tmax, stn_ids_rst, True)] = netCDF4.default_fillvals['i2']
    ds_tmax.close()
    
def reset_imputes_by_mae():
        
    ds_tmin = Dataset('/projects/daymet2/station_data/infill/impute_tair/infill_tmin.nc','r+')
    ds_tmax = Dataset('/projects/daymet2/station_data/infill/impute_tair/infill_tmax.nc','r+')
    
    stnids_tmin = ds_tmin.variables['stn_id'][:].astype("<S16")
    stnids_tmax = ds_tmax.variables['stn_id'][:].astype("<S16")
    
    mae_tmin = ds_tmin.variables['mae'][:]
    mae_tmax = ds_tmax.variables['mae'][:]
    
    pct99_mae_tmin = np.percentile(mae_tmin,99)
    pct99_mae_tmax = np.percentile(mae_tmax,99)
    
    print np.sum(mae_tmin >= pct99_mae_tmin),np.sum(mae_tmax >= pct99_mae_tmax)
    
    ds_tmin.variables['nnghs'][mae_tmin >= pct99_mae_tmin] = netCDF4.default_fillvals['i2']
    ds_tmin.close()
    
    ds_tmax.variables['nnghs'][mae_tmax >= pct99_mae_tmax] = netCDF4.default_fillvals['i2']
    ds_tmax.close()
    
def test_modis_sin_rast():
    
    lst_rast = modis_sin_rast("/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs/mosaic_mean_gdal2.tif")
    print lst_rast.getDataValue(-124.167,49.35)

def testTopoDisectDEM():
    dem_rast = TopoDisectDEM('/projects/daymet2/dem/dem_orig.tif')
    lon, lat = dem_rast.getGeoLocation(1303, 445)
    elev = dem_rast.a[445,1303]
    print elev
    tdi_dists = [3,6,9,12,15]
    
    print dem_rast.get_tdi(445,1303,tdi_dists)

def xval_tmin_mae_analysis():
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc','tmin')
    ds = Dataset('/projects/daymet2/station_data/infill/impute_tair/xval_tmin_pc_all.nc')
    mae = ds.variables['mae'][:]
    min_nghs = ds.variables['min_nghs'][:]
    stn_id = ds.variables['stn_id'][:].astype("<S16")
    mask_stns = np.in1d(stn_da.stn_ids, stn_id, True)
    lon = stn_da.stns[LON][mask_stns]
    lat = stn_da.stns[LAT][mask_stns]
    elev = stn_da.stns[ELEV][mask_stns]
    
    mae_fnl = []
    nngh_fnl = []
    lines = []
    for x in np.arange(mae.shape[1]):
        
        mae_stn = mae[:,x]
        min_mae = np.min(mae_stn)
        nngh = min_nghs[min_mae==mae_stn][0]
        
        mae_fnl.append(min_mae)
        nngh_fnl.append(nngh)
        
        lines.append(",".join([stn_id[x],str(lat[x]),str(lon[x]),str(elev[x]),str(min_mae),str(nngh)+"\n"]))
        
    fout = open("/projects/daymet2/station_data/infill/impute_tair/stns_xval.csv","w")
    fout.write(",".join([STN_ID,LAT,LON,ELEV,"MAE","NNGH\n"]))
    fout.writelines(lines)
    fout.close()
    
    print np.mean(mae_fnl),np.mean(nngh_fnl)
    plt.hist(nngh_fnl)
    plt.show()
    plt.boxplot(mae_fnl)
    plt.show()

def find_dup_stns():
    stn_da = station_data_infill("/projects/daymet2/station_data/infill/impute_tair/infill_tmax.nc","tmax",stn_dtype=DTYPE_STN_DFLT)
    dup_stnids = []
    fout = open("/projects/daymet2/station_data/infill/impute_tair/dup_tmax_stns_rm.txt","w")
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
                    fout.write(",".join([stn_id,RM_STN_DUP+"\n"]))
        stat_chk.increment()
    fout.close()
        
def reset_infill_stn():
    ds = Dataset('/projects/daymet2/station_data/infill/impute_tair/infill_tmax.nc','r+')
    stn_ids = ds.variables['stn_id'][:]
    ds.variables['nnghs'][stn_ids=='SNOTEL_13A19S'] = netCDF4.default_fillvals['i2']
    ds.sync()
    ds.close()

def check_spatial_qa():
    
    db = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    
    lat_mask = db.stns[LAT] > 50
    
    #stn_idxs = np.nonzero(np.logical_and(np.char.startswith(db.stn_ids, "SNOTEL"),lat_mask))[0]
    stn_idxs = np.nonzero(lat_mask)[0]
    
    flags_tmin = db.var_ftmin[:,stn_idxs]
    print np.sum(flags_tmin!="")

def fix_gate_park():
    ds = Dataset('/projects/daymet2/station_data/infill/impute_tair/serial_tmax.nc','r+')
    i = np.nonzero(ds.variables['stn_id'][:]=='RAWS_MGAT')[0][0]
    print i
    ds.variables[LON][i] = -112.941028
    ds.variables[LAT][i] = 47.789662
    ds.variables[ELEV][i] = 1627
    ds.sync()
    ds.close()
    
def test_neon_varios():
    r.source('/nfshome/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R')
    
    stn_da = station_data_infill("/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc","tmin")
    
    dup_stn_ids = np.loadtxt("/projects/daymet2/station_data/infill/impute_tair/dup_tmin_stns_rm.txt" ,dtype="<S16")
    rm_stnids = np.concatenate([dup_stn_ids,RM_STN_IDS_TAIR])
    
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[NEON]),
                            np.logical_and(np.isfinite(stn_da.stns[TDI]),
                            np.logical_not(np.in1d(stn_da.stns[STN_ID], rm_stnids, True))))
    
    stns = stn_da.stns[mask_stns]
    stns_df = {LON:stns[LON],LAT:stns[LAT],ELEV:stns[ELEV],TDI:stns[TDI],LST:stns[LST],NEON:stns[NEON],'tair':stns[MEAN_OBS]}
    stns_df = robjects.DataFrame(stns_df)
    df_varios = r.build_neon_varios(17,stns_df)
    #print np.array(df_varios)
    

def tmin_xval_neon_stats():
    
    ds = Dataset("/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmin.nc")
    divs = ds.variables['neon'][:]
    neon_rngs = np.unique(divs.data[np.logical_not(divs.mask)])
    
    #neon_rngs = np.array([0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17])
    neon_rngs = np.arange(1,17)#np.arange(1,17)#np.array([13])#np.array([5,6,7,13,15])
    path = '/projects/daymet2/station_data/infill/infill_20130518/xval/xval_tmin_mean'#xvalgwr_tmin_mean
    #path = '/projects/daymet2/station_data/infill/impute_tair/xval_tmin_gwneon'
    #path = '/projects/daymet2/station_data/cce/xval_tmin_cce'
    #neon_rngs = np.array([12])
    #path = '/projects/daymet2/station_data/infill/impute_tair/xval_tmin_gwneon'  
    
    results = np.zeros((neon_rngs.size,5))
    
    x = 0
    for neon in neon_rngs:
        
        fpath = "".join([path,"_",str(neon),".nc"])
        
        ds = Dataset(fpath)
        mae = ds.variables['mae'][:]
        bias = ds.variables['bias'][:]
        nngh = ds.variables['min_nghs'][:]
        gw_p = ds.variables['gw_p'][:]
        in_ci = ds.variables['in_ci'][:]
        
        print nngh
        
        #print "######################" + str(neon) + "######################"
        
        in_ci_pct = np.sum(in_ci,axis=2)/np.float(in_ci.shape[2])
        mmae = np.mean(mae,axis=2)
        mbias = np.mean(bias,axis=2)
        
        ci_dif = np.abs((in_ci_pct*100)-95)
        
        
        smmae = np.sort(mmae.ravel())
        
        for amae in smmae[0:10]:
            
            r,c = np.nonzero(amae==mmae)
            
            if r.size == 1:
            
                print "%d|%f|%f|%f|%d|%d"%(neon,mmae[r,c],mbias[r,c],in_ci_pct[r,c],nngh[r],gw_p[c])
            
            else:
                
                for i in np.arange(r.size):
                    print "%d|%f|%f|%f|%d|%d"%(neon,mmae[r[i],c[i]],mbias[r[i],c[i]],in_ci_pct[r[i],c[i]],nngh[r[i]],gw_p[c[i]])
        
        print "############################################################################"
        #row, col of min mae
        r,c = np.nonzero(mmae==np.min(mmae))
        #r,c = np.nonzero(ci_dif==np.min(ci_dif))
        #print in_ci_pct[ci_dif==np.min(ci_dif)]
        
        if r.size == 1:
        
            print "%d|%f|%f|%f"%(neon,mmae[r,c],mbias[r,c],in_ci_pct[r,c])
        
        else:
            
            for i in np.arange(r.size):
                print "%d|%f|%f|%f"%(neon,mmae[r[i],c[i]],mbias[r[i],c[i]],in_ci_pct[r[i],c[i]])
                
        print "############################################################################"
        r,c = np.nonzero(mmae==np.min(mmae))
        #print in_ci_pct[ci_dif==np.min(ci_dif)]
        
        if r.size == 1:
        
            print "%d|%f|%f|%f|%d"%(neon,mmae[r,c],mbias[r,c],in_ci_pct[r,c],nngh[r])
        
        else:
            
            for i in np.arange(r.size):
                print "%d|%f|%f|%f|%d"%(neon,mmae[r[i],c[i]],mbias[r[i],c[i]],in_ci_pct[r[i],c[i],nngh[r[i]]])
        

        plt.clf()            
        plt.subplot(121)
        for i in np.arange(mmae.shape[1]):
            plt.plot(nngh,mmae[:,i],".-")
        plt.legend(gw_p)
        
        plt.subplot(122)
        for i in np.arange(ci_dif.shape[1]):
            plt.plot(nngh,ci_dif[:,i],".-")
        xmin,xmax = plt.xlim()   
        plt.hlines(1,xmin,xmax)
        plt.hlines(2,xmin,xmax)
        plt.legend(gw_p)
        plt.suptitle("NEON "+str(neon))
        
        
        plt.show()
        
#        print in_ci_pct[r,c]
#        print mmae[r,c]
#        print mbias[r,c]
        
        
        
        
        # print in_ci_pct
        #print mmae
        
       # min_mae = np.min(mmae)
        #i = np.nonzero(mmae==min_mae)[0][0]
        
        #results[x,:] = [neon,nngh[i],min_mae,mbias[i],in_ci_pct[i]]
        #print "%d|%d|%f|%f|%f"%tuple(results[x,:])

def tmin_xval_neon_stats2():
    #neon_rngs = np.array([0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17])
    neon_rngs = np.array([12])
    path = '/projects/daymet2/station_data/infill/infill_fnl/xval_tmin_mean'
        
    results = np.zeros((neon_rngs.size,5))
    
    x = 0
    for neon in neon_rngs:
        
        fpath = "".join([path,"_",str(neon),".nc"])
        
        ds = Dataset(fpath)
        mae = ds.variables['mae'][:]
        bias = ds.variables['bias'][:]
        nngh = ds.variables['min_nghs'][:]
        gw_p = ds.variables['gw_p'][:]
        in_ci = ds.variables['in_ci'][:]
        
        #print nngh
        
        #print "######################" + str(neon) + "######################"
        
        in_ci_pct = np.sum(in_ci,axis=2)/np.float(in_ci.shape[2])
        mmae = np.mean(mae,axis=2)
        mbias = np.mean(bias,axis=2)
        
        ci_dif = np.abs((in_ci_pct*100)-95)
        ci_dif[ci_dif==0] = 0.0001
        
        
        min_mmae = np.min(mmae)
        min_ci_dif = np.min(ci_dif)
        
        minr_mmae = mmae/min_mmae
        minr_ci_dif = ci_dif/min_ci_dif
        
        mr = ((minr_mmae*.90) + (minr_ci_dif*.10))
        r,c = np.nonzero(mr==np.min(mr))
                
        if r.size == 1:
        
            print "%d|%f|%f|%f|%d|%d"%(neon,mmae[r,c],mbias[r,c],in_ci_pct[r,c],nngh[r],gw_p[c])
        
        else:
            
            for i in np.arange(r.size):
                print "%d|%f|%f|%f|%d|%d"%(neon,mmae[r[i],c[i]],mbias[r[i],c[i]],in_ci_pct[r[i],c[i]],nngh[r[i]],gw_p[c[i]])

def anomaly_stats():
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc')
    path = '/projects/daymet2/station_data/infill/infill_20130518/xval/optimTairAnom/tmax/xval_tmax_anom'
    
    divs = ds.variables['neon'][:]
    if np.ma.isMA(divs):
        neon_rngs = np.unique(divs.data[np.logical_not(divs.mask)])
    else:
        neon_rngs = np.unique(divs)
    
    optimNgh = []
    print path
    for neon in neon_rngs:
        
        fpath = "".join([path,"_",str(neon),".nc"])
        
        ds = Dataset(fpath)
        mae = ds.variables['mae'][:]
        bias = ds.variables['bias'][:]
        nngh = ds.variables['min_nghs'][:]
        r2 = ds.variables['r2'][:]
        
        mmae = np.mean(mae,axis=1)
        mbias = np.mean(bias,axis=1)
        mr2 = np.mean(r2,axis=1)
        
        x = np.argmin(mmae)
        print "%d|%f|%f|%f|%d"%(neon,mmae[x],mbias[x],mr2[x],nngh[x])
        optimNgh.append(nngh[x])
    print optimNgh
                
#        smmae = np.sort(mmae)
#        print "######################################################"
#        for amae in smmae[0:10]:
#            
#            x = np.nonzero(amae==mmae)[0]
#            
#            if x.size == 1:
#            
#                print "%d|%f|%f|%f|%d"%(neon,mmae[x],mbias[x],mr2[x],nngh[x])
#            
#            else:
#                
#                for i in x:
#                    print "%d|%f|%f|%f|%d"%(neon,mmae[i],mbias[i],mr2[i],nngh[i])
#        
#        
#        plt.plot(nngh,mmae,".-")
#        plt.title("NEON "+str(neon))        
#        plt.show()
    

def test_stn_slct_neon():
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc','tmin')
        
    dup_stn_ids = np.loadtxt('/projects/daymet2/station_data/infill/impute_tair/dup_tmin_stns_rm.txt',dtype="<S16")
    rm_stnids = np.concatenate([dup_stn_ids,RM_STN_IDS_TAIR])
            
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[TDI]),
                            np.logical_not(np.in1d(stn_da.stns[STN_ID], rm_stnids, True)))
    
    stns = stn_da.stns[mask_stns]
    
    stn_slct = station_select_neon(stns, stn_da,'/projects/daymet2/station_data/infill/impute_tair/xval_gwneon2/neon_params_tmin.csv')   
    
    m = modeler_RkrigMean2('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                       '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R', stn_slct)
    
    a_pt = np.empty(1, dtype=[(LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
                                   (TDI, np.float64),(LST, np.float64),(NEON, np.float64)])
        
    a_pt[LON] = -113.545515
    a_pt[LAT] = 48.853358
    a_pt[ELEV] = 1500
    a_pt[TDI] = 3
    a_pt[LST] = -1.5
    a_pt[NEON] = 12
    a_pt = a_pt[0]
    print "START"
    print m.model_local(a_pt)

def neon_buf_stn_cnts():
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc','tmin')
        
    dup_stn_ids = np.loadtxt('/projects/daymet2/station_data/infill/impute_tair/dup_tmin_stns_rm.txt',dtype="<S16")
    rm_stnids = np.concatenate([dup_stn_ids,RM_STN_IDS_TAIR])
            
    mask_stns = np.logical_and(np.isfinite(stn_da.stns[TDI]),
                            np.logical_not(np.in1d(stn_da.stns[STN_ID], rm_stnids, True)))
    
    stns = stn_da.stns[mask_stns]
    
    stn_slct = station_select_neon(stns, stn_da,'/projects/daymet2/station_data/infill/impute_tair/xval_gwneon2/neon_params_tmin.csv')   
    
    m = modeler_RkrigMean2('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                       '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R', stn_slct)

    neon_params = np.loadtxt('/projects/daymet2/station_data/infill/impute_tair/xval_gwneon2/neon_params_tmin.csv',skiprows=1,delimiter=",")
    max_ngh = neon_params[:,1]+np.round(neon_params[:,1]*.2)
    
    for x in np.arange(neon_params.shape[0]):
        print "NEON REGION: "+str(neon_params[x,0])
        r.build_neon_vcld(max_ngh[x],m.stns_all,neon_params[x,0],max_dist_scale=1.4) 

def build_gvar():
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc','tmin')
    mask_stns = it.build_stn_mask(stn_da.stn_ids, '/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmin.csv')      
    stns = stn_da.stns[mask_stns]
    
    gvar,binmasks = it.build_global_variogram_cld(stns,'/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R')

    #np.save('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram.npy',gvar)
    #np.save('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram_binmasks.npy',binmasks)

    return gvar

def test_KrigTair():
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc','tmin')
    mask_stns = it.build_stn_mask(stn_da.stn_ids, '/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmin.csv')  
    stn_slct = station_select(stn_da,stn_mask=mask_stns)
    
    gvar = np.load('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram.npy')
    gvar_binidxs = np.load('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram_binmasks.npy')
    
    
    a_pt = np.empty(1, dtype=[(LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
                                   (TDI, np.float64),(LST, np.float64),(NEON, np.float64)])
    
#    params[MIN_NNGH] = 93
#    params[it.MAX_NNGH] = 112
        
#    a_pt[LON] = -113.545515
#    a_pt[LAT] = 48.853358
#    a_pt[ELEV] = 1500
#    a_pt[TDI] = 3
#    a_pt[LST] = -1.5
#    a_pt[NEON] = 12
#    a_pt = a_pt[0]
    
    a_pt[LON] = -113.283732
    a_pt[LAT] =  47.033762
    a_pt[ELEV] = 1300
    a_pt[TDI] = 1.184159
    a_pt[LST] = -2.642857
    a_pt[NEON] = 12
    a_pt = a_pt[0]
    
    
    gwvario = it.GwVarioSetRegion(stn_slct, gvar, gvar_binidxs, 30, 40, 1.0, NEON, 12, NEON_AREAS[12])

#    m = Basemap(projection='cyl',llcrnrlat=25.0,urcrnrlat=50.0,\
#            llcrnrlon=-126,urcrnrlon=-66,resolution='c')
#    m.drawcoastlines()
#    m.drawstates()
#    m.drawcountries()
#    m.scatter(gwvario.stn_slct.stns[LON][gwvario.vcld_stnmask],gwvario.stn_slct.stns[LAT][gwvario.vcld_stnmask])
#    plt.show()


    it.init_interp_R_env('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R')
    krig = KrigTair(stn_slct,gwvario)
    
    #krig.save_r_env("/projects/daymet2/rdata/KrigTair_TEST.Rdata")
    start = time.time()
    print krig.krig(a_pt)
    end = time.time()
    print end-start

def test_KrigTairPtRadius():
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc','tmin')
    mask_stns = it.build_stn_mask(stn_da.stn_ids, '/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmin.csv')  
    stn_slct = station_select(stn_da,stn_mask=mask_stns)
    
    params = it.load_neon_params('/projects/daymet2/station_data/infill/impute_tair/param_files/neon_mean_tmin_params.csv')
        
    gvar = np.load('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram.npy')
    gvar_binidxs = np.load('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram_binmasks.npy')
        
    gwvario = it.GwVarioPtRadius(stn_slct, gvar, gvar_binidxs, params, NEON)
    
    a_pt = np.empty(1, dtype=[(LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
                                   (TDI, np.float64),(LST, np.float64),(NEON, np.float64)])
    
#    params[MIN_NNGH] = 93
#    params[it.MAX_NNGH] = 112
    #Smaller error   
    a_pt[LON] = -113.392032  
    a_pt[LAT] =  46.467471
    a_pt[ELEV] = 1726
    a_pt[TDI] = 1.406070
    a_pt[LST] = -1.298849
    a_pt[NEON] = 12
    a_pt = a_pt[0]
    
    #Larger error   
#    a_pt[LON] = -113.383942  
#    a_pt[LAT] =   46.467114
#    a_pt[ELEV] = 1900
#    a_pt[TDI] = 2.635647
#    a_pt[LST] = -1.167447
#    a_pt[NEON] = 12
#    a_pt = a_pt[0]
    
    
    print "setup...."
    
    it.init_interp_R_env('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R')
    krig = it.KrigTair(stn_slct, gwvario)
    
    #krig.save_r_env("/projects/daymet2/rdata/KrigTair_TEST.Rdata")
    print "krig..."
    start = time.time()
    print krig.krig(a_pt)
    end = time.time()
    print end-start
    
#    print "krig..."
#    start = time.time()
#    print krig.krig(a_pt)
#    end = time.time()
#    print end-start

def testGwrPcaTair():
    
    params = {}
    P_PATH_DB = 'P_PATH_DB'
    P_PATH_DB_XVAL = 'P_PATH_DB_XVAL'
    P_PATH_RMSTNS = 'P_PATH_RMSTNS'
    P_VARNAME = 'P_VARNAME'
    P_VARNAME_XVAL = 'P_VARNAME_XVAL'
    P_PATH_CLIB = 'P_PATH_CLIB'
    
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc"
    params[P_PATH_DB_XVAL] = '/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc'
    params[P_PATH_RMSTNS] = "/projects/daymet2/station_data/infill/infill_fnl/rm_stns_all.csv"
    params[P_VARNAME] = 'tmax'
    params[P_VARNAME_XVAL] = 'tmax'
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    stn_id = 'GHCN_USW00013985'
    
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME],vcc_size=470560000*2)
    stn_da_xval = station_data_infill(params[P_PATH_DB_XVAL], params[P_VARNAME_XVAL])
    mask_stns = it.build_stn_mask(stn_da.stn_ids, params[P_PATH_RMSTNS])
        
    stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)

    gwr_pca = it.GwrPcaTairStatic(stn_slct,params[P_PATH_CLIB],84,101,1.0)
    
    xval_stn = stn_da_xval.stns[stn_da.stn_idxs[stn_id]]
    print xval_stn
    xval_obs = stn_da_xval.load_obs(xval_stn[STN_ID])
    xval_anom = xval_obs - xval_stn[MEAN_OBS]
    

    
    gwr_pca.setup_for_pt(xval_stn, np.array([xval_stn[STN_ID]]))
    
    interp_tair = gwr_pca.gwr_pca()
    interp_anom = interp_tair - xval_stn[MEAN_OBS]
    
    difs = interp_anom - xval_anom
                    
    bias = np.mean(difs)
    mae = np.mean(np.abs(difs))
    
    r_value = stats.linregress(interp_anom, xval_anom)[2]
    r2 = r_value**2 #r-squared value; variance explained
    
    print mae,bias,r2
    plt.subplot(211)
    plt.plot(interp_anom)
    plt.subplot(212)
    plt.plot(xval_anom)
    plt.show()
    
    plt.plot(interp_anom,xval_anom,'.')
    plt.show()
    

def testInterpTair():
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc','tmin')
    mask_stns = it.build_stn_mask(stn_da.stn_ids, '/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmin.csv')  
    stn_slct = station_select(stn_da,stn_mask=mask_stns)
    
    params = it.load_neon_params('/projects/daymet2/station_data/infill/impute_tair/xval_gwneon2/neon_params_tmin.csv')
    
    gvar = np.load('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram.npy')
    gvar_binidxs = np.load('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram_binmasks.npy')
    
    it.init_interp_R_env('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R')
    gwvario = it.GwVarioPtRadius(stn_slct, gvar, gvar_binidxs, params, NEON)
    krig = it.KrigTair(stn_slct, gwvario)
    
    gwr = it.GwrPcaTairDynamic(stn_slct, '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                         params, NEON, False)
    
    interp = it.InterpTair(krig, gwr)
    
    a_pt = np.empty(1, dtype=[(LON, np.float64), (LAT, np.float64), (ELEV, np.float64),
                                   (TDI, np.float64),(LST, np.float64),(NEON, np.float64),(MEAN_OBS,np.float64)])
    
#    params[MIN_NNGH] = 93
#    params[it.MAX_NNGH] = 112
        
    a_pt[LON] = -113.545515
    a_pt[LAT] = 48.853358
    a_pt[ELEV] = 1500
    a_pt[TDI] = 3
    a_pt[LST] = -1.5
    a_pt[NEON] = 12
    a_pt[MEAN_OBS] = -2.0
    a_pt = a_pt[0]
    
    stat_chk = status_check(1, 1)
    tair_daily, tair_mean, std_err, ci = interp.interp(a_pt,np.array(['SNOTEL_13C01S']))
    stat_chk.increment()
    print tair_mean, std_err, ci
    plt.plot(tair_daily)
    plt.show()

def testInterpTairAnalyzeBias():
    
    stn_id = 'RAWS_AHIL'#'RAWS_KALP'
    
    P_PATH_DB = 'P_PATH_DB'
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
    
    params = {}
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc"
    params[P_PATH_DB_XVAL] = '/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc'
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_PATH_RMSTNS] = "/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmin.csv" 
    params[P_PATH_RLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R'
    params[P_PATH_GLOBAL_VARIO] = '/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram.npy'
    params[P_PATH_GLOBAL_VARIO_BINMASKS] = '/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram_binmasks.npy'
    params[P_PATH_PARAMS_MEAN] = '/projects/daymet2/station_data/infill/impute_tair/param_files/neon_mean_tmin_params.csv'
    params[P_PATH_PARAMS_ANOM] = '/projects/daymet2/station_data/infill/impute_tair/param_files/neon_anom_tmin_params.csv'
    params[P_VARNAME] = 'tmin'
    params[P_VARNAME_XVAL] = 'tmin'
    
    stn_da_infill = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/infill_tmin.nc', 
                                        'tmin',stn_dtype=DTYPE_STN_DFLT)
    
    stn_da_infill2 = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/infill_tmin.nc', 
                                        'tmin_imp',stn_dtype=DTYPE_STN_DFLT)
    
    ds_xval = Dataset('/projects/daymet2/station_data/infill/impute_tair/xval_overall_tmin_exp.nc')
    overall_bias = np.abs(ds_xval.variables['overall_bias'][:])
    stn_ids = ds_xval.variables['stn_id'][:]
    ids_chk = stn_ids[overall_bias>=np.percentile(overall_bias, 99)]
    bias_chk = overall_bias[overall_bias>=np.percentile(overall_bias, 99)]
    print "Total ids to check: "+str(ids_chk.size)
    for x in np.arange(bias_chk.size):
    
        stn_id = ids_chk[x]
        abias = bias_chk[x]
    
        print stn_id+" "+str(abias)
        obs = stn_da_infill.load_obs(stn_id)
        imp = stn_da_infill2.load_obs(stn_id)
        
        obs[obs == imp] = np.nan
        plt.subplot(211)
        plt.plot(imp)
        plt.plot(obs)
        plt.title(stn_id)
        
        plt.subplot(212)
        plt.plot(imp)
        plt.title(stn_id)
        plt.show()
        
        
        
#        stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
#        stn_da_xval = station_data_infill(params[P_PATH_DB_XVAL], params[P_VARNAME_XVAL])
#        mask_stns = it.build_stn_mask(stn_da.stn_ids, params[P_PATH_RMSTNS])    
#        stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
#        
#        p_mean = it.load_neon_params(params[P_PATH_PARAMS_MEAN])
#        p_anom = it.load_neon_params(params[P_PATH_PARAMS_ANOM])
#        
#        gvcld = np.load(params[P_PATH_GLOBAL_VARIO])
#        gvcld_binidxs = np.load(params[P_PATH_GLOBAL_VARIO_BINMASKS])    
#        it.init_interp_R_env(params[P_PATH_RLIB])
#        
#        gwvario = it.GwVarioPtRadius(stn_slct, gvcld, gvcld_binidxs, p_mean, NEON)
#        krig = it.KrigTair(stn_slct, gwvario)
#        gwrpca = it.GwrPcaTairDynamic(stn_slct, params[P_PATH_CLIB], p_anom, NEON, set_pt=False)
#        
#        #start = time.time()
#        xval_stn = stn_da_xval.stns[stn_da.stn_idxs[stn_id]]
#        print xval_stn
#        xval_mean = xval_stn[MEAN_OBS]
#        
#        xval_obs = stn_da_xval.load_obs(xval_stn[STN_ID])
#        xval_anom = xval_obs - xval_stn[MEAN_OBS]
#        
#        rm_stnid = np.array([xval_stn[STN_ID]])
#        
#        tair_mean,tair_var = krig.krig(xval_stn, np.array([xval_stn[STN_ID]]))
#        std_err,ci = krig.std_err_ci(tair_mean, tair_var)
#        
#        mean_bias = tair_mean - xval_mean
#        mean_mae = np.abs(mean_bias)
#        cir = ci[1] - ci[0]
#        in_ci = True if  xval_stn[MEAN_OBS] >= ci[0] and  xval_stn[MEAN_OBS] <= ci[1] else False
#    
#        gwrpca.setup_for_pt(xval_stn,rm_stnid)
#        tair_daily = gwrpca.gwr_pca()
#        interp_anom = tair_daily - xval_stn[MEAN_OBS]                
#        difs = interp_anom - xval_anom
#        anom_bias = np.mean(difs)
#        anom_mae = np.mean(np.abs(difs))
#        
#        xval_stn[MEAN_OBS] = tair_mean
#        #gwrpca.setup_for_pt(xval_stn,rm_stnid)
#        tair_daily = gwrpca.gwr_pca()
#        difs = tair_daily - xval_obs
#        overall_bias = np.mean(difs)
#        overall_mae = np.mean(np.abs(difs))
#        
#        plt.plot(xval_obs)
#        plt.plot(tair_daily)
#        
#        print np.mean(xval_obs)
#        print np.mean(tair_daily)
#        
#        plt.show()
    #end = time.time()
    

def testInterpTairDynParams():
    
    stn_id = 'RAWS_CLAR'#'RAWS_AHAV'#RAWS_AMOS'#'SNOTEL_10J01S'#RAWS_AMOS'#'RAWS_AMOS'#'RAWS_AMOS'#'RAWS_KALP'#'GHCN_USC00025270'
    
    P_PATH_DB = 'P_PATH_DB'
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
    
    params = {}
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc"
    params[P_PATH_DB_XVAL] = '/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc'
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_PATH_RMSTNS] = "/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmin.csv" 
    params[P_PATH_RLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R'
    params[P_PATH_GLOBAL_VARIO] = '/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram.npy'
    params[P_PATH_GLOBAL_VARIO_BINMASKS] = '/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram_binmasks.npy'
    params[P_PATH_PARAMS_MEAN] = '/projects/daymet2/station_data/infill/impute_tair/param_files/neon_mean_tmin_params.csv'
    params[P_PATH_PARAMS_ANOM] = '/projects/daymet2/station_data/infill/impute_tair/param_files/neon_anom_tmin_params.csv'
    params[P_VARNAME] = 'tmin'
    params[P_VARNAME_XVAL] = 'tmin'

    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    stn_da_xval = station_data_infill(params[P_PATH_DB_XVAL], params[P_VARNAME_XVAL])
    mask_stns = it.build_stn_mask(stn_da.stn_ids, params[P_PATH_RMSTNS])    
    stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
    
    p_mean = it.load_neon_params(params[P_PATH_PARAMS_MEAN])
    p_anom = it.load_neon_params(params[P_PATH_PARAMS_ANOM])
    
    gvcld = np.load(params[P_PATH_GLOBAL_VARIO])
    gvcld_binidxs = np.load(params[P_PATH_GLOBAL_VARIO_BINMASKS])    
    it.init_interp_R_env(params[P_PATH_RLIB])
    
    gwvario = it.GwVarioPtRadius(stn_slct, gvcld, gvcld_binidxs, p_mean, NEON)
    krig = it.KrigTair(stn_slct, gwvario)
    gwrpca = it.GwrPcaTairDynamic(stn_slct, params[P_PATH_CLIB], p_anom, NEON, set_pt=False)
    
    #start = time.time()
    xval_stn = stn_da_xval.stns[stn_da.stn_idxs[stn_id]]
#    print xval_stn[LST],xval_stn[TDI]
#    xval_stn[LAT] =   37.365814
#    xval_stn[LON] =  -118.666948 
#    xval_stn[LST] =  0.823252
#    xval_stn[TDI] =  2.848142
#    xval_stn[ELEV] =  2400
    print xval_stn
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
    #gwrpca.setup_for_pt(xval_stn,rm_stnid)
    tair_daily = gwrpca.gwr_pca()
    difs = tair_daily - xval_obs
    overall_bias = np.mean(difs)
    overall_mae = np.mean(np.abs(difs))
    
    print "ANOME MAE "+str(anom_mae)
    
    plt.plot(xval_obs)
    plt.plot(tair_daily)
    
    print np.mean(xval_obs)
    print np.mean(tair_daily)
    #print std_err
    
    plt.show()
    #end = time.time()

def testInterpTairDynParams2():
    #SNOTEL_12A02S
    stn_id = 'SNOTEL_13B07S'#'SNOTEL_13A15S'#'RAWS_MGOL'#'GHCN_USC00243489'#####'RAWS_AHAV'#RAWS_AMOS'#'SNOTEL_10J01S'#RAWS_AMOS'#'RAWS_AMOS'#'RAWS_AMOS'#'RAWS_KALP'#'GHCN_USC00025270'
    
    P_PATH_DB = 'P_PATH_DB'
    P_PATH_DB_XVAL = 'P_PATH_DB_XVAL'
    P_PATH_RLIB = 'P_PATH_RLIB'
    P_PATH_CLIB = 'P_PATH_CLIB'
    P_PATH_RMSTNS = 'P_PATH_RMSTNS'
    P_PATH_PARAMS_MEAN = 'P_PATH_PARAMS_MEAN'
    P_PATH_PARAMS_ANOM = 'P_PATH_PARAMS_ANOM'
    
    P_VARNAME = 'P_VARNAME'
    P_VARNAME_XVAL = 'P_VARNAME_XVAL'    
    
    params = {}
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/infill_fnl/serial_tmin.nc"
    params[P_PATH_DB_XVAL] = '/projects/daymet2/station_data/infill/infill_fnl/serial_tmin.nc'
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_PATH_RMSTNS] = "/projects/daymet2/station_data/infill/infill_fnl/rm_stns_all.csv"
    params[P_PATH_RLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R'
    params[P_PATH_PARAMS_MEAN] = '/projects/daymet2/station_data/cce/param_files/cce_mean_tmin_params.csv'
    params[P_PATH_PARAMS_ANOM] = '/projects/daymet2/station_data/cce/param_files/cce_anom_tmin_params.csv'
    params[P_VARNAME] = 'tmin'
    params[P_VARNAME_XVAL] = 'tmin'
        
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    stn_da_xval = station_data_infill(params[P_PATH_DB_XVAL], params[P_VARNAME_XVAL])
    mask_stns = it.build_stn_mask(stn_da.stn_ids, params[P_PATH_RMSTNS])    
    stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
    
    p_mean = it.load_neon_params_mean(params[P_PATH_PARAMS_MEAN])[2]
    p_anom = it.load_neon_params_anom(params[P_PATH_PARAMS_ANOM])
      
    it.init_interp_R_env(params[P_PATH_RLIB])
    
    krig_params = it.KrigParamsStatic2(stn_slct, p_mean[0], p_mean[1], p_mean[2], 
                                            p_mean[3], p_mean[4])
    
    #krig_params = it.KrigParamsDynamic2(stn_slct, p_mean, NEON)
    
    krig = it.KrigTair2(stn_slct, krig_params)
    
    #start = time.time()
    xval_stn = stn_da_xval.stns[stn_da.stn_idxs[stn_id]]
#    print xval_stn[LST],xval_stn[TDI]
#    xval_stn[LAT] =   47.675729
#    xval_stn[LON] =  -113.126078 
#    xval_stn[LST] =  -7.371884
#    xval_stn[TDI] =  2.440777
#    xval_stn[ELEV] =  2105
    print xval_stn
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
    print tair_mean
    print mean_bias,in_ci,std_err
#    gpca = it.GwrPcaTairStatic(stn_slct, params[P_PATH_CLIB], 63, 76,2,False)
#    gpca.setup_for_pt(xval_stn, np.array([xval_stn[STN_ID]]))
#    dtair = gpca.gwr_pca()
#    plt.plot(dtair)
#    plt.show()

def run_output_no_tdi_stns():

#    output_no_tdi_stns("/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc",
#                       "/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmin.csv","a")
    
    output_no_tdi_stns("/projects/daymet2/station_data/infill/impute_tair/serial_tmax.nc",
                       "/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmax.csv","a")


def linear_interp_envlimits():
    
    xp = [0,100]
    fp = [0,255]
    
    in_rast = input_raster('/home/jared.oyler/env_lim/rad3.tif')
    a = in_rast.readEntireRaster()
    a = np.interp(a, xp, fp, left=np.nan, right=np.nan)
    a[np.isnan(a)] = in_rast.ndata

    out_rast = output_raster('/home/jared.oyler/env_lim/rad_final.tif', 
                             in_rast, 1, noDataVal=in_rast.ndata, datatype=gdalconst.GDT_UInt16)
    
    out_rast.writeDataArray(a, 0, 0)


def mean_mt_ndvi():
    
    yrs = np.arange(2000,2013)
    mths = np.array(["june","july","august"])
    
    path = "/projects/daymet2/climate_office/modis/MOD13A3/jja_mt_tifs/"
    
    ndvi_means = np.zeros((mths.size,yrs.size+1))
    
    for x in np.arange(mths.size):
        
        mth = mths[x]
        
        for i in np.arange(yrs.size):
            
            yr = yrs[i]
            
            r = input_raster(''.join([path,'mosaic.ndvi.',str(yr),".",mth,".tif"]))
            a = r.readEntireRaster()
            a = np.ma.masked_array(a,mask=a==r.ndata)
            ndvi_means[x,i] = np.ma.mean(a)
            
    for x in np.arange(mths.size):
        
        mth = mths[x]
        r = input_raster(''.join([path,'mosaic.ndvi.mean.',mth,".tif"]))
        a = r.readEntireRaster()
        a = np.ma.masked_array(a,mask=a==r.ndata)
        ndvi_means[x,-1] = np.ma.mean(a)
    
    np.savetxt("".join([path,"ndvi_means.csv"]),ndvi_means, fmt="%0.4f", delimiter=",")

def testGwSampleVario():
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc','tmin')
    mask_stns = it.build_stn_mask(stn_da.stn_ids, '/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmin.csv')  
    stn_slct = station_select(stn_da,stn_mask=mask_stns)
    
    gvar = np.load('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram.npy')
    gvar_binidxs = np.load('/projects/daymet2/station_data/infill/impute_tair/tmin_global_variogram_binmasks.npy')
        
    stn_slct.set_params(30,40,1.0)
    stn_slct.set_pt(48.853358,  -113.545515)
    
    interp_stns,ngh_obs,ngh_dist,wgt = stn_slct.get_ngh_stns(False)
    
    print interp_stns.dtype
    
    gwvario = it.GwSampleVario(stn_slct, gvar, gvar_binidxs)
    start = time.time()
    gwvario.build_gw_sample_vario(ngh_dist,1.0)
    end = time.time()
    
    print end-start


def neon_stn_counts():
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmin.nc','tmin')
    mask_stns = it.build_stn_mask(stn_da.stn_ids, '/projects/daymet2/station_data/infill/impute_tair/rm_stns_tmin.csv')
    
    stns = stn_da.stns[np.logical_and(mask_stns,np.isfinite(stn_da.stns[NEON]))]
    
    for neon in np.unique(stns[NEON]):
        
        print "".join(["Neon ",str(neon),": ",str(np.sum(stns[NEON]==neon))])
    

def fix_stn_loc_in_serial():
    ds_lst = input_raster('/projects/daymet2/dem/interp_grids/tifs/lst_tmax.tif')
    ds_tdi = input_raster('/projects/daymet2/dem/interp_grids/tifs/tdi.tif')
    
    stn_id = 'GHCN_USC00040824'
    new_lon = -118.666948
    new_lat = 37.365814
    new_tdi = ds_tdi.getDataValue(new_lon, new_lat)
    new_lst = ds_lst.getDataValue(new_lon, new_lat)
    print new_lst,new_tdi
    
    ds_stns = Dataset('/projects/daymet2/station_data/infill/impute_tair/serial_tmax.nc','r+')
    stn_ids = ds_stns.variables['stn_id'][:]
    x = np.nonzero(stn_ids==stn_id)[0][0]
    ds_stns.variables['lon'][x] = new_lon
    ds_stns.variables['lat'][x] = new_lat
    ds_stns.variables['tdi'][x] = new_tdi
    ds_stns.variables['lst'][x] = new_lst
    ds_stns.sync()
    ds_stns.close()

def save_stns_to_R():

    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc','tmax',)
    imp_flag = stn_da.ds.variables['flag_impute'][:]
    ndays = imp_flag.shape[0]
    imp_flag = np.sum(imp_flag,axis=0) == ndays

    rm_ids = np.loadtxt('/projects/daymet2/station_data/infill/infill_fnl/rm_stns_all.csv', np.str)
    
    mask_rm_stns = np.logical_not(np.in1d(stn_da.stn_ids, rm_ids, True))
    mask_mask = stn_da.stns['mask']==1
    mask_all = np.logical_and(mask_mask,mask_rm_stns)
    
    stns = stn_da.stns[mask_all]
    imp_flag = imp_flag[mask_all]    
    r.source("/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/krig.R")
    
    #(lon,lat,elev,tdi,lst,neon,tair,fpath_out)
    r.save_stn_spatial_df(stns[STN_ID],stns[LON],stns[LAT],stns[ELEV],stns[TDI],stns[LST],stns[NEON],stns[MEAN_OBS],imp_flag,
                          "/projects/daymet2/station_data/infill/infill_fnl/stns_tmax.RData")

def save_stns_to_csv():

    stndtype = copy(DTYPE_STN_DFLT)
    stndtype.append(('xval_overall_bias',np.float64))
    stndtype.append(('xval_overall_mae',np.float64))
    stndtype.append(('xval_overall_r2',np.float64))

    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax',stn_dtype=stndtype)
    stn_mask = np.logical_and(np.isfinite(stn_da.stns[MASK]),np.isnan(stn_da.stns[BAD]))   
    stns = stn_da.stns[stn_mask]
        
    fout = open('/projects/daymet2/station_data/infill/infill_20130725/stns_tmax.csv','w')
    fout.write(",".join(['STNID','LON','LAT','ELEV','TDI','LST','BIAS','MAE','NEON\n']))

    for stn in stns:

        fout.write(",".join([stn[STN_ID],str(stn[LON]),str(stn[LAT]),str(stn[ELEV]),
                             str(stn[TDI]),str(stn[LST]),str(stn['xval_overall_bias']),
                             str(stn['xval_overall_mae']),"".join([str(stn[NEON]),"\n"])]))
        
    fout.close()

def create_empty_neon_var():
    
    ds = Dataset('/projects/daymet2/station_data/cce/serial_tmax.nc','r+')
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    stn_id = ds.variables['stn_id'][:]
    nvar = ds.variables['neon']
    
    elev = input_raster('/projects/daymet2/dem/cce/interp_grids/cce_elev.tif')
    
    for x in np.arange(lon.size):
        
        try:
        
            elev_val = elev.getDataValue(lon[x], lat[x])
            
            if elev_val != elev.ndata:
                
                nvar[x] = 2
                print "Set CCE station"
                
        except Exception:
            continue

#    nvar = ds.createVariable('neon','f8',('stn_id',),fill_value=255)
#    nvar.long_name = "neon ecoregions"
#    nvar.units = "NA"
#    nvar.standard_name = "neon ecoregions"
#    nvar[:] = 1
    
    ds.sync()
    

def arcgic_ncdf_test():
    ds_elev = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_elev.nc')
    lons = ds_elev.variables['lon'][:]
    lats = ds_elev.variables['lat'][:]
    elev = ds_elev.variables['elev'][:]
    ndata = ds_elev.variables['elev'].missing_value
    
    ds_test = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_test.nc','w')
        
    ds_test.createDimension('y',lats.size)
    ds_test.createDimension('x',lons.size)
    
    y = ds_test.createVariable('y','f8',('y',),fill_value=False)
    y.units = "degrees"
    y.long_name = "y coordinate of projection"
    y.standard_name = "y coordinate of projection"
    y[:] = lats
    
    x = ds_test.createVariable('x','f8',('x',),fill_value=False)
    x.units = "degrees"
    x.long_name = "x coordinate of projection"
    x.standard_name = "x coordinate of projection"
    x[:] = lons
    
    latitudes = ds_test.createVariable('lat','f8',('y','x'),fill_value=False)
    latitudes.long_name = "latitude coordinate"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    for x in np.arange(lats.size):
        latitudes[x,:] = lats[x]

    longitudes = ds_test.createVariable('lon','f8',('y','x'),fill_value=False)
    longitudes.long_name = "longitude coordinate"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    for x in np.arange(lons.size):
        longitudes[:,x] = lons[x]
        
    crs = ds_test.createVariable("crs",'i2')
    crs.grid_mapping_name = "latitude_longitude"
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = 6378137.0
    crs.inverse_flattening = 298.257223563
    
    velev = ds_test.createVariable('elev','f4',('y','x'),fill_value=ndata)
    velev.long_name = "elevation"
    velev.units = "m"
    velev.standard_name = "elevation"
    velev.coordinates = "lat lon"
    velev.grid_mapping = "crs"
    velev[:] = elev
    
    ds_test.Conventions = "CF-1.6"
    ds_test.sync()
    ds_test.close()

def test_single_tile():
    
    stn_da_tmin = station_data_infill('/projects/daymet2/station_data/cce/serial_tmin.nc', 'tmin',stn_dtype=DTYPE_STN_MEAN_LST_TDI)
    
    ds_mask = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_mask.nc')    
    ds_elev = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_elev.nc')
    ds_tdi = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_tdi.nc')
    ds_lsttmin = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_lst_tmin.nc')
    ds_lsttmax = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_lst_tmax.nc')
    ds_neon = Dataset('/projects/daymet2/dem/cce/interp_grids/ncdf/500_neon.nc')
    
    ds_attrs = [('elev',ds_elev),('tdi',ds_tdi),
                ('lst_tmin',ds_lsttmin),('lst_tmax',ds_lsttmax),('neon',ds_neon)]
    
    atiler = tl.tiler(ds_mask, ds_attrs, 500,500, 50, 50,np.array([0]))
    tile_num,wrk_chk = atiler.next()
    
    tile_grid_info = atiler.build_tile_grid_info()
    
    awriter = tl.tile_writer(tile_grid_info, '/projects/daymet2/dem/cce/interp_grids/ncdf/test_tile/')
    
    tile_id = tile_grid_info.get_tile_id(tile_num)
    
    rslt_tmin = np.zeros((stn_da_tmin.days.size,50,50))
    
    BLANK_CHK = np.zeros((50,50))
    rslt_tmin_mean = BLANK_CHK
    rslt_tmin_cil = BLANK_CHK
    rslt_tmin_ciu = BLANK_CHK
    rslt_tmin_se = BLANK_CHK
    
    awriter.write_rslts(tile_id,'tmin', stn_da_tmin.days, 0,0,
                        rslt_tmin, rslt_tmin_mean, rslt_tmin_cil, rslt_tmin_ciu, rslt_tmin_se)

def test_impute_tair_norm():
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    stn_id = 'GHCN_CA001047179'
    tair_var = 'tmax'
    
    stn_da = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc')
        
    source_r('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R')
    
    por = load_por_csv('/projects/daymet2/station_data/all/tairHomog_por_1948_2012.csv')
    mask_por_tmin,mask_por_tmax,mask_por_prcp = build_valid_por_masks(por)
    
    stn_masks = {'tmin':mask_por_tmin,'tmax':mask_por_tmax}
    
    nnr_ds = NNRNghData('/projects/daymet2/reanalysis_data/conus_subset/', (19480101,20121231))
        
    aclib = clib_wxTopo('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C')
    
    impute_tair_norm(stn_id, stn_da, stn_masks[tair_var], tair_var, nnr_ds, aclib)
    

def coast_tmax_analysis():
    
    ds = Dataset('/projects/daymet2/station_data/infill/impute_tair/xval_tmax_mean_17.nc')
    bias = ds.variables['bias'][:]
    bias.shape = (bias.shape[0],bias.shape[2])
    mbias = np.mean(bias,axis=1)
    
    mae = ds.variables['mae'][:]
    mae.shape = (mae.shape[0],mae.shape[2])
    mmae = np.mean(mae,axis=1)
    x = np.nonzero(mmae == np.min(mmae))[0][0]
    print np.min(mmae)
    mae = mae[x,:]
    bias = bias[x,:]
    
    stn_id = ds.variables['stn_id'][:].astype("<S16")
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/impute_tair/serial_tmax.nc', 'tmax',stn_dtype=DTYPE_STN_MEAN_LST_TDI)
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stn_id, True)]
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(stns[LAT]),urcrnrlat=np.max(stns[LAT]),\
            llcrnrlon=np.min(stns[LON]),urcrnrlon=np.max(stns[LON]),resolution='l')
    m.drawcoastlines()
    m.drawstates()
    #m.imshow(vals,origin='upper')
    
    colors = mae
    
    sizes = mae/np.max(mae)
    
    m.scatter(stns[LON],stns[LAT],c=colors,vmin=0,vmax=3)#,s=sizes*100)
    m.colorbar()
    
    ds.close()
    
    plt.show()

def ghcn_raws():
    
    raws_ids_orig = np.loadtxt('/projects/daymet2/station_data/raws/raws_stnids.txt',dtype="<S6")
    raws_ids = np.array([x[2:] for x in raws_ids_orig],dtype="<S4")
    
    ghcn_stns = open('/projects/daymet2/station_data/ghcn/ghcnd-stations.txt')
    
    ghcn_raws = []
    for aline in ghcn_stns.readlines():
        
        stn_id = aline[0:11].strip()
        
        #prefix for a raws station
        if stn_id[0:3] == "USR":
            ghcn_raws.append(stn_id[-4:])
            
    ghcn_raws = np.array(ghcn_raws,dtype="<S4")
    
    fnl_ids = raws_ids_orig[np.in1d(raws_ids, ghcn_raws, True)]
    np.savetxt('/projects/daymet2/station_data/raws/raws_ghcn_stnids.txt', fnl_ids, "%s")
    
#        fo = open(out_path,"w")
#    for stn_id in stn_ids:
#        fo.write("".join([stn_id,"\n"]))

def test_impute_tair_norm_new():
    
    stn_id = "RAWS_MMIS"
    stn_da = station_data_ncdb('/projects/daymet2/station_data/all/all.nc',(19480101,20111231))
    por = load_por_csv('/projects/daymet2/station_data/all/all_por_1948_2011.csv')
    mask_por_tmin,mask_por_tmax,mask_por_prcp = build_valid_por_masks(por,5,(-126.0,-64.0,22.0,50.0))
    
    source_r('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R')
    
    ds_nnr = NNRNghData('/projects/daymet2/reanalysis_data/conus_subset/',(19480101,20111231))
    
    fit_tair,obs_tair = impute_tair_norm(stn_id, stn_da, mask_por_tmax,"tmax",ds_nnr)
    print fit_tair

def xval_stats_impute_norm():
    
    stn_da = station_data_ncdb('/projects/daymet2/station_data/all/all.nc')
    ds = Dataset('/projects/daymet2/station_data/infill/xval_impute_norm_tair_sntlraws_nonnr.nc')
    
    stn_ids = ds.variables['stn_id'][:].astype("<S16")
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stn_ids, True)]
    
    print ds.variables['bias'][:,0,stn_ids=='SNOTEL_11D26S']
    
    obs = stn_da.load_all_stn_obs(np.array(['SNOTEL_11D26S']))
    print np.sum(obs[TMAX_FLAG] != '')
    plt.plot(obs[TMAX]-obs[TMIN])
    #plt.plot(obs[TMAX])
    plt.show()
    sys.exit()
    
    mae = ds.variables['mae'][:]
    mae_tmin = mae[0,0,:]
    mae_tmax = mae[1,0,:]
    
    bias = ds.variables['bias'][:]
    bias_tmin = bias[0,0,:]
    bias_tmax = bias[1,0,:]
    
    print stn_ids[mae_tmax==np.max(mae_tmax)]
    
    print np.mean(mae_tmin),np.mean(bias_tmin),np.max(mae_tmin)
    print np.mean(mae_tmax),np.mean(bias_tmax),np.max(mae_tmax)
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(stns[LAT]),urcrnrlat=np.max(stns[LAT]),\
            llcrnrlon=np.min(stns[LON]),urcrnrlon=np.max(stns[LON]),resolution='c')
    m.drawcountries()
    m.drawcoastlines()
    m.drawstates()
    m.scatter(stns[LON],stns[LAT])
    
    
    colors = mae_tmax
    #sizes = hss_fnl/np.max(hss_fnl)
    m.scatter(stns[LON],stns[LAT],c=colors)#,s=sizes*100)
    m.colorbar()
    plt.show()
    
    
    plt.boxplot(bias_tmax)
    plt.show()
    
def xval_stats_impute_daily():
    
    stn_da = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    ds = Dataset('/projects/daymet2/station_data/infill/xval_impute_daily_tair.nc')
    
    stn_ids = ds.variables['stn_id'][:].astype("<S16")
    stns = stn_da.stns[np.in1d(stn_da.stn_ids, stn_ids, True)]
    
    mae = ds.variables['mae'][:]
    mae_tmin = mae[0,0,:]
    mae_tmax = mae[1,0,:]
    
    print stn_ids[np.max(mae_tmax)==mae_tmax]
    
    print np.mean(mae_tmin),np.mean(mae_tmax)
    
    bias = ds.variables['bias'][:]
    bias_tmin = bias[0,0,:]
    bias_tmax = bias[1,0,:]
    
    print np.mean(bias_tmin),np.mean(bias_tmax)
    
    var_pct = ds.variables['var_pct'][:]
    var_pct_tmin = var_pct[0,0,:]
    var_pct_tmax = var_pct[1,0,:]
    print np.mean(var_pct_tmin),np.mean(var_pct_tmax)
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(stns[LAT]),urcrnrlat=np.max(stns[LAT]),\
            llcrnrlon=np.min(stns[LON]),urcrnrlon=np.max(stns[LON]),resolution='c')
    m.drawcountries()
    m.drawcoastlines()
    m.drawstates()
    m.scatter(stns[LON],stns[LAT])
    plt.show()
    
    plt.boxplot(mae_tmax)
    plt.show()
    

def test_impute_norm():
    
    np.seterr(all='raise')
    np.seterr(under='ignore')    
    
    ymd_start = 19480101
    ymd_end = 20121231
    path_db = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    path_rfuncs = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R'
    path_nnr = '/projects/daymet2/reanalysis_data/conus_subset/'
    path_por = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    path_clib = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    stn_id = 'RAWS_AQUA'#'GHCN_USC00040693'#GHCN_USC00041614 Bad Tmax converge
    nyrs_mod = 5
    tair_var = 'tmin'
    nngh_nnr = 4
    nngh_stns = 3
    
    aclib = clib_wxTopo(path_clib)
    
    stn_da = station_data_ncdb(path_db,(ymd_start,ymd_end))
    print stn_da.stns[stn_id==stn_da.stn_ids]
    
    source_r(path_rfuncs)

    #Load the period-of-record datafile
    por = load_por_csv(path_por)
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por)[0:2]

    stn_masks = {}
    stn_masks['tmin'] = mask_por_tmin
    stn_masks['tmax'] = mask_por_tmax
    
    ds_nnr = NNRNghData(path_nnr, (ymd_start,ymd_end))
    
    obs_tmin = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), 'tmin')[0],dtype=np.float64)
    obs_tmax = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), 'tmax')[0],dtype=np.float64)
    
    plt.plot(obs_tmin)
    plt.show()
                 
    fin_tmin = np.isfinite(obs_tmin)
    fin_tmax = np.isfinite(obs_tmax)
    
    #The number of observations that should not be set to nan
    #and are used to build the infill model
    nmask = int(np.round(nyrs_mod*365.25))
    idxs = np.arange(stn_da.days.size)
    last_idxs = np.nonzero(fin_tmin)[0][-nmask:]
    xval_mask_tmin = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_tmin)
    last_idxs = np.nonzero(fin_tmax)[0][-nmask:]
    xval_mask_tmax = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_tmax)
    
    #xval_mask_tmin = np.zeros(xval_mask_tmin.size,dtype=np.bool)
    #xval_mask_tmax = np.zeros(xval_mask_tmax.size,dtype=np.bool)
    
    xval_masks = {'tmin':xval_mask_tmin,'tmax':xval_mask_tmax}
    
    ################################################
    
    tair_mask = xval_masks[tair_var]
    
    obs_tair = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0],dtype=np.float64)
    fin_mask = np.isfinite(obs_tair) 
    days = stn_da.days
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    fit_tair = infill_tair(stn_id, stn_da, days, tair_var,stn_masks[tair_var], mth_masks, mthbuf_masks, tair_mask)
    print fit_tair - np.mean(obs_tair[fin_mask])
    #print np.mean(fit_tair[tair_mask])-np.mean(obs_tair[tair_mask])
    
    imp_mean,imp_var = impute_tair_norm(stn_id, stn_da, stn_masks[tair_var],tair_var,ds_nnr,aclib,tair_mask=tair_mask,nnghs=nngh_stns,nnghs_nnr=nngh_nnr)[0]
        
    
    obs_mean,obs_var = np.mean(obs_tair[fin_mask]),np.var(obs_tair[fin_mask], ddof=1)
    
    bias = imp_mean-obs_mean
    mae = np.abs(bias)
    var_pct = (imp_var/obs_var)*100.
    print mae,bias,var_pct

def test_impute_daily():
    
    ymd_start = 19480101
    ymd_end = 20121231
    path_db = '/projects/daymet2/station_data/all/bak/tairHomog_1948_2012.nc'
    path_rfuncs = ['/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R',
                   '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R']
    path_nnr = '/projects/daymet2/reanalysis_data/conus_subset/'
    path_clib = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    stn_id = 'GHCN_USC00115079'#'GHCN_USC00123418'#'GHCN_USC00040693'#GHCN_USC00041614 Bad Tmax converge
    nyrs_mod = 5
    tair_var = 'tmax'
    nngh_nnr = 4
    nngh_stns = 3
    
    aclib = clib_wxTopo(path_clib)
    
    stn_da = station_data_ncdb(path_db,(ymd_start,ymd_end))
    print stn_da.stns[stn_id==stn_da.stn_ids]
    
    for path in path_rfuncs:    
        source_r(path)

    stn_masks = {}
    stn_masks['tmin'] = np.isfinite(stn_da.stns[MEAN_TMIN])
    stn_masks['tmax'] = np.isfinite(stn_da.stns[MEAN_TMAX])
    
    ds_nnr = NNRNghData(path_nnr, (ymd_start,ymd_end))
    
    obs_tmin = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), 'tmin')[0],dtype=np.float64)
    obs_tmax = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), 'tmax')[0],dtype=np.float64)
    
    plt.plot(obs_tmin)
    plt.show()
                 
    fin_tmin = np.isfinite(obs_tmin)
    fin_tmax = np.isfinite(obs_tmax)
    
    #The number of observations that should not be set to nan
    #and are used to build the infill model
    nmask = int(np.round(nyrs_mod*365.25))
    idxs = np.arange(stn_da.days.size)
    last_idxs = np.nonzero(fin_tmin)[0][-nmask:]
    xval_mask_tmin = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_tmin)
    last_idxs = np.nonzero(fin_tmax)[0][-nmask:]
    xval_mask_tmax = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_tmax)
    
#    xval_mask_tmin = np.isnan(obs_tmin)
#    xval_mask_tmax = np.isnan(obs_tmax)
    
    xval_masks = {'tmin':xval_mask_tmin,'tmax':xval_mask_tmax}
    
    ################################################
    
    tair_mask = xval_masks[tair_var]
    
    obs_tair = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0],dtype=np.float64)
    mask_modobs = np.logical_and(np.isfinite(obs_tair),np.logical_not(tair_mask))
#    #Estimate normal
    norm_est,va_est = impute_tair_norm(stn_id, stn_da, stn_masks[tair_var], tair_var,ds_nnr,aclib,tair_mask=tair_mask,trim_nan=False,nnghs=nngh_stns, nnghs_nnr=nngh_nnr)[0]
    
    i = np.nonzero(stn_da.stn_ids==stn_id)[0][0]
        
    stn_da.stns["_".join(["mean",tair_var])][i] = norm_est
    stn_da.stns["_".join(["var",tair_var])][i] = va_est
    
    a_pca_matrix = ImputeMatrixPCA(stn_id, stn_da, tair_var,ds_nnr,aclib=aclib,tair_mask=tair_mask,add_bestngh=True)
                
    fit_tair =  a_pca_matrix.impute(nngh_stns,nngh_nnr)[0]   
    
    xval_fit = fit_tair[tair_mask]
    xval_obs = obs_tair[tair_mask]
    
    mod_fit = fit_tair[mask_modobs]
    mod_obs = obs_tair[mask_modobs]
    
    mae = np.mean(np.abs(xval_fit-xval_obs))
    bias = np.mean(xval_fit-xval_obs)
    r_value = stats.linregress(xval_fit, xval_obs)[2]
    var_pct = r_value**2 #r-squared value; variance explained
    print mae,bias,var_pct
    
    mae = np.mean(np.abs(mod_fit-mod_obs))
    bias = np.mean(mod_fit-mod_obs)
    r_value = stats.linregress(mod_fit, mod_obs)[2]
    var_pct = r_value**2 #r-squared value; variance explained
    print mae,bias,var_pct
    
    plt.subplot(311)
    plt.plot(xval_obs)
    plt.subplot(312)
    plt.plot(xval_fit)
    plt.subplot(313)
    plt.plot(xval_fit-xval_obs)
    plt.show()
    
    plt.plot(fit_tair)
    plt.show()
    
    plt.clf()
    plt.plot(xval_fit,xval_obs,'.')
    plt.show()
    

def test_impute_daily_noxval():
    
    ymd_start = 19480101
    ymd_end = 20121231
    path_db = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    path_rfuncs = ['/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R',
                   '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R']
    path_nnr = '/projects/daymet2/reanalysis_data/conus_subset/'
    stn_id = 'GHCN_USC00115079'#'#'SNOTEL_13A19S'#'GHCN_USC00040693'#GHCN_USC00041614 Bad Tmax converge
    tair_var = 'tmax'
    path_clib = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    nngh_nnr = 4
    nngh_stns = 3
    
    aclib = clib_wxTopo(path_clib)
    
    stn_da = station_data_ncdb(path_db,(ymd_start,ymd_end))
    print stn_da.stns[stn_id==stn_da.stn_ids]
    
    for path in path_rfuncs:    
        source_r(path)

    stn_masks = {}
    stn_masks['tmin'] = np.isfinite(stn_da.stns[MEAN_TMIN])
    stn_masks['tmax'] = np.isfinite(stn_da.stns[MEAN_TMAX])
    
    ds_nnr = NNRNghData(path_nnr, (ymd_start,ymd_end))
    
    obs_tmin = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), 'tmin')[0],dtype=np.float64)
    obs_tmax = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), 'tmax')[0],dtype=np.float64)
    
#    plt.plot(obs_tmin)
#    plt.show()
                 
    fin_tmin = np.isfinite(obs_tmin)
    fin_tmax = np.isfinite(obs_tmax)
    xval_masks = {'tmin':fin_tmin,'tmax':fin_tmax}
    
    tair_mask = xval_masks[tair_var]
    obs_tair = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0],dtype=np.float64)
    
    a_pca_matrix = ImputeMatrixPCA(stn_id, stn_da, tair_var,ds_nnr,aclib)
                
    fit_tair =  a_pca_matrix.impute(nngh_stns,nngh_nnr)[0]
    xval_fit = fit_tair[tair_mask]
    xval_obs = obs_tair[tair_mask]    
    
    mae = np.mean(np.abs(xval_fit-xval_obs))
    bias = np.mean(xval_fit-xval_obs)
    r_value = stats.linregress(xval_fit, xval_obs)[2]
    var_pct = r_value**2 #r-squared value; variance explained
    print mae,bias,var_pct
    
    plt.subplot(211)
    plt.plot(obs_tair)
    plt.subplot(212)
    plt.plot(fit_tair)
    plt.show()
    
    plt.clf()
    plt.plot(xval_fit,xval_obs,'.')
    plt.show()

    plt.clf()
    ann_vals = []
    for yr in np.unique(stn_da.days[YEAR]):
        ann_vals.append(np.mean(fit_tair[stn_da.days[YEAR]==yr]))
    plt.plot(ann_vals,'.-')
    plt.show()

def stn_cnts():

    ymd_start = 19480101
    ymd_end = 20121231
    path_db = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    path_por = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    
    #Load the period-of-record datafile
    por = load_por_csv(path_por)
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por,.001)[0:2]
    mask_tair = np.logical_or(mask_por_tmin,mask_por_tmax)
    
    stn_da = station_data_ncdb(path_db,(ymd_start,ymd_end))
    
    print np.sum(np.logical_and(mask_tair,np.char.startswith(stn_da.stns[STN_ID], 'RAWS')))
    
def qa_testing():
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    ymd_start = 19480101
    ymd_end = 20121231
    path_db = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    
    path_por = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    #tair_var = 'tmin'
    #Load the period-of-record datafile
    por = load_por_csv(path_por)
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por,1)[0:2]
    
    stn_da = station_data_ncdb(path_db,(ymd_start,ymd_end))
    
    p = 90
    
    #RAWS discrepency
    ####################################################################################
#    ds_chgpt = Dataset('/projects/daymet2/station_data/all/snht_chgpt2.nc')
#    ymd_maxT_tmin = ds_chgpt.variables['ymd'][0,:]
#    ymd_maxT_tmax = ds_chgpt.variables['ymd'][1,:]
#    stn_ids = ds_chgpt.variables['stn_id'][:].astype("<S16")
#    mask_tmin = np.logical_and(ymd_maxT_tmin >= 20091201,ymd_maxT_tmin <= 20091231)
#    mask_tmax = np.logical_and(ymd_maxT_tmax >= 20091201,ymd_maxT_tmax <= 20091231)
#    mask_tair = np.logical_or(mask_tmin,mask_tmax)
#    mask_badraws = np.logical_and(mask_tair,np.char.startswith(stn_ids,'RAWS'))
#    stn_ids = stn_ids[mask_badraws]
#    print stn_ids
    ####################################################################################
    
    ds_chgpt = Dataset('/projects/daymet2/station_data/all/snht_chgpt3.nc')
    stn_ids = ds_chgpt.variables['stn_id'][:].astype("<S16")
    maxT_tmax = ds_chgpt.variables['T'][1,:]
    
    if np.ma.is_masked(maxT_tmax):
        mask_maxT = np.logical_not(maxT_tmax.mask)
    else:
        mask_maxT = np.ones(maxT_tmax.size,dtype=np.bool)
    
    
    #mask_maxT = np.ones(maxT_tmax.size,dtype=np.bool)#np.ones(maxT_tmax.size,dtype=np.bool)#np.logical_not(maxT_tmax.mask)
    maxT_tmax = maxT_tmax[mask_maxT]
    stn_ids_tmax = stn_ids[mask_maxT]
    stn_ids_tmax = stn_ids_tmax[maxT_tmax >= np.percentile(maxT_tmax,p)]
    
    maxT_tmin = ds_chgpt.variables['T'][0,:]
    
    if np.ma.is_masked(maxT_tmin):
        mask_maxT_tmin = np.logical_not(maxT_tmin.mask)
    else:
        mask_maxT_tmin = np.ones(maxT_tmin.size,dtype=np.bool)
    
    maxT_tmin = maxT_tmin[mask_maxT_tmin]
    stn_ids_tmin = stn_ids[mask_maxT_tmin]
    stn_ids_tmin = stn_ids_tmin[maxT_tmin >= np.percentile(maxT_tmin,p)]
    
    print "Percentile Tmin: "+str(np.percentile(maxT_tmin,p))
    print "Percentile Tmax: "+str(np.percentile(maxT_tmin,p))
    
    #r.source('/home/jared.oyler/ecl_juno_workspace/r_sandbox/snht.R')
    chg_pt_mthly = qa_cp.ChgPtMthly(stn_da, mask_por_tmin, mask_por_tmax)
    chg_pt_dly = qa_cp.ChgPtDaily(stn_da, mask_por_tmin, mask_por_tmax)
    
    stn_ids = np.unique(np.concatenate([stn_ids_tmax,stn_ids_tmin]))
    
    chgpts = np.loadtxt('/projects/daymet2/station_data/all/qa_change_pts.csv', 
                        dtype=qa_cp.DTYPE_CHGPTS, delimiter=",",skiprows=1)
    stn_ids = stn_ids[np.logical_not(np.in1d(stn_ids,chgpts[STN_ID], True))]
    
    stn_ids = stn_ids[np.char.startswith(stn_ids,'SNOTEL')]
    
    xstart = np.nonzero(stn_ids=='SNOTEL_19F01S')[0][0]
    stn_ids = stn_ids[xstart:]
    
    #last station 'SNOTEL_19F01S', 'OR', 'SNOW MOUNTAIN'
    print stn_ids.size
    #sys.exit()
    #stn_id = 'RAWS_NLIM'#'SNOTEL_11D26S'#'GHCN_USC00040693'#'SNOTEL_11D26S'
#    obs = stn_da.load_all_stn_obs(np.array([stn_id]), False)
#    
#    plt.plot(obs[TMIN])
#    plt.plot(obs[TMAX])
#    plt.show()
    
    #stn = stn_da.stns[stn_da.stn_ids==stn_id][0]
    

    #mask_tair = np.logical_or(mask_por_tmin,mask_por_tmax)
    
    stn_ids= np.array(['GHCN_USC00428705'])#'SNOTEL_11D26S'
    
    for stn_id in stn_ids:
        
        #stn = stn_da.stns[stn_da.stn_ids==stn_id][0]
        #chg_pt.find_stn_chg_pt(stn_id, tair_var)
        print "DAILY"
        chg_pt_dly.find_stn_chg_pt_tmin_tmax(stn_id)
        chg_pt_dly.print_chg_pt()
        print "MONTHLY"
        chg_pt_mthly.find_stn_chg_pt_tmin_tmax(stn_id)
        chg_pt_mthly.print_chg_pt()
        chg_pt_dly.plot_chg_pt(nrows=6)
        chg_pt_mthly.plot_chg_pt(nrows=6,startnum=7)
#        plt.suptitle('Table Mountain')
        plt.show()
        plt.clf()
    
    #print chg_pt.stn_startend,chg_pt.stn_maxT_ymd
    #print chg_pt.stn_obs[chg_pt.stn_maxT_idx+1]
    
#    plt.subplot(211)
#    plt.plot(chg_pt.stn_T)
#    plt.subplot(212)
#    plt.plot(chg_pt.stn_obs)
#    ymin,ymax = plt.ylim()
#    plt.vlines(chg_pt.stn_maxT_idx, ymin, ymax,'r')
#    plt.show()
    
    

#    flags_tmin, flags_tmax = qa_temp.run_qa_all(stn, stn_da, obs[TMIN], obs[TMAX],stn_da.days)
#    
#    for flag in np.unique(flags_tmax):
#        
#        print "|".join([str(flag),str(np.sum(flags_tmax==flag))])


def stn_infill_bias_map():
    
    ymd_start = 19480101
    ymd_end = 20121231
    path_db = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    
    stn_da = station_data_ncdb(path_db,(ymd_start,ymd_end))
    
    ds = Dataset('/projects/daymet2/station_data/infill/xval_impute_norm_tair_sntlraws.nc')
    stn_ids = np.array(ds.variables['stn_id'][:],dtype="<S16")
    bias = ds.variables['bias'][1,0,:]
    stn_ids = stn_ids[np.char.startswith(stn_ids,'SNOTEL')]
    bias = bias[np.char.startswith(stn_ids,'SNOTEL')]
    stns = stn_da.stns[np.in1d(stn_da.stn_ids,stn_ids, True)]
    ds.close()
        
    #stns = stns[np.logical_and(stns[LON]>-360,stns[LON]<-48)]
    print stns.size
    m = Basemap(projection='cyl',llcrnrlat=np.min(stns[LAT]),urcrnrlat=np.max(stns[LAT]),\
            llcrnrlon=np.min(stns[LON]),urcrnrlon=np.max(stns[LON]),resolution='c')
    m.drawcountries()
    m.drawcoastlines()
    m.drawstates()
    colors = bias
    
    cmap = cm.jet
    cmap.set_under("purple",1.)
    
    norm = None#matplotlib.colors.Normalize(vmin=0)
    sizes = bias/np.max(bias)
    
    m.scatter(stns[LON],stns[LAT],c=colors,cmap=cmap,norm=norm)
    m.colorbar()
    plt.show()  

def qa_mtsnotel_locs():
    
    #stn_da = station_data_ncdb(path_db)
    #stns = stn_da.stns[np.logical_and()]
    
    path_out = "/projects/daymet2/mt_snotel_qa.csv"
    
    sntl_dtype = [(STN_ID,"<S16"),(STN_NAME,"<S30"),(STATE, "<S2"),('dsource',"S1"),(LAT, np.float64),(LON, np.float64), (ELEV, np.float64)]
    
    f_in = open('/projects/daymet2/station_data/snotel/cleaned/snotel_stns.csv')
    f_in.readline()
    
    stns = []
    
    for line in f_in.readlines():
        #["STN_ID","NAME","STATE","DSOURCE","LAT","LON","ELEV"]
        vals = line.split(',')
        lat = float(vals[4])
        lon = float(vals[5])
        
        #Change to (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        stns.append(("".join(["SNOTEL_",vals[0]]).upper(),vals[1],vals[2],vals[3],float(vals[4]),float(vals[5]),float(vals[6])))
    
    stns = np.array(stns,dtype=sntl_dtype)
    stns = stns[stns[STATE]=='MT']
    
    #stns = np.loadtxt('/projects/daymet2/station_data/snotel/cleaned/snotel_stns.csv',dtype=sntl_dtype, delimiter=",",skiprows=1)
    print "Total num of stations for location qa: "+str(stns.size)
    
    fout = open(path_out,"w")
    fout.write(",".join(["STN_ID","NAME","LAT", "LON", "ELEV", "DEM", "DIF","\n"]))
    
    for stn in stns:
 
        elev_dem = get_elev(stn)
        dif = stn[ELEV] - elev_dem
        print dif,stn
        fout.write(",".join([stn[STN_ID],stn[STN_NAME], str(stn[LAT]), str(stn[LON]), str(stn[ELEV]), str(elev_dem), str(dif),"","","","\n"]))
    
    fout.close()

def qa_cnts():
    
    ymd_start = 19480101
    ymd_end = 20121231
    path_db = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    path_por = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    
    stn_da = station_data_ncdb(path_db,(ymd_start,ymd_end))
    mask_sntl = np.char.startswith(stn_da.stn_ids,'SNOTEL')
        
    por = load_por_csv(path_por)
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por)[0:2]
    
    mask_tmin = np.logical_and(mask_sntl,mask_por_tmin)
    mask_tmax = np.logical_and(mask_sntl,mask_por_tmax)
    
    print "TMIN"
    stnids = stn_da.stn_ids[mask_tmin]
    print stnids.size    
    obs,flags = stn_da.load_all_stn_obs_var(stnids, 'tmin',False)
    nobs = np.sum(np.isfinite(obs))
    nflags = np.sum(np.logical_and(flags != "",flags != ""))
    print (np.float(nflags)/np.float(nobs))*100.
    
    print "TMAX"
    stnids = stn_da.stn_ids[mask_tmax]
    print stnids.size    
    obs,flags = stn_da.load_all_stn_obs_var(stnids, 'tmax',False)
    nobs = np.sum(np.isfinite(obs))
    nflags = np.sum(np.logical_and(flags != "",flags != ""))
    print (np.float(nflags)/np.float(nobs))*100.
    
def lst_mean_byyrmth():
    
    ds = Dataset('/projects/daymet2/climate_office/modis/MYD11A2/h11v04_tmin2.nc')
    mth = ds.variables['mth'][:]
    yr = ds.variables['yr'][:]
        
    mth_lsts = {}
    for x in np.arange(1,13):
        mth_lsts[x] = None
    
    for x in np.arange(1,13):
        
        for i in np.arange(2003,2013):
            
            print x,i
            lst_yrmth = ds.variables['LST_Night_1km'][np.logical_and(mth==x,yr==i),:,:]
            lst_yrmth_mean = np.ma.mean(lst_yrmth,axis=0)
    
            if mth_lsts[x] is None:
                mth_lsts[x] = lst_yrmth_mean
            else:
                mth_lsts[x] = np.dstack((mth_lsts[x],lst_yrmth_mean))
        
    mth_means = []
    for x in np.arange(1,13):
        mth_means.append(np.ma.mean(mth_lsts[x],axis=2))
    
    lst_fnl = np.dstack(mth_means)
    lst_fnl = np.mean(lst_fnl,axis=2)
    
    #cmap = cm.jet
    #norm = matplotlib.colors.Normalize(vmin=265,vmax=295)
    
    plt.imshow(lst_fnl)#,cmap=cmap,norm=norm)
    plt.colorbar()
    plt.show()
    
def lst_mean_bymth():
    
    ds = Dataset('/projects/daymet2/climate_office/modis/MYD11A2/h09v04_tmin.nc')
    mth = ds.variables['mth'][:]
        
    lst_fnl = None
    
    for x in np.arange(1,13):
        print x
        lst_mth = ds.variables['LST_Night_1km'][mth==x,:,:]
        lst_mth_mean = np.ma.mean(lst_mth,axis=0)
#        plt.imshow(lst_mth_mean)
#        plt.colorbar()
#        plt.show()
    
        if lst_fnl is None:
            lst_fnl = lst_mth_mean
        else:
            lst_fnl = np.ma.dstack((lst_fnl,lst_mth_mean))
            #lst_fnl = np.dstack((lst_fnl,lst_mth_mean))

    lst_fnl = np.ma.mean(lst_fnl,axis=2)
    
    lst = ds.variables['LST_Night_1km'][:]
    mask = np.logical_not(lst.mask)
    pct_obs = np.sum(mask,axis=0)/np.float(mask.shape[0])
    lst_fnl.mask = np.logical_or(lst_fnl.mask,pct_obs < 0.25)
    
    #cmap = cm.jet
    #norm = matplotlib.colors.Normalize(vmin=265,vmax=285)
    lst_fnl = KtoC(lst_fnl)
    
    grid_sd = EOSGridSD('/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.h09v04/MYD11A2.A2005201.h09v04.005.2008051082431.hdf')
    grid_sd.to_geotiff(lst_fnl,'/projects/daymet2/climate_office/modis/MYD11A2/h09v04_tmin.tif')
    
    plt.imshow(lst_fnl)#,cmap=cmap,norm=norm)
    plt.colorbar()
    plt.show()
    
    
def lst_bitmask_testing():
    from pyhdf.SD import SD, SDC
    
    qc_bitmask1 = 0b00000011
    qc_bitmask2 = 0b00001100
    qc_bitmask3 = 0b00110000
    qc_bitmask4 = 0b11000000
    
    #Bit1 values: Basic QA flags
    QC_BIT1_GOOD = 0b00000000
    QC_BIT1_OTHER = 0b00000001
    QC_BIT1_CLOUD = 0b00000010
    QC_BIT1_NA = 0b00000011
    
    #Bit3 values: Emis Error
    QC_BIT3_1 = 0b00000000 #Average emissivity error <= 0.01
    QC_BIT3_2 = 0b00010000 #Average emissivity error <= 0.02
    QC_BIT3_3 = 0b00100000 #Average emissivity error <= 0.04
    QC_BIT3_4 = 0b00110000 #Average emissivity error > 0.04
    
    #Bit4 values: LST Error
    QC_BIT4_1 = 0b00000000 #Average LST error <= 1K
    QC_BIT4_2 = 0b01000000 #Average LST error <= 2K
    QC_BIT4_3 = 0b10000000 #Average LST error <= 3K
    QC_BIT4_4 = 0b11000000 #Average LST error > 3K
        
    sd = SD('/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.h10v04/MYD11A2.A2008217.h10v04.005.2008233195913.hdf',SDC.READ)
    sds_lst = sd.select('LST_Day_1km')
    lst_scale = sds_lst.attributes()['scale_factor']
    lst_fill = sds_lst.attributes()['_FillValue']
    lst = np.array(sds_lst[:],dtype = np.float64)
    
    qc = sd.select('QC_Day')[:]
    qc1 = qc & qc_bitmask1
    qc3 = qc & qc_bitmask3
    qc4 = qc & qc_bitmask4
    
    sd_dva = sd.select('Day_view_angl')
    dva_scale = sd_dva.attributes()['scale_factor']
    dva_offset = sd_dva.attributes()['add_offset']
    dva_fill = sd_dva.attributes()['_FillValue']
    dva = sd_dva[:].astype(np.float64)
    dva[dva==dva_fill]  = np.nan
    dva = (dva*dva_scale) + dva_offset
    dva = np.abs(dva)
    dva_mask = dva > 40
#    print np.sum(dva > 40)/np.float(dva.size)
#    
#    plt.imshow(dva)
#    plt.colorbar()
#    plt.show()
    
    qc_mask1 =  np.logical_or(qc1==QC_BIT1_CLOUD,qc1==QC_BIT1_NA)
    qc_mask2 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_3)
    qc_mask3 = np.logical_and(qc1==QC_BIT1_OTHER,qc3==QC_BIT3_4)
    qc_mask4 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_3)
    qc_mask5 = np.logical_and(qc1==QC_BIT1_OTHER,qc4==QC_BIT4_4)
    qc_fnl_mask = np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(qc_mask1,qc_mask2),qc_mask3),qc_mask4),qc_mask5),dva_mask)
    
    lst[qc_fnl_mask] = np.nan
    plt.imshow(lst)
    plt.colorbar()
    plt.show()
    
    
    print np.sum(qc_fnl_mask)

def extreme_infill_qa():
    ds = Dataset('/projects/daymet2/station_data/infill/infill_tmax.nc')
    stn_ids = ds.variables['stn_id'][:]
    avar = ds.variables['tmax_imp']
    
    schk = status_check(stn_ids.size,100)
    for x in np.arange(stn_ids.size):
        
        vals = avar[:,x]
        #if np.sum(vals < -89.4) > 0:
        if np.sum(vals > 57.7) > 0:
            if not np.char.startswith(stn_ids[x],'GHCN_CA'):
                print stn_ids[x]
        #schk.increment()

def chk_lc_stns():
    
    ds_tmin = Dataset('/projects/daymet2/station_data/infill/infill_fnl/infill_tmin.nc')
    ds_tmax = Dataset('/projects/daymet2/station_data/infill/infill_fnl/infill_tmax.nc')
    
    stnids_tmin = ds_tmin.variables['stn_id'][:].astype("<S16")
    stnids_tmax = ds_tmax.variables['stn_id'][:].astype("<S16")
    stnids = np.unique(np.concatenate((stnids_tmin,stnids_tmax)))
    
    stn_da = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc',(19480101,20121231))

    stns = stn_da.stns[np.in1d(stn_da.stns[STN_ID], stnids, assume_unique=True)]
    
    #a_rast = modis_sin_rast('/projects/daymet2/climate_office/modis/MOD12Q1/mosaic_lc.tif')
    a_rast = input_raster('/projects/daymet2/climate_office/modis/MOD12Q1/mosaic_lc_wgs84.tif')
    a_lc = a_rast.readEntireRaster()
    
    #schk = status_check(stns.size,500)
    cnt=0
    for stn in stns:
        
        if a_rast.getDataValue(stn[LON],stn[LAT]) == 0:
            
            x,y = a_rast.getGridCellOffset(stn[LON],stn[LAT])
            
            r = 1
            nn = []
            
            while len(nn) == 0:
                
                lcol = x-r
                rcol = x+r
                trow = y-r
                brow = y+r
                
                #top of ring
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
            print stn[STN_ID],d[j],nlat,nlon,nlc       
                            
            cnt+=1
        #schk.increment()
    print cnt
    
def dem_nad_to_wgs():
    dem_nad = gdal.Open('/projects/daymet2/dem/interp_grids/tifs/tdi.tif')
    dem_band = dem_nad.GetRasterBand(1)
    #dst_proj = dem_nad.GetProjection()
    dst_geot = dem_nad.GetGeoTransform()
    dst_dtype = dem_band.DataType
    dst_ndata = dem_band.GetNoDataValue()
    nrows = dem_nad.RasterYSize
    ncols = dem_nad.RasterXSize
    a = dem_nad.ReadAsArray()
    
    driver = gdal.GetDriverByName("GTiff")
    dem_wgs = driver.Create('/projects/daymet2/dem/gtopo/srtm30/tdi_wgs84.tif',
                            ncols,nrows,1,dst_dtype)
    
    wgs84SR = osr.SpatialReference()
    wgs84SR.ImportFromEPSG(4326)
    
    dem_wgs.SetGeoTransform(dst_geot)
    dem_wgs.SetProjection(wgs84SR.ExportToWkt())
    band = dem_wgs.GetRasterBand(1)
    band.SetNoDataValue(dst_ndata)
    band.WriteArray(a,0,0)
    band.FlushCache() 

def chk_rastvals_stns():
    
    ds_path = '/projects/daymet2/station_data/infill/infill_fnl/infill_tmax.nc'
    varname = 'tmax'
    
    stn_da = station_data_infill(ds_path, varname,stn_dtype=DTYPE_STN_BASIC)
    stns= stn_da.stns
    
    #a_rast = modis_sin_rast('/projects/daymet2/climate_office/modis/MOD12Q1/mosaic_lc.tif')
    #a_rast = input_raster('/projects/daymet2/climate_office/modis/MOD12Q1/mosaic_lc_wgs84.tif')
    #a_rast = input_raster('/projects/daymet2/climate_office/modis/MYD11A2/mean_gtiffs3/night/mosaic_mean_lst_tmin_wgs84.tif')
    a_rast = input_raster('/projects/daymet2/climate_office/modis/MOD44B/mosaic_vcf_reclass1km_wgs84.tif')
    ndata = a_rast.ndata
    
    cnt=0
    for stn in stns:
        
        if a_rast.getDataValue(stn[LON],stn[LAT]) == ndata:
            
            print stn
            cnt+=1
    
    print cnt
            
def chk_impute_cnts():
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_fnl/infill_tmin.nc')
    var_impflg = ds.variables['flag_impute']
    stnids = ds.variables['stn_id'][:]
    
    engh_obs = np.zeros(var_impflg.shape[1],dtype=np.bool)
    
    five_yrs = np.round(365.25 * 5.0)
    
    for x in np.arange(var_impflg.shape[1]):
        
        flg = var_impflg[:,x].astype(np.bool)
        
        imp_runs = runs_of_ones_array(flg)
        
        if imp_runs.size > 0:
            max_imp = np.max(imp_runs)
        else:
            max_imp = 0
        
        if max_imp < five_yrs:
            engh_obs[x] = True
    
    print np.sum(engh_obs) 
    print (np.sum(engh_obs)/np.float(engh_obs.size))*100.

def output_tile_nc():     
    ds_mask = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_mask.nc')
    ds_elev = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc')
    ds_attrs = [('elev',ds_elev)]
    atiler = tl.tiler(ds_mask,ds_attrs,250,250,50,50)
    tinfo = atiler.build_tile_grid_info()
    lon = ds_mask.variables['lon'][:]
    lat = ds_mask.variables['lat'][:]
    
    
    ds_tile = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/tiles.nc','w')
    
    #Create 2-dimensions
    ds_tile.createDimension('lat',lat.size)
    ds_tile.createDimension('lon',lon.size)

    latitudes = ds_tile.createVariable('lat','f8',('lat',))
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = lat

    longitudes = ds_tile.createVariable('lon','f8',('lon',))
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = lon
    
    ncdf_var = ds_tile.createVariable('tile',np.int16,('lat','lon',),fill_value=-1)
    ncdf_var.missing_value = -1
    
    for ntile,idtile in tinfo.tile_ids.items():
        
        row,col = tinfo.tile_rc[idtile]
        ncdf_var[row:row+250,col:col+250] = ntile
    
    ds_tile.sync()
    to_geotiff(ds_tile,'tile', '/projects/daymet2/dem/interp_grids/conus/tifs/tiles.tif',nodata_val=-1)

def output_tile_csv():     
    ds_mask = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_mask.nc')
    ds_elev = Dataset('/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc')
    ds_attrs = [('elev',ds_elev)]
    atiler = tl.tiler(ds_mask,ds_attrs,250,250,50,50)
    tinfo = atiler.build_tile_grid_info()
    
    tileIds =  np.sort(np.array(tinfo.tile_ids.keys()))
    
    fout = open('/projects/daymet2/dem/interp_grids/conus/tifs/tileList.csv','w')
    
    for aId in tileIds:
        fout.write(",".join([str(aId),str(tinfo.tile_ids[aId])+"\n"]))
    fout.close()
    
def lcc_for_montana_aoi(lcc_path):
    
    ds = gdal.Open(lcc_path, gdalconst.GA_Update)
    a = ds.ReadAsArray()
    a[a==0] = 13
    a[a==15] = 6 
    
    band = ds.GetRasterBand(1)
    band.WriteArray(a,0,0)
    band.FlushCache() 

def combine_montana_conus_mask():
    
    ds = gdal.Open('/projects/daymet2/dem/interp_grids/tifs/mask_montana_aoi.tif')
    mt_mask = ds.ReadAsArray()
    ds = None
    
    ds = gdal.Open('/projects/daymet2/dem/interp_grids/conus/tifs/mask.tif')
    conus_mask = ds.ReadAsArray()
    ds = None
    
    mask = np.logical_or(conus_mask==1,mt_mask==0)
    
    ds = gdal.Open('/projects/daymet2/dem/interp_grids/tifs/mask_all.tif', gdalconst.GA_Update)
    #a = ds.ReadAsArray()
    #a[mask] = 1
    
#    plt.imshow(a)
#    plt.show()
    
    band = ds.GetRasterBand(1)
    band.WriteArray(mask.astype(np.int8),0,0)
    band.FlushCache()
    ds = None
    

def set_0_lcc():
    
    ds = gdal.Open('/projects/daymet2/dem/interp_grids/tifs/mask_all.tif')
    mask = ds.ReadAsArray()
    ds = None
    
    ds = gdal.Open('/projects/daymet2/dem/interp_grids/tifs/fwpusgs_lcc.tif')
    lcc = ds.ReadAsArray()
    ds = None
    
    ds = gdal.Open('/projects/daymet2/dem/interp_grids/tifs/fwpusgs_lcc2.tif', gdalconst.GA_Update)
    a = ds.ReadAsArray()

    idxs = np.nonzero(np.logical_and(mask==1,lcc==0))

    for r,c in zip(idxs[0],idxs[1]):
        
        if r < 1000:
            a[r,c] = 13
        else:
            a[r,c] = 2
        
    band = ds.GetRasterBand(1)
    band.WriteArray(a,0,0)
    band.FlushCache()
    ds = None
    

def temporal_variability_analysis():
     
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_fnl/serial_tmin.nc','tmin')
    #tmax = stnda.load_obs('SNOTEL_13C01S')#'GHCN_USC00053553')
    
    rm_stns = np.loadtxt('/projects/daymet2/station_data/infill/infill_fnl/rm_stns_all.csv',np.str)
    rm_ids = np.unique(rm_stns)
    stn_ids = stnda.stn_ids[np.logical_not(np.in1d(stnda.stn_ids, rm_ids, True))]
    
    yrs = np.unique(stnda.days[YEAR])
    yr_masks = []
    for yr in yrs:
        yr_masks.append(stnda.days[YEAR]==yr)
    
    ann = np.zeros(yrs.size)
    
    for stn_id in stn_ids:
        
        tair = stnda.load_obs(stn_id)
        
        for x in np.arange(ann.size):
            
            ann[x] = np.std(tair[yr_masks[x]],ddof=1)
    
        annstd = np.std(ann,ddof=1)
        if annstd > 1.2:
            print stn_id,annstd
            plt.plot(tair)
            plt.show()
    
#    ann = []
#    for yr in yrs:
#        
#        ann.append(np.std(tmax[stnda.days[YEAR]==yr],ddof=1))
#
#    print np.std(ann,ddof=1)
#    plt.plot(ann)
#    plt.show()

def overall_interp_err_stats():
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_fnl/xval/xval_tmax_overall.nc')
    rgns = ds.variables['neon'][:]
    urgns = np.unique(rgns)
    
    for rgn in urgns:
        
        mask_rgn = rgns == rgn
        
        print "####################################################"
        print rgn
        print "####################################################"
        
        for astat in ['mean_mae','anom_mae','overall_mae',
                      'mean_bias','anom_bias','overall_bias',
                      'anom_r2','overall_r2']:
            
            print "".join([astat,": ",str(np.mean(ds.variables[astat][mask_rgn]))])
            
def montana_aoi_interp_err_stats():
    ds = Dataset('/projects/daymet2/station_data/infill/infill_fnl/xval/xval_tmax_overall.nc')    
    stnids = ds.variables['neon'][:].astype("<S16")
    
    stnids_aoi = np.loadtxt('/projects/daymet2/station_data/infill/infill_fnl/montana_aoi_stns.csv',np.str, delimiter=",",usecols=[0])
    
    aoi_mask = np.in1d(stnids, stnids_aoi, True)
    
    for astat in ['mean_mae','anom_mae','overall_mae',
                  'mean_bias','anom_bias','overall_bias',
                  'anom_r2','overall_r2']:
        
        print "".join([astat,": ",str(np.mean(ds.variables[astat][aoi_mask]))])
    
def montana_aoi_por_csv():
    
    stnids_aoi = np.loadtxt('/projects/daymet2/station_data/infill/infill_fnl/montana_aoi_stns.csv',np.str, delimiter=",",usecols=[0])
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_fnl/infill_tmax.nc','tmax',stn_dtype=DTYPE_STN_BASIC)
    
    idxs = np.nonzero(np.in1d(stnda.stn_ids,stnids_aoi,True))[0]
    
    limit_miss = (1.0/3.0)*stnda.days.size
    
    fout = open('/projects/daymet2/station_data/infill/infill_fnl/montana_aoi_longstns.csv','w')
    fout.write(",".join(['stnid','lon','lat','lterm\n']))
    
    for x in idxs:
        
        if np.sum(stnda.ds.variables['flag_impute'][:,x]) < limit_miss:
            lterm = 1
        else:
            lterm = 0
       
        fout.write(",".join([stnda.stn_ids[x],str(stnda.stns[LON][x]),str(stnda.stns[LAT][x]),str(lterm)+'\n']))
    fout.close()

def imputation_plot():
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_fnl/infill_tmax.nc')
    stnids = ds.variables['stn_id'][:].astype("<S16")
    
    x = np.nonzero(stnids=='SNOTEL_13C01S')[0][0]
    
    tair = ds.variables['tmax'][:,x]
    fimp = ds.variables['flag_impute'][:,x].astype(np.bool)
    imp =  ds.variables['tmax_imp'][:,x]
    
    tair[fimp] = np.nan
    imp[np.logical_not(fimp)] = np.nan
    
    plt.plot(tair)
    plt.plot(imp)
    plt.ylabel("Celsius")
    plt.xlabel('Day #: 1948-2012')
    plt.title('Stuart Montain SNOTEL: Daily Tmax')
    plt.legend(('Obs','Imp'))
    plt.savefig('/projects/daymet2/docs/magip_presentation/snotel_imp_legend.png')
    plt.show()
    

def fix_lst_data():
    
    fpath = '/projects/daymet2/climate_office/modis/MYD11A2/'
    fnames = np.array(os.listdir(fpath))
    dirs = fnames[np.char.startswith(fnames,'MYD11A2')]
    
    for dirname in dirs:
        
        os.chdir("".join([fpath,dirname]))
        fnames = os.listdir("".join([fpath,dirname]))
        
        for fname in fnames:
        
            d = datetime.strptime(time.ctime(os.path.getmtime(fname)),"%a %b %d %H:%M:%S %Y")
            if d.month == 4 and d.year == 2012:
                print fname 
    
    
    #os.path.getmtime('/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.h10v03/MYD11A2.A2012361.h10v03.005.2013015232747.hdf')
    #d = time.ctime(os.path.getmtime('/projects/daymet2/climate_office/modis/MYD11A2/MYD11A2.005.h10v03/MYD11A2.A2012361.h10v03.005.2013015232747.hdf'))
    #datetime.datetime.strptime(d,"%a %b %d %H:%M:%S %Y")
    
def set_optim_nnghs():
    min_nghs = np.array([   134,
                            52,
                            147,
                            111,
                            92,
                            76,
                            57,
                            92,
                            92,
                            57,
                            101,
                            35,
                            147,
                            101,
                            111,
                            57])
    
    
    
    lccs = np.arange(1,17)
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_fnl/serial_tmin.nc','tmin')
    stn_lcc = stn_da.stns[NEON]
    stn_mask = np.logical_and(stn_da.stns[MASK],np.isnan(stn_da.stns[BAD]))
    stn_da.ds.close()
    stn_da = None
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_fnl/serial_tmin.nc','r+')
    optimnghs = ds.variables[OPTIM_NNGH]
    optimnghs[:] = np.float64(9.96920996838687e+36)
    ds.sync()
    
    for lcc in lccs:
        
        lcc_mask = stn_lcc == np.logical_and(lcc,stn_mask)
        optimnghs[lcc_mask] = min_nghs[lcc-1]
    ds.sync()

def plot_krig_params():
    
    stn_id = 'GHCN_USC00452531'

    stn_da = station_data_infill("/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc", 'tmax')
    mask_stns = it.build_stn_mask(stn_da.stn_ids, "/projects/daymet2/station_data/infill/infill_fnl/rm_stns_all.csv")
    stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=False)
    
    it.init_interp_R_env('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R')

    krigparams = it.BuildKrigParams(stn_slct)
    
    stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
                
    min_ngh = stn[OPTIM_NNGH]
    max_ngh = min_ngh+np.round(min_ngh*0.20) 
    nug,psill,rng = krigparams.get_krig_params(stn, min_ngh, max_ngh)
    print nug,psill,rng
    print "done"


def test_xval_krig():
    
    stn_id = 'SNOTEL_13C01S'
    
    stn_da = station_data_infill("/projects/daymet2/station_data/infill/infill_fnl/serial_tmin.nc", 'tmin')
    mask_stns = np.isnan(stn_da.stns[BAD])         
    stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
    xval_stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
    
    it.init_interp_R_env('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R')
        
    krig = it.KrigTair2(stn_slct)
    tair_mean,tair_var = krig.krig(xval_stn, np.array([xval_stn[STN_ID]]))
    print tair_mean,tair_var

def new_krig_xval_stats():
    ds = Dataset('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmin.nc')
    #ds = Dataset('/projects/daymet2/station_data/infill/infill_fnl/serial_tmin.nc')
    inci = ds.variables['xval_inci'][:]
    err = ds.variables['xval_err'][:]
    
    divs = ds.variables['lcc'][:]
    if np.ma.isMA(divs):
        lccs = np.unique(divs.data[np.logical_not(divs.mask)])
    else:
        lccs = np.unique(divs)
    
    test_mask = np.logical_not(inci.mask)
    
    o_err = err[test_mask]
    o_inci = inci[test_mask]
    
    lcc_mae = np.mean(np.abs(o_err))
    lcc_bias = np.mean(o_err)
    lcc_pctci = (np.sum(o_inci==1)/np.float(o_inci.size))*100.0
    
    print "|".join(["Overall",str(lcc_mae),str(lcc_bias),str(lcc_pctci)])
    
    
    for lcc in lccs:
        
        lcc_mask = np.logical_and(test_mask,divs==lcc)
        
        lcc_err = err[lcc_mask]
        lcc_inci = inci[lcc_mask]
        
        lcc_mae = np.mean(np.abs(lcc_err))
        lcc_bias = np.mean(lcc_err)
        lcc_pctci = (np.sum(lcc_inci==1)/np.float(lcc_inci.size))*100.0
        
        print "|".join([str(lcc),str(lcc_mae),str(lcc_bias),str(lcc_pctci)])
        
        
def optim_krigsmth():
    ds = Dataset('/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc')
    inci = ds.variables['inci'][:]
    lcc_stns = ds.variables['neon'][:]      
    
    test_mask = np.logical_not(inci[0,:].mask)
    
    lccs = np.arange(1,17)
    
    for lcc in lccs:
        
        lcc_mask = np.logical_and(test_mask,lcc_stns==lcc)
        lcc_inci = inci[:,lcc_mask]
        
        print np.sum(lcc_inci,axis=1)/np.float(lcc_inci.shape[1])

def run_full_interp():
    
    stn_id = 'GHCN_USC00146549'
    
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc', 'tmax')
    mask_stns = np.isnan(stn_da.stns[BAD])         
    stn_slct = station_select(stn_da, stn_mask=mask_stns, rm_zero_dist_stns=True)
    print stn_da.stns[stn_da.stn_ids==stn_id][0]
    
    it.init_interp_R_env('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R')
        
    krig_tair = it.KrigTair(stn_slct)
    gwr_tair = it.GwrPcaTairDynamic(stn_slct, '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', set_pt=False)
    interp_tair = it.InterpTair(krig_tair, gwr_tair)
    
    xval_stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
    xval_mean = xval_stn[MEAN_OBS]
    xval_obs = stn_da.load_obs(xval_stn[STN_ID])
    
    tair_daily, tair_mean, std_err, ci = interp_tair.interp(xval_stn, np.array([xval_stn[STN_ID]]))
                                        
    err = tair_daily - xval_obs
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    print mae,bias
    
def overall_xval_stats():
        
    ds = Dataset('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmin.nc')
    mae = ds.variables['xval_overall_mae'][:]
    bias = ds.variables['xval_overall_bias'][:]
    
    divs = ds.variables['neon'][:]
    if np.ma.isMA(divs):
        rgns = np.unique(divs.data[np.logical_not(divs.mask)])
    else:
        rgns = np.unique(divs)
    
    
    test_mask = np.logical_not(mae.mask)
    
    o_mae = mae[test_mask]
    o_bias = bias[test_mask]
    
    mmae = np.mean(o_mae)
    mbias = np.mean(o_bias)
    abias = np.mean(np.abs(o_bias))
    print "|".join(["Overall",str(mmae),str(mbias),str(abias)])
    
    for rgn in rgns:
        
        rgn_mask = np.logical_and(test_mask,divs==rgn)
        
        rgn_mae = mae[rgn_mask]
        rgn_bias = bias[rgn_mask]
        
        rgn_mmae = np.mean(rgn_mae)
        rgn_mbias = np.mean(rgn_bias)
        rgn_abias = np.mean(np.abs(rgn_bias))
        
        print "|".join([str(rgn),str(rgn_mmae),str(rgn_mbias),str(rgn_abias)])    

def summary_stats_chunks():
    
    ds = Dataset('/projects/daymet2/interp_output/montana/h05v02/h05v02_tmin.nc')
    #ds = Dataset('/projects/daymet2/daymet_oakridge/multi_yr_tiles/12274_tmin_1980_2011.nc')
    varname = 'tmin'
    dates = num2date(ds.variables['time'][:],units=ds.variables['time'].units)
    days = utld.get_days_metadata_dates(dates)
    
    mthMask = np.logical_and(days[MONTH]>=6,days[MONTH]<=8)
    yr_masks = []
    yrs = np.unique(days[YEAR])
    for yr in yrs:
        
        yr_masks.append(np.logical_and(mthMask,days[YEAR] == yr))
    

    idx2007 = np.int(np.argwhere(yrs==1993))
    
    avar = ds.variables[varname]
    rstep,cstep = avar.chunking()[1:]
    #rstep,cstep = 2,2
    nrow,ncol = avar.shape[1:]
    
    a_out = np.zeros(avar.shape[1:])
    
    for y in np.arange(nrow,step=rstep):
        
        
        for x in np.arange(ncol,step=cstep):
            
            print y,x
            
            a = avar[:,y:y+rstep,x:x+cstep].astype(np.float)
            
            yrmeans = np.zeros((yrs.size,a.shape[1],a.shape[2]))
            
            for i in np.arange(yrs.size):
                
                yrmeans[i,:,:] = np.mean(a[yr_masks[i],:,:],axis=0)
            
            norm = np.mean(yrmeans,axis=0)
            anom2007 = yrmeans[idx2007,:,:] - norm
            
            a_out[y:y+rstep,x:x+cstep] = anom2007
    
    #toGTiff(a_out, ds,'/projects/daymet2/interp_output/montana/h05v02/h05v02_tmin_2007anom.tif')
    
    plt.imshow(a_out)
    plt.colorbar()
    plt.show()
    

def summary_stats_daymet():
    
    #ds = Dataset('/projects/daymet2/interp_output/montana/h05v02/h05v02_tmin.nc')
    ds = Dataset('/projects/daymet2/daymet_oakridge/multi_yr_tiles/12274_tmax_1980_2011.nc')
    varname = 'tmax'
    dates = num2date(ds.variables['time'][:],units=ds.variables['time'].units)
    days = utld.get_days_metadata_dates(dates)
    
    yr_masks = []
    yrs = np.unique(days[YEAR])
    for yr in yrs:
  
        yr_masks.append(days[YEAR] == yr)
        
    avar = ds.variables[varname]
    
    yrMeans = None
    
    for yrMask in yr_masks:
        
        avarYr = np.ma.mean(avar[yrMask,:,:],axis=0)
        avarYr.shape = (1,avarYr.shape[0],avarYr.shape[1])
    
        if yrMeans == None:
            yrMeans = avarYr
        else:
            yrMeans = np.ma.vstack((yrMeans,avarYr))
        print yrMeans.shape
    
    idx2007 = np.nonzero(yrs==2007)[0][0]
    
    annMean = np.ma.mean(yrMeans,axis=0)
    plt.imshow(yrMeans[idx2007,:,:]-annMean)  
    plt.colorbar()
    plt.show()       

def discont_analysis_daily():
    
    lon1,lat1 = -112.602009, 45.358391
    lon2,lat2 = -112.532766,45.276300
    
    tair_daily1, tair_mean1, std_err1, ci1 = run_full_interp_pt(lon1, lat1)
    tair_daily2, tair_mean2, std_err2, ci2 = run_full_interp_pt(lon2, lat2)
    
    plt.plot((tair_daily1-tair_mean1)-(tair_daily2-tair_mean2))
    plt.show()

def run_full_interp_pt_xval():
    
    #lon,lat,elev = -113.430344,47.230488,1405 #seeley
    #lon,lat,elev = -110.587747,45.096985,2389 #gardiner
    
    auxFpaths = ['/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_climdiv.nc']
            
    ptInterper = it.PtInterpTair('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc',
                    '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc',
                    '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                    '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                    auxFpaths)
    xvalId = 'RAWS_MGAT'
    xvalStnTmax = ptInterper.stn_da_tmax.stns[ptInterper.stn_da_tmax.stn_ids==xvalId][0]
    xvalStnTmin = ptInterper.stn_da_tmin.stns[ptInterper.stn_da_tmin.stn_ids==xvalId][0]
    xvalStn = xvalStnTmax
    
    rm_stnid = np.array([xvalId])
    obstmax = ptInterper.stn_da_tmax.load_obs(xvalId)
    obstmin = ptInterper.stn_da_tmin.load_obs(xvalId)
    
    
    agg = ushcn.TairAggregate(ptInterper.stn_da_tmax.days)
        
    aPt = it.build_empty_pt()
    aPt[LON] = xvalStn[LON]
    aPt[LAT] = xvalStn[LAT]
    aPt[ELEV] = xvalStn[ELEV]
    aPt[TDI] = xvalStn[TDI]
    aPt[LST_TMAX] = xvalStnTmax[LST]
    aPt[LST_TMIN] = xvalStnTmin[LST]
    aPt[NEON] = xvalStn[NEON]
    aPt[MEAN_OBS] = xvalStn[MEAN_OBS]
    aPt[OPTIM_NNGH] = xvalStn[OPTIM_NNGH]
    aPt[OPTIM_NNGH_ANOM] = xvalStn[OPTIM_NNGH_ANOM]
    
#    aPt[LON] = lon
#    aPt[LAT] = lat
#    ptInterper.pGrids.setPtValues(aPt, chgLatLon=False)
#    aPt[ELEV] = elev
    
    ptInterper.a_pt = aPt
    
#    stn_id = 'GHCN_USC00366879'#'GHCN_USW00014837'#SNOTEL_13C01S
#    idx_stn = ptInterper.interp_tmin.krig_tair.stn_slct.stn_da.stn_idxs[stn_id]
#    xval_stn = ptInterper.interp_tmin.krig_tair.stn_slct.stn_da.stns[idx_stn]
#    stn_da = ptInterper.interp_tmax.krig_tair.stn_slct.stn_da
#    aPt = it.build_empty_pt()
#    aPt[LON] = xval_stn[LON]
#    aPt[LAT] = xval_stn[LAT]
#    elev = xval_stn[ELEV]
#    ptInterper.pGrids.setPtValues(aPt, chgLatLon=False)
#    aPt[ELEV] = elev
#    ptInterper.a_pt = aPt
    tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpPt(rm_stnid=rm_stnid)
    #tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpLonLatPt(-113.755,47.675)
    print np.mean(obstmax),tmax_mean
    print tmin_se,tmax_se,ninvalid
    plt.subplot(121)
    plt.plot(tmin_dly)
    ylim = plt.ylim()
    plt.title('tmin')
    plt.subplot(122)
    plt.plot(obstmin)
    plt.title('Obs tmin')
    plt.ylim(ylim)
    plt.show()
    
    print np.mean(tmin_dly-obstmin),np.mean(np.abs(tmin_dly-obstmin))
#    plt.subplot(121)
#    plt.plot(tmax_dly-obstmax)
#    plt.subplot(122)
#    plt.plot(tmax_dly,obstmax,'*')
#    plt.show()

    tminMth = agg.dailyToAnn(tmin_dly)
    obtminMth = agg.dailyToAnn(obstmin)
    tmaxMth = tminMth - np.mean(tminMth)
    obtmaxMth = obtminMth - np.mean(obtminMth)
    plt.plot(obtminMth)
    plt.plot(tminMth)
    plt.show()
#    tairObs = stn_da.load_obs(stn_id)
#    
#    plt.plot(tairObs[-100:])
#    plt.plot(tmax_dly[-100:])
#    plt.show()
    
#    fout = open('/projects/daymet2/hare_sites/hare_wxTopo/tairTopoWx20130627/TairOldSeeley.csv','w')
#    fout.write(",".join(['YMD','TMIN','TMAX\n']))
#    for x in np.arange(tmin_dly.size):
#        fout.write("{0:d},{1:.2f},{2:.2f}\n".format(days[YMD][x],tmin_dly[x],tmax_dly[x]))
#    fout.close()

def run_full_interp_pt():
    
    #lon,lat,elev = -113.430344,47.230488,1405 #seeley
    #lon,lat,elev = -110.587747,45.096985,2389 #gardiner
    
    path = '/projects/daymet2/dem/interp_grids/conus/ncdf/'
    
    auxFpaths = ["".join([path,'fnl_elev.nc']),
                 "".join([path,'fnl_tdi.nc']),
                 "".join([path,'fnl_climdiv.nc'])]
    
    for mth in np.arange(1,13):
        auxFpaths.append("".join([path,'fnl_lst_tmin%02d.nc'%mth]))
        auxFpaths.append("".join([path,'fnl_lst_tmax%02d.nc'%mth]))
    
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
    
    ptInterper = it.PtInterpTair(stndaTmin,stndaTmax,
                                 '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                                 '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                                 auxFpaths)
    
    tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterper.interpLonLatPt(-114.016758,48.825263)
    print tmin_norms[1]
    tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterper.interpLonLatPt(-114.033652,48.825501)
    print tmin_norms[1]
    
#    print ninvalid
#    plt.plot(tmin_dly)
#    plt.show()
    
def discont_analysis():
    
    ds = Dataset('/projects/daymet2/interp_output/montana/h05v02/h05v02_tmin.nc')
    dates = num2date(ds.variables['time'][:],units=ds.variables['time'].units)
    days = utld.get_days_metadata(dates[0], dates[-1])
    
    yr_masks = []
    yrs = np.unique(days[YEAR])
    for yr in yrs:
        yr_masks.append(days[YEAR] == yr)
    
    lon1,lat1 = -112.602009, 45.358391
    lon2,lat2 = -112.532766,45.276300
    
    gds = GeoNc(ds)      
    row1,col1 =  gds.getRowCol(lon1,lat1)  
    row2,col2 =  gds.getRowCol(lon2,lat2)  
    
    a1 = ds.variables['tmin'][:,row1,col1].astype(np.float)
    a2 = ds.variables['tmin'][:,row2,col2].astype(np.float)
    
    yrmeans1 = np.zeros(yrs.size)
    yrmeans2 = np.zeros(yrs.size)
    
    for i in np.arange(yrs.size):
        
        yrmeans1[i] = np.mean(a1[yr_masks[i]])
        yrmeans2[i] = np.mean(a2[yr_masks[i]])
    
    plt.plot(yrs,(yrmeans1-np.mean(yrmeans1))-(yrmeans2-np.mean(yrmeans2)),'.-')
    plt.show()
    
def unusable_pha_stns():
     
    stn_da = station_data_infill('/projects/daymet2/station_data/infill/infill_fnl/serial_tmax.nc', 'tmax')
    stn_da_homog = station_data_infill('/projects/daymet2/station_data/infill/infill_fnl/serialhomog_tmax.nc', 'tmax')
     
    stns = stn_da.stns[np.isnan(stn_da.stns[BAD])]
    stnsh = stn_da_homog.stns[np.isnan(stn_da_homog.stns[BAD])]
    
    stns_nopha = stns[np.logical_not(np.in1d(stns[STN_ID], stnsh[STN_ID], True))]
    
#    stnda = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
#    ds = Dataset('/projects/daymet2/station_data/infill/xval_impute_norm.nc')
#    stnids = ds.variables['stn_id'][:].astype("<S16")
#    
#    stns_nopha = stnda.stns[np.in1d(stnda.stn_ids, stnids, True)]
    
    print stns_nopha[STN_ID]
    m = Basemap(projection='cyl',llcrnrlat=np.min(stns_nopha[LAT])-1,urcrnrlat=np.max(stns_nopha[LAT])+1,\
                llcrnrlon=np.min(stns_nopha[LON])-1,urcrnrlon=np.max(stns_nopha[LON])+1,resolution='c')
    
    m.scatter(stns_nopha[LON],stns_nopha[LAT])
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    
    plt.show()
    
    
def imputeDailyNoXval():
    
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
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/infill_20130518/' 
    params[P_PATH_NNR] = '/projects/daymet2/reanalysis_data/conus_subset/'
    params[P_PATH_R_FUNCS] = ['/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R',
                              '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R']
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    params[P_MIN_NNGH_DAILY] = 3
    params[P_NNGH_NNR] = 4
    params[P_NNR_VARYEXPLAIN] = 0.99
    params[P_FRACOBS_INIT_PCS] = 0.5
    params[P_PPCA_VARYEXPLAIN] = 0.99
    params[P_CHCK_IMP_PERF] = True
    params[P_NPCS_PPCA] = 0
    
    for path in params[P_PATH_R_FUNCS]:    
        source_r(path)
    
    stn_id = 'GHCN_USC00247204'
    tair_var = 'tmin'
    update_serial = False
    doXval = False
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    stn_masks = {}
    stn_masks['tmin'] = np.isfinite(stn_da.stns[MEAN_TMIN])
    stn_masks['tmax'] = np.isfinite(stn_da.stns[MEAN_TMAX])
    
    
    print stn_da.stns[stn_da.stn_ids==stn_id][0]
    
    xval_mask = xval.build_xval_masks(np.array([stn_id]), 5, stn_da, tair_var)[0]
    if not doXval:
        xval_mask[:] = False
    
    obs = stn_da.load_all_stn_obs_var(stn_id, tair_var)[0]
    
    plt.plot(np.arange(stn_da.days.size),obs)
    plt.show()
    
    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD],params[P_END_YMD]))
    aclib = clib_wxTopo(params[P_PATH_CLIB])
    
    #Estimate normal
    norm_est,va_est = impute_tair_norm(stn_id, stn_da, stn_masks[tair_var], tair_var,ds_nnr,aclib,
                                       nnghs=params[P_MIN_NNGH_DAILY],tair_mask=xval_mask,trim_nan=False)[0]
                                       
    i = np.nonzero(stn_da.stn_ids==stn_id)[0][0]
    
    norm_orig = stn_da.stns["_".join(["mean",tair_var])][i]
    va_orig = stn_da.stns["_".join(["var",tair_var])][i]
    
    stn_da.stns["_".join(["mean",tair_var])][i] = norm_est
    stn_da.stns["_".join(["var",tair_var])][i] = va_est
    
    a_pca_matrix = ImputeMatrixPCA(stn_id, stn_da, tair_var,ds_nnr,aclib,tair_mask=xval_mask)
    
    fit_tair, obs_tair, npcs, fnl_nnghs, max_dist = a_pca_matrix.impute(min_daily_nnghs=params[P_MIN_NNGH_DAILY],
                                                                        nnghs_nnr=params[P_NNGH_NNR],
                                                                        max_nnr_var=params[P_NNR_VARYEXPLAIN],
                                                                        chk_perf=params[P_CHCK_IMP_PERF],
                                                                        npcs=params[P_NPCS_PPCA],
                                                                        frac_obs_initnpcs=params[P_FRACOBS_INIT_PCS],
                                                                        ppca_varyexplain=params[P_PPCA_VARYEXPLAIN])
    #np.savetxt('/projects/daymet2/station_data/infill/infill_20130518/chgptTest_GHCN_CA001175122.csv', fit_tair,fmt="%.2f", delimiter=',')
    #Check for extreme values to see if the imputation converged to a reasonable solution
#    if np.sum(fit_tair > 57.7) > 0 or np.sum(fit_tair < -89.4) > 0:
#        print "".join(["WARNING|",a_pca_matrix.stn_id,
#                       " appears to have bad imputation convergence for ",a_pca_matrix.tair_var])

    fnl_tair = np.copy(obs_tair)
    fill_mask = np.isnan(fnl_tair)
    fnl_tair[fill_mask] = fit_tair[fill_mask]
    
    if not doXval:
        xval_mask = np.logical_not(fill_mask)
    xval_fit = fit_tair[xval_mask]
    xval_obs = obs[xval_mask]
    
    mae = np.mean(np.abs(xval_fit-xval_obs))
    bias = np.mean(xval_fit-xval_obs)
    r_value = stats.linregress(xval_fit, xval_obs)[2]
    var_pct = r_value**2 #r-squared value; variance explained
    
    
    
#    difs = fit_tair[fin_mask] - obs_tair[fin_mask]
#    mae = np.mean(np.abs(difs))
#    bias = np.mean(difs)
#    
#    r_value = stats.linregress(fit_tair[fin_mask], obs_tair[fin_mask])[2]
#    vary_pct = r_value**2 #r-squared value; variance explained
    
    print mae,bias,var_pct
    plt.plot(fit_tair)
    plt.show()
    
    #return fnl_tair,fill_mask,fit_tair,npcs,fnl_nnghs,max_dist,mae,bias,vary_pct
#    if update_serial:
#        ds_out = Dataset("".join([params[P_PATH_OUT],'infill_',tair_var,'.nc']),'r+')
#        stnids_out = np.array(ds_out.variables['stn_id'][:], dtype="<S16")
#        stn_idx = np.nonzero(stnids_out == stn_id)[0][0]
#        
#        ds_out.variables['npcs'][stn_idx] = npcs
#        ds_out.variables['mae'][stn_idx] = mae
#        ds_out.variables['bias'][stn_idx] = bias
#        ds_out.variables['r2'][stn_idx] = vary_pct
#        ds_out.variables['max_dist'][stn_idx] = max_dist
#        ds_out.variables[tair_var][:,stn_idx] = fnl_tair
#        ds_out.variables["".join([tair_var,"_imp"])][:,stn_idx] = fit_tair
#        ds_out.variables["".join([tair_var,"_mean"])][stn_idx] = np.mean(fnl_tair,dtype=np.float64)
#        ds_out.variables['flag_impute'][:,stn_idx] = fill_mask
#        ds_out.variables[LAST_VAR_WRITTEN][stn_idx] = fnl_nnghs

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
    params[P_PATH_OUT] = '' 
    params[P_PATH_NNR] = '/projects/daymet2/reanalysis_data/conus_subset/'
    params[P_PATH_R_FUNCS] = ['/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R',
                              '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R']
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    params[P_MIN_NNGH_DAILY] = 3
    params[P_NNGH_NNR] = 4
    params[P_NNR_VARYEXPLAIN] = 0.99
    params[P_FRACOBS_INIT_PCS] = 0.5
    params[P_PPCA_VARYEXPLAIN] = 0.99
    params[P_CHCK_IMP_PERF] = True
    params[P_NPCS_PPCA] = 0
    
    for path in params[P_PATH_R_FUNCS]:    
        source_r(path)
    
    stn_id = 'GHCN_CA001016203'
    tair_var = 'tmax'
    ppcaConThres=1e-5
    runUpdate = False
    tair_mask = None
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    stn_masks = {}
    stn_masks['tmin'] = np.isfinite(stn_da.stns[MEAN_TMIN])
    stn_masks['tmax'] = np.isfinite(stn_da.stns[MEAN_TMAX])
    
    print stn_da.stns[stn_da.stn_ids==stn_id][0]
    
    obs = stn_da.load_all_stn_obs_var(stn_id, tair_var)[0]
#    tair_mask = np.arange(obs.size) > 20000
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

def imputeNormNoXval():
    
    P_PATH_DB = 'P_PATH_DB'
    P_PATH_POR = 'P_PATH_POR'
    P_PATH_R_FUNCS = 'P_PATH_R_FUNCS'
    P_PATH_NNR = 'P_PATH_NNR'
    P_PATH_CLIB = 'P_PATH_CLIB'
    
    P_START_YMD = 'P_START_YMD'
    P_END_YMD = 'P_END_YMD'
    P_MIN_POR = 'P_MIN_POR'
    P_STN_LOC_BNDS = 'P_STN_LOC_BNDS'
    P_MIN_NNGH_DAILY = 'P_MIN_NNGH_DAILY'
    
    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    params[P_PATH_R_FUNCS] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R'
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_PATH_NNR] = '/projects/daymet2/reanalysis_data/conus_subset/'
    params[P_MIN_POR] = 5  #2 for CCE
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    params[P_MIN_NNGH_DAILY] = 7
    #left,right,bottom,top
    #params[P_STN_LOC_BNDS] = (-118.5,-109.2,44.0,52.6) #Crown of the Continent
    params[P_STN_LOC_BNDS] = (-126.0,-64.0,22.0,53.0) #CONUS
    
    stn_id = 'RAWS_NTEWGHCN_USC00244038'
    tair_var = 'tmax'
    
    por = load_por_csv(params[P_PATH_POR])
    mask_por_tmin,mask_por_tmax,mask_por_prcp = build_valid_por_masks(por,params[P_MIN_POR],params[P_STN_LOC_BNDS])
    
    source_r(params[P_PATH_R_FUNCS])
    
    stn_masks = {'tmin':mask_por_tmin,'tmax':mask_por_tmax}
    
    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD],params[P_END_YMD]))
        
    aclib = clib_wxTopo(params[P_PATH_CLIB])
    
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    norm,vary = impute_tair_norm(stn_id, stn_da, stn_masks[tair_var],tair_var,ds_nnr,aclib,nnghs=params[P_MIN_NNGH_DAILY])[0]
    print norm,vary

def resetInfills():
    
    LAST_VAR_WRITTEN = 'nnghs'
    fillval = netCDF4.default_fillvals['i2']
    
    log = open('/projects/daymet2/station_data/infill/infill_20130518/impute_20130522.log')
    
    dstmin = Dataset('/projects/daymet2/station_data/infill/infill_20130518/infill_tmin.nc','r+')
    dstmax = Dataset('/projects/daymet2/station_data/infill/infill_20130518/infill_tmax.nc','r+')
    
    stnidsTmin = dstmin.variables['stn_id'][:].astype("<S16")
    stnidsTmax = dstmax.variables['stn_id'][:].astype("<S16")
    
    varTmin = dstmin.variables[LAST_VAR_WRITTEN]
    varTmax = dstmax.variables[LAST_VAR_WRITTEN]
    
    rstnids_tmin = []
    rstnids_tmax = []
    
    for aline in log.readlines():
        
        if aline.startswith('ERROR'):
            
            asplit = aline.split()
            
            stn_id = asplit[0].split("|")[1]
            
            tair_var = asplit[5]
            
            if tair_var == 'tmin':
                rstnids_tmin.append(stn_id)
            elif tair_var == 'tmax':
                rstnids_tmax.append(stn_id)
     
    rstnids_tmin = np.unique(np.array(rstnids_tmin,dtype="<S16"))
    rstnids_tmax = np.unique(np.array(rstnids_tmax,dtype="<S16"))
    
    varTmin[np.in1d(stnidsTmin, rstnids_tmin, True)] = fillval
    dstmin.sync()
    varTmax[np.in1d(stnidsTmax, rstnids_tmax, True)] = fillval
    dstmax.sync()

def getNobsPerDay(masks,missflgs):
    
    nobs = []
    for aMask in masks:
        nobs.append(np.sum(missflgs[aMask]))
    return np.array(nobs)

def stnDataFilesRuben():
    
    nodata = -999.00
    
    stnda = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    stndaH = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc')
    stndaSTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stndaSTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
    
    days = stnda.days
    
    allNoData = np.ones(stnda.days.size)*nodata
    allNoFlgs = np.array(['']*stnda.days.size)
    
    ds = Dataset('/projects/daymet2/station_data/forRuben/StnGridCellInterps.nc')
    stnids = ds.variables['stn_id'][:].astype('<S16')
    
    pathOut = "/projects/daymet2/station_data/forRuben/StnInterps/obsFiles/"
    
    stchk = status_check(stnids.size, 50)
    
    for aId,x in zip(stnids,np.arange(stnids.size)):
        
        if stndaSTmin.stn_idxs.has_key(aId):
            
            rawTmin,qaTmin = stnda.load_all_stn_obs_var(aId, 'tmin', set_flagged_nan=True)
            rawTmin[np.isnan(rawTmin)] = nodata
            
            hTmin = stndaH.load_all_stn_obs_var(aId, 'tmin', set_flagged_nan=True)[0]
            hTmin[np.isnan(hTmin)] = nodata
            
            sTmin = stndaSTmin.load_obs(aId)
        
        else:
            
            rawTmin,qaTmin,hTmin,sTmin = allNoData,allNoFlgs,allNoData,allNoData
            
        if stndaSTmax.stn_idxs.has_key(aId):
            
            rawTmax,qaTmax = stnda.load_all_stn_obs_var(aId, 'tmax', set_flagged_nan=True)
            rawTmax[np.isnan(rawTmax)] = nodata
            
            hTmax = stndaH.load_all_stn_obs_var(aId, 'tmax', set_flagged_nan=True)[0]
            hTmax[np.isnan(hTmax)] = nodata
            
            sTmax = stndaSTmax.load_obs(aId)
        
        else:
            
            rawTmax,qaTmax,hTmax,sTmax = allNoData,allNoFlgs,allNoData,allNoData
    
        interpTmin = ds.variables['tmin'][:,x]
        interpTmax = ds.variables['tmax'][:,x]
        
        fout = open("".join([pathOut,aId,".csv"]),'w')
        fout.write(",".join(['YMD','RAW_TMAX','RAW_TMIN','HOMOG_TMAX','HOMOG_TMIN',
                             'SERIAL_TMAX','SERIAL_TMIN','INTERP_TMAX','INTERP_TMIN\n']))
        
        for i in np.arange(days.size):
            
            line = [str(days[YMD][i]),"{0:.2f}".format(rawTmax[i]),"{0:.2f}".format(rawTmin[i]),
                    "{0:.2f}".format(hTmax[i]),"{0:.2f}".format(hTmin[i]),
                    "{0:.2f}".format(sTmax[i]),"{0:.2f}".format(sTmin[i]),
                    "{0:.2f}".format(interpTmax[i]),"{0:.2f}".format(interpTmin[i])]
            
            fout.write(",".join(line)+"\n")
        stchk.increment()
    
def prcpStnDataFilesRuben():
    
    nodata = -999.00
    
    stnda = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    stnids1 = np.loadtxt('/projects/daymet2/station_data/forRuben/prcp19710101-20001231.csv',np.str,delimiter=",",skiprows=1, usecols=[0])
    stnids2 = np.loadtxt('/projects/daymet2/station_data/forRuben/prcp19810101-20101231.csv',np.str,delimiter=",",skiprows=1, usecols=[0])
    stnids = np.unique(np.concatenate((stnids1,stnids2)))
    days = stnda.days
            
    pathOut = "/projects/daymet2/station_data/forRuben/prcpObsFiles/"
    
    stchk = status_check(stnids.size, 50)
    
    for aId,x in zip(stnids,np.arange(stnids.size)):
            
        rawPrcp,qaPrcp = stnda.load_all_stn_obs_var(aId, 'prcp', set_flagged_nan=False)
        rawPrcp[np.isnan(rawPrcp)] = nodata
                
        fout = open("".join([pathOut,aId,".csv"]),'w')
        fout.write(",".join(['YMD','RAW_PRCP','RAW_PRCP_QA\n']))
        
        for i in np.arange(days.size):
            
            line = [str(days[YMD][i]),"{0:.4f}".format(rawPrcp[i]),str(qaPrcp[i])]
            fout.write(",".join(line)+"\n")
        
        stchk.increment()

def stnMetaFileRuben():
    
    stndaSTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/infill_tmin.nc','tmin',stn_dtype=[(STN_ID, "<S16"), (STN_NAME, "<S30"), (LON, np.float64), (LAT, np.float64), (ELEV, np.float64)])
    
    stndaSTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc','tmax')
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc','tmax',stn_dtype=[(STN_ID, "<S16"), (STN_NAME, "<S30"), (LON, np.float64), (LAT, np.float64), (ELEV, np.float64)])
    
    missFlgsTmin = np.logical_not(stndaTmin.ds.variables['flag_impute'][:].astype(np.bool))
    missFlgsTmax = np.logical_not(stndaTmax.ds.variables['flag_impute'][:].astype(np.bool))
    
    ds = Dataset('/projects/daymet2/station_data/forRuben/StnGridCellInterps.nc')
    stnids = ds.variables['stn_id'][:].astype('<S16')
    
    days365 = stndaSTmin.days[stndaSTmin.days[YEAR]==2003]
    zeroObs = np.zeros(days365.size,dtype=np.int)
    days = stndaSTmin.days
    dayHeader = [adate.strftime("%d-%b") for adate in days365['DATE']]
    dayHeader[-1] = dayHeader[-1]+"\n"
    dayHeader = "STNID,"+",".join(dayHeader)
    
    foutMeta = open('/projects/daymet2/station_data/forRuben/StnInterps/stns.csv','w')
    foutMeta.write(",".join(["STNID","NAME","LON","LAT","ELEV","GRIDCELL_LON","GRIDCELL_LAT","GRIDCELL_ELEV",
                             "GRIDCELL_LST_TMAX","GRIDCELL_LST_TMIN","GRIDCELL_TDI"
                             "GRIDCELL_MEAN_TMAX","GRIDCELL_MEAN_TMIN","GRIDCELL_SE_TMAX","GRIDCELL_SE_TMIN"
                             "TMAX_1971-2000_20obs","TMAX_1981-2010_20obs","TMIN_1971-2000_20obs","TMIN_1981-2010_20obs\n"]))
    
    foutTmax71 = open('/projects/daymet2/station_data/forRuben/StnInterps/nobsTmax_1971-2000.csv','w')
    foutTmax81 = open('/projects/daymet2/station_data/forRuben/StnInterps/nobsTmax_1981-2010.csv','w')
    foutTmin71 = open('/projects/daymet2/station_data/forRuben/StnInterps/nobsTmin_1971-2000.csv','w')
    foutTmin81 = open('/projects/daymet2/station_data/forRuben/StnInterps/nobsTmin_1981-2010.csv','w')
    nobsFiles = [foutTmax71,foutTmax81,foutTmin71,foutTmin81]
    
    for afile in nobsFiles:
        afile.write(dayHeader)
    
    normMask71 = np.logical_and(days[YEAR]>=1971,days[YEAR]<=2000)
    dayMasks71 = []
    for aDay in days365:
        dayMasks71.append(np.logical_and(np.logical_and(days[MONTH]==aDay[MONTH],days[utld.DAY]==aDay[utld.DAY]),normMask71))
    
    normMask81 = np.logical_and(days[YEAR]>=1981,days[YEAR]<=2010)
    dayMasks81 = []
    for aDay in days365:
        dayMasks81.append(np.logical_and(np.logical_and(days[MONTH]==aDay[MONTH],days[utld.DAY]==aDay[utld.DAY]),normMask81))
     
    stchk = status_check(stnids.size,100)
    for aId,x in zip(stnids,np.arange(stnids.size)):
        
        try:
            idx_stn = stndaSTmin.stn_idxs[aId]
            stn = stndaSTmin.stns[idx_stn]
        except KeyError:
            idx_stn = stndaSTmax.stn_idxs[aId]
            stn = stndaSTmax.stns[idx_stn]
                
        name = stn[STN_NAME].replace(","," ")
        lon = "{0:.5f}".format(stn[LON])
        lat = "{0:.5f}".format(stn[LAT])
        elev = "{0:d}".format(np.int(np.round(stn[ELEV])))
        gcLon = "{0:.5f}".format(ds.variables['lon'][x])
        gcLat = "{0:.5f}".format(ds.variables['lat'][x])
        gcElev = "{0:d}".format(np.int(np.round(ds.variables['elev'][x])))
        gcLstTmin = "{0:.2f}".format(ds.variables['lst_tmin'][x])
        gcLstTmax = "{0:.2f}".format(ds.variables['lst_tmax'][x])
        gcTdi = "{0:.2f}".format(ds.variables['tdi'][x])
        gcMeanTmax = "{0:.2f}".format(ds.variables['mean_tmax'][x])
        gcMeanTmin = "{0:.2f}".format(ds.variables['mean_tmin'][x])
        gcSeTmax = "{0:.5f}".format(ds.variables['se_tmax'][x])
        gcSeTmin = "{0:.5f}".format(ds.variables['se_tmin'][x])
        
        idxStnTmin = None
        idxStnTmax = None
        
        if stndaTmin.stn_idxs.has_key(aId):
            idxStnTmin = stndaTmin.stn_idxs[aId]
            
        if stndaTmax.stn_idxs.has_key(aId):
            idxStnTmax = stndaTmax.stn_idxs[aId]
        
        if idxStnTmin is not None:
            flgTminStn = missFlgsTmin[:,idxStnTmin]
            nObsTmin71 = getNobsPerDay(dayMasks71, flgTminStn)
            nObsTmin81 = getNobsPerDay(dayMasks81, flgTminStn)
            tmin71 = np.sum(nObsTmin71 >= 20) == days365.size
            tmin81 = np.sum(nObsTmin81 >= 20) == days365.size
        else:
            nObsTmin71 = zeroObs
            nObsTmin81 = zeroObs
            tmin71 = False
            tmin81 = False
            
        if idxStnTmax is not None:
            flgTmaxStn = missFlgsTmax[:,idxStnTmax]
            nObsTmax71 = getNobsPerDay(dayMasks71, flgTmaxStn)
            nObsTmax81 = getNobsPerDay(dayMasks81, flgTmaxStn)
            tmax71 = np.sum(nObsTmax71 >= 20) == days365.size
            tmax81 = np.sum(nObsTmax81 >= 20) == days365.size
        else:
            nObsTmax71 = zeroObs
            nObsTmax81 = zeroObs
            tmax71 = False
            tmax81 = False
            
        
        tmax71 = str(tmax71)
        tmax81 = str(tmax81)
        tmin71 = str(tmin71)
        tmin81 = str(tmin81)+"\n"
        
        aline = [aId,name,lon,lat,elev,gcLon,gcLat,gcElev,gcLstTmax,gcLstTmin,
                 gcTdi,gcMeanTmax,gcMeanTmin,gcSeTmax,gcSeTmin,tmax71,tmax81,tmin71,tmin81]
        
        foutMeta.write(",".join(aline))
        
        foutTmax71.write(aId+","+",".join([str(n) for n in nObsTmax71])+"\n")
        foutTmax81.write(aId+","+",".join([str(n) for n in nObsTmax81])+"\n")
        foutTmin71.write(aId+","+",".join([str(n) for n in nObsTmin71])+"\n")
        foutTmin81.write(aId+","+",".join([str(n) for n in nObsTmin81])+"\n")
        
        stchk.increment()
        

def prcpStnMetaFileRuben():
    
    yrStart = 19710101#1981#1971
    yrEnd = 20101231#2010#2000
    stnda = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc', (yrStart,yrEnd))
    stnids1 = np.loadtxt('/projects/daymet2/station_data/forRuben/prcp19710101-20001231.csv',np.str,delimiter=",",skiprows=1, usecols=[0])
    stnids2 = np.loadtxt('/projects/daymet2/station_data/forRuben/prcp19810101-20101231.csv',np.str,delimiter=",",skiprows=1, usecols=[0])
    stnids = np.unique(np.concatenate((stnids1,stnids2)))
    
    prcp = stnda.load_all_stn_obs_var(stnids,'prcp', set_flagged_nan=True)[0]
    missFlgs = np.isfinite(prcp)
    
    
    days365 = stnda.days[stnda.days[YEAR]==2003]
    days = stnda.days
    dayHeader = [adate.strftime("%d-%b") for adate in days365['DATE']]
    dayHeader[-1] = dayHeader[-1]+"\n"
    dayHeader = "STNID,"+",".join(dayHeader)
    
    foutMeta = open('/projects/daymet2/station_data/forRuben/prcpStns.csv','w')
    foutMeta.write(",".join(["STNID","NAME","LON","LAT","ELEV","PRCP_1971-2000_20obs","PRCP_1981-2010_20obs\n"]))
    
    foutPrcp71 = open('/projects/daymet2/station_data/forRuben/nobsPrcp_1971-2000.csv','w')
    foutPrcp81 = open('/projects/daymet2/station_data/forRuben/nobsPrcp_1981-2010.csv','w')
    nobsFiles = [foutPrcp71,foutPrcp81]
    
    for afile in nobsFiles:
        afile.write(dayHeader)
    
    normMask71 = np.logical_and(days[YEAR]>=1971,days[YEAR]<=2000)
    dayMasks71 = []
    for aDay in days365:
        dayMasks71.append(np.logical_and(np.logical_and(days[MONTH]==aDay[MONTH],days[utld.DAY]==aDay[utld.DAY]),normMask71))
    
    normMask81 = np.logical_and(days[YEAR]>=1981,days[YEAR]<=2010)
    dayMasks81 = []
    for aDay in days365:
        dayMasks81.append(np.logical_and(np.logical_and(days[MONTH]==aDay[MONTH],days[utld.DAY]==aDay[utld.DAY]),normMask81))
     
    stchk = status_check(stnids.size,100)
    for aId,x in zip(stnids,np.arange(stnids.size)):
        
        stn = stnda.stns[stnda.stn_idxs[aId]]
                        
        name = stn[STN_NAME].replace(","," ")
        lon = "{0:.5f}".format(stn[LON])
        lat = "{0:.5f}".format(stn[LAT])
        elev = "{0:d}".format(np.int(np.round(stn[ELEV])))
        
        flgPrcpStn = missFlgs[:,x]
        nObsPrcp71 = getNobsPerDay(dayMasks71, flgPrcpStn)
        nObsPrcp81 = getNobsPerDay(dayMasks81, flgPrcpStn)
        
        if aId[0:6] == 'SNOTEL':
            prcp71 = np.sum(nObsPrcp71 >= 20) == 364
            prcp81 = np.sum(nObsPrcp81 >= 20) == 364
        else:
            prcp71 = np.sum(nObsPrcp71 >= 20) == days365.size
            prcp81 = np.sum(nObsPrcp81 >= 20) == days365.size
        
        prcp71 = str(prcp71)
        prcp81 = str(prcp81)+"\n"
        
        aline = [aId,name,lon,lat,elev,prcp71,prcp81]
        
        foutMeta.write(",".join(aline))
        
        foutPrcp71.write(aId+","+",".join([str(n) for n in nObsPrcp71])+"\n")
        foutPrcp81.write(aId+","+",".join([str(n) for n in nObsPrcp81])+"\n")
        
        stchk.increment()

def prcpStnsForRuben():
    
    minObs = 20
    yrStart = 19810101#1981#1971
    yrEnd = 20101231#2010#2000
    
    stnda = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc', (yrStart,yrEnd))
    stnMask = np.logical_or(np.logical_or(np.char.startswith(stnda.stn_ids,'GHCN_US'),
                                          np.char.startswith(stnda.stn_ids,'SNOTEL')),
                            np.char.startswith(stnda.stn_ids,'RAWS'))
    stnMask = np.logical_and(stnMask,stnda.stns[STATE] != 'AK')

    #stnMask = np.char.startswith(stnda.stn_ids,'SNOTEL')
    
    stns = stnda.stns[stnMask]
    
    uYrs = np.unique(stnda.days[YEAR]) 
    days365 = stnda.days[stnda.days[YEAR]==uYrs[0]]
    
    x = 1
    while days365.size != 365:
        days365 = stnda.days[stnda.days[YEAR]==uYrs[x]]
        x+=1
        
    stnOutMask = np.zeros((days365.size,stns.size),dtype=np.bool)
    
    prcp = stnda.load_all_stn_obs_var(stns[STN_ID],'prcp', set_flagged_nan=True)[0]
    missFlgs = np.isfinite(prcp)
    
    i = 0
    for aDay in days365:
        
        dayMask = np.logical_and(stnda.days[MONTH]==aDay[MONTH],stnda.days[utld.DAY]==aDay[utld.DAY])
        
        flgs = missFlgs[dayMask,:]
                
        stnOutMask[i,:] = np.sum(flgs,axis=0)>=minObs
        i+=1
        
    stnOutMask1 = np.sum(stnOutMask,axis=0)==days365.size
    stnOutMask = np.logical_or(np.logical_and(np.sum(stnOutMask,axis=0)==364,np.char.startswith(stns[STN_ID],'SNOTEL')),stnOutMask1)
    
    stnsOut = stns[stnOutMask]
    print stnsOut.size
    
    print "WRITING...."
    
    with open("".join(['/projects/daymet2/station_data/forRuben/prcp',str(yrStart),"-",str(yrEnd),".csv"]),'w') as f:
                
        stnLists = [[stnLine[STN_ID],stnLine[STN_NAME],stnLine[LON],stnLine[LAT],stnLine[ELEV]] for stnLine in stnsOut]        
        f.write(",".join(['STNID','NAME','LON','LAT','ELEV\n']))
        
        for stnLine in stnLists:
            
            aLine = [str(item) for item in stnLine]
            aLine[-1] = "".join([aLine[-1],"\n"])
            #print aLine
            f.write(','.join(aLine))
    
def stnsForRuben():
    
    minObs = 20
    yrStart = 1981#1981#1971
    yrEnd = 2010#2010#2000
    tairVar = 'tmin'
    
    stndaS = station_data_infill("".join(['/projects/daymet2/station_data/infill/infill_20130518/serialhomog_',tairVar,".nc"]),tairVar)
    stnMask = np.logical_and(np.isfinite(stndaS.stns[MASK]),np.isnan(stndaS.stns[BAD]))  
    
    stnda = station_data_infill("".join(['/projects/daymet2/station_data/infill/infill_20130518/infill_',tairVar,".nc"]),tairVar,stn_dtype=[(STN_ID, "<S16"), (STN_NAME, "<S30"), (LON, np.float64), (LAT, np.float64), (ELEV, np.float64)])
    missFlgs = np.logical_not(stnda.ds.variables['flag_impute'][:].astype(np.bool))
    
    stns = stnda.stns[stnMask]
    days365 = stnda.days[stnda.days[YEAR]==2003]
    normMask = np.logical_and(stnda.days[YEAR]>=yrStart,stnda.days[YEAR]<=yrEnd)
    
    stnOutMask = np.zeros((days365.size,stns.size),dtype=np.bool)
    
    i = 0
    for aDay in days365:
        
        dayMask = np.logical_and(np.logical_and(stnda.days[MONTH]==aDay[MONTH],stnda.days[utld.DAY]==aDay[utld.DAY]),normMask)
        
        flgs = missFlgs[dayMask,:]
        flgs = flgs[:,stnMask]
                
        stnOutMask[i,:] = np.sum(flgs,axis=0)>=minObs
        i+=1
    
    stnOutMask = np.sum(stnOutMask,axis=0)==days365.size
    stnsOut = stns[stnOutMask]
    print stnsOut.size
    print "WRITING...."
    
    with open("".join(['/projects/daymet2/station_data/forRuben/',tairVar,str(yrStart),"-",str(yrEnd),".csv"]),'w') as f:
                
        stnLists = [list(stnLine) for stnLine in stnsOut]        
        f.write(",".join(['STNID','NAME','LON','LAT','ELEV\n']))
        
        for stnLine in stnLists:
            
            aLine = [str(item) for item in stnLine]
            aLine[-1] = "".join([aLine[-1],"\n"])
            #print aLine
            f.write(','.join(aLine))

def analyzeBadImpsHighMAE():
    
    source_r('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R')
    
    f = open('/projects/daymet2/station_data/infill/infill_20130518/impute_20130528.log')
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
    
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/infill_tmax.nc','tmax_imp',stn_dtype=DTYPE_STN_BASIC)
    stnda2 = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/infill_tmax.nc','tmax',stn_dtype=DTYPE_STN_BASIC)
    
    
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
        
        print "".join(["% Imped: ",str(np.sum(flg)/np.float(flg.size)*100.)])
        
        plt.show()

def analyzeBadImps():
    
    source_r('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/pca_infill.R')
    
    f = open('/projects/daymet2/station_data/infill/infill_20130518/impute_20130528.log')
    lines = np.array(f.readlines())
    lmask = np.logical_and(np.logical_and(np.char.find(lines, 'ERROR|') != -1,
                                          np.char.find(lines, 'variance') != -1),
                           np.char.find(lines, 'tmin') != -1)
    print np.sum(lmask)
    
    stnids = np.unique(np.array([aline.split('|')[1].split()[0] for aline in lines[lmask]]))
    
    for stnid in stnids:
        print stnid
    
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/infill_tmin.nc','tmin_imp',stn_dtype=DTYPE_STN_BASIC)
    stnda2 = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/infill_tmin.nc','tmin',stn_dtype=DTYPE_STN_BASIC)
    
    
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
        
        print "".join(["% Imped: ",str(np.sum(flg)/np.float(flg.size)*100.)])
        
        plt.show()

def analyzeRmStns():
    dupsTmin = np.loadtxt('/projects/daymet2/station_data/infill/infill_fnl/dupstns_tmin.csv',delimiter=",",dtype=np.str,skiprows=1,usecols=(0,))
    dupsTmax = np.loadtxt('/projects/daymet2/station_data/infill/infill_fnl/dupstns_tmax.csv',delimiter=",",dtype=np.str,skiprows=1,usecols=(0,))
    dups = np.unique(np.concatenate([dupsTmin,dupsTmax]))
    rmIds = np.loadtxt('/projects/daymet2/station_data/infill/infill_fnl/rm_stns_all.csv',dtype=np.str)
    
    rmIds = np.unique(rmIds[np.logical_not(np.in1d(rmIds, dups, True))])
    print rmIds.size
    for stnid in rmIds:
        print stnid
    stndaTmin = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/infill_tmin.nc','tmin_imp',stn_dtype=DTYPE_STN_BASIC)
    stndaTmax = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/infill_tmax.nc','tmax_imp',stn_dtype=DTYPE_STN_BASIC)
    
    stnda = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    
    rmStns = stnda.stns[np.in1d(stnda.stn_ids, rmIds, True)]
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(rmStns[LAT])-1,urcrnrlat=np.max(rmStns[LAT])+1,\
                llcrnrlon=np.min(rmStns[LON])-1,urcrnrlon=np.max(rmStns[LON])+1,resolution='c')
    
    m.scatter(rmStns[LON],rmStns[LAT])
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    
    plt.show()
    
    for stn in rmStns:
    
        print stn
        plt.clf()
        plt.subplot(221)
        try:
            plt.plot(stndaTmin.load_obs(stn[STN_ID]))
            plt.title('TMIN IMP')
        except KeyError:
            pass
        plt.subplot(222)
        try:
            plt.plot(stnda.load_all_stn_obs_var(stn[STN_ID],'tmin')[0])
            plt.title('TMIN RAW')
        except KeyError:
            pass
        plt.subplot(223)
        try:
            plt.plot(stndaTmax.load_obs(stn[STN_ID]))
            plt.title('TMAX')
        except KeyError:
            pass
        plt.subplot(224)
        try:
            plt.plot(stnda.load_all_stn_obs_var(stn[STN_ID],'tmax')[0])
            plt.title('TMAX RAW')
        except KeyError:
            pass
        plt.suptitle(stn[STN_ID])
        plt.show()

def testGhcnStnList():
    f = open('/projects/daymet2/inhomo_software/mydata/tmax/world1_stnlist.tmax')
    lines = f.readlines()
    lon = np.array([np.float(x.split()[2]) for x in lines])
    lat = np.array([np.float(x.split()[1]) for x in lines])
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(lat)-1,urcrnrlat=np.max(lat)+1,\
                llcrnrlon=np.min(lon)-1,urcrnrlon=np.max(lon)+1,resolution='c')
    
    m.scatter(lon,lat)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    
    plt.show()

def ghcnStnCsv():

    f = open("/projects/daymet2/station_data/ghcn/ghcnd-stations.txt")
    fout = open("/projects/daymet2/station_data/ghcn/ghcnd-stations.csv",'w')
    fout.write(",".join(["FIPS","STNID","LAT","LON","ELEV","NAME\n"]))

    for line in f.readlines():
        
        fips_code = line[0:2]
                    
        stn_id_orig = line[0:11].strip()
        lat = line[12:20].strip()
        lon = line[21:30].strip()
        elev = line[31:37].strip()
        name = unicode(line[41:71].strip())
        name = name.replace(","," ")
        
        fout.write(",".join([fips_code,stn_id_orig,lat,lon,elev,"".join([name,"\n"])]))
                
    fout.close()

def optimTairMeanStats():
    ds = Dataset("/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc")
    divs = ds.variables['neon'][:]
    lccs = np.unique(divs.data[np.logical_not(divs.mask)])
    #lccs = np.array([1209.0])
    fpath = '/projects/daymet2/station_data/infill/infill_20130518/xval/optimTairMean/tmax/'
    #fpath = '/projects/daymet2/station_data/infill/infill_fnl/xval/'
    fprefix = "xval_tmax_mean"
    #lccs = np.arange(1,17)
    
    
    biasAll = np.array([])
    stnidAll = np.array([],dtype="<S16")
    print fprefix
    fnlNghsAll = []
    for alcc in lccs:
        
        ds = Dataset("".join([fpath,fprefix,"_",str(alcc),".nc"]))
        minNghs = ds.variables['min_nghs'][:]
        mae = ds.variables['mae'][:]
        bias = ds.variables['bias'][:]
        stnid = ds.variables['stn_id'][:].astype("<S16")
        
        mmae = np.mean(mae,axis=1)
        mbias = np.mean(bias,axis=1)
        x = np.argmin(mmae)
#        plt.plot(mbias)
#        plt.title("LCC "+str(alcc))
#        plt.show()
        
        fnlMmae = mmae[x]
        fnlNghs = minNghs[x]
        fnlMbias = mbias[x]
        
        biasRow = bias[x,:]
        biasAll = np.concatenate((biasAll,biasRow))
        stnidAll = np.concatenate((stnidAll,stnid))
        
#        print "############################################################"
        print "|".join([str(alcc),str(fnlMmae),str(fnlMbias),str(fnlNghs)])
        
        fnlNghsAll.append(fnlNghs)
      
    #biasAllStd = np.abs((biasAll - np.mean(biasAll)) / np.std(biasAll))  
#    print np.sum(biasAllStd >= 4)
#    print stnidAll[biasAllStd >= 4]  
#    print biasAll[biasAllStd >= 4]
    print fnlNghsAll
    plt.boxplot(biasAll)
    plt.show()
    
    maeAll = np.abs(biasAll)
    print np.sum(maeAll >= 5)
    print stnidAll[maeAll >= 5]  
    print biasAll[maeAll >= 5]    

def optimTairMeanHomogVsRaw():
    
    fpath = '/projects/daymet2/station_data/infill/infill_20130518/xval/'
    #fpath = '/projects/daymet2/station_data/infill/infill_fnl/xval/'
    fprefixHomog = "xval100_tmin_mean"
    fprefixRaw = "xval100raw_tmin_mean"
    lccs = np.arange(1,17)
    
    
    difAll = np.array([])
    stnidAll = np.array([],dtype="<S16")
    for alcc in lccs:
        
        dsHomog = Dataset("".join([fpath,fprefixHomog,"_",str(alcc),".nc"]))
        dsRaw = Dataset("".join([fpath,fprefixRaw,"_",str(alcc),".nc"]))
        minNghs = dsHomog.variables['min_nghs'][:]
        stnid = dsHomog.variables['stn_id'][:].astype("<S16")
        
        maeHomog = dsHomog.variables['mae'][:].ravel()
        biasHomog = dsHomog.variables['bias'][:].ravel()
        maeRaw = dsRaw.variables['mae'][:].ravel()
        biasRaw = dsRaw.variables['bias'][:].ravel()
        
        maeDif = maeHomog-maeRaw
        
        difAll = np.concatenate((difAll,maeDif))
        stnidAll = np.concatenate((stnidAll,stnid))
    
    difMask = difAll > 0
    difAll = np.abs(difAll[difMask])
    stnidAll = stnidAll[difMask]
    
    difAllStd = np.abs((difAll - np.mean(difAll)) / np.std(difAll))  
    
    print np.sum(np.logical_and(difAllStd > 4,difAll > 2))
    print stnidAll[np.logical_and(difAllStd > 4,difAll > 2)]
    plt.boxplot(difAll)
    plt.show()

def rmStnMap():
    
    stnda = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    
    rmIds = np.unique(np.array(['GHCN_USC00094688','RAWS_AGUT','SNOTEL_08S08S','SNOTEL_11D26S',
                      'RAWS_CBUR','RAWS_AHAV','RAWS_AHIL','GHCN_USC00267750','RAWS_IWEI']))
    
    stns = stnda.stns[np.in1d(stnda.stn_ids, rmIds, True)]
    lon = stns[LON]
    lat = stns[LAT]
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(lat)-1,urcrnrlat=np.max(lat)+1,\
                llcrnrlon=np.min(lon)-1,urcrnrlon=np.max(lon)+1,resolution='l')
    
    m.scatter(lon,lat)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    
    plt.show()
    
def usClimDivStnsMap():
    ds_path_tmin = '/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc'
    stnda = station_data_infill(ds_path_tmin, 'tmax')
    stns = stnda.stns[np.isfinite(stnda.stns[NEON])]
    
    lon = stns[LON]
    lat = stns[LAT]
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(lat)-1,urcrnrlat=np.max(lat)+1,\
                llcrnrlon=np.min(lon)-1,urcrnrlon=np.max(lon)+1,resolution='l')
    
    m.scatter(lon,lat)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    
    plt.show()

def tairTrends():
    
    stnda1 = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmin.nc','tmin')
    stnda2 = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serial_tmin.nc','tmin',stn_dtype=DTYPE_STN_BASIC)
    days = stnda1.days
    
    mthMask = np.logical_and(days[MONTH]>=6,days[MONTH]<=8)
    yrMasks = []
    for yr in np.unique(stnda1.days[YEAR]):
        
        yrMasks.append(np.logical_and(yr==stnda1.days[YEAR],mthMask))
    
    ds = Dataset('/projects/daymet2/interp_output/montana/h05v02/h05v02_tmin.nc')
    tairvar = 'tmin'
    
    p1 = ds.variables[tairvar][:,185,139]
    p2 = ds.variables[tairvar][:,175,157]
    
    stnid = 'SNOTEL_13D19S'
    obs1 = stnda1.load_obs(stnid)
    obs2 = stnda2.load_obs(stnid)
    print stnda1.stns[stnda1.stn_ids==stnid]
        
    annP1 = np.array([np.mean(p1[aMask]) for aMask in yrMasks])
    annP2 = np.array([np.mean(p2[aMask]) for aMask in yrMasks])
    yrs = np.unique(stnda1.days[YEAR])
    
    plt.plot(obs1-obs2)
    plt.show()
    
    annO1 = np.array([np.mean(obs1[aMask]) for aMask in yrMasks])
    annO2 = np.array([np.mean(obs2[aMask]) for aMask in yrMasks])
    
    plt.plot(yrs,annO1-np.mean(annO1))
    plt.plot(yrs,annO2-np.mean(annO2))
    xmin, xmax = plt.xlim()
    plt.hlines(0, xmin, xmax)
    plt.show()
    print yrs[np.argmin(annO2-np.mean(annO2))]
    plt.plot(yrs,annP1-np.mean(annP1))
    plt.plot(yrs,annP2-np.mean(annP2))
    xmin, xmax = plt.xlim()
    plt.hlines(0, xmin, xmax)
    plt.show()

def hcnMap():
    
    stnda = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    
    hcnIds = np.unique(np.loadtxt('/projects/daymet2/station_data/ghcn/hcnXvalStns.txt',dtype=np.str))
    
    stns = stnda.stns[np.in1d(stnda.stn_ids, hcnIds, True)]
    lon = stns[LON]
    lat = stns[LAT]
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(lat)-1,urcrnrlat=np.max(lat)+1,\
                llcrnrlon=np.min(lon)-1,urcrnrlon=np.max(lon)+1,resolution='l')
    
    m.scatter(lon,lat)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    
    plt.show()

def buildLowRestSnotel():
    stns = parse_hist_stns('/projects/daymet2/station_data/snotel/historical/')
    ids = stns.keys()
    outids = np.array([('SNOTEL_'+aid).upper() for aid in ids])
    name = np.array([stns[aid][0] for aid in ids])
    name = np.char.replace(name, ",", " ")
    name = np.char.replace(name, "#", " ")
    lat = np.array([stns[aid][1] for aid in ids])
    lon = np.array([stns[aid][2] for aid in ids])
    elev = np.array([stns[aid][3] for aid in ids])
    state = np.array([stns[aid][4] for aid in ids])
    
    stnsLow = np.empty(elev.size,dtype=[('stn_id',"<S16"),('name', "<S30"),('lon', np.float64),('lat', np.float64),('elev', np.float64),('state', "<S2")])
    stnsLow['stn_id'] = outids
    stnsLow['name'] = name
    stnsLow['lat'] = lat
    stnsLow['lon'] = lon
    stnsLow['elev'] = elev
    stnsLow['state'] = state
    
    np.savetxt('/projects/daymet2/station_data/snotel/snotel_lowres.csv', stnsLow, fmt=("%s","%s","%f","%f","%f","%s"), delimiter=",")

def saveSerialStnList():
    
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmin.nc','tmin')
    stns = stnda.stns[np.isnan(stnda.stns[BAD])]
    stns = filterToLowResSnotel(stns,'/projects/daymet2/station_data/snotel/snotel_lowres.csv')
    
    fout = open("/projects/daymet2/station_data/infill/infill_20130518/tminStns.csv",'w')
    
    fout.write(",".join(['STNID','NAME','LON','LAT','ELEV\n']))

    for stn in stns:

        fout.write(",".join([stn[STN_ID],stn[STN_NAME],str(stn[LON]),str(stn[LAT]),str(stn[ELEV])+"\n"]))

def mooseRasters():
#    te = ti.tile_extract('/projects/daymet2/interp_output/montana/',
#                    '/projects/daymet2/climate_office/interp_grid_geotifs/FwpMooseRasters/', 'tmax', 'tmax_mean',None)#gdalconst.GDT_Float32)
    
    te = ti.tile_extract_predictor('/projects/daymet2/interp_output/montana/',
                                   '/projects/daymet2/climate_office/interp_grid_geotifs/FwpMooseRasters/',
                                   'tmin', '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc', 'tdi')
    
    te.extract_all()
    te.mosaic()

def subsetPOR():
    stnda = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc')
    
    fin = open('/projects/daymet2/station_data/all/all_por_1948_2012.csv')
    fout = open('/projects/daymet2/station_data/all/tairHomog_por_1948_2012.csv','w')
    fout.write(fin.readline())
    
    for aline in fin.readlines():
        
        stnid = aline.split(',')[0]
        if stnid in stnda.stn_ids:
            fout.write(aline)
    fin.close()
    fout.close()

def addUTC():
    db = station_data_ncdb('/projects/daymet2/station_data/all/tairHomog_1948_2012.nc',mode='r+',stnDtype=DTYPE_STN_BASIC)
    varutc = db.add_stn_variable("utc_offset","utc offset","hours","i2")
    
    dsin = Dataset('/projects/daymet2/station_data/all/all_1948_2012.nc')
    utcs = dsin.variables['utc_offset'][:]
    stnids = dsin.variables['stn_id'][:].astype("<S16")
    
    for x in np.arange(db.stn_ids.size):
        
        i = np.nonzero(stnids==db.stn_ids[x])[0][0]
        varutc[x] = utcs[i]
    db.ds.sync()

def xvalStatsImpute():
    
    #ds = Dataset('/projects/daymet2/station_data/infill/xval_impute_homog_tair_sntlraws.nc')      
    ds = Dataset('/projects/daymet2/station_data/infill/xval_impute_homog_tair_hcn.nc') 
    stnids = ds.variables['stn_id'][:].astype("<S16")
    
    idPfxs = ['GHCN']#['RAWS','SNOTEL']['GHCN']
    
    for aPfx in idPfxs:
        
        idx = np.nonzero(np.char.startswith(stnids, aPfx))[0]
        
        tminMae = np.mean(ds.variables['mae'][0,0,idx])
        tminMab = np.mean(np.abs(ds.variables['bias'][0,0,idx]))
        tminBias = np.mean(ds.variables['bias'][0,0,idx])
        tminVar = np.mean(ds.variables['var_pct'][0,0,idx])
        
        tmaxMae = np.mean(ds.variables['mae'][1,0,idx])
        tmaxMab = np.mean(np.abs(ds.variables['bias'][1,0,idx]))
        tmaxBias = np.mean(ds.variables['bias'][1,0,idx])
        tmaxVar = np.mean(ds.variables['var_pct'][1,0,idx])
        
        print "|".join([aPfx,str(idx.size)," total stations"])
        print "|".join([aPfx,"TMIN",str(tminMae),str(tminMab),str(tminBias),str(tminVar)])
        print "|".join([aPfx,"TMAX",str(tmaxMae),str(tmaxMab),str(tmaxBias),str(tmaxVar)])
        
def statsImpute():
    
    stndaSerial = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc', 'tmax')
    stndaInfill = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/infill_tmax.nc', 'tmax',stn_dtype=DTYPE_STN_BASIC)
    
    stnMask = np.logical_and(np.isfinite(stndaSerial.stns[MASK]),np.isnan(stndaSerial.stns[BAD]))
    #stnMask = np.isnan(stndaSerial.stns[BAD])  
    
    idPfxs = ['RAWS','SNOTEL','GHCN_US']#['RAWS','SNOTEL']['GHCN']
    
    for aPfx in idPfxs:
        
        idx = np.nonzero(np.logical_and(np.char.startswith(stndaInfill.stn_ids, aPfx),stnMask))[0]
        
        mae = np.mean(stndaInfill.ds.variables['mae'][idx])
        mab = np.mean(np.abs(stndaInfill.ds.variables['bias'][idx]))
        bias = np.mean(stndaInfill.ds.variables['bias'][idx])
        var = np.mean(stndaInfill.ds.variables['r2'][idx])
        
        print "|".join([aPfx,str(idx.size)," total stations"])
        print "|".join([aPfx,str(mae),str(mab),str(bias),str(var)])
        
def daymetAnnPlot():
    yr,yday,tmin = np.loadtxt('/projects/daymet2/daymet_oakridge/12274_07-31-2013.csv',dtype = [('year',np.int),('yday',np.int),('tmin',np.float)],delimiter=',', skiprows=7,unpack=True)
    tmin = np.ma.masked_array(tmin,tmin==-9999)
    days = utld.get_days_metadata_daymet(yr, yday)
    
    agg = ushcn.TairAggregate(days)
    
    tminAnn = agg.dailyToAnn(tmin)
    
    plt.plot(tmin)
    plt.show()

def imputeLogAnalysis():
    
    logFile = open('/projects/daymet2/station_data/infill/infill_20130725/impute.log')
    
    warns = {}
    succ = {}
    errs = {}
    
    for aline in logFile.readlines():
        
        if aline[0:7] == "WARNING" and aline.find('variance') != -1:
            
            stnid = aline.split("|")[1].split()[0]
            warns[stnid] = aline
        
        elif aline[0:7] == "SUCCESS":
            
            stnid = aline.split("|")[1].split()[0]
            succ[stnid] = aline
        
        elif aline[0:5] == "ERROR" and aline.find('variance') != -1:
            
            stnid = aline.split("|")[1].split()[0]
            errs[stnid] = aline
    
    stnidWarn = warns.keys()
    stnidSucc = succ.keys()
    stnidErrs = errs.keys()
    
    for astnid in stnidWarn:
        
        if astnid not in stnidSucc and astnid not in stnidErrs:
            print astnid

def tmaxShiftValidation():
    
    ds1 = Dataset('/projects/daymet2/station_data/infill/infill_20130518/serialhomog_tmax.nc')
    ds2 = Dataset('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
    
    mae1 = ds1.variables['xval_overall_mae'][:]
    mae2 = ds2.variables['xval_overall_mae'][:]
    
    stnid1 = ds1.variables['stn_id'][:].astype("<S16")
    stnid2 = ds2.variables['stn_id'][:].astype("<S16")
    
    mae1 = mae1[np.char.startswith(stnid1, "GHCN_W")]
    mae2 = mae2[np.char.startswith(stnid2, "GHCN_W")]
    
    print np.mean(mae1)
    print np.mean(mae2)
    print (np.mean(mae2)-np.mean(mae1))/np.mean(mae1)*100

def nearestGridCellTest():
    
    r = input_raster('/projects/daymet2/dem/interp_grids/conus/tifs/crop_elev.tif')
    r.getDataValue(-124.5550003,47.9375)




def randomCcePts():
    PATH_CCE_MASK = '/projects/daymet2/dem/interp_grids/cce/crp_cce_us_mask.tif'
    ds = RasterDataset(PATH_CCE_MASK)
    a = ds.readAsArray()
    
    r = np.random.randint(low=0,high=a.shape[0]-1,size=500)
    c = np.random.randint(low=0,high=a.shape[1]-1,size=500)
    
    vals = a[r,c]
    mask = ~vals.mask
    r = r[mask]
    c = c[mask]
    
    plt.imshow(a)
    plt.scatter(c, r,color="red")
    plt.show()
    
    auxFpaths = ['/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_climdiv.nc']
            
    ptInterper = it.PtInterpTair('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc',
                    '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc',
                    '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                    '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                    auxFpaths)
    
    for x in np.arange(r.size):
        
        lat,lon = ds.getCoord(r[x], c[x])
        ptInterper.interpLonLatPt(lon,lat)
        print x

def plotCceStns():
    
    stnda = station_data_infill('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc','tmin')
    stnIds = np.unique(np.loadtxt('/projects/daymet2/docs/final_writeup/cce_stnids.txt', dtype=np.str))
    
    mask = np.in1d(stnda.stn_ids, stnIds, True)
    
    print np.sum(mask),stnIds.size
    
    stns = stnda.stns[mask]
    
    
    
    m = Basemap(projection='cyl',llcrnrlat=np.min(stns[LAT]),urcrnrlat=np.max(stns[LAT]),\
                llcrnrlon=np.min(stns[LON]),urcrnrlon=np.max(stns[LON]),resolution='i')
    m.scatter(stns[LON],stns[LAT])
    m.drawcountries()
    m.drawcoastlines()
    m.drawstates()
    m.readshapefile('/projects/daymet2/dem/interp_grids/cce/CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True,linewidth=1)
    plt.show()

def glacStnValidation():
    
    auxFpaths = ['/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin.nc',
                 '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_climdiv.nc']
            
    ptInterper = it.PtInterpTair('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc',
                    '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc',
                    '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                    '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                    auxFpaths)
    
    stnda = station_data_ncdb('/projects/daymet2/station_data/crown_stns_final.nc',stnDtype=DTYPE_STN_BASIC)
    stn = stnda.stns[stnda.stn_ids=='GLAC_7'][0]
    tminObs = stnda.load_all_stn_obs_var('GLAC_7','tmin')[0]
    tmaxObs = stnda.load_all_stn_obs_var('GLAC_7','tmax')[0]
    print stn
    tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpLonLatPt(stn[LON], stn[LAT])


    print ptInterper.a_pt[ELEV],stn[ELEV]
    print tmin_mean,tmax_mean,ninvalid
    
    ymdMask = np.in1d(ptInterper.days[YMD], stnda.days[YMD], True)
    tmin_dly = tmin_dly[ymdMask]
    tminObs = tminObs[np.in1d(stnda.days[YMD],ptInterper.days[YMD], True)]
    finMask = np.isfinite(tminObs)
    print np.mean(tmin_dly[finMask]-tminObs[finMask]),stats.linregress(tmin_dly[finMask], tminObs[finMask])[2]**2
    
    tmax_dly = tmax_dly[ymdMask]
    tmaxObs = tmaxObs[np.in1d(stnda.days[YMD],ptInterper.days[YMD], True)]
    finMask = np.isfinite(tmaxObs)
    print np.mean(tmax_dly[finMask]-tmaxObs[finMask]),stats.linregress(tmax_dly[finMask], tmaxObs[finMask])[2]**2
    plt.plot(tmax_dly[finMask])
    plt.plot(tmaxObs[finMask])
    plt.show()

def glacStnValidationDaymet():
    
    stnda = station_data_ncdb('/projects/daymet2/station_data/crown_stns_final.nc',stnDtype=DTYPE_STN_BASIC)
    stn = stnda.stns[stnda.stn_ids=='GLAC_7'][0]
    tminObs = stnda.load_all_stn_obs_var('GLAC_7','tmin')[0]
    tmaxObs = stnda.load_all_stn_obs_var('GLAC_7','tmax')[0]
    print stn

    a = np.loadtxt('/projects/daymet2/compare/DaymetLoganPass.csv', delimiter=",",skiprows=7)
    days = utld.get_days_metadata_daymet(a[:,0],a[:,1])
    tmin_dly = a[:,3]
    tmax_dly = a[:,3]

    
    ymdMask = np.in1d(days[YMD], stnda.days[YMD], True)
    tmin_dly = tmin_dly[ymdMask]
    tminObs = tminObs[np.in1d(stnda.days[YMD],days[YMD], True)]
    finMask = np.isfinite(tminObs)
    print np.mean(tmin_dly[finMask]-tminObs[finMask]),stats.linregress(tmin_dly[finMask], tminObs[finMask])[2]**2
    
    tmax_dly = tmax_dly[ymdMask]
    
    tmaxObs = tmaxObs[np.in1d(stnda.days[YMD],days[YMD], True)]
    finMask = np.isfinite(tmaxObs)
    print np.mean(tmax_dly[finMask]-tmaxObs[finMask]),stats.linregress(tmax_dly[finMask], tmaxObs[finMask])[2]**2
    plt.plot(tmax_dly[finMask])
    plt.plot(tmaxObs[finMask])
    plt.show()

def plotGlacStns():
    stnda = station_data_ncdb('/projects/daymet2/station_data/crown_stns_final.nc',stnDtype=DTYPE_STN_BASIC)
    glacStns = stnda.stns[np.char.startswith(stnda.stn_ids, 'GLAC')]
    
    for aStn in glacStns:
        
        tminObs = stnda.load_all_stn_obs_var(aStn[STN_ID],'tmin')[0]
        tmaxObs = stnda.load_all_stn_obs_var(aStn[STN_ID],'tmax')[0]
        
        plt.clf()
        plt.subplot(211)
        plt.plot(tminObs)
        plt.subplot(212)
        plt.plot(tmaxObs)
        plt.suptitle("|".join([aStn[STN_ID],aStn[STN_NAME],str(tminObs[np.isfinite(tminObs)].size),str(tmaxObs[np.isfinite(tmaxObs)].size)]))
        plt.show()
    
def get_lst_stn(x,ds):
    lst = []
    for mth in np.arange(1,13):
        lst.append(ds.variables[get_lst_varname(mth)][x])
    return lst
    
def plotLstStn():
    
    ds = Dataset('/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc')
    stnids = ds.variables['stn_id'][:].astype("<S16")
    x = np.nonzero(stnids=='GHCN_USW00023272')[0][0]
    lst1 = get_lst_stn(x, ds)
    x = np.nonzero(stnids=='GHCN_USC00044997')[0][0]
    lst2 = get_lst_stn(x, ds)
    
    plt.plot(lst1)
    plt.plot(lst2)
    plt.show()

def perfInterpTair():
    P_PATH_MASK = 'P_PATH_MASK'
    P_PATH_ELEV = 'P_PATH_ELEV'
    P_PATH_TDI = 'P_PATH_TDI'
    P_PATH_LST_TMIN = 'P_PATH_LST_TMIN'
    P_PATH_LST_TMAX = 'P_PATH_LST_TMAX'
    P_PATH_LC = 'P_PATH_LC'
    P_PATH_VCF = 'P_PATH_VCF'
    P_PATH_NEON = 'P_PATH_NEON'
    P_PATH_DB_TMIN = 'P_PATH_DB_TMIN'
    P_PATH_DB_TMAX = 'P_PATH_DB_TMAX'
    P_PATH_OUT = 'P_PATH_OUT'
    P_PATH_DUPS_TMIN = 'P_PATH_DUPS_TMIN'
    P_PATH_DUPS_TMAX = 'P_PATH_DUPS_TMAX'
    P_PATH_CLIB = 'P_PATH_CLIB'
    P_PATH_RLIB = 'P_PATH_RLIB'
    P_PATH_NEON_PARAMS_TMIN = 'P_PATH_NEON_PARAMS_TMIN'
    P_PATH_RMSTNS_TMIN = "P_PATH_RMSTNS_TMIN"
    P_PATH_RMSTNS_TMAX = "P_PATH_RMSTNS_TMAX"
    P_PATH_PARAMS_MEAN_TMIN = 'P_PATH_PARAMS_MEAN_TMIN'
    P_PATH_PARAMS_ANOM_TMIN = 'P_PATH_PARAMS_ANOM_TMIN'
    P_PATH_PARAMS_MEAN_TMAX = 'P_PATH_PARAMS_MEAN_TMAX'
    P_PATH_PARAMS_ANOM_TMAX = 'P_PATH_PARAMS_ANOM_TMAX'
    
    P_TILESIZE_X = 'P_TILESIZE_X'
    P_TILESIZE_Y = 'P_TILESIZE_Y'
    P_CHCKSIZE_X = 'P_CHCKSIZE_X'
    P_CHCKSIZE_Y = 'P_CHCKSIZE_Y'
    P_TILE_INFO = 'P_TILE_INFO'
    P_TILES_PROCESS = 'P_TILES_PROCESS'
    
    SCALE_FACTOR = np.float32(0.01)
    
    #long name, units, standard name, missing value
    VAR_ATTRS = {'tmin':("minimum air temperature","C","minimum_air_temperature",netCDF4.default_fillvals['i2']),
                 'tmax':("maximum air temperature","C","maximum_air_temperature",netCDF4.default_fillvals['i2'])}
    params = {}
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_PATH_RLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R'
    
    params[P_PATH_DB_TMIN] = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc'
    params[P_PATH_DB_TMAX] = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc'

    gridPath = '/projects/daymet2/dem/interp_grids/conus/ncdf/'
    params[P_PATH_MASK] = "".join([gridPath,'fnl_mask.nc'])
    params[P_PATH_ELEV] = "".join([gridPath,'fnl_elev.nc'])
    params[P_PATH_TDI] = "".join([gridPath,'fnl_tdi.nc'])
    params[P_PATH_LST_TMIN] = ["".join([gridPath,'fnl_lst_tmin%02d.nc'%mth]) for mth in np.arange(1,13)]
    params[P_PATH_LST_TMAX] = ["".join([gridPath,'fnl_lst_tmax%02d.nc'%mth]) for mth in np.arange(1,13)]
    params[P_PATH_NEON] = "".join([gridPath,'fnl_climdiv.nc'])
    
    params[P_PATH_OUT] = '/projects/daymet2/interp_output/'#topowx_tile_output/'
    params[P_TILESIZE_X] = 250
    params[P_TILESIZE_Y] = 250
    params[P_CHCKSIZE_X] = 50
    params[P_CHCKSIZE_Y] = 50
    params[P_TILES_PROCESS] = np.array([17])#np.array([16,17,18,37,38])
    ####################################################
    
    ds_mask = Dataset(params[P_PATH_MASK])    
    ds_elev = Dataset(params[P_PATH_ELEV])
    ds_tdi = Dataset(params[P_PATH_TDI])
    ds_neon = Dataset(params[P_PATH_NEON])
    
    ds_attrs = [('elev',ds_elev),('tdi',ds_tdi),('neon',ds_neon)]
    ds_attrs_lst_tmin = [('tmin%02d'%mth,Dataset(params[P_PATH_LST_TMIN][mth-1])) for mth in np.arange(1,13)]
    ds_attrs_lst_tmax = [('tmax%02d'%mth,Dataset(params[P_PATH_LST_TMAX][mth-1])) for mth in np.arange(1,13)]
    ds_attrs.extend(ds_attrs_lst_tmin)
    ds_attrs.extend(ds_attrs_lst_tmax)
    
    atiler = tl.tiler(ds_mask, ds_attrs, params[P_TILESIZE_Y], params[P_TILESIZE_X], 
                      params[P_CHCKSIZE_Y], params[P_CHCKSIZE_X], params[P_TILES_PROCESS])
    
    tile_num,wrk_chk = atiler.next()
    
#    tileInfo = atiler.build_tile_grid_info()
#    awriter = tl.tile_writer(tileInfo, params[P_PATH_OUT])
#    tile_id = tileInfo.get_tile_id(tile_num)
    
    print wrk_chk.shape
    
    stndaTmin = StationDataWrkChk(params[P_PATH_DB_TMIN], 'tmin')
    stndaTmax = StationDataWrkChk(params[P_PATH_DB_TMAX], 'tmax')
    ptInterp = it.PtInterpTair(stndaTmin, stndaTmax, params[P_PATH_RLIB], params[P_PATH_CLIB]) 
    days = stndaTmin.days
    
#    str_row = np.int(wrk_chk[0,0,0])
#    str_col = np.int(wrk_chk[1,0,0])
    
#    rslt_tmin = np.empty((stndaTmin.days.size,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int16)
#    rslt_tmin.fill(netCDF4.default_fillvals['i2'])
#    rslt_tmin_norm = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
#    rslt_tmin_se = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
#    rslt_ninvalid = np.ones((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int32)*netCDF4.default_fillvals['i4']
#    awriter.write_rslts(tile_id,'tmin', stndaTmin.days, str_row, str_col,rslt_tmin, rslt_tmin_norm, rslt_tmin_se,rslt_ninvalid)
#    sys.exit()
    
    minLat = wrk_chk[3,-1,0]
    maxLat = wrk_chk[3,0,0]
    minLon = wrk_chk[4,0,0]
    maxLon = wrk_chk[4,0,-1]
    
    #print wrk_chk[2,-1,0],wrk_chk[2,0,0],wrk_chk[2,0,0],wrk_chk[2,0,0]
    
    bnds = (minLat,maxLat,minLon,maxLon)
    stndaTmin.set_obs(bnds)
    stndaTmax.set_obs(bnds)
    
    mths = np.arange(1,13)
    
    rslt_tmin = np.empty((days.size,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int16)
    rslt_tmin.fill(netCDF4.default_fillvals['i2'])
    rslt_tmin_norm = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    rslt_tmin_se = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    
    rslt_tmax = np.empty((days.size,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int16)
    rslt_tmax.fill(netCDF4.default_fillvals['i2'])
    rslt_tmax_norm = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    rslt_tmax_se = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    
    def runInterp():
        ptInterp.a_pt[LAT] = wrk_chk[3,r,c]
        ptInterp.a_pt[LON] = wrk_chk[4,r,c]
        ptInterp.a_pt[ELEV] = wrk_chk[5,r,c]
        ptInterp.a_pt[TDI] = wrk_chk[6,r,c]
        ptInterp.a_pt[NEON] = wrk_chk[7,r,c]
        for mth in mths:
            ptInterp.a_pt["tmin%02d"%mth] = wrk_chk[8+(mth-1),r,c]
            ptInterp.a_pt["tmax%02d"%mth] = wrk_chk[20+(mth-1),r,c]
        ptInterp.a_pt[NEON]
    
        tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterp.interpPt()
        
        plt.plot(tmin_se)
        plt.plot(tmax_se)
        plt.show()
        
        rslt_tmin[:,r,c] = np.round(tmin_dly,2)/SCALE_FACTOR
        rslt_tmax[:,r,c] = np.round(tmax_dly,2)/SCALE_FACTOR
        
        rslt_tmin_norm[:,r,c] = tmin_norms
        rslt_tmax_norm[:,r,c] = tmax_norms
                               
        rslt_tmin_se[:,r,c] = tmin_se
        rslt_tmax_se[:,r,c] = tmax_se
        
#        plt.plot(tmin_dly)
#        plt.plot(tmax_dly)
#        plt.show()
#        plt.clf()
#        plt.plot(tmax_dly-tmin_dly)
#        plt.show()
#        plt.clf()
#        plt.plot(tmin_norms)
#        plt.plot(tmax_norms)
#        plt.show()
#        plt.clf()
#        plt.plot(tmin_se)
#        plt.plot(tmax_se)
#        plt.show()
    
    global runAInterp
    global r
    global c
    runAInterp = runInterp
    
    for i in np.arange(params[P_CHCKSIZE_Y]):
            
        for j in np.arange(params[P_CHCKSIZE_X]):
            r = i
            c = j
            
            if c == 40 and r == 40:
                cProfile.run("runAInterp()")
                sys.exit()
            
            #runInterp()
            #print r,c
            


def test_exp_gauss():
    
    def wgt_exp(d,r):
        return np.exp(-(d/r))**2
    
    def wgt_gauss(d,r):
        return np.exp(-(d/r)**2)
    
    def wgt_gauss2(d,r):
        return np.exp(-0.5*(d/r)**2)
        
    def wgt_cos(d,r):
        return ((1.0+np.cos(np.pi*(d/r)))/2.0)**2
    
    r = 100.0
    d = np.arange(0,101,dtype=np.float)
    
    #plt.plot(d,wgt_exp(d, r))
    plt.plot(d,wgt_gauss(d, r))
    plt.plot(d,wgt_gauss2(d, r))
    #plt.plot(d,wgt_cos(d, r))
    plt.show()

def reverseCmapDict(cdict):
    cdict_r = {}
    
    r = np.array(cdict['red'])
    r[:,1] = r[:,1][::-1]
    r[:,2] = r[:,2][::-1]
    
    g = np.array(cdict['green'])
    g[:,1] = g[:,1][::-1]
    g[:,2] = g[:,2][::-1]
    
    b = np.array(cdict['blue'])
    b[:,1] = b[:,1][::-1]
    b[:,2] = b[:,2][::-1]
    
    cdict_r['red'] = [tuple(x) for x in tuple(r)]
    cdict_r['green'] = [tuple(x) for x in tuple(g)]
    cdict_r['blue'] = [tuple(x) for x in tuple(b)]
    
    return cdict_r
    

def colorMapTest():
    cdict = {'red':[(0.0,   194/255.0,      194/255.0),
                    (0.2,   237.0/255.0,    237.0/255.0),
                    (0.4,   255.0/255.0,    255.0/255.0),
                    (0.6,   0.0/255.0,      0.0/255.0),
                    (0.8,   32.0/255.0,     32.0/255.0),
                    (1.0,   11.0/255.0,     11.0/255.0)],
         'green': [ (0.0,   80/255.0,       80/255.0),
                    (0.2,   161/255.0,      161/255.0),
                    (0.4,   255.0/255.0,    255.0/255.0),
                    (0.6,   219.0/255.0,    219.0/255.0),
                    (0.8,   153.0/255.0,    153.0/255.0),
                    (1.0,   44.0/255.0,     44.0/255.0)],
         'blue': [  (0.0,   60/255.0,       60/255.0),
                    (0.2,   19/255.0,       19/255.0),
                    (0.4,   0.0/255.0,      0.0/255.0),
                    (0.6,   0.0/255.0,      0.0/255.0),
                    (0.8,   143.0/255.0,    143.0/255.0),
                    (1.0,   122.0/255.0,    122.0/255.0)]}
    
    
    my_cmap = matplotlib.colors.LinearSegmentedColormap('prcp',reverseCmapDict(cdict),256)
    #cm.register_cmap('prcp', cmap=my_cmap)
    
    ds = Dataset('/stage/climate/topowx_tile_output/h05v02/h05v02_tmin.nc')
    plt.imshow(ds.variables['tmin_normal'][11,:,:],cmap=my_cmap)
    plt.show()
    
 
if __name__ == '__main__':
    
    colorMapTest()
    #test_exp_gauss()
    #perfInterpTair()
    #plotLstStn()
    #plotGlacStns()
    #glacStnValidationDaymet()
    #glacStnValidation()
    #plotCceStns()
    #randomCcePts()
    #output_tile_nc()
    #output_tile_csv()
    #statsImpute()
    #xvalStatsImpute()
    #save_stns_to_csv()
    #cProfile.run('run_full_interp_pt()')
    #run_full_interp_pt_xval()
    
    #nearestGridCellTest()
    #stnMetaFileRuben()
    #stnDataFilesRuben()
    
    #tmaxShiftValidation()
    #imputeLogAnalysis()
    #daymetAnnPlot()
    #updateImputeDaily()
    #xvalStatsImpute()
    #Optim/xval stations methods
    ###################################
    #optimTairMeanStats()
    #new_krig_xval_stats()
    #overall_xval_stats()
    #anomaly_stats()
    ###################################
    
    #test_impute_tair_norm()
    #addUTC()
    #subsetPOR()
    
    #run_full_interp_pt()
    
    #usClimDivStnsMap()
    #mooseRasters()
    #saveSerialStnList()
    #buildLowRestSnotel()
    #hcnMap()
    #summary_stats_chunks()
    #tairTrends()
    #overall_xval_stats()
    #anomaly_stats()
    #new_krig_xval_stats()
    #rmStnMap()
    #optimTairMeanHomogVsRaw()
    #optimTairMeanStats()
    #tmin_xval_neon_stats()
    #ghcnStnCsv()
    #testGhcnStnList()
    #analyzeRmStns()
    #analyzeBadImpsHighMAE()
    #stnsForRuben()
    #updateImputeDaily()
    #analyzeBadImps()
    #summary_stats_daymet()
    #stnsForRuben()
    #prcpStnsForRuben()
    #prcpStnMetaFileRuben()
    #stnMetaFileRuben()
    #prcpStnDataFilesRuben()
    #stnDataFilesRuben()
    #run_full_interp()
    #resetInfills()
    #imputeNormNoXval()
    #imputeDailyNoXval()
    #unusable_pha_stns()
    #discont_analysis()
    #overall_xval_stats()
    #discont_analysis_daily()
    #optim_krigsmth()
    #new_krig_xval_stats()
    #test_xval_krig()
    #plot_krig_params()
    #set_optim_nnghs()
    #testGwrPcaTair()
    #fix_lst_data()
    #imputation_plot()
    #montana_aoi_por_csv()
    #montana_aoi_interp_err_stats()
    #overall_interp_err_stats()
    #test_impute_daily_noxval()
    #temporal_variability_analysis()
    #set_0_lcc()
    #combine_montana_conus_mask()
    #lcc_for_montana_aoi('/projects/daymet2/dem/interp_grids/montana/tifs/crop_lcc.tif')
    #save_stns_to_csv()
    #testInterpTairDynParams2()
    #output_tile_nc()
    #save_stns_to_R()
    #chk_impute_cnts()
    #chk_rastvals_stns()
    #dem_nad_to_wgs()
    #chk_lc_stns()
    #extreme_infill_qa()
    #lst_mean_bymth()
    #lst_bitmask_testing()
    #test_impute_daily()
    #test_impute_daily_noxval()
    #test_impute_norm()
    #qa_cnts()
    #qa_mtsnotel_locs()
    #stn_infill_bias_map()
    #qa_testing()
    #stn_cnts()
    #test_impute_norm()
    #xval_stats_impute_norm()
    #xval_stats_impute_daily()
    #test_impute_daily()
    #xval_stats_impute_norm()
    #test_impute_tair_norm_new()
    #ghcn_raws()
    #coast_tmax_analysis()
    #test_impute_tair_norm()
    #test_single_tile()
    #arcgic_ncdf_test()
    #create_cce_blank_neon_nc()
    #create_nc_grids()
    #create_empty_neon_var()
    #save_stns_to_R()
    #tmin_xval_neon_stats()
    #fix_stn_loc_in_serial()
    #neon_stn_counts()
    #testInterpTairDynParams2()
    #anomaly_stats()
    #test_KrigTairPtRadius()
    #testGwSampleVario()an
    #build_gvar()
    #mean_mt_ndvi()
    #linear_interp_envlimits()
    #run_output_no_tdi_stns()
    #testInterpTair()
    #testGwrPcaTair()
    #test_KrigTairDynamic()
    #build_gvar()
    #test_KrigTair()
    #neon_buf_stn_cnts()
    #test_stn_slct_neon()
    #tmin_xval_neon_stats()
    #test_neon_varios()
    #fix_gate_park()
    #check_spatial_qa()
    #TEST_INTERP_TMINTMAX()
    #TEST_INTERP_TAIR()
    #create_nc_grids()
    #reset_infill_stn()
    #find_dup_stns()
    #xval_tmin_mae_analysis()
    #testTopoDisectDEM()
    #TEST_GWPCA()
    #load_test_pca()
    #test_modis_sin_rast()
    #TEST_INTERP_TAIR()
    #OUTPUT_STN_CSV()
    #TEST_PCACOR_IMPUTE_TAIR()
    #TEST_INTERP_TMAX()
    #TEST_INTERP_TAIR()
    #TEST_INTERP_TAIR_NNR()
    #reset_imputes_by_mae()
    #extract_tiles()
    #TEST_MICROMET_IMPUTE_TAIR()
    #TEST_PCACOR_IMPUTE_TAIR()
    #TEST_AMELIA_IMPUTE_TAIR()
    #TEST_AMELIA_IMPUTE_TAIR()
    #TEST_GWRPCA_INFILL_TAIR()
    #TEST_AMELIA_MTH_IMPUTE_TAIR()
    #TEST_AMELIA_IMPUTE_TAIR()
    #TEST_NNRPCA_INFILL_TAIR()
    #TEST_GWRPCA_INFILL_TAIR()
    #TEST_INFILL_TAIR()
    #xval_stats_by_neon()
    #TEST_INFILL_PRCP_XVAL()
    #test_infill_prcp_pct()
    #transform_prcp_norms('/projects/daymet2/station_data/infill/normals_prcp_log10.nc')
    #test_infills_prcp_norms_error()
    #test_dist()
    #test_infill_prcp_transform()
    #test_infill_prcp_transform2()
    #test_infill_prcp_by_mth()
    #test_infill_po_norms() 
    #test_infill_extrapolate_prcp()
    #error_stats_tair()
    #infill_error_map()
    #montana_stns_kml()
    #po_analysis()
    #quick_interp_cory_prcp()
    #quick_interp_cory_tair()
    #sites_dm_dd()
    #infill_micromet()
    #create_nometa_db()
    #stn_id_list()
    #pnw_raws()
    #infill_micromet()
    #qa_stats()
    #debug_qa()
    #reset_raw_flags()
    #debug_qa()
    #gtopo_prism_mask_merge()
    #stn_map()
    #gtopo_prism_merge2()
    #fix_raws_loc_qa()
    #srtm_to_ncdf()
    #fix_raws_prcp()
    #srtm_prism_clip()
    #stn_names_location_qa()
    #gtopo_prism_merge()
    #co_site()
    #stn_csv_from_por()
    #fix_ca_locs()
    #stn_csv_from_por()
    #gtopo_prism_merge()
    #output_stn_csv()
    #mean_val_to_geotiff()
    #test_trends()
    #test_quick_interps()
    #xval_cir_mae()
    #interp_tile_map()
    #find_optim_po_nghs()
    #hare_snotels()
    #hare_closest_sites()
    #hare_site_interp()
    #xval_prcp_map()
    #test_po_mth_thres_interp()
    #test_prcp_interp()
    #create_centered_prcp_ds()
    #create_center_prcp_var()
    #test_po_mth_thres_interp()
    #create_neon_nc()
    #test_po_interp()
    #make_tiles_ncdf()
    #test_tair_interp_xval()
    #interp_tair_map()
    #test_infill_prcp_with_stats()
    #extrapolate_map()
    #error_map()
    #test_sngl_tair_interp()
    #test_tair_interp()
    #center_dataset(Dataset('/projects/daymet2/station_data/infill/infill_tmin_center.nc','r+'), 'tmin')
    #center_dataset(Dataset('/projects/daymet2/station_data/infill/infill_tmax_center.nc','r+'), 'tmax')

    #add_fill_means()
    #analyze_infill_po()
    #extrapolate_map_po()
    #test_infill_extrapolate_po()
    #test_infill_extrapolate_prcp()
    #test_moments_extrapolate_prcp()
    #test_infill_outputs()
    #test_infill()
    #hcn_map()
    #test_infill_extrapolate_new()
    #extrapolate_map()
    #analyze_infill_error()
    #test_po_infill()
    #hare_closest_sites()
    #glac_lapse_rate_stns()
    #test_infill_extrapolate()
    #check_stn_locs()
    #test_ncdf_db()
    #station_netcdf()
    #reset_glac_flags()
    #station_netcdf_insert()
    #get_date_indices()
    #check_qaflags()
    #station_netcdf()
    #check_station_netcdf()
    #station_netcdf_insert()
    #station_netcdf()
    #check_smry_file()
    #broot_subset()
    
#    aspect = input_raster("/projects/daymet2/dem/aspect.tif")
#    slope = input_raster("/projects/daymet2/dem/slope.tif")
#    horiz_e = input_raster("/projects/daymet2/dem/horiz_e.tif")
#    horiz_w = input_raster("/projects/daymet2/dem/horiz_w.tif")
#    
#    aspect.to_ncdf("/projects/daymet2/dem/smoothed/ncdf/aspect.nc","aspect","f4")
#    slope.to_ncdf("/projects/daymet2/dem/smoothed/ncdf/slope.nc","slope","f4")
#    horiz_e.to_ncdf("/projects/daymet2/dem/smoothed/ncdf/horiz_e.nc","horiz_e","f4")
#    horiz_w.to_ncdf("/projects/daymet2/dem/smoothed/ncdf/horiz_w.nc","horiz_w","f4")
    
#    dem = Dataset("/projects/daymet2/dem/smoothed/ncdf/dem_orig.nc")
#    
#    #Convert original flt64 DEM to flt32
#    dem = Dataset("/projects/daymet2/dem/smoothed/ncdf/dem_orig.nc")
#    elev = dem.variables['elev'][:,:]
#    elev[elev==netCDF4._default_fillvals['f8']] = np.nan
#    elev32 = np.array(elev,dtype=np.float32)
#    elev32[elev.mask] = netCDF4._default_fillvals['f4']
#    
#    lats = dem.variables['lat'][:]
#    lons = dem.variables['lon'][:]
#    dem_new = Dataset("/projects/daymet2/dem/smoothed/ncdf/dem_orig2.nc","w")
#    dem_new.createDimension('lat',lats.size)
#    dem_new.createDimension('lon',lons.size)
#    latitudes = dem_new.createVariable('lat','f8',('lat',),fill_value=False)
#    latitudes.long_name = "latitude"
#    latitudes.units = "degrees_north"
#    latitudes.standard_name = "latitude"
#    latitudes[:] = lats
#    longitudes = dem_new.createVariable('lon','f8',('lon',),fill_value=False)
#    longitudes.long_name = "longitude"
#    longitudes.units = "degrees_east"
#    longitudes.standard_name = "longitude"
#    longitudes[:] = lons
#    
#    elev_new = dem_new.createVariable('elev','f4',('lat','lon'),fill_value=False)
#    elev_new.missing = netCDF4._default_fillvals['f4']
#    elev_new[:,:] = elev32
#    dem_new.close()
#    sys.exit()
#    
##    load_stns()
##    sys.exit()
#    stns = np.load("/projects/daymet2/station_data/all_infill/decade_split/stns_infill.npy")
#    ncdf_data = Dataset('/projects/daymet2/station_data/all_infill/decade_split/test_stns_infill.nc','w')
#    
#    stns_new = np.array(stns,dtype=[(STN_ID,"S1",16),(STATE,"S1",2),(LON,np.float64),(LAT,np.float64),(ELEV,np.float64),
#                              (INFILLED_TMIN,np.int),(INFILLED_TMAX,np.int),(INFILLED_PRCP,np.int),
#                              (POPCRIT_JAN,np.float64),(POPCRIT_FEB,np.float64),(POPCRIT_MAR,np.float64),
#                              (POPCRIT_APR,np.float64),(POPCRIT_MAY,np.float64),(POPCRIT_JUN,np.float64),
#                              (POPCRIT_JUL,np.float64),(POPCRIT_AUG,np.float64),(POPCRIT_SEP,np.float64),
#                              (POPCRIT_OCT,np.float64),(POPCRIT_NOV,np.float64),(POPCRIT_DEC,np.float64)])
#    
#    
#    
#    for x in np.arange(stns[STN_ID].size):
#        a_chr = np.zeros(16,dtype=np.character)
#        a_chr[:len(stns[STN_ID][x])] = list(stns[STN_ID][x])
#        stns_new[STN_ID][x,:] = a_chr
#        
#        a_chr = np.zeros(2,dtype=np.character)
#        a_chr[:] = list(stns[STATE][x])
#        stns_new[STATE][x,:] = a_chr
#        
#    a_dtype = stns_new.dtype
#
#    a_dtype_ncdf = ncdf_data.createCompoundType(a_dtype, "stn_dtype")
#    ncdf_data.createDimension("a_dim", size=None)
#    v = ncdf_data.createVariable("stns",a_dtype_ncdf,"a_dim")
#    v[:] = stns_new
#    
#    print stns_new[LON]
#    ncdf_data.close()
#    
#    ncdf_data = Dataset('/projects/daymet2/station_data/all_infill/decade_split/test_stns_infill.nc')
#    stns_new = ncdf_data.variables['stns'][:]
#    print stns_new[LON]
#    ncdf_data.close()
#    np.sa
#    mask = input_raster('/projects/daymet2/dem/interp_mask_montana.tif')
#    a_mask = mask.readEntireRaster()
#    mask.to_ncdf("/projects/daymet2/dem/smoothed/ncdf/interp_mask_montana.nc","mask", 'i2')
    
#    dem = input_raster("/projects/daymet2/dem/smoothed/dem_1_0.tif")
#    a_dem = dem.readEntireRaster()
#    dem.to_ncdf("/projects/daymet2/dem/smoothed/ncdf/dem_1_0.nc","elev",'f8')
    #ncdf_dem = Dataset("/projects/daymet2/dem/smoothed/ncdf/dem_1_0.nc","r")
#    elev_var = ncdf_dem.variables['elev']
#    lon_var = ncdf_dem.variables['lon']
#    lat_var = ncdf_dem.variables['lat']
#    a_dem_ncdf = elev_var[:,:]
#    lat1,lon1 = dem.getLatLon(500, 500, transform=False)
#    lat2,lon2 = dem.getLatLon(500, 501, transform=False)
#    print lat1 - lat2
#    
#    
#    print lat_var[400] - lat_var[401]
#    print lat_var[501] - lat_var[500]
#    print lon_var[200] - lon_var[199]
