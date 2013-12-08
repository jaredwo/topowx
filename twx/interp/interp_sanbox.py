'''
Created on Nov 18, 2011

@author: jared.oyler
'''
import cProfile
from station_select import station_select
from db.station_data import station_data_ncdf,LON,LAT,ELEV,NO_DATA,YMD,STN_ID
import numpy as np
from netCDF4 import Dataset
from reanalysis import free_air_interp
import utils.util_dates as utld
import matplotlib.pyplot as plt
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
from utils.input_raster import input_raster
r = robjects.r


def get_topo_disect_pt(lon,lat):
    #alter table stations add column NEON_RGN; 
    r_td = input_raster('/projects/daymet2/dem/topo_disect_msd.tif')
    return r_td.getDataValue(lon, lat)

def get_topo_disect(stns):
    #alter table stations add column NEON_RGN; 
    r_td = input_raster('/projects/daymet2/dem/topo_disect_msd.tif')
    a_td = r_td.readEntireRaster()
    
    td = np.zeros(stns.size)
    
    for i in np.arange(stns.size):
        
        x,y = r_td.getGridCellOffset(stns[LON][i],stns[LAT][i])
        td[i] = a_td[y,x]

    return td
    

def linear_regression():
    
    x = np.array([1.0,2.0,3.0,4.0,5,6,7,8,9,10])
    y = x+5.0
    
    X = np.mat(np.column_stack((np.ones(x.size),x)))
    Xt = np.transpose(X)
    
    np.linalg.inv(Xt*X)*Xt*y

def free_air_regress(fair,obs,wgt=None):
    
    robjects.globalenv["x"]=fair
    robjects.globalenv["y"]=obs
    if wgt == None:
        a_lm = r.lm("y~x")
    else:
        a_lm = r.lm("y~x",weights=wgt)
#    a_smry = r.summary(a_lm)
#    print a_smry.rx2('r.squared')[0]
    b0,b1 = np.asarray(a_lm.rx2('coefficients'))
    resid = np.asarray(a_lm.rx2('residuals'))
    
    return b0,b1,resid

def daymet_regress(stn_pairs,tair_pairs):
    
#    robjects.globalenv["delta_lon"]=stn_pairs[:,0]
#    robjects.globalenv["delta_lat"]=stn_pairs[:,1]
#    robjects.globalenv["delta_elev"]=stn_pairs[:,2]
    robjects.globalenv["delta_tair"]=tair_pairs
    
    a_lm = r.lm("delta_tair~delta_lon+delta_lat+delta_elev+delta_td-1",weights=stn_pairs[:,4])
    b1,b2,b3,b4 = np.asarray(a_lm.rx2('coefficients'))
    
#    a_lm = r.lm("delta_tair~delta_lon+delta_lat+delta_elev-1",weights=stn_pairs[:,4])
#    b1,b2,b3 = np.asarray(a_lm.rx2('coefficients'))
    
    b0 = 0.0
    #b4 = 0.0
    return b0,b1,b2,b3,b4 

def setup_r_env(stn_pairs):
    
    #lon,lat,elev,td,wgt
    robjects.globalenv["delta_lon"]=stn_pairs[:,0]
    robjects.globalenv["delta_lat"]=stn_pairs[:,1]
    robjects.globalenv["delta_elev"]=stn_pairs[:,2]
    robjects.globalenv["delta_td"]=stn_pairs[:,3]

def resid_regress(stn_pairs,delta_pairs):
    
    #lon,lat,elev,td,wgt
#    robjects.globalenv["delta_lon"]=stn_pairs[:,0]
#    robjects.globalenv["delta_lat"]=stn_pairs[:,1]
#    robjects.globalenv["delta_elev"]=stn_pairs[:,2]
#    robjects.globalenv["delta_td"]=stn_pairs[:,3]
    robjects.globalenv["delta"]=delta_pairs
    
    
#    plt.subplot(2,2,1)
#    plt.title('LON')
#    plt.plot(stn_pairs[:,0],delta_pairs,'.')
#    plt.subplot(2,2,2)
#    plt.title('LAT')
#    plt.plot(stn_pairs[:,1],delta_pairs,'.')    
#    plt.subplot(2,2,3)
#    plt.title('ELEV')
#    plt.plot(stn_pairs[:,2],delta_pairs,'.')
#    plt.subplot(2,2,4)
#    plt.title('TD')
#    plt.plot(stn_pairs[:,3],delta_pairs,'.')
#    plt.show()
    
    a_lm = r.lm("delta~delta_lon+delta_lat+delta_elev+delta_td-1",weights=stn_pairs[:,4])
    #a_lm = r.lm("delta~delta_lon+delta_lat+delta_elev",weights=stn_pairs[:,4])
    b1,b2,b3,b4 = np.asarray(a_lm.rx2('coefficients'))
    b0 = 0.0
    # b0,b1,b2,b3 = np.asarray(a_lm.rx2('coefficients'))
#    a_smry = r.summary(a_lm)
#    r2 = a_smry.rx2('r.squared')[0]
    #b4 = 0.0
    
    return b0,b1,b2,b3,b4
  
def build_dif_pairs(vals,npairs,sign_init):
    
    count = vals.size
    
    pairs = np.zeros(npairs)
    n = 0
    for i in np.arange(count):
        
        if i == count - 1:
            break
        
        val_i = vals[i]
        sign = sign_init
        
        for j in np.arange(i+1,count):
            
            pairs[n] = sign * (val_i - vals[j]);
            n+=1
            sign = -sign
    return pairs
 
def build_uniq_stn_pairs(lon,lat,elev,td,wgt,sign_init):
    
    count = lon.size
    npairs = ((count * count) - count) / 2;
    
    pairs = np.zeros((npairs,5))
    
    n = 0
    for stn_i in np.arange(count):
        
        if stn_i == count - 1:
            break
        
        sign = sign_init
        lon_i = lon[stn_i]
        lat_i = lat[stn_i]
        elev_i = elev[stn_i]
        td_i = td[stn_i]
        wgt_i = wgt[stn_i]
        
        for stn_j in np.arange(stn_i+1,count):
            
            pairs[n,0] = sign * (lon_i -lon[stn_j])
            pairs[n,1] = sign * (lat_i -lat[stn_j])
            pairs[n,2] = sign * (elev_i -elev[stn_j])
            pairs[n,3] = sign * (td_i -td[stn_j])
            pairs[n,4] = wgt_i * wgt[stn_j]
            
            n+=1
            sign = -sign
    
    return pairs

def predict_tair(coef,lon,lat,elev,td,lon_interp_stns,lat_interp_stns,elev_interp_stns,td_interp_stns,tair,wgt):
    
    sum_tair = 0
    sum_wgt = 0
    b0,b1,b2,b3,b4 = coef
    
    for x in np.arange(lon_interp_stns.size):
        #removed intercept
        sum_tair+= wgt[x]*(tair[x] + (b1*(lon - lon_interp_stns[x])) + (b2*(lat - lat_interp_stns[x])) + (b3*(elev - elev_interp_stns[x])) + (b4*(td - td_interp_stns[x])))
        sum_wgt+=wgt[x]
        
    return  sum_tair/sum_wgt 

def predict_resid(coef,lon,lat,elev,td,lon_interp_stns,lat_interp_stns,elev_interp_stns,td_interp_stns,resid,wgt):
    
    sum_resid = 0
    sum_wgt = 0
    b0,b1,b2,b3,b4 = coef
    
    for x in np.arange(lon_interp_stns.size):
        #removed intercept
        sum_resid+= wgt[x]*(resid[x] + (b1*(lon - lon_interp_stns[x])) + (b3*(elev - elev_interp_stns[x])) + (b2*(lat - lat_interp_stns[x])) + (b4*(td - td_interp_stns[x])))
        sum_wgt+=wgt[x]
        
    return  sum_resid/sum_wgt   

def run_interp_fair():
    
    #xval_stnid = 'GHCN_USC00246218' #olney
    #xval_stnid = 'SNOTEL_13A19S'
    #xval_stnid = 'GHCN_USW00024153' #missoula ap
    xval_stnid = 'GHCN_USC00247448' #seeley rs
    #xval_stnid = 'SNOTEL_13C01S'
    
    stn_path = '/projects/daymet2/station_data/all_infill/nc_files/stn_infill_tair.nc'
    obs_path = '/projects/daymet2/station_data/all_infill/nc_files/obs_infill_tmin.nc'
    obs_fair_path = '/projects/daymet2/station_data/all_infill/nc_files/obs_infill_fair1.nc'
    var = 'tmin'
    
    area = 326671.41
    num_stns = 549
    min_stns = 40
    rng_stns = 10
    max_stns = min_stns+rng_stns
    avg_stns = np.round(float(min_stns+max_stns)/2.0,0) 
    
    days = utld.get_days_metadata()
    
    stn_da = station_data_ncdf(stn_path)
    stns = stn_da.load_stns()
    
    obs_ds = Dataset(obs_path)
    obs = obs_ds.variables[var]
    stn_mask = np.logical_not(obs[0,:].mask)
    stn_idxs = np.nonzero(stn_mask)[0]
    stns = stns[stn_mask]
    
    xval_stn = stns[stns[STN_ID]==xval_stnid][0]
    lat = xval_stn[LAT]
    lon = xval_stn[LON]
    elev = xval_stn[ELEV]
    td = get_topo_disect_pt(lon, lat)
    stn_idx = stn_idxs[np.nonzero(stns[STN_ID]==xval_stnid)[0][0]]
    
    obs_fair_ds = Dataset(obs_fair_path)
    obs_fair = obs_fair_ds.variables["fair"]
    
    fair_interp = free_air_interp('/projects/daymet2/reanalysis_data/ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/conus.air.hgt.1948.2011.nc')
    
    stn_slct = station_select(stns, min_stns, max_stns, avg_stns, num_stns, area)
    interp_stn_idxs, wghts, radius = stn_slct.get_interp_stns((lat,lon),stns_rm=np.array([xval_stnid]))
    interp_obs_idxs = stn_idxs[interp_stn_idxs]
    
    obs_interp_stns = np.array(obs[:,interp_obs_idxs],dtype=np.float64)
    obs_fair_interp_stns = np.array(obs_fair[:,interp_obs_idxs],dtype=np.float64)
    lon_interp_stns = stns[LON][interp_stn_idxs]
    lat_interp_stns = stns[LAT][interp_stn_idxs]
    elev_interp_stns = stns[ELEV][interp_stn_idxs]
    td_interp_stns = get_topo_disect(stns[interp_stn_idxs])
    
#    elev_difs = np.abs(elev_interp_stns-elev)
#    elev_difs[elev_difs < 1.0] = 1.0
#    wghts_elev = 1.0/elev_difs
#    wghts_elev = wghts_elev/np.sum(wghts_elev)
#    
#    td_difs = np.abs(td_interp_stns-td)
#    td_difs[td_difs < 1.0] = 1.0
#    wghts_td = 1.0/td_difs
#    wghts_td = wghts_td/np.sum(wghts_td)
#    
#    wghts = wghts*wghts_elev*wghts_td
#    wghts = wghts/np.sum(wghts)
    
    obs_fair_pt = fair_interp.get_free_air(lon, lat, elev)
    
    predict_vals = np.zeros(days[YMD].size)

    #build weight matrix 
    W = np.mat(np.diag(wghts))
    
    #Build design matrix
    X = np.mat(np.column_stack((np.ones(lon_interp_stns.size),lon_interp_stns,lat_interp_stns,elev_interp_stns,td_interp_stns)))
    Xt = np.transpose(X)
    #Get inverse
    xterm = np.linalg.inv((Xt*W*X))*Xt*W
    
    X_fair = np.ones((lon_interp_stns.size,2))
    
    for x in np.arange(days[YMD].size):
        
        X_fair[:,1] = obs_fair_interp_stns[x,:]
        X_fair_mat = np.mat(X_fair)
        Xt_fair = np.transpose(X_fair_mat)
        
        coef = np.linalg.inv((Xt_fair*W*X_fair_mat))*Xt_fair*W*np.transpose(np.mat(obs_interp_stns[x,:]))
        b0_fair,b1_fair = coef[0,0],coef[1,0]
        resid1 = obs_interp_stns[x,:]-(b0_fair + b1_fair*obs_fair_interp_stns[x,:])
        
        coef2 = xterm*np.transpose(np.mat(resid1))
        b0,b1,b2,b3,b4 = coef2[0,0],coef2[1,0],coef2[2,0],coef2[3,0],coef2[4,0]
        
        resid2 = resid1-(b0 + b1*lon_interp_stns + b2*lat_interp_stns + b3*elev_interp_stns + b4*td_interp_stns)
        
        resid2_interp = np.average(resid2,weights=wghts)
        
        resid_interp = (b0 + (b1*lon) + (b2*lat) + (b3*elev) + (b4*td))- resid2_interp
        
        predict_vals[x] = (b0_fair + b1_fair*obs_fair_pt[x]) - resid_interp
            
    obs_vals = obs[:,stn_idx]    
    
    obs_vals = obs_vals
    predict_vals = predict_vals
    
    print np.mean(obs_vals)
    print np.mean(predict_vals)
    print np.mean(np.abs(obs_vals-predict_vals))     
    plt.plot(obs_vals)
    plt.plot(predict_vals)
    plt.show()

def run_interp_no_fair():
    
    #xval_stnid = 'GHCN_USC00246218' #olney
    #xval_stnid = 'SNOTEL_13A19S'
    xval_stnid = 'GHCN_USW00024153' #missoula ap
    #xval_stnid = 'GHCN_USC00247448' #seeley rs
    #xval_stnid = 'SNOTEL_13C01S'
    
    stn_path = '/projects/daymet2/station_data/all_infill/nc_files/stn_infill_tair.nc'
    obs_path = '/projects/daymet2/station_data/all_infill/nc_files/obs_infill_tmin.nc'
    var = 'tmin'
    
    area = 326671.41
    num_stns = 549
    min_stns = 40
    rng_stns = 10
    max_stns = min_stns+rng_stns
    avg_stns = np.round(float(min_stns+max_stns)/2.0,0) 
    
    days = utld.get_days_metadata()
    
    stn_da = station_data_ncdf(stn_path)
    stns = stn_da.load_stns()
    
    obs_ds = Dataset(obs_path)
    obs = obs_ds.variables[var]
    stn_mask = np.logical_not(obs[0,:].mask)
    stn_idxs = np.nonzero(stn_mask)[0]
    stns = stns[stn_mask]
    
    xval_stn = stns[stns[STN_ID]==xval_stnid][0]
    lat = xval_stn[LAT]
    lon = xval_stn[LON]
    elev = xval_stn[ELEV]
    td = get_topo_disect_pt(lon, lat)
    stn_idx = stn_idxs[np.nonzero(stns[STN_ID]==xval_stnid)[0][0]]
    
    stn_slct = station_select(stns, min_stns, max_stns, avg_stns, num_stns, area)
    interp_stn_idxs, wghts, radius = stn_slct.get_interp_stns((lat,lon),stns_rm=np.array([xval_stnid]))
    interp_obs_idxs = stn_idxs[interp_stn_idxs]
    
    obs_interp_stns = np.array(obs[:,interp_obs_idxs],dtype=np.float64)
    lon_interp_stns = stns[LON][interp_stn_idxs]
    lat_interp_stns = stns[LAT][interp_stn_idxs]
    elev_interp_stns = stns[ELEV][interp_stn_idxs]
    td_interp_stns = get_topo_disect(stns[interp_stn_idxs])
    
#    elev_difs = np.abs(elev_interp_stns-elev)
#    elev_difs[elev_difs < 1.0] = 1.0
#    wghts_elev = 1.0/elev_difs
#    wghts_elev = wghts_elev/np.sum(wghts_elev)
#    
#    td_difs = np.abs(td_interp_stns-td)
#    td_difs[td_difs < 1.0] = 1.0
#    wghts_td = 1.0/td_difs
#    wghts_td = wghts_td/np.sum(wghts_td)
#    
#    wghts = wghts*wghts_td*wghts_td
#    wghts = wghts/np.sum(wghts)
    
    predict_vals = np.zeros(days[YMD].size)

    #build weight matrix 
    W = np.mat(np.diag(wghts))
    
    #Build design matrix
    X = np.mat(np.column_stack((np.ones(lon_interp_stns.size),lon_interp_stns,lat_interp_stns,elev_interp_stns)))
    Xt = np.transpose(X)
    #Get inverse
    xterm = np.linalg.inv((Xt*W*X))*Xt*W
    
    for x in np.arange(days[YMD].size):
        
        ##############################
        # old daymet
        y = np.transpose(np.mat(obs_interp_stns[x,:]))
        coef = xterm*y
        b0,b1,b2,b3 = coef[0,0],coef[1,0],coef[2,0],coef[3,0]
        resid1 = obs_interp_stns[x,:]-(b0 + b1*lon_interp_stns + b2*lat_interp_stns + b3*elev_interp_stns)
        resid1_interp = np.average(resid1,weights=wghts)
        predict_vals[x] =  (b0 + (b1*lon) + (b2*lat) + (b3*elev))- resid1_interp
        ##############################
    
    obs_vals = obs[:,stn_idx]    
    
    obs_vals = obs_vals#[0:1000]
    predict_vals = predict_vals#[0:1000]
    
    print np.mean(obs_vals)
    print np.mean(predict_vals)
    print np.mean(np.abs(obs_vals-predict_vals))     
    plt.plot(obs_vals)
    plt.plot(predict_vals)
    plt.show()

def run_interp():
    
    #Seeley
#    lat = 47.230488
#    lon = -113.430344
#    elev = 1405.46972
#    td = 2.6764722
    #SNOTEL_13C01S
    #xval_stnid = 'GHCN_USC00247448' #seeley rs
    xval_stnid = 'SNOTEL_13C01S'
    
    stn_path = '/projects/daymet2/station_data/all_infill/nc_files/stn_infill_tair.nc'
    obs_path = '/projects/daymet2/station_data/all_infill/nc_files/obs_infill_tmin.nc'
    obs_fair_path = '/projects/daymet2/station_data/all_infill/nc_files/obs_infill_fair1.nc'
    var = 'tmin'
    
    area = 326671.41
    num_stns = 549
    min_stns = 40
    rng_stns = 10
    max_stns = min_stns+rng_stns
    avg_stns = np.round(float(min_stns+max_stns)/2.0,0) 
    
    days = utld.get_days_metadata()
    
    stn_da = station_data_ncdf(stn_path)
    stns = stn_da.load_stns()
    
    obs_ds = Dataset(obs_path)
    obs = obs_ds.variables[var]
    stn_mask = np.logical_not(obs[0,:].mask)
    stn_idxs = np.nonzero(stn_mask)[0]
    stns = stns[stn_mask]
    
    xval_stn = stns[stns[STN_ID]==xval_stnid][0]
    lat = xval_stn[LAT]
    lon = xval_stn[LON]
    elev = xval_stn[ELEV]
    td = get_topo_disect_pt(lon, lat)
    stn_idx = stn_idxs[np.nonzero(stns[STN_ID]==xval_stnid)[0][0]]
    
    obs_fair_ds = Dataset(obs_fair_path)
    obs_fair = obs_fair_ds.variables["fair"]
    
    fair_interp = free_air_interp('/projects/daymet2/reanalysis_data/ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/conus.air.hgt.1948.2011.nc')
    
    stn_slct = station_select(stns, min_stns, max_stns, avg_stns, num_stns, area)
    interp_stn_idxs, wghts, radius = stn_slct.get_interp_stns((lat,lon),stns_rm=np.array([xval_stnid]))
    interp_obs_idxs = stn_idxs[interp_stn_idxs]
    
    print interp_stn_idxs
    print interp_obs_idxs
    
    obs_interp_stns = np.array(obs[:,interp_obs_idxs],dtype=np.float64)
    obs_fair_interp_stns = np.array(obs_fair[:,interp_obs_idxs],dtype=np.float64)
    lon_interp_stns = stns[LON][interp_stn_idxs]
    lat_interp_stns = stns[LAT][interp_stn_idxs]
    elev_interp_stns = stns[ELEV][interp_stn_idxs]
    td_interp_stns = get_topo_disect(stns[interp_stn_idxs])
    
#    elev_difs = np.abs(elev_interp_stns-elev)
#    elev_difs[elev_difs < 1.0] = 1.0
#    wghts_elev = 1.0/elev_difs
#    wghts_elev = wghts_elev/np.sum(wghts_elev)
#    
#    td_difs = np.abs(td_interp_stns-td)
#    td_difs[td_difs < 1.0] = 1.0
#    wghts_td = 1.0/td_difs
#    wghts_td = wghts_td/np.sum(wghts_td)
#    
#    wghts = wghts*wghts_elev*wghts_td
#    wghts = wghts/np.sum(wghts)
    
    wgt_matrix = np.empty_like(obs_interp_stns)
    for x in np.arange(interp_stn_idxs.size):
        
        wgt_matrix[:,x] = wghts[x]
    
    wgt_matrix = wgt_matrix/np.sum(wgt_matrix)
    
    b0, b1, resid = free_air_regress(np.ravel(obs_fair_interp_stns,order='F'), np.ravel(obs_interp_stns,order='F'),np.ravel(wgt_matrix,order='F'))
    resid_matrix = np.reshape(resid, wgt_matrix.shape, order='F')
    
#    obs_fair_interp_stns = np.empty_like(obs_interp_stns)
#    for x in np.arange(interp_stn_idxs.size):
#        
#        obs_fair_interp_stns[:,x] = fair_interp.get_free_air_hgt(lon_interp_stns[x], lat_interp_stns[x],850)
#    
#    plt.plot(np.ravel(obs_fair_interp_stns,order='F'),np.ravel(obs_interp_stns,order='F'),".")
#    plt.show()
    
    print "creating free air..."
    obs_fair_pt = fair_interp.get_free_air(lon, lat, elev)
    #obs_fair_pt = fair_interp.get_free_air_hgt(lon, lat,850)
    
    stn_pairs = build_uniq_stn_pairs(lon_interp_stns, lat_interp_stns, elev_interp_stns, td_interp_stns, wghts,1.0)
    setup_r_env(stn_pairs)
    
    predict_vals = np.zeros(days[YMD].size)
    
    resid_predicts = []
    r2s = []
    
    for x in np.arange(days[YMD].size):
        
#        if x == 1000:
#            break
        #Old Daymet regressions
        #################################################
#        tair_pairs = build_dif_pairs(obs_interp_stns[x,:], stn_pairs.shape[0], 1.0)
#        coef = daymet_regress(stn_pairs, tair_pairs)
#        predict_vals[x] = predict_tair(coef, lon, lat, elev,td,lon_interp_stns, lat_interp_stns, elev_interp_stns,td_interp_stns, obs_interp_stns[x,:], wghts)
        #####################################################
        
        resid = resid_matrix[x,:]
#        b0,b1,resid = free_air_regress(obs_fair_interp_stns[x,:],obs_interp_stns[x,:],wghts)
        resid_pairs = build_dif_pairs(resid, stn_pairs.shape[0],1.0)
#        
#        #difs = obs_interp_stns[x,:]-obs_fair_interp_stns[x,:]
#        #dif_pairs = build_dif_pairs(difs, stn_pairs.shape[0], 1.0)
#        
        coef = resid_regress(stn_pairs, resid_pairs)
        #r2s.append(r2)
        resid_predict = predict_resid(coef, lon, lat, elev, td, lon_interp_stns, 
                              lat_interp_stns, elev_interp_stns, td_interp_stns, resid, wghts)
        #predict_vals[x] = obs_fair_pt[x] + resid_predict
        #resid_predicts.append(resid_predict)
        predict_vals[x] = (b0 + (b1*obs_fair_pt[x]))-resid_predict
    
    
    obs_vals = obs[:,stn_idx]    
    
    obs_vals = obs_vals#[0:1000]
    predict_vals = predict_vals#[0:1000]
    
    print np.mean(obs_vals)
    print np.mean(predict_vals)
    print np.mean(np.abs(obs_vals-predict_vals))     
    plt.plot(obs_vals)
    plt.plot(predict_vals)
    plt.show()
    
    #print "lm call....."
    #free_air_regress(np.ravel(obs_fair_interp_stns,order='F'), np.ravel(obs_interp_stns,order='F'))
    
    
    
#    plt.plot(np.ravel(obs_fair_interp_stns,order='F'),np.ravel(obs_interp_stns,order='F'),".")
#    plt.show()

if __name__ == '__main__':
    #run_interp_new()
    #cProfile.run('run_interp()')
    #run_interp()
    run_interp_no_fair()
    #run_interp_fair()
    