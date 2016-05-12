'''
Copyright 2014,2015, Jared Oyler.

This file is part of TopoWx.

TopoWx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TopoWx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.
'''
from twx.infill.infill_daily import NONOPTIM_IMPOSS_VAL, NONOPTIM_VARI_CHGPT

__all__ = ['add_monthly_normals', 'add_stn_raster_values',
           'create_serially_complete_db', 'find_dup_stns',
           'set_bad_stations', 'update_daily_infill', 'get_bad_infill_stnids']

import numpy as np
from netCDF4 import Dataset, num2date
import netCDF4
from twx.db import create_quick_db, STN_ID, BAD, LON, LAT
from twx.db.station_data import _build_stn_struct
from twx.utils import get_days_metadata, StatusCheck, grt_circle_dist, TairAggregate
import mpl_toolkits.basemap as bm

SERIAL_DB_VARIABLES = {'tmin':[('tmin', 'f4', netCDF4.default_fillvals['f4'], 'minimum air temperature', 'C'),
                               ('flag_infilled', 'i1', netCDF4.default_fillvals['i1'], 'infilled flag', '')],
                       'tmax':[('tmax', 'f4', netCDF4.default_fillvals['f4'], 'maximum air temperature', 'C'),
                               ('flag_infilled', 'i1', netCDF4.default_fillvals['i1'], 'infilled flag', '')]}

# 5 years of data
USE_ALL_INFILL_THRESHOLD = np.round(365.25 * 5.0)

def update_daily_infill(stnid, tair_var, fpath_db, fnl_tair, mask_infill, infill_tair):
    '''
    Update the daily values for a station in an infilled netCDF station database
    
    Parameters
    ----------
    stnid : str
        The station id of the station to update
    tair_var : str
        The temperature variable ('tmin' or 'tmax') to update
    fpath_db : str
        The file path to the infilled netCDF station database
    fnl_tair : ndarray
        Time series of station observations with missing values infilled
    mask_infill : ndarray
        Boolean array specifying which values were infilled in fnl_tair
    infill_tair : ndarray
        Time series of station observations with all observations replaced with
        values from the infill model regardless of whether or not an
        observation was originally missing. 
    '''
    
    obs_mask = np.logical_not(mask_infill)
    difs = infill_tair[obs_mask] - fnl_tair[obs_mask]
    mae = np.mean(np.abs(difs))
    bias = np.mean(difs)

    ds_out = Dataset(fpath_db, 'r+')
    stnids = ds_out.variables['stn_id'][:].astype(np.str)
    i = np.nonzero(stnids == stnid)[0][0]

    ds_out.variables[tair_var][:, i] = fnl_tair
    ds_out.variables["".join([tair_var, "_infilled"])][:, i] = infill_tair
    ds_out.variables['flag_infilled'][:, i] = mask_infill
    ds_out.variables['bias'][i] = bias
    ds_out.variables['mae'][i] = mae
    ds_out.close()
    
def create_serially_complete_db(fpath_infill_db, tair_var, fpath_out_serial_db):
    '''
    Create a netCDF single variable, serially-complete station database and
    insert serially-complete station observations. Based on a specific threshold
    of total missing data, a station's serially-complete time series will either
    consist of a mix of actual and infilled observations or entirely of infilled
    observations from the infill model.
    
    Parameters
    ----------
    fpath_infill_db : str
        The file path to the infilled station database
    tair_var : str
        The temperature variable ('tmin' or 'tmax') for the database
    fpath_out_serial_db : str
        The file path for the output serially-complete database
    '''
    
    ds_infill = Dataset(fpath_infill_db)
    var_time = ds_infill.variables['time']
    stns = _build_stn_struct(ds_infill)
    start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
    days = get_days_metadata(start, end)
    
    create_quick_db(fpath_out_serial_db, stns, days, SERIAL_DB_VARIABLES[tair_var])
    ds_out = Dataset(fpath_out_serial_db, 'r+')
    
    all_infill_flags = np.ones(days.size, dtype=np.bool)
    all_infill_stns = np.zeros(stns.size, dtype=np.bool)
    
    stat_chk = StatusCheck(stns.size, 100)
    
    for x in np.arange(stns.size):
        
        infill_mask = ds_infill.variables['flag_infilled'][:, x].astype(np.bool)
        infill_runs = _runs_of_ones_array(infill_mask)
        
        if infill_runs.size > 0:
            max_infill = np.max(infill_runs)
        else:
            max_infill = 0
        
        if max_infill >= USE_ALL_INFILL_THRESHOLD:
            
            # This station has greater than USE_ALL_INFILL_THRESHOLD continuous
            # years of missing data. Use all infilled values for this station to avoid 
            # discontinuities between infilled and observed portions of time series
            tair_stn = ds_infill.variables["".join([tair_var, "_infilled"])][:, x]
            flag_stn = all_infill_flags
            
            all_infill_stns[x] = True
        
        else:
            
            tair_stn = ds_infill.variables[tair_var][:, x]
            flag_stn = infill_mask
            
        ds_out.variables[tair_var][:, x] = tair_stn
        ds_out.variables['flag_infilled'][:, x] = flag_stn
        ds_out.sync()
        
        stat_chk.increment()
    
    ds_out.close()
    
    print "% of stns with all infilled values: " + str((np.sum(all_infill_stns) / np.float(all_infill_stns.size)) * 100.)

def set_bad_stations(ds_serial, bad_ids, reset=True):
    '''
    Set the bad flag on a specific set of stations.
    
    Parameters
    ----------
    ds_serial : netCDF4.Dataset
        The station database on which to set the bad
        station flags.
    bad_ids : ndarray
        An array station ids that should be flagged
        as bad. If a specified station id is not in
        the database, it is ignored
    reset : boolean
        If true, reset all station bad flags to okay
        before flagging the passed bad_ids.
    '''
    
    db_stnids = ds_serial.variables[STN_ID][:].astype(np.str)
    
    if reset:
        ds_serial.variables[BAD][:] = 0
    
    for aid in bad_ids:
        
        try:
            x = np.nonzero(db_stnids == aid)[0][0]
        except IndexError:
            # Ignore station id not in database
            pass
        
        ds_serial.variables[BAD][x] = 1
    
    ds_serial.sync()

def find_dup_stns(stnda):
    '''
    Find duplicate stations in a netCDF4 infilled station database. Two or
    more stations are considered duplicates if they are at the exact
    same location. For two or more stations with the same
    location, the one with the longest non-infilled period-of-record is
    kept and the others are considered duplicates and will be returned
    by this function.
    
    Parameters
    ----------
    stnda : twx.db.StationSerialDataDb
        A StationSerialDataDb object pointing to the infilled 
        database that should be searched for duplicate stations.
        
    Returns
    ----------
    rm_stnids : ndarray
        An array of duplicate station ids
    '''
    
    dup_stnids = []
    rm_stnids = []
    
    stat_chk = StatusCheck(stnda.stns.size, 1000)
    
    for stn in stnda.stns:
        
        if stn[STN_ID] not in dup_stnids:
        
            ngh_stns = stnda.stns[stnda.stn_ids != stn[STN_ID]]
            dists = grt_circle_dist(stn[LON], stn[LAT], ngh_stns[LON], ngh_stns[LAT])
            
            dup_nghs = ngh_stns[dists == 0]
            
            if dup_nghs.size > 0:

                dup_stnids.extend(dup_nghs[STN_ID])
    
                stn_ids_load = np.sort(np.concatenate([np.array([stn[STN_ID]]).ravel(), np.array([dup_nghs[STN_ID]]).ravel()]))
                # print stn_ids_load
                stn_idxs = np.nonzero(np.in1d(stnda.stn_ids, stn_ids_load, True))[0]
                imp_flgs = stnda.ds.variables['flag_infilled'][:, stn_idxs]
                imp_flg_sum = np.sum(imp_flgs, axis=0)
                
                stn_ids_rm = stn_ids_load[imp_flg_sum != np.min(imp_flg_sum)]
                
                rm_stnids.extend(stn_ids_rm)
                
        stat_chk.increment()
    
    rm_stnids = np.array(rm_stnids)
    
    return rm_stnids

def add_stn_raster_values(stnda, var_name, long_name, units, a_rast, extract_method=1, revert_nn=False, force_data_value=False):
    '''
    Extract raster values for station locations and add them to a 
    serially-complete netCDF station database. Uses mpl_toolkits.basemap.interp
    to extract raster values. 
    
    Parameters
    ----------
    stnda : twx.db.StationSerialDataDb
        A StationSerialDataDb object pointing to the
        database to which station raster values should
        be added.
    var_name : str
        The netCDF variable name to be used for the raster values
    long_name : str
        The long netCDF variable name to be used for the raster values
    units : str
        The units of the raster values
    a_rast : RasterDataset
        The raster dataset from which to extract values
    extract_method : int, optional
        The mpl_toolkits.basemap.interp interpolation method to use for
        extraction specified as an integer (0 = nearest neighbor, 1 = bilinear,
        3 = cubic spline). Default = 1.
    revert_nn : boolean, optional
        Set to True if extract_methods > 0 should revert to nearest neighbor
        if a value cannot be extracted with the specified extract method. Default = False.
    force_data_value : boolean, optional
        If True, for station locations that have no data raster values or are outside
        the extent of the raster, the nearest grid cell with a value will be used.
        Default = False.
        
    Returns
    ----------
    stnids_ndata : ndarray
        An array of station ids for which raster values could not be extracted
    '''
    
    lon = stnda.stns[LON]
    lat = stnda.stns[LAT]
    
    newvar = stnda.add_stn_variable(var_name, long_name, units, 'f8', fill_value=a_rast.ndata)
    
    # Setup data and coordinates for mpl_toolkits.basemap.interp
    a = a_rast.read_as_array()
    aflip = np.flipud(a)
    aflip = aflip.astype(np.float)
    a = a.data
    
    yGrid, xGrid = a_rast.get_coord_grid_1d()
    yGrid = np.sort(yGrid)
        
    # Initialize output array
    rvals = np.zeros(len(newvar[:]))
    
    # Loop through stations
    schk = StatusCheck(lon.size, 5000) 
    for x in np.arange(lon.size):
        
        try:
            rval = bm.interp(aflip, xGrid, yGrid, np.array(lon[x]), np.array(lat[x]),
                             checkbounds=True, masked=False, order=extract_method)
        except ValueError:
            # ValueError means that station point is outside the bounds of the raster
            rval = np.ma.masked
        
        # Re-run nearest neighbor extraction with no checkbounds constraint if rval is 
        # masked (i.e.--no data) and at least one of the following conditions is met:
        # 1.) The station point is in bounds and extract method is nearest neighbor or
        # revert_nn is True.Since mpl_toolkits.basemap.interp uses coordinates based on cell centers,
        # stations near very edges of raster will produce ValueErrors with a checkbounds restriction
        # when they aren't actually out-of-bounds.
        # 2.) force_data_value = True. If point is outside bounds of raster, the returned
        # value will be clipped to the edge of raster.
        if (np.ma.is_masked(rval) and ((a_rast.is_inbounds(lon[x], lat[x]) 
                                        and (extract_method == 0 or revert_nn))
                                       or force_data_value)):
            
            rval = bm.interp(aflip, xGrid, yGrid, np.array(lon[x]), np.array(lat[x]),
                             checkbounds=False, masked=False, order=0)
        
        # If rval is still masked (i.e.--no data) and force_data_value is True,
        # find the nearest grid cell with data to the point
        if np.ma.is_masked(rval) and force_data_value:
                        
            if np.ma.is_masked(rval):
                
                row, col = a_rast.get_row_col(lon[x], lat[x], check_bounds=False)
                rval, dist = _find_nn_data(a, a_rast, col, row)
        
        if np.ma.is_masked(rval):
            rval = a_rast.ndata
          
        rvals[x] = rval
        schk.increment()
    
    newvar[:] = rvals
    stnda.ds.sync()
    
    stnids_ndata = stnda.stn_ids[rvals == a_rast.ndata]
    
    if force_data_value and stnids_ndata.size > 0:
        raise Exception('force_data_value turned on, but station points still had no data values.')
    
    return stnids_ndata

def add_monthly_normals(stnda, start_norm_yr=1981, end_norm_yr=2010):
    '''
    Calculate and add station monthly normals to a serially-complete 
    netCDF station database.
    
    Parameters
    ----------
    stnda : twx.db.StationSerialDataDb
        A StationSerialDataDb object pointing to the
        database to which station monthly normals should
        be added.
    start_norm_yr : int, optional
        The start year for the normals.
    end_norm_yr : int, optional
        The end year for the normals
    '''
    
    tagg = TairAggregate(stnda.days)
    stns = stnda.stns
    
    norm_vars = {}
    for mth in np.arange(1, 13):
        
        norm_var_name = 'norm%02d' % mth
        long_name = "%d - %d Monthly Normal" % (start_norm_yr, end_norm_yr)
        norm_var = stnda.add_stn_variable(norm_var_name, long_name, units='C',
                                         dtype='f8', fill_value=netCDF4.default_fillvals['f8'])
        norm_vars[mth] = norm_var
            
    dly_var = stnda.var
    
    chk_size = 50
    stchk = StatusCheck(np.int(np.round(stns.size / np.float(chk_size))), 10)
    
    for i in np.arange(0, stns.size, chk_size):
        
        if i + chk_size < stns.size:
            nstns = chk_size
        else:
            nstns = stns.size - i

        dly_vals = np.ma.masked_equal(dly_var[:, i:i + nstns], dly_var._FillValue)
        norm_vals = tagg.daily_to_mthly_norms(dly_vals, start_norm_yr, end_norm_yr)
                
        for mth in np.arange(1, 13):
            norm_vars[mth][i:i + nstns] = norm_vals[mth - 1, :]
        
        stnda.ds.sync()
        stchk.increment()

def get_bad_infill_stnids(fpath_log):
    
    afile = open(fpath_log)
    lines = np.array(afile.readlines())
    
    mask_err = np.char.startswith(lines, 'ERROR')
    mask_fail = np.char.find(lines, 'Could not infill') != -1
    mask_imposs = np.char.find(lines, NONOPTIM_IMPOSS_VAL) != -1
    mask_vaript = np.char.find(lines, NONOPTIM_VARI_CHGPT) != -1
    
    stnids_all = []
    
    # Get stnids that completely failed the infill process
    if np.sum(mask_fail) > 0:
        
        lines_fail = lines[mask_fail]
        lines_fail = np.char.split(lines_fail, "|")
        lines_fail = np.array([a_line[0] for a_line in lines_fail])
        lines_fail = np.char.split(lines_fail, " ")
        stnids_fail = np.sort(np.array([a_line[-1] for a_line in lines_fail]))
        stnids_all.append(stnids_fail)
    
    # Get stnids that had an error with a variance change point in the infilled
    # observations or an impossible value in the infilled observations
    mask_err2 = np.logical_and(mask_err, np.logical_or(mask_imposs, mask_vaript))
    
    if np.sum(mask_err2) > 0:     
        
        lines_err = lines[mask_err2]
        lines_err = np.char.split(lines_err, "|")
        lines_err = np.array([a_line[1] for a_line in lines_err])
        lines_err = np.char.split(lines_err, " ")
        stnids_err = np.sort(np.array([a_line[0] for a_line in lines_err]))
        stnids_all.append(stnids_err)
    
    stnids = np.unique(np.concatenate(stnids_all))
        
    return stnids

def _find_nn_data(a_data, a_rast, x, y):
                    
    r = 1
    nn = []
    nn_vals = []
    
    while len(nn) == 0:
        
        lcol = x - r
        rcol = x + r
        trow = y - r
        brow = y + r
        
        # top ring
        if trow > 0 and trow < a_rast.rows:
            
            for i in np.arange(lcol, rcol + 1):
                
                if i > 0 and i < a_rast.cols:
                                           
                    if a_data[trow, i] != a_rast.ndata:                                
                        nn.append((trow, i))
                        nn_vals.append(a_data[trow, i])
        
        # left ring
        if lcol > 0 and lcol < a_rast.cols:
            
            for i in np.arange(trow, brow + 1):
                
                if i > 0 and i < a_rast.rows:
                    
                    if a_data[i, lcol] != a_rast.ndata:                                                              
                        nn.append((i, lcol))
                        nn_vals.append(a_data[i, lcol])
                    
        # bottom ring
        if brow > 0 and brow < a_rast.rows:
            
            for i in np.arange(rcol, lcol, -1):
                
                if i > 0 and i < a_rast.cols:
                    
                    if a_data[brow, i] != a_rast.ndata:                                                                                          
                        nn.append((brow, i))
                        nn_vals.append(a_data[brow, i])
                        
        # right ring
        if rcol > 0 and rcol < a_rast.cols:
            
            for i in np.arange(brow, trow, -1):
                
                if i > 0 and i < a_rast.rows:
                    
                    if a_data[i, rcol] != a_rast.ndata:   
                        nn.append((i, rcol))
                        nn_vals.append(a_data[i, rcol])
        
        r += 1
    

    nn = np.array(nn)
    nn_vals = np.array(nn_vals)
    lats, lons = a_rast.get_coord(nn[:, 0], nn[:, 1])
    pt_lat, pt_lon = a_rast.get_coord(y, x)
    d = grt_circle_dist(pt_lon, pt_lat, lons, lats)
    j = np.argsort(d)[0]
    nval = nn_vals[j]
    return nval, d[j]

def _runs_of_ones_array(bits):
    # http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts
