'''
Functions and classes for inserting station observation data from multiple sources
into a single database format.

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
from twx.db.station_data import StationSerialDataDb, BAD, StationDataDb
from datetime import date

__all__ = ['Insert', 'InsertGhcn', 'InsertRaws', 'InsertSnotel',
           'create_netcdf_db', 'insert_data_netcdf_db',
           'MISSING', 'DTYPE_STNOBS', 'NCDF_CHK_COLS', 'add_monthly_means',
           'add_utc_offset','dbDataset','create_quick_db', 'SnotelPreciseLoc',
           'add_obs_cnt', 'create_aux_db']

import os
import numpy as np
import datetime
from twx.utils import StatusCheck, get_days_metadata, ymdL, DATE, YMD, YEAR
from netCDF4 import Dataset
from netCDF4 import date2num
import netCDF4
from netCDF4 import num2date
from station_data import STN_NAME, ELEV, LAT, LON, STATE, STN_ID
import twx
import pandas as pd
from twx.db.download_stndata import load_snotel_stn_inventory
from twx.db.stn_utc_offsets import GeonamesError
import xray as xr

MISSING = -9999.
NCDF_CHK_COLS = 50
DTYPE_STNOBS = [('year', np.int), ('month', np.int), ('day', np.int), ('ymd', np.int),
               ('tmin', np.float64), ('tmax', np.float64), ('prcp', np.float64), ('swe', np.float64),
               ('qflag_tmin', "S1"), ('qflag_tmax', "S1"), ('qflag_prcp', "S1")]
SNOTEL_TMIN = 'TMIN.D-1 (degC) '
SNOTEL_TMAX = 'TMAX.D-1 (degC) '
SNOTEL_MISSING = -99.9

class dbDataset(Dataset):
    '''
    Subclass of netCDF4.Dataset. Provides utility methods for creating basic dimensions and variables
    most often used in this application
    '''

    def db_create_global_attributes(self, title):

        self.title = title
        self.institution = "University of Montana Numerical Terradynamics Simulation Group"
        self.history = "".join(["Created on: ", datetime.datetime.strftime(datetime.date.today(), "%Y-%m-%d")])

    def db_create_time_dimvar(self, days):

        self.createDimension('time', days.size)

        times = self.createVariable('time', 'f8', ('time',))
        times.long_name = "time"
        times.units = "".join(["days since ", str(np.min(days[YEAR])), "-1-1 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times[:] = date2num(days[DATE], times.units)

    def db_create_level_dimvar(self, levels):

        self.createDimension('level', levels.size)

        l = self.createVariable('level', 'f8', ('level',))
        l.long_name = "Level"
        l.units = "millibar"
        l.standard_name = "level"
        l[:] = levels

    def db_create_stnid_dimvar(self, stn_ids):

        self.createDimension('stn_id', stn_ids.size)

        stations = self.createVariable('stn_id', np.str, ('stn_id',))
        stations.long_name = "station id"
        stations.standard_name = "station id"
        stations[:] = stn_ids.astype(np.object)

    def db_create_names_var(self, names):

        namesvar = self.createVariable('name', np.str, ('stn_id',))
        namesvar.long_name = "station name"
        namesvar.standard_name = "name"
        namesvar[:] = names.astype(np.object)

    def db_create_states_var(self, states):

        statesvar = self.createVariable('state', np.str, ('stn_id',))
        statesvar.long_name = "state"
        statesvar.standard_name = "state"
        statesvar[:] = states.astype(np.object)

    def db_create_lonlat_dimvar(self, lons, lats):

        self.createDimension('lon', lons.size)
        self.createDimension('lat', lats.size)

        self.db_create_lon_var(lons, ('lon',))
        self.db_create_lat_var(lats, ('lat',))

    def db_create_lat_var(self, lats, dim=('stn_id',)):

        latitudes = self.createVariable('lat', 'f8', dim)
        latitudes.long_name = "latitude"
        latitudes.units = "degrees_north"
        latitudes.standard_name = "latitude"
        latitudes[:] = lats

    def db_create_lon_var(self, lons, dim=('stn_id',)):

        longitudes = self.createVariable('lon', 'f8', dim)
        longitudes.long_name = "longitude"
        longitudes.units = "degrees_east"
        longitudes.standard_name = "longitude"
        longitudes[:] = lons

    def db_create_elev_var(self, elev):

        elevvar = self.createVariable('elev', 'f8', ('stn_id',))
        elevvar.long_name = "elevation"
        elevvar.units = "m"
        elevvar.standard_name = "elevation"
        elevvar[:] = elev

    def db_create_stn_vars(self, stns):

        self.db_create_names_var(stns[STN_NAME])
        self.db_create_states_var(stns[STATE])
        self.db_create_lat_var(stns[LAT])
        self.db_create_lon_var(stns[LON])
        self.db_create_elev_var(stns[ELEV])

    def db_create_binflag_var(self, varname, lname, sname, dims=('time', 'stn_id'), chunk=None):

        flag_var = self.createVariable(varname, 'i1', dims, chunksizes=chunk)
        flag_var.long_name = lname
        flag_var.standard_name = sname
        flag_var.missing_value = netCDF4.default_fillvals['i1']

    def db_create_mae_var(self, dims=('stn_id',), units="C"):

        mae_var = self.createVariable('mae', 'f8', dims)
        mae_var.long_name = "mean absolute error"
        mae_var.units = units
        mae_var.standard_name = "mean_absolute_error"
        mae_var.missing_value = netCDF4.default_fillvals['f8']

    def db_create_bias_var(self, dims=('stn_id',), units="C"):

        bias_var = self.createVariable('bias', 'f8', dims)
        bias_var.long_name = "bias"
        bias_var.units = units
        bias_var.standard_name = "bias"
        bias_var.missing_value = netCDF4.default_fillvals['f8']

    def db_create_tmin_var(self, chunk=None, fill_value=netCDF4.default_fillvals['f4']):

        tmin_var = self.createVariable('tmin', 'f4', ('time', 'stn_id'), fill_value=fill_value, chunksizes=chunk)
        tmin_var.long_name = "minimum air temperature"
        tmin_var.units = "C"
        tmin_var.standard_name = "minimum_air_temperature"
        tmin_var.missing_value = fill_value

    def db_create_tmax_var(self, chunk=None, fill_value=netCDF4.default_fillvals['f4']):

        tmax_var = self.createVariable('tmax', 'f4', ('time', 'stn_id'), fill_value=fill_value, chunksizes=chunk)
        tmax_var.long_name = "minimum air temperature"
        tmax_var.units = "C"
        tmax_var.standard_name = "minimum_air_temperature"
        tmax_var.missing_value = fill_value

    def db_create_tair_var(self, var, chunk=None, fill_value=netCDF4.default_fillvals['f4']):

        if var == "tmin":
            self.db_create_tmin_var(chunk, fill_value)
        elif var == "tmax":
            self.db_create_tmax_var(chunk, fill_value)

    def db_create_prcp_var(self, chunk=None, fill_value=netCDF4.default_fillvals['f4']):

        ncdf_var = self.createVariable('prcp', 'f4', ('time', 'stn_id'), chunksizes=chunk)
        ncdf_var.long_name = "precipitation amount"
        ncdf_var.units = "cm"
        ncdf_var.standard_name = "precipitation_amount"
        ncdf_var.missing_value = fill_value

    def db_create_swe_var(self, chunk=None, fill_value=netCDF4.default_fillvals['f4']):

        ncdf_var = self.createVariable('swe', 'f4', ('time', 'stn_id'), chunksizes=chunk)
        ncdf_var.long_name = "snow water equivalent"
        ncdf_var.units = "cm"
        ncdf_var.standard_name = "snow_water_equivalent"
        ncdf_var.missing_value = fill_value

    def db_create_char_var(self, varname, lname, sname, dims=('time', 'stn_id'), chunk=None):

        ncdf_var = self.createVariable(varname, 'S1', dims, chunksizes=chunk)
        ncdf_var.long_name = lname
        ncdf_var.standard_name = sname
        ncdf_var.missing_value = ''

    def db_create_qflagtmin_var(self, dim=('time', 'stn_id'), chunk=None):

        self.db_create_char_var('qflag_tmin', "quality assurance flag tmin", "quality assurance flag tmin", dim, chunk)

    def db_create_qflagtmax_var(self, dim=('time', 'stn_id'), chunk=None):

        self.db_create_char_var('qflag_tmax', "quality assurance flag tmax", "quality assurance flag tmax", dim, chunk)

    def db_create_qflagprcp_var(self, dim=('time', 'stn_id'), chunk=None):

        self.db_create_char_var('qflag_prcp', "quality assurance flag prcp", "quality assurance flag prcp", dim, chunk)


def create_quick_db(path,stns,days,variables):
    '''
    Quickly create a netCDF4 weather station database for a specific set of stations,
    time period, and set of variables.
    
    Parameters
    ----------
    path : str
        File path for the database
    stns : structured array
        A structured array of stations from twx.db.StationDataDb.
    days : structured array
        A days array produced by twx.utils.get_days_metadata
    variables : list of tuples
        A list of tuples specifying the (time,stn_id) variables
        to be created. Each tuple is of size 5 and should contain: 
        variable name, data type, fill value, long name, and units.
    '''

    ncdf_file = Dataset(path, 'w')

    # Set global attributes
    title = "Weather Station Database"
    ncdf_file.title = title
    ncdf_file.institution = "Pennsylvania State University"
    ncdf_file.history = "".join(["Created on: ", datetime.datetime.strftime(datetime.date.today(), "%Y-%m-%d")])

    print "Creating netCDF4 database for " + str(days[DATE][0]) + " to " + str(days[DATE][-1]) + " for " + str(stns.size) + " stations."

    dim_time = ncdf_file.createDimension('time', days.size)
    dim_station = ncdf_file.createDimension(STN_ID, stns.size)

    times = ncdf_file.createVariable('time', 'f8', ('time',), fill_value=False)
    times.long_name = "time"
    times.units = "".join(["days since ", str(days[DATE][0].year), "-",
                           str(days[DATE][0].month), "-",
                           str(days[DATE][0].day), " 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE], times.units)

    stations = ncdf_file.createVariable(STN_ID, np.str, (STN_ID,))
    stations.long_name = "station id"
    stations.standard_name = "station id"
    stations[:] = stns[STN_ID].astype(np.object)

    names = ncdf_file.createVariable(STN_NAME, np.str, (STN_ID,))
    names.long_name = "station name"
    names.standard_name = "name"
    names[:] = stns[STN_NAME].astype(np.object)

    states = ncdf_file.createVariable(STATE, np.str, (STN_ID,))
    states.long_name = "state"
    states.standard_name = "state"
    states[:] = stns[STATE].astype(np.object)

    latitudes = ncdf_file.createVariable(LAT, 'f8', (STN_ID,), fill_value=MISSING)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = stns[LAT]

    longitudes = ncdf_file.createVariable(LON, 'f8', (STN_ID,), fill_value=MISSING)
    longitudes.long_name = 'longitude'
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = stns[LON]

    elevs = ncdf_file.createVariable(ELEV, 'f8', (STN_ID,), fill_value=MISSING)
    elevs.long_name = "elevation"
    elevs.units = "m"
    elevs.standard_name = "elevation"
    elevs[:] = stns[ELEV]
    
    for varname,dtype,fill_value,long_name,units in variables:
    
        a_var = ncdf_file.createVariable(varname, dtype, ('time', STN_ID),
                                         fill_value=fill_value, zlib=True,
                                         chunksizes=(days[DATE].size, 1))
        a_var.long_name = long_name
        a_var.units = units

    ncdf_file.close()

    print "Done creating NCDF Database......"


def create_netcdf_db(path, min_date, max_date, inserts):
    '''
    Create a netCDF4 weather station database. At this time, all dimensions are
    set and fixed at creation time (i.e.--dates covered and number of stations).
    Current variables are TMIN, TMAX,and associated quality flags.
    All data is prefilled with missing values. The dimensions of the main variables
    TMIN and TMAX are (time,stn_id) with chunks of size (time,50). TODO: see if this
    can be better optimized. Different dimension ordering? Chunk size? Compression?
    Starting in  netCDF4 version 4.3, performance access to this setup decreased 
    dramatically. Why?
    
    Parameters
    ----------
    path : str
        File path for the database
    min_date : datetime
        The earliest observation date
    max_date : datetime
        The latest observation date
    inserts : sequence
        A list of Insert objects. These are used to determine the length
        of the station dimension
    '''

    ncdf_file = Dataset(path, 'w')

    # Set global attributes
    title = "Weather Station Database"
    ncdf_file.title = title
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.history = "".join(["Created on: ", datetime.datetime.strftime(datetime.date.today(), "%Y-%m-%d")])

    days = get_days_metadata(min_date, max_date)

    nstns = 0
    for a_insert in inserts:
        nstns += len(a_insert.get_stns())

    print "Creating netCDF4 database for " + str(min_date) + " to " + str(max_date) + " for " + str(nstns) + " stations."

    dim_time = ncdf_file.createDimension('time', days.size)
    dim_station = ncdf_file.createDimension(STN_ID, nstns)

    times = ncdf_file.createVariable('time', 'f8', ('time',), fill_value=False)
    times.long_name = "time"
    times.units = "".join(["days since ", str(min_date.year), "-", str(min_date.month), "-", str(min_date.day), " 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE], times.units)

    stations = ncdf_file.createVariable(STN_ID, np.str, (STN_ID,))
    stations.long_name = "station id"
    stations.standard_name = "station id"

    names = ncdf_file.createVariable(STN_NAME, np.str, (STN_ID,))
    names.long_name = "station name"
    names.standard_name = "name"

    states = ncdf_file.createVariable(STATE, np.str, (STN_ID,))
    states.long_name = "state"
    states.standard_name = "state"

    latitudes = ncdf_file.createVariable(LAT, 'f8', (STN_ID,), fill_value=MISSING)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = MISSING

    longitudes = ncdf_file.createVariable(LON, 'f8', (STN_ID,), fill_value=MISSING)
    longitudes.long_name = 'longitude'
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = MISSING

    elevs = ncdf_file.createVariable(ELEV, 'f8', (STN_ID,), fill_value=MISSING)
    elevs.long_name = "elevation"
    elevs.units = "m"
    elevs.standard_name = "elevation"
    elevs[:] = MISSING

    tmin_var = ncdf_file.createVariable('tmin', 'f4', ('time', STN_ID),
                                        fill_value=MISSING,zlib=True,
                                        chunksizes=(days[DATE].size, 1))
    tmin_var.long_name = "minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "minimum_air_temperature"
    tmin_var.missing_value = MISSING
    #tmin_var[:, :] = MISSING

    tmax_var = ncdf_file.createVariable('tmax', 'f4', ('time', STN_ID),
                                        fill_value=MISSING,zlib=True,
                                        chunksizes=(days[DATE].size, 1))
    tmax_var.long_name = "maximum air temperature"
    tmax_var.units = "C"
    tmax_var.standard_name = "maximum_air_temperature"
    tmax_var.missing_value = MISSING
    #tmax_var[:, :] = MISSING

#    ncdf_var = ncdf_file.createVariable('prcp','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(days[DATE].size,NCDF_CHK_COLS))
#    ncdf_var.long_name = "precipitation amount"
#    ncdf_var.units = "cm"
#    ncdf_var.standard_name = "precipitation_amount"
#    ncdf_var.missing_value = MISSING
#    ncdf_var[:,:] = MISSING

    ncdf_var = ncdf_file.createVariable('qflag_tmin', 'S1', ('time', STN_ID),
                                        fill_value='',zlib=True,
                                        chunksizes=(days[DATE].size, 1))
    ncdf_var.long_name = "quality assurance flag tmin"
    ncdf_var.standard_name = "quality assurance flag tmin"
    ncdf_var.missing_value = ""
    #ncdf_var[:, :] = ""

    ncdf_var = ncdf_file.createVariable('qflag_tmax', 'S1', ('time', STN_ID),
                                        fill_value='',zlib=True,
                                        chunksizes=(days[DATE].size, 1))
    ncdf_var.long_name = "quality assurance flag tmax"
    ncdf_var.standard_name = "quality assurance flag tmax"
    ncdf_var.missing_value = ""
    #ncdf_var[:, :] = ""

#    ncdf_var = ncdf_file.createVariable('qflag_prcp','S1',('time','stn_id'),fill_value='',chunksizes=(days[DATE].size,NCDF_CHK_COLS))
#    ncdf_var.long_name = "quality assurance flag prcp"
#    ncdf_var.standard_name = "quality assurance flag prcp"
#    ncdf_var.missing_value = ""
#    ncdf_var[:,:] = ""

#    ncdf_var = ncdf_file.createVariable('swe','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(days[DATE].size,NCDF_CHK_COLS))
#    ncdf_var.long_name = "snow water equivalent"
#    ncdf_var.units = "cm"
#    ncdf_var.standard_name = "snow_water_equivalent"
#    ncdf_var.missing_value = MISSING
#    ncdf_var[:,:] = MISSING

    ncdf_file.close()

    print "Done creating NCDF Database......"


def insert_data_netcdf_db(db_path, insert_objs):
    '''
    Insert data into a netCDF4 weather station database
    that was created by create_netcdf_db.
    
    Parameters
    ----------
    db_path : str
        The file path to the database to which 
        the data should be inserted.
    insert_objs : sequence
        A list of Insert objects from which the 
        insert data will be obtained.
    '''

    ds = Dataset(db_path, 'r+')
    time_units = ds.variables['time'].units

    time_vals = ds.variables['time'][:]

    date_start = num2date(time_vals[0], units=time_units)
    date_end = num2date(time_vals[-1], units=time_units)
    days = get_days_metadata(date_start, date_end)

    ymd_idx = {}
    for x in np.arange(days[YMD].size):
        ymd_idx[days[YMD][x]] = x

    ########################################################
    # Get and insert all station metadata ordered by stn_id
    ########################################################
    
    print "Inserting station metadata..."
    
    all_stn_rows = []
    all_stn_rows_ls = []
    for insert in insert_objs:

        stn_rows = insert.get_stns()

        all_stn_rows.extend(stn_rows)
        all_stn_rows_ls.append(stn_rows)

    stn_ids = np.array([row[0] for row in all_stn_rows], dtype=np.object)
    lats = np.array([row[1] for row in all_stn_rows])
    lons = np.array([row[2] for row in all_stn_rows])
    elev = np.array([row[3] for row in all_stn_rows])
    state = np.array([row[4] for row in all_stn_rows], dtype=np.object)
    name = np.array([unicode(str(row[5]), errors="ignore") for row in all_stn_rows], dtype=np.object)

    sort_stnid = np.argsort(stn_ids)
    stn_ids = stn_ids[sort_stnid]
    lats = lats[sort_stnid]
    lons = lons[sort_stnid]
    elev = elev[sort_stnid]
    state = state[sort_stnid]
    name = name[sort_stnid]

    for x in np.arange(stn_ids.size):

        ds.variables[STN_ID][x] = str(stn_ids[x])
        ds.variables[LAT][x] = lats[x]
        ds.variables[LON][x] = lons[x]
        ds.variables[ELEV][x] = elev[x]
        ds.variables[STATE][x] = str(state[x])
        ds.variables[STN_NAME][x] = name[x]

    ds.sync()
    ########################################################
    ########################################################

    ########################################################
    # Get and insert all station observations
    ########################################################
    
    print "Inserting observations for each station..."

    stat_chk = StatusCheck(stn_ids.size, 100)

    for insert, stn_rows in zip(insert_objs, all_stn_rows_ls):

        stn_ids = ds.variables[STN_ID][:]

        stn_id = ""
        stn_idx = 0

        var_tmin = ds.variables['tmin']
        var_tmax = ds.variables['tmax']
        # var_prcp = ds.variables['prcp']
        var_qtmin = ds.variables['qflag_tmin']
        var_qtmax = ds.variables['qflag_tmax']
        # var_qprcp = ds.variables['qflag_prcp']
        # var_swe = ds.variables['swe']

        chkc = list(var_tmin.get_var_chunk_cache())
        chkc[0] = 1095750000  # bytes
        var_tmin.set_var_chunk_cache(chkc[0], chkc[1], chkc[2])
        var_tmax.set_var_chunk_cache(chkc[0], chkc[1], chkc[2])
        # var_prcp.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        var_qtmin.set_var_chunk_cache(chkc[0], chkc[1], chkc[2])
        var_qtmax.set_var_chunk_cache(chkc[0], chkc[1], chkc[2])
        # var_qprcp.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        # var_swe.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])

        stn_id = None
        stn_idx = []
        stn_cnt = 0

        tmin_chk = np.ones((days.size, NCDF_CHK_COLS), dtype=np.float32) * MISSING
        tmax_chk = np.ones((days.size, NCDF_CHK_COLS), dtype=np.float32) * MISSING
        # prcp_chk = np.ones((days.size,NCDF_CHK_COLS),dtype=np.float32)*MISSING
        qtmin_chk = np.zeros((days.size, NCDF_CHK_COLS), dtype=np.character)
        qtmax_chk = np.zeros((days.size, NCDF_CHK_COLS), dtype=np.character)
        # qprcp_chk = np.zeros((days.size,NCDF_CHK_COLS),dtype=np.character)
        # swe_chk = np.ones((days.size,NCDF_CHK_COLS),dtype=np.float32)*MISSING

        stn_id = None
        for stn_row in stn_rows:

            if stn_id != None:

                stn_cnt += 1

                if stn_cnt == NCDF_CHK_COLS:
                    
                    #Slicing by indices in netCDF4 now requires
                    #that indices are in order
                    stn_idx = np.array(stn_idx)
                    idx_sort = np.argsort(stn_idx)
                    
                    stn_idx = np.take(stn_idx,idx_sort)
                    tmin_chk = np.take(tmin_chk, idx_sort, axis=1)
                    tmax_chk = np.take(tmax_chk, idx_sort, axis=1)
                    qtmin_chk = np.take(qtmin_chk, idx_sort, axis=1)
                    qtmax_chk = np.take(qtmax_chk, idx_sort, axis=1)
                    
                    var_tmin[:, stn_idx] = tmin_chk
                    var_tmax[:, stn_idx] = tmax_chk
                    # var_prcp[:,stn_idx] = prcp_chk
                    var_qtmin[:, stn_idx] = qtmin_chk
                    var_qtmax[:, stn_idx] = qtmax_chk
                    # var_qprcp[:,stn_idx] = qprcp_chk
                    # var_swe[:,stn_idx] = swe_chk
                    tmin_chk[:, :] = MISSING
                    tmax_chk[:, :] = MISSING
                    # prcp_chk[:,:] = MISSING
                    qtmin_chk[:, :] = ""
                    qtmax_chk[:, :] = ""
                    # qprcp_chk[:,:] = ""
                    # swe_chk[:,:] = MISSING
                    stn_idx = []
                    stn_cnt = 0

            stn_id = stn_row[0]
            stn_idx.append(np.nonzero(stn_ids == stn_id)[0][0])
     
            obs = insert.parse_stn_obs(stn_id)

            if obs is not None:

                time_idx = np.in1d(days[YMD], obs['ymd'], assume_unique=True)
                # obs [stn_id,year,month,day,ymd,tmin,tmax,prcp,swe,"","",""]
                tmin_chk[time_idx, stn_cnt] = obs['tmin']
                tmax_chk[time_idx, stn_cnt] = obs['tmax']
                # prcp_chk[time_idx,stn_cnt] = obs['prcp']
                qtmin_chk[time_idx, stn_cnt] = obs['qflag_tmin']
                qtmax_chk[time_idx, stn_cnt] = obs['qflag_tmax']
                # qprcp_chk[time_idx,stn_cnt] = obs['qflag_prcp']
                # swe_chk[time_idx,stn_cnt] = obs['swe']

            stat_chk.increment()

        if len(stn_idx) > 0:

            idx = stn_cnt + 1

            tmin_chk_s = tmin_chk[:, 0:idx]
            tmax_chk_s = tmax_chk[:, 0:idx]
            qtmin_chk_s = qtmin_chk[:, 0:idx]
            qtmax_chk_s = qtmax_chk[:, 0:idx]
            
            stn_idx = np.array(stn_idx)
            idx_sort = np.argsort(stn_idx)
            
            stn_idx = np.take(stn_idx,idx_sort)
            tmin_chk_s = np.take(tmin_chk_s, idx_sort, axis=1)
            tmax_chk_s = np.take(tmax_chk_s, idx_sort, axis=1)
            qtmin_chk_s = np.take(qtmin_chk_s, idx_sort, axis=1)
            qtmax_chk_s = np.take(qtmax_chk_s, idx_sort, axis=1)
            
            
            var_tmin[:, stn_idx] = tmin_chk_s
            var_tmax[:, stn_idx] = tmax_chk_s
            # var_prcp[:,stn_idx] = prcp_chk[:,0:idx]
            var_qtmin[:, stn_idx] = qtmin_chk_s
            var_qtmax[:, stn_idx] = qtmax_chk_s
            # var_qprcp[:,stn_idx] = qprcp_chk[:,0:idx]
            # var_swe[:,stn_idx] = swe_chk[:,0:idx]
        ds.sync()

    ds.close()
    ########################################################
    ########################################################

class Insert:
    '''
    Parent class for inserting station data from multiple different sources 
    into a single netCDF database
    '''

    def __init__(self, min_date, max_date):
        '''
        Parameters
        ----------
        min_date : datetime
            The earliest observation date
        max_date : datetime
            The latest observation date
        '''

        self.min_ymd = ymdL(min_date)
        self.max_ymd = ymdL(max_date)

    def get_stns(self):
        '''
        Retrieve a list of station rows (i.e.--tuples) to be inserted.
        
        Returns
        -------
        stns : list
            A list of tuples. Each tuple is of format:
            (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        '''
        pass

    def parse_stn_obs(self, stn_id):
        '''
        Parse  observations for a station
        
        Parameters
        ----------
        stn_id : str
            The station id for which to parse observations
            
        Returns
        -------
        obs : ndarray
            An array of observations in chronological order
            with dtype DTYPE_STNOBS          
        '''
        pass

    def is_obs_inbounds(self, ymd):
        '''
        Return True if the year is between the specific min/max year bounds
        '''
        return ymd >= self.min_ymd and ymd <= self.max_ymd


class SnotelPreciseLoc():
    
    def __init__(self, fpath_precise_loc):
        
        self.stns_precise = self._parse_highres_stns(fpath_precise_loc)
    
    def get_precise_loc(self, stn_id, stn_name):
        

        name, lat, lon, elev, state = self.stns_precise[stn_id]
        
        if name != stn_name:
            
            print "Warning: Station %s non-precise name (%s) does not match precise name (%s)"%(stn_id,stn_name,name)
                    
        return lat, lon, elev
    
    def _parse_highres_stns(self,fpath):
    
        a_file = file(fpath)
        a_file.readline()
    
        locs_highres = {}
        for line in a_file.readlines():
            vals = line.split(",")
            st = vals[0].strip().upper()
            name = vals[1].strip().upper()
            stnid = vals[2].strip().upper()
    
            if stnid == '':
                continue
    
            lat = float(vals[5].strip())
            lon = float(vals[6].strip())
            elev = float(vals[8].strip()) * 0.3048  # convert from feet to meters
    
            locs_highres[stnid] = [name, lat, lon, elev, st]
        
        return locs_highres
    

class InsertSnotel(Insert):
    '''
    Class for inserting stations and observations from the SNOTEL/SCAN networks.
    Requires SNOTEL ASCII data to first be cleaned and formatted using the
    snotel_clean module
    '''

    def __init__(self, min_date, max_date, path_stn_obs_csv, fpath_stn_inventory=None, fpath_precise_loc=None, obs_prefix_field='cdbs_id'):
        '''
        Parameters
        ----------
        path_stn_file : str
            File path to the SNOTEL station metadata csv file
        path_clean_obs : str
            Path to SNOTEL clean observation files
        min_date : datetime
            The earliest observation date
        max_date : datetime
            The latest observation date
        '''

        Insert.__init__(self, min_date, max_date)
                
        self.stns_df = load_snotel_stn_inventory(fpath_stn_inventory)
        
        if fpath_precise_loc is not None:
            self.sntl_loc = SnotelPreciseLoc(fpath_precise_loc)
        else:
            self.sntl_loc = None
        
        self.path_stn_obs_csv = path_stn_obs_csv
        self.obs_prefix_field = obs_prefix_field

    def get_stns(self):
        '''
        Retrieve a list of station rows (i.e.--tuples) to be inserted.
        
        Returns
        -------
        stns : list
            A list of tuples. Each tuple is of format:
            (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        '''
        
        print "SNOTEL INSERT: Building list of stations..."
        
        print "SNOTEL INSERT: Determining which stations have observations..."
        
        has_obs = np.zeros(len(self.stns_df),dtype = np.bool)
        
        for i,stn_id in enumerate(self.stns_df[self.obs_prefix_field]):
            
            if os.path.exists(os.path.join(self.path_stn_obs_csv,stn_id)) and stn_id != '':
                
                has_obs[i] = True
                
        stns = self.stns_df[has_obs].copy()
        
        print "SNOTEL INSERT: A total of %d stations in inventory do not have observations and will not be inserted."%np.sum(~has_obs)
        
        if self.sntl_loc is None:
            
            print "SNOTEL INSERT: No high precise station locations provided. Will use imprecise locations."
            
        else:
            
            n_no_prec = 0
            
            print "SNOTEL INSERT: Setting high precision SNOTEL locations..."
            
            for i in np.arange(len(stns)):
                
                try:
                
                    p_lat, p_lon, p_elev = self.sntl_loc.get_precise_loc(stns['cdbs_id'].iloc[i], stns['site_name'].iloc[i])
                    x = stns.index[i]
                    stns.loc[x,['lat','lon','elev']] = p_lat, p_lon, p_elev
                    
                except KeyError:
                    
                    n_no_prec+=1
            
            print "SNOTEL INSERT: %d out of %d stations did not have high precision locations."%(n_no_prec,len(stns))
                    
        stns[self.obs_prefix_field] = ['SNOTEL_'+a_id for a_id in stns[self.obs_prefix_field]]
                
        stns = [tuple(x) for x in stns[[self.obs_prefix_field,'lat','lon','elev','state','site_name']].values]
        
        print "SNOTEL INSERT: Done building stations. Number of stns: %d" % (len(stns),)

        return stns

    def parse_stn_obs(self, stn_id):
        '''
        Parse observations for a station
        
        Parameters
        ----------
        stn_id : str
            The station id for which to parse observations
            
        Returns
        -------
        obs : ndarray
            An array of observations in chronological order
            with dtype DTYPE_STNOBS          
        '''

        # remove snotel prefix
        stn_id = stn_id.split("_")[1]

        path_stnobs = os.path.join(self.path_stn_obs_csv,stn_id)
        
        fnames = np.array(os.listdir(path_stnobs))
        fnames = fnames[np.logical_and(np.char.startswith(fnames, stn_id),np.char.endswith(fnames, '.csv'))]
        fnames = np.sort(fnames)
        
        obs_ls = []
        
        tdelta = np.timedelta64(1,'D')
        
        for a_fname in fnames:
            
            try:
            
                obs_df = pd.read_csv(os.path.join(path_stnobs,a_fname),skiprows=2, index_col=1, parse_dates=True)
                obs_df.index = obs_df.index - tdelta
            
            except Exception as e:
                
                print "SNOTEL INSERT: Warning, could not parse observations from %s. Skipping..."%(os.path.join(path_stnobs,a_fname))
                continue
            
            colnames = np.array(obs_df.columns,dtype=np.str)
            
#             icol_tmin = np.nonzero(np.char.startswith(colnames, 'TMIN'))[0]
#             icol_tmax = np.nonzero(np.char.startswith(colnames, 'TMAX'))[0]
            
            icol_tmin = np.nonzero(colnames == SNOTEL_TMIN)[0]
            icol_tmax = np.nonzero(colnames == SNOTEL_TMAX)[0]
        
            if icol_tmin.size == 0 and icol_tmax.size == 0:
            
                #No Tmin or Tmax observations in this file
                continue 
            
            elif icol_tmin.size > 1 or icol_tmax.size > 1:
                
                raise Exception('Unexpected observation column in %s'%os.path.join(path_stnobs,a_fname))
        
            if icol_tmin.size == 0:
                obs_df[SNOTEL_TMIN] = pd.Series(MISSING,index=obs_df.index)
                icol_tmin = obs_df.columns.get_loc(SNOTEL_TMIN)
            else:
                icol_tmin = icol_tmin[0]
                obs_df.loc[obs_df[obs_df.columns[icol_tmin]]==SNOTEL_MISSING,obs_df.columns[icol_tmin]] = MISSING
                
            if icol_tmax.size == 0:
                obs_df[SNOTEL_TMAX] = pd.Series(MISSING,index=obs_df.index)
                icol_tmax = obs_df.columns.get_loc(SNOTEL_TMAX)
            else:
                icol_tmax = icol_tmax[0]
                obs_df.loc[obs_df[obs_df.columns[icol_tmax]]==SNOTEL_MISSING,obs_df.columns[icol_tmax]] = MISSING
            
            # stn_id,year,month,day,ymd,tmin,tmax,prcp,swe,qflag_tmin,qflag_tmax,qflag_prcp
            obs_ls.extend([(year, month, day, ymd, a_obs[0], a_obs[1], MISSING, MISSING, "", "", "") for a_obs,year,month,day,ymd in zip(obs_df.iloc[:,[icol_tmin,icol_tmax]].values,
                                                                                                                                         obs_df.index.year,obs_df.index.month,obs_df.index.day,
                                                                                                                                         obs_df.index.format(formatter=lambda x: x.strftime("%Y%m%d")))])
    
        obs = np.empty(len(obs_ls), dtype=DTYPE_STNOBS)
        obs[:] = obs_ls
        obs = obs[np.argsort(obs['ymd'])]
        #only send back observations within time period of interest
        obs = obs[np.logical_and(obs['ymd'] >= self.min_ymd, obs['ymd'] <= self.max_ymd)]
        
        return obs


class InsertGhcn(Insert):
    '''
    Class for inserting stations and observations from the GHCN-Daily network
    into a netCDF database file
    '''

    VALID_FIPS_CODES = ["US", "CA", "MX"]
    VALID_ELEMENTS = ["TMIN", "TMAX", "PRCP"]
    RM_STATES = ["HI", "AK"]  # don't insert stations from HI or AK
    RM_NETWORK_CODES = ["S", "R"]  # don't insert SNOTEL and RAWS stations
    ELEMENT_INDICES = {"TMIN":(4, 8), "TMAX":(5, 9), "PRCP":(6, 10)}
    MONTH_DAYS = range(1, 32)
    OBS_COLUMN_SIZE = 8

    def __init__(self, path_stn_file, path_obs_files, min_date, max_date, check_obs_exist=False):
        '''
        Parameters
        ----------
        path_stn_file : str
            File path to the GHCN-D station metadata file (ghcnd-stations.txt)
        path_obs_files : str
            Path to GHCN-D observation files
        min_date : datetime
            The earliest observation date
        max_date : datetime
            The latest observation date
        max_date : datetime
            The latest observation date
        check_obs_exist : bool, optional
            If True, each GHCN station id in ghcnd-stations.txt will be 
            checked to make sure it has a corresponding *.dly file. If it does
            not, the station will skipped and not inserted. Checking for *.dly
            existance will significantly lengthen insert time, so only use
            if necessary..
        '''

        Insert.__init__(self, min_date, max_date)
        self.path_stn_file = path_stn_file
        self.path_obs_files = path_obs_files
        self.check_obs_exist = check_obs_exist

    def get_stns(self):
        '''
        Retrieve a list of station rows (i.e.--tuples) to be inserted.
        
        Returns
        -------
        stns : list
            A list of tuples. Each tuple is of format:
            (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        '''

        print "GHCN: Building list of stations"

        afile = open(self.path_stn_file)

        stns = []
        for line in afile.readlines():

            fips_code = line[0:2]

            if fips_code in self.VALID_FIPS_CODES:

                stn_id_orig = line[0:11].strip()
                network_id = stn_id_orig[2]
                stn_id = "".join(["GHCN_", stn_id_orig])
                lat = float(line[12:20].strip())
                lon = float(line[21:30].strip())
                elev = float(line[31:37].strip())
                state = line[38:40].strip().upper()
                name = unicode(line[41:71].strip(), errors='ignore')

                if state not in self.RM_STATES and network_id not in self.RM_NETWORK_CODES:

                    insert_stn = True

                    if self.check_obs_exist:

                        if  not os.path.exists(os.path.join(self.path_obs_files, "%s.dly" % (stn_id_orig,))):

                            insert_stn = False
                            print "".join([stn_id_orig, " in GHCN station list but no observations."])

                    if insert_stn:

                        stns.append((stn_id, lat, lon, elev, state, name))

        print "GHCN: Done building stations. Number of stns: %d" % (len(stns),)

        return stns

    def __convert_units(self, element, value):

        if value == -9999:
            # NO DATA, no conversion
            return value
        elif element == "PRCP":
            # tenths of mm to cm
            return value / 100.0
        elif element == "TMAX" or element == "TMIN" or element == "TOBS":
            # tenths of degrees C to degrees C
            return value / 10.0
        else:
            raise Exception("".join(["Invalid element type: ", element]))

    def parse_stn_obs(self, stn_id):
        '''
        Parse  observations for a station
        
        Parameters
        ----------
        stn_id : str
            The station id for which to parse observations
            
        Returns
        -------
        obs : ndarray
            An array of observations in chronological order
            with dtype DTYPE_STNOBS          
        '''

        obs_file = open(os.path.join(self.path_obs_files, "%s.dly" % (stn_id[5:],)))
        line = obs_file.readline()
        obs_dict = {}
        while len(line) > 0:

            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21].strip()

            if element in self.VALID_ELEMENTS:

                offset = 0

                for day in self.MONTH_DAYS:

                    try:

                        ymd = long("".join([str(year), "%02d" % (month,), "%02d" % (day,)]))
                        datetime.date(year, month, day)  # throw error if not valid date

                        if self.is_obs_inbounds(ymd):

                            value = self.__convert_units(element, float(line[21 + offset:offset + 26]))
                            qflag = line[27 + offset:offset + 28].strip()

                            if not obs_dict.has_key(ymd):

                                # obs [year,month,day,ymd,tmin,tmax,prcp,swe,"","",""]
                                obs_dict[ymd] = [year, month, day, ymd, MISSING, MISSING, MISSING, MISSING, "", "", ""]

                            obs_dict[ymd][self.ELEMENT_INDICES[element][0]] = value
                            obs_dict[ymd][self.ELEMENT_INDICES[element][1]] = qflag

                    except ValueError:
                        # Indicates invalid date, do not insert a record
                        pass

                    offset += self.OBS_COLUMN_SIZE

            line = obs_file.readline()

        obs = np.empty(len(obs_dict), dtype=DTYPE_STNOBS)
        obs[:] = [tuple(x) for x in obs_dict.values()]
        obs = obs[np.argsort(obs['ymd'])]

        return obs


class InsertRaws(Insert):
    '''
    Class for inserting stations and observations from the RAWS network
    '''

    def __init__(self, path_stn_file, path_stnid_file, path_obs_files, min_date, max_date):
        '''
        Parameters
        ----------
        path_stn_file : str
            File path to the raws station metadata file 
            generated by raws_build_stn_metadata
        path_stnid_file : str
            File path to the raws station id
            generated by raws_save_stnid_list or raws_to_ghcn_subset
        min_date : datetime
            The earliest observation date
        max_date : datetime
            The latest observation date
        '''

        Insert.__init__(self, min_date, max_date)
        self.path_stn_file = path_stn_file
        self.path_stnid_file = path_stnid_file
        self.path_obs_files = path_obs_files

    def get_stns(self):
        '''
        Retrieve a list of station rows (i.e.--tuples) to be inserted.
        
        Returns
        -------
        stns : list
            A list of tuples. Each tuple is of format:
            (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        '''

        print "RAWS: Building list of stations"

        fstnids = open(self.path_stnid_file)

        states = {}
        for line in fstnids.readlines():

            states[line[2:6]] = line[0:2].upper()

        fmeta = open(self.path_stn_file)

        stns = []
        for line in fmeta.readlines():

            stn_id_orig = line[0:4].strip()
            stn_id = "".join(["RAWS_", stn_id_orig])

            vals = line.split()

            lat = float(vals[3])
            lon = -float(vals[4])
            elev = float(vals[5])
            state = states[stn_id_orig]
            name = unicode(" ".join(vals[6:]))

            if os.path.exists(os.path.join(self.path_obs_files, '%s.txt' % (stn_id_orig,))):
                stns.append((stn_id, lat, lon, elev, state, name))
            else:
                print "".join([stn_id_orig, " in RAWS station list but no observations."])

        print "RAWS: Done building stations. Number of stns: %d" % (len(stns),)

        return stns

    def parse_stn_obs(self, stn_id):
        '''
        Parse  observations for a station
        
        Parameters
        ----------
        stn_id : str
            The station id for which to parse observations
            
        Returns
        -------
        obs : ndarray
            An array of observations in chronological order
            with dtype DTYPE_STNOBS          
        '''

        obs_file = open(os.path.join(self.path_obs_files, "%s.txt" % (stn_id[5:],)))

        # skip first 7 lines
        for x in np.arange(7):
            obs_file.readline()

        obs_ls = []

        for line in obs_file.readlines():

            if "Copyright" in line:
                break  # EOF reached
            else:

                try:
                    vals = line.split()
                    year = int(vals[0][6:])
                    month = int(vals[0][0:2])
                    day = int(vals[0][3:5])
                    ymd = long("".join([str(year), "%02d" % (month,), "%02d" % (day,)]))
                    datetime.date(year, month, day)  # throw error if not valid date

                except ValueError:
                    print "RAWS: Error in parsing a observation for", stn_id
                    continue

                if self.is_obs_inbounds(ymd):

                    tmax = float(vals[9])
                    tmin = float(vals[10])
                    prcp = float(vals[14])
                    prcp = prcp * 0.1 if prcp != MISSING else MISSING  # mm -> cm

                    # obs [stn_id,year,month,day,ymd,tmin,tmax,prcp,swe,"","",""]
                    obs_ls.append((year, month, day, ymd, tmin, tmax, prcp, MISSING, "", "", ""))

        obs = np.empty(len(obs_ls), dtype=DTYPE_STNOBS)
        obs[:] = obs_ls
        obs = obs[np.argsort(obs['ymd'])]

        return obs

def add_monthly_means(ds_path, var_name, max_miss=9):
    '''
    Calculate and add monthly temperature means to a
    netCDF station database. The new monthly temperature
    variable will be [var_name]_mth. Another new variable
    ([var_name]_mthmiss) will also be added to keep track
    of the number of missing daily observations in each
    month.
    
    Parameters
    ----------
    ds_path : str
        File path to a netCDF station database
    var_name : str
        The daily temperature variable name (eg- tmin, tmax)
    max_miss : int, optional
        The maximum # of missing daily observations in a month
        for a monthly mean to be calculated. If # of missing 
        observations for a month is > max_miss, the month's mean
        will be marked as missing. Set max_miss to None if there
        should not be a max_miss threshold.
    '''

    stnda = twx.db.StationDataDb(ds_path)
    tagg = twx.utils.TairAggregate(stnda.days)
    min_date = stnda.days[DATE][0]
    stns = stnda.stns
    stnda.ds.close()
    stnda = None
    ds = Dataset(ds_path, 'r+')

    if 'time_mth' not in ds.variables.keys():

        ds.createDimension('time_mth', tagg.yr_mths.size)
        times = ds.createVariable('time_mth', 'f8', ('time_mth',), fill_value=False)
        times.units = "".join(["days since ", str(min_date.year), "-", str(min_date.month), "-", str(min_date.day), " 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times[:] = date2num(tagg.yr_mths[DATE], times.units)

    var_mthly_name = "_".join([var_name, "mth"])
    
    if var_mthly_name not in ds.variables.keys():

        var_mthly = ds.createVariable(var_mthly_name, 'f4', ('time_mth', STN_ID),
                                      zlib=True, chunksizes=(tagg.yr_mths.size, 1),
                                      fill_value=netCDF4.default_fillvals['f4'])

    else:

        var_mthly = ds.variables[var_mthly_name]
    
    var_miss_name = "_".join([var_name,"mthmiss"])
    
    if var_miss_name not in ds.variables.keys():
         
        var_miss = ds.createVariable(var_miss_name,'i2',('time_mth', STN_ID),
                                     zlib=True, chunksizes=(tagg.yr_mths.size, 1),
                                     fill_value=netCDF4.default_fillvals['i2'])
    
    else:
        
        var_miss = ds.variables[var_miss_name]
    

    var_dly = ds.variables[var_name]
    var_dly_qa = ds.variables["_".join(["qflag", var_name])]
    chk_size = 50

    stchk = StatusCheck(np.int(np.round(stns.size / np.float(chk_size))), 10)
    for i in np.arange(0, stns.size, chk_size):

        if i + chk_size < stns.size:
            n_stns = chk_size
        else:
            n_stns = stns.size - i

        dly_vals = var_dly[:, i:i + n_stns]
        dly_vals_qa = var_dly_qa[:, i:i + n_stns]

        if np.ma.isMA(dly_vals):
            dly_vals[np.logical_not(dly_vals_qa.mask)] = np.ma.masked
        else:
            dly_vals = np.ma.masked_array(dly_vals, mask=np.logical_not(dly_vals_qa.mask))

        mth_vals,n_miss = tagg.daily_to_mthly(dly_vals, max_miss=max_miss)

        if np.ma.isMA(mth_vals):

            tmth_vals = mth_vals.data
            tmth_vals[mth_vals.mask] = var_mthly._FillValue
            mth_vals = tmth_vals

        var_mthly[:, i:i + n_stns] = mth_vals
        var_miss[:,i:i+n_stns] = n_miss
        ds.sync()
        stchk.increment()

def add_utc_offset(ds_path, geonames_usrname=None):
    '''
    Add a UTC offset station attribute to a netCDF database
    
    Parameters
    ----------
    ds_path : str
        File path to a netCDF station database
    geonames_usrname : str, optional
        A Geonames username. If not None,
        the Geonames time zone data web service will be
        used if time zone information for a point cannot
        be determined locally.
    '''
    
    stnda = twx.db.StationDataDb(ds_path, mode='r+') 
    
    var_utc = stnda.add_stn_variable(twx.db.UTC_OFFSET, twx.db.UTC_OFFSET,
                                     "", "i2")
    
    ndata = netCDF4.default_fillvals['i2']
    
    utc = twx.db.UtcOffset(ndata, geonames_usrname)
    
    print "Starting to get station UTC offset data..."
    
    schk = StatusCheck(stnda.stns.size, 1000)
    
    for x in np.arange(stnda.stns.size):
        
        try:
        
            a_utc = utc.get_utc_offset(stnda.stns[LON][x], stnda.stns[LAT][x])
        
        except GeonamesError:
            
            a_utc = ndata
        
        if a_utc == ndata:
            
            print "Error: Couldn't determine UTC offset for: %s"%stnda.stns[STN_ID][x]
        
        else:
        
            var_utc[x] = a_utc
        
        schk.increment()
    
    stnda.ds.sync()
    stnda.ds.close()
    stnda = None
    
def add_obs_cnt(ds_path, elem, start_date, end_date, stn_chk=500):
    '''Add period-of-record observation count netCDF variable
    
    For each station, calculates the number of daily observations in each month
    over a specified time period. Adds the observation counts as a
    netCDF variables of name: obs_cnt_[elem]_[ymd-start]_[ymd-end]
    
    Parameters
    ----------
    ds_path : str
        File path to a netCDF station database
    elem : str
        Element name for which to calculation observation count (e.g.-tmin)
    start_date : date-like
        The start date for time period over which to calculation obs counts
    end_date : date-like
        The end date for time period over which to calculation obs counts
    stn_chk : int, optional
        The number of stations to process in memory at a time. Default: 500.
    '''
    
    ds = xr.open_dataset(ds_path)
    
    cnts = []
    
    schk = StatusCheck(ds.station_id.size, stn_chk)
    
    for i in np.arange(ds.station_id.size, step=stn_chk):
    
        da = ds[elem][:,i:(i+stn_chk)].load().loc[start_date:end_date,:]
        
        cnt = da.groupby('time.month').count(dim='time')
        
        cnts.append(cnt)
        
        schk.increment(stn_chk)
    
    cnts = xr.concat(cnts, dim='station_id')
    ds.close()
    del ds
    
    ds = Dataset(ds_path,'r+')
    vname = "obs_cnt_%s_%d_%d"%(elem,ymdL(start_date),ymdL(end_date))
    
    if "mth" not in ds.dimensions.keys():
    
        ds.createDimension('mth', 12)
        vmth = ds.createVariable('mth', np.int, ('mth',),
                                 fill_value=False)
        vmth[:] = np.arange(1,13)
        
    if vname not in ds.variables.keys():
        
        vcnt = ds.createVariable(vname, np.int, ('mth','station_id'),
                                 fill_value=False)
        vcnt.comments = "Number of observations per calendar month"
        
    else:
        
        vcnt = ds.variables[vname]
        
    vcnt[:] = cnts.values
    
    ds.sync()
    ds.close()

def create_aux_db(path_all_db, path_infill_db_in, path_serial_db_in,
                  path_db_out, varname, start_year, end_year,
                  ds_version_str):
    '''Create a station netCDF database file to include in the public auxiliary data
        
    Parameters
    ----------
    path_all_db : str
        File path to the initial netCDF station database with raw observations from
        all networks 
    path_infill_db_in : str
        File path to the netCDF station database with infilled observations
    path_serial_db_in : str
        File path to the netCDF station database with final observation time
        series used as input to TopoWx
    path_db_out : str
        File path to write the output netCDF file
    varname : str
        The observation variable to write (tmin or tmax)
    start_year : int
        The start year of the observation time series
    end_year : int
        The end year of the observation time series
    ds_version_str : str
        The TopoWx dataset version string
    '''
    
    out_variables_dict = {'tmax':[('tmax_raw', 'f4',
                                   netCDF4.default_fillvals['f4'],
                                   'maximum air temperature', 'C'),
                                  ('tmax_homog', 'f4',
                                   netCDF4.default_fillvals['f4'],
                                   'maximum air temperature', 'C'),
                                  ('tmax_infilled', 'f4',
                                   netCDF4.default_fillvals['f4'],
                                   'maximum air temperature', 'C')],
                          'tmin':[('tmin_raw', 'f4',
                                   netCDF4.default_fillvals['f4'],
                                   'minimum air temperature', 'C'),
                                  ('tmin_homog', 'f4',
                                   netCDF4.default_fillvals['f4'],
                                   'minimum air temperature', 'C'),
                                  ('tmin_infilled', 'f4', 
                                   netCDF4.default_fillvals['f4'],
                                   'minimum air temperature', 'C')]}
    
    out_variables = out_variables_dict[varname]
    
    stnda_serial = StationSerialDataDb(path_serial_db_in, varname)
    mask_stns = np.nonzero(np.isnan(stnda_serial.stns[BAD]))[0]
    stns = stnda_serial.stns[mask_stns]
    days = stnda_serial.days
    mask_days = np.nonzero(np.logical_and(days[YEAR]>=start_year,
                                          days[YEAR]<=end_year))[0]
    stnda_serial.ds.close()
    del stnda_serial
    
    print "Creating %s..."%path_db_out
    create_quick_db(path_db_out, stns, days, out_variables)
    
    ds_in = Dataset(path_serial_db_in)
    ds_out = Dataset(path_db_out,'r+')
    
    print "Reading data from %s..."%path_serial_db_in
    tair_in = ds_in.variables[varname][:,mask_stns]
    tair_in = tair_in[mask_days,:]
    
    print "Writing data to %s..."%path_db_out
    ds_out.variables["%s_infilled"%varname][:] = tair_in
    ds_out.sync()
    ds_in.close()
    
    stnda = StationDataDb(path_all_db)
    mask_stns = np.nonzero(np.in1d(stnda.stn_ids, stns[STN_ID], True))[0]
    days = stnda.days
    mask_days = np.nonzero(np.logical_and(days[YEAR]>=start_year,
                                          days[YEAR]<=end_year))[0]
    stnda.ds.close()
    del stnda
    
    ds_in = Dataset(path_all_db)
    
    print "Reading data from %s..."%path_all_db
    tair_in = ds_in.variables[varname][:,mask_stns]
    tair_in = tair_in[mask_days,:]
    
    print "Writing data to %s..."%path_db_out
    ds_out.variables["%s_raw"%varname][:] = tair_in
    ds_out.sync()
    ds_in.close()
    
    stnda_infill = StationDataDb(path_infill_db_in)
    mask_stns = np.nonzero(np.in1d(stnda_infill.stn_ids,
                                   stns[STN_ID], True))[0]
    days = stnda_infill.days
    mask_days = np.nonzero(np.logical_and(days[YEAR]>=start_year,
                                          days[YEAR]<=end_year))[0]
    stnda_infill.ds.close()
    del stnda_infill
    
    ds_in = Dataset(path_infill_db_in)
    
    print "Reading data from %s..."%path_infill_db_in
    tair_in = ds_in.variables[varname][:,mask_stns]
    tair_in = tair_in[mask_days,:]
    flag_infill = ds_in.variables['flag_infilled'][:,mask_stns].astype(np.bool)
    flag_infill = flag_infill[mask_days,:]
    tair_in = np.ma.masked_array(tair_in,flag_infill)
        
    print "Writing data to %s..."%path_db_out
    ds_out.variables["%s_homog"%varname][:] = tair_in
    ds_out.sync()
    ds_in.close()
    
    ds_out.variables["%s_raw"%varname].comment = ("Original %s "
                                                  "observations from "
                                                  "data provider")%varname
    
    ds_out.variables["%s_homog"%varname].comment = ("QA'd and homogenized "
                                                    "daily %s observations "
                                                    "with missing values NOT "
                                                    "infilled.")%varname

    ds_out.variables['%s_infilled'%varname].comment = ("Final QA'd, "
                                                       "homogenized, and "
                                                       "missing value "
                                                       "infilled daily %s "
                                                       "observations used as "
                                                       "input to TopoWx. For "
                                                       "a station with more "
                                                       "than 5 continuous "
                                                       "years of missing "
                                                       "observations, all the "
                                                       "station's observations"
                                                       " were replaced with "
                                                       "values from the "
                                                       "station's infill "
                                                       "model. This was done "
                                                       "to avoid "
                                                       "inhomogeneities "
                                                       "between long segments "
                                                       "of infilled vs. "
                                                       "observed "
                                                       "values.")%varname
    
    
    ds_out.title = ("TopoWx weather station database of %d-%d daily %s "
                    "observations")%(start_year,end_year,varname)
    
    ds_out.comment = ("Original raw, QA'd, homogenized, and infilled %d-%d "
                      "daily %s temperature observations for stations used as "
                      "input to TopoWx. Includes stations with at least 5 "
                      "years of observations in each month from GHCN-Daily, "
                      "NRCS SNOTEL/SCAN, and WRCC RAWS.")%(start_year,end_year,varname)
                      
    ds_out.references = ("http://dx.doi.org/10.1002/joc.4127 , "
                         "http://dx.doi.org/10.1002/2014GL062803 , "
                         "http://dx.doi.org/10.1175/JAMC-D-15-0276.1")
    
    ds_out.source =  ("TopoWx software version %s "
                      "(https://github.com/jaredwo/topowx)")%twx.__version__
    ds_out.history = "".join(["Created on: ",datetime.datetime.strftime(date.today(),
                                                               "%Y-%m-%d"),
                              " , ","dataset version %s"%ds_version_str])
    
    ds_out.variables[STN_ID].comment = ("Original network station IDs with "
                                        "added prefix. GHCND station IDs start "
                                        "with 'GHCND_', NRCS SNOTEL/SCAN station "
                                        "IDs start with 'NRCS_' and RAWS "
                                        "station IDs start with 'RAWS_'. "
                                        "Stored as a variable-length string "
                                        "array.")
    
    ds_out.variables[STN_NAME].comment = ("Stored as a variable-length string "
                                          "array.")
    
    ds_out.variables[STATE].comment = ("Stored as a variable-length string array. "
                                       "Note: Some RAWS station states provided "
                                       "by the WRCC do not appear to be correct.")
    
    ds_out.close()
    
        