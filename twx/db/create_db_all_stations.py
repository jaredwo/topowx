'''
Functions and classes for inserting station observation data from multiple sources
into a single database format
'''

__all__ = ['Insert', 'InsertGhcn', 'InsertRaws', 'InsertSnotel',
           'create_netcdf_db', 'insert_data_netcdf_db',
           'MISSING', 'DTYPE_STNOBS', 'NCDF_CHK_COLS', 'add_monthly_means']

import os
import numpy as np
import datetime
from twx.utils import status_check, get_days_metadata, ymdL, DATE, YMD, YEAR
from netCDF4 import Dataset
from netCDF4 import date2num
import netCDF4
from netcdftime import num2date
from station_data import STN_NAME, ELEV, LAT, LON, STATE
import twx

MISSING = -9999.
NCDF_CHK_COLS = 50
DTYPE_STNOBS = [('year', np.int), ('month', np.int), ('day', np.int), ('ymd', np.int),
               ('tmin', np.float64), ('tmax', np.float64), ('prcp', np.float64), ('swe', np.float64),
               ('qflag_tmin', "S1"), ('qflag_tmax', "S1"), ('qflag_prcp', "S1")]

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

        stations = self.createVariable('stn_id', 'str', ('stn_id',))
        stations.long_name = "station id"
        stations.standard_name = "station id"
        stations[:] = stn_ids.astype(np.object)

    def db_create_names_var(self, names):

        namesvar = self.createVariable('name', 'str', ('stn_id',))
        namesvar.long_name = "station name"
        namesvar.standard_name = "name"
        namesvar[:] = names.astype(np.object)

    def db_create_states_var(self, states):

        statesvar = self.createVariable('state', 'str', ('stn_id',))
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
    dim_station = ncdf_file.createDimension('stn_id', nstns)

    times = ncdf_file.createVariable('time', 'f8', ('time',), fill_value=False)
    times.long_name = "time"
    times.units = "".join(["days since ", str(min_date.year), "-", str(min_date.month), "-", str(min_date.day), " 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE], times.units)

    stations = ncdf_file.createVariable('stn_id', 'str', ('stn_id',))
    stations.long_name = "station id"
    stations.standard_name = "station id"

    names = ncdf_file.createVariable('name', 'str', ('stn_id',))
    names.long_name = "station name"
    names.standard_name = "name"

    states = ncdf_file.createVariable('state', 'str', ('stn_id',))
    states.long_name = "state"
    states.standard_name = "state"

    latitudes = ncdf_file.createVariable('lat', 'f8', ('stn_id',), fill_value=MISSING)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = MISSING

    longitudes = ncdf_file.createVariable('lon', 'f8', ('stn_id',), fill_value=MISSING)
    longitudes.long_name = 'longitude'
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = MISSING

    elevs = ncdf_file.createVariable('elev', 'f8', ('stn_id',), fill_value=MISSING)
    elevs.long_name = "elevation"
    elevs.units = "m"
    elevs.standard_name = "elevation"
    elevs[:] = MISSING

    tmin_var = ncdf_file.createVariable('tmin', 'f4', ('time', 'stn_id'), fill_value=MISSING,
                                        chunksizes=(days[DATE].size, NCDF_CHK_COLS))
    tmin_var.long_name = "minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "minimum_air_temperature"
    tmin_var.missing_value = MISSING
    tmin_var[:, :] = MISSING

    tmax_var = ncdf_file.createVariable('tmax', 'f4', ('time', 'stn_id'), fill_value=MISSING,
                                        chunksizes=(days[DATE].size, NCDF_CHK_COLS))
    tmax_var.long_name = "maximum air temperature"
    tmax_var.units = "C"
    tmax_var.standard_name = "maximum_air_temperature"
    tmax_var.missing_value = MISSING
    tmax_var[:, :] = MISSING

#    ncdf_var = ncdf_file.createVariable('prcp','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(days[DATE].size,NCDF_CHK_COLS))
#    ncdf_var.long_name = "precipitation amount"
#    ncdf_var.units = "cm"
#    ncdf_var.standard_name = "precipitation_amount"
#    ncdf_var.missing_value = MISSING
#    ncdf_var[:,:] = MISSING

    ncdf_var = ncdf_file.createVariable('qflag_tmin', 'S1', ('time', 'stn_id'), fill_value='',
                                        chunksizes=(days[DATE].size, NCDF_CHK_COLS))
    ncdf_var.long_name = "quality assurance flag tmin"
    ncdf_var.standard_name = "quality assurance flag tmin"
    ncdf_var.missing_value = ""
    ncdf_var[:, :] = ""

    ncdf_var = ncdf_file.createVariable('qflag_tmax', 'S1', ('time', 'stn_id'), fill_value='',
                                        chunksizes=(days[DATE].size, NCDF_CHK_COLS))
    ncdf_var.long_name = "quality assurance flag tmax"
    ncdf_var.standard_name = "quality assurance flag tmax"
    ncdf_var.missing_value = ""
    ncdf_var[:, :] = ""

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

        ds.variables['stn_id'][x] = str(stn_ids[x])
        ds.variables['lat'][x] = lats[x]
        ds.variables['lon'][x] = lons[x]
        ds.variables['elev'][x] = elev[x]
        ds.variables['state'][x] = str(state[x])
        ds.variables['name'][x] = name[x]

    ds.sync()
    ########################################################
    ########################################################

    ########################################################
    # Get and insert all station observations
    ########################################################
    
    print "Inserting observations for each station..."

    stat_chk = status_check(stn_ids.size, 100)

    for insert, stn_rows in zip(insert_objs, all_stn_rows_ls):

        stn_ids = ds.variables['stn_id'][:]

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

            var_tmin[:, stn_idx] = tmin_chk[:, 0:idx]
            var_tmax[:, stn_idx] = tmax_chk[:, 0:idx]
            # var_prcp[:,stn_idx] = prcp_chk[:,0:idx]
            var_qtmin[:, stn_idx] = qtmin_chk[:, 0:idx]
            var_qtmax[:, stn_idx] = qtmax_chk[:, 0:idx]
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


class InsertSnotel(Insert):
    '''
    Class for inserting stations and observations from the SNOTEL network.
    Requires SNOTEL ASCII data to first be cleaned and formatted using the
    snotel_clean module
    '''

    def __init__(self, path_stn_file, path_clean_obs, min_date, max_date):
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
        self.path_stn_file = path_stn_file
        self.path_clean = path_clean_obs
        self.stns = []

    def get_stns(self):
        '''
        Retrieve a list of station rows (i.e.--tuples) to be inserted.
        
        Returns
        -------
        stns : list
            A list of tuples. Each tuple is of format:
            (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        '''

        print "SNOTEL: Building list of stations"

        f_in = open(self.path_stn_file)
        f_in.readline()

        stns = []

        for line in f_in.readlines():
            # ["STN_ID","NAME","STATE","DSOURCE","LAT","LON","ELEV"]
            vals = line.split(',')

            # Change to (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
            stns.append(("".join(["SNOTEL_", vals[0]]).upper(), float(vals[4]), float(vals[5]), float(vals[6]), vals[2], vals[1]))

        print "SNOTEL: Done building stations. Number of stns: %d" % (len(stns),)

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

        # remove snotel prefix and make lowercase
        stn_id_l = stn_id.split("_")[1].lower()


        try:
            obs_file = open(os.path.join(self.path_clean, "%s.csv" % (stn_id_l,)))
        except IOError:
            print "No observations for SNOTEL: " + stn_id
            return None

        line = obs_file.readline()
        obs_ls = []
        line = obs_file.readline()
        while len(line) > 0:
            # YMD,TMIN,TMAX,PRCP,TAVG,PREC,PILL
            vals = line.split(",")

            year = int(vals[0][0:4])
            month = int(vals[0][4:6])
            day = int(vals[0][6:])
            ymd = long(vals[0])

            if self.is_obs_inbounds(ymd):
                # stn_id,year,month,day,ymd,tmin,tmax,prcp,swe,qflag_tmin,qflag_tmax,qflag_prcp
                obs_ls.append((year, month, day, ymd, float(vals[1]), float(vals[2]), float(vals[3]), float(vals[6]), "", "", ""))

            line = obs_file.readline()

        obs = np.empty(len(obs_ls), dtype=DTYPE_STNOBS)
        obs[:] = obs_ls
        obs = obs[np.argsort(obs['ymd'])]

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

        var_mthly = ds.createVariable(var_mthly_name, 'f4', ('time_mth', 'stn_id'),
                                     fill_value=netCDF4.default_fillvals['f4'])

    else:

        var_mthly = ds.variables[var_mthly_name]
    
    var_miss_name = "_".join([var_name,"mthmiss"])
    
    if var_miss_name not in ds.variables.keys():
         
        var_miss = ds.createVariable(var_miss_name,'i2',('time_mth','stn_id'),
                                    fill_value=netCDF4.default_fillvals['i2'])
    
    else:
        
        var_miss = ds.variables[var_miss_name]
    

    var_dly = ds.variables[var_name]
    var_dly_qa = ds.variables["_".join(["qflag", var_name])]
    chk_size = 50

    stchk = status_check(np.int(np.round(stns.size / np.float(chk_size))), 10)
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

# if __name__ == '__main__':

    # add_monthly_means("/projects/daymet2/station_data/all/all_1948_2012.nc", 'tmin')
    # add_monthly_means("/projects/daymet2/station_data/all/all_1948_2012.nc", 'tmax')
