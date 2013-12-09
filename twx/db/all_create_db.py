'''
Functions and classes for inserting station observation data from multiple sources
into a single database format 

@author: jared.oyler
'''
import os
import numpy as np
import datetime
from twx.utils.status_check import status_check
import twx.utils.util_dates as utld
from twx.utils.util_dates import DATE,YMD,YEAR,MONTH,DAY
from netCDF4 import Dataset
from netCDF4 import date2num
import netCDF4
from netcdftime import num2date
import sys
from twx.db.station_data import STN_NAME,ELEV,LAT,LON,STATE, station_data_ncdb
import twx.db.ushcn as ushcn
import matplotlib.pyplot as plt
import twx.infill.obs_por as por

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

MISSING = -9999.
NCDF_CHK_COLS = 50
DTYPE_STNOBS = [('year', np.int), ('month', np.int), ('day', np.int), ('ymd', np.int),
               ('tmin',np.float64),('tmax',np.float64),('prcp',np.float64),('swe',np.float64),
               ('qflag_tmin',"S1"),('qflag_tmax',"S1"),('qflag_prcp',"S1")]

class dbDataset(Dataset):
    '''
    Subclass of netCDF4.Dataset. Provides utility methods for creating basic dimensions and variables
    most often used in this application
    '''
    
    def db_create_global_attributes(self,title):
        
        self.title = title
        self.institution = "University of Montana Numerical Terradynamics Simulation Group"
        self.history = "".join(["Created on: ",datetime.datetime.strftime(datetime.date.today(),"%Y-%m-%d")]) 
    
    def db_create_time_dimvar(self,days):
        
        self.createDimension('time',days.size)
        
        times = self.createVariable('time','f8',('time',))
        times.long_name = "time"
        times.units = "".join(["days since ",str(np.min(days[YEAR])),"-1-1 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times[:] = date2num(days[DATE],times.units)

    def db_create_level_dimvar(self,levels):
        
        self.createDimension('level',levels.size)
        
        l = self.createVariable('level','f8',('level',))
        l.long_name = "Level"
        l.units = "millibar"
        l.standard_name = "level"
        l[:] = levels

    def db_create_stnid_dimvar(self,stn_ids):
        
        self.createDimension('stn_id',stn_ids.size)
                
        stations = self.createVariable('stn_id','str',('stn_id',))
        stations.long_name = "station id"
        stations.standard_name = "station id"
        stations[:] = stn_ids.astype(np.object)
        
    def db_create_names_var(self,names):
        
        namesvar = self.createVariable('name','str',('stn_id',))
        namesvar.long_name = "station name"
        namesvar.standard_name = "name"
        namesvar[:] = names.astype(np.object)
        
    def db_create_states_var(self,states):
    
        statesvar = self.createVariable('state','str',('stn_id',))
        statesvar.long_name = "state"
        statesvar.standard_name = "state"
        statesvar[:] = states.astype(np.object)
    
    def db_create_lonlat_dimvar(self,lons,lats):
        
        self.createDimension('lon',lons.size)
        self.createDimension('lat',lats.size)
        
        self.db_create_lon_var(lons,('lon',))
        self.db_create_lat_var(lats,('lat',))
    
    def db_create_lat_var(self,lats,dim=('stn_id',)):
    
        latitudes = self.createVariable('lat','f8',dim)
        latitudes.long_name = "latitude"
        latitudes.units = "degrees_north"
        latitudes.standard_name = "latitude"
        latitudes[:] = lats
    
    def db_create_lon_var(self,lons,dim=('stn_id',)):
    
        longitudes = self.createVariable('lon','f8',dim)
        longitudes.long_name = "longitude"
        longitudes.units = "degrees_east"
        longitudes.standard_name = "longitude"
        longitudes[:] = lons
    
    def db_create_elev_var(self,elev):
    
        elevvar = self.createVariable('elev','f8',('stn_id',))
        elevvar.long_name = "elevation"
        elevvar.units = "m"
        elevvar.standard_name = "elevation"
        elevvar[:] = elev

    def db_create_stn_vars(self,stns):
        
        self.db_create_names_var(stns[STN_NAME])
        self.db_create_states_var(stns[STATE])
        self.db_create_lat_var(stns[LAT])
        self.db_create_lon_var(stns[LON])
        self.db_create_elev_var(stns[ELEV])
    
    def db_create_binflag_var(self,varname,lname,sname,dims=('time','stn_id'),chunk=None):
        
        flag_var = self.createVariable(varname,'i1',dims,chunksizes=chunk)
        flag_var.long_name = lname
        flag_var.standard_name = sname
        flag_var.missing_value = netCDF4.default_fillvals['i1']
        
    def db_create_mae_var(self,dims=('stn_id',),units="C"): 
        
        mae_var = self.createVariable('mae','f8',dims)
        mae_var.long_name = "mean absolute error"
        mae_var.units = units
        mae_var.standard_name = "mean_absolute_error"
        mae_var.missing_value = netCDF4.default_fillvals['f8']
    
    def db_create_bias_var(self,dims=('stn_id',),units="C"): 
    
        bias_var = self.createVariable('bias','f8',dims)
        bias_var.long_name = "bias"
        bias_var.units = units
        bias_var.standard_name = "bias"
        bias_var.missing_value = netCDF4.default_fillvals['f8']
    
    def db_create_tmin_var(self,chunk=None,fill_value=netCDF4.default_fillvals['f4']):
        
        tmin_var = self.createVariable('tmin','f4',('time','stn_id'),fill_value=fill_value,chunksizes=chunk)
        tmin_var.long_name = "minimum air temperature"
        tmin_var.units = "C"
        tmin_var.standard_name = "minimum_air_temperature"
        tmin_var.missing_value = fill_value
    
    def db_create_tmax_var(self,chunk=None,fill_value=netCDF4.default_fillvals['f4']):
        
        tmax_var = self.createVariable('tmax','f4',('time','stn_id'),fill_value=fill_value,chunksizes=chunk)
        tmax_var.long_name = "minimum air temperature"
        tmax_var.units = "C"
        tmax_var.standard_name = "minimum_air_temperature"
        tmax_var.missing_value = fill_value
        
    def db_create_tair_var(self,var,chunk=None,fill_value=netCDF4.default_fillvals['f4']):
        
        if var == "tmin":
            self.db_create_tmin_var(chunk,fill_value)
        elif var == "tmax":
            self.db_create_tmax_var(chunk,fill_value)
    
    def db_create_prcp_var(self,chunk=None,fill_value=netCDF4.default_fillvals['f4']):
    
        ncdf_var = self.createVariable('prcp','f4',('time','stn_id'),chunksizes=chunk)
        ncdf_var.long_name = "precipitation amount"
        ncdf_var.units = "cm"
        ncdf_var.standard_name = "precipitation_amount"
        ncdf_var.missing_value = fill_value
    
    def db_create_swe_var(self,chunk=None,fill_value=netCDF4.default_fillvals['f4']):
     
        ncdf_var = self.createVariable('swe','f4',('time','stn_id'),chunksizes=chunk)
        ncdf_var.long_name = "snow water equivalent"
        ncdf_var.units = "cm"
        ncdf_var.standard_name = "snow_water_equivalent"
        ncdf_var.missing_value = fill_value
    
    def db_create_char_var(self,varname,lname,sname,dims=('time','stn_id'),chunk=None):
        
        ncdf_var = self.createVariable(varname,'S1',dims,chunksizes=chunk)
        ncdf_var.long_name = lname
        ncdf_var.standard_name = sname
        ncdf_var.missing_value = ''
        
    def db_create_qflagtmin_var(self,dim=('time','stn_id'),chunk=None):
        
        self.db_create_char_var('qflag_tmin', "quality assurance flag tmin", "quality assurance flag tmin",dim, chunk)
        
    def db_create_qflagtmax_var(self,dim=('time','stn_id'),chunk=None):
        
        self.db_create_char_var('qflag_tmax', "quality assurance flag tmax", "quality assurance flag tmax",dim, chunk)
        
    def db_create_qflagprcp_var(self,dim=('time','stn_id'),chunk=None):
        
        self.db_create_char_var('qflag_prcp', "quality assurance flag prcp", "quality assurance flag prcp",dim, chunk)

def build_hcn_mask(stn_ids,ghcn_stn_fpath):
    
    #Build dict of HCN flags
    afile = open(ghcn_stn_fpath)
    hcn_dict = {}
    
    for line in afile.readlines():
        
        stn_id = "".join(["GHCN_",line[0:11].strip()])
        hcn_flg = line[76:79]
        hcn_dict[stn_id] = hcn_flg
    
    #Build mask of HCN flags using dict
    hcn_mask = np.zeros(stn_ids.size,dtype=np.bool)
    for x in np.arange(stn_ids.size):
    
        try:
            if hcn_dict[stn_ids[x]] == "HCN":
                hcn_mask[x] = True
        except KeyError:
            pass
    
    return hcn_mask

def create_db_ncdf(path,min_date,max_date,inserts):
    '''
    Creates a netCDF4 weather station database. At this time, all dimensions are
    set and fixed at creation time (i.e.--dates covered and number of stations). Current variables are
    TMIN, TMAX, PRCP and associated quality flags, and SWE (snotel only). All data is prefilled with missing values.
    
    @param path: the file path for the database
    @param min_date: the minimum date to be covered by the database (a datetime.datetime object)
    @param max_date: the maximum date to be covered by the database (a datetime.datetime object)
    @param inserts: a list of insert objects. These are used to determine the length of the stn dimension
    '''
    
    ncdf_file = Dataset(path,'w')
    
    #Set global attributes
    title = "Weather Station Database"
    ncdf_file.title = title
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.history = "".join(["Created on: ",datetime.datetime.strftime(datetime.date.today(),"%Y-%m-%d")]) 
    
    days = utld.get_days_metadata(min_date,max_date)
    
    nstns = 0
    for a_insert in inserts:
        nstns+=len(a_insert.get_stns())
    
    print "Creating netCDF4 database for "+str(min_date)+" to "+str(max_date)+" for "+str(nstns)+" stations."
    
    dim_time = ncdf_file.createDimension('time',days.size)
    dim_station = ncdf_file.createDimension('stn_id',nstns)
    
    times = ncdf_file.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.units = "".join(["days since ",str(min_date.year),"-",str(min_date.month),"-",str(min_date.day)," 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE],times.units)
    
    stations = ncdf_file.createVariable('stn_id','str',('stn_id',))
    stations.long_name = "station id"
    stations.standard_name = "station id"
    
    names = ncdf_file.createVariable('name','str',('stn_id',))
    names.long_name = "station name"
    names.standard_name = "name"
    
    states = ncdf_file.createVariable('state','str',('stn_id',))
    states.long_name = "state"
    states.standard_name = "state"
    
    latitudes = ncdf_file.createVariable('lat','f8',('stn_id',),fill_value=MISSING)
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = MISSING
    
    longitudes = ncdf_file.createVariable('lon','f8',('stn_id',),fill_value=MISSING)
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = MISSING
    
    elevs = ncdf_file.createVariable('elev','f8',('stn_id',),fill_value=MISSING)
    elevs.long_name = "elevation"
    elevs.units = "m"
    elevs.standard_name = "elevation"
    elevs[:] = MISSING
    
    tmin_var = ncdf_file.createVariable('tmin','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(days[DATE].size,NCDF_CHK_COLS))
    tmin_var.long_name = "minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "minimum_air_temperature"
    tmin_var.missing_value = MISSING
    tmin_var[:,:] = MISSING
    
    tmax_var = ncdf_file.createVariable('tmax','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(days[DATE].size,NCDF_CHK_COLS))
    tmax_var.long_name = "maximum air temperature"
    tmax_var.units = "C"
    tmax_var.standard_name = "maximum_air_temperature"
    tmax_var.missing_value = MISSING
    tmax_var[:,:] = MISSING

#    ncdf_var = ncdf_file.createVariable('prcp','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(days[DATE].size,NCDF_CHK_COLS))
#    ncdf_var.long_name = "precipitation amount"
#    ncdf_var.units = "cm"
#    ncdf_var.standard_name = "precipitation_amount"
#    ncdf_var.missing_value = MISSING
#    ncdf_var[:,:] = MISSING
    
    ncdf_var = ncdf_file.createVariable('qflag_tmin','S1',('time','stn_id'),fill_value='',chunksizes=(days[DATE].size,NCDF_CHK_COLS))
    ncdf_var.long_name = "quality assurance flag tmin"
    ncdf_var.standard_name = "quality assurance flag tmin"
    ncdf_var.missing_value = ""
    ncdf_var[:,:] = ""
    
    ncdf_var = ncdf_file.createVariable('qflag_tmax','S1',('time','stn_id'),fill_value='',chunksizes=(days[DATE].size,NCDF_CHK_COLS))
    ncdf_var.long_name = "quality assurance flag tmax"
    ncdf_var.standard_name = "quality assurance flag tmax"
    ncdf_var.missing_value = ""
    ncdf_var[:,:] = ""
    
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

def copy_db_ncdf_nometa(in_db_path,out_db_path,out_meta_path):
    
    ds = Dataset(in_db_path)
    stn_ids = np.array(ds.variables['stn_id'][:], dtype="<S16")
    names = [x.encode('ascii','replace') for x in ds.variables['name'][:]]
    names = [x.replace(","," ") for x in names]
    elev = ds.variables['elev'][:]
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    state = ds.variables['state'][:]
    
    times = ds.variables['time'][:]
    units = ds.variables['time'].units
    
    fout = open(out_meta_path,"w")
    fout.write(",".join(["STN_ID","NAME","STATE","LAT","LON","ELEV\n"]))
    for x in np.arange(stn_ids.size):
        fout.write(",".join([stn_ids[x],names[x],state[x],"%.5f"%(lat[x],),"%.5f"%(lon[x],),"%.2f\n"%(elev[x],)]))
    fout.close()
    
    ds_out = Dataset(out_db_path,'w')
    
    ds_out.createDimension('time',times.size)
    ds_out.createDimension('stn_id',stn_ids.size)
    
    times = ds_out.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.units = units
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = times
    
    tmin_var = ds_out.createVariable('tmin','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(times.size,NCDF_CHK_COLS))
    tmin_var.long_name = "minimum air temperature"
    tmin_var.units = "C"
    tmin_var.standard_name = "minimum_air_temperature"
    tmin_var.missing_value = MISSING
    
    tmax_var = ds_out.createVariable('tmax','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(times.size,NCDF_CHK_COLS))
    tmax_var.long_name = "maximum air temperature"
    tmax_var.units = "C"
    tmax_var.standard_name = "maximum_air_temperature"
    tmax_var.missing_value = MISSING

    ncdf_var = ds_out.createVariable('prcp','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(times.size,NCDF_CHK_COLS))
    ncdf_var.long_name = "precipitation amount"
    ncdf_var.units = "cm"
    ncdf_var.standard_name = "precipitation_amount"
    ncdf_var.missing_value = MISSING
    
    ncdf_var = ds_out.createVariable('qflag_tmin','S1',('time','stn_id'),fill_value='',chunksizes=(times.size,NCDF_CHK_COLS))
    ncdf_var.long_name = "quality assurance flag tmin"
    ncdf_var.standard_name = "quality assurance flag tmin"
    ncdf_var.missing_value = ""
    
    ncdf_var = ds_out.createVariable('qflag_tmax','S1',('time','stn_id'),fill_value='',chunksizes=(times.size,NCDF_CHK_COLS))
    ncdf_var.long_name = "quality assurance flag tmax"
    ncdf_var.standard_name = "quality assurance flag tmax"
    ncdf_var.missing_value = ""
    
    ncdf_var = ds_out.createVariable('qflag_prcp','S1',('time','stn_id'),fill_value='',chunksizes=(times.size,NCDF_CHK_COLS))
    ncdf_var.long_name = "quality assurance flag prcp"
    ncdf_var.standard_name = "quality assurance flag prcp"
    ncdf_var.missing_value = ""
    
    ncdf_var = ds_out.createVariable('swe','f4',('time','stn_id'),fill_value=MISSING,chunksizes=(times.size,NCDF_CHK_COLS))
    ncdf_var.long_name = "snow water equivalent"
    ncdf_var.units = "cm"
    ncdf_var.standard_name = "snow_water_equivalent"
    ncdf_var.missing_value = MISSING
    
    stat_chk = status_check(stn_ids.size,250)
    
    for x in np.arange(0,stn_ids.size,NCDF_CHK_COLS):
    
        if x + NCDF_CHK_COLS < stn_ids.size:
            nchk = NCDF_CHK_COLS
        else:
            nchk = stn_ids.size - x

        ds_out.variables['tmin'][:,x:x+nchk] = ds.variables['tmin'][:,x:x+nchk]
        ds_out.variables['tmax'][:,x:x+nchk] = ds.variables['tmax'][:,x:x+nchk]
        ds_out.variables['prcp'][:,x:x+nchk] = ds.variables['prcp'][:,x:x+nchk]
        ds_out.variables['qflag_tmin'][:,x:x+nchk] = ds.variables['qflag_tmin'][:,x:x+nchk]
        ds_out.variables['qflag_tmax'][:,x:x+nchk] = ds.variables['qflag_tmax'][:,x:x+nchk]
        ds_out.variables['qflag_prcp'][:,x:x+nchk] = ds.variables['qflag_prcp'][:,x:x+nchk]
        ds_out.variables['swe'][:,x:x+nchk] = ds.variables['swe'][:,x:x+nchk]
        
        stat_chk.increment(nchk)
        
    ds.close()
    ds_out.close()
    

def insert_data_ncdf(db_path,insert_objs):
    '''
    Inserts data into a netCDF4 weather station database.
    
    @param db_path: the path to the database in which the data should be inserted
    @param insert_objs:  a list of insert objects from which the insert data will be obtained
    '''
    
    ds = Dataset(db_path,'r+')
    time_units = ds.variables['time'].units
    
    time_vals = ds.variables['time'][:]
    
    date_start = num2date(time_vals[0],units=time_units)
    date_end = num2date(time_vals[-1],units=time_units)
    days = utld.get_days_metadata(date_start,date_end)
    
    ymd_idx = {}
    for x in np.arange(days[YMD].size):
        ymd_idx[days[YMD][x]] = x
        
    ########################################################
    #Get and insert all station metadata ordered by stn_id
    ########################################################
    all_stn_rows = []
    all_stn_rows_ls = []
    for insert in insert_objs:
        
        stn_rows = insert.get_stns()
        
        all_stn_rows.extend(stn_rows)
        all_stn_rows_ls.append(stn_rows)
    
    stn_ids = np.array([row[0] for row in all_stn_rows],dtype=np.object)
    lats = np.array([row[1] for row in all_stn_rows])
    lons = np.array([row[2] for row in all_stn_rows])
    elev = np.array([row[3] for row in all_stn_rows])
    state = np.array([row[4] for row in all_stn_rows],dtype=np.object)
    name = np.array([unicode(str(row[5]),"utf-8","replace") for row in all_stn_rows],dtype=np.object)
    
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
    #Get and insert all station observations
    ########################################################
    
    stat_chk = status_check(stn_ids.size, 100)
    
    for insert,stn_rows in zip(insert_objs,all_stn_rows_ls):
        
        stn_ids = ds.variables['stn_id'][:]
        
        stn_id = ""
        stn_idx = 0
        
        var_tmin = ds.variables['tmin']
        var_tmax = ds.variables['tmax']
        #var_prcp = ds.variables['prcp']
        var_qtmin = ds.variables['qflag_tmin']
        var_qtmax = ds.variables['qflag_tmax']
        #var_qprcp = ds.variables['qflag_prcp']
        #var_swe = ds.variables['swe']
        
        chkc = list(var_tmin.get_var_chunk_cache())
        chkc[0] = 1095750000 #bytes
        var_tmin.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        var_tmax.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        #var_prcp.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        var_qtmin.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        var_qtmax.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        #var_qprcp.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        #var_swe.set_var_chunk_cache(chkc[0],chkc[1],chkc[2])
        
        stn_id = None
        stn_idx = []
        stn_cnt = 0
        
        tmin_chk = np.ones((days.size,NCDF_CHK_COLS),dtype=np.float32)*MISSING
        tmax_chk = np.ones((days.size,NCDF_CHK_COLS),dtype=np.float32)*MISSING
        #prcp_chk = np.ones((days.size,NCDF_CHK_COLS),dtype=np.float32)*MISSING
        qtmin_chk = np.zeros((days.size,NCDF_CHK_COLS),dtype=np.character)
        qtmax_chk = np.zeros((days.size,NCDF_CHK_COLS),dtype=np.character)
        #qprcp_chk = np.zeros((days.size,NCDF_CHK_COLS),dtype=np.character)
        #swe_chk = np.ones((days.size,NCDF_CHK_COLS),dtype=np.float32)*MISSING
        
        stn_id = None
        for stn_row in stn_rows:
            
            if stn_id != None:
                
                stn_cnt+=1
                        
                if stn_cnt == NCDF_CHK_COLS:
                    
                    var_tmin[:,stn_idx] = tmin_chk
                    var_tmax[:,stn_idx] = tmax_chk
                    #var_prcp[:,stn_idx] = prcp_chk
                    var_qtmin[:,stn_idx] = qtmin_chk
                    var_qtmax[:,stn_idx] = qtmax_chk
                    #var_qprcp[:,stn_idx] = qprcp_chk
                    #var_swe[:,stn_idx] = swe_chk
                    tmin_chk[:,:] = MISSING
                    tmax_chk[:,:] = MISSING
                    #prcp_chk[:,:] = MISSING
                    qtmin_chk[:,:] = ""
                    qtmax_chk[:,:] = ""
                    #qprcp_chk[:,:] = ""
                    #swe_chk[:,:] = MISSING
                    stn_idx = []
                    stn_cnt = 0
            
            stn_id = stn_row[0]
            stn_idx.append(np.nonzero(stn_ids==stn_id)[0][0])
            
            obs = insert.parse_stn_obs(stn_id)
            
            if obs is not None:
            
                time_idx = np.in1d(days[YMD], obs['ymd'], assume_unique=True)
                #obs [stn_id,year,month,day,ymd,tmin,tmax,prcp,swe,"","",""]
                tmin_chk[time_idx,stn_cnt] = obs['tmin']
                tmax_chk[time_idx,stn_cnt] = obs['tmax']
                #prcp_chk[time_idx,stn_cnt] = obs['prcp']
                qtmin_chk[time_idx,stn_cnt] = obs['qflag_tmin']
                qtmax_chk[time_idx,stn_cnt] = obs['qflag_tmax']
                #qprcp_chk[time_idx,stn_cnt] = obs['qflag_prcp']
                #swe_chk[time_idx,stn_cnt] = obs['swe']
            
            stat_chk.increment()
        
        if len(stn_idx) > 0:
            
            idx = stn_cnt+1
            
            var_tmin[:,stn_idx] = tmin_chk[:,0:idx]
            var_tmax[:,stn_idx] = tmax_chk[:,0:idx]
            #var_prcp[:,stn_idx] = prcp_chk[:,0:idx]
            var_qtmin[:,stn_idx] = qtmin_chk[:,0:idx]
            var_qtmax[:,stn_idx] = qtmax_chk[:,0:idx]
            #var_qprcp[:,stn_idx] = qprcp_chk[:,0:idx]
            #var_swe[:,stn_idx] = swe_chk[:,0:idx]
        ds.sync()
            
    ds.close()
    ########################################################
    ######################################################## 

class insert:
    '''
    Parent class for inserting station data from multiple different sources into a single database format 
    '''
    
    def __init__(self,min_date,max_date,raster_mask=None):
        '''
        
        @param min_yr: the min year from which to insert observations
        @param max_yr: the max year from which to insert observations
        @param raster_mask: a utils.input_raster.input_raster class to specify the spatial extent 
        from which stations should be inserted (1=within extent,0=outside extent) 
        '''
        
        self.raster_mask = raster_mask
        if raster_mask is not None:
            self.mask = raster_mask.readEntireRaster()
        else:
            self.mask = None
        
        self.min_ymd = utld.ymdL(min_date)
        self.max_ymd = utld.ymdL(max_date)
    
    def get_stns(self):
        '''
        Retrieves a list of station rows (i.e.--tuples) to be inserted.  The rows are of the format:
        (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        '''
        pass
    
    def parse_stn_obs(self,stn_id):
        
        pass
    
    def is_obs_inbounds(self,ymd):
        '''
        Returns True if the year is between the specific min/max year bounds
        '''
        return ymd >= self.min_ymd and ymd <= self.max_ymd
    
    def is_stn_inbounds(self,lat,lon):
        '''
        Returns True if there is no mask or if the mask specifies the locations as being within bounds
        '''
        
        inbounds = True
        
        if self.raster_mask is not None:
                    
            try:
                c,r = self.raster_mask.getGridCellOffset(lon,lat)
                mask_val = self.mask[r,c] 
                #mask_val = self.raster_mask.getDataValue(lon,lat)
                if mask_val != 1:
                    #stn is not within the bounds dictated by the mask
                    inbounds = False
            
            except:
                #stn is completely outside the extent of the mask
                inbounds = False
        
        return inbounds


class insert_glac(insert):
    '''
    Class for inserting stations and observations from the Glacier National Park (GLAC) alpine
    network run by the USGS NOROCK: http://nrmsc.usgs.gov/AlpineClim_GNP 
    '''
    
    def __init__(self,stn_file_path,obs_path,min_date,max_date,raster_mask=None):
        '''
        
        @param stn_file_path: path to the file containing station metadata
        @param obs_path: directory path containing observations for each station
        @param min_yr: the min year for a station time series
        @param max_yr: a max year for a station time series
        @param raster_mask: a raster mask for which stations to insert
        '''
        
        insert.__init__(self,min_date,max_date,raster_mask)
        
        self.stn_file_path = stn_file_path
        self.obs_path = obs_path
        self.obs_dict = self.load_all_obs()
    
    def get_stns(self):
        '''
        Retrieves a list of station rows (i.e.--tuples) to be inserted.  The rows are of the format:
        (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        '''
        fin = file(self.stn_file_path)
        #skip header
        fin.readline()
        
        x = 1

        stns = []
        for line in fin.readlines():
    
            vals = line.split(',')
            
            name = vals[0]
            lon = float(vals[1])
            lat = float(vals[2])
            elev = float(vals[3])
            
            stns.append(("".join(["GLAC_",str(x)]),lat,lon,elev,"MT",name))
            #print ("".join(["GLAC_",str(x)]),lat,lon,elev,"MT",name)
            x+=1
            
        return stns
    
    def parse_stn_obs(self,stn_id):
        
        try:
            return self.obs_dict[stn_id]
        except KeyError:
            return None #No obs for stn_id
        
        
    def load_all_obs(self):

        fnames = np.array(os.listdir(self.obs_path))
        #Don't include backup files
        fnames = fnames[np.core.defchararray.find(fnames,"[bak]") == -1]
        
        prcp = MISSING
        swe = MISSING
        qflag_tmin = ''
        qflag_tmax = ''
        qflag_prcp = ''
        
        obs_dict = {}
        
        for fname in fnames:
            
            afile = open("".join([self.obs_path,fname]))
            
            #skip header
            afile.readline()
            
            stn_id = fname.split('-')[0]
            
            obs_rows = []
            
            for line in afile.readlines():
                    
                vals = line.split(',')
                
                if len(vals[0].strip()) == 0:
                    #reached end of file
                    break
                #Two different file formats; check which one
                #3 column: YMD,TMAX,TMIN
                #8 column: #Date,Tmax_C,Topomet_Tmax_C,Tmax_offset,Tmin_C,Topomet_Tmin_C,Tmin_offset_C,# of missing hours
                if len(vals) == 3:
                    
                    ymd = long(vals[0])
                    a_date = utld.ymdL_to_date(ymd)
                    
                    tmin = MISSING if vals[2].strip() == 'NA' else float(vals[2])
                    tmax = MISSING if vals[1].strip() == 'NA' else float(vals[1])
                    
                else:
                    
                    a_date = datetime.datetime.strptime(vals[0],'%m/%d/%Y')
                    ymd = utld.ymdL(a_date)
                    
                    tmin = MISSING if vals[4] == 'NA' else float(vals[4])
                    tmax = MISSING if vals[1] == 'NA' else float(vals[1])
                    
                if tmin == MISSING and tmax == MISSING:
                    continue
                obs_rows.append((a_date.year,a_date.month,a_date.day,ymd,tmin,tmax,prcp,swe,qflag_tmin,qflag_tmax,qflag_prcp))
            
            obs = np.empty(len(obs_rows), dtype=DTYPE_STNOBS)
            obs[:] = obs_rows
            obs = obs[np.argsort(obs['ymd'])]
            obs_dict[stn_id] = obs
        
        return obs_dict

class insert_china(insert):
    '''
    Class for inserting stations and observations from Chinese networks
    '''
    
    def __init__(self,path_data,min_date,max_date,raster_mask=None):
        '''
        
        @param path_clean: the directory path to cleaned and formatted SNOTEL data
        @param min_yr: the min year for a station time series
        @param max_yr: the max year for a station time series
        @param raster_mask: a raster mask for which stations to insert
        '''
        
        insert.__init__(self,min_date,max_date,raster_mask)
        self.path_data = path_data
        self.stns = self.load_all_stns()
        self.stnids = np.array([astn[0] for astn in self.stns])
        
        sIds = np.argsort(self.stnids)
        
        sortStns = []
        for x in sIds:
            sortStns.append(self.stns[x])
        self.stns = sortStns
        self.stnids = self.stnids[sIds]
        
        if np.unique(self.stnids).size != self.stnids.size:
            raise Exception("Non-unique stations")
        
        self.days = utld.get_days_metadata(min_date, max_date)
        
        self.tmin,self.tmax,self.prcp = self.load_all_obs()
        
        self.emptyQA = np.zeros(self.days.size,dtype=np.str)
        self.emptySWE = np.zeros(self.days.size)
    
    def get_stns(self):
        return self.stns
    
    def load_all_stns(self):
        
        #(STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        f_in = open("".join([self.path_data,'STATION_CHINA.TXT']))
        
        stns = []
        
        for line in f_in.readlines():

            vals = line.split()
            stnid = str(vals[0][-5:])
            lon = float(vals[1])
            lat = float(vals[2])
            elev = float(vals[3])
            
            if self.is_stn_inbounds(lat, lon):
                #Change to (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
                stns.append((stnid,lat,lon,elev,"",""))
        
        
        stnList2 = np.loadtxt("".join([self.path_data,'Station_updated.txt']),delimiter=',')
        for x in np.arange(stnList2.shape[0]):
            
            astn = stnList2[x,:]
            stns.append((str(int(astn[0])),astn[2],astn[1],astn[3],"",""))
        
        return stns   
    
    def load_all_obs(self):
        
        tmin = np.ones((self.days.size,self.stnids.size),dtype=np.int16)*32744
        tmax = np.ones((self.days.size,self.stnids.size),dtype=np.int16)*32744
        prcp = np.ones((self.days.size,self.stnids.size),dtype=np.int16)*32744
        
        missIds = []
        
        for yr in np.unique(self.days[YEAR]):
            
            print yr
            
            yrMask = self.days[YEAR] == yr
            
            #tmin
            tminYr = np.loadtxt("".join([self.path_data,"TMIN/TMIN_",str(yr),".csv"]),delimiter=',')
            stnidYr = tminYr[:,0].astype("<S5")
            tminYr = tminYr[:,1:]
            
            for x in np.arange(stnidYr.size):
                
                try:
                    idxId = np.nonzero(self.stnids==stnidYr[x])[0][0]
                    tmin[yrMask,idxId] = tminYr[x,:]
                except IndexError:
                    missIds.append(stnidYr[x])
                
            #tmax
            tmaxYr = np.loadtxt("".join([self.path_data,"TMAX/TMAX_",str(yr),".csv"]),delimiter=',')
            stnidYr = tmaxYr[:,0].astype("<S5")
            tmaxYr = tmaxYr[:,1:]
            
            for x in np.arange(stnidYr.size):
                
                try:
                    idxId = np.nonzero(self.stnids==stnidYr[x])[0][0]
                    tmax[yrMask,idxId] = tmaxYr[x,:]
                except IndexError:
                    missIds.append(stnidYr[x])
                
            #prcp
            prcpYr = np.loadtxt("".join([self.path_data,"PRCP/PRCP_",str(yr),".csv"]),delimiter=',')
            stnidYr = prcpYr[:,0].astype("<S5")
            prcpYr = prcpYr[:,1:]
            
            for x in np.arange(stnidYr.size):
                
                try:
                    idxId = np.nonzero(self.stnids==stnidYr[x])[0][0]
                    prcp[yrMask,idxId] = prcpYr[x,:]
                except IndexError:
                    missIds.append(stnidYr[x])
                
        print "# of stnids with obs but not in stn list: "+str(np.unique(missIds).size)
        
        return tmin,tmax,prcp
                    
    def parse_stn_obs(self,stn_id):
           
        idxId = np.nonzero(self.stnids==stn_id)[0][0]
        
        obs = np.empty(self.days.size, dtype=DTYPE_STNOBS)
        
        tminStn = self.tmin[:,idxId].astype(np.float)
        tmaxStn = self.tmax[:,idxId].astype(np.float)
        prcpStn = self.prcp[:,idxId].astype(np.float)
        
        #Set blank/missing to 0
        tminStn[np.logical_or(tminStn==32744,tminStn==32766)] = MISSING
        tmaxStn[np.logical_or(tmaxStn==32744,tmaxStn==32766)] = MISSING
        prcpStn[np.logical_or(prcpStn==32744,prcpStn==32766)] = MISSING
        
        #Set trace to 0
        prcpStn[prcpStn==32700] = 0
        
        #Snow mask
        snowMask = np.logical_and(prcpStn >= 31000,prcpStn <= 31999)
        prcpStn[snowMask] = prcpStn[snowMask] - 31000
        
        #Rain and snow mask
        snowMask = np.logical_and(prcpStn >= 30000,prcpStn <= 30999)
        prcpStn[snowMask] = prcpStn[snowMask] - 30000
        
        #fog dew frost mask
        fdfMask = np.logical_and(prcpStn >= 32000,prcpStn <= 32999)
        prcpStn[fdfMask] = prcpStn[fdfMask] - 32000
        
        #Convert units
        
        #tenths of degrees C to degrees C
        missMask = tminStn != MISSING
        tminStn[missMask] = tminStn[missMask]/10.0
        
        #tenths of degrees C to degrees C
        missMask = tmaxStn != MISSING
        tmaxStn[missMask] = tmaxStn[missMask]/10.0
        
        #tenths of mm to cm
        missMask = prcpStn != MISSING
        prcpStn[missMask] = prcpStn[missMask]/100.0
                
        obs['year'] = self.days[YEAR]
        obs['month'] = self.days[MONTH]
        obs['day'] = self.days[DAY]
        obs['ymd'] = self.days[YMD]
        obs['tmin'] = tminStn
        obs['tmax'] = tmaxStn
        obs['prcp'] = prcpStn
        obs['swe'] = self.emptySWE
        obs['qflag_tmin'] = self.emptyQA
        obs['qflag_tmax'] = self.emptyQA
        obs['qflag_prcp'] = self.emptyQA
                
        return obs

class insert_snotel(insert):
    '''
    Class for inserting stations and observations from the SNOTEL network. Requires SNOTEL ASCII data to first be
    cleaned and formatted using the snotel_clean module
    '''
    
    def __init__(self,path_clean,min_date,max_date,raster_mask=None):
        '''
        
        @param path_clean: the directory path to cleaned and formatted SNOTEL data
        @param min_yr: the min year for a station time series
        @param max_yr: the max year for a station time series
        @param raster_mask: a raster mask for which stations to insert
        '''
        
        insert.__init__(self,min_date,max_date,raster_mask)
        self.path_clean = path_clean
        self.stns = []
    
    def get_stns(self):
        
        #(STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
        f_in = open("".join([self.path_clean,'snotel_stns.csv']))
        f_in.readline()
        
        stns = []
        
        for line in f_in.readlines():
            #["STN_ID","NAME","STATE","DSOURCE","LAT","LON","ELEV"]
            vals = line.split(',')
            lat = float(vals[4])
            lon = float(vals[5])
            
            if self.is_stn_inbounds(lat, lon):
                #Change to (STN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME)
                stns.append(("".join(["SNOTEL_",vals[0]]).upper(),float(vals[4]),float(vals[5]),float(vals[6]),vals[2],vals[1]))
        
        return stns   
    
    def parse_stn_obs(self,stn_id):
        
        #remove snotel prefix and make lowercase
        stn_id_l = stn_id.split("_")[1].lower()
        
        
        try:
            obs_file = open("".join([self.path_clean,stn_id_l,".csv"]))
        except IOError:
            print "No observations for SNOTEL: "+stn_id
            return None
        
        line = obs_file.readline()
        obs_ls = []
        line = obs_file.readline()
        while len(line) > 0:
            #YMD,TMIN,TMAX,PRCP,TAVG,PREC,PILL
            vals = line.split(",")
            
            year = int(vals[0][0:4])
            month = int(vals[0][4:6])
            day = int(vals[0][6:])
            ymd = long(vals[0])
            
            if self.is_obs_inbounds(ymd):
                #stn_id,year,month,day,ymd,tmin,tmax,prcp,swe,qflag_tmin,qflag_tmax,qflag_prcp
                obs_ls.append((year,month,day,ymd,float(vals[1]),float(vals[2]),float(vals[3]),float(vals[6]),"","",""))
            
            line = obs_file.readline()
        
        obs = np.empty(len(obs_ls), dtype=DTYPE_STNOBS)
        obs[:] = obs_ls
        obs = obs[np.argsort(obs['ymd'])]
        
        return obs
  
class insert_ghcn(insert):
    '''
    Class for inserting stations and observations from the GHCN-Daily network
    '''

    VALID_FIPS_CODES = ["US","CA","MX"]
    VALID_ELEMENTS = ["TMIN","TMAX","PRCP"]
    RM_STATES = ["HI","AK"] #don't insert stations from HI or AK
    RM_NETWORK_CODES = ["S","R"] #don't insert SNOTEL and RAWS stations
    ELEMENT_INDICES = {"TMIN":(4,8),"TMAX":(5,9),"PRCP":(6,10)}
    MONTH_DAYS = range(1,32)
    OBS_COLUMN_SIZE = 8
    
    def __init__(self,path_stn_file,path_obs_files,min_date,max_date,raster_mask=None):
        '''
        
        @param path_stn_file: file path to the ghcn station metadata file (ghcnd-stations.txt)
        @param path_obs_files: directory path to ghcn observation files
        @param min_yr: the min date for a station time series
        @param max_yr: the max date for a station time series
        @param raster_mask: a raster mask for which stations to insert
        '''
        
        insert.__init__(self,min_date,max_date,raster_mask)
        self.path_stn_file = path_stn_file
        self.path_obs_files = path_obs_files
        
    def get_stns(self):
        
        print "GHCN: Building list of stations"
        
        file = open(self.path_stn_file)
            
        stns = []
        for line in file.readlines():
            
            fips_code = line[0:2]
            
            if fips_code in self.VALID_FIPS_CODES:
                
                stn_id_orig = line[0:11].strip()
                network_id = stn_id_orig[2]
                stn_id = "".join(["GHCN_",stn_id_orig])
                lat = float(line[12:20].strip())
                lon = float(line[21:30].strip())
                elev = float(line[31:37].strip())
                state = line[38:40].strip().upper()
                name = unicode(line[41:71].strip())
                
                if self.is_stn_inbounds(lat, lon) and state not in self.RM_STATES and network_id not in self.RM_NETWORK_CODES:
                    
                    if os.path.exists("".join([self.path_obs_files,stn_id_orig,".dly"])):
                        
                        stns.append((stn_id,lat,lon,elev,state,name))
                    else:
                        
                        print "".join([stn_id_orig," in GHCN station list but no observations."])
        print "GHCN: Done building stations. Number of stns: %d"%(len(stns),)
        
        return stns
    
    def convert_units(self,ELEMENT,VALUE):
    
        if VALUE == -9999:
            #NO DATA, no conversion
            return VALUE
        elif ELEMENT == "PRCP":
            #tenths of mm to cm
            return VALUE / 100.0
        elif ELEMENT == "TMAX" or ELEMENT == "TMIN" or ELEMENT == "TOBS":
            #tenths of degrees C to degrees C
            return VALUE/10.0
        else:
            raise Exception("".join(["Invalid ELEMENT type: ",ELEMENT])) 
    
    def parse_stn_obs(self,stn_id):

        obs_file = open("".join([self.path_obs_files,stn_id[5:],".dly"]))
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
                        
                        ymd = long("".join([str(year),"%02d"%(month,),"%02d"%(day,)]))
                        datetime.date(year,month,day) #throw error if not valid date
                        
                        if self.is_obs_inbounds(ymd):
                        
                            value = self.convert_units(element,float(line[21+offset:offset+26]))
                            qflag = line[27+offset:offset+28].strip()
              
                            if not obs_dict.has_key(ymd):
                                
                                #obs [year,month,day,ymd,tmin,tmax,prcp,swe,"","",""]
                                obs_dict[ymd] = [year,month,day,ymd,MISSING,MISSING,MISSING,MISSING,"","",""]
                            
                            obs_dict[ymd][self.ELEMENT_INDICES[element][0]]= value
                            obs_dict[ymd][self.ELEMENT_INDICES[element][1]]= qflag
                    
                    except ValueError:
                        #Indicates invalid date, do not insert a record
                        pass  
                    
                    offset+=self.OBS_COLUMN_SIZE
            
            line = obs_file.readline()
       
        obs = np.empty(len(obs_dict), dtype=DTYPE_STNOBS)
        obs[:] = [tuple(x) for x in obs_dict.values()]
        obs = obs[np.argsort(obs['ymd'])]
        
        return obs


class insert_ca(insert):
    '''
    Class for inserting stations and observations from non-GHCN Canada stations
    '''
    
    MISSING_CA = -99.9
    
    def __init__(self,path_stn_files,path_obs_files,path_ghcn_stn_file,min_date,max_date,raster_mask=None):
        '''
        
        @param path_stn_file: directory path to station metadata files
        @param path_obs_files: directory path to observation files
        @param path_ghcn_stn_file: file path to the ghcn station metadata file (ghcnd-stations.txt). used to remove stations
        that are duplicates with GHCN stations
        @param min_date: the min date for a station time series
        @param max_date: the max date for a station time series
        @param raster_mask: a raster mask for which stations to insert
        '''
        
        insert.__init__(self,min_date,max_date,raster_mask)
        self.path_stn_files = path_stn_files
        self.path_obs_files = path_obs_files
        self.path_ghcn = path_ghcn_stn_file
        
    def get_stns(self):
        
        #Get CA GHCN stations to check for dups
        #########################################
        ghcn_file = open(self.path_ghcn)
            
        ghcn_ids = []
        for line in ghcn_file.readlines():
            
            fips_code = line[0:2]
            
            if fips_code == "CA":
                
                stn_id = line[0:11].strip()
                
                #Last 7 digits are the Canada Station ID
                ghcn_ids.append(stn_id[-7:])
        #########################################
        
        print "CANADA: Building list of stations"
        
        fnames = np.array(os.listdir(self.path_stn_files))
        
        stns = []
        stn_ids = []
        ghcn_dups = []
        
        for fname in fnames:
            
            afile = open("".join([self.path_stn_files,fname]))
            afile.readline() #skip header
            
            for line in afile.readlines():
                
                vals = line.split("@")
                stn_id = vals[0].strip()
                
                if not stn_id in stn_ids:
                    
                    if not stn_id in ghcn_ids:
                    
                        stn_ids.append(stn_id)
                        stn_id = "".join(["CA_",stn_id])
                        name = "-".join([vals[1],vals[2].strip()])
                        state = ""
                        lon = float(vals[4])
                        lat = float(vals[3])
                        elev = float(vals[5])/1000.
                        
                        if self.is_stn_inbounds(lat, lon):
                            
                            if os.path.exists("".join("".join([self.path_obs_files,stn_id[3:],".txt"]))):
                        
                                stns.append((stn_id,lat,lon,elev,state,name))
                            
                            else:
                        
                                print "".join([stn_id," in Canada station list but no observations."])
                            
                    elif not stn_id in ghcn_dups:
                        
                        ghcn_dups.append(stn_id)

        print "CANADA: Done building stations. Number of stns: %d. Number of dup GHCN stations removed: %d"%(len(stns),len(ghcn_dups))
        
        return stns
    
    def parse_stn_obs(self,stn_id):

        obs_file = open("".join([self.path_obs_files,stn_id[3:],".txt"]))
        
        obs_ls = []
        
        for line in obs_file.readlines():
            
            vals = line.split()
            year = int(vals[1])
            month = int(vals[2])
            day = int(vals[3])
            ymd = long("".join([str(year),"%02d"%(month,),"%02d"%(day,)]))
            
            if self.is_obs_inbounds(ymd):
                tmin = float(vals[5])
                tmax = float(vals[4])
                tmin = tmin if tmin != self.MISSING_CA else MISSING
                tmax = tmax if tmax != self.MISSING_CA else MISSING
                
                prcp = float(vals[6])
                prcp = prcp*0.1 if prcp != self.MISSING_CA else MISSING #mm -> cm
                
                #year,month,day,ymd,tmin,tmax,prcp,swe,qflag_tmin,qflag_tmax,qflag_prcp
                obs_ls.append((year,month,day,ymd,tmin,tmax,prcp,MISSING,"","",""))
        
        obs = np.empty(len(obs_ls), dtype=DTYPE_STNOBS)
        obs[:] = obs_ls
        
        return obs

class insert_mx(insert):
    '''
    Class for inserting stations and observations from non-GHCN Mexico stations
    '''
    
    MISSING_MX = "NO_D"
    BREAK_STNMETA = "="
    INVALID_DATE = "~~~~"
    ELEMENT_INDICES = {"TMin":4,"TMax":5,"Prec":6}
    FILE_PREFIX_TMIN = "TMinuptoStation"
    FILE_PREFIX_TMAX = "TMaxuptoStation"
    FILE_PREFIX_PRCP = "PrecipuptoStation" 
    
    def __init__(self,path_data,path_ghcn_stn_file,min_date,max_date,raster_mask=None):
        '''
        
        @param path_data: directory path to Mexico station observation files
        @param path_obs_files: directory path to observation files
        @param path_ghcn_stn_file: file path to the ghcn station metadata file (ghcnd-stations.txt). used to remove stations
        that are duplicates with GHCN stations
        @param min_date: the min date for a station time series
        @param max_date: the max date for a station time series
        @param raster_mask: a raster mask for which stations to insert
        '''
        
        insert.__init__(self,min_date,max_date,raster_mask)
        self.path_data = path_data
        self.path_ghcn = path_ghcn_stn_file
        self.fnames = np.array(os.listdir(self.path_data))
        self.obs_files = self.preload_obs_files()
        self.cutoffs = self.build_cutoffs(self.fnames)
    
    def build_cutoffs(self,fnames):
        
        cutoffs = []
        
        for fname in fnames:
        
            cutoffs.append(int(fname[-9:-4]))
        
        cutoffs = np.unique(cutoffs)
        return cutoffs
      
    def get_stns(self):
        
        #Get MX GHCN stations to check for dups
        #########################################
        ghcn_file = open(self.path_ghcn)
            
        ghcn_ids = []
        for line in ghcn_file.readlines():
            
            fips_code = line[0:2]
            
            if fips_code == "MX":
                
                stn_id = line[0:11].strip()
                
                #Last 6 digits are the Mexico Station ID
                ghcn_ids.append(stn_id[-6:])
        #########################################
        
        print "MEXICO: Building list of stations"
        
        stns = []
        stn_ids = []
        ghcn_dups = []
        
        for fname in self.fnames:
            
            afile = open("".join([self.path_data,fname]))
            
            for x in np.arange(3):
                afile.readline() #skip first 3 lines
            
            line = afile.readline()
            
            while line[0] != self.BREAK_STNMETA:
                
                stn_id = line[7:13]
                
                if not stn_id in stn_ids:
                    
                    if not stn_id in ghcn_ids:
                        
                        try:
                        
                            lon = float(line[39:47])
                            lat = float(line[49:55])
                            elev = float(line[56:60])
                        
                        except ValueError:
                            
                            print "".join(["MEXICO: No location information for ",stn_id])
                            line = afile.readline()
                            continue
                        
                        stn_ids.append(stn_id)
                        stn_id = "".join(["MX_",stn_id])
                        name = line[14:39].strip()
                        state = ""
   
                        if self.is_stn_inbounds(lat, lon):
                            stns.append((stn_id,lat,lon,elev,state,name))
                    
                    elif not stn_id in ghcn_dups:
                        
                        ghcn_dups.append(stn_id)
                
                line = afile.readline()

        print "MEXICO: Done building stations. Number of stns: %d. Number of dup GHCN stations removed: %d"%(len(stns),len(ghcn_dups))
        
        return stns
    
    def preload_obs_files(self):
        
        file_dict = {}
        
        print "MEXICO: Preloading obs files into memory..."
        for fname in self.fnames:
            print fname
            
            obs_file = open("".join([self.path_data,fname]))
            lines = obs_file.readlines()
            fnl_lines = []
            
            
            x = 0
            while lines[x][0] != self.BREAK_STNMETA:
                x+=1
            x+=1
            
            while lines[x][0] != self.BREAK_STNMETA:
                
                fnl_lines.append(lines[x])
                x+=1
            
            fnl_lines.append("END")
            file_dict[fname] = fnl_lines
            
        return file_dict
    
    def get_obs_fnames(self,stn_num):
        
        stn_cutoff = None
        for x in np.arange(self.cutoffs.size):
            
            if x == 0:
                
                if stn_num <= self.cutoffs[x]:
                    stn_cutoff = str(self.cutoffs[x])
                    break
            
            if stn_num > self.cutoffs[x] and stn_num <= self.cutoffs[x+1]:
                
                stn_cutoff = str(self.cutoffs[x+1])
                break
        
        if stn_cutoff == None:
            print "debug"
        
        return "".join([self.FILE_PREFIX_TMIN,stn_cutoff,".txt"]),"".join([self.FILE_PREFIX_TMAX,stn_cutoff,".txt"]),"".join([self.FILE_PREFIX_PRCP,stn_cutoff,".txt"])
                
    def parse_stn_obs(self,stn_id):
        
        obs_dict = {}
        stn_fnd = False
        stn_num = int(stn_id[3:])
        
        for fname in self.get_obs_fnames(stn_num):
                
            elem = fname[0:4]
            
            lines = self.obs_files[fname]
            
            x = 0
            stn_fnd = False
            
            while lines[x][0:3] != "END":
                
                if lines[x][0:5] == "Clave":
                    
                    if stn_num == int(lines[x].split()[1]):
                        
                        stn_fnd = True
                        year = int(lines[x].split()[-1])
                        x+=2
                        
                        for day in np.arange(1,32):
                            
                            vals = lines[x].split()[1:]
                            
                            for month in np.arange(1,13):
                                
                                aobs = vals[month-1]
                                
                                if aobs == self.INVALID_DATE:
                                    continue
                                elif aobs == self.MISSING_MX:
                                    aobs = MISSING
                                else:
                                    aobs = float(aobs)
                                    if elem == "Prec":
                                        aobs = aobs*0.1 #mm -> cm
                                
                                ymd = long("".join([str(year),"%02d"%(month,),"%02d"%(day,)]))
                                
                                if self.is_obs_inbounds(ymd):
                                    
                                    if not obs_dict.has_key(ymd):
                            
                                        #obs [year,month,day,ymd,tmin,tmax,prcp,swe,"","",""]
                                        obs_dict[ymd] = [year,month,day,ymd,MISSING,MISSING,MISSING,MISSING,"","",""]
                                    
                                    obs_dict[ymd][self.ELEMENT_INDICES[elem]]= aobs
                            
                            x+=1
                    
                    elif stn_fnd:
                        break
                    else:
                        x+=33
                        
                else:
                    x+=1
               
        obs = np.empty(len(obs_dict), dtype=DTYPE_STNOBS)
        obs[:] = [tuple(x) for x in obs_dict.values()]
        obs = obs[np.argsort(obs['ymd'])]
        
        return obs

class insert_raws(insert):
    '''
    Class for inserting stations and observations from the RAWS network
    '''
    
    def __init__(self,path_stn_file,path_stnid_file,path_obs_files,min_date,max_date,raster_mask=None):
        '''
        
        @param path_stn_file: file path to the raws station metadata file (raws_meta.txt)
        @param path_stnid_file: file path to the raws stn id file (raws_stnids.txt)
        @param path_obs_files: directory path to raws observation files
        @param min_date: the min date for a station time series
        @param max_date: the max date for a station time series
        @param raster_mask: a raster mask for which stations to insert
        '''
        
        insert.__init__(self,min_date,max_date,raster_mask)
        self.path_stn_file = path_stn_file
        self.path_stnid_file = path_stnid_file
        self.path_obs_files = path_obs_files
        
    def get_stns(self):
        
        print "RAWS: Building list of stations"
        
        fstnids = open(self.path_stnid_file)
        
        states = {}
        for line in fstnids.readlines():
            
            states[line[2:6]] = line[0:2].upper()
        
        fmeta = open(self.path_stn_file)
            
        stns = []
        for line in fmeta.readlines():
                
            stn_id_orig = line[0:4].strip()
            stn_id = "".join(["RAWS_",stn_id_orig])
            
            vals = line.split()
            
            lat = float(vals[3])
            lon = -float(vals[4])
            elev = float(vals[5])
            state = states[stn_id_orig]
            name = unicode(" ".join(vals[6:]))
                
            if self.is_stn_inbounds(lat, lon):
                
                if os.path.exists("".join([self.path_obs_files,stn_id_orig,".txt"])):
                    stns.append((stn_id,lat,lon,elev,state,name))
                else:
                    print "".join([stn_id_orig," in RAWS station list but no observations."])
        print "RAWS: Done building stations. Number of stns: %d"%(len(stns),)
        
        return stns
    
    def parse_stn_obs(self,stn_id):

        obs_file = open("".join([self.path_obs_files,stn_id[5:],".txt"]))
        
        #skip first 7 lines
        for x in np.arange(7):
            obs_file.readline()
        
        obs_ls = []
        
        for line in obs_file.readlines():
            
            if "Copyright" in line:
                break #EOF reached
            else:
            
                try:
                    vals = line.split()
                    year = int(vals[0][6:])
                    month = int(vals[0][0:2])
                    day = int(vals[0][3:5])        
                    ymd = long("".join([str(year),"%02d"%(month,),"%02d"%(day,)]))
                    datetime.date(year,month,day) #throw error if not valid date
                        
                except ValueError:
                    print "RAWS: Error in parsing a observation for",stn_id
                    continue
                        
                if self.is_obs_inbounds(ymd):
                    
                    tmax = float(vals[9])
                    tmin = float(vals[10])
                    prcp = float(vals[14])
                    prcp = prcp*0.1 if prcp != MISSING else MISSING #mm -> cm
                    
                    #obs [stn_id,year,month,day,ymd,tmin,tmax,prcp,swe,"","",""]
                    obs_ls.append((year,month,day,ymd,tmin,tmax,prcp,MISSING,"","",""))
        
        obs = np.empty(len(obs_ls), dtype=DTYPE_STNOBS)
        obs[:] = obs_ls
        obs = obs[np.argsort(obs['ymd'])]
        
        return obs


class insert_usfs(insert):
    '''
    Class for inserting stations and observations from USFS mircomet data provided by
    Zack Holden (zaholdenfs@gmail.com)
    '''
    
    def __init__(self,path_obs_files,min_ymd,max_ymd,raster_mask=None):
        '''
        @param path_obs_files: directory path to observation files
        @param min_date: the min date for a station time series
        @param max_date: the max date for a station time series
        @param raster_mask: a raster mask for which stations to insert
        '''
        
        min_date,max_date = utld.ymdL_to_date(min_ymd),utld.ymdL_to_date(max_ymd)
        insert.__init__(self,min_date,max_date,raster_mask)
        self.path_obs_files = path_obs_files
        self.fnames = np.array(os.listdir(self.path_obs_files))
        
    def get_stns(self):
        
        print "USFS Micro: Building list of stations"
        
        stns = []
        for fname in self.fnames:
            
            fobs = open("".join([self.path_obs_files,fname]))
            fobs.readline() #skip header
            
            vals = fobs.readline().split(",")
            stn_id = vals[6].strip('"')
            stn_id = "".join(['USFS_',stn_id])
            lat = float(vals[7])
            lon = float(vals[8])
            elev = float(vals[22])
            state = "ID" #TODO: dynamically generate state
            name = fname.split(".")[0]
        
            if self.is_stn_inbounds(lat, lon):
                stns.append((stn_id,lat,lon,elev,state,name))
        
        print "USFS Micro: Done building stations. Number of stns: %d"%(len(stns),)
        
        return stns
    
    def parse_stn_obs(self,stn_id):

        fobs = None
        
        stn_id_orig = stn_id[5:]
        for fname in self.fnames:

            if stn_id_orig in fname:
                fobs = open("".join([self.path_obs_files,fname]))
                break
        
        fobs.readline() #skip header
        
        obs_ls = []
        for line in fobs.readlines():
            
            vals = line.split(",")
            adate = datetime.datetime.strptime(vals[0].strip('"'),"%Y-%m-%d")
            year = adate.year
            month = adate.month
            day = adate.day
            ymd = long("".join([str(year),"%02d"%(month,),"%02d"%(day,)]))
            
            tmin = float(vals[3])
            tmax = float(vals[4])
            
            obs_ls.append((year,month,day,ymd,tmin,tmax,MISSING,MISSING,"","",""))
        
        obs = np.empty(len(obs_ls), dtype=DTYPE_STNOBS)
        obs[:] = obs_ls
        obs['tmin'] = (obs['tmin']-32)/1.8
        obs['tmax'] = (obs['tmax']-32)/1.8
        obs = obs[np.argsort(obs['ymd'])]
        
        return obs


class insert_previous_all(insert):
    '''
    Class for inserting stations and observations from a previous netcdf database
    '''
    
    def __init__(self,db_path,min_date,max_date,raster_mask=None):
        '''
        
        @param db_path: file path to previous netcdf db
        @param min_date: the min date for a station time series
        @param max_date: the max date for a station time series
        @param raster_mask: a raster mask for which stations to insert
        '''
        
        insert.__init__(self,min_date,max_date,raster_mask)
        self.db_path = db_path
        self.ds = Dataset(self.db_path)
        self.stn_ids = np.array(self.ds.variables['stn_id'][:], dtype="<S16")
        self.stn_idxs = {}
        for x in np.arange(self.stn_ids.size):
            self.stn_idxs[self.stn_ids[x]] = x
        
        var_time = self.ds.variables['time']
        start, end = num2date([var_time[0], var_time[-1]], var_time.units)  
        self.days = utld.get_days_metadata(start, end)
        
    def get_stns(self):
        
        print "PREVIOUS DB: Building list of stations"
        
        stns = np.empty(self.stn_ids.size, dtype=[('stn_id', "<S16"), ('state', "<S2"), ('name', "<S30"), ('lon', np.float64), ('lat', np.float64), ('elev', np.float64)])
        
        stns['stn_id'] = self.stn_ids
        stns['state'] = self.ds.variables['state'][:]
        stns['name'] = self.ds.variables['name'][:]
        stns['lon'] = self.ds.variables['lon'][:]
        stns['lat'] = self.ds.variables['lat'][:]
        stns['elev'] = self.ds.variables['elev'][:]
        
        stns_ls = []
        
        for stn in stns:
        
            if self.is_stn_inbounds(stn['lat'], stn['lon']):
                stns_ls.append((stn['stn_id'],stn['lat'],stn['lon'],stn['elev'],stn['state'],stn['name']))
        
        print "PREVIOUS DB: Done building stations. Number of stns: %d"%(len(stns),)
        
        return stns_ls
    
    def parse_stn_obs(self,stn_id):
        
        stn_idx = self.stn_idxs[stn_id]
            
        tmin = self.ds.variables['tmin'][:,stn_idx]
        tmax = self.ds.variables['tmax'][:,stn_idx]
        prcp = self.ds.variables['prcp'][:,stn_idx]
        swe = self.ds.variables['swe'][:,stn_idx]
        
        flag_tmin = self.ds.variables['qflag_tmin'][:,stn_idx]
        flag_tmax = self.ds.variables['qflag_tmax'][:,stn_idx]
        flag_prcp = self.ds.variables['qflag_prcp'][:,stn_idx]
        
        if np.ma.isMA(tmin): tmin = tmin.data
        if np.ma.isMA(tmax): tmax = tmax.data
        if np.ma.isMA(prcp): prcp = prcp.data
        if np.ma.isMA(swe): swe = swe.data
        if np.ma.isMA(flag_tmin): flag_tmin = flag_tmin.data
        if np.ma.isMA(flag_tmax): flag_tmax = flag_tmax.data
        if np.ma.isMA(flag_prcp): flag_prcp = flag_prcp.data
        
#        valid_mask = np.logical_not(np.logical_and(np.logical_and(np.logical_and(tmin == MISSING,tmax == MISSING),prcp == MISSING),swe==MISSING))
#        valid_mask = np.logical_and(valid_mask,self.is_obs_inbounds(self.days[YMD]))
#        
#        tmin = tmin[valid_mask]
#        tmax = tmax[valid_mask]
#        prcp = prcp[valid_mask]
#        swe = swe[valid_mask]
#        flag_tmin = flag_tmin[valid_mask]
#        flag_tmax = flag_tmax[valid_mask]
#        flag_prcp = flag_prcp[valid_mask]
        
#DTYPE_STNOBS = [('year', np.int), ('month', np.int), ('day', np.int), ('ymd', np.int),
#               ('tmin',np.float64),('tmax',np.float64),('prcp',np.float64),('swe',np.float64),
#               ('qflag_tmin',"S1"),('qflag_tmax',"S1"),('qflag_prcp',"S1")]
        
        obs = np.empty(tmin.size, dtype=DTYPE_STNOBS)
        obs['year'][:] = self.days[YEAR]
        obs['month'][:] = self.days[MONTH]
        obs['day'][:] = self.days[DAY]
        obs['ymd'][:] = self.days[YMD]
        obs['tmin'][:] = tmin
        obs['tmax'][:] = tmax
        obs['prcp'][:] = prcp
        obs['swe'][:] = swe
        obs['qflag_tmin'][:] = flag_tmin
        obs['qflag_tmax'][:] = flag_tmax
        obs['qflag_prcp'][:] = flag_prcp
        
        return obs

def add_monthly_means(dsPath,varName):
    
    stnda = station_data_ncdb(dsPath)
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
    
    varMthlyName = "_".join([varName,"mth"])
    if varMthlyName not in ds.variables.keys(): 
        varMthly = ds.createVariable(varMthlyName,'f4',('time_mth','stn_id'),fill_value=netCDF4.default_fillvals['f4'])
    else:
        varMthly = ds.variables[varMthlyName]
        
    varMissName = "_".join([varName,"mthmiss"])
    if varMissName not in ds.variables.keys(): 
        varMiss = ds.createVariable(varMissName,'i2',('time_mth','stn_id'),fill_value=netCDF4.default_fillvals['i2'])
    else:
        varMiss = ds.variables[varMissName]
        
    varDly = ds.variables[varName]
    varDlyQA = ds.variables["_".join(["qflag",varName])]
    chkSize = 50
    
    stchk = status_check(np.int(np.round(stns.size/np.float(chkSize))), 10)
    for i in np.arange(0,stns.size,chkSize):
        
        if i + chkSize < stns.size:
            nStns = chkSize
        else:
            nStns = stns.size - i
        
        dlyVals = varDly[:,i:i+nStns]
        dlyValsQA = varDlyQA[:,i:i+nStns]
                
        if np.ma.isMA(dlyVals):
            dlyVals[np.logical_not(dlyValsQA.mask)] = np.ma.masked
        else:
            dlyVals = np.ma.masked_array(dlyVals,mask=np.logical_not(dlyValsQA.mask))
        
        mthVals,nMiss = tagg.dailyToMthly(dlyVals,maxMiss=9)
        
        if np.ma.isMA(mthVals):
            tmthVals = mthVals.data
            tmthVals[mthVals.mask] = varMthly._FillValue
            mthVals = tmthVals
            
        #varMthly[:,i:i+nStns] = mthVals
        varMiss[:,i:i+nStns] = nMiss
        ds.sync()
        stchk.increment()

def createTobsFile(pathTobsFiles,fpathOut,fpathDsOut,stnids,yrs):
    
    for yr in yrs:
        print yr
        os.system("".join(["zgrep 'TMIN\|TMAX' ",pathTobsFiles,str(yr),".csv.gz | zgrep -E '^US|^CA|^MX' | zgrep '[0-9]$' | zgrep -v '2400$' >> ",fpathOut]))
    
def createTobsDs(pathTobsFile,fpathDsOut,stnids,min_date,max_date):
    
    ds = Dataset(fpathDsOut,'w')
    
    #Set global attributes
    ds.title = "Time-of-Observation Database"
    ds.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ds.history = "".join(["Created on: ",datetime.datetime.strftime(datetime.date.today(),"%Y-%m-%d")]) 
    
    days = utld.get_days_metadata(min_date,max_date)
        
    print "Creating netCDF4 Time-of-Observation Database for "+str(min_date)+" to "+str(max_date)+" for "+str(stnids.size)+" stations."
    
    ds.createDimension('time',days.size)
    ds.createDimension('stn_id',stnids.size)
    
    times = ds.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.units = "".join(["days since ",str(min_date.year),"-",str(min_date.month),"-",str(min_date.day)," 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE],times.units)
    
    stations = ds.createVariable('stn_id','str',('stn_id',))
    stations.long_name = "station id"
    
    for x in np.arange(stnids.size):
        
        ds.variables['stn_id'][x] = str(stnids[x])
    
    tobs = ds.createVariable('tobs',np.int16,('time','stn_id'),fill_value=-1,chunksizes=(days[DATE].size,NCDF_CHK_COLS))
    tobs.long_name = "time-of-observation"
    tobs.missing_value = -1
    
    stnidsOrig = np.char.replace(stnids, "GHCN_", "", 1)
    
    fileTobs = open(pathTobsFile)
    aline = fileTobs.readline()
    
    
    
    atobs = np.ones((days.size,stnids.size))*-1
    
    curYmd = days[YMD][0]
    time_idx = 0
    #curStnId = None
    stn_idx = 0
    
    stchk = status_check(104075406,1000000)
    
    stn_idxs = {}
    for x in np.arange(stnidsOrig.size):
        stn_idxs[stnidsOrig[x]] = x
        
    while aline != "":
        #vals = aline.split(",")
        
        try:
#            
#            #if vals[0] != curStnId:
            #stn_idx = np.nonzero(stnidsOrig==aline[0:11])[0][0]
            stn_idx = stn_idxs[aline[0:11]]
#            #curStnId = vals[0]
#                
            aYmd = np.int(aline[12:20])
            if aYmd != curYmd:
                time_idx = np.nonzero(days[YMD]==aYmd)[0][0]
                curYmd = aYmd
                
            atobs[time_idx,stn_idx] = np.int(aline[-5:])
#                
        except KeyError:
            pass
        stchk.increment()
        aline = fileTobs.readline()
    tobs[:] = atobs
      

if __name__ == '__main__':
    
#    dem_mask = None#input_raster("/projects/daymet2/dem/dem_mask.tif")
#    min_date = datetime.datetime(1948,1,1)
#    max_date = datetime.datetime(2012,12,31)
#    
#    raws = insert_raws("/projects/daymet2/station_data/raws/raws_meta.txt", 
#                       "/projects/daymet2/station_data/raws/raws_stnids.txt",
#                       "/projects/daymet2/station_data/raws/raws_data/", min_date, max_date, dem_mask)
#    
##    ca = insert_ca('/projects/daymet2/station_data/ca_data/stn_metadata/',
##                   '/projects/daymet2/station_data/ca_data/obs_data/',
##                   '/projects/daymet2/station_data/ghcn/ghcnd-stations.txt',min_date,max_date,dem_mask)
##
##    mx = insert_mx('/projects/daymet2/station_data/mx_data/',
##                   '/projects/daymet2/station_data/ghcn/ghcnd-stations.txt',min_date,max_date,dem_mask)
#        
##    prev = insert_previous_all("/projects/daymet2/station_data/all/all.nc", min_date, max_date, dem_mask)
#    
#    ghcn = insert_ghcn("/projects/daymet2/station_data/ghcn/ghcnd-stations.txt", 
#                       "/projects/daymet2/station_data/ghcn/ghcnd_all/",min_date,max_date,dem_mask)
#    
#    snotel = insert_snotel('/projects/daymet2/station_data/snotel/cleaned/', min_date, max_date, dem_mask)
#    
##    glac = insert_glac('/projects/daymet2/station_data/glac/gnp_alpine_lonlat.csv',
##                       '/projects/daymet2/station_data/glac/csv/',
##                       min_date,max_date)
#    
#    #insert_objs = [mx,snotel,raws,ca,glac,ghcn]
#    insert_objs = [snotel,raws,ghcn]
#    create_db_ncdf("/projects/daymet2/station_data/all/all_1948_2012.nc",min_date,max_date,insert_objs)
#    insert_data_ncdf("/projects/daymet2/station_data/all/all_1948_2012.nc",insert_objs)
    
    #add_monthly_means("/projects/daymet2/station_data/all/all_1948_2012.nc", 'tmin')
    #add_monthly_means("/projects/daymet2/station_data/all/all_1948_2012.nc", 'tmax')
    
#    createTobsFile('/projects/daymet2/station_data/ghcn/ghcn_yrly/',
#                   '/projects/daymet2/station_data/ghcn/ghcn_yrly/AllTobs.csv',np.arange(1948,2013))
    
    porData = por.load_por_csv('/projects/daymet2/station_data/all/all_por_1948_2012.csv')
    mask_por_tmin, mask_por_tmax, mask_por_prcp = por.build_valid_por_masks(porData)
    stnda = station_data_ncdb('/projects/daymet2/station_data/all/all_1948_2012.nc')
    
    stnids = stnda.stn_ids[np.logical_or(mask_por_tmin,mask_por_tmax)]
    createTobsDs('/projects/daymet2/station_data/ghcn/ghcn_yrly/AllTobsTmax.csv',
                 '/projects/daymet2/station_data/ghcn/ghcn_yrly/AllTobsTmax.nc', stnids, datetime.datetime(1948,1,1), datetime.datetime(2012,12,31))
    
#    china = insert_china('/projects/daymet2/station_data/china/', datetime.datetime(1951,1,1), datetime.datetime(2012,12,31))
#    #obs = china.parse_stn_obs(china.stnids[0])
#    #plt.plot(obs['tmin'])
#    #plt.show()
#    create_db_ncdf('/projects/daymet2/station_data/china/chinaStns.nc', datetime.datetime(1951,1,1), 
#                   datetime.datetime(2012,12,31), (china,))
#    insert_data_ncdf('/projects/daymet2/station_data/china/chinaStns.nc', (china,))

    
    
